import json

from tau2.agent.base.streaming import (
    LinearizationStrategy,
    ParticipantTick,
    linearize_ticks,
)
from tau2.config import DEFAULT_LLM_NL_ASSERTIONS, DEFAULT_LLM_NL_ASSERTIONS_ARGS
from tau2.data_model.message import Message, SystemMessage, Tick, UserMessage
from tau2.data_model.simulation import NLAssertionCheck, RewardInfo
from tau2.data_model.tasks import RewardType, Task
from tau2.evaluator.evaluator_base import EvaluatorBase
from tau2.utils.llm_utils import generate


class NLAssertionsEvaluator(EvaluatorBase[Message]):
    """
    Judge that evaluates whether a trajectory adheres to all the natural-language assertions.
    """

    @classmethod
    def calculate_reward(
        cls,
        task: Task,
        full_trajectory: list[Message],
    ) -> RewardInfo:
        """
        Calculate the reward for the simulation by using an LLM to evaluate whether the trajectory adheres to all the natural-language assertions
        """
        if task.evaluation_criteria is None:
            return RewardInfo(
                reward=1.0,
                nl_assertions=[],
                info={"note": "No evaluation criteria"},
                reward_breakdown={RewardType.NL_ASSERTION: 1.0},
            )
        nl_assertions = task.evaluation_criteria.nl_assertions
        if not nl_assertions:
            return RewardInfo(
                reward=1.0,
                nl_assertions=[],
                info={"note": "No nl_assertions to evaluate"},
                reward_breakdown={RewardType.NL_ASSERTION: 1.0},
            )

        nl_assertions_checks = cls.evaluate_nl_assertions(
            full_trajectory, nl_assertions
        )

        # Calculate reward: 1 if all expectations are met, 0 otherwise
        all_expectations_met = all(result.met for result in nl_assertions_checks)
        reward = 1.0 if all_expectations_met else 0.0

        return RewardInfo(
            reward=reward,
            nl_assertions=nl_assertions_checks,
            reward_breakdown={RewardType.NL_ASSERTION: reward},
        )

    @classmethod
    def evaluate_nl_assertions(
        cls,
        trajectory: list[Message],
        nl_assertions: list[str],
    ) -> list[NLAssertionCheck]:
        """
        Evaluate whether the trajectory meets each expected outcome.

        Args:
            trajectory: List of messages from the conversation
            nl_assertions: List of natural-language assertions to evaluate

        Returns:
            List of evaluation results for each NL assertion, containing:
            - nl_assertion: The NL assertion being evaluated
            - metExpectation: Boolean indicating if the assertion was met
            - reasoning: Explanation for the evaluation
        """
        trajectory_str = "\n".join(
            [f"{message.role}: {message.content}" for message in trajectory]
        )
        # System prompt similar to the TypeScript implementation
        system_prompt = """
        TASK
        - You will be given a list of expected outcomes and a conversation that was collected during a test case run.
        - The conversation is between an agent and a customer.
        - Your job is to evaluate whether the agent satisfies each of the expected outcomes.
        - Grade each expected outcome individually.

        FORMAT
        - Your response should be a JSON object with the following fields:
        - `reasoning`: a short explanation for your classification
        - `metExpectation`: `true` if the agent satisfies the expected outcomes, `false` otherwise
        - `expectedOutcome`: repeat the expectation from the input that you are grading
        
        Example response structure:
        {
            "results": [
                {
                    "expectedOutcome": "<one of the expected outcomes from the input>",
                    "reasoning": "<reasoning trace>",
                    "metExpectation": <false or true>,
                }
            ]
        }
        """

        user_prompt = f"""
        conversation:
        {trajectory_str}
        
        expectedOutcomes:
        {nl_assertions}
        """

        messages = [
            SystemMessage(role="system", content=system_prompt),
            UserMessage(role="user", content=user_prompt),
        ]

        assistant_message = generate(
            model=DEFAULT_LLM_NL_ASSERTIONS,
            messages=messages,
            call_name="nl_assertions_eval",
            **DEFAULT_LLM_NL_ASSERTIONS_ARGS,
        )
        result_data = json.loads(assistant_message.content)
        return [
            NLAssertionCheck(
                nl_assertion=result["expectedOutcome"],
                met=result["metExpectation"],
                justification=result["reasoning"],
            )
            for result in result_data.get("results", [])
        ]


class FullDuplexNLAssertionsEvaluator(EvaluatorBase[Tick]):
    """
    Judge that evaluates whether a full-duplex trajectory adheres to all the
    natural-language assertions.
    """

    @classmethod
    def ticks_to_message_history(cls, ticks: list[Tick]) -> list[Message]:
        """
        Convert a list of Ticks to a linearized message history suitable for NL evaluation.

        This converts orchestrator Ticks to ParticipantTicks (from the agent's perspective),
        then uses containment-aware linearization to create a sequential message list.
        Only speech content is included (tool calls are ignored for NL evaluation).

        Args:
            ticks: List of Tick objects from full-duplex simulation.

        Returns:
            List of Messages linearized using containment-aware strategy.
        """
        # Convert orchestrator Ticks to ParticipantTicks from agent's perspective
        # self_chunk = agent_chunk, other_chunk = user_chunk
        # We only care about speech content, not tool calls
        participant_ticks: list[ParticipantTick] = []

        for tick in ticks:
            # Only include chunks that have content (not tool calls)
            agent_chunk = tick.agent_chunk
            user_chunk = tick.user_chunk

            # Skip tool call messages - we only want speech content
            if agent_chunk is not None and agent_chunk.is_tool_call():
                agent_chunk = None
            if user_chunk is not None and user_chunk.is_tool_call():
                user_chunk = None

            participant_tick = ParticipantTick(
                tick_id=tick.tick_id,
                timestamp=tick.timestamp,
                self_chunk=agent_chunk,
                other_chunk=user_chunk,
            )
            participant_ticks.append(participant_tick)

        # Linearize using containment-aware strategy
        messages = linearize_ticks(
            participant_ticks,
            strategy=LinearizationStrategy.CONTAINMENT_AWARE,
        )

        return messages

    @classmethod
    def calculate_reward(
        cls,
        task: Task,
        full_trajectory: list[Tick],
    ) -> RewardInfo:
        """
        Calculate the reward for the simulation by using an LLM to evaluate whether
        the trajectory adheres to all the natural-language assertions.
        """
        if task.evaluation_criteria is None:
            return RewardInfo(
                reward=1.0,
                nl_assertions=[],
                info={"note": "No evaluation criteria"},
                reward_breakdown={RewardType.NL_ASSERTION: 1.0},
            )
        nl_assertions = task.evaluation_criteria.nl_assertions
        if not nl_assertions:
            return RewardInfo(
                reward=1.0,
                nl_assertions=[],
                info={"note": "No nl_assertions to evaluate"},
                reward_breakdown={RewardType.NL_ASSERTION: 1.0},
            )

        # Convert ticks to linearized message history
        messages = cls.ticks_to_message_history(full_trajectory)

        nl_assertions_checks = cls.evaluate_nl_assertions(messages, nl_assertions)

        # Calculate reward: 1 if all expectations are met, 0 otherwise
        all_expectations_met = all(result.met for result in nl_assertions_checks)
        reward = 1.0 if all_expectations_met else 0.0

        return RewardInfo(
            reward=reward,
            nl_assertions=nl_assertions_checks,
            reward_breakdown={RewardType.NL_ASSERTION: reward},
        )

    @classmethod
    def evaluate_nl_assertions(
        cls,
        trajectory: list[Message],
        nl_assertions: list[str],
    ) -> list[NLAssertionCheck]:
        """
        Evaluate whether the trajectory meets each expected outcome.

        Delegates to NLAssertionsEvaluator.evaluate_nl_assertions.
        """
        return NLAssertionsEvaluator.evaluate_nl_assertions(trajectory, nl_assertions)
