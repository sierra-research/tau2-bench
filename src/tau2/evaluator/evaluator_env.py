from typing import Callable, Optional

from loguru import logger

from tau2.data_model.evaluation import DBEvaluation, EnvAssertionEvaluation
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    Tick,
    UserMessage,
)
from tau2.data_model.simulation import DBCheck, EnvAssertionCheck, RewardInfo
from tau2.data_model.tasks import RewardType, Task
from tau2.environment.environment import Environment
from tau2.evaluator.evaluator_base import EvaluatorBase


def _evaluate_environment_state(
    environment_constructor: Callable[[], Environment],
    task: Task,
    predicted_message_history: list[Message],
    *,
    solo_mode: bool = False,
    env_kwargs: dict | None = None,
) -> tuple[Optional[DBEvaluation], list[EnvAssertionEvaluation], dict]:
    if task.evaluation_criteria is None:
        return None, [], {"note": "No evaluation criteria"}

    expected_actions = task.evaluation_criteria.actions
    env_assertions = task.evaluation_criteria.env_assertions
    if expected_actions is None and env_assertions is None:
        return DBEvaluation(db_match=True), [], {
            "note": "No expected actions or env assertions"
        }

    initialization_data = None
    if task.initial_state is not None and task.initial_state.initialization_data is not None:
        initialization_data = task.initial_state.initialization_data

    initialization_actions = None
    if (
        task.initial_state is not None
        and task.initial_state.initialization_actions is not None
    ):
        initialization_actions = task.initial_state.initialization_actions

    message_history = []
    if task.initial_state is not None and task.initial_state.message_history is not None:
        message_history = task.initial_state.message_history

    if env_kwargs is None:
        env_kwargs = {}

    predicted_environment = environment_constructor(solo_mode=solo_mode, **env_kwargs)
    predicted_environment.set_state(
        initialization_data=initialization_data,
        initialization_actions=initialization_actions,
        message_history=predicted_message_history,
    )

    gold_environment = environment_constructor(**env_kwargs)
    gold_environment.set_state(
        initialization_data=initialization_data,
        initialization_actions=initialization_actions,
        message_history=message_history,
    )
    golden_actions = task.evaluation_criteria.actions or []
    for action in golden_actions:
        try:
            gold_environment.make_tool_call(
                tool_name=action.name,
                requestor=action.requestor,
                **action.arguments,
            )
        except Exception as e:
            logger.warning(
                f"Error in golden actions {action.name}({action.arguments}): {e}"
            )

    agent_db_hash = gold_environment.get_db_hash()
    user_db_hash = gold_environment.get_user_db_hash()
    predicted_agent_db_hash = predicted_environment.get_db_hash()
    predicted_user_db_hash = predicted_environment.get_user_db_hash()
    db_match = agent_db_hash == predicted_agent_db_hash and user_db_hash == predicted_user_db_hash

    env_assertion_checks = []
    for env_assertion in task.evaluation_criteria.env_assertions or []:
        success = predicted_environment.run_env_assertion(
            env_assertion,
            raise_assertion_error=False,
        )
        env_assertion_checks.append(
            EnvAssertionEvaluation(
                env_assertion=env_assertion,
                met=success,
            )
        )

    return DBEvaluation(db_match=db_match), env_assertion_checks, {}


class EnvironmentEvaluator(EvaluatorBase[Message]):
    """
    Evaluator focuses on endstate of the simulation environment.
    """

    @classmethod
    def calculate_reward(
        cls,
        environment_constructor: Callable[[], Environment],
        task: Task,
        full_trajectory: list[
            Message
        ],  # FIXME: It would be better to be able to get only the messages that are after the initial state
        solo_mode: bool = False,
        env_kwargs: dict = None,
    ) -> RewardInfo:
        """
        Calculate the reward for the simulation.
        Args:
            environment_constructor: Callable[[], Environment]
            task: Task
            full_trajectory: list[Message] (Must include the message history from task initial state)
            solo_mode: bool
        Returns:
            RewardInfo
        """
        db_evaluation, env_evaluations, info = _evaluate_environment_state(
            environment_constructor,
            task,
            list(full_trajectory),
            solo_mode=solo_mode,
            env_kwargs=env_kwargs,
        )

        if db_evaluation is None:
            return RewardInfo(
                reward=1.0,
                info=info,
            )

        db_reward = 1.0 if db_evaluation.db_match else 0.0
        db_check = DBCheck(db_match=db_evaluation.db_match, db_reward=db_reward)
        env_assertion_checks = [
            EnvAssertionCheck(
                env_assertion=env_eval.env_assertion,
                met=env_eval.met,
                reward=1.0 if env_eval.met else 0.0,
            )
            for env_eval in env_evaluations
        ]
        env_assertion_reward = 1.0
        for env_assertion_check in env_assertion_checks:
            env_assertion_reward *= env_assertion_check.reward

        reward = 1.0
        reward_breakdown = {}
        if RewardType.DB in task.evaluation_criteria.reward_basis:
            reward_breakdown[RewardType.DB] = db_reward
            reward *= db_reward
        if RewardType.ENV_ASSERTION in task.evaluation_criteria.reward_basis:
            reward_breakdown[RewardType.ENV_ASSERTION] = env_assertion_reward
            reward *= env_assertion_reward

        return RewardInfo(
            reward=reward,
            db_check=db_check,
            env_assertions=env_assertion_checks,
            reward_basis=task.evaluation_criteria.reward_basis,
            reward_breakdown=reward_breakdown,
            info=info or None,
        )

    @classmethod
    def evaluate_environment(
        cls,
        environment_constructor: Callable[[], Environment],
        task: Task,
        full_trajectory: list[Message],
        solo_mode: bool = False,
        env_kwargs: dict = None,
    ) -> tuple[Optional[DBEvaluation], list[EnvAssertionEvaluation], dict]:
        return _evaluate_environment_state(
            environment_constructor,
            task,
            list(full_trajectory),
            solo_mode=solo_mode,
            env_kwargs=env_kwargs,
        )


class FullDuplexEnvironmentEvaluator(EvaluatorBase[Tick]):
    """
    Evaluator focuses on endstate of the simulation environment.
    """

    @classmethod
    def ticks_to_message_history(cls, ticks: list[Tick]) -> list[Message]:
        """
        Convert a list of Ticks to a message history suitable for Environment.set_state().

        The order follows the execution order in FullDuplexOrchestrator:
        - User tool calls are processed before agent tool calls within each tick
        - Each tool call message is followed by its corresponding tool results

        Args:
            ticks: List of Tick objects from full-duplex simulation.

        Returns:
            List of Messages in the format expected by Environment.set_state():
            [UserMessage with tool_calls, ToolMessage results, AssistantMessage with tool_calls, ToolMessage results, ...]
        """
        messages: list[Message] = []

        for tick in ticks:
            # 1. User tool calls first (processed before agent in orchestrator)
            if tick.user_tool_calls:
                user_msg = UserMessage(
                    role="user",
                    content=tick.user_chunk.content if tick.user_chunk else None,
                    tool_calls=tick.user_tool_calls,
                    timestamp=(
                        tick.user_chunk.timestamp if tick.user_chunk else tick.timestamp
                    ),
                    contains_speech=(
                        tick.user_chunk.contains_speech if tick.user_chunk else False
                    ),
                )
                messages.append(user_msg)
                messages.extend(tick.user_tool_results)

            # 2. Agent tool calls second
            if tick.agent_tool_calls:
                agent_msg = AssistantMessage(
                    role="assistant",
                    content=tick.agent_chunk.content if tick.agent_chunk else None,
                    tool_calls=tick.agent_tool_calls,
                    timestamp=(
                        tick.agent_chunk.timestamp
                        if tick.agent_chunk
                        else tick.timestamp
                    ),
                    contains_speech=(
                        tick.agent_chunk.contains_speech if tick.agent_chunk else False
                    ),
                )
                messages.append(agent_msg)
                messages.extend(tick.agent_tool_results)

        return messages

    @classmethod
    def calculate_reward(
        cls,
        environment_constructor: Callable[[], Environment],
        task: Task,
        full_trajectory: list[Tick],
        solo_mode: bool = False,
        env_kwargs: dict = None,
    ) -> RewardInfo:
        """
        Calculate the reward for the simulation.
        Args:
            environment_constructor: Callable[[], Environment]
            task: Task
            full_trajectory: list[Tick]
            solo_mode: bool
            env_kwargs: dict
        Returns:
            RewardInfo
        """
        predicted_message_history = cls.ticks_to_message_history(full_trajectory)
        db_evaluation, env_evaluations, info = _evaluate_environment_state(
            environment_constructor,
            task,
            predicted_message_history,
            solo_mode=solo_mode,
            env_kwargs=env_kwargs,
        )

        if db_evaluation is None:
            return RewardInfo(
                reward=1.0,
                info=info,
            )

        db_reward = 1.0 if db_evaluation.db_match else 0.0
        db_check = DBCheck(db_match=db_evaluation.db_match, db_reward=db_reward)
        env_assertion_checks = [
            EnvAssertionCheck(
                env_assertion=env_eval.env_assertion,
                met=env_eval.met,
                reward=1.0 if env_eval.met else 0.0,
            )
            for env_eval in env_evaluations
        ]
        env_assertion_reward = 1.0
        for env_assertion_check in env_assertion_checks:
            env_assertion_reward *= env_assertion_check.reward

        reward = 1.0
        reward_breakdown = {}
        if RewardType.DB in task.evaluation_criteria.reward_basis:
            reward_breakdown[RewardType.DB] = db_reward
            reward *= db_reward
        if RewardType.ENV_ASSERTION in task.evaluation_criteria.reward_basis:
            reward_breakdown[RewardType.ENV_ASSERTION] = env_assertion_reward
            reward *= env_assertion_reward

        return RewardInfo(
            reward=reward,
            db_check=db_check,
            env_assertions=env_assertion_checks,
            reward_basis=task.evaluation_criteria.reward_basis,
            reward_breakdown=reward_breakdown,
            info=info or None,
        )

    @classmethod
    def evaluate_environment(
        cls,
        environment_constructor: Callable[[], Environment],
        task: Task,
        full_trajectory: list[Tick],
        solo_mode: bool = False,
        env_kwargs: dict = None,
    ) -> tuple[Optional[DBEvaluation], list[EnvAssertionEvaluation], dict]:
        return _evaluate_environment_state(
            environment_constructor,
            task,
            cls.ticks_to_message_history(full_trajectory),
            solo_mode=solo_mode,
            env_kwargs=env_kwargs,
        )
