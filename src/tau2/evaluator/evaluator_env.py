from typing import Callable

from loguru import logger

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    Tick,
    ToolCall,
    UserMessage,
)
from tau2.data_model.simulation import (
    DBCheck,
    EnvAssertionCheck,
    ReplayedAction,
    RewardInfo,
)
from tau2.data_model.tasks import RewardType, Task
from tau2.environment.environment import Environment
from tau2.evaluator.evaluator_base import EvaluatorBase


def _tool_calls_from_messages(messages: list[Message]) -> list[ToolCall]:
    """Extract recorded tool calls in the order used by environment replay."""
    tool_calls = []
    for message in messages:
        if isinstance(message, (AssistantMessage, UserMessage)):
            tool_calls.extend(message.tool_calls or [])
    return tool_calls


def _replayed_trajectory_tool_calls(
    environment: Environment,
    messages: list[Message],
) -> list[ToolCall]:
    """Return the trajectory calls that ``Environment.set_state`` replays."""
    return [
        tool_call
        for tool_call in _tool_calls_from_messages(messages)
        if environment._has_tool(tool_call.name)
        and environment._is_mutating_tool(tool_call.name)
    ]


def _reference_tool_calls(task: Task) -> list[ToolCall]:
    """Convert the task's reference actions into serializable tool calls."""
    return [
        ToolCall(
            id=action.action_id,
            name=action.name,
            arguments=action.arguments,
            requestor=action.requestor,
        )
        for action in (task.evaluation_criteria.actions or [])
    ]


def _provenance_warnings(
    evaluation_mode: str,
    strict_replay: bool,
) -> list[str]:
    """Build warnings that keep replay results distinct from live results."""
    warnings = []
    if evaluation_mode == "replay":
        warnings.append(
            "evaluation_mode='replay': the candidate state was reconstructed "
            "from recorded tool calls and does not certify live environment state."
        )
    if not strict_replay:
        warning = (
            "strict_replay=False: recorded tool outputs were not required to "
            "match replayed outputs."
        )
        if evaluation_mode == "replay":
            warning += " Reward reflects replayed state only."
        else:
            warning += " The live candidate state was not reconstructed."
        warnings.append(warning)
    return warnings


def _build_state_reward(
    predicted_environment: Environment,
    gold_environment: Environment,
    task: Task,
    *,
    evaluation_mode: str,
    state_source: str,
    replayed_actions: list[ReplayedAction],
    strict_replay: bool,
) -> RewardInfo:
    """Score two environments and attach explicit evaluator provenance."""
    agent_db_hash = gold_environment.get_db_hash()
    user_db_hash = gold_environment.get_user_db_hash()
    predicted_agent_db_hash = predicted_environment.get_db_hash()
    predicted_user_db_hash = predicted_environment.get_user_db_hash()
    agent_db_match = agent_db_hash == predicted_agent_db_hash
    user_db_match = user_db_hash == predicted_user_db_hash
    db_match = agent_db_match and user_db_match
    db_reward = 1.0 if db_match else 0.0
    db_check = DBCheck(db_match=db_match, db_reward=db_reward)

    env_assertions = task.evaluation_criteria.env_assertions or []
    env_assertion_checks = []
    env_assertion_reward = 1.0
    for env_assertion in env_assertions:
        success = predicted_environment.run_env_assertion(
            env_assertion,
            raise_assertion_error=False,
        )
        res = EnvAssertionCheck(
            env_assertion=env_assertion,
            met=success,
            reward=1.0 if success else 0.0,
        )
        env_assertion_checks.append(res)
        env_assertion_reward *= res.reward

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
        evaluation_mode=evaluation_mode,
        state_source=state_source,
        replayed_actions=replayed_actions,
        warnings=_provenance_warnings(evaluation_mode, strict_replay),
    )


class EnvironmentEvaluator(EvaluatorBase[Message]):
    """
    Evaluator focuses on endstate of the simulation environment.
    """

    @classmethod
    def evaluate_live(
        cls,
        environment: Environment,
        environment_constructor: Callable[[], Environment],
        task: Task,
        solo_mode: bool = False,
        env_kwargs: dict = None,
        strict_replay: bool = True,
    ) -> RewardInfo:
        """Evaluate the current environment state without replaying a trajectory.

        The reference state is still constructed by replaying the task's
        reference actions in a fresh environment.  The candidate state remains
        the caller-provided live environment, so a proxy-intercepted write is
        not mistaken for a successful live task merely because replay succeeds.
        """
        if task.evaluation_criteria is None:
            return RewardInfo(
                reward=1.0,
                evaluation_mode="live",
                state_source="live",
                info={"note": "No evaluation criteria"},
            )
        expected_actions = task.evaluation_criteria.actions
        env_assertions = task.evaluation_criteria.env_assertions
        if expected_actions is None and env_assertions is None:
            return RewardInfo(
                reward=1.0,
                db_check=DBCheck(db_match=True, db_reward=1.0),
                evaluation_mode="live",
                state_source="live",
                info={"note": "No expected actions or env assertions"},
            )

        initialization_data = None
        if (
            task.initial_state is not None
            and task.initial_state.initialization_data is not None
        ):
            initialization_data = task.initial_state.initialization_data

        initialization_actions = None
        if (
            task.initial_state is not None
            and task.initial_state.initialization_actions is not None
        ):
            initialization_actions = task.initial_state.initialization_actions

        message_history = []
        if (
            task.initial_state is not None
            and task.initial_state.message_history is not None
        ):
            message_history = task.initial_state.message_history

        if env_kwargs is None:
            env_kwargs = {}

        gold_environment = environment_constructor(**env_kwargs)
        gold_environment.set_state(
            initialization_data=initialization_data,
            initialization_actions=initialization_actions,
            message_history=message_history,
            strict=strict_replay,
        )
        for action in task.evaluation_criteria.actions or []:
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

        replayed_actions = [
            ReplayedAction(source="reference", tool_call=tool_call)
            for tool_call in _reference_tool_calls(task)
        ]
        return _build_state_reward(
            environment,
            gold_environment,
            task,
            evaluation_mode="live",
            state_source="live",
            replayed_actions=replayed_actions,
            strict_replay=strict_replay,
        )

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
        strict_replay: bool = True,
    ) -> RewardInfo:
        """
        Calculate the reward for the simulation.
        Args:
            environment_constructor: Callable[[], Environment]
            task: Task
            full_trajectory: list[Message] (Must include the message history from task initial state)
            solo_mode: bool
            strict_replay: forwarded to Environment.set_state(strict=...). Set
                False when re-grading historical trajectories whose recorded
                tool outputs may cosmetically differ from current tool code.
        Returns:
            RewardInfo
        """
        if task.evaluation_criteria is None:
            return RewardInfo(
                reward=1.0,
                info={"note": "No evaluation criteria"},
            )
        expected_actions = task.evaluation_criteria.actions
        env_assertions = task.evaluation_criteria.env_assertions
        if expected_actions is None and env_assertions is None:
            return RewardInfo(
                reward=1.0,
                db_check=DBCheck(db_match=True, db_reward=1.0),
                info={"note": "No expected actions or env assertions"},
            )

        initialization_data = None
        if (
            task.initial_state is not None
            and task.initial_state.initialization_data is not None
        ):
            initialization_data = task.initial_state.initialization_data

        initialization_actions = None
        if (
            task.initial_state is not None
            and task.initial_state.initialization_actions is not None
        ):
            initialization_actions = task.initial_state.initialization_actions

        message_history = []
        if (
            task.initial_state is not None
            and task.initial_state.message_history is not None
        ):
            message_history = task.initial_state.message_history

        if env_kwargs is None:
            env_kwargs = {}

        predicted_environment = environment_constructor(
            solo_mode=solo_mode, **env_kwargs
        )

        predicted_environment.set_state(
            initialization_data=initialization_data,
            initialization_actions=initialization_actions,
            message_history=list(full_trajectory),
            strict=strict_replay,
        )

        # Setting up gold environment
        gold_environment = environment_constructor(**env_kwargs)
        gold_environment.set_state(
            initialization_data=initialization_data,
            initialization_actions=initialization_actions,
            message_history=message_history,
            strict=strict_replay,
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

        replayed_actions = [
            *(
                ReplayedAction(source="trajectory", tool_call=tool_call)
                for tool_call in _replayed_trajectory_tool_calls(
                    predicted_environment,
                    full_trajectory,
                )
            ),
            *(
                ReplayedAction(source="reference", tool_call=tool_call)
                for tool_call in _reference_tool_calls(task)
            ),
        ]
        return _build_state_reward(
            predicted_environment,
            gold_environment,
            task,
            evaluation_mode="replay",
            state_source="replayed",
            replayed_actions=replayed_actions,
            strict_replay=strict_replay,
        )


class FullDuplexEnvironmentEvaluator(EvaluatorBase[Tick]):
    """
    Evaluator focuses on endstate of the simulation environment.
    """

    @classmethod
    def evaluate_live(
        cls,
        environment: Environment,
        environment_constructor: Callable[[], Environment],
        task: Task,
        solo_mode: bool = False,
        env_kwargs: dict = None,
        strict_replay: bool = True,
    ) -> RewardInfo:
        """Evaluate a current live state for a full-duplex simulation."""
        return EnvironmentEvaluator.evaluate_live(
            environment=environment,
            environment_constructor=environment_constructor,
            task=task,
            solo_mode=solo_mode,
            env_kwargs=env_kwargs,
            strict_replay=strict_replay,
        )

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
        strict_replay: bool = True,
    ) -> RewardInfo:
        """
        Calculate the reward for the simulation.
        Args:
            environment_constructor: Callable[[], Environment]
            task: Task
            full_trajectory: list[Tick]
            solo_mode: bool
            env_kwargs: dict
            strict_replay: forwarded to Environment.set_state(strict=...). Set
                False when re-grading historical trajectories whose recorded
                tool outputs may cosmetically differ from current tool code.
        Returns:
            RewardInfo
        """
        if env_kwargs is None:
            env_kwargs = {}
        if task.evaluation_criteria is None:
            return RewardInfo(
                reward=1.0,
                info={"note": "No evaluation criteria"},
            )
        expected_actions = task.evaluation_criteria.actions
        env_assertions = task.evaluation_criteria.env_assertions
        if expected_actions is None and env_assertions is None:
            return RewardInfo(
                reward=1.0,
                db_check=DBCheck(db_match=True, db_reward=1.0),
                info={"note": "No expected actions or env assertions"},
            )

        initialization_data = None
        if (
            task.initial_state is not None
            and task.initial_state.initialization_data is not None
        ):
            initialization_data = task.initial_state.initialization_data

        initialization_actions = None
        if (
            task.initial_state is not None
            and task.initial_state.initialization_actions is not None
        ):
            initialization_actions = task.initial_state.initialization_actions

        message_history = []
        if (
            task.initial_state is not None
            and task.initial_state.message_history is not None
        ):
            message_history = task.initial_state.message_history

        # Convert ticks to message history for set_state
        # Note: Audio native does not support task history, so we only use the simulation trajectory
        predicted_message_history = cls.ticks_to_message_history(full_trajectory)

        predicted_environment = environment_constructor(
            solo_mode=solo_mode, **env_kwargs
        )
        predicted_environment.set_state(
            initialization_data=initialization_data,
            initialization_actions=initialization_actions,
            message_history=predicted_message_history,
            strict=strict_replay,
        )

        # Setting up gold environment
        gold_environment = environment_constructor(**env_kwargs)
        gold_environment.set_state(
            initialization_data=initialization_data,
            initialization_actions=initialization_actions,
            message_history=message_history,
            strict=strict_replay,
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

        replayed_actions = [
            *(
                ReplayedAction(
                    source="trajectory",
                    tool_call=tool_call,
                )
                for tool_call in _replayed_trajectory_tool_calls(
                    predicted_environment,
                    predicted_message_history,
                )
            ),
            *(
                ReplayedAction(source="reference", tool_call=tool_call)
                for tool_call in _reference_tool_calls(task)
            ),
        ]
        return _build_state_reward(
            predicted_environment,
            gold_environment,
            task,
            evaluation_mode="replay",
            state_source="replayed",
            replayed_actions=replayed_actions,
            strict_replay=strict_replay,
        )
