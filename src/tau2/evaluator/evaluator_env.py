from typing import Callable

from loguru import logger

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    Tick,
    UserMessage,
)
from tau2.data_model.simulation import DBCheck, EnvAssertionCheck, RewardInfo
from tau2.data_model.tasks import Action, RewardType, Task
from tau2.environment.environment import Environment
from tau2.evaluator.evaluator_base import EvaluatorBase


def replay_golden_actions(
    gold_environment: Environment,
    golden_actions: list[Action],
    task_id: str,
) -> list[dict]:
    """Replay the reference trajectory on a fresh gold environment.

    Every golden action is executed, in order, exactly as the loop this helper
    replaces always did: the calls the gold environment receives, and therefore
    the DB end state that is hashed afterwards, are unchanged. What it adds is a
    *classification* of the failures it swallows, not a filter on the replay.

    A read-only action that raises cannot have moved the DB, so it keeps the old
    treatment -- a warning naming the task, the action and the exception -- and
    does not invalidate the gold state. A *state-mutating* action that raised
    asked for a DB change that was never applied, so the gold end state is
    incomplete and no verdict taken from it is usable; those are returned to the
    caller (see ``_resolve_db_check``).

    Skipping the non-mutating actions instead of running them -- the way
    ``Environment.set_state`` does for the *predicted* trajectory -- was measured
    and rejected: it changes no gold hash on the shipped tasks, so it buys no
    correctness, and it erases the only record that those references point at
    users, orders and products the DB does not contain.

    Args:
        gold_environment: The fresh environment the reference is replayed on.
        golden_actions: ``task.evaluation_criteria.actions``, one reference
            trajectory. Not a per-call requirement on the agent.
        task_id: Used to make a failure attributable in the log.

    Returns:
        One record per state-mutating action that raised. Such an action asked
        for a DB change that the gold environment never applied, so the gold end
        state -- and therefore any verdict derived from it -- is unavailable.
    """
    failed_actions = []
    for action in golden_actions:
        mutating = gold_environment.is_mutating_tool(action.name)
        try:
            gold_environment.make_tool_call(
                tool_name=action.name,
                requestor=action.requestor,
                **action.arguments,
            )
        except Exception as e:
            if not mutating:
                logger.warning(
                    f"Gold replay for task {task_id}: read-only golden action "
                    f"{action.action_id} ({action.name}, "
                    f"requestor={action.requestor}) raised "
                    f"{type(e).__name__}: {e} (arguments={action.arguments}). The "
                    "call cannot change the database, so the gold state and the "
                    "DB verdict are unaffected by it."
                )
                continue
            failed_actions.append(
                {
                    "action_id": action.action_id,
                    "tool_name": action.name,
                    "requestor": action.requestor,
                    "error_type": type(e).__name__,
                    "error": str(e),
                }
            )
            logger.error(
                f"Gold replay failed for task {task_id}: action "
                f"{action.action_id} ({action.name}, requestor={action.requestor}) "
                f"raised {type(e).__name__}: {e}. The reference trajectory could "
                "not be applied, so the gold database state is incomplete."
            )
    return failed_actions


def _resolve_db_check(
    db_match: bool,
    db_reward: float,
    failed_golden_actions: list[dict],
    task_id: str,
) -> tuple[DBCheck | None, dict]:
    """Build the DB check, refusing to certify a match on an incomplete gold state.

    A reference trajectory whose state-mutating actions raised never reached the
    end state the task author described, so the gold hash is one the agent is not
    expected to reproduce -- a match against it says nothing. Rule 4 of
    ``src/tau2/evaluator/AGENTS.md`` defines an unavailable criterion as "not
    evaluated" rather than "failed", and ``RewardInfo.db_check`` already has a
    value for that (``None``, which ``metrics/agent_metrics.py`` counts in
    ``db_not_checked``).

    A *mismatch* keeps reporting ``db_reward=0.0``: it is already the failing
    verdict, and downgrading it to "not evaluated" would raise the reward of an
    agent that did the wrong thing. Since only a match (``db_reward=1.0``) is
    dropped, and 1.0 is the identity of the multiplicative reward, this cannot
    change any total reward.

    The diagnostics travel with the verdict in both directions. A ``0.0`` scored
    against a gold state that could not be built is the outcome most likely to be
    mis-read as "the model failed", so it has to carry its own provenance; a
    trajectory whose reference replayed cleanly gets an empty ``info``, exactly
    as before.

    Returns:
        The DB check (``None`` when the match is not certifiable) and the info
        records to attach to ``RewardInfo.info``.
    """
    if not failed_golden_actions:
        return DBCheck(db_match=db_match, db_reward=db_reward), {}
    if db_match:
        logger.error(
            f"Task {task_id}: {len(failed_golden_actions)} state-mutating golden "
            "action(s) could not be applied, so the DB hash match is not a pass. "
            "Reporting db_check=None instead of db_reward=1.0."
        )
        return None, {
            "gold_replay_incomplete": True,
            "failed_mutating_actions": failed_golden_actions,
            "db_evaluated": False,
        }
    logger.warning(
        f"Task {task_id}: the DB check reports a mismatch, but "
        f"{len(failed_golden_actions)} state-mutating golden action(s) could not "
        "be applied, so the gold state it was compared against is incomplete. "
        "Keeping db_reward=0.0 -- downgrading a failing verdict to 'not "
        "evaluated' would upgrade the score of an agent that did the wrong "
        "thing -- and recording the reason in RewardInfo.info."
    )
    return DBCheck(db_match=db_match, db_reward=db_reward), {
        "gold_replay_incomplete": True,
        "failed_mutating_actions": failed_golden_actions,
        "db_evaluated": True,
    }


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
        failed_golden_actions = replay_golden_actions(
            gold_environment, golden_actions, task_id=task.id
        )

        # Comparing the environments
        agent_db_hash = gold_environment.get_db_hash()
        user_db_hash = gold_environment.get_user_db_hash()
        predicted_agent_db_hash = predicted_environment.get_db_hash()
        predicted_user_db_hash = predicted_environment.get_user_db_hash()
        agent_db_match = agent_db_hash == predicted_agent_db_hash
        user_db_match = user_db_hash == predicted_user_db_hash
        if agent_db_match and user_db_match:
            db_reward = 1.0
            db_match = True
        else:
            db_reward = 0.0
            db_match = False

        db_check, replay_info = _resolve_db_check(
            db_match=db_match,
            db_reward=db_reward,
            failed_golden_actions=failed_golden_actions,
            task_id=task.id,
        )

        # Run env assertions
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
        if (
            RewardType.DB in task.evaluation_criteria.reward_basis
            and db_check is not None
        ):
            reward_breakdown[RewardType.DB] = db_check.db_reward
            reward *= db_check.db_reward
        if RewardType.ENV_ASSERTION in task.evaluation_criteria.reward_basis:
            reward_breakdown[RewardType.ENV_ASSERTION] = env_assertion_reward
            reward *= env_assertion_reward

        return RewardInfo(
            reward=reward,
            db_check=db_check,
            env_assertions=env_assertion_checks,
            reward_basis=task.evaluation_criteria.reward_basis,
            reward_breakdown=reward_breakdown,
            info=replay_info or None,
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
        failed_golden_actions = replay_golden_actions(
            gold_environment, golden_actions, task_id=task.id
        )

        # Comparing the environments
        agent_db_hash = gold_environment.get_db_hash()
        user_db_hash = gold_environment.get_user_db_hash()
        predicted_agent_db_hash = predicted_environment.get_db_hash()
        predicted_user_db_hash = predicted_environment.get_user_db_hash()
        agent_db_match = agent_db_hash == predicted_agent_db_hash
        user_db_match = user_db_hash == predicted_user_db_hash
        if agent_db_match and user_db_match:
            db_reward = 1.0
            db_match = True
        else:
            db_reward = 0.0
            db_match = False

        db_check, replay_info = _resolve_db_check(
            db_match=db_match,
            db_reward=db_reward,
            failed_golden_actions=failed_golden_actions,
            task_id=task.id,
        )

        # Run env assertions
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
        if (
            RewardType.DB in task.evaluation_criteria.reward_basis
            and db_check is not None
        ):
            reward_breakdown[RewardType.DB] = db_check.db_reward
            reward *= db_check.db_reward
        if RewardType.ENV_ASSERTION in task.evaluation_criteria.reward_basis:
            reward_breakdown[RewardType.ENV_ASSERTION] = env_assertion_reward
            reward *= env_assertion_reward

        return RewardInfo(
            reward=reward,
            db_check=db_check,
            env_assertions=env_assertion_checks,
            reward_basis=task.evaluation_criteria.reward_basis,
            reward_breakdown=reward_breakdown,
            info=replay_info or None,
        )
