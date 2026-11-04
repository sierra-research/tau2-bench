# Copyright Sierra
"""Intake domain construction and task loading for the selected strategy."""

from collections import Counter
from copy import deepcopy
from typing import Annotated, Optional

from pydantic import Field, RootModel, ValidationError, model_validator

from tau2.config import DEFAULT_ELICITATION_STRATEGY
from tau2.data_model.elicitation import ElicitationStrategy
from tau2.data_model.tasks import Task
from tau2.domains.intake.data_model import IntakeDB
from tau2.domains.intake.prompts import render_callback_order
from tau2.domains.intake.tools import ScaffoldedIntakeTools
from tau2.domains.intake.user_data_model import IntakeUserDB
from tau2.domains.intake.user_tools import IntakeUserTools
from tau2.domains.intake.utils import (
    INTAKE_DATA_DIR,
    INTAKE_DB_PATH,
    INTAKE_SCAFFOLDED_POLICY_PATH,
    INTAKE_SCAFFOLDED_TEXT_POLICY_PATH,
    INTAKE_TASK_SET_PATH,
    INTAKE_USER_DB_PATH,
)
from tau2.environment.environment import Environment
from tau2.utils import load_file
from tau2.utils.pydantic_utils import BaseModelNoExtra

# Agent policy per run channel. The voice policy scripts spoken-channel
# mechanics — spell-out pinning, per-entity read-backs, capture logging —
# that have no typed analogue, so text runs load the text policy instead
# (the typed value is authoritative as written). Tools, DB, and tasks are
# channel-invariant; only the policy the agent reads changes.
_POLICY_PATHS = {
    "voice": INTAKE_SCAFFOLDED_POLICY_PATH,
    "text": INTAKE_SCAFFOLDED_TEXT_POLICY_PATH,
}


def populate_callback_order(environment: Environment, task: Task) -> None:
    """Add the selected task's work order to its policy before agent construction.

    Replay setup on an isolated copy: the orchestrator still owns initialization
    of the live environment. Only the typed callback order reaches the prompt;
    seeded record values and the caller's private entities remain hidden.
    """
    prepared = deepcopy(environment)
    initial = deepcopy(task.initial_state)
    prepared.set_state(
        initialization_data=initial.initialization_data if initial else None,
        initialization_actions=initial.initialization_actions if initial else None,
        message_history=(initial.message_history or []) if initial else [],
    )
    order = prepared.tools.get_callback_order()
    environment.policy = (
        f"{environment.policy.rstrip()}\n\n{render_callback_order(order)}"
    )


def get_environment(
    db: Optional[IntakeDB] = None,
    policies: dict[str, str] | None = None,
    user_db: Optional[IntakeUserDB] = None,
    solo_mode: bool = False,
    channel: str = "voice",
    task: Optional[Task] = None,
    strategy: ElicitationStrategy = ElicitationStrategy(DEFAULT_ELICITATION_STRATEGY),
) -> Environment:
    """Build the intake environment with its configured elicitation strategy.

    Agent-directed voice capture is the default. The scaffolded ablation
    supports voice and text, with a separate policy for each channel. The
    runner supplies the strategy, channel, and selected task. When present,
    the task's work order is added before agent construction, without
    initializing the live database early.
    """
    strategy = ElicitationStrategy(strategy)
    if strategy is ElicitationStrategy.AGENT_DIRECTED:
        from tau2.domains.intake.agent_directed import (
            get_environment as get_agent_directed_environment,
        )

        return get_agent_directed_environment(
            db=db,
            user_db=user_db,
            solo_mode=solo_mode,
            channel=channel,
            task=task,
            policy=policies["agent_directed_policy"] if policies else None,
        )
    if solo_mode:
        raise ValueError("Solo mode not supported for intake domain")
    if channel not in _POLICY_PATHS:
        raise ValueError(
            f"Unknown scaffolded intake channel {channel!r}: the policy is per channel "
            f"and only knows {sorted(_POLICY_PATHS)}"
        )
    if db is None:
        db = IntakeDB.load(INTAKE_DB_PATH)
    if user_db is None:
        user_db = IntakeUserDB.load(INTAKE_USER_DB_PATH)
    policy = (
        policies[_POLICY_PATHS[channel].stem]
        if policies
        else load_file(_POLICY_PATHS[channel])
    )
    environment = Environment(
        domain_name="intake",
        policy=policy,
        tools=ScaffoldedIntakeTools(db),
        user_tools=IntakeUserTools(user_db),
    )
    if task is not None:
        populate_callback_order(environment, task)
    return environment


class IntakeTaskFile(BaseModelNoExtra):
    """The serialized collection of elicitation tasks."""

    tasks: Annotated[list[Task], Field(description="The frozen task collection.")]

    @model_validator(mode="after")
    def _unique_task_ids(self) -> "IntakeTaskFile":
        duplicates = sorted(
            task_id
            for task_id, count in Counter(task.id for task in self.tasks).items()
            if count > 1
        )
        if duplicates:
            raise ValueError(f"Duplicate task id(s): {', '.join(duplicates)}")
        return self


class IntakeTaskSplits(RootModel[dict[str, list[str]]]):
    """Named task selections, with a required base evaluation split."""

    @model_validator(mode="after")
    def _valid_splits(self) -> "IntakeTaskSplits":
        if "base" not in self.root:
            raise ValueError("Task splits must contain a 'base' split")
        for name, task_ids in self.root.items():
            if len(task_ids) != len(set(task_ids)):
                raise ValueError(f"Split {name!r} contains duplicate task ids")
        return self


def get_tasks(
    task_split_name: Optional[str] = "base",
    strategy: ElicitationStrategy = ElicitationStrategy(DEFAULT_ELICITATION_STRATEGY),
) -> list[Task]:
    """Load the shared task set with the selected strategy's caller/tool contract."""
    from tau2.domains.intake.scoring import project_scoring_folds

    strategy = ElicitationStrategy(strategy)
    if strategy is ElicitationStrategy.AGENT_DIRECTED:
        from tau2.domains.intake.agent_directed import (
            get_tasks as get_agent_directed_tasks,
        )

        return get_agent_directed_tasks(task_split_name)
    return [
        english_time_display_task(project_scoring_folds(task))
        for task in load_frozen_tasks(task_split_name)
    ]


def english_time_display_task(task: Task) -> Task:
    """Project English caller clock phrases without changing records or tool gold.

    Keep this outside ``load_frozen_tasks``: inventory extraction and localized
    compilation must continue to receive the exact frozen protocol values.
    """
    from tau2.domains.intake.folds import FoldKind
    from tau2.domains.intake.time_display import (
        TIME_CALLER_GUIDANCE,
        display_time_payload,
    )

    task = task.model_copy(deep=True)
    calls = {
        action.func_name: action.arguments
        for action in task.initial_state.initialization_actions
    }
    time_fields = {
        field["name"]
        for field in calls["seed_record"]["record"]["fields"]
        if field["fold"] == FoldKind.TIME.value
    }
    entities = calls["set_entities"]["entities"]
    caller_time_fields = time_fields & entities.keys()
    for name in caller_time_fields:
        entities[name] = display_time_payload(entities[name], "en")
    if caller_time_fields:
        instructions = task.user_scenario.instructions
        if isinstance(instructions, str):
            raise ValueError(
                "English time display requires structured intake instructions"
            )
        if TIME_CALLER_GUIDANCE not in instructions.task_instructions:
            instructions.task_instructions += (
                "\n\n# Intake written clarification\n" + TIME_CALLER_GUIDANCE
            )
    return task


def load_frozen_tasks(task_split_name: Optional[str] = "base") -> list[Task]:
    """Load raw validated tasks before strategy or caller time adaptation."""
    try:
        tasks = IntakeTaskFile.model_validate(load_file(INTAKE_TASK_SET_PATH)).tasks
    except ValidationError as error:
        raise ValueError(
            f"Malformed task file {INTAKE_TASK_SET_PATH}: {error}"
        ) from error
    task_ids = {task.id for task in tasks}
    if task_split_name is None:
        return tasks
    task_splits = get_tasks_split()
    if task_split_name not in task_splits:
        raise ValueError(
            f"Invalid task split name: {task_split_name}. "
            f"Valid splits are: {sorted(task_splits.keys())}"
        )
    split_ids = task_splits[task_split_name]
    unknown_ids = sorted(set(split_ids) - task_ids)
    if unknown_ids:
        raise ValueError(
            f"Split '{task_split_name}' references unknown task id(s): {', '.join(unknown_ids)}"
        )
    return [task for task in tasks if task.id in split_ids]


def get_tasks_split() -> dict[str, list[str]]:
    """Load the task splits. The file must exist and contain a 'base' split."""
    split_path = INTAKE_DATA_DIR / f"split_{INTAKE_TASK_SET_PATH.stem}.json"
    if not split_path.exists():
        raise FileNotFoundError(f"Task split file not found: {split_path}")
    return IntakeTaskSplits.model_validate(load_file(split_path)).root
