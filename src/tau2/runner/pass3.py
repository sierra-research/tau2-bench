"""Two explicit Pass^3 designs: repeated trials and crossed voice conditions."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from tau2.data_model.simulation import Info, Results, TerminationReason
from tau2.data_model.tasks import Task


class CrossedCondition(BaseModel):
    """One versioned realization of the established R/C/S voice cross."""

    model_config = ConfigDict(frozen=True)

    label: str
    seed: int
    channel_effects_mode: Literal["light", "regular", "heavy"]
    speech_effects_mode: Literal["light", "regular", "heavy"]


# Match the established robustness v1.2.0 whole-run seeds. The regular seed
# also matches ordinary seed-42 voice runs, which can be supplied read-only.
CROSSED_PASS3_VERSION = "voice-v2-rcs-v3"
CROSSED_CONDITIONS = (
    CrossedCondition(
        label="regular",
        seed=42,
        channel_effects_mode="regular",
        speech_effects_mode="regular",
    ),
    CrossedCondition(
        label="channel_heavy",
        seed=401,
        channel_effects_mode="heavy",
        speech_effects_mode="light",
    ),
    CrossedCondition(
        label="speech_heavy",
        seed=16319,
        channel_effects_mode="light",
        speech_effects_mode="heavy",
    ),
)


class TaskCrossedPass3(BaseModel):
    """One task's success vector in R/C/S order."""

    task_id: str
    passes: tuple[bool | None, bool | None, bool | None]
    passed_all: bool


class CrossedPass3Report(BaseModel):
    """Provenance-bearing, reproducible aggregate of three aligned runs."""

    version: str = CROSSED_PASS3_VERSION
    domain: str
    language: str | None
    conditions: tuple[CrossedCondition, CrossedCondition, CrossedCondition]
    source_paths: dict[str, str]
    source_git_commits: dict[str, str]
    tasks: list[TaskCrossedPass3]
    passed_all: int = Field(ge=0)
    total_tasks: int = Field(gt=0)
    scored_tasks: int = Field(ge=1)
    infrastructure_error_task_ids: list[str]
    pass3: float = Field(ge=0, le=1)


STABLE_INFO_FIELDS = (
    "domain_tools_version",
    "voice_instruction_prompt_version",
    "language",
    "pack_extension",
    "localize_db",
    "language_artifacts",
    "language_tasks",
    "communicate_scoring",
    "benchmark",
    "elicitation",
    "elicitation_scoring_version",
    "elicitation_prompt_version",
    "elicitation_catalog_version",
    "elicitation_spell_detection_version",
    "num_trials",
    "max_steps",
    "max_errors",
    "user_info",
    "agent_info",
    "environment_info",
    "text_streaming_config",
    "speech_complexity",
    "audio_native_config",
    "retrieval_config",
    "retrieval_config_kwargs",
)


def _stable_value(value: object) -> object:
    """Ignore newly added, unset optional fields in stored model contracts."""
    if isinstance(value, BaseModel):
        return value.model_dump(exclude_none=True)
    return value


def validate_regular_source(
    result: Results, expected_info: Info, expected_tasks: list[Task]
) -> None:
    """Verify an ordinary regular run against today's resolved benchmark frame."""
    mismatched = [
        field
        for field in STABLE_INFO_FIELDS
        if _stable_value(getattr(result.info, field))
        != _stable_value(getattr(expected_info, field))
    ]
    if result.tasks != expected_tasks:
        mismatched.append("tasks")
    if mismatched:
        raise ValueError(
            f"Regular source settings, scoring, pins or tasks differ: {mismatched}"
        )
    if (
        result.info.seed != CROSSED_CONDITIONS[0].seed
        or result.info.voice_sampling_seed is not None
        or result.info.pass3_condition is not None
        or result.info.channel_effects_mode != "regular"
        or result.info.speech_effects_mode != "regular"
    ):
        raise ValueError("Regular source is not an ordinary seed-42 run")
    index = result.simulation_index or result._build_simulation_index()
    task_ids = {str(task.id) for task in expected_tasks}
    if (
        len(task_ids) != len(expected_tasks)
        or len(index) != len(task_ids)
        or {str(entry.task_id) for entry in index} != task_ids
        or any(entry.trial != 0 for entry in index)
    ):
        raise ValueError("Regular source needs one indexed call per task")
    if any(
        entry.reward is None
        and entry.termination_reason != TerminationReason.INFRASTRUCTURE_ERROR
        for entry in index
    ):
        raise ValueError("Regular source has unscored non-infrastructure calls")


def summarize_crossed_pass3(
    results: dict[str, Results], source_paths: dict[str, str]
) -> CrossedPass3Report:
    """Score one result per task in each pinned condition; refuse partial grids."""
    expected = {condition.label for condition in CROSSED_CONDITIONS}
    if set(results) != expected or set(source_paths) != expected:
        raise ValueError(f"Crossed Pass^3 needs exactly {sorted(expected)}")
    first = results[CROSSED_CONDITIONS[0].label]
    domain = first.info.environment_info.domain_name
    language = first.info.language
    rows = []
    for condition in CROSSED_CONDITIONS:
        result = results[condition.label]
        if result.info.environment_info.domain_name != domain:
            raise ValueError("Crossed Pass^3 domains differ")
        if result.info.language != language:
            raise ValueError("Crossed Pass^3 languages differ")
        if (
            any(
                _stable_value(getattr(result.info, field))
                != _stable_value(getattr(first.info, field))
                for field in STABLE_INFO_FIELDS
            )
            or result.tasks != first.tasks
        ):
            raise ValueError("Crossed Pass^3 run settings or task frames differ")
        if (
            result.info.seed != condition.seed
            or result.info.voice_sampling_seed is not None
        ):
            raise ValueError("Crossed Pass^3 run seed does not match catalog")
        if (
            result.info.pass3_condition
            not in (
                (None, "regular")
                if condition.label == "regular"
                else (condition.label,)
            )
            or result.info.num_trials != 1
            or result.info.channel_effects_mode != condition.channel_effects_mode
            or result.info.speech_effects_mode != condition.speech_effects_mode
        ):
            raise ValueError("Crossed Pass^3 effects do not match catalog")
        index = result.simulation_index or result._build_simulation_index()
        if any(entry.trial != 0 for entry in index):
            raise ValueError("Crossed Pass^3 requires one trial per task")
        rewards = {str(entry.task_id): entry.reward for entry in index}
        if not rewards or len(rewards) != len(index):
            raise ValueError("Crossed Pass^3 needs unique completed task results")
        expected_tasks = {str(task.id) for task in result.tasks}
        if len(expected_tasks) != len(result.tasks) or set(rewards) != expected_tasks:
            raise ValueError("Crossed Pass^3 has missing or unexpected task results")
        if any(
            entry.reward is None
            and entry.termination_reason != TerminationReason.INFRASTRUCTURE_ERROR
            for entry in index
        ):
            raise ValueError("Crossed Pass^3 has unscored non-infrastructure results")
        rows.append(rewards)
    task_ids = set(rows[0])
    if any(set(row) != task_ids for row in rows[1:]):
        raise ValueError("Crossed Pass^3 conditions have different task sets")
    infrastructure_error_task_ids = sorted(
        task_id for task_id in task_ids if any(row[task_id] is None for row in rows)
    )
    scored_tasks = len(task_ids) - len(infrastructure_error_task_ids)
    if not scored_tasks:
        raise ValueError("Crossed Pass^3 has no fully scored tasks")
    tasks = [
        TaskCrossedPass3(
            task_id=task_id,
            passes=tuple(
                row[task_id] == 1.0 if row[task_id] is not None else None
                for row in rows
            ),
            passed_all=all(row[task_id] == 1.0 for row in rows),
        )
        for task_id in sorted(task_ids)
    ]
    passed_all = sum(task.passed_all for task in tasks)
    return CrossedPass3Report(
        domain=domain,
        language=language,
        conditions=CROSSED_CONDITIONS,
        source_paths=source_paths,
        source_git_commits={
            condition.label: results[condition.label].info.git_commit
            for condition in CROSSED_CONDITIONS
        },
        tasks=tasks,
        passed_all=passed_all,
        total_tasks=len(tasks),
        scored_tasks=scored_tasks,
        infrastructure_error_task_ids=infrastructure_error_task_ids,
        pass3=passed_all / scored_tasks,
    )
