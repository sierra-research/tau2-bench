"""Opt-in grading corrections for task/grader defects found in published tasks.

Official scoring never loads this module: a correction set is applied only when a
caller names one explicitly (``tau2 evaluate-trajs --task-corrections <name>``).
Corrected scores are therefore not leaderboard-comparable and must be reported
next to, never instead of, the official scores.

A correction changes only how a saved trajectory is graded; it never changes
what the user simulator was told. Defects whose fix needs different user
instructions or different data (for example an unaffordable gold exchange that
should become affordable) need a re-simulation and are out of scope here.

A correction set is a JSON file under ``data/tau2/task_corrections/``. Each entry
targets one canonical task (shared by every language variant) and may:

- edit the official gold actions into the corrected primary gold
  (``gold_edits``);
- accept further gold variants, each derived from the corrected primary gold
  (``alternatives``); the DB check passes when the predicted database matches
  any accepted gold database;
- replace the NL assertions or the communicate_info of the task, per language.

Edits address gold actions by tool name plus a subset of top-level arguments,
using only machine identifiers that the multilingual packs keep unchanged
(reservation, order, item and payment ids). Every edit must match at least one
gold action; a correction that no longer matches its task raises instead of
silently grading against drifted data.

Two scoping guards (``TaskCorrection.check_applies``) refuse a correction
outside the tasks it was written for:

- per-language values (amounts, prices, communicate_info) are written for the
  regional bundle of each language, so a non-English call is corrected only in
  a ``regional`` scenario; a ``canonical`` (USD) or unrecorded scenario raises;
- each entry pins ``task_content_digest`` of every task copy it was written
  against, per call language, so a task with the same id but different content
  (an older or newer task version) raises instead of being "corrected".
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from tau2.data_model.tasks import Action, Task
from tau2.utils.utils import DATA_DIR

TASK_CORRECTIONS_DIR = DATA_DIR / "tau2" / "task_corrections"

# Language key used when a per-language mapping has no entry for the call's
# language.
ANY_LANGUAGE = "*"

# Call language whose values are written for the canonical (registry) scenario;
# every other language's values are written for its regional bundle.
CANONICAL_LANGUAGE = "en"
REGIONAL_SCENARIO_MODE = "regional"


class TaskCorrectionMismatch(ValueError):
    """A correction is asked to grade a task or scenario it was not written for."""


def task_content_digest(task: Task) -> str:
    """sha256 of the canonical JSON of a task's user scenario and criteria.

    These are the parts a correction is written against: the user instructions
    that define the defect and the gold criteria it edits. The task id is
    excluded, so localized copies with different ids but equal content match.
    """
    payload = task.model_dump(
        mode="json", include={"user_scenario", "evaluation_criteria"}
    )
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


class ActionMatch(BaseModel):
    """Select gold actions by tool name and a subset of top-level arguments."""

    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)

    def matches(self, action: Action) -> bool:
        if action.name != self.name:
            return False
        return all(
            key in action.arguments and action.arguments[key] == value
            for key, value in self.arguments.items()
        )


class ActionEdit(BaseModel):
    """Drop the matched gold actions, or overwrite some of their arguments.

    ``arguments_by_language`` holds overrides whose values differ per language
    pack, such as amounts in the pack's currency scale; a ``set_arguments`` edit
    that has them must have an entry for every graded language.
    """

    model_config = ConfigDict(extra="forbid")

    match: ActionMatch
    op: Literal["drop", "set_arguments"]
    arguments: dict[str, Any] = Field(default_factory=dict)
    arguments_by_language: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def arguments_for(self, language: str) -> dict[str, Any]:
        if not self.arguments_by_language:
            return dict(self.arguments)
        return {
            **self.arguments,
            **_for_language(self.arguments_by_language, language, "edit arguments"),
        }


class GoldVariant(BaseModel):
    """One further accepted gold, derived from the corrected primary gold."""

    model_config = ConfigDict(extra="forbid")

    description: str
    edits: list[ActionEdit]


class TaskCorrection(BaseModel):
    """Grading correction for one canonical task in one domain."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Defect id, e.g. 'T1'.")
    domain: str
    canonical_task_id: str
    summary: str
    gold_edits: list[ActionEdit] = Field(default_factory=list)
    alternatives: list[GoldVariant] = Field(default_factory=list)
    nl_assertions: Optional[dict[str, list[str]]] = Field(
        default=None, description="Replacement NL assertions keyed by language."
    )
    communicate_info_any_of: Optional[dict[str, list[list[str]]]] = Field(
        default=None,
        description=(
            "Accepted communicate_info alternatives keyed by language; the check "
            "passes when any alternative passes."
        ),
    )
    task_digests: dict[str, list[str]] = Field(
        min_length=1,
        description=(
            "Accepted task_content_digest values keyed by call language: the "
            "task copies this correction was written against."
        ),
    )

    def check_applies(
        self, task: Task, *, language: str, scenario_mode: Optional[str]
    ) -> None:
        """Raise ``TaskCorrectionMismatch`` unless this correction fits ``task``.

        ``scenario_mode`` is the recorded ``LanguageTaskInfo.scenario_mode`` of
        the call, or None when the run recorded none.
        """
        where = f"{self.id} ({self.domain} task {self.canonical_task_id})"
        if language != CANONICAL_LANGUAGE and scenario_mode != REGIONAL_SCENARIO_MODE:
            raise TaskCorrectionMismatch(
                f"Correction {where} is written for regional {language!r} "
                f"scenarios, but the call's scenario mode is {scenario_mode!r}; "
                "its per-language values would not match this scenario."
            )
        accepted = self.task_digests.get(language)
        if accepted is None:
            raise TaskCorrectionMismatch(
                f"Correction {where} pins no task content for language {language!r}"
            )
        digest = task_content_digest(task)
        if digest not in accepted:
            raise TaskCorrectionMismatch(
                f"Correction {where} was written for different task content in "
                f"{language!r} (task {task.id} has digest {digest}); the task "
                "changed, so the correction must be re-validated."
            )


class TaskCorrectionSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    corrections: list[TaskCorrection]

    def content_digest(self) -> str:
        """Canonical digest of the complete grading definition recorded in results."""
        payload = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def lookup(self, domain: str, canonical_task_id: str) -> Optional[TaskCorrection]:
        """The entry for a canonical task id, without any applicability check.

        Grading must use ``for_task``, which also enforces the scoping guards.
        """
        found = [
            c
            for c in self.corrections
            if c.domain == domain and c.canonical_task_id == canonical_task_id
        ]
        if len(found) > 1:
            raise ValueError(
                f"Correction set {self.name} has {len(found)} entries for "
                f"{domain} task {canonical_task_id}"
            )
        return found[0] if found else None

    def for_task(
        self,
        domain: str,
        task: Task,
        *,
        canonical_task_id: str,
        language: str,
        scenario_mode: Optional[str],
    ) -> Optional[TaskCorrection]:
        """The entry that grades ``task``, or None; raises if it does not fit."""
        correction = self.lookup(domain, canonical_task_id)
        if correction is not None:
            correction.check_applies(
                task, language=language, scenario_mode=scenario_mode
            )
        return correction


def load_correction_set(
    name: str, directory: Path = TASK_CORRECTIONS_DIR
) -> TaskCorrectionSet:
    """Load ``<directory>/<name>.json``; the file's ``name`` must match."""
    path = directory / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Unknown task correction set: {path}")
    correction_set = TaskCorrectionSet.model_validate(json.loads(path.read_text()))
    if correction_set.name != name:
        raise ValueError(
            f"Correction set file {path} declares name {correction_set.name!r}"
        )
    return correction_set


def apply_edits(
    actions: list[Action], edits: list[ActionEdit], language: str = "en"
) -> list[Action]:
    """Return a new action list with the edits applied in order."""
    result = [action.model_copy(deep=True) for action in actions]
    for edit in edits:
        hits = [i for i, action in enumerate(result) if edit.match.matches(action)]
        if not hits:
            raise ValueError(
                f"Correction edit matches no gold action: {edit.match.model_dump()}"
            )
        if edit.op == "drop":
            result = [a for i, a in enumerate(result) if i not in set(hits)]
        else:
            for i in hits:
                result[i].arguments = {
                    **result[i].arguments,
                    **edit.arguments_for(language),
                }
    return result


def gold_variants(
    task: Task, correction: TaskCorrection, language: str = "en"
) -> list[list[Action]]:
    """Accepted gold action lists for ``language``, the corrected primary first."""
    if task.evaluation_criteria is None:
        raise ValueError(f"Task {task.id} has no evaluation criteria to correct")
    official = list(task.evaluation_criteria.actions or [])
    primary = apply_edits(official, correction.gold_edits, language)
    variants = [primary]
    variants.extend(
        apply_edits(primary, alt.edits, language) for alt in correction.alternatives
    )
    return variants


def _for_language(mapping: dict[str, Any], language: str, what: str) -> Any:
    if language in mapping:
        return mapping[language]
    if ANY_LANGUAGE in mapping:
        return mapping[ANY_LANGUAGE]
    raise ValueError(f"Correction has no {what} for language {language!r}")


def corrected_task(
    task: Task,
    correction: TaskCorrection,
    language: str,
    *,
    actions: Optional[list[Action]] = None,
    communicate_info: Optional[list[str]] = None,
) -> Task:
    """Copy of ``task`` with corrected criteria.

    ``actions`` defaults to the corrected primary gold and ``communicate_info``
    to the first accepted alternative for ``language``.
    """
    fixed = deepcopy(task)
    criteria = fixed.evaluation_criteria
    criteria.actions = (
        actions if actions is not None else gold_variants(task, correction, language)[0]
    )
    if correction.nl_assertions is not None:
        criteria.nl_assertions = list(
            _for_language(correction.nl_assertions, language, "NL assertions")
        )
    if correction.communicate_info_any_of is not None:
        criteria.communicate_info = (
            communicate_info
            if communicate_info is not None
            else list(
                _for_language(
                    correction.communicate_info_any_of, language, "communicate_info"
                )[0]
            )
        )
    return fixed


def communicate_alternatives(
    correction: TaskCorrection, language: str
) -> list[list[str]]:
    if correction.communicate_info_any_of is None:
        return []
    return [
        list(alt)
        for alt in _for_language(
            correction.communicate_info_any_of, language, "communicate_info"
        )
    ]


def _evaluate(evaluation_type, task: Task, **evaluate_kwargs):
    # Imported lazily: evaluator.py imports the domain registry.
    from tau2.evaluator.evaluator import evaluate_simulation

    return evaluate_simulation(
        task=task, evaluation_type=evaluation_type, **evaluate_kwargs
    )


def corrected_db_check(
    *, task: Task, correction: TaskCorrection, language: str, **evaluate_kwargs
) -> tuple[float, Optional[int]]:
    """DB reward against every accepted gold, without any LLM call.

    Returns ``(db_reward, index of the first matching gold variant or None)``.
    ``evaluate_kwargs`` are the keyword arguments of ``evaluate_simulation``
    other than ``task`` and ``evaluation_type``.
    """
    from tau2.evaluator.evaluator import EvaluationType

    for index, actions in enumerate(gold_variants(task, correction, language)):
        variant = corrected_task(task, correction, language, actions=actions)
        info = _evaluate(EvaluationType.ENV, variant, **evaluate_kwargs)
        if info.db_check is not None and info.db_check.db_match:
            return 1.0, index
    return 0.0, None


def evaluate_simulation_with_correction(
    *,
    task: Task,
    correction: TaskCorrection,
    language: str,
    evaluation_type,
    **evaluate_kwargs,
):
    """Full grading of one simulation under ``correction``.

    Grades the corrected primary task with ``evaluate_simulation`` (which calls
    the NL/communicate judges the task's reward basis needs), then accepts any
    further gold variant for the DB check and any further communicate_info
    alternative. The returned ``RewardInfo.info["task_correction"]`` records
    which alternatives matched.
    """
    from tau2.data_model.simulation import DBCheck
    from tau2.data_model.tasks import RewardType
    from tau2.evaluator.evaluator import EvaluationType

    primary = corrected_task(task, correction, language)
    reward_info = _evaluate(evaluation_type, primary, **evaluate_kwargs)
    breakdown = dict(reward_info.reward_breakdown or {})
    note: dict[str, Any] = {"id": correction.id, "gold_variant": None}
    if reward_info.db_check is not None and reward_info.db_check.db_match:
        note["gold_variant"] = 0
    elif RewardType.DB in breakdown:
        db_reward, index = corrected_db_check(
            task=task, correction=correction, language=language, **evaluate_kwargs
        )
        if index is not None:
            breakdown[RewardType.DB] = db_reward
            reward_info.db_check = DBCheck(db_match=True, db_reward=db_reward)
            note["gold_variant"] = index
    alternatives = communicate_alternatives(correction, language)
    note["communicate_alternative"] = (
        0 if breakdown.get(RewardType.COMMUNICATE) else None
    )
    if breakdown.get(RewardType.COMMUNICATE) == 0.0:
        for index, info in enumerate(alternatives[1:], start=1):
            variant = corrected_task(task, correction, language, communicate_info=info)
            comm = _evaluate(EvaluationType.COMMUNICATE, variant, **evaluate_kwargs)
            if comm.reward == 1.0:
                breakdown[RewardType.COMMUNICATE] = 1.0
                reward_info.communicate_checks = comm.communicate_checks
                note["communicate_alternative"] = index
                break
    if breakdown:
        reward = 1.0
        for value in breakdown.values():
            reward *= value
        reward_info.reward = reward
        reward_info.reward_breakdown = breakdown
    reward_info.info = {**(reward_info.info or {}), "task_correction": note}
    return reward_info
