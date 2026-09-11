# Copyright Sierra
"""Refill individual simulations in an existing run directory.

The engine behind ``tau2 run refill``. A run directory is a checkpoint: the
runner fills gaps rather than redoing work, so re-running a single simulation
has always been "delete the recorded one, then resume". Doing that by hand
means retyping the run's configuration from memory, and a flag left off the
retyped command does not fail — it writes a simulation the rest of the
directory does not agree with (an omitted ``--user-persona-id`` put five calls
of the preference pool in a stock English voice).

So the config is not retyped: it is READ BACK from the results' ``Info``,
which records what the run was configured to do. Three steps:

1. Rebuild the ``RunConfig`` from ``Info`` (:func:`reconstruct_config`), and
   report every knob the checkpoint did not record rather than quietly
   defaulting it.
2. Delete the named simulations and their index entries
   (``checkpoint.drop_simulations``).
3. Re-run into the same directory with ``--auto-resume``, which fills exactly
   the deleted cells (:func:`execute_refill`).

Everything that decides whether a refill is safe happens BEFORE step 2: an
unresolvable caller persona, an unknown task id, a trial outside the grid.
A refill that deletes and then fails has destroyed the only copy.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Annotated, Optional, Union

from loguru import logger
from pydantic import BaseModel, Field

from tau2.config import SUBSET_AUTO
from tau2.data_model.simulation import (
    DeliveryJudgeSettings,
    Info,
    NativenessJudgeSettings,
    QualityJudgeSettings,
    Results,
    RunConfig,
    SimulationRun,
    TerminationReason,
    TextRunConfig,
    VoiceRunConfig,
    parse_scores,
)
from tau2.data_model.tasks import Task
from tau2.multilingual.registry import require_caller_persona
from tau2.registry import registry
from tau2.runner.checkpoint import drop_simulations
from tau2.runner.helpers import load_tasks, resolve_timeout, trial_seeds
from tau2.utils.utils import DATA_DIR


class RefillError(Exception):
    """A refill cannot proceed. Raised before anything is deleted."""


class RefillOverrides(BaseModel):
    """Values supplied on the command line rather than read off the run.

    Two kinds live here, and they are treated differently:

    - ``max_concurrency`` is an execution knob no result records, because it
      changes how fast the run goes and nothing about what it produces. It is
      always taken from here.
    - the rest are run knobs that ``Info`` records for every run written since
      the field existed. They may only fill a HOLE: if the checkpoint records
      the field, an override that contradicts it is an error, because the
      refilled simulation has to be the same kind of call as the ones beside
      it. An override that agrees with the record is accepted and ignored.
    """

    max_concurrency: Annotated[
        Optional[int],
        Field(description="Concurrent simulations for the refill run.", default=None),
    ]
    verbose_logs: Annotated[
        Optional[bool],
        Field(description="Save per-task and LLM call logs.", default=None),
    ]
    timeout: Annotated[
        Optional[float],
        Field(
            description="Per-simulation wallclock guard, in seconds. 0 for a "
            "run that had none, as on `tau2 run`.",
            default=None,
        ),
    ]
    scores_spec: Annotated[
        Optional[str],
        Field(
            description="Scoring axes the run computed inline, as the "
            "comma-separated `--scores` value. Unparsed, because 'all' expands "
            "differently for voice and text and the modality is only known "
            "once the results are read.",
            default=None,
        ),
    ]
    nativeness_llm_judge: Annotated[
        Optional[bool],
        Field(description="Run the LLM nativeness judge.", default=None),
    ]
    user_persona_id: Annotated[
        Optional[str],
        Field(
            description="The caller the run spoke as: a language code or a "
            "pack persona id. Checked against the personas the directory's "
            "recorded calls actually used.",
            default=None,
        ),
    ]
    task_set_name: Annotated[
        Optional[str],
        Field(
            description="The task set the run's tasks came from. Checked "
            "against the task ids the directory holds.",
            default=None,
        ),
    ]


class SimulationCell(BaseModel):
    """One ``(task_id, trial)`` cell of a run's grid.

    ``seed`` completes the triple resume matches on. It is None for a cell the
    directory holds nothing for.
    """

    task_id: str
    trial: int
    seed: Optional[int] = None
    sim_id: Optional[str] = None
    termination_reason: Optional[TerminationReason] = None
    reward: Optional[float] = None

    @classmethod
    def from_simulation(cls, sim: SimulationRun) -> "SimulationCell":
        return cls(
            task_id=sim.task_id,
            trial=sim.trial if sim.trial is not None else 0,
            seed=sim.seed,
            sim_id=sim.id,
            termination_reason=sim.termination_reason,
            reward=sim.reward_info.reward if sim.reward_info else None,
        )

    def describe(self) -> str:
        parts = [f"task {self.task_id}", f"trial {self.trial}"]
        if self.termination_reason is not None:
            parts.append(self.termination_reason.value)
        if self.reward is not None:
            parts.append(f"reward {self.reward:g}")
        return ", ".join(parts)


class RefillPlan(BaseModel):
    """What a refill would delete, what it would then run, and on what config.

    Produced before anything is written, so ``--dry-run`` and the real thing
    see exactly the same plan.
    """

    results_path: Path
    config: Union[VoiceRunConfig, TextRunConfig]
    recorded_commit: str = Field(
        description="git commit the run directory was written at. The refill "
        "runs at whatever this checkout is; a difference is not fatal but it "
        "is worth seeing, since the directory keeps the ORIGINAL commit in "
        "its Info."
    )
    unrecorded: list[str] = Field(
        description="Info fields the checkpoint does not carry, so the config "
        "above holds a fallback rather than what the run actually used.",
        default_factory=list,
    )
    selection_note: Optional[str] = Field(
        description="Set when the refill had to select the directory's tasks "
        "differently than the run itself did (see pin_task_selection).",
        default=None,
    )
    drop: list[SimulationCell] = Field(
        description="Recorded simulations the refill deletes.",
        default_factory=list,
    )
    requested_empty: list[SimulationCell] = Field(
        description="Requested cells the directory already holds nothing for; "
        "nothing to delete, and the resume fills them.",
        default_factory=list,
    )
    other_gaps: list[SimulationCell] = Field(
        description="Cells of the grid that are empty and were NOT requested. "
        "The resume fills these too — it fills every gap it finds — so they "
        "are spend the refill incurs and are reported up front.",
        default_factory=list,
    )
    seed_mismatch: list[SimulationCell] = Field(
        description="Stored simulations whose seed this config does not "
        "derive (e.g. a hallucination retry re-ran with a fresh seed). Resume "
        "does not recognise them, so their cell counts as a gap and gets a "
        "SECOND simulation. Reported because that duplication is silent.",
        default_factory=list,
    )

    @property
    def total_to_run(self) -> int:
        return len(self.drop) + len(self.requested_empty) + len(self.other_gaps)


class RefillReport(BaseModel):
    """Outcome of a refill: what was deleted and what came back."""

    plan: RefillPlan
    filled: list[SimulationCell] = Field(default_factory=list)
    missing: list[SimulationCell] = Field(
        description="Requested cells the directory still holds nothing for "
        "after the run — a refill that did not refill.",
        default_factory=list,
    )


# =============================================================================
# Config reconstruction
# =============================================================================


class ReconstructedConfig(BaseModel):
    """A run config rebuilt from a result, with the gaps in that result named."""

    config: Union[VoiceRunConfig, TextRunConfig]
    unrecorded: list[str] = Field(default_factory=list)


def _fill(
    unrecorded: list[str],
    field: str,
    recorded,
    fallback,
    override=None,
    report_gap: bool = True,
):
    """One recorded-or-fallback decision, recording the gap and refusing drift.

    ``recorded`` None means the results file predates the field (that is what
    every knob added for refill documents on ``Info``). An override may fill
    that hole; an override that contradicts a recorded value is a
    ``RefillError``, because it would make the refilled simulation differ from
    the ones already in the directory.

    ``report_gap=False`` is for the fields where None is also a legitimate
    recorded value — a run on a domain's own task set records no task set name
    — so listing them as unrecorded would cry wolf on every English run. Those
    fields are checked against the directory's contents instead.
    """
    if recorded is None:
        if report_gap:
            unrecorded.append(field)
        return fallback if override is None else override
    if override is not None and override != recorded:
        raise RefillError(
            f"the checkpoint records {field}={recorded!r}, but the command "
            f"line asks for {override!r}. The refilled simulation has to match "
            f"the ones beside it — drop the flag to use the recorded value."
        )
    return recorded


def _config_default(field: str):
    """The repo default for a run-config field.

    Read off the model rather than repeated as a literal here: a fallback that
    drifts from the default the runner actually applies would describe a run
    nobody ever ran.
    """
    fields = {**TextRunConfig.model_fields, **VoiceRunConfig.model_fields}
    return fields[field].get_default(call_default_factory=True)


def reconstruct_config(
    info: Info,
    *,
    run_dir: Path,
    task_ids: list[str],
    overrides: Optional[RefillOverrides] = None,
) -> ReconstructedConfig:
    """Rebuild the ``RunConfig`` a run directory was produced with.

    Task selection is reproduced the way the run made it, not re-derived: a
    run that scored a fixed subset gets ``--task-subset <name>`` again (so the
    subset's own frame check still fires if the artifact moved under it),
    while any other run is pinned to the exact task ids the checkpoint holds.

    ``save_to`` is set to the absolute run directory, so a refill works on a
    directory that lives outside ``data/simulations`` (results directories get
    moved; see ``Results.load``).

    Raises:
        RefillError: If the results carry too little to identify the run at
            all (no user model, or a text run with no agent model), or an
            override contradicts the record.
    """
    overrides = overrides or RefillOverrides()
    unrecorded: list[str] = []
    is_voice = info.audio_native_config is not None

    if not info.user_info.llm:
        raise RefillError(
            "the results record no user simulator model (info.user_info.llm), "
            "so the run cannot be rebuilt from them."
        )

    # 0 means the run had no wallclock guard, on both sides: that is how Info
    # records the `--timeout 0` opt-out, and how the escape says so here.
    # RunConfig spells the same thing None, so the resolved value converts —
    # a literal 0 there would time every simulation out instantly.
    timeout = (
        _fill(
            unrecorded,
            "timeout",
            info.timeout,
            resolve_timeout(None, info.audio_native_config),
            overrides.timeout,
        )
        or None
    )
    # `--scores all` means one thing for a text run and another for a voice one
    # (delivery is voice-only), and which this is is only known now, from the
    # results — so the spec is parsed here rather than at parse time.
    override_scores = None
    if overrides.scores_spec is not None:
        try:
            override_scores = parse_scores(overrides.scores_spec, voice=is_voice)
        except ValueError as exc:
            raise RefillError(f"--scores: {exc}") from None
    scores = _fill(
        unrecorded,
        "scores",
        set(info.scores) if info.scores is not None else None,
        _config_default("scores"),
        override_scores,
    )
    # The escape says whether the LLM judge ran, and nothing about which model
    # judged. So it is checked against that one field: a run recorded with
    # llm_judge=True and a non-default judge model agrees with
    # --nativeness-llm-judge, and keeps its model.
    judge_override = (
        None
        if overrides.nativeness_llm_judge is None
        else NativenessJudgeSettings(llm_judge=overrides.nativeness_llm_judge)
    )
    if info.nativeness_judge is not None and overrides.nativeness_llm_judge is not None:
        if info.nativeness_judge.llm_judge != overrides.nativeness_llm_judge:
            raise RefillError(
                "the checkpoint records nativeness llm_judge="
                f"{info.nativeness_judge.llm_judge!r}, but the command line "
                f"asks for {overrides.nativeness_llm_judge!r}. The refilled "
                "simulation has to match the ones beside it — drop the flag to "
                "use the recorded value."
            )
        judge_override = None
    nativeness_judge = _fill(
        unrecorded,
        "nativeness_judge",
        info.nativeness_judge,
        NativenessJudgeSettings(),
        judge_override,
    )

    shared = dict(
        domain=info.environment_info.domain_name,
        task_set_name=_fill(
            unrecorded,
            "task_set_name",
            info.task_set_name,
            None,
            overrides.task_set_name,
            report_gap=False,
        ),
        task_split_name=_fill(
            unrecorded,
            "task_split_name",
            info.task_split_name,
            _config_default("task_split_name"),
        ),
        # None here is ambiguous by construction — a stock English run and a
        # results file predating the field look the same — so it is reported
        # as unrecorded either way, and the personas the recorded calls used
        # decide whether that matters (see check_recorded_callers).
        user_persona_id=_fill(
            unrecorded,
            "user_persona_id",
            info.user_persona_id,
            None,
            overrides.user_persona_id,
        ),
        communicate_judge_mode=_fill(
            unrecorded,
            "communicate_judge_mode",
            info.communicate_judge_mode,
            _config_default("communicate_judge_mode"),
        ),
        llm_user=info.user_info.llm,
        llm_args_user=dict(info.user_info.llm_args or {}),
        num_trials=info.num_trials,
        max_errors=info.max_errors,
        seed=info.seed,
        timeout=timeout,
        scores=scores,
        nativeness_judge=nativeness_judge,
        quality_judge=_fill(
            unrecorded,
            "quality_judge",
            info.quality_judge,
            QualityJudgeSettings(),
        ),
        hallucination_retries=_fill(
            unrecorded,
            "hallucination_retries",
            info.hallucination_retries,
            _config_default("hallucination_retries"),
        ),
        auto_review=_fill(
            unrecorded, "auto_review", info.auto_review, _config_default("auto_review")
        ),
        review_mode=_fill(
            unrecorded, "review_mode", info.review_mode, _config_default("review_mode")
        ),
        review_model=_fill(
            unrecorded,
            "review_model",
            info.review_model,
            _config_default("review_model"),
        ),
        verbose_logs=_fill(
            unrecorded,
            "verbose_logs",
            info.verbose_logs,
            _config_default("verbose_logs"),
            overrides.verbose_logs,
        ),
        retrieval_config=info.retrieval_config,
        retrieval_config_kwargs=info.retrieval_config_kwargs,
        # The whole point of the verb: fill gaps in place, never prompt, never
        # start a second directory.
        save_to=str(run_dir),
        auto_resume=True,
    )
    if overrides.max_concurrency is not None:
        shared["max_concurrency"] = overrides.max_concurrency

    # Task selection: the subset if the run scored one, else the exact ids.
    if info.task_subset is not None:
        shared["task_subset"] = info.task_subset.name
    else:
        shared["task_ids"] = list(task_ids)
        shared["task_subset"] = SUBSET_AUTO

    if is_voice:
        config: RunConfig = VoiceRunConfig(
            **shared,
            audio_native_config=info.audio_native_config.model_copy(deep=True),
            speech_complexity=_fill(
                unrecorded,
                "speech_complexity",
                info.speech_complexity,
                _config_default("speech_complexity"),
            ),
            delivery_judge=_fill(
                unrecorded,
                "delivery_judge",
                info.delivery_judge,
                DeliveryJudgeSettings(),
            ),
        )
    else:
        if not info.agent_info.llm:
            raise RefillError(
                "the results record no agent model (info.agent_info.llm), so "
                "the text run cannot be rebuilt from them."
            )
        config = TextRunConfig(
            **shared,
            agent=info.agent_info.implementation,
            llm_agent=info.agent_info.llm,
            llm_args_agent=dict(info.agent_info.llm_args or {}),
            user=info.user_info.implementation,
            max_steps=info.max_steps,
            enforce_communication_protocol=_fill(
                unrecorded,
                "enforce_communication_protocol",
                info.enforce_communication_protocol,
                _config_default("enforce_communication_protocol"),
            ),
        )
    return ReconstructedConfig(config=config, unrecorded=unrecorded)


# =============================================================================
# Planning
# =============================================================================


def resolve_run_dir(save_to: str | Path) -> Path:
    """The run directory named by ``--save-to``.

    Accepts what the operator has in hand: the directory, its results.json, or
    the ``--save-to`` name the run was launched with (resolved under
    data/simulations/).
    """
    path = Path(save_to)
    # Resolved to an absolute path because it becomes the config's `save_to`,
    # and a relative one there would be read as a name under
    # data/simulations/ — i.e. the refill would write somewhere else entirely.
    candidates = [path, DATA_DIR / "simulations" / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.parent.resolve()
        if candidate.is_dir():
            return candidate.resolve()
    raise RefillError(
        f"no run directory at {path} (also looked under {DATA_DIR / 'simulations'})."
    )


class RunCensus(BaseModel):
    """What one streaming pass over a run directory's simulations found.

    One pass, not two: the seed completes the triple resume matches on and the
    index does not carry it, so the simulations themselves have to be read —
    and a voice directory's simulations are large enough that reading them
    twice is a minute of I/O.
    """

    cells: list[SimulationCell] = Field(default_factory=list)
    caller_persona_ids: set[str] = Field(
        description="Language-pack persona ids the recorded calls were spoken "
        "with (from each simulation's speech environment). Empty for text runs "
        "and for stock English voices.",
        default_factory=set,
    )

    def by_cell(self) -> dict[tuple[int, str], SimulationCell]:
        return {(cell.trial, cell.task_id): cell for cell in self.cells}


def census_run(results_path: Path) -> RunCensus:
    """Read a run directory's simulations once, keeping only what a plan needs."""
    census = RunCensus()
    for sim in Results.iter_simulations(results_path):
        census.cells.append(SimulationCell.from_simulation(sim))
        environment = sim.speech_environment
        if environment is not None and environment.persona_id:
            census.caller_persona_ids.add(environment.persona_id)
    return census


def check_recorded_callers(
    user_persona_id: Optional[str], caller_persona_ids: set[str]
) -> None:
    """Assert the rebuilt config speaks as the caller the recorded calls used.

    ``Info.user_persona_id`` only exists on runs written after it was added, so
    for every directory produced before that the config's caller is either an
    operator's ``--user-persona-id`` or nothing at all. The calls themselves
    are the evidence: each one records the language-pack persona it was spoken
    with, so a claimed caller can be checked against what the directory
    actually holds instead of being taken on trust.

    Raises:
        RefillError: If the directory's calls used language-pack personas and
            the config would not reproduce them.
    """
    from tau2.multilingual.registry import get_language_pack, resolve_run_language

    if not caller_persona_ids:
        return

    languages = {resolve_run_language(persona) for persona in caller_persona_ids}
    languages.discard(None)
    example = ", ".join(sorted(caller_persona_ids)[:3])

    if user_persona_id is None:
        raise RefillError(
            f"this run's calls were spoken by language-pack personas "
            f"({example}), but its results record no --user-persona-id (they "
            "predate the field). Pass --user-persona-id "
            f"{'/'.join(sorted(languages)) or '<language>'} so the refill "
            "speaks as the same caller instead of a stock English voice."
        )

    run_language = resolve_run_language(user_persona_id)
    if languages and {run_language} != languages:
        raise RefillError(
            f"--user-persona-id {user_persona_id!r} resolves to language "
            f"{run_language!r}, but this run's calls were spoken by "
            f"{'/'.join(sorted(languages))} personas ({example})."
        )
    if get_language_pack(user_persona_id) is None and len(caller_persona_ids) > 1:
        # A pack persona id pins ONE speaker for every task; this directory
        # assigned a different one per task, which is what the bare language
        # code does.
        raise RefillError(
            f"--user-persona-id {user_persona_id!r} pins one speaker, but this "
            f"run assigned {len(caller_persona_ids)} personas across its tasks "
            f"({example}). Pass the language code "
            f"{'/'.join(sorted(languages))} instead."
        )


def _coverage_mismatch(config: RunConfig, recorded_task_ids: set[str]) -> Optional[str]:
    """How the config's task selection differs from the directory's, if it does.

    Runs the selection the runner will run — same function, not a re-derivation
    — so the answer is what ``run_domain`` would actually load.
    """
    from tau2.runner.helpers import resolve_tasks

    try:
        selected = {task.id for task in resolve_tasks(config).tasks}
    except ValueError as exc:  # task ids not in the set, subset off its frame
        return str(exc)
    missing = sorted(recorded_task_ids - selected)
    unexpected = sorted(selected - recorded_task_ids)
    if not missing and not unexpected:
        return None
    parts = []
    if missing:
        parts.append(
            f"{len(missing)} task(s) the directory holds are not selected "
            f"(e.g. {', '.join(missing[:2])})"
        )
    if unexpected:
        parts.append(
            f"{len(unexpected)} task(s) it does not hold are "
            f"(e.g. {', '.join(unexpected[:2])})"
        )
    return "; ".join(parts)


def _task_sets_holding(task_id: str) -> list[str]:
    """Registered task sets that contain a task id — the hint for a lost set."""
    holders = []
    for task_set_name in registry.get_task_sets():
        try:
            tasks = load_tasks(task_set_name=task_set_name)
        except Exception:  # a set whose loader needs arguments we do not have
            continue
        if any(task.id == task_id for task in tasks):
            holders.append(task_set_name)
    return holders


def pin_task_selection(
    config: RunConfig, recorded_task_ids: list[str]
) -> tuple[RunConfig, Optional[str]]:
    """Make the config select exactly the tasks the directory holds.

    The reconstructed config prefers the run's own selection *mechanism* — the
    subset name when one was recorded — so the subset's frame check still fires
    if the artifact moved underneath it. When that mechanism no longer picks
    the directory's tasks (a subset recorded as provenance for a run that was
    really a ``--num-tasks`` prefix), the ids themselves are unambiguous and
    take over.

    Returns the config to run and a note for the plan when the mechanism
    changed.

    Raises:
        RefillError: If the task set does not hold the directory's tasks at
            all — which is what a results file predating ``Info.task_set_name``
            looks like, since the refill then falls back to the domain's own
            English set. Raised here, in planning, because the alternative is
            discovering it after the delete.
    """
    task_set_name = config.task_set_name or config.domain
    if task_set_name not in registry.get_task_sets():
        # A typo in --task-set-name, or a set the checkout no longer registers.
        raise RefillError(f"no task set '{task_set_name}' is registered.")

    recorded = set(recorded_task_ids)
    mismatch = _coverage_mismatch(config, recorded)
    if mismatch is None:
        return config, None

    if not config.task_ids:
        pinned = config.model_copy(
            update={"task_ids": list(recorded_task_ids), "task_subset": SUBSET_AUTO}
        )
        pinned_mismatch = _coverage_mismatch(pinned, recorded)
        if pinned_mismatch is None:
            return pinned, (
                f"task subset '{config.task_subset}' no longer selects this "
                "directory's tasks, so the refill pins them by id instead "
                f"({mismatch})."
            )
        # Pinning did not help: the ids are not in this task set at all, which
        # is the more useful thing to report.
        mismatch = pinned_mismatch

    hint = ""
    if config.task_set_name is None:
        holders = _task_sets_holding(recorded_task_ids[0])
        hint = (
            " These results record no task set, so the refill fell back to the "
            f"domain's own set '{config.domain}'; pass --task-set-name"
            + (f" (try: {', '.join(holders)})" if holders else "")
            + "."
        )
    raise RefillError(
        f"task set '{task_set_name}' does not select the simulations this "
        f"directory holds: {mismatch}.{hint}"
    )


def check_task_drift(config: RunConfig, checkpoint_tasks: list[Task]) -> None:
    """Assert the task set still says what the directory's copy of it says.

    ``try_resume`` compares the checkpoint's task objects to the ones the run
    loads and aborts when any differ — a refill would hit that INSIDE
    ``run_domain``, with the simulations already deleted. Same comparison,
    same hash function, run here instead.

    Rewriting a task after a run is not a resumable state: the recorded calls
    scored the old text, so a call refilled against the new one is not
    comparable to the ones beside it. There is no flag for it here; the answer
    is to re-run the arm.

    Raises:
        RefillError: If any task the directory holds has changed.
    """
    from tau2.runner.checkpoint import task_drift_hash
    from tau2.runner.helpers import resolve_tasks

    loaded = {task.id: task for task in resolve_tasks(config).tasks}
    drifted = [
        task.id
        for task in checkpoint_tasks
        if task.id in loaded
        and task_drift_hash(task) != task_drift_hash(loaded[task.id])
    ]
    if drifted:
        raise RefillError(
            f"{len(drifted)} task(s) changed since this directory was written "
            f"(e.g. {', '.join(drifted[:2])}). The recorded calls scored the "
            "old text, so a refilled call would not be comparable to the ones "
            "beside it — re-run the arm instead of refilling it."
        )


def plan_refill(
    save_to: str | Path,
    *,
    task_ids: list[str],
    trials: Optional[list[int]] = None,
    overrides: Optional[RefillOverrides] = None,
) -> RefillPlan:
    """Work out what a refill would do, without touching the directory."""
    run_dir = resolve_run_dir(save_to)
    results_path = run_dir / "results.json"
    if not results_path.exists():
        raise RefillError(f"{run_dir} holds no results.json.")

    # Metadata plus a stream of simulations: a voice run directory holds every
    # tick of every call, and a plan does not need any of it.
    metadata = Results.load_metadata(results_path)
    info = metadata.info
    known_task_ids = [task.id for task in metadata.tasks]

    unknown = [task_id for task_id in task_ids if task_id not in set(known_task_ids)]
    if unknown:
        raise RefillError(
            f"{run_dir.name} does not hold task(s) {unknown}. Its tasks are: "
            f"{', '.join(known_task_ids)}"
        )
    if trials:
        out_of_range = [t for t in trials if t < 0 or t >= info.num_trials]
        if out_of_range:
            raise RefillError(
                f"trial(s) {out_of_range} are outside the run's grid of "
                f"{info.num_trials} trial(s) (0-{info.num_trials - 1})."
            )

    census = census_run(results_path)
    reconstructed = reconstruct_config(
        info, run_dir=run_dir, task_ids=known_task_ids, overrides=overrides
    )
    config = reconstructed.config

    # Both caller checks run here, before the delete rather than after it: a
    # run that cannot speak as its caller must not first destroy the
    # simulation it was asked to replace. The first is the guard PR #530
    # added (config against tasks); the second is against the calls the
    # directory already holds, which is all a pre-#530 directory has.
    check_recorded_callers(config.user_persona_id, census.caller_persona_ids)
    require_caller_persona(config.user_persona_id, known_task_ids, config.domain)

    # And the last pre-delete checks: that the config still loads the tasks
    # this directory was built from, and that they still say the same thing.
    config, selection_note = pin_task_selection(config, known_task_ids)
    check_task_drift(config, metadata.tasks)

    wanted_trials = set(trials) if trials else set(range(info.num_trials))
    wanted_tasks = set(task_ids)
    # trial_seeds reseeds the global RNG, which is how the batch runner has
    # always derived these and is load-bearing for what the simulations draw
    # downstream. Planning is not a run, so it puts the RNG back.
    rng_state = random.getstate()
    try:
        seeds = trial_seeds(info.seed, info.num_trials)
    finally:
        random.setstate(rng_state)
    stored = census.by_cell()
    seed_mismatch = [
        cell
        for cell in stored.values()
        if 0 <= cell.trial < len(seeds) and cell.seed != seeds[cell.trial]
    ]

    drop: list[SimulationCell] = []
    requested_empty: list[SimulationCell] = []
    other_gaps: list[SimulationCell] = []
    for trial in range(info.num_trials):
        for task_id in known_task_ids:
            cell = stored.get((trial, task_id))
            requested = task_id in wanted_tasks and trial in wanted_trials
            if requested and cell is not None:
                drop.append(cell)
            elif requested:
                requested_empty.append(SimulationCell(task_id=task_id, trial=trial))
            elif cell is None:
                other_gaps.append(SimulationCell(task_id=task_id, trial=trial))

    return RefillPlan(
        results_path=results_path,
        config=config,
        recorded_commit=info.git_commit,
        unrecorded=reconstructed.unrecorded,
        selection_note=selection_note,
        drop=drop,
        requested_empty=requested_empty,
        other_gaps=other_gaps,
        seed_mismatch=seed_mismatch,
    )


def format_plan(plan: RefillPlan) -> str:
    """The plan as the operator sees it before anything is deleted."""
    config = plan.config
    lines = [
        f"Refill {plan.results_path.parent}",
        f"  domain          {config.domain}",
        f"  task set        {config.task_set_name or config.domain}",
        f"  caller persona  {config.user_persona_id or '(stock English sampling)'}",
        f"  agent           {config.effective_agent_model}"
        + (
            f" ({config.effective_agent_provider}, "
            f"{config.audio_native_config.reasoning_effort})"
            if isinstance(config, VoiceRunConfig)
            else ""
        ),
        f"  user simulator  {config.llm_user}",
        f"  seed / trials   {config.seed} / {config.num_trials}",
        f"  recorded at     commit {plan.recorded_commit[:12]}",
    ]
    if plan.unrecorded:
        lines.append(
            "  UNRECORDED      "
            + ", ".join(plan.unrecorded)
            + " — the checkpoint predates these fields, so the values above "
            "are repo defaults, not what the run used."
        )
    if plan.selection_note:
        lines.append(f"  TASK SELECTION  {plan.selection_note}")
    if plan.seed_mismatch:
        lines.append(
            f"  DUPLICATE RISK  {len(plan.seed_mismatch)} stored simulation(s) "
            "carry a seed this config does not derive; resume will not "
            "recognise them and their cells will be run a second time."
        )
    lines.append(f"  deleting {len(plan.drop)} simulation(s):")
    for cell in plan.drop:
        lines.append(f"    - {cell.describe()}")
    for cell in plan.requested_empty:
        lines.append(f"    - {cell.describe()} (nothing recorded; will be run)")
    if plan.other_gaps:
        lines.append(
            f"  plus {len(plan.other_gaps)} pre-existing gap(s) the resume "
            "will also fill:"
        )
        for cell in plan.other_gaps:
            lines.append(f"    - {cell.describe()}")
    lines.append(f"  running {plan.total_to_run} simulation(s) in total")
    return "\n".join(lines)


# =============================================================================
# Execution
# =============================================================================


def execute_refill(plan: RefillPlan) -> RefillReport:
    """Delete the planned simulations and re-run them into the same directory."""
    from tau2.runner.batch import run_domain

    if plan.total_to_run == 0:
        logger.info("Nothing to refill: every requested cell is already filled.")
        return RefillReport(plan=plan)

    if plan.drop:
        # Logged in full before deletion: these rewards and termination
        # reasons are the only record of what the directory held, and after
        # this call they are gone.
        for cell in plan.drop:
            logger.info(f"Refill dropping {cell.describe()}")
        # One call per trial, never one call with every task id and every
        # trial: drop_simulations ANDs its filters, so the combined form would
        # delete the cross product — trial 1 of a task whose trial 0 was the
        # only one asked for.
        by_trial: dict[int, list[str]] = {}
        for cell in plan.drop:
            by_trial.setdefault(cell.trial, []).append(cell.task_id)
        for trial, task_ids_in_trial in sorted(by_trial.items()):
            drop_simulations(
                results_path=plan.results_path,
                task_ids=sorted(task_ids_in_trial),
                trials=[trial],
            )

    run_domain(plan.config)

    stored = census_run(plan.results_path).by_cell()
    filled: list[SimulationCell] = []
    missing: list[SimulationCell] = []
    for cell in plan.drop + plan.requested_empty:
        refilled = stored.get((cell.trial, cell.task_id))
        if refilled is None:
            missing.append(cell)
        else:
            filled.append(refilled)
    if missing:
        logger.warning(
            f"{len(missing)} requested cell(s) are still empty after the "
            "refill run: " + "; ".join(cell.describe() for cell in missing)
        )
    return RefillReport(plan=plan, filled=filled, missing=missing)


def refill_simulations(
    save_to: str | Path,
    *,
    task_ids: list[str],
    trials: Optional[list[int]] = None,
    overrides: Optional[RefillOverrides] = None,
) -> RefillReport:
    """Plan and execute a refill. See the module docstring for the sequence."""
    return execute_refill(
        plan_refill(save_to, task_ids=task_ids, trials=trials, overrides=overrides)
    )
