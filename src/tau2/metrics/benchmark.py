"""Strict read-only adapters and submissions over ordinary saved cell Results."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from tau2.config import DOMAIN_TOOLS_VERSION, VOICE_INSTRUCTION_PROMPT_VERSION
from tau2.data_model.benchmark import (
    BenchmarkFrame,
    BenchmarkPlan,
    BenchmarkSelection,
    BenchmarkSubmission,
    BenchmarkSummary,
    CellFrame,
    CellSummary,
    SavedBenchmark,
    SavedBenchmarkCell,
    ScoreCoverage,
    TaskTrial,
    TrialStatus,
    cell_audio_config,
    conditions_mode,
    encode_conditions,
    shared_audio_config,
)
from tau2.data_model.elicitation import ElicitationStrategy
from tau2.data_model.semantic_communication import CommunicateScorer
from tau2.data_model.simulation import (
    Info,
    Results,
    RunConfig,
    TerminationReason,
    TextRunConfig,
    VoiceRunConfig,
)
from tau2.runner.presets import (
    binding_revision,
    load_cell_language,
    uses_canonical_tasks,
)


class IncompleteSubmissionError(ValueError):
    """Official coverage is incomplete; the message lists counts and remedies."""


def describe_reward_gaps(summary: BenchmarkSummary) -> list[str]:
    """Plain-text reward coverage gaps: an overall count, then one line per cell.

    Infrastructure-errored calls are not gaps: they are excluded from the score
    and only mentioned, unless a cell has no scored call at all.
    """
    if summary.complete:
        return []
    unsettled = summary.missing_trials + summary.unscored_trials
    lines = []
    if unsettled:
        lines.append(
            f"Rewards: {unsettled}/{summary.expected_trials} calls lack a scored "
            f"reward ({summary.missing_trials} missing, "
            f"{summary.unscored_trials} unscored)."
        )
    if summary.error_trials:
        lines.append(
            f"Rewards: {summary.error_trials} infrastructure-errored calls are "
            "excluded from the score, not counted as gaps."
        )
    for cell in summary.cells:
        if cell.complete:
            continue
        parts = [
            f"{count} {label}"
            for count, label in (
                (cell.missing_trials, "missing"),
                (cell.unscored_trials, "unscored"),
            )
            if count
        ]
        if not cell.scored_trials:
            parts.append("no scored call")
        if cell.error_trials:
            parts.append(f"{cell.error_trials} errored (excluded)")
        lines.append(f"  {cell.language}-{cell.domain}: " + ", ".join(parts))
    return lines


def _digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _recorded_config(info: Info) -> RunConfig:
    """Reconstruct the complete declared condition contract, without defaults drift."""
    record = info.benchmark
    assert record is not None
    conditions = record.conditions
    voice = conditions_mode(conditions) == "voice"
    # Agent-directed elicitation calls use their own versioned prompt instead.
    shared_prompt = (
        conditions.elicitation is None
        or conditions.elicitation.strategy != ElicitationStrategy.AGENT_DIRECTED
    )
    if voice and (
        info.voice_instruction_prompt_version
        != (VOICE_INSTRUCTION_PROMPT_VERSION if shared_prompt else None)
        or conditions.voice_instruction_prompt_version
        != VOICE_INSTRUCTION_PROMPT_VERSION
    ):
        raise ValueError("Missing or mismatched shared voice prompt revision")
    if (
        info.domain_tools_version != DOMAIN_TOOLS_VERSION
        or conditions.domain_tools_version != DOMAIN_TOOLS_VERSION
    ):
        raise ValueError("Missing or mismatched domain tools revision")
    if info.localize_db is False:
        raise ValueError("Released benchmark results require regional scenarios")
    values = conditions.model_dump(mode="python")
    # Result provenance belongs in the fingerprint but is not configurable.
    values.pop("voice_instruction_prompt_version", None)
    values.pop("domain_tools_version")
    values.update(language=record.language, domain=record.domain)
    config_type = VoiceRunConfig if voice else TextRunConfig
    if voice:
        values["audio_native_config"] = cell_audio_config(
            conditions.audio_native_config,
            record.language,
        )
    try:
        config = config_type.model_validate(values)
    except ValueError as exc:
        raise ValueError(
            f"Recorded benchmark conditions are not a valid {config_type.__name__}"
        ) from exc
    common = encode_conditions(config)
    if common != conditions:
        raise ValueError("Recorded benchmark conditions are incomplete or unresolved")
    return config


def _authoritative_language(
    language: str,
    domain: str,
    expected_binding: str,
    mode: str,
    *,
    canonical_tasks: bool = False,
):
    """Require exact published source assets; saved metadata cannot authenticate itself.

    Sources resolve through the ordinary DATA_DIR mechanism, so a relocated data
    installation is supported. Missing or different versions fail explicitly;
    there is no fallback to a task or persona supplied inside the saved result.
    """
    try:
        loaded = load_cell_language(
            language, domain, mode, canonical_tasks=canonical_tasks
        )
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Cannot load authoritative benchmark artifacts for {language}/{domain}; "
            "install the pinned release data or set TAU2_DATA_DIR to that data root"
        ) from exc
    if binding_revision(loaded) != expected_binding:
        raise ValueError(
            f"Installed authoritative benchmark artifacts differ for {language}/{domain}; "
            "install the pinned release data or set TAU2_DATA_DIR to that data root"
        )
    return loaded


def read_task_trials(path: Path, frame: BenchmarkFrame) -> list[TaskTrial]:
    """Stream one saved cell, validating effective config and explicit correspondence."""
    metadata = Results.load_metadata(path)
    info = metadata.info
    record = info.benchmark
    if record is None or not record.conditions:
        raise ValueError("Results lack complete benchmark condition provenance")
    if (
        record.benchmark_version != frame.benchmark_version
        or record.frame_revision != frame.revision
        or record.trials != frame.trials
    ):
        raise ValueError("Recorded benchmark frame/trials differ")
    cell = next(
        (
            c
            for c in frame.cells
            if (c.language, c.domain) == (record.language, record.domain)
        ),
        None,
    )
    if cell is None:
        raise ValueError("Recorded cell is outside the released benchmark matrix")
    if (
        record.canonical_task_ids != cell.canonical_task_ids
        or record.binding_revision != cell.binding_revision
    ):
        raise ValueError("Recorded benchmark binding/task frame differs")
    if (
        _digest(record.conditions.model_dump(mode="json"))
        != record.condition_fingerprint
    ):
        raise ValueError(
            "Recorded condition fingerprint does not match effective conditions"
        )
    if (
        getattr(info, "language", None) != record.language
        or info.environment_info.domain_name != record.domain
    ):
        raise ValueError("Result language/domain conflicts with benchmark identity")
    conditions = record.conditions
    config = _recorded_config(info)
    if (
        info.num_trials != len(frame.trials)
        or info.effective_communicate_scoring.scorer != CommunicateScorer.SEMANTIC
    ):
        raise ValueError("Missing semantic evaluator or mismatched trial count")
    if conditions_mode(conditions) != frame.mode:
        raise ValueError("Recorded conditions differ from the benchmark call mode")
    checks = {
        "llm_user": info.user_info.llm,
        "llm_args_user": info.user_info.llm_args,
        "communicate_scoring": info.communicate_scoring,
        "max_errors": info.max_errors,
        "seed": info.seed,
        "num_trials": info.num_trials,
        "retrieval_config": info.retrieval_config,
        "retrieval_config_kwargs": info.retrieval_config_kwargs,
    }
    if frame.mode == "voice":
        if info.audio_native_config is None:
            raise ValueError("Missing audio-native configuration")
        checks["speech_complexity"] = info.speech_complexity
        checks["audio_native_config"] = shared_audio_config(
            info.audio_native_config, record.language
        )
        agent_model = (
            f"{config.audio_native_config.provider}:{config.audio_native_config.model}"
        )
        agent_args = None
    else:
        if info.audio_native_config is not None:
            raise ValueError(
                "Text benchmark results carry an audio-native configuration"
            )
        agent_model = config.llm_agent
        agent_args = config.llm_args_agent
    if any(getattr(conditions, key) != value for key, value in checks.items()):
        raise ValueError(
            "Effective saved settings differ from benchmark condition contract"
        )
    derived_checks = {
        "max_steps": (info.max_steps, config.effective_max_steps),
        "agent implementation": (
            info.agent_info.implementation,
            config.effective_agent,
        ),
        "user implementation": (info.user_info.implementation, config.effective_user),
        "agent model": (info.agent_info.llm, agent_model),
        "agent arguments": (info.agent_info.llm_args, agent_args),
    }
    changed = [
        name
        for name, (actual, expected) in derived_checks.items()
        if actual != expected
    ]
    if changed:
        raise ValueError(
            "Effective saved settings differ from benchmark condition contract: "
            + ", ".join(changed)
        )
    authoritative = _authoritative_language(
        record.language,
        record.domain,
        cell.binding_revision,
        frame.mode,
        canonical_tasks=uses_canonical_tasks(frame),
    )
    if info.language_artifacts != authoritative.artifact_pins:
        raise ValueError(
            "Saved run artifact pins differ from authoritative benchmark artifacts"
        )
    canonical = {
        entry.canonical_task_id: task_id
        for task_id, entry in authoritative.metadata.items()
    }
    if not set(cell.canonical_task_ids) <= set(canonical):
        raise ValueError(
            "Frozen frame tasks are absent from authoritative benchmark artifacts"
        )
    selected = {canonical[task_id] for task_id in record.selected_task_ids}
    source_tasks = {
        task.id: task for task in authoritative.tasks if task.id in selected
    }
    contexts = getattr(info, "language_tasks", {})
    tasks = {t.id: t for t in metadata.tasks}
    if len(tasks) != len(metadata.tasks) or set(tasks) != set(contexts):
        raise ValueError("Saved task/context coverage is inconsistent")
    if {c.canonical_task_id for c in contexts.values()} != set(
        record.selected_task_ids
    ):
        raise ValueError("Saved canonical task population is incomplete or foreign")
    if set(tasks) != set(source_tasks):
        raise ValueError(
            "Saved task selection differs from authoritative benchmark artifacts"
        )
    from tau2.multilingual.runtime import resolve_task, without_delivery_details

    for task_id, context in contexts.items():
        source_task = source_tasks[task_id]
        if tasks[task_id] != source_task:
            raise ValueError(
                "Saved task content differs from authoritative frozen task"
            )
        expected_context = resolve_task(authoritative, source_task, config.seed or 0)
        # Submitters use their own caller voices (TAU2_VOICE_ID_*), which change
        # delivery, not the graded task or persona.
        if without_delivery_details(context) != without_delivery_details(
            expected_context
        ):
            raise ValueError(
                "Saved language/task/persona context differs from authoritative artifacts"
            )
        if _digest(tasks[task_id].model_dump(mode="json")) != context.task_sha256:
            raise ValueError("Saved task content or artifact fingerprints disagree")
    observations = []
    seen = set()
    for simulation in Results.iter_simulations(path):
        if simulation.task_id not in contexts or simulation.trial not in frame.trials:
            raise ValueError("Stored trial lies outside selected task population")
        context = contexts[simulation.task_id]
        if getattr(simulation, "language_context", None) != context and not (
            getattr(simulation, "language_context", None) is None
            and simulation.termination_reason
            in {
                TerminationReason.INFRASTRUCTURE_ERROR,
                TerminationReason.UNEXPECTED_ERROR,
            }
        ):
            raise ValueError(
                "Simulation language context differs from recorded run context"
            )
        key = (context.canonical_task_id, simulation.trial)
        if key in seen:
            raise ValueError(
                "Duplicate task/trial attempts require explicit resolution"
            )
        seen.add(key)
        if simulation.termination_reason in {
            TerminationReason.INFRASTRUCTURE_ERROR,
            TerminationReason.UNEXPECTED_ERROR,
        }:
            status = TrialStatus.ERROR
            reward = None
        elif simulation.reward_info is None:
            status = TrialStatus.UNSCORED
            reward = None
        else:
            status = TrialStatus.SCORED
            reward = simulation.reward_info.reward
        observations.append(
            TaskTrial(
                language=record.language,
                domain=record.domain,
                canonical_task_id=key[0],
                trial=key[1],
                condition_fingerprint=record.condition_fingerprint,
                binding_revision=record.binding_revision,
                status=status,
                reward=reward,
            )
        )
    return observations


def export_submission(
    root: Path, frame: BenchmarkFrame, *, require_complete: bool = False
) -> BenchmarkSubmission:
    """Read stored cell outputs, keeping missing cells explicit and never judging."""
    trials = []
    sources = {}
    for cell in frame.cells:
        # The results file, not its directory: a directory always reads as the
        # per-simulation (voice) layout, which would drop text cells' inline runs.
        path = root / f"{cell.language}-{cell.domain}" / "results.json"
        if not path.exists():
            continue
        trials.extend(read_task_trials(path, frame))
        sources[f"{cell.language}/{cell.domain}"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    selection = None
    plan_path = root / "benchmark-plan.json"
    if plan_path.exists():
        plan = BenchmarkPlan.model_validate_json(plan_path.read_text())
        if plan.frame != frame:
            raise ValueError("Stored execution plan differs from the benchmark frame")
        selection = plan.selection
    summary = summarize(frame, trials)
    if require_complete and not summary.complete:
        raise IncompleteSubmissionError(
            "\n".join(["Submission is incomplete.", *describe_reward_gaps(summary)])
        )
    submission = BenchmarkSubmission(
        frame=frame,
        trials=trials,
        summary=summary,
        sources=sources,
        selection=selection,
        requested=summarize_selection(frame, trials, selection) if selection else None,
    )

    validate_submission(submission)
    return submission


def load_submission(path: Path) -> BenchmarkSubmission:
    """Read a portable submission and recompute its scores from retained evidence."""
    submission = BenchmarkSubmission.model_validate_json(path.read_text())
    validate_submission(submission)
    return submission


def validate_submission(submission: BenchmarkSubmission) -> None:
    """Reject forged overall and requested summaries at the scoring boundary."""
    if summarize(submission.frame, submission.trials) != submission.summary:
        raise ValueError("Submission summary does not match task observations")
    expected = (
        summarize_selection(submission.frame, submission.trials, submission.selection)
        if submission.selection is not None
        else None
    )
    if submission.requested != expected:
        raise ValueError("Requested summary does not match the recorded selection")


def summarize_selection(
    frame: BenchmarkFrame, trials: list[TaskTrial], selection: BenchmarkSelection
) -> BenchmarkSummary:
    """Summarize a declared execution slice without calling it the official score."""
    selection.validate_against(frame)
    membership = {
        (cell.language, cell.domain, task_id)
        for cell in selection.cells
        for task_id in cell.canonical_task_ids
    }
    selected_trials = [
        trial
        for trial in trials
        if (trial.language, trial.domain, trial.canonical_task_id) in membership
    ]
    return _summarize(frame, selected_trials, selection.cells)


def summarize(frame: BenchmarkFrame, trials: list[TaskTrial]) -> BenchmarkSummary:
    """Compute only the released benchmark's full denominator."""
    return _summarize(frame, trials, frame.cells)


def _summarize(
    frame: BenchmarkFrame, trials: list[TaskTrial], cells: list[CellFrame]
) -> BenchmarkSummary:
    """Average trials within task, tasks within cell, and all cells equally.

    Infrastructure-errored trials are harness faults, not model failures: they
    leave the denominator (a task whose trials all errored drops out of its cell
    mean) and stay counted in ``error_trials``. A cell is complete once every
    owed trial is scored or errored and at least one is scored; missing and
    unscored trials still block it. Partial means are explicitly labeled and
    never substituted for an official score. Repeated observations, unknown
    membership and incompatible conditions fail rather than quietly changing the
    sample or its weights.
    """
    conditions = {trial.condition_fingerprint for trial in trials}
    if len(conditions) > 1:
        raise ValueError("Cannot combine different effective benchmark conditions")
    expected = {(cell.language, cell.domain): cell for cell in cells}
    seen = set()
    grouped = defaultdict(list)
    for trial in trials:
        cell_key = (trial.language, trial.domain)
        if cell_key not in expected:
            raise ValueError("Trial is outside the frozen language/domain population")
        cell = expected[cell_key]
        if trial.binding_revision != cell.binding_revision:
            raise ValueError("Recorded task/pack binding differs from the frozen frame")
        if (
            trial.canonical_task_id not in cell.canonical_task_ids
            or trial.trial not in frame.trials
        ):
            raise ValueError("Trial is outside the frozen task/trial population")
        key = (*cell_key, trial.canonical_task_id, trial.trial)
        if key in seen:
            raise ValueError(
                "Duplicate task/trial observation; select one recorded attempt explicitly"
            )
        seen.add(key)
        grouped[cell_key].append(trial)
    summaries = []
    for cell in cells:
        observations = grouped[(cell.language, cell.domain)]
        task_rewards = defaultdict(list)
        for trial in observations:
            if trial.status == TrialStatus.SCORED:
                task_rewards[trial.canonical_task_id].append(trial.reward)
        observed_score = (
            mean(mean(values) for values in task_rewards.values())
            if task_rewards
            else None
        )
        owed = len(cell.canonical_task_ids) * len(frame.trials)
        scored = sum(trial.status == TrialStatus.SCORED for trial in observations)
        errored = sum(trial.status == TrialStatus.ERROR for trial in observations)
        # An all-errored cell has no score to report, so it never counts.
        complete = scored > 0 and scored + errored == owed
        summaries.append(
            CellSummary(
                language=cell.language,
                domain=cell.domain,
                expected_trials=owed,
                scored_trials=scored,
                error_trials=errored,
                unscored_trials=sum(
                    trial.status == TrialStatus.UNSCORED for trial in observations
                ),
                missing_trials=owed - len(observations),
                complete=complete,
                score=observed_score if complete else None,
                observed_score=observed_score,
            )
        )
    observed = [
        cell.observed_score for cell in summaries if cell.observed_score is not None
    ]
    complete = all(cell.complete for cell in summaries)
    return BenchmarkSummary(
        benchmark_version=frame.benchmark_version,
        frame_revision=frame.revision,
        condition_fingerprint=next(iter(conditions), None),
        expected_trials=sum(cell.expected_trials for cell in summaries),
        scored_trials=sum(cell.scored_trials for cell in summaries),
        error_trials=sum(cell.error_trials for cell in summaries),
        unscored_trials=sum(cell.unscored_trials for cell in summaries),
        missing_trials=sum(cell.missing_trials for cell in summaries),
        complete=complete,
        score=mean(observed) if complete else None,
        observed_score=mean(observed) if observed else None,
        cells=summaries,
        by_language=_breakdown(summaries, "language"),
        by_domain=_breakdown(summaries, "domain"),
    )


def _breakdown(cells: list[CellSummary], dimension: str) -> dict[str, ScoreCoverage]:
    groups = defaultdict(list)
    for cell in cells:
        groups[getattr(cell, dimension)].append(cell)
    result = {}
    for name, group in groups.items():
        complete = all(cell.complete for cell in group)
        observed = [
            cell.observed_score for cell in group if cell.observed_score is not None
        ]
        result[name] = ScoreCoverage(
            expected_trials=sum(cell.expected_trials for cell in group),
            scored_trials=sum(cell.scored_trials for cell in group),
            error_trials=sum(cell.error_trials for cell in group),
            unscored_trials=sum(cell.unscored_trials for cell in group),
            missing_trials=sum(cell.missing_trials for cell in group),
            complete=complete,
            score=mean(observed) if complete else None,
            observed_score=mean(observed) if observed else None,
        )
    return result


def load_saved_benchmark(root: Path) -> SavedBenchmark:
    """Read a released matrix without discovering membership from result directories."""
    from tau2.runner.presets import infer_preset

    root = root.resolve()
    frame = BenchmarkFrame.model_validate_json(
        (root / "benchmark-frame.json").read_text()
    )
    preset = infer_preset(frame)
    submission = export_submission(root, frame)
    cells = []
    for cell in frame.cells:
        path = root / f"{cell.language}-{cell.domain}" / "results.json"
        cells.append(
            SavedBenchmarkCell(cell=cell, results_path=path if path.is_file() else None)
        )
    return SavedBenchmark(root=root, preset=preset, submission=submission, cells=cells)
