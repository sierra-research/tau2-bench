# Copyright Sierra
"""Streaming, resumable producer for the conversation-judge artifact.

Headline: per-dimension progression verdicts and per-turn conciseness failure
modes, aggregated per (language x domain) cell inside the artifact. Shadow:
the EVA-X conjunction and the τ quality composite, computed exactly as before
the facts-first conversion but demoted to ``shadow_scores`` (labeled
``uncalibrated-shadow``, never headlined).

The paid-judgment cache is unchanged from the previous experience runner
(same envelope, component ids, versions, and input digest), so judgments
paid for under the old verb are reused, never re-bought.
"""

import hashlib
import os
import tempfile
from collections.abc import Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from itertools import islice
from pathlib import Path
from statistics import mean
from typing import Optional, TypeVar, Union

from loguru import logger
from pydantic import BaseModel, Field

from tau2.data_model.simulation import (
    Info,
    QualityInfo,
    QualityJudgeSettings,
    Results,
    SimulationRun,
)
from tau2.data_model.tasks import Task
from tau2.evaluator.evaluator import get_simulation_language_info
from tau2.judges.conversation.models import (
    EVA_X_CONCISENESS_THRESHOLD,
    EVA_X_PROGRESSION_THRESHOLD,
    EVA_X_SEMANTIC_PROMPT_VERSION,
    ConcisenessFailureModeSummary,
    ConversationArtifactProvenance,
    ConversationCallError,
    ConversationCallResult,
    ConversationCellSummary,
    ConversationJudgeArtifact,
    ConversationJudgeConfig,
    ConversationShadowCellSummary,
    ConversationShadowScores,
    EvaConcisenessFailureMode,
    EvaConcisenessTurnResult,
    EvaProgressionDimension,
    EvaProgressionDimensionName,
    EvaXScore,
    ProgressionDimensionSummary,
)
from tau2.judges.conversation.semantic import (
    aggregate_conciseness,
    aggregate_progression,
    extract_agent_turns,
    judge_conciseness_turn,
    judge_progression,
)
from tau2.judges.quality.factors import QUALITY_RUBRIC_VERSION
from tau2.judges.quality.harness import evaluate_quality
from tau2.judges.quality.judge import QUALITY_JUDGE_PROMPT_VERSION
from tau2.metrics.interaction_quality import QUALITY_METRICS_VERSION
from tau2.metrics.turn_taking import (
    EVA_X_PASS_THRESHOLD,
    evaluate_turn_taking,
    resolve_turn_taking_params,
)
from tau2.utils.utils import get_commit_hash, get_now


class _InputUnit(BaseModel):
    source: str = Field(description="Resolved source results path.")
    source_key: str = Field(description="Filesystem-safe cache namespace.")
    simulation: SimulationRun = Field(description="Stored call to score.")
    task: Task = Field(description="Task contract for the stored call.")
    domain: str = Field(description="Task domain used for semantic context.")
    language: str = Field(description="Resolved conversation language.")


class _CacheEnvelope(BaseModel):
    """Validated identity around one independently reusable paid judgment."""

    input_sha256: str = Field(description="Digest of all component inputs.")
    component: str = Field(description="Stable paid-unit component identifier.")
    component_version: str = Field(description="Metric or prompt contract version.")
    model: Optional[str] = Field(
        default=None, description="Model used by this component, when applicable."
    )
    model_args: dict = Field(description="Effective model or component settings.")
    payload: dict = Field(description="Validated serialized component result.")


T = TypeVar("T", bound=BaseModel)


def _input_sha256(unit: _InputUnit) -> str:
    """Hash exactly the judge-relevant inputs.

    Everything the suites read: the message trajectory, the tick stream, the
    effective policy, the task contract, domain, and language. Deliberately
    NOT the whole simulation record — an in-place re-evaluation that rewrites
    rewards or annotations must not torch the paid-judgment cache.
    """
    simulation = unit.simulation
    digest = hashlib.sha256()
    for payload in (
        "\n".join(
            message.model_dump_json(exclude_none=False)
            for message in simulation.messages or []
        ),
        "\n".join(
            tick.model_dump_json(exclude_none=False) for tick in simulation.ticks or []
        ),
        simulation.policy or "",
        unit.task.model_dump_json(exclude_none=False),
        unit.domain,
        unit.language,
    ):
        digest.update(payload.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _source_key(path: Path, info: Info) -> str:
    """Move-stable cache namespace for one results input.

    Keyed to the run-dir basename plus a digest of the run's own metadata —
    never the absolute path, because relocating run directories is a normal
    workflow and must not orphan the paid-judgment cache. Content identity is
    still enforced per unit by ``_input_sha256``.
    """
    digest = hashlib.sha256(
        info.model_dump_json(exclude_none=False).encode("utf-8")
    ).hexdigest()[:12]
    stem = path.name.replace(".json", "") or "results"
    safe = "".join(character if character.isalnum() else "-" for character in stem)
    return f"{safe}-{digest}"


def _atomic_write_json(path: Path, model: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(model.model_dump_json(indent=2))
            handle.write("\n")
        Path(temporary).replace(path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _load_cache(
    path: Path,
    *,
    input_sha256: str,
    component: str,
    component_version: str,
    model: Optional[str],
    model_args: dict,
    payload_model: type[T],
    force: bool,
) -> Optional[T]:
    if force or not path.is_file():
        return None
    try:
        envelope = _CacheEnvelope.model_validate_json(path.read_text())
    except Exception as exc:  # noqa: BLE001 - corrupt cache is safely recomputed
        logger.warning(f"ignoring invalid conversation-judge cache {path}: {exc}")
        return None
    if (
        envelope.input_sha256 != input_sha256
        or envelope.component != component
        or envelope.component_version != component_version
        or envelope.model != model
        or envelope.model_args != model_args
    ):
        return None
    return payload_model.model_validate(envelope.payload)


def _store_cache(
    path: Path,
    *,
    input_sha256: str,
    component: str,
    component_version: str,
    model: Optional[str],
    model_args: dict,
    payload: BaseModel,
) -> None:
    _atomic_write_json(
        path,
        _CacheEnvelope(
            input_sha256=input_sha256,
            component=component,
            component_version=component_version,
            model=model,
            model_args=model_args,
            payload=payload.model_dump(mode="json"),
        ),
    )


def _score_ours(
    unit: _InputUnit,
    *,
    cache_root: Path,
    input_sha256: str,
    config: ConversationJudgeConfig,
    force: bool,
) -> QualityInfo:
    settings = QualityJudgeSettings(
        llm_judge=config.ours_llm_judge,
        model=config.ours_model,
        model_args=config.ours_model_args,
    )
    component_version = ":".join(
        [QUALITY_RUBRIC_VERSION, QUALITY_METRICS_VERSION, QUALITY_JUDGE_PROMPT_VERSION]
    )
    path = cache_root / unit.source_key / unit.simulation.id / "ours.json"
    cached = _load_cache(
        path,
        input_sha256=input_sha256,
        component="ours",
        component_version=component_version,
        model=config.ours_model,
        model_args=settings.model_dump(mode="json"),
        payload_model=QualityInfo,
        force=force,
    )
    # A cached suite with errored factor checks is a partial result, not a
    # verdict: treat it as a miss so reruns retry instead of resurfacing it.
    if cached is not None and cached.num_errors == 0:
        return cached
    result = evaluate_quality(
        unit.simulation,
        unit.task,
        domain=unit.domain,
        settings=settings,
    )
    if result.num_errors == 0:
        _store_cache(
            path,
            input_sha256=input_sha256,
            component="ours",
            component_version=component_version,
            model=config.ours_model,
            model_args=settings.model_dump(mode="json"),
            payload=result,
        )
    return result


class _ProgressionCache(BaseModel):
    dimensions: list[EvaProgressionDimension] = Field(
        description="Validated closed progression dimensions."
    )


def _score_eva_x(
    unit: _InputUnit,
    *,
    cache_root: Path,
    input_sha256: str,
    config: ConversationJudgeConfig,
    force: bool,
) -> EvaXScore:
    if not unit.simulation.ticks:
        raise ValueError("EVA-X turn-taking requires a stored full-duplex tick stream")
    turn_taking = evaluate_turn_taking(
        unit.simulation.ticks,
        params=resolve_turn_taking_params(unit.language),
    )
    sim_cache = cache_root / unit.source_key / unit.simulation.id / "eva-x"
    conciseness_turns: list[EvaConcisenessTurnResult] = []
    for turn_id, turn_text in extract_agent_turns(unit.simulation):
        component = f"conciseness-turn-{turn_id}"
        path = sim_cache / "conciseness" / f"turn-{turn_id:04d}.json"
        result = _load_cache(
            path,
            input_sha256=input_sha256,
            component=component,
            component_version=EVA_X_SEMANTIC_PROMPT_VERSION,
            model=config.eva_x_model,
            model_args=config.eva_x_model_args,
            payload_model=EvaConcisenessTurnResult,
            force=force,
        )
        if result is None:
            result = judge_conciseness_turn(
                unit.simulation,
                turn_id=turn_id,
                turn_text=turn_text,
                language=unit.language,
                model=config.eva_x_model,
                model_args=config.eva_x_model_args,
            )
            _store_cache(
                path,
                input_sha256=input_sha256,
                component=component,
                component_version=EVA_X_SEMANTIC_PROMPT_VERSION,
                model=config.eva_x_model,
                model_args=config.eva_x_model_args,
                payload=result,
            )
        conciseness_turns.append(result)
    conciseness = aggregate_conciseness(conciseness_turns)

    progression_path = sim_cache / "progression.json"
    progression_reply = _load_cache(
        progression_path,
        input_sha256=input_sha256,
        component="conversation-progression",
        component_version=EVA_X_SEMANTIC_PROMPT_VERSION,
        model=config.eva_x_model,
        model_args=config.eva_x_model_args,
        payload_model=_ProgressionCache,
        force=force,
    )
    if progression_reply is None:
        progression_reply = _ProgressionCache(
            dimensions=judge_progression(
                unit.simulation,
                unit.task,
                domain=unit.domain,
                language=unit.language,
                model=config.eva_x_model,
                model_args=config.eva_x_model_args,
            )
        )
        _store_cache(
            progression_path,
            input_sha256=input_sha256,
            component="conversation-progression",
            component_version=EVA_X_SEMANTIC_PROMPT_VERSION,
            model=config.eva_x_model,
            model_args=config.eva_x_model_args,
            payload=progression_reply,
        )
    progression = aggregate_progression(progression_reply.dimensions)
    # N/A components (score None: no scoreable turns, an agent-mute call, or
    # no agent turns) are excluded from the conjunction — zero or invalidated
    # evidence is never a fail. The conjunction itself is None only when
    # every component is N/A.
    component_scores = [
        (turn_taking.score, EVA_X_PASS_THRESHOLD),
        (progression.score, EVA_X_PROGRESSION_THRESHOLD),
        (conciseness.score, EVA_X_CONCISENESS_THRESHOLD),
    ]
    evidenced = [
        (score, threshold) for score, threshold in component_scores if score is not None
    ]
    passed = (
        all(score >= threshold for score, threshold in evidenced) if evidenced else None
    )
    return EvaXScore(
        passed=passed,
        turn_taking=turn_taking,
        conversation_progression=progression,
        conciseness=conciseness,
        judge_model=config.eva_x_model,
        judge_args=config.eva_x_model_args,
    )


def _score_one(
    unit: _InputUnit,
    *,
    cache_root: Path,
    config: ConversationJudgeConfig,
    force: bool,
) -> ConversationCallResult:
    digest = _input_sha256(unit)
    ours: Optional[QualityInfo] = None
    eva_x: Optional[EvaXScore] = None
    errors: list[ConversationCallError] = []
    if config.include_ours:
        try:
            ours = _score_ours(
                unit,
                cache_root=cache_root,
                input_sha256=digest,
                config=config,
                force=force,
            )
        except Exception as exc:  # noqa: BLE001 - suite-local failure is recorded
            logger.warning(f"ours failed for {unit.simulation.id}: {exc}")
            errors.append(ConversationCallError(suite="ours", message=str(exc)))
    try:
        eva_x = _score_eva_x(
            unit,
            cache_root=cache_root,
            input_sha256=digest,
            config=config,
            force=force,
        )
    except Exception as exc:  # noqa: BLE001 - suite-local failure is recorded
        logger.warning(f"eva-x failed for {unit.simulation.id}: {exc}")
        errors.append(ConversationCallError(suite="eva-x", message=str(exc)))
    return ConversationCallResult(
        source=unit.source,
        sim_id=unit.simulation.id,
        task_id=str(unit.simulation.task_id),
        domain=unit.domain,
        language=unit.language,
        input_sha256=digest,
        progression_dimensions=(
            list(eva_x.conversation_progression.dimensions) if eva_x is not None else []
        ),
        conciseness_turns=(list(eva_x.conciseness.turns) if eva_x is not None else []),
        shadow_scores=ConversationShadowScores(eva_x=eva_x, ours=ours),
        errors=errors,
    )


def _input_error(
    *,
    source: str,
    simulation: SimulationRun,
    domain: str,
    message: str,
) -> ConversationCallResult:
    """An unscoreable call, recorded on the error channel — never an abort."""
    logger.warning(message)
    return ConversationCallResult(
        source=source,
        sim_id=simulation.id,
        task_id=str(simulation.task_id),
        domain=domain,
        language="unknown",
        input_sha256="",
        errors=[ConversationCallError(suite="input", message=message)],
    )


def _iter_inputs(
    paths: list[Path],
) -> Iterator[Union[_InputUnit, ConversationCallResult]]:
    """Yield scoreable units, or pre-built error records for broken inputs.

    One broken simulation (missing task contract, duplicate identity,
    unresolvable multilingual language) must never abort the batch after
    paid judge spend: it is skipped with a warning and recorded on the
    per-call error channel instead. Duplicate (source, sim_id) pairs are
    dropped here, keeping the artifact validator as a backstop.
    """
    seen: set[tuple[str, str]] = set()
    for path in paths:
        meta = Results.load_metadata(path)
        tasks = {str(task.id): task for task in meta.tasks}
        domain = meta.info.environment_info.domain_name
        source = str(path.resolve())
        source_key = _source_key(path, meta.info)
        for simulation in Results.iter_simulations(path):
            identity = (source, simulation.id)
            if identity in seen:
                logger.warning(
                    f"skipping duplicate simulation {simulation.id} in {path}; "
                    "keeping the first occurrence"
                )
                continue
            seen.add(identity)
            task = tasks.get(str(simulation.task_id))
            if task is None:
                yield _input_error(
                    source=source,
                    simulation=simulation,
                    domain=domain,
                    message=(
                        f"simulation {simulation.id} references missing task "
                        f"{simulation.task_id} in {path}; skipping"
                    ),
                )
                continue
            language, _script = get_simulation_language_info(simulation)
            speech_environment = simulation.speech_environment
            if (
                language is None
                and speech_environment is not None
                and speech_environment.persona_id is not None
            ):
                # A language-pack persona is on record but its language is
                # unresolvable: judging it under English norms would be a
                # silent mislabel, so it goes to the error channel instead.
                yield _input_error(
                    source=source,
                    simulation=simulation,
                    domain=domain,
                    message=(
                        f"simulation {simulation.id} in {path} carries "
                        f"multilingual persona {speech_environment.persona_id} "
                        "but resolves no language; refusing the English "
                        "default"
                    ),
                )
                continue
            if simulation.policy is None:
                simulation = simulation.model_copy(
                    update={"policy": meta.info.environment_info.policy}
                )
            yield _InputUnit(
                source=source,
                source_key=source_key,
                simulation=simulation,
                task=task,
                domain=domain,
                # None with no multilingual marker is the documented sentinel
                # for genuine English/text runs.
                language=language or "en",
            )


def _bounded_map(
    fn,
    items: Iterable[Union[_InputUnit, ConversationCallResult]],
    concurrency: int,
) -> Iterator[ConversationCallResult]:
    iterator = iter(items)
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(fn, item) for item in islice(iterator, concurrency)}
        while futures:
            done, futures = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                yield future.result()
            futures |= {
                executor.submit(fn, item) for item in islice(iterator, len(done))
            }


def aggregate_cells(
    calls: list[ConversationCallResult],
) -> list[ConversationCellSummary]:
    """Per (language x domain) headline aggregates plus shadow means."""
    by_cell: dict[tuple[str, str], list[ConversationCallResult]] = {}
    for call in calls:
        by_cell.setdefault((call.language, call.domain), []).append(call)
    cells: list[ConversationCellSummary] = []
    for (language, domain), cell_calls in sorted(by_cell.items()):
        judged = [call for call in cell_calls if call.progression_dimensions]
        dimensions: list[ProgressionDimensionSummary] = []
        for name in EvaProgressionDimensionName:
            verdicts = [
                dimension
                for call in judged
                for dimension in call.progression_dimensions
                if dimension.name is name
            ]
            # The model validator pins flagged <-> rating in {1, 2}, so the
            # rating alone carries the severity split.
            majors = sum(1 for dimension in verdicts if dimension.rating == 1)
            minors = sum(1 for dimension in verdicts if dimension.rating == 2)
            dimensions.append(
                ProgressionDimensionSummary(
                    name=name,
                    n_calls=len(verdicts),
                    major_count=majors,
                    major_rate=(majors / len(verdicts) if verdicts else None),
                    minor_count=minors,
                    minor_rate=(minors / len(verdicts) if verdicts else None),
                    mean_rating=(
                        mean(dimension.rating for dimension in verdicts)
                        if verdicts
                        else None
                    ),
                )
            )
        turns = [turn for call in cell_calls for turn in call.conciseness_turns]
        failure_modes = [
            ConcisenessFailureModeSummary(
                mode=mode,
                turn_count=(
                    count := sum(1 for turn in turns if mode in turn.failure_modes)
                ),
                rate=(count / len(turns) if turns else None),
            )
            for mode in EvaConcisenessFailureMode
        ]
        eva_scores = [
            call.shadow_scores.eva_x
            for call in cell_calls
            if call.shadow_scores.eva_x is not None
        ]
        gated = [score for score in eva_scores if score.passed is not None]
        tt_scores = [
            score.turn_taking.score
            for score in eva_scores
            if score.turn_taking.score is not None
        ]
        conc_scores = [
            score.conciseness.score
            for score in eva_scores
            if score.conciseness.score is not None
        ]
        ours_scores = [
            call.shadow_scores.ours.score
            for call in cell_calls
            if call.shadow_scores.ours is not None
            and call.shadow_scores.ours.score is not None
        ]
        cells.append(
            ConversationCellSummary(
                language=language,
                domain=domain,
                n_calls=len(cell_calls),
                n_judged=len(judged),
                n_judged_turns=len(turns),
                dimensions=dimensions,
                conciseness_failure_modes=failure_modes,
                conciseness_mean_rating=(
                    mean(turn.rating for turn in turns) if turns else None
                ),
                shadow_scores=ConversationShadowCellSummary(
                    eva_x_pass_rate=(
                        mean(float(score.passed) for score in gated) if gated else None
                    ),
                    progression_score_mean=(
                        mean(
                            score.conversation_progression.score for score in eva_scores
                        )
                        if eva_scores
                        else None
                    ),
                    conciseness_score_mean=(mean(conc_scores) if conc_scores else None),
                    turn_taking_score_mean=(mean(tt_scores) if tt_scores else None),
                    ours_score_mean=(mean(ours_scores) if ours_scores else None),
                    n_gated=len(gated),
                ),
            )
        )
    return cells


def run_conversation_judging(
    paths: list[Path],
    *,
    out_path: Path,
    cache_root: Path,
    config: ConversationJudgeConfig,
    force: bool = False,
) -> ConversationJudgeArtifact:
    """Score stored results, resume paid units, and atomically emit one artifact."""
    if not paths:
        raise ValueError("at least one results path is required")
    units: Iterable[Union[_InputUnit, ConversationCallResult]] = _iter_inputs(paths)
    if config.limit is not None:
        units = islice(units, config.limit)
    cache_root.mkdir(parents=True, exist_ok=True)

    def scorer(
        unit: Union[_InputUnit, ConversationCallResult],
    ) -> ConversationCallResult:
        if isinstance(unit, ConversationCallResult):
            return unit  # already an input-error record
        return _score_one(unit, cache_root=cache_root, config=config, force=force)

    calls = list(_bounded_map(scorer, units, config.max_concurrency))
    calls.sort(key=lambda call: (call.source, call.sim_id))
    artifact = ConversationJudgeArtifact(
        provenance=ConversationArtifactProvenance(
            created_at=get_now(),
            git_commit=get_commit_hash(),
            source_paths=[str(path.resolve()) for path in paths],
            config=config,
        ),
        calls=calls,
        cells=aggregate_cells(calls),
        num_calls=len(calls),
        num_errors=sum(len(call.errors) for call in calls),
    )
    _atomic_write_json(out_path, artifact)
    return artifact
