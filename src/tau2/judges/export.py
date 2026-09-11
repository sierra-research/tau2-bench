# Copyright Sierra
"""The judges → annotation interface: typed verdict records over stored results.

Downstream consumers (the annotation factory, calibration tooling) import ONLY
from here + ``tau2.data_model``:

- ``iter_judged_sims_detailed`` — stream sims from a results dir with
                            ``nativeness_info`` populated (reusing complete
                            stored verdicts unless ``reuse_existing`` is off),
                            memory-bounded for audio-bearing runs.
- annotation rubrics and verdict records — pydantic
  records, one per (language, factor) / (sim, judge factor) / (sim, utterance).
- ``write_verdicts_jsonl`` — serialize any record stream to JSONL.
- ``save_judged_results`` — persist judged sims back to a results location,
  preserving its storage format (dir stays dir, json stays json).
"""

import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from itertools import islice
from pathlib import Path
from typing import Annotated, Callable, Iterable, Iterator, Optional, TypeVar

from loguru import logger
from pydantic import BaseModel, Field

from tau2.config import DEFAULT_JUDGE_STREAM_CONCURRENCY
from tau2.data_model.simulation import (
    DeliveryFactorCheck,
    DeliveryFinding,
    DeliveryJudgeSettings,
    JudgeOutcome,
    NativenessFactorCheck,
    NativenessJudgeSettings,
    Results,
    SimulationRun,
)
from tau2.judges.delivery.judge import DELIVERY_JUDGE_PROMPT_VERSION
from tau2.judges.nativeness.factors import (
    DEFAULT_FACTORS,
    NativenessRubricText,
    judge_factors_for,
)
from tau2.judges.nativeness.harness import (
    NATIVENESS_RUBRIC_VERSION,
    build_agent_corpus,
)
from tau2.judges.nativeness.judge import NATIVENESS_JUDGE_PROMPT_VERSION

T = TypeVar("T")
U = TypeVar("U")


class UnregisteredLanguageError(RuntimeError):
    """A stored sim's nativeness language has no usable registered pack.

    Judging/exporting such a sim would silently destroy value: re-judging
    rebuilds ``NativenessInfo`` with deterministic factors only (overwriting
    the stored judge verdicts), and exporting emits vacuously-empty verdicts.
    """


class AnnotationRubric(BaseModel):
    """Catalog metadata shared by annotation rubrics."""

    language: Annotated[str, Field(description="ISO 639-1 language code.")]
    factor_id: Annotated[str, Field(description="Catalog factor id.")]
    category: Annotated[str, Field(description="Universal axis bucket.")]
    severity: Annotated[
        int, Field(description="Effective severity (pack override or catalog default).")
    ]
    annotator_label: Annotated[
        str,
        Field(description="Catalog plain-language violation phrasing for humans."),
    ]
    description: Annotated[
        str,
        Field(description="Catalog language-independent explanation of the axis."),
    ]


class NativenessAnnotationRubric(AnnotationRubric):
    """One language's rubric for one judge factor (the annotator's factor key)."""

    nuance: Annotated[str, Field(description="Short name shown to judge/annotator.")]
    question: Annotated[
        Optional[str], Field(description="Pack-authored binary rubric question.")
    ] = None
    native_does: Annotated[
        str, Field(description="What a native speaker does (the standard).")
    ]
    ai_likely_does: Annotated[
        str,
        Field(
            description="Common non-native error (reference only — priming risk; "
            "hidden from blind annotators)."
        ),
    ]


class NativenessVerdict(BaseModel):
    """The judge's stored verdict for one (simulation, judge factor) pair."""

    sim_id: str
    task_id: str
    language: Annotated[str, Field(description="ISO 639-1 language code of the run.")]
    factor_id: Annotated[str, Field(description="Catalog factor id.")]
    outcome: Annotated[
        Optional[JudgeOutcome],
        Field(
            description="Stored judge outcome; None when the sim carries no "
            "verdict for this factor (judge never ran / legacy result)."
        ),
    ] = None
    reasoning: Annotated[
        Optional[str], Field(description="Judge reasoning (FAIL evidence).")
    ] = None
    quote: Annotated[
        Optional[str],
        Field(description="Exact offending span, when the FAIL localizes to one."),
    ] = None
    transcript: Annotated[
        str, Field(description="Agent-only transcript the judge scored.")
    ]
    bucket: Annotated[
        Optional[str],
        Field(
            description="Construct-level calibration bucket this factor rolls "
            "into (tau2.annotation.buckets); None for retired/unbucketed."
        ),
    ] = None
    disposition: Annotated[
        Optional[str],
        Field(description="FactorDisposition value (scored/retired/audition/shadow)."),
    ] = None


class QualityVerdict(BaseModel):
    """Stored verdict for one simulation and universal quality factor."""

    sim_id: str
    task_id: str
    factor_id: str
    category: str
    severity: int
    evaluator: str
    outcome: JudgeOutcome
    evidence: Optional[str] = None
    quote: Optional[str] = None
    metrics: dict[str, float] = Field(default_factory=dict)
    thresholds: dict[str, float] = Field(default_factory=dict)
    rubric_version: str
    metrics_version: str
    bucket: Annotated[
        Optional[str],
        Field(
            description="Construct-level calibration bucket this factor rolls "
            "into (tau2.annotation.buckets); None for deterministic interaction "
            "metrics and other unbucketed factors."
        ),
    ] = None
    disposition: Annotated[
        Optional[str],
        Field(description="FactorDisposition value (scored/retired/audition/shadow)."),
    ] = None


class DeliveryVerdict(BaseModel):
    """The delivery judge's stored verdict for one (simulation, utterance) pair."""

    sim_id: str
    task_id: str
    language: Optional[str] = None
    utterance_idx: int
    outcome: JudgeOutcome
    severity: Annotated[int, Field(description="Overall 0 (clean) .. 3 (critical).")]
    flag_for_review: bool = False
    confidence: Optional[float] = None
    summary: Optional[str] = None
    expected_text: Annotated[
        Optional[str],
        Field(description="Excerpt of the intended synthesis text."),
    ] = None
    findings: Annotated[
        list[DeliveryFinding],
        Field(default_factory=list, description="Axis-tagged issues in the clip."),
    ]
    factor_checks: Annotated[
        list[DeliveryFactorCheck],
        Field(
            default_factory=list,
            description="Per-factor verdicts against the language's pack "
            "delivery rubric for this clip (id + outcome + evidence/quote, "
            "nativeness-shaped); empty in fallback/generic mode.",
        ),
    ]
    rubric_source: Annotated[
        Optional[str],
        Field(
            description="Where the judge's language-specific criteria came "
            "from ('pack' / 'fallback'); None for language-less runs."
        ),
    ] = None


def judge_factor_ids(language: Optional[str]) -> set[str]:
    """The set of ENABLED judge-factor ids a language defines (empty if none).

    Mirrors the harness, which skips disabled factors and never writes a check
    for them. Including a disabled factor here would make
    ``has_complete_nativeness_verdicts`` permanently False (its check can never
    be stored), so ``--reuse-existing`` would re-invoke the judge forever.
    """
    return {
        f.id for f in judge_factors_for(language) if f.type == "judge" and f.enabled
    }


def judge_checks(sim: SimulationRun) -> list[NativenessFactorCheck]:
    """Judge-factor checks on a sim (deterministic factors aren't judge verdicts)."""
    if sim.nativeness_info is None:
        return []
    judge_ids = judge_factor_ids(sim.nativeness_info.language)
    return [c for c in sim.nativeness_info.factor_checks if c.id in judge_ids]


def has_complete_nativeness_verdicts(sim: SimulationRun) -> bool:
    """True iff every judge factor the sim's language defines has a stored,
    usable verdict from the CURRENT judge prompt — i.e. re-running the judge
    on this stored result would buy nothing.

    Used to decide reuse. A missing factor (legacy/incomplete result), a
    DEFERRED one (judge was off), or an ERROR one (the call failed) means the
    sim must be re-judged, so gap-filling heals errors instead of treating
    them as done forever. Verdicts from an older prompt version are stale for
    the same reason: reusing them would silently mix judge designs in one
    result set.
    """
    if sim.nativeness_info is None:
        return False
    if sim.nativeness_info.rubric_version != NATIVENESS_RUBRIC_VERSION:
        return False
    expected = judge_factor_ids(sim.nativeness_info.language)
    if not expected:
        # No judge factors defined (e.g. en): nothing can be missing or stale.
        return True
    by_id = {c.id: c for c in judge_checks(sim)}
    unusable = (JudgeOutcome.DEFERRED, JudgeOutcome.ERROR)
    if not all(fid in by_id and by_id[fid].outcome not in unusable for fid in expected):
        return False
    if all(by_id[fid].outcome is JudgeOutcome.NO_OPPORTUNITY for fid in expected):
        # The judge never ran because there was nothing to judge (e.g. an
        # empty agent transcript): no prompt was involved, so no prompt
        # version can be stale — re-judging would rebuild the same
        # NO_OPPORTUNITY rows forever.
        return True
    return sim.nativeness_info.judge_prompt_version == NATIVENESS_JUDGE_PROMPT_VERSION


def has_complete_quality_verdicts(sim: SimulationRun, *, llm_judge: bool) -> bool:
    """True iff the stored quality verdicts are complete at the CURRENT rubric,
    metrics, and (for LLM factors) judge prompt versions — i.e. re-running the
    quality rubric on this stored result would buy nothing.

    An ERROR check is a gap (the call failed), and with ``llm_judge`` a
    DEFERRED LLM check is too (the judge was off when the result was scored) —
    gap-filling heals both instead of treating them as done forever.
    """
    # Lazy import: keep the quality catalog off the nativeness-only path.
    from tau2.judges.quality.factors import QUALITY_FACTORS, QUALITY_RUBRIC_VERSION
    from tau2.judges.quality.judge import QUALITY_JUDGE_PROMPT_VERSION
    from tau2.metrics.interaction_quality import QUALITY_METRICS_VERSION

    info = sim.quality_info
    if info is None:
        return False
    if info.rubric_version != QUALITY_RUBRIC_VERSION:
        return False
    if info.metrics_version != QUALITY_METRICS_VERSION:
        return False
    by_id = {c.id: c for c in info.factor_checks}
    if {f.id for f in QUALITY_FACTORS} - set(by_id):
        return False
    for check in by_id.values():
        if check.outcome is JudgeOutcome.ERROR:
            return False
        if check.evaluator == "deterministic":
            continue
        if llm_judge and check.outcome is JudgeOutcome.DEFERRED:
            return False
        # A verdict an LLM actually produced must come from the current prompt.
        if (
            check.judge_prompt_version is not None
            and check.judge_prompt_version != QUALITY_JUDGE_PROMPT_VERSION
        ):
            return False
    return True


def has_complete_delivery_verdicts(sim: SimulationRun) -> bool:
    """True iff the delivery judge already ran on this stored result with the
    CURRENT prompt and every judged utterance carries a usable (non-ERROR)
    verdict. Older-prompt verdicts are stale, never reused."""
    info = sim.delivery_info
    if info is None or not info.utterance_results:
        return False
    if info.judge_prompt_version != DELIVERY_JUDGE_PROMPT_VERSION:
        return False
    return all(r.outcome != JudgeOutcome.ERROR for r in info.utterance_results)


def require_registered_language(sim: SimulationRun) -> None:
    """Fail loudly when a sim's stored nativeness language has no usable pack.

    Without a pack, ``judge_factors_for`` returns ``[]``, so re-judging would
    rebuild ``NativenessInfo`` with only deterministic factors — dropping the
    stored judge verdicts on save — and export would emit vacuously-empty
    verdicts. Raises ``UnregisteredLanguageError`` naming the language and the
    pack data dir.
    """
    info = sim.nativeness_info
    if info is None or not info.language:
        return
    language = info.language.lower()
    if language == "en":
        return
    # Lazy import: registry pulls the pack loader; judges must not import it
    # at module import time.
    from tau2.multilingual.registry import get_language_pack
    from tau2.utils import DATA_DIR

    data_dir = DATA_DIR / "tau2" / "multilingual"
    pack = get_language_pack(language)
    deterministic_ids = {f.id for f in DEFAULT_FACTORS}
    stored_judge_ids = sorted({c.id for c in info.factor_checks} - deterministic_ids)
    if pack is None:
        raise UnregisteredLanguageError(
            f"sim '{sim.id}' records nativeness language '{language}' but no "
            f"language pack is registered for it. Expected a pack at "
            f"{data_dir / language / 'pack.yaml'} — is the multilingual data "
            f"dir missing or misconfigured? Refusing to judge/export: doing "
            f"so would drop the stored judge verdicts "
            f"({stored_judge_ids or 'none stored'})."
        )
    if stored_judge_ids and not judge_factor_ids(language):
        raise UnregisteredLanguageError(
            f"sim '{sim.id}' carries stored judge verdicts for language "
            f"'{language}' ({stored_judge_ids}) but the registered pack "
            f"defines no enabled judge factors (see "
            f"{data_dir / language / 'pack.yaml'}). Refusing to judge/export: "
            f"the stored verdicts would be silently dropped."
        )


_PROGRESS_EVERY = 10  # log a progress line every N completed sims


def bounded_map(
    fn: Callable[[T], U],
    items: Iterable[T],
    concurrency: int,
) -> Iterator[U]:
    """Map ``fn`` over a (possibly streaming) iterable with bounded concurrency.

    Keeps at most ``concurrency`` tasks in flight, advancing the source iterator
    only as slots free up — so large audio-bearing sims stay bounded in memory
    while the I/O-bound judge calls run in parallel. Yields in completion order.
    """
    it = iter(items)
    done_count = 0
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        # Prime up to `concurrency` tasks, then refill one per completion so the
        # source iterator is only pulled as fast as work finishes.
        futures = {ex.submit(fn, x) for x in islice(it, concurrency)}
        while futures:
            done, futures = wait(futures, return_when=FIRST_COMPLETED)
            for f in done:
                done_count += 1
                if done_count % _PROGRESS_EVERY == 0:
                    logger.info(f"judged {done_count} sims...")
                yield f.result()
            futures |= {ex.submit(fn, x) for x in islice(it, len(done))}


class JudgeStreamStats(BaseModel):
    """Counters for one judged stream, reported to the user."""

    judged: Annotated[
        int, Field(description="Sims the judge actually (re-)ran on.")
    ] = 0
    reused: Annotated[
        int, Field(description="Sims whose complete stored verdicts were reused.")
    ] = 0
    skipped: Annotated[
        int,
        Field(
            description="Sims skipped on per-sim failure (judge error, missing "
            "task) — logged and counted, never aborting the batch."
        ),
    ] = 0
    skipped_sim_ids: Annotated[
        list[str], Field(default_factory=list, description="Ids of skipped sims.")
    ]
    delivery_judged: Annotated[
        int, Field(description="Sims the delivery judge (re-)ran on from disk.")
    ] = 0
    delivery_reused: Annotated[
        int, Field(description="Sims whose stored delivery verdicts were reused.")
    ] = 0
    delivery_unavailable: Annotated[
        int,
        Field(
            description="Voice sims whose delivery could not be judged from "
            "disk (missing both.wav, half-duplex/mono layout, timeline "
            "mismatch) — logged and counted, nativeness verdicts unaffected."
        ),
    ] = 0
    delivery_unavailable_sim_ids: Annotated[
        list[str],
        Field(default_factory=list, description="Ids of delivery-unavailable sims."),
    ]


class JudgedSim(BaseModel):
    """One streamed sim plus whether the judge actually ran on it."""

    sim: SimulationRun
    was_judged: Annotated[
        bool,
        Field(
            description="True when the judge (re-)ran on this sim — its "
            "verdicts are new and must be persisted; False when complete "
            "stored verdicts were reused as-is."
        ),
    ]


def iter_judged_sims_detailed(
    path: Path,
    reuse_existing: bool = True,
    concurrency: int = DEFAULT_JUDGE_STREAM_CONCURRENCY,
    *,
    settings: Optional[NativenessJudgeSettings] = None,
    delivery_settings: Optional["DeliveryJudgeSettings"] = None,
    judge_nativeness: bool = True,
    stats: Optional[JudgeStreamStats] = None,
) -> Iterator[JudgedSim]:
    """Stream sims from a results location with nativeness_info populated,
    tagged with whether the judge actually ran (so callers can persist
    exactly the sims whose verdicts are new).

    Streams (``load_metadata`` for tasks + ``iter_simulations`` for sims) so we
    never hold every SimulationRun — each carries large audio — in memory at
    once. The judge calls (I/O-bound) run with bounded concurrency.

    Reuse is the default: a sim whose stored verdicts are complete is yielded
    as-is; gaps are judged. ``reuse_existing=False`` forces a full re-judge.

    ``delivery_settings`` additionally (re-)runs the DELIVERY judge from disk
    audio (``both.wav`` right channel sliced per utterance) on every streamed
    full-duplex sim lacking complete stored delivery verdicts. Delivery is
    defined for every language, so an English voice sim without stored
    nativeness results is still delivery-judged and yielded for persistence; a
    sim whose delivery judging fails keeps its nativeness verdicts and streams
    on.

    ``judge_nativeness=False`` makes the stream delivery-only: stored
    nativeness verdicts stream through untouched (never re-judged), so a
    delivery-only rescore cannot mutate the nativeness axis.

    Error policy: a sim whose judging fails (or whose task is missing) is
    logged, counted on ``stats``, and skipped — one bad sim never aborts the
    batch. The one deliberate exception is ``UnregisteredLanguageError``
    (missing language pack / data dir): that is a config error that would
    silently destroy stored verdicts on every sim, so it aborts loudly.
    """
    settings = (
        settings if settings is not None else NativenessJudgeSettings(llm_judge=True)
    )
    stats = stats if stats is not None else JudgeStreamStats()
    # judge_one runs on worker threads; stats counters are shared.
    stats_lock = threading.Lock()
    path = Path(path)
    # Audio artifacts live relative to the results dir root (dir-format runs).
    results_root = path if path.is_dir() else path.parent
    meta = Results.load_metadata(path)
    tasks_by_id = {t.id: t for t in meta.tasks}
    # The agent's perceived gender comes from the provider voice (see
    # judges.nativeness.agent_voice). attach_nativeness prefers per-sim
    # provider+voice; this run-level provider is the fallback when a sim
    # records neither.
    audio_cfg = meta.info.audio_native_config
    agent_provider = audio_cfg.provider if audio_cfg is not None else None
    domain = meta.info.environment_info.domain_name

    def judge_delivery(sim: SimulationRun) -> Optional[bool]:
        """Run/reuse the delivery judge on one sim.

        Returns True when fresh verdicts were paid for (must be persisted),
        False when complete stored verdicts were reused, None when delivery
        was not requested / not applicable / unavailable.
        """
        if delivery_settings is None or not sim.ticks:
            return None
        if reuse_existing and has_complete_delivery_verdicts(sim):
            with stats_lock:
                stats.delivery_reused += 1
            return False
        # Lazy import: keeps numpy/audio_io off the nativeness-only path.
        from tau2.judges.attach import attach_delivery_from_disk

        try:
            attach_delivery_from_disk(sim, results_root, settings=delivery_settings)
        except Exception as exc:  # noqa: BLE001 - never abort the batch
            logger.warning(f"delivery-from-disk unavailable for sim {sim.id}: {exc}")
            with stats_lock:
                stats.delivery_unavailable += 1
                stats.delivery_unavailable_sim_ids.append(sim.id)
            return None
        with stats_lock:
            stats.delivery_judged += 1
        return True

    def skip(sim: SimulationRun) -> None:
        with stats_lock:
            stats.skipped += 1
            stats.skipped_sim_ids.append(sim.id)

    def judge_one(sim: SimulationRun) -> Optional[JudgedSim]:
        if not judge_nativeness:
            # Delivery-only stream: nativeness verdicts pass through untouched
            # (nothing rebuilt → the pack guard is safely skipped).
            fresh_delivery = judge_delivery(sim)
            if sim.nativeness_info is not None:
                with stats_lock:
                    stats.reused += 1
                return JudgedSim(sim=sim, was_judged=fresh_delivery is True)
            if fresh_delivery is None:
                return None
            return JudgedSim(sim=sim, was_judged=fresh_delivery)
        # Pack guard BEFORE the reuse decision: a missing pack makes verdict
        # completeness vacuously true, and both reuse and re-judge would then
        # silently lose data. Raises — deliberately not caught below.
        require_registered_language(sim)
        if reuse_existing and has_complete_nativeness_verdicts(sim):
            with stats_lock:
                stats.reused += 1
            # Freshly paid delivery verdicts on a nativeness-reused sim must
            # still be persisted.
            return JudgedSim(sim=sim, was_judged=judge_delivery(sim) is True)
        task = tasks_by_id.get(sim.task_id)
        if task is None:
            logger.warning(f"no task for sim {sim.id} (task_id={sim.task_id}); skip")
            skip(sim)
            return None
        # Lazy import: attach pulls the evaluator graph; keep it off the import
        # path for callers that only want the pure record builders above.
        from tau2.judges.attach import attach_nativeness

        try:
            # The judge is forced ON for re-judging regardless of the run's
            # settings, else the very verdicts we are exporting would be
            # DEFERRED (callers may still override model/args via settings).
            attach_nativeness(
                sim,
                task,
                settings=settings,
                agent_provider=agent_provider,
                domain=domain,
            )
        except Exception as exc:  # noqa: BLE001 - one bad sim never aborts the batch
            logger.warning(f"judging failed for sim {sim.id}: {exc}; skipping it")
            skip(sim)
            return None
        if sim.nativeness_info is None:
            # Nativeness N/A (e.g. an English run); delivery still applies —
            # yield only when there are delivery verdicts to carry.
            fresh_delivery = judge_delivery(sim)
            if fresh_delivery is None:
                return None
            return JudgedSim(sim=sim, was_judged=fresh_delivery)
        with stats_lock:
            stats.judged += 1
        judge_delivery(sim)
        return JudgedSim(sim=sim, was_judged=True)

    sims = Results.iter_simulations(path)
    if concurrency <= 1:
        outcomes = (judge_one(sim) for sim in sims)
    else:
        outcomes = bounded_map(judge_one, sims, concurrency)
    for judged in outcomes:
        if judged is not None:
            yield judged


def save_judged_results(
    path: Path,
    judged_sims: dict[str, SimulationRun],
    output: Optional[Path] = None,
) -> Path:
    """Persist judged sims back into a results location, preserving its format.

    Loads the results at ``path``, swaps in the judged sims by id, and saves to
    ``output`` (default: back over ``path``) in the format ``path`` was stored
    in — a dir-format (voice) run stays dir-format (so the fresh verdicts are
    never stranded in a monolithic results.json that the next load would
    ignore in favor of stale simulations/*.json), and a monolithic run stays
    monolithic.
    """
    path = Path(path)
    fmt = Results.detect_format(path)
    results = Results.load(path)
    results.simulations = [judged_sims.get(s.id, s) for s in results.simulations]
    target = Path(output) if output is not None else path
    if fmt == "dir" and target.suffix != ".json":
        # A fresh dir-format target must exist as a directory before save()
        # resolves it (otherwise it is treated as a metadata FILE path and the
        # simulations/ dir lands next to it in the parent).
        target.mkdir(parents=True, exist_ok=True)
    results.save(target, format=fmt)
    return target


class AdoptVerdictStats(BaseModel):
    """Counters for one adopt-verdicts pass over a destination results set."""

    dest_sims: Annotated[int, Field(description="Sims in the destination.")] = 0
    matched: Annotated[
        int, Field(description="Destination sims with a source sim of the same id.")
    ] = 0
    nativeness_adopted: Annotated[
        int,
        Field(description="Sims that took the source's nativeness verdicts."),
    ] = 0
    delivery_adopted: Annotated[
        int, Field(description="Sims that took the source's delivery verdicts.")
    ] = 0


def _index_source_sims(src_root: Path) -> dict[str, Path]:
    """Map sim id -> sim JSON path for every dir-format sim under ``src_root``.

    Dir-format results name each sim file ``simulations/<sim_id>.json``, so the
    index is built from filenames alone — no sim is loaded until a destination
    sim actually matches.
    """
    index = {p.stem: p for p in sorted(src_root.rglob("simulations/*.json"))}
    if not index:
        raise FileNotFoundError(
            f"no dir-format sims (simulations/*.json) found under {src_root}"
        )
    return index


def adopt_stored_verdicts(src_root: Path, dst_path: Path) -> AdoptVerdictStats:
    """Copy current-version judge verdicts from sims under ``src_root`` into
    the results at ``dst_path``, matched by sim id. Never invokes a judge.

    Per axis (nativeness, delivery), a destination sim adopts the source sim's
    stored verdicts iff the source's are complete at the CURRENT judge prompt
    version and the destination's are not — so a corpus whose calls were
    freshly judged elsewhere (e.g. a materialized recall subset drawn from it)
    inherits those verdicts instead of paying for them again, while verdicts
    that are already current are never overwritten.
    """
    src_root, dst_path = Path(src_root), Path(dst_path)
    index = _index_source_sims(src_root)
    stats = AdoptVerdictStats()
    fmt = Results.detect_format(dst_path)
    results = Results.load(dst_path)
    stats.dest_sims = len(results.simulations)
    changed = False
    for sim in results.simulations:
        src_file = index.get(sim.id)
        if src_file is None:
            continue
        stats.matched += 1
        src_sim = SimulationRun.model_validate_json(src_file.read_text())
        if has_complete_nativeness_verdicts(
            src_sim
        ) and not has_complete_nativeness_verdicts(sim):
            sim.nativeness_info = src_sim.nativeness_info
            stats.nativeness_adopted += 1
            changed = True
        if has_complete_delivery_verdicts(
            src_sim
        ) and not has_complete_delivery_verdicts(sim):
            sim.delivery_info = src_sim.delivery_info
            stats.delivery_adopted += 1
            changed = True
    if changed:
        results.save(dst_path, format=fmt)
    return stats


def factor_rubrics_for(language: str) -> list[NativenessAnnotationRubric]:
    """The judge-factor rubrics a language defines, as typed records."""
    from tau2.multilingual.nativeness_catalog import get_catalog_factor

    rubrics: list[NativenessAnnotationRubric] = []
    for factor in judge_factors_for(language):
        if factor.type != "judge" or not isinstance(
            factor.params, NativenessRubricText
        ):
            continue
        catalog = get_catalog_factor(factor.id)
        rubrics.append(
            NativenessAnnotationRubric(
                language=(language or "").lower(),
                factor_id=factor.id,
                category=factor.category,
                severity=factor.severity,
                annotator_label=catalog.annotator_label,
                description=catalog.description,
                nuance=factor.params.language_criterion,
                question=factor.params.question,
                native_does=(
                    f"Shared rule: {factor.params.shared_allowed}\n"
                    f"Language-specific: {factor.params.language_allowed}"
                ),
                ai_likely_does=(
                    f"Shared violation: {factor.params.shared_violation}\n"
                    f"Language-specific: {factor.params.language_violation}"
                ),
            )
        )
    return rubrics


class DeliveryAnnotationRubric(AnnotationRubric):
    """One language's pack delivery factor, described blind-safely.

    Deliberately carries the CATALOG's language-independent ``description``
    and NOT the pack's ``listen_for`` text: listen_for names the expected
    error (like ``ai_likely_does``), which would prime blind annotators.
    """


def delivery_rubrics_for(language: str) -> list[DeliveryAnnotationRubric]:
    """The enabled pack delivery factors a language defines, blind-safely.

    Empty for a language whose pack has no ``delivery:`` block (fallback-mode
    delivery judging scores no per-factor checks, so there is nothing for an
    annotator to verify).
    """
    # Lazy imports: keep the pack registry off this module's import path.
    from tau2.judges.delivery.factors import delivery_factors_for
    from tau2.multilingual.delivery_catalog import get_delivery_catalog_factor

    rubrics: list[DeliveryAnnotationRubric] = []
    for factor in delivery_factors_for((language or "").lower()):
        if not factor.enabled:
            continue
        catalog = get_delivery_catalog_factor(factor.id)
        rubrics.append(
            DeliveryAnnotationRubric(
                language=(language or "").lower(),
                factor_id=factor.id,
                category=factor.category,
                severity=factor.severity,
                description=catalog.description,
                annotator_label=catalog.annotator_label,
            )
        )
    return rubrics


def _bucket_stamp(judge_family: str, factor_id: str) -> tuple[Optional[str], str]:
    """(bucket, disposition) for one factor under the calibration catalog.

    Lazy import: ``tau2.annotation`` consumes this module, so the catalog is
    pulled at call time only (same pattern as the quality-catalog import).
    """
    from tau2.annotation.buckets import bucket_for_factor, factor_disposition
    from tau2.annotation.calibration_draw import CalibrationJudge

    judge = CalibrationJudge(judge_family)
    return (
        bucket_for_factor(judge, factor_id),
        factor_disposition(judge, factor_id).value,
    )


def nativeness_verdicts(sim: SimulationRun) -> list[NativenessVerdict]:
    """One record per (sim, judge factor) the sim's language defines.

    A factor with no stored check is emitted with ``outcome=None`` (no verdict)
    so completeness gaps stay visible downstream rather than silently vanishing.
    """
    info = sim.nativeness_info
    if info is None or not info.language:
        return []
    transcript = build_agent_corpus(sim)
    by_id = {c.id: c for c in judge_checks(sim)}
    records: list[NativenessVerdict] = []
    for factor_id in sorted(judge_factor_ids(info.language)):
        check = by_id.get(factor_id)
        bucket, disposition = _bucket_stamp("nativeness", factor_id)
        records.append(
            NativenessVerdict(
                sim_id=sim.id,
                task_id=sim.task_id,
                language=info.language,
                factor_id=factor_id,
                outcome=check.outcome if check else None,
                reasoning=(check.evidence if check else None) or None,
                quote=(check.quote if check else None) or None,
                transcript=transcript,
                bucket=bucket,
                disposition=disposition,
            )
        )
    return records


def delivery_verdicts(sim: SimulationRun) -> list[DeliveryVerdict]:
    """One record per judged utterance stored on the sim (empty if not judged)."""
    info = sim.delivery_info
    if info is None:
        return []
    return [
        DeliveryVerdict(
            sim_id=sim.id,
            task_id=sim.task_id,
            language=info.language,
            utterance_idx=r.utterance_idx,
            outcome=r.outcome,
            severity=r.severity,
            flag_for_review=r.flag_for_review,
            confidence=r.confidence,
            summary=r.summary,
            expected_text=r.expected_text,
            findings=list(r.findings),
            factor_checks=list(r.factor_checks),
            rubric_source=info.rubric_source,
        )
        for r in info.utterance_results
    ]


def quality_verdicts(sim: SimulationRun) -> list[QualityVerdict]:
    """One typed record per stored universal quality-factor check."""
    info = sim.quality_info
    if info is None:
        return []
    records: list[QualityVerdict] = []
    for check in info.factor_checks:
        bucket, disposition = _bucket_stamp("quality", check.id)
        records.append(
            QualityVerdict(
                sim_id=sim.id,
                task_id=sim.task_id,
                factor_id=check.id,
                category=check.category,
                severity=check.severity,
                evaluator=check.evaluator,
                outcome=check.outcome,
                evidence=check.evidence,
                quote=check.quote,
                metrics=dict(check.metrics),
                thresholds=dict(check.thresholds),
                rubric_version=info.rubric_version,
                metrics_version=info.metrics_version,
                bucket=bucket,
                disposition=disposition,
            )
        )
    return records


class BucketRollupRow(BaseModel):
    """Call-level any-FAIL rollup of one calibration bucket over an export.

    A call is scored for a bucket when the merged outcome of its constituent
    factor verdicts is PASS or FAIL (``tau2.annotation.buckets`` semantics);
    it fails when any constituent fails. Retired/audition/shadow factors never
    contribute.
    """

    group: Annotated[
        str, Field(description="Display group (efficiency/nativeness/delivery).")
    ]
    judge: Annotated[str, Field(description="Owning judge family.")]
    bucket_id: str
    factor_ids: list[str]
    calls_scored: Annotated[
        int, Field(description="Calls whose merged outcome is PASS or FAIL.", ge=0)
    ]
    calls_failed: Annotated[
        int, Field(description="Calls whose merged outcome is FAIL.", ge=0)
    ]
    fail_rate: Annotated[
        Optional[float],
        Field(description="calls_failed / calls_scored; None when unscored."),
    ] = None


def bucket_rollup(
    nat_records: Iterable[NativenessVerdict],
    quality_records: Iterable[QualityVerdict],
    delivery_records: Iterable[DeliveryVerdict],
) -> list[BucketRollupRow]:
    """Roll exported verdicts up into the calibration buckets, per call.

    Nativeness and quality verdicts join their bucket by stamped factor id.
    Delivery joins per constituent, NOT per utterance outcome: the stored
    utterance ``outcome`` fails on ANY axis finding (intonation included, and
    intonation is retired from buckets), so each delivery bucket instead
    fails on axis findings whose axis is a constituent (``fidelity``) or on
    failing pack factor checks whose id is a constituent
    (``tone_meaning_flip``). Errored utterances stay unscorable.
    """
    from tau2.annotation.buckets import (
        CALIBRATION_BUCKETS,
        JUDGE_GROUP_LABELS,
        bucket_outcome,
    )
    from tau2.annotation.calibration_draw import CalibrationJudge

    outcomes: dict[tuple[str, str], dict[str, list[Optional[JudgeOutcome]]]] = {}

    def feed(judge: str, bucket_id: Optional[str], sim_id: str, outcome) -> None:
        if bucket_id is None:
            return
        outcomes.setdefault((judge, bucket_id), {}).setdefault(sim_id, []).append(
            outcome
        )

    delivery_buckets = [
        bucket
        for bucket in CALIBRATION_BUCKETS
        if bucket.judge is CalibrationJudge.DELIVERY
    ]

    def delivery_constituent_outcome(
        deliv: DeliveryVerdict, factor_ids: tuple[str, ...]
    ) -> JudgeOutcome:
        if deliv.outcome is JudgeOutcome.ERROR:
            return JudgeOutcome.ERROR
        axis_fail = any(f.axis in factor_ids for f in deliv.findings)
        factor_fail = any(
            c.id in factor_ids and c.outcome is JudgeOutcome.FAIL
            for c in deliv.factor_checks
        )
        return JudgeOutcome.FAIL if (axis_fail or factor_fail) else JudgeOutcome.PASS

    for nat in nat_records:
        feed("nativeness", nat.bucket, nat.sim_id, nat.outcome)
    for qual in quality_records:
        feed("quality", qual.bucket, qual.sim_id, qual.outcome)
    for deliv in delivery_records:
        for bucket in delivery_buckets:
            feed(
                "delivery",
                bucket.bucket_id,
                deliv.sim_id,
                delivery_constituent_outcome(deliv, bucket.factor_ids),
            )

    rows: list[BucketRollupRow] = []
    for bucket in CALIBRATION_BUCKETS:
        per_call = outcomes.get((bucket.judge.value, bucket.bucket_id), {})
        merged = [bucket_outcome(cell) for cell in per_call.values()]
        scored = sum(m in (JudgeOutcome.PASS, JudgeOutcome.FAIL) for m in merged)
        failed = sum(m is JudgeOutcome.FAIL for m in merged)
        rows.append(
            BucketRollupRow(
                group=JUDGE_GROUP_LABELS.get(
                    CalibrationJudge(bucket.judge.value), bucket.judge.value
                ),
                judge=bucket.judge.value,
                bucket_id=bucket.bucket_id,
                factor_ids=list(bucket.factor_ids),
                calls_scored=scored,
                calls_failed=failed,
                fail_rate=(failed / scored) if scored else None,
            )
        )
    rows.sort(key=lambda r: (r.group, r.bucket_id))
    return rows


def render_bucket_rollup(rows: Iterable[BucketRollupRow]) -> str:
    """Fixed-width table of the bucket rollup for terminal output."""
    lines = [f"{'group':<12} {'bucket':<18} {'scored':>6} {'failed':>6} {'rate':>6}"]
    for row in rows:
        rate = f"{row.fail_rate:.2f}" if row.fail_rate is not None else "-"
        lines.append(
            f"{row.group:<12} {row.bucket_id:<18} "
            f"{row.calls_scored:>6} {row.calls_failed:>6} {rate:>6}"
        )
        lines.append(f"{'':<12}   = {' + '.join(row.factor_ids)}")
    return "\n".join(lines)


def write_verdicts_jsonl(records: Iterable[BaseModel], path: Path) -> int:
    """Write pydantic records to ``path`` as JSONL; returns the row count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as fp:
        for record in records:
            fp.write(record.model_dump_json())
            fp.write("\n")
            count += 1
    logger.info(f"wrote {count} records -> {path}")
    return count
