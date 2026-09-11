# Copyright Sierra
"""One-pass annotation-calibrated judging over stored results.

``tau2 judges suite`` runs exactly the calibrated judge set — nativeness,
quality, delivery — over one or more results roots (paths only), in two
sequential phases:

1. **text** — nativeness + quality, fanned across ``text_processes`` worker
   processes;
2. **audio** — the per-utterance delivery judge from disk audio, fanned across
   ``audio_processes`` (delivery payloads are heavier, so fewer processes).

Each worker process runs ``concurrency`` judge calls in flight: the
concurrency ceiling in this codebase is GIL-bound PER PROCESS (the same reason
``tau2.runner`` fans simulations across worker processes), so the suite
multiplies per-process thread concurrency by a process count per phase.

Semantics carried over from the individual verbs:

- **gap-filling by default** — complete stored verdicts at the current judge
  prompt / rubric versions are reused (``--rejudge`` forces a full re-judge);
- **one trial only by default** — sims outside ``trials`` (default ``{0}``)
  are not judged at all (``--trials all`` opts into every trial);
- **English nativeness skip** — en has zero nativeness judge factors, so
  English (and language-less) cells silently skip that family;
- **no delivery sampling cap** — every call with stored audio is judged
  (``sample_rate=1.0``; the per-conversation utterance cap still applies);
- **format-preserving persistence** — judged sims are written back into the
  results in their original storage format (per-sim files for dir-format
  roots, so disjoint shards never contend; a whole-root save for monolithic
  JSON, which therefore always runs as a single shard);
- **provenance** — one typed suite-level record per results root
  (``judge_suite_provenance.json``) recording, per judge family, the judge
  model, prompt / rubric versions, the invocation that last ran it (flags,
  trials, timestamp), and which sims it touched. The record accumulates
  ACROSS invocations: a later ``--only delivery`` pass updates the delivery
  entry and keeps the text families' entries intact (newest entry per
  family), so partial invocations never erase each other's provenance.
"""

import json
import multiprocessing
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Annotated, Iterator, Literal, Optional

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator

from tau2.config import (
    DEFAULT_JUDGE_STREAM_CONCURRENCY,
    DEFAULT_JUDGE_SUITE_AUDIO_PROCESSES,
    DEFAULT_JUDGE_SUITE_TEXT_PROCESSES,
    DEFAULT_LLM_DELIVERY_JUDGE,
)
from tau2.data_model.simulation import (
    DeliveryJudgeSettings,
    NativenessJudgeSettings,
    QualityJudgeSettings,
    Results,
    SimulationRun,
)
from tau2.evaluator.evaluator import get_simulation_language_info
from tau2.judges.export import (
    bounded_map,
    has_complete_delivery_verdicts,
    has_complete_nativeness_verdicts,
    has_complete_quality_verdicts,
    require_registered_language,
    save_judged_results,
)

PROVENANCE_SCHEMA_VERSION = "judge-suite-provenance-v2"
PROVENANCE_FILENAME = "judge_suite_provenance.json"
# v1 records (one flat invocation per file, overwritten on every run) are
# upgraded on read so a partial re-run merges into them instead of erasing them.
PROVENANCE_SCHEMA_VERSION_V1 = "judge-suite-provenance-v1"


class SuiteFamily(str, Enum):
    """One judge family in the annotation-calibrated set."""

    NATIVENESS = "nativeness"
    QUALITY = "quality"
    DELIVERY = "delivery"


ALL_FAMILIES: frozenset[SuiteFamily] = frozenset(SuiteFamily)
# Which phase runs each family: text fans wider than audio.
TEXT_FAMILIES: frozenset[SuiteFamily] = frozenset(
    {SuiteFamily.NATIVENESS, SuiteFamily.QUALITY}
)
AUDIO_FAMILIES: frozenset[SuiteFamily] = frozenset({SuiteFamily.DELIVERY})


class SuiteConfig(BaseModel):
    """Validated configuration for one ``tau2 judges suite`` invocation."""

    model_config = ConfigDict(extra="forbid")

    results: Annotated[
        list[Path],
        Field(min_length=1, description="Results dir(s) or results.json file(s)."),
    ]
    concurrency: Annotated[
        int,
        Field(
            ge=1,
            default=DEFAULT_JUDGE_STREAM_CONCURRENCY,
            description="Per-process simulations in flight (I/O-bound judge calls).",
        ),
    ]
    text_processes: Annotated[
        int,
        Field(
            ge=1,
            default=DEFAULT_JUDGE_SUITE_TEXT_PROCESSES,
            description="Worker processes for the nativeness + quality phase.",
        ),
    ]
    audio_processes: Annotated[
        int,
        Field(
            ge=1,
            default=DEFAULT_JUDGE_SUITE_AUDIO_PROCESSES,
            description="Worker processes for the delivery phase.",
        ),
    ]
    rejudge: Annotated[
        bool,
        Field(
            default=False,
            description="Force a full re-judge instead of filling gaps.",
        ),
    ]
    only: Annotated[
        frozenset[SuiteFamily],
        Field(
            default=ALL_FAMILIES,
            description="Judge families to run (default: all three).",
        ),
    ]
    trials: Annotated[
        Optional[frozenset[int]],
        Field(
            default=frozenset({0}),
            description="Trial indices to judge; sims outside the selection are "
            "not judged at all (owner decision: judge ONE trial by default). "
            "None means every trial (--trials all).",
        ),
    ]

    @field_validator("only")
    @classmethod
    def _non_empty_only(cls, value: frozenset[SuiteFamily]) -> frozenset[SuiteFamily]:
        if not value:
            raise ValueError("at least one judge family must be selected")
        return value

    @field_validator("trials")
    @classmethod
    def _non_empty_trials(
        cls, value: Optional[frozenset[int]]
    ) -> Optional[frozenset[int]]:
        if value is not None and not value:
            raise ValueError("at least one trial index must be selected")
        return value

    def trial_selected(self, trial: Optional[int]) -> bool:
        """Whether a sim with this ``trial`` is judged under the selection."""
        if self.trials is None:
            return True
        return trial in self.trials


class FamilyTouches(BaseModel):
    """Which sims one judge family touched, and how."""

    judged: Annotated[
        list[str],
        Field(default_factory=list, description="Sims the judge (re-)ran on."),
    ]
    reused: Annotated[
        list[str],
        Field(
            default_factory=list,
            description="Sims whose complete stored verdicts were reused.",
        ),
    ]
    skipped: Annotated[
        list[str],
        Field(
            default_factory=list,
            description="Sims not judged: unselected trial, no applicable "
            "judge (en/language-less nativeness, tick-less delivery), a "
            "missing task, or a per-sim judge failure.",
        ),
    ]
    unavailable: Annotated[
        list[str],
        Field(
            default_factory=list,
            description="Voice sims whose delivery could not be judged from "
            "disk (missing/mono both.wav, timeline mismatch).",
        ),
    ]

    def merge(self, other: "FamilyTouches") -> "FamilyTouches":
        """Combine two shards' touches (sorted, ids are disjoint by design)."""
        return FamilyTouches(
            judged=sorted(self.judged + other.judged),
            reused=sorted(self.reused + other.reused),
            skipped=sorted(self.skipped + other.skipped),
            unavailable=sorted(self.unavailable + other.unavailable),
        )


class FamilyInvocation(BaseModel):
    """The suite invocation that produced one family's provenance entry."""

    created_at: Annotated[str, Field(description="UTC ISO-8601 timestamp.")]
    concurrency: Annotated[int, Field(description="Per-process simulations in flight.")]
    rejudge: Annotated[
        bool,
        Field(description="Whether the pass forced a full re-judge (no reuse)."),
    ]
    trials: Annotated[
        Optional[list[int]],
        Field(
            description="Trial indices judged (sorted); None means every trial "
            "(--trials all)."
        ),
    ]


class FamilyProvenance(BaseModel):
    """Judge configuration + touches for one family over one results root."""

    invocation: Annotated[
        FamilyInvocation,
        Field(description="The suite invocation that last ran this family."),
    ]
    judge_model: Annotated[
        str,
        Field(
            description="Model id, or 'factor-catalog-default' when each "
            "factor pins its own model."
        ),
    ]
    judge_prompt_version: Annotated[
        str, Field(description="Versioned judge prompt in force for this pass.")
    ]
    rubric_version: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Versioned factor catalog, for families that have one.",
        ),
    ]
    metrics_version: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Deterministic extractor version (quality only).",
        ),
    ]
    touches: FamilyTouches


class SuiteProvenance(BaseModel):
    """The suite-level provenance record for one results root.

    Accumulates across invocations: each write merges this invocation's
    families over the entries already on disk (newest per family), so
    invocation-scoped fields live per family, not at the top level.
    """

    schema_version: Literal["judge-suite-provenance-v2"] = PROVENANCE_SCHEMA_VERSION
    updated_at: Annotated[
        str, Field(description="UTC ISO-8601 timestamp of the last write.")
    ]
    results_path: Annotated[str, Field(description="The judged results root.")]
    families: Annotated[
        dict[str, FamilyProvenance],
        Field(
            description="Per-family judge configuration, invocation metadata, "
            "and touched sims — the newest entry per family across every "
            "suite invocation on this root."
        ),
    ]


def _nativeness_settings() -> NativenessJudgeSettings:
    """The calibrated nativeness judge settings (LLM judge forced ON, as in
    rejudge: without it the very verdicts the suite exists to fill would be
    recorded DEFERRED)."""
    return NativenessJudgeSettings(llm_judge=True)


def _quality_settings() -> QualityJudgeSettings:
    """The calibrated quality settings: the task-aware semantic factors run
    (deterministic-only quality needs no fan-out and is not what annotation
    calibrates against)."""
    return QualityJudgeSettings(llm_judge=True)


def _delivery_settings() -> DeliveryJudgeSettings:
    """The calibrated delivery settings: judge EVERY call that has stored
    audio (no conversation sampling; the per-conversation utterance cap
    stays)."""
    return DeliveryJudgeSettings(sample_rate=1.0)


def _family_provenance(
    family: SuiteFamily,
    touches: FamilyTouches,
    invocation: FamilyInvocation,
) -> FamilyProvenance:
    """The recorded judge configuration for one family, from the live constants."""
    if family is SuiteFamily.NATIVENESS:
        from tau2.judges.nativeness.judge import NATIVENESS_JUDGE_PROMPT_VERSION

        return FamilyProvenance(
            invocation=invocation,
            judge_model=_nativeness_settings().model,
            judge_prompt_version=NATIVENESS_JUDGE_PROMPT_VERSION,
            touches=touches,
        )
    if family is SuiteFamily.QUALITY:
        from tau2.judges.quality.factors import QUALITY_RUBRIC_VERSION
        from tau2.judges.quality.judge import QUALITY_JUDGE_PROMPT_VERSION
        from tau2.metrics.interaction_quality import QUALITY_METRICS_VERSION

        settings = _quality_settings()
        return FamilyProvenance(
            invocation=invocation,
            judge_model=settings.model or "factor-catalog-default",
            judge_prompt_version=QUALITY_JUDGE_PROMPT_VERSION,
            rubric_version=QUALITY_RUBRIC_VERSION,
            metrics_version=QUALITY_METRICS_VERSION,
            touches=touches,
        )
    from tau2.judges.delivery.judge import DELIVERY_JUDGE_PROMPT_VERSION

    return FamilyProvenance(
        invocation=invocation,
        judge_model=DEFAULT_LLM_DELIVERY_JUDGE,
        judge_prompt_version=DELIVERY_JUDGE_PROMPT_VERSION,
        touches=touches,
    )


# --- sharded per-root work ----------------------------------------------------


class ShardTask(BaseModel):
    """One worker process's slice of one results root for one phase.

    Crosses the process boundary as a validated model (dumped to JSON on the
    way in, revalidated in the worker), so the worker seam stays typed.
    """

    model_config = ConfigDict(extra="forbid")

    root: str
    shard_index: Annotated[int, Field(ge=0)]
    num_shards: Annotated[int, Field(ge=1)]
    phase: Literal["text", "audio"]
    families: list[SuiteFamily]
    concurrency: Annotated[int, Field(ge=1)]
    rejudge: bool
    trials: Optional[list[int]]

    def trial_selected(self, trial: Optional[int]) -> bool:
        if self.trials is None:
            return True
        return trial in self.trials


class ShardReport(BaseModel):
    """What one shard touched, per family."""

    root: str
    touches: dict[str, FamilyTouches] = Field(default_factory=dict)


def _iter_shard_sims(task: ShardTask) -> Iterator[SimulationRun]:
    """Stream this shard's sims: every ``num_shards``-th sim in stable
    (sorted) order, so shards are disjoint and cover the root exactly."""
    for idx, sim in enumerate(Results.iter_simulations(Path(task.root))):
        if idx % task.num_shards == task.shard_index:
            yield sim


def _dir_results_root(root: Path) -> Path:
    """The directory that owns ``simulations/`` (and the run's audio files),
    whether the root was given as the dir itself or its results.json."""
    return root if root.is_dir() else root.parent


def _persist_sim(root: Path, sim: SimulationRun) -> None:
    """Write one judged sim back into a DIR-format root (its own file, so
    disjoint shards never contend; matches ``Results.save`` dir output)."""
    sim_path = _dir_results_root(root) / "simulations" / f"{sim.id}.json"
    if not sim_path.parent.is_dir():
        raise FileNotFoundError(f"not a dir-format results root: {root}")
    sim_path.write_text(sim.model_dump_json(indent=2))


def _nativeness_language(sim: SimulationRun) -> Optional[str]:
    """The language the nativeness family would judge under, if any."""
    if sim.nativeness_info is not None and sim.nativeness_info.language:
        return sim.nativeness_info.language.lower()
    language, _script = get_simulation_language_info(sim)
    return language.lower() if language else None


def run_shard(task: ShardTask) -> ShardReport:
    """Judge one shard of one root for one phase; persist what was judged.

    Per-sim failures are logged and recorded as skipped — one bad sim never
    aborts the shard (mirroring the individual verbs' error policy).
    """
    root = Path(task.root)
    fmt = Results.detect_format(root)
    meta = Results.load_metadata(root)
    tasks_by_id = {t.id: t for t in meta.tasks}
    audio_cfg = meta.info.audio_native_config
    agent_provider = audio_cfg.provider if audio_cfg is not None else None
    domain = meta.info.environment_info.domain_name
    families = set(task.families)
    touches = {family.value: FamilyTouches() for family in sorted(families)}
    nativeness_settings = _nativeness_settings()
    quality_settings = _quality_settings()
    delivery_settings = _delivery_settings()

    def judge_nativeness(sim: SimulationRun) -> bool:
        record = touches[SuiteFamily.NATIVENESS.value]
        language = _nativeness_language(sim)
        if language is None or language == "en":
            # en defines zero nativeness judge factors: skipped silently.
            record.skipped.append(sim.id)
            return False
        # Pack guard BEFORE the reuse decision (see iter_judged_sims_detailed):
        # a missing pack makes completeness vacuously true and would lose data.
        require_registered_language(sim)
        if not task.rejudge and has_complete_nativeness_verdicts(sim):
            record.reused.append(sim.id)
            return False
        sim_task = tasks_by_id.get(sim.task_id)
        if sim_task is None:
            logger.warning(f"no task for sim {sim.id} (task_id={sim.task_id}); skip")
            record.skipped.append(sim.id)
            return False
        from tau2.judges.attach import attach_nativeness

        try:
            attach_nativeness(
                sim,
                sim_task,
                settings=nativeness_settings,
                agent_provider=agent_provider,
                domain=domain,
            )
        except Exception as exc:  # noqa: BLE001 - one bad sim never aborts the shard
            logger.warning(f"nativeness judging failed for sim {sim.id}: {exc}")
            record.skipped.append(sim.id)
            return False
        record.judged.append(sim.id)
        return True

    def judge_quality(sim: SimulationRun) -> bool:
        record = touches[SuiteFamily.QUALITY.value]
        if not task.rejudge and has_complete_quality_verdicts(sim, llm_judge=True):
            record.reused.append(sim.id)
            return False
        sim_task = tasks_by_id.get(sim.task_id)
        if sim_task is None:
            logger.warning(f"no task for sim {sim.id} (task_id={sim.task_id}); skip")
            record.skipped.append(sim.id)
            return False
        from tau2.judges.attach import attach_quality

        try:
            attach_quality(sim, sim_task, domain=domain, settings=quality_settings)
        except Exception as exc:  # noqa: BLE001 - one bad sim never aborts the shard
            logger.warning(f"quality judging failed for sim {sim.id}: {exc}")
            record.skipped.append(sim.id)
            return False
        record.judged.append(sim.id)
        return True

    def judge_delivery(sim: SimulationRun) -> bool:
        record = touches[SuiteFamily.DELIVERY.value]
        if not sim.ticks:
            # No tick timeline (text / half-duplex run): no stored audio to
            # judge — skipped silently.
            record.skipped.append(sim.id)
            return False
        if not task.rejudge and has_complete_delivery_verdicts(sim):
            record.reused.append(sim.id)
            return False
        from tau2.judges.attach import attach_delivery_from_disk

        try:
            # Audio artifacts live relative to the results dir root.
            attach_delivery_from_disk(
                sim, _dir_results_root(root), settings=delivery_settings
            )
        except Exception as exc:  # noqa: BLE001 - one bad sim never aborts the shard
            logger.warning(f"delivery-from-disk unavailable for sim {sim.id}: {exc}")
            record.unavailable.append(sim.id)
            return False
        record.judged.append(sim.id)
        return True

    def judge_one(sim: SimulationRun) -> Optional[SimulationRun]:
        """Run this phase's selected families on one sim; return it if judged."""
        if not task.trial_selected(sim.trial):
            for family in families:
                touches[family.value].skipped.append(sim.id)
            return None
        modified = False
        if SuiteFamily.NATIVENESS in families:
            modified |= judge_nativeness(sim)
        if SuiteFamily.QUALITY in families:
            modified |= judge_quality(sim)
        if SuiteFamily.DELIVERY in families:
            modified |= judge_delivery(sim)
        return sim if modified else None

    judged: dict[str, SimulationRun] = {}
    sims = _iter_shard_sims(task)
    if task.concurrency <= 1:
        outcomes = (judge_one(sim) for sim in sims)
    else:
        outcomes = bounded_map(judge_one, sims, task.concurrency)
    for sim in outcomes:
        if sim is None:
            continue
        if fmt == "dir":
            # Immediate per-sim persistence: paid verdicts survive a crash.
            _persist_sim(root, sim)
        else:
            judged[sim.id] = sim
    if judged:
        # Monolithic JSON cannot take disjoint per-sim writes, which is why a
        # json-format root always runs as a single shard.
        save_judged_results(root, judged)
    # Deterministic report: ids sorted, not completion-ordered.
    for record in touches.values():
        record.judged.sort()
        record.reused.sort()
        record.skipped.sort()
        record.unavailable.sort()
    return ShardReport(root=task.root, touches=touches)


def _run_shard_payload(payload: dict) -> dict:
    """Pickle-friendly worker entrypoint: typed models on both sides."""
    return run_shard(ShardTask.model_validate(payload)).model_dump(mode="json")


# --- phase orchestration ------------------------------------------------------


def _count_sims(root: Path) -> int:
    """How many sims a root holds, without loading them (dir) or their models
    (json)."""
    if Results.detect_format(root) == "dir":
        sims_dir = (root if root.is_dir() else root.parent) / "simulations"
        return len(list(sims_dir.glob("*.json"))) if sims_dir.is_dir() else 0
    with open(root, "r") as f:
        return len(json.load(f).get("simulations", []))


def plan_phase_shards(
    config: SuiteConfig,
    phase: Literal["text", "audio"],
    families: list[SuiteFamily],
) -> list[ShardTask]:
    """The phase's work items: every root split into disjoint shards.

    A dir-format root gets up to ``processes`` shards (never more than it has
    sims); a monolithic JSON root always gets exactly one (whole-root save).
    """
    processes = config.text_processes if phase == "text" else config.audio_processes
    tasks: list[ShardTask] = []
    for root in config.results:
        if Results.detect_format(root) == "json":
            num_shards = 1
        else:
            num_shards = max(1, min(processes, _count_sims(root)))
        tasks.extend(
            ShardTask(
                root=str(root),
                shard_index=shard,
                num_shards=num_shards,
                phase=phase,
                families=sorted(families),
                concurrency=config.concurrency,
                rejudge=config.rejudge,
                trials=sorted(config.trials) if config.trials is not None else None,
            )
            for shard in range(num_shards)
        )
    return tasks


def _run_phase(
    config: SuiteConfig,
    phase: Literal["text", "audio"],
) -> dict[str, dict[str, FamilyTouches]]:
    """Run one phase to completion; returns per-root, per-family touches."""
    phase_families = TEXT_FAMILIES if phase == "text" else AUDIO_FAMILIES
    families = sorted(config.only & phase_families)
    merged: dict[str, dict[str, FamilyTouches]] = {}
    if not families:
        return merged
    processes = config.text_processes if phase == "text" else config.audio_processes
    shard_tasks = plan_phase_shards(config, phase, families)
    logger.info(
        f"suite phase {phase}: {[f.value for f in families]} over "
        f"{len(config.results)} root(s), {len(shard_tasks)} shard(s), "
        f"{min(processes, len(shard_tasks))} process(es) x "
        f"{config.concurrency} in flight"
    )
    if processes == 1 or len(shard_tasks) == 1:
        reports = [run_shard(task) for task in shard_tasks]
    else:
        payloads = [task.model_dump(mode="json") for task in shard_tasks]
        # spawn: never fork a process that may hold live client/thread state.
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=min(processes, len(shard_tasks))) as pool:
            reports = [
                ShardReport.model_validate(raw)
                for raw in pool.map(_run_shard_payload, payloads)
            ]
    for report in reports:
        root_touches = merged.setdefault(report.root, {})
        for family, touches in report.touches.items():
            existing = root_touches.get(family)
            root_touches[family] = (
                touches if existing is None else existing.merge(touches)
            )
    return merged


def provenance_path(root: Path) -> Path:
    """Where the suite provenance record lives for a results root."""
    root = Path(root)
    if Results.detect_format(root) == "dir" and root.is_dir():
        return root / PROVENANCE_FILENAME
    return root.parent / f"{root.stem}.{PROVENANCE_FILENAME}"


def _load_prior_families(path: Path) -> dict[str, FamilyProvenance]:
    """The family entries already recorded for a root, for merge-on-write.

    A v1 record (flat invocation fields at the top level, overwritten on every
    run) is upgraded in place: its top-level invocation metadata is copied into
    each family entry. An unreadable or unknown-version file is logged and
    treated as absent — the current invocation's provenance must still land.
    """
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
        version = raw.get("schema_version")
        if version == PROVENANCE_SCHEMA_VERSION:
            return SuiteProvenance.model_validate(raw).families
        if version == PROVENANCE_SCHEMA_VERSION_V1:
            invocation = FamilyInvocation(
                created_at=raw["created_at"],
                concurrency=raw["concurrency"],
                rejudge=raw["rejudge"],
                trials=raw["trials"],
            )
            return {
                name: FamilyProvenance.model_validate(
                    {**entry, "invocation": invocation.model_dump()}
                )
                for name, entry in raw["families"].items()
            }
        raise ValueError(f"unknown schema_version: {version!r}")
    except Exception as exc:  # noqa: BLE001 - a bad record never aborts the write
        logger.warning(
            f"cannot merge existing suite provenance at {path} ({exc}); "
            "its family entries are dropped from the rewritten record"
        )
        return {}


def _refresh_simulation_index(root: Path) -> None:
    """Rebuild a dir-format root's simulation index from the sims on disk.

    Per-sim shard writes update verdicts without touching results.json, so the
    index's headline score columns go stale; this streams the sims back
    through once (memory bounded by one sim) and rewrites metadata only.
    """
    meta = Results.load_metadata(root)
    meta.simulation_index = [
        Results.index_entry(sim) for sim in Results.iter_simulations(root)
    ]
    meta.save_metadata(root if root.is_dir() else root.parent)


def run_suite(config: SuiteConfig) -> list[SuiteProvenance]:
    """Run the two suite phases over every results root; returns (and writes)
    one provenance record per root."""
    all_touches: dict[str, dict[str, FamilyTouches]] = {
        str(root): {} for root in config.results
    }
    for phase in ("text", "audio"):
        for root, families in _run_phase(config, phase).items():
            all_touches[root].update(families)
    records: list[SuiteProvenance] = []
    for root in config.results:
        touches = all_touches[str(root)]
        if Results.detect_format(root) == "dir" and any(
            t.judged for t in touches.values()
        ):
            _refresh_simulation_index(Path(root))
        now = datetime.now(timezone.utc).isoformat()
        invocation = FamilyInvocation(
            created_at=now,
            concurrency=config.concurrency,
            rejudge=config.rejudge,
            trials=sorted(config.trials) if config.trials is not None else None,
        )
        ran = {
            family: _family_provenance(SuiteFamily(family), family_touches, invocation)
            for family, family_touches in touches.items()
        }
        out = provenance_path(Path(root))
        # Merge-on-write: this invocation's families over the entries already
        # on disk (this pass is by construction the newest per family it ran),
        # so a partial --only pass never erases the other families' record.
        families = {**_load_prior_families(out), **ran}
        record = SuiteProvenance(
            updated_at=now,
            results_path=str(root),
            families=dict(sorted(families.items())),
        )
        out.write_text(record.model_dump_json(indent=2))
        logger.info(f"suite provenance -> {out}")
        records.append(record)
    return records
