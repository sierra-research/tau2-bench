# Copyright Sierra
"""Judge-calibration packets: the blind annotate wave and the adjudication view.

Two modes over ONE cohort (`tau2 annotate calibration-packets`):

- **annotate** — the rater-facing packet. Runs the defect-enriched draw
  (``tau2.annotation.calibration_draw``) over stored judge results, then
  renders the drawn calls through the SAME single-page dual-evidence UI as the
  combined rubric packet, with the calibration section set (interaction +
  semantic + nativeness + audio incl. pack delivery factors). JUDGE-BLIND BY
  CONSTRUCTION: the renderer receives only call evidence and pack rubric text
  — never judge verdicts, arms, rewards, scores, or sampling reasons — and the
  in-packet manifest entries (``CalibrationPacketEntry``) carry no arm or
  results-path fields. The sampling frame and the coverage sidecar (which DO
  say why each call was drawn) are written OUTSIDE the packet directory and
  never ship.

  ``--show-judge`` is the OPT-IN judge-visible variant: a TWO-PHASE
  blind-then-reveal instrument, never the plain blind wave. Per call, phase 1
  is answered with every judge finding hidden (nothing judge-derived in the
  visible DOM); locking the answers freezes them as the immutable pre-reveal
  layer and reveals the stored judge verdicts, evidence, and pinned quotes
  inline beneath their calibrating questions (uniform closed toggles on EVERY
  question, so the collapsed page never says which were flagged); phase 2
  revisions are recorded as a distinct post-reveal layer and completing the
  call means confirming phase 2. Exports carry BOTH layers (a ``phase``
  column on every row); ``calibration-agreement`` routes pre-reveal rows to
  the blind (recall) pool and post-reveal rows to the precision pool. A fixed
  ``JUDGE_VISIBLE_BANNER`` strips across the page, the manifest and coverage
  sidecar record ``judge_visible: true`` (blind builds record false), and the
  batch name is force-branded with ``JUDGE_VISIBLE_BATCH_SUFFIX`` so every
  exported CSV row carries the mark. The default build stays blind by
  construction: the renderer is handed no judge material at all.

- **adjudicate** — the owner-facing packet, with two explicit cohort sources:

  * ``--sidecar`` — the cohort is the annotate build's, pinned by its
    coverage sidecar. With returned rater CSVs it shows, per call and per
    factor, the stored judge verdict next to every rater's label, with
    disagreements highlighted; rater CSVs are ingested through the one
    sanctioned browser-CSV path (``ingest_browser_csv``). WITHOUT any rater
    CSVs it renders the same page as a judge-only review view ("show the LLM
    annotations" for the calibration draw): judge verdict/evidence/quote
    columns populated, no rater columns, no disagreement highlighting, and a
    fixed "Judge-only view" banner.
  * ``--results`` — an INDEPENDENT per-factor draw straight off the stored
    results (``build_calibration_adjudication_draw``): every judge factor's
    defect stratum takes ``min(--per-factor-cap, #flagged-calls)`` flagged
    calls — capped, never padded — with no clean controls. The draw persists
    its own frame + coverage sidecar (per-factor counts recorded on both)
    and then renders the SAME judge-only review page through the sidecar
    path, so downstream consumption is identical.

  In BOTH forms every judge-POSITIVE cell also becomes a decision candidate:
  the owner confirms or rejects it directly on the page (the
  ``nativeness-adjudication-packets`` interaction pattern — Confirmed/Rejected
  buttons, per-card completion blocker, localStorage persistence, CSV export).
  Call-level positives (quality factors, semantic dimensions, call-level
  nativeness factors) render their decision beneath the call's verdict table
  and transcript; utterance-level positives (nativeness utterance factors,
  delivery per-utterance findings, pack delivery factors) pin the EXACT
  flagged agent turn — the judge's pinned quote highlighted inside the turn
  text, with the immediately preceding caller turn as context. Non-positive
  cells (PASS / n-a / deferred) stay read-only table rows. The exported
  decisions CSV is the ``CalibrationDecisionRow`` contract
  (``NativenessHumanLabelRow``'s candidate_id + adjudication_decision contract
  extended minimally with the judge family), which ``calibration-agreement``
  ingests as owner adjudications.

Rater returns reuse the existing CSV contracts (``RubricAnnotationRow`` +
``NativenessHumanLabelRow``); the decisions CSV adds the one
``packet_judge_calibration_decisions`` dispatch entry.
"""

import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape
from loguru import logger
from markupsafe import escape
from pydantic import BaseModel, Field, field_validator

from tau2.annotation.artifacts import (
    ArtifactKind,
    ArtifactManifest,
    CalibrationDecisionEntry,
    CalibrationPacketEntry,
    NativenessAgentTurn,
    derive_batch_id,
    git_sha,
    write_json_artifact,
)
from tau2.annotation.calibration_draw import (
    CALIBRATION_CELLS_VERSION,
    CALIBRATION_SAMPLER_VERSION,
    CLEAN_CONTROL_STRATUM,
    DELIVERY_AXES,
    FOLDED_FIDELITY_CELLS_VERSION,
    JUDGE_VISIBLE_BATCH_SUFFIX,
    AdjudicationDrawConfig,
    CalibrationCoverageRow,
    CalibrationCoverageSidecar,
    CalibrationDrawConfig,
    CalibrationJudge,
    FrameCall,
    JudgeVerdictCell,
    build_adjudication_frame,
    build_calibration_frame,
    calibration_instrument_cells,
    load_calibration_sidecar,
    load_semantic_artifact,
    observed_judge_versions,
)
from tau2.annotation.calibration_report import (
    HUMAN_STATE_LABELS,
    JUDGE_STATE_LABELS,
    CohortEvidence,
    collect_human_labels,
    load_cohort_evidence,
    tri_state_disagreement,
)
from tau2.annotation.loading import LoadedSim, iter_loaded_sims
from tau2.annotation.models import CalibrationDecisionRow
from tau2.annotation.packets.audio_quality import (
    agent_speech_transcript,
    concatenate_agent_speech,
)
from tau2.annotation.packets.builder import (
    DEFAULT_OUTPUT_ROOT,
    PACKET_MANIFEST_NAME,
    find_audio,
    sync_packet_zip,
)
from tau2.annotation.packets.rubric import (
    JudgeAnnotationVM,
    RubricCallVM,
    RubricSectionVM,
    _copy_audio,
    _duration_label,
    calibration_sections_for,
    render_rubric_page,
    rubric_instrument_sha256,
)
from tau2.annotation.packets.transcript import (
    generate_message_rows,
    generate_tick_rows,
)
from tau2.config import DEFAULT_CALIBRATION_SEED
from tau2.data_model.simulation import JudgeOutcome, SimulationRun
from tau2.judges.nativeness.factors import judge_factors_for
from tau2.judges.nativeness.harness import _delivered_text, build_agent_turns
from tau2.multilingual.factory.entity_localization import caller_gender_for_task
from tau2.voice.utils.audio_io import save_wav_file
from tau2.voice.voice_gender import resolve_agent_gender

CALIBRATION_PACKET_VERSION = "judge-calibration-packet-v11"

#: Default per-factor cap on PRESENTED decision rows. The call draw bounds
#: which calls enter the cohort, but every judge positive on a drawn call is
#: a potential decision row — utterance-level factors mint one row per
#: finding, so an uncapped page runs to ~1,000 decisions/language. Precision
#: is a per-factor proportion: ~20 sampled rows give a ±0.19 CI, all the
#: calibration gates need. Rows beyond the cap are dropped from the page by
#: a seeded draw (fidelity stratified by finding category); the per-factor
#: total/presented accounting lands in the manifest provenance.
DEFAULT_DECISION_ROW_CAP = 20

#: Fixed banner for an adjudication page built with no rater returns; an
#: explicit ``watermark`` option takes precedence over it.
JUDGE_ONLY_BANNER = "Judge-only view — no rater returns"

#: Fixed banner of a --show-judge annotate build. Never configurable: the one
#: reason the page exists blind is the calibration wave, and an enriched page
#: must always say what it is instead — the two-phase blind-then-reveal pass.
JUDGE_VISIBLE_BANNER = (
    "TWO-PHASE PASS — answer blind, lock, then judge findings reveal; "
    "pre-reveal answers score recall, post-reveal revisions score precision"
)
JUDGE_CALIBRATION_KIND: ArtifactKind = "packet_judge_calibration"
JUDGE_CALIBRATION_ADJUDICATION_KIND: ArtifactKind = (
    "packet_judge_calibration_adjudication"
)


class CalibrationPacketOptions(BaseModel):
    """Everything one annotate-mode calibration packet build needs."""

    results: Annotated[
        list[Path],
        Field(min_length=1, description="Results dirs / results.json files."),
    ]
    language: Annotated[str, Field(description="ISO 639-1 language code.")]
    conversation_artifact: Annotated[
        Optional[Path],
        Field(
            description="Stored conversation-judge artifact. The instrument "
            "no longer asks semantic questions, but on corpora whose "
            "LLM-quality verdicts live in the artifact (``ours``) rather "
            "than on the sim it is the only source of the quality cells — "
            "without it those strata and questions render judge-empty."
        ),
    ] = None
    n_calls: Annotated[
        Optional[int],
        Field(gt=0, description="Cohort size override (default from config)."),
    ] = None
    control_fraction: Annotated[
        Optional[float], Field(description="Control-share override.")
    ] = None
    rule: Annotated[
        Literal["enriched", "random"],
        Field(
            description="Cohort selection rule: 'enriched' (default) is the "
            "defect-enriched draw; 'random' is the judge-blind uniform draw "
            "for the recall arm (control_fraction is ignored — clean calls "
            "enter at their natural rate)."
        ),
    ] = "enriched"
    seed: Annotated[int, Field(description="Draw seed.")] = DEFAULT_CALIBRATION_SEED
    batch_name: Annotated[Optional[str], Field(description="Packet label.")] = None
    out_dir: Annotated[Optional[Path], Field(description="Packet directory.")] = None
    emit_zip: Annotated[bool, Field(description="Emit the shipping ZIP.")] = True
    show_judge: Annotated[
        bool,
        Field(
            description="Render the stored judge annotations inline beneath "
            "each calibrating question (enriched internal review ONLY, never "
            "the calibration wave). The DEFAULT stays blind; a show-judge "
            "build is banner-marked, brands its batch name with "
            "JUDGE_VISIBLE_BATCH_SUFFIX, and records judge_visible in its "
            "manifest and coverage sidecar."
        ),
    ] = False

    @field_validator("language")
    @classmethod
    def _normalize_language(cls, value: str) -> str:
        return value.strip().lower()

    @property
    def effective_batch_name(self) -> str:
        name = self.batch_name or f"judge_calibration_{self.language}"
        if self.show_judge and not name.endswith(JUDGE_VISIBLE_BATCH_SUFFIX):
            return f"{name}{JUDGE_VISIBLE_BATCH_SUFFIX}"
        return name

    @property
    def packet_dir(self) -> Path:
        return self.out_dir or (DEFAULT_OUTPUT_ROOT / self.effective_batch_name)

    @property
    def frame_path(self) -> Path:
        return self.packet_dir.parent / f"{self.packet_dir.name}_frame.json"

    @property
    def sidecar_path(self) -> Path:
        return self.packet_dir.parent / f"{self.packet_dir.name}_coverage.json"

    def draw_config(self) -> CalibrationDrawConfig:
        overrides: dict = {
            "language": self.language,
            "seed": self.seed,
            "rule": self.rule,
        }
        if self.n_calls is not None:
            overrides["n_calls"] = self.n_calls
        if self.control_fraction is not None:
            overrides["control_fraction"] = self.control_fraction
        return CalibrationDrawConfig(**overrides)


class CalibrationPacketBuild(BaseModel):
    """The three annotate-mode artifacts, by path."""

    manifest_path: Path
    frame_path: Path
    sidecar_path: Path


def _selection_hash(
    cohort: list[FrameCall],
    sections: list[RubricSectionVM],
    language: str,
    *,
    judge_visible: bool,
) -> str:
    payload = {
        "packet_version": CALIBRATION_PACKET_VERSION,
        "language": language,
        "judge_visible": judge_visible,
        "cohort": [
            {"results_path": call.results_path, "sim_id": call.sim_id}
            for call in cohort
        ],
        "sections": [section.model_dump() for section in sections],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _presentation_order(cohort: list[FrameCall], seed: int) -> list[FrameCall]:
    """Deterministic arm-hiding shuffle of the drawn calls."""
    import random

    ordered = sorted(
        cohort, key=lambda c: (c.results_path, c.task_id, c.trial, c.sim_id)
    )
    random.Random(f"tau2-calib-order|{seed}").shuffle(ordered)
    return ordered


def _transcript_rows(sim) -> str:
    return generate_tick_rows(sim) if sim.ticks else generate_message_rows(sim)


def _render_call_audio(
    item: LoadedSim, packet_dir: Path, clip_id: str
) -> tuple[str, str, str, str]:
    """Render one call's packet audio pair under ``calls/``: the raw stereo
    call and the agent-only concatenation. Returns ``(full_relative,
    agent_relative, full_duration, agent_duration)``. Writes are atomic, so
    an interrupted build never leaves a truncated WAV behind."""
    full_source = find_audio(item.results_dir, item.sim)
    if full_source is None:
        raise FileNotFoundError(
            f"sim {item.sim.id}: no both.wav under {item.results_dir}"
        )
    (packet_dir / "calls").mkdir(parents=True, exist_ok=True)
    full_relative = f"calls/{clip_id}_full.wav"
    agent_relative = f"calls/{clip_id}_agent.wav"
    _copy_audio(full_source, packet_dir / full_relative)
    agent_output = packet_dir / agent_relative
    staging = agent_output.with_name(f".{agent_output.stem}.tmp.wav")
    staging.unlink(missing_ok=True)
    save_wav_file(concatenate_agent_speech(item.sim, item.results_dir), staging)
    staging.replace(agent_output)
    return (
        full_relative,
        agent_relative,
        _duration_label(packet_dir / full_relative),
        _duration_label(agent_output),
    )


def _agent_turns(sim) -> list[NativenessAgentTurn]:
    return [
        NativenessAgentTurn(
            index=turn.index,
            turn_id=f"agent-turn-{turn.index:03d}",
            text=turn.text,
            preceding_customer_text=turn.preceding_user_text,
            interrupted=turn.interrupted,
        )
        for turn in build_agent_turns(sim)
    ]


def _question_sources(
    section: RubricSectionVM, question_id: str, answer_kind: str
) -> list[tuple[CalibrationJudge, str]]:
    """Which judge cells one instrument question calibrates.

    The inverse of the agreement report's fixed label mapping: interaction
    binaries → quality factors, nativeness questions → nativeness factors,
    and the audio section → the matching delivery cell (``fidelity``,
    ``intonation``, or a pack factor). The overall 1-4 score questions calibrate no single judge cell.
    """
    if answer_kind == "likert4":
        return []
    if section.id == "interaction":
        return [(CalibrationJudge.QUALITY, question_id)]
    if section.id == "nativeness":
        return [(CalibrationJudge.NATIVENESS, question_id)]
    if section.id == "audio":
        return [(CalibrationJudge.DELIVERY, question_id)]
    raise ValueError(f"unknown calibration section {section.id!r}")


def _question_annotations(
    sim: SimulationRun,
    language: str,
    cells: list[JudgeVerdictCell],
    sections: list[RubricSectionVM],
) -> dict[str, list[JudgeAnnotationVM]]:
    """'section:question_id' → inline judge annotations for one call
    (--show-judge builds only).

    Precision pass: only the judge's POSITIVES are presented for verification
    (passes/N-A verdicts would only anchor the rater), one card per resolved
    finding — utterance-level positives pin the exact flagged agent turn with
    the judge's quote highlighted, same as the adjudication candidates. The
    single fidelity question's findings each carry their dimension category
    in the evidence text, so the rater sees which kind of defect the judge
    heard."""
    by_key = {(cell.judge, cell.factor_id): cell for cell in cells}
    findings = _positive_findings(sim, language, by_key)
    annotations: dict[str, list[JudgeAnnotationVM]] = {}
    for section in sections:
        for question in section.questions:
            vms: list[JudgeAnnotationVM] = []
            for source in _question_sources(section, question.id, question.answer_kind):
                source_findings = findings.get(source, [])
                vms.extend(
                    JudgeAnnotationVM(
                        judge=source[0].value,
                        factor_id=source[1],
                        verdict=JUDGE_STATE_LABELS[JudgeOutcome.FAIL],
                        flagged=True,
                        evidence=finding.evidence,
                        quote=finding.quote,
                        turn_index=(
                            finding.turn.index if finding.turn is not None else None
                        ),
                        turn_html=(
                            _highlight_quote(finding.turn.text, finding.quote)
                            if finding.turn is not None
                            else ""
                        ),
                        context_text=(
                            finding.turn.preceding_customer_text or ""
                            if finding.turn is not None
                            else ""
                        ),
                    )
                    for finding in source_findings
                )
            if vms:
                annotations[f"{section.id}:{question.id}"] = vms
    return annotations


def build_calibration_packet(
    options: CalibrationPacketOptions,
) -> CalibrationPacketBuild:
    """Draw the defect-enriched cohort and build the annotate packet.

    Blind by default; ``options.show_judge`` builds the banner-marked
    judge-visible variant (see the module docstring)."""
    config = options.draw_config()
    semantic_by_key = {}
    if options.conversation_artifact is not None:
        semantic_by_key = load_semantic_artifact(options.conversation_artifact).by_key()
    frame = build_calibration_frame(
        iter_loaded_sims(options.results),
        config,
        results=options.results,
        conversation_artifact=options.conversation_artifact,
        semantic_by_key=semantic_by_key,
    )
    cohort = _presentation_order(frame.drawn_calls(), config.seed)
    if not cohort:
        raise ValueError(
            "the draw selected no calls — the inputs hold no judged calls "
            f"for language '{config.language}' (see the frame drop counters)"
        )
    sections = calibration_sections_for(config.language)
    selection_sha = _selection_hash(
        cohort, sections, config.language, judge_visible=options.show_judge
    )
    batch_name = options.effective_batch_name
    batch_id = derive_batch_id(
        JUDGE_CALIBRATION_KIND, batch_name, {"selection": selection_sha}
    )

    packet_dir = options.packet_dir
    manifest_path = packet_dir / PACKET_MANIFEST_NAME
    instrument_sha = rubric_instrument_sha256()
    if manifest_path.is_file():
        prior = ArtifactManifest.model_validate_json(manifest_path.read_text())
        same_draw = (
            prior.kind == JUDGE_CALIBRATION_KIND
            and prior.batch_id == batch_id
            and prior.provenance.get("selection_sha256") == selection_sha
        )
        if (
            same_draw
            and prior.provenance.get("instrument_sha256") == instrument_sha
            and (packet_dir / "index.html").is_file()
        ):
            logger.info(f"calibration packet unchanged -> {packet_dir}")
            sync_packet_zip(packet_dir, emit=options.emit_zip)
            return CalibrationPacketBuild(
                manifest_path=manifest_path,
                frame_path=options.frame_path,
                sidecar_path=options.sidecar_path,
            )
        if same_draw:
            # Same draw, different rendered instrument (template/CSS/JS edit,
            # or a pre-fingerprint manifest). Never silently keep the stale
            # page and never silently clobber a possibly-distributed one.
            raise FileExistsError(
                "output packet holds the same draw but a DIFFERENT instrument "
                f"(template/CSS/JS changed): {packet_dir} — delete or rename "
                "the packet directory to re-render it"
            )
        raise FileExistsError(
            f"output packet already exists with different inputs: {packet_dir}"
        )

    # Pass 2: stream again, rendering only the drawn calls.
    wanted = {call.key: call for call in cohort}
    loaded: dict[tuple[str, str], LoadedSim] = {}
    for item in iter_loaded_sims(options.results):
        key = (str((item.results_dir / "results.json").resolve()), item.sim.id)
        if key in wanted and key not in loaded:
            loaded[key] = item
    missing = sorted(set(wanted) - set(loaded))
    if missing:
        raise ValueError(f"drawn calls missing on the second pass: {missing[:5]}")

    calls: list[RubricCallVM] = []
    entries: list[CalibrationPacketEntry] = []
    coverage_rows: list[CalibrationCoverageRow] = []
    annotations: Optional[dict[str, dict[str, list[JudgeAnnotationVM]]]] = (
        {} if options.show_judge else None
    )
    for ordinal, frame_call in enumerate(cohort, 1):
        clip_id = f"clip_{ordinal:03d}"
        item = loaded[frame_call.key]
        sim = item.sim
        full_relative, agent_relative, full_duration, agent_duration = (
            _render_call_audio(item, packet_dir, clip_id)
        )

        agent_gender = (
            resolve_agent_gender(sim.agent_provider, sim.agent_voice) or "unknown"
        )
        caller_gender = (
            caller_gender_for_task(
                frame_call.task_id, config.language, frame_call.domain
            )
            or "unknown"
        )
        turns = _agent_turns(sim)
        calls.append(
            RubricCallVM(
                clip_id=clip_id,
                ordinal=ordinal,
                full_audio_file=full_relative,
                agent_audio_file=agent_relative,
                full_duration=full_duration,
                agent_duration=agent_duration,
                transcript_rows=_transcript_rows(sim),
                agent_transcript=agent_speech_transcript(sim),
                sim_id=sim.id,
                task_id=str(sim.task_id),
                agent_turns=turns,
                agent_gender=agent_gender,
                caller_gender=caller_gender,
            )
        )
        entries.append(
            CalibrationPacketEntry(
                clip_id=clip_id,
                sim_id=sim.id,
                task_id=str(sim.task_id),
                trial=frame_call.trial,
                language=config.language,
                domain=frame_call.domain,
                full_audio_file=full_relative,
                agent_audio_file=agent_relative,
                agent_gender=agent_gender,
                caller_gender=caller_gender,
                agent_turns=turns,
            )
        )
        coverage_rows.append(
            CalibrationCoverageRow(
                clip_id=clip_id,
                sim_id=sim.id,
                task_id=str(sim.task_id),
                trial=frame_call.trial,
                results_path=frame_call.results_path,
                provider=frame_call.provider,
                experiment=frame_call.experiment,
                judges_present=frame_call.judges_present,
                strata=frame_call.strata,
                drawn_for=frame_call.drawn_for,
            )
        )
        if annotations is not None:
            annotations[clip_id] = _question_annotations(
                sim,
                config.language,
                calibration_instrument_cells(
                    sim, config.language, semantic_by_key.get(frame_call.key)
                ),
                sections,
            )
        logger.info(
            f"calibration {clip_id}: {frame_call.domain} task "
            f"{frame_call.task_id} / sim {sim.id}"
        )

    html = render_rubric_page(
        batch_id=batch_id,
        batch_name=batch_name,
        language=config.language,
        calls=calls,
        sections=sections,
        packet_version=CALIBRATION_PACKET_VERSION,
        judge_annotations=annotations,
        banner=JUDGE_VISIBLE_BANNER if options.show_judge else None,
    )
    (packet_dir / "index.html").write_text(html)

    questions = [question for section in sections for question in section.questions]
    manifest = ArtifactManifest(
        kind=JUDGE_CALIBRATION_KIND,
        batch_id=batch_id,
        batch_name=batch_name,
        created_at=datetime.now(timezone.utc).isoformat(),
        git_sha=git_sha(),
        language=config.language,
        form="judge_calibration",
        provenance={
            "selection_sha256": selection_sha,
            "instrument_sha256": instrument_sha,
            "packet_version": CALIBRATION_PACKET_VERSION,
            "sampler_version": CALIBRATION_SAMPLER_VERSION,
            "cells_version": CALIBRATION_CELLS_VERSION,
            "seed": config.seed,
            "question_count": len(questions),
            "sections": [section.model_dump() for section in sections],
            "frame_path": str(options.frame_path),
            "judge_visible": options.show_judge,
            "blinding_policy": (
                (
                    "JUDGE-VISIBLE two-phase build: per call, the rater "
                    "answers every question with the stored judge findings "
                    "hidden, locks those answers as the immutable pre-reveal "
                    "layer, and only then do the verdicts, evidence, and "
                    "pinned quotes render inline beneath their calibrating "
                    "questions (uniform closed toggles — the collapsed page "
                    "never says which questions were flagged). Post-reveal "
                    "revisions are a distinct layer; exports carry BOTH, and "
                    "calibration-agreement routes pre-reveal rows to the "
                    "blind (recall) pool and post-reveal rows to the "
                    "precision pool."
                )
                if options.show_judge
                else (
                    "Judge verdicts, arms, rewards, scores, results paths, and "
                    "sampling rationale are never rendered in this packet or "
                    "its manifest; the clip-to-source join lives only in the "
                    "builder's coverage sidecar, outside the packet directory."
                )
            ),
        },
        files=[],
        calibration_entries=entries,
    )
    manifest_path = write_json_artifact(packet_dir / PACKET_MANIFEST_NAME, manifest)
    frame_path = write_json_artifact(options.frame_path, frame)
    sidecar = CalibrationCoverageSidecar(
        created_at=datetime.now(timezone.utc).isoformat(),
        git_sha=git_sha(),
        batch_id=batch_id,
        batch_name=batch_name,
        language=config.language,
        frame_path=str(frame_path),
        conversation_artifact=(
            str(options.conversation_artifact)
            if options.conversation_artifact is not None
            else None
        ),
        cells_version=CALIBRATION_CELLS_VERSION,
        judge_visible=options.show_judge,
        judge_versions=observed_judge_versions(
            [loaded[call.key].sim for call in cohort]
        ),
        rows=coverage_rows,
    )
    sidecar_path = write_json_artifact(options.sidecar_path, sidecar)
    # Zip AFTER all packet files exist; the frame/sidecar live OUTSIDE the
    # packet dir, so the archive can never carry them.
    sync_packet_zip(packet_dir, emit=options.emit_zip)
    logger.info(
        f"judge-calibration packet '{batch_name}' ({batch_id}): "
        f"{len(entries)} calls x {len(questions)} questions -> {packet_dir}"
    )
    return CalibrationPacketBuild(
        manifest_path=manifest_path,
        frame_path=frame_path,
        sidecar_path=sidecar_path,
    )


# ---------------------------------------------------------------------------
# Adjudicate mode (owner-facing)
# ---------------------------------------------------------------------------


class CalibrationAdjudicationOptions(BaseModel):
    """Inputs for the owner-facing judge-vs-raters adjudication packet."""

    sidecar: Annotated[
        Path, Field(description="The annotate build's coverage sidecar.")
    ]
    filled: Annotated[
        list[Path],
        Field(
            default_factory=list,
            description="Returned rater CSVs (rubric-answer and/or nativeness "
            "label exports from the calibration packet). Empty = judge-only "
            "review view: judge verdicts rendered, no rater columns, no "
            "disagreement highlighting.",
        ),
    ]
    batch_name: Annotated[Optional[str], Field(description="Packet label.")] = None
    out_dir: Annotated[Optional[Path], Field(description="Packet directory.")] = None
    emit_zip: Annotated[bool, Field(description="Emit the shipping ZIP.")] = True
    watermark: Annotated[
        Optional[str],
        Field(
            description="Banner rendered across the page (e.g. to mark a "
            "sample built from synthetic labels)."
        ),
    ] = None
    decision_row_cap: Annotated[
        int,
        Field(
            ge=1,
            description="Per-factor cap on presented decision rows "
            "(``DEFAULT_DECISION_ROW_CAP``): factors with more judge-positive "
            "findings get a seeded sample of this many rows; the rest render "
            "in the verdict tables but carry no decision control.",
        ),
    ] = DEFAULT_DECISION_ROW_CAP

    @property
    def effective_batch_name(self) -> str:
        return self.batch_name or f"{Path(self.sidecar).stem}_adjudication"

    @property
    def packet_dir(self) -> Path:
        return self.out_dir or (DEFAULT_OUTPUT_ROOT / self.effective_batch_name)


class AdjudicationRowVM(BaseModel):
    """One factor's side-by-side comparison on one call."""

    judge: Annotated[str, Field(description="Judge family.")]
    factor_id: Annotated[str, Field(description="Factor / axis / dimension id.")]
    judge_state: Annotated[str, Field(description="Judge verdict display label.")]
    judge_evidence: Annotated[str, Field(description="Judge rationale.")] = ""
    judge_quote: Annotated[str, Field(description="Judge's pinned span.")] = ""
    rater_states: Annotated[
        dict[str, str],
        Field(description="Rater id -> label display (missing = 'unlabeled')."),
    ]
    disagree: Annotated[
        bool, Field(description="Any substantive judge/rater or rater/rater split.")
    ]


class CalibrationDecisionCandidateVM(BaseModel):
    """One judge-positive candidate card on the adjudication page."""

    entry: CalibrationDecisionEntry
    ordinal: Annotated[int, Field(description="One-based position on the page.")]
    agent_text_html: Annotated[
        str,
        Field(
            description="Escaped flagged-turn text with the judge's pinned "
            "quote wrapped in <mark>; empty for call-level candidates."
        ),
    ] = ""


class AdjudicationCallVM(BaseModel):
    """One cohort call and all its comparison rows."""

    clip_id: str
    ordinal: int
    sim_id: str
    task_id: str
    experiment: str
    provider: str
    strata: list[str]
    transcript_rows: str
    full_audio_file: Annotated[
        str, Field(description="Packet-relative raw stereo call WAV.")
    ]
    agent_audio_file: Annotated[
        str, Field(description="Packet-relative agent-only concatenation WAV.")
    ]
    full_duration: Annotated[str, Field(description="mm:ss of the full call.")]
    agent_duration: Annotated[str, Field(description="mm:ss of the agent track.")]
    rows: list[AdjudicationRowVM]
    candidates: list[CalibrationDecisionCandidateVM] = Field(default_factory=list)

    @property
    def n_disagreements(self) -> int:
        return sum(1 for row in self.rows if row.disagree)


def _delivery_turn_map(
    sim: SimulationRun, turns: list[NativenessAgentTurn]
) -> dict[int, NativenessAgentTurn]:
    """Map the delivery judge's ``utterance_idx`` onto the rendered agent turns.

    The delivery judge indexes the merged full-duplex agent message history
    (``ticks_to_message_history``); ``build_agent_turns`` applies the SAME
    utterance-id grouping and then drops merges with no delivered text — so
    walking the message history and counting only text-bearing messages
    reproduces the turn indices exactly. Judged utterances always carry text
    (the judge skips silent clips), so every judged index maps.
    """
    if not sim.ticks:
        return {}
    from tau2.evaluator.evaluator_communicate import FullDuplexCommunicateEvaluator

    mapping: dict[int, NativenessAgentTurn] = {}
    position = 0
    for idx, message in enumerate(
        FullDuplexCommunicateEvaluator.ticks_to_message_history(sim.ticks)
    ):
        delivered = _delivered_text(message)
        if not delivered:
            continue
        if position >= len(turns):
            raise ValueError(
                f"sim {sim.id}: delivered agent message {idx} has no rendered "
                "agent turn — transcript and delivery indices diverged"
            )
        mapping[idx] = turns[position]
        position += 1
    return mapping


def _highlight_quote(text: str, quote: str) -> str:
    """Escape the flagged turn text, wrapping the judge's pinned quote (first
    occurrence) in ``<mark>`` when it is verbatim-present."""
    escaped = str(escape(text))
    needle = str(escape(quote)) if quote else ""
    if needle and needle in escaped:
        return escaped.replace(needle, f"<mark>{needle}</mark>", 1)
    return escaped


def _decision_candidate_id(
    sidecar_batch_id: str,
    sim_id: str,
    judge: CalibrationJudge,
    factor_id: str,
    level: str,
    turn_index: Optional[int],
) -> str:
    raw = json.dumps(
        [sidecar_batch_id, sim_id, judge.value, factor_id, level, turn_index],
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class PositiveFinding(BaseModel):
    """One judge-POSITIVE cell resolved to its display evidence — the shared
    substrate of the adjudication decision candidates and the --show-judge
    inline annotation cards."""

    evidence: Annotated[str, Field(description="Judge rationale.")] = ""
    quote: Annotated[str, Field(description="Judge's pinned span.")] = ""
    level: Annotated[
        Literal["call", "utterance"],
        Field(description="Granularity the positive was stored at."),
    ] = "call"
    turn: Annotated[
        Optional[NativenessAgentTurn],
        Field(description="The flagged agent turn (utterance-level only)."),
    ] = None


def _delivery_axis_findings(
    sim: SimulationRun,
    delivery_turns: dict[int, NativenessAgentTurn],
    axes: tuple[str, ...],
) -> list[PositiveFinding]:
    """One finding per flagged utterance on the given delivery axes,
    aggregating that utterance's findings. Each finding's evidence is
    prefixed with its dimension category, so a folded cell still says which
    kind of defect the judge heard."""
    pinned: list[PositiveFinding] = []
    for result in sim.delivery_info.utterance_results if sim.delivery_info else []:
        axis_findings = [finding for finding in result.findings if finding.axis in axes]
        turn = delivery_turns.get(result.utterance_idx)
        if not axis_findings or turn is None:
            continue
        pinned.append(
            PositiveFinding(
                evidence="; ".join(
                    f"{finding.category}: {finding.issue or ''}".strip(": ")
                    for finding in axis_findings
                ),
                quote="",
                level="utterance",
                turn=turn,
            )
        )
    return pinned


def _positive_findings(
    sim: SimulationRun,
    language: str,
    cells: dict[tuple[CalibrationJudge, str], JudgeVerdictCell],
    *,
    folded_fidelity: bool = False,
) -> dict[tuple[CalibrationJudge, str], list[PositiveFinding]]:
    """Every judge-POSITIVE cell of one cohort call, resolved to findings.

    Utterance-level positives pin the exact flagged agent turn:

    - nativeness utterance factors — one finding per stored violated unit
      (``unit_results``), carrying that unit's reasoning and quote;
    - delivery axes — one finding per flagged utterance, aggregating that
      utterance's findings on the axis; with ``folded_fidelity`` (the
      current instrument's cell view) the ``fidelity`` cell covers ALL
      generic delivery axes, each finding labeled with its dimension
      category;
    - pack delivery factors — one finding per per-utterance factor FAIL.

    Call-level positives (quality factors, semantic dimensions, call-level
    nativeness factors) yield one finding each. A stored positive whose
    per-utterance detail is missing falls back to one call-level finding —
    every positive is shown, nothing is invented.
    """
    turns = _agent_turns(sim)
    delivery_turns = _delivery_turn_map(sim, turns)
    nativeness_levels = {
        factor.id: factor.evaluation_level
        for factor in judge_factors_for(language)
        if factor.enabled
    }
    nativeness_checks = {
        check.id: check
        for check in (sim.nativeness_info.factor_checks if sim.nativeness_info else [])
    }

    findings: dict[tuple[CalibrationJudge, str], list[PositiveFinding]] = {}
    for (judge, factor_id), cell in sorted(
        cells.items(), key=lambda kv: (kv[0][0].value, kv[0][1])
    ):
        if cell.outcome is not JudgeOutcome.FAIL:
            continue
        pinned: list[PositiveFinding] = []
        if (
            judge is CalibrationJudge.NATIVENESS
            and nativeness_levels.get(factor_id) == "utterance"
        ):
            check = nativeness_checks.get(factor_id)
            for unit in check.unit_results if check is not None else []:
                if not unit.opportunity or not unit.violated:
                    continue
                index = unit.unit_index
                if index is None or not 0 <= index < len(turns):
                    raise ValueError(
                        f"stored nativeness positive {sim.id}/{factor_id} "
                        f"points to missing agent turn {index}"
                    )
                pinned.append(
                    PositiveFinding(
                        evidence=unit.reasoning,
                        quote=unit.quote,
                        level="utterance",
                        turn=turns[index],
                    )
                )
        elif judge is CalibrationJudge.DELIVERY and factor_id in DELIVERY_AXES:
            axes = (
                DELIVERY_AXES
                if folded_fidelity and factor_id == "fidelity"
                else (factor_id,)
            )
            pinned = _delivery_axis_findings(sim, delivery_turns, axes)
        elif judge is CalibrationJudge.DELIVERY:
            for result in (
                sim.delivery_info.utterance_results if sim.delivery_info else []
            ):
                turn = delivery_turns.get(result.utterance_idx)
                if turn is None:
                    continue
                for check in result.factor_checks:
                    if check.id != factor_id or check.outcome is not JudgeOutcome.FAIL:
                        continue
                    pinned.append(
                        PositiveFinding(
                            evidence=check.evidence or "",
                            quote=check.quote or "",
                            level="utterance",
                            turn=turn,
                        )
                    )
        findings[(judge, factor_id)] = pinned or [
            PositiveFinding(evidence=cell.evidence, quote=cell.quote)
        ]
    return findings


def _decision_candidates(
    sidecar: CalibrationCoverageSidecar,
    row: CalibrationCoverageRow,
    sim: SimulationRun,
    cells: dict[tuple[CalibrationJudge, str], JudgeVerdictCell],
) -> list[CalibrationDecisionEntry]:
    """Every judge-POSITIVE cell of one cohort call, as decision candidates —
    one per resolved ``_positive_findings`` finding."""
    candidates: list[CalibrationDecisionEntry] = []
    for (judge, factor_id), findings in _positive_findings(
        sim,
        sidecar.language,
        cells,
        folded_fidelity=sidecar.cells_version == FOLDED_FIDELITY_CELLS_VERSION,
    ).items():
        for finding in findings:
            candidates.append(
                CalibrationDecisionEntry(
                    candidate_id=_decision_candidate_id(
                        sidecar.batch_id,
                        row.sim_id,
                        judge,
                        factor_id,
                        finding.level,
                        finding.turn.index if finding.turn is not None else None,
                    ),
                    clip_id=row.clip_id,
                    sim_id=row.sim_id,
                    task_id=row.task_id,
                    language=sidecar.language,
                    judge=judge.value,
                    factor_id=factor_id,
                    evaluation_level=finding.level,
                    judge_verdict=JudgeOutcome.FAIL.value,
                    judge_evidence=finding.evidence,
                    judge_quote=finding.quote,
                    agent_turn=finding.turn,
                )
            )
    return list(
        {candidate.candidate_id: candidate for candidate in candidates}.values()
    )


def _template_environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(Path(__file__).parent / "templates"),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _sample_decision_rows(
    candidates: list[CalibrationDecisionEntry],
    cap: int,
    salt: str,
) -> tuple[set[str], dict[str, dict[str, int]]]:
    """Seeded per-factor cap on PRESENTED decision rows.

    Returns the kept candidate ids plus per-factor accounting
    ``{"judge:factor": {"total": n, "presented": k}}``. Factors at or under
    the cap keep every row. Over-cap factors get a deterministic draw seeded
    by ``salt`` + the factor key: the folded delivery ``fidelity`` cell is
    stratified by finding category (round-robin across the categories named
    in each finding's evidence prefix, shuffled within category) so the
    per-category precision slices stay populated; every other factor draws
    uniformly. Presentation order elsewhere stays cohort order — this only
    decides membership.
    """
    by_factor: dict[str, list[CalibrationDecisionEntry]] = {}
    for entry in candidates:
        by_factor.setdefault(f"{entry.judge}:{entry.factor_id}", []).append(entry)
    kept: set[str] = set()
    accounting: dict[str, dict[str, int]] = {}
    for key in sorted(by_factor):
        entries = by_factor[key]
        accounting[key] = {"total": len(entries), "presented": min(cap, len(entries))}
        if len(entries) <= cap:
            kept.update(entry.candidate_id for entry in entries)
            continue
        rng = random.Random(f"{salt}:{key}:decision-rows-v1")
        if key == f"{CalibrationJudge.DELIVERY.value}:fidelity":
            groups: dict[str, list[CalibrationDecisionEntry]] = {}
            for entry in entries:
                category = (entry.judge_evidence or "").split(":", 1)[0].strip()
                groups.setdefault(category or "other", []).append(entry)
            for group in groups.values():
                rng.shuffle(group)
            picked: list[CalibrationDecisionEntry] = []
            while len(picked) < cap:
                for category in sorted(groups):
                    if groups[category] and len(picked) < cap:
                        picked.append(groups[category].pop())
            kept.update(entry.candidate_id for entry in picked)
        else:
            kept.update(entry.candidate_id for entry in rng.sample(entries, cap))
    return kept, accounting


#: Section id → the judge family its questions calibrate, for the
#: reclassification dropdown's closed catalog.
_SECTION_JUDGE = {
    "interaction": CalibrationJudge.QUALITY,
    "nativeness": CalibrationJudge.NATIVENESS,
    "audio": CalibrationJudge.DELIVERY,
}


def _reclassify_options(language: str) -> list[dict[str, str]]:
    """The closed catalog for the per-row "different issue" dropdown: every
    binary/severity factor in the language's calibration instrument, as
    ``judge:factor_id`` values. Rejecting a row while naming another factor
    records that the finding is REAL but mis-filed — the cross-factor
    attribution the pt forensics found dominating apparent recall loss."""
    options: list[dict[str, str]] = []
    for section in calibration_sections_for(language):
        judge = _SECTION_JUDGE[section.id]
        for question in section.questions:
            if question.answer_kind == "likert4":
                continue
            options.append(
                {
                    "value": f"{judge.value}:{question.id}",
                    "label": f"{judge.value}: {question.id}",
                }
            )
    return options


def build_calibration_adjudication(
    options: CalibrationAdjudicationOptions,
) -> Path:
    """Build the adjudication page; returns the manifest path.

    With ``filled`` CSVs: judge-vs-raters, disagreements highlighted. Without:
    a judge-only review view of the same cohort — no rater columns, no
    disagreement highlighting (there is nothing to disagree with), and the
    fixed ``JUDGE_ONLY_BANNER`` unless an explicit watermark overrides it.
    In BOTH forms every judge-positive cell carries a Confirmed/Rejected
    decision control, exported as the ``CalibrationDecisionRow`` CSV.
    """
    sidecar = load_calibration_sidecar(options.sidecar)
    evidence: CohortEvidence = load_cohort_evidence(sidecar)
    if options.filled:
        collected = collect_human_labels(options.filled)
        labels, raters = collected.labels, collected.raters
        # The judge-vs-raters view compares the judge against BLIND returns
        # only; precision-pass (judge-visible) returns are scored by
        # calibration-agreement instead. Never drop them silently.
        if collected.verifiers and not raters:
            raise ValueError(
                "every supplied CSV is a judge-visible (precision-pass) "
                "return — adjudication compares the judge against BLIND "
                "returns; score these with `tau2 annotate "
                "calibration-agreement`"
            )
        if collected.verifiers:
            logger.warning(
                f"excluding {len(collected.verifiers)} judge-visible "
                f"(precision-pass) return(s) from the judge-vs-raters view "
                f"({', '.join(sorted(collected.verifiers))}); they are scored "
                "by calibration-agreement"
            )
    else:
        labels, raters = {}, []
    banner = options.watermark or (JUDGE_ONLY_BANNER if not options.filled else None)

    # Pass 1: every judge positive on every cohort call is a POTENTIAL
    # decision row. Pass 2 (below) presents only the per-factor capped
    # sample — utterance-level factors mint one row per finding, so the
    # uncapped page runs to ~1,000 decisions per language.
    per_call_candidates: dict[str, list[CalibrationDecisionEntry]] = {}
    for row in sidecar.rows:
        sim = evidence.sims[row.clip_id].sim
        cells = evidence.judge_cells.get(row.clip_id, {})
        per_call_candidates[row.clip_id] = list(
            _decision_candidates(sidecar, row, sim, cells)
        )
    kept_ids, row_accounting = _sample_decision_rows(
        [c for row in sidecar.rows for c in per_call_candidates[row.clip_id]],
        options.decision_row_cap,
        salt=sidecar.batch_id,
    )

    packet_dir = options.packet_dir
    calls: list[AdjudicationCallVM] = []
    all_candidates: list[CalibrationDecisionEntry] = []
    dropped_clips: list[str] = []
    for row in sidecar.rows:
        cells = evidence.judge_cells.get(row.clip_id, {})
        vm_rows: list[AdjudicationRowVM] = []
        for (judge, factor_id), cell in sorted(
            cells.items(), key=lambda kv: (kv[0][0].value, kv[0][1])
        ):
            # Judge-only form is a precision view: positives only. With
            # rater returns, PASS rows stay — judge-pass vs rater-violation
            # disagreement is the recall signal the comparison exists for.
            if not raters and cell.outcome is not JudgeOutcome.FAIL:
                continue
            rater_states = {}
            human_states = []
            for rater in raters:
                label = labels.get((judge, factor_id, row.clip_id, rater))
                rater_states[rater] = HUMAN_STATE_LABELS.get(label, "unlabeled")
                human_states.append(label)
            vm_rows.append(
                AdjudicationRowVM(
                    judge=judge.value,
                    factor_id=factor_id,
                    judge_state=JUDGE_STATE_LABELS[cell.outcome],
                    judge_evidence=cell.evidence,
                    judge_quote=cell.quote,
                    rater_states=rater_states,
                    disagree=tri_state_disagreement(cell.outcome, human_states),
                )
            )
        kept_candidates = [
            candidate
            for candidate in per_call_candidates[row.clip_id]
            if candidate.candidate_id in kept_ids
        ]
        # A drawn call whose candidates ALL capped out (and which, with rater
        # returns, carries no disagreement either) would render as an empty
        # page with nothing to rate — drop it from the packet and keep its
        # audio out of the bundle.
        if not kept_candidates and not any(vm.disagree for vm in vm_rows):
            dropped_clips.append(row.clip_id)
            for relative in (
                f"calls/{row.clip_id}_full.wav",
                f"calls/{row.clip_id}_agent.wav",
            ):
                (packet_dir / relative).unlink(missing_ok=True)
            continue
        item = evidence.sims[row.clip_id]
        sim = item.sim
        # Audio-judge (fidelity / accent) candidates cannot be adjudicated
        # from the transcript — ship the same audio pair the annotate packet
        # carries so the decision is made against what was actually said.
        full_audio, agent_audio, full_duration, agent_duration = _render_call_audio(
            item, packet_dir, row.clip_id
        )
        candidate_vms = []
        for candidate in kept_candidates:
            all_candidates.append(candidate)
            candidate_vms.append(
                CalibrationDecisionCandidateVM(
                    entry=candidate,
                    ordinal=len(all_candidates),
                    agent_text_html=(
                        _highlight_quote(
                            candidate.agent_turn.text, candidate.judge_quote
                        )
                        if candidate.agent_turn is not None
                        else ""
                    ),
                )
            )
        calls.append(
            AdjudicationCallVM(
                clip_id=row.clip_id,
                ordinal=len(calls) + 1,
                sim_id=row.sim_id,
                task_id=row.task_id,
                experiment=row.experiment,
                provider=row.provider or "",
                strata=row.strata,
                transcript_rows=_transcript_rows(sim),
                full_audio_file=full_audio,
                agent_audio_file=agent_audio,
                full_duration=full_duration,
                agent_duration=agent_duration,
                rows=vm_rows,
                candidates=candidate_vms,
            )
        )

    if dropped_clips:
        logger.warning(
            f"dropping {len(dropped_clips)} cohort call(s) with zero rendered "
            f"decision candidates (no judge positives, or every one capped "
            f"out by decision_row_cap={options.decision_row_cap}): "
            f"{', '.join(dropped_clips)}"
        )

    selection = hashlib.sha256(
        json.dumps(
            {
                "sidecar_batch_id": sidecar.batch_id,
                "filled": sorted(
                    hashlib.sha256(Path(path).read_bytes()).hexdigest()
                    for path in options.filled
                ),
                "packet_version": CALIBRATION_PACKET_VERSION,
                "decision_row_cap": options.decision_row_cap,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    batch_name = options.effective_batch_name
    batch_id = derive_batch_id(
        JUDGE_CALIBRATION_ADJUDICATION_KIND, batch_name, {"selection": selection}
    )
    packet_dir.mkdir(parents=True, exist_ok=True)
    template_root = Path(__file__).parent / "templates"
    config = {
        "batch_id": batch_id,
        "batch_name": batch_name,
        "packet_version": CALIBRATION_PACKET_VERSION,
        "language": sidecar.language,
        "sidecar_batch_id": sidecar.batch_id,
        "csv_headers": CalibrationDecisionRow.headers(),
        "candidates": [candidate.model_dump() for candidate in all_candidates],
    }
    reclassify_options = _reclassify_options(sidecar.language)
    html = (
        _template_environment()
        .get_template("calibration_adjudication.html.j2")
        .render(
            packet_config=json.dumps(config, ensure_ascii=False).replace("</", "<\\/"),
            batch_name=batch_name,
            language=sidecar.language,
            raters=raters,
            calls=calls,
            watermark=banner,
            judge_only=not options.filled,
            n_disagreements=sum(call.n_disagreements for call in calls),
            n_candidates=len(all_candidates),
            reclassify_options=reclassify_options,
            css=(template_root / "static" / "rubric.css").read_text()
            + "\n"
            + (template_root / "static" / "nativeness_adjudication.css").read_text()
            + "\n"
            + (template_root / "static" / "calibration_adjudication.css").read_text(),
            js=(template_root / "static" / "calibration_adjudication.js").read_text(),
        )
    )
    (packet_dir / "index.html").write_text(html)
    manifest = ArtifactManifest(
        kind=JUDGE_CALIBRATION_ADJUDICATION_KIND,
        batch_id=batch_id,
        batch_name=batch_name,
        created_at=datetime.now(timezone.utc).isoformat(),
        git_sha=git_sha(),
        language=sidecar.language,
        form="judge_calibration_adjudication",
        provenance={
            "selection_sha256": selection,
            "packet_version": CALIBRATION_PACKET_VERSION,
            "sidecar_path": str(options.sidecar),
            "sidecar_batch_id": sidecar.batch_id,
            "filled": [str(path) for path in options.filled],
            "raters": raters,
            "judge_only": not options.filled,
            "judge_versions": sidecar.judge_versions,
            "watermark": banner,
            "n_decision_candidates": len(all_candidates),
            "decision_row_cap": options.decision_row_cap,
            "dropped_empty_clips": dropped_clips,
            "audio": (
                "Per-call audio pair under calls/: {clip}_full.wav (raw "
                "stereo call) and {clip}_agent.wav (agent-only "
                "concatenation). Listening is required for audio-judge "
                "(fidelity / accent) decisions; transcript times seek the "
                "full-call track."
            ),
            "decision_rows_per_factor": row_accounting,
            "decision_policy": (
                "Judge positives render Confirmed/Rejected decision controls "
                "(utterance positives pin the exact flagged turn), capped per "
                "factor by a seeded draw (decision_row_cap; fidelity "
                "stratified by finding category — decision_rows_per_factor "
                "records total vs presented). A rejected row may name the "
                "factor the finding ACTUALLY belongs to via the "
                "reclassified_factor column. Decisions export as the "
                "CalibrationDecisionRow CSV and feed calibration-agreement "
                "as owner adjudications."
            ),
        },
        files=[],
        calibration_decision_entries=all_candidates,
    )
    manifest_path = write_json_artifact(packet_dir / PACKET_MANIFEST_NAME, manifest)
    sync_packet_zip(packet_dir, emit=options.emit_zip)
    logger.info(
        f"calibration adjudication '{batch_name}' ({batch_id}): "
        f"{len(calls)} calls, {sum(c.n_disagreements for c in calls)} "
        f"disagreement(s), {len(all_candidates)} decision candidate(s) "
        f"-> {packet_dir}"
    )
    return manifest_path


# ---------------------------------------------------------------------------
# Adjudicate mode, per-factor draw (independent of any annotate build)
# ---------------------------------------------------------------------------


class CalibrationAdjudicationDrawOptions(BaseModel):
    """Inputs for the adjudicate-mode PER-FACTOR draw over stored results.

    Independent of any annotate build: the cohort comes straight off the
    results — every judge factor's defect stratum takes
    ``min(per_factor_cap, #flagged-calls)`` flagged calls (capped, never
    padded), no clean controls. The draw persists its own frame + coverage
    sidecar, then renders the judge-only adjudication page through the same
    sidecar path as the pinned-cohort form.
    """

    results: Annotated[
        list[Path],
        Field(min_length=1, description="Results dirs / results.json files."),
    ]
    language: Annotated[str, Field(description="ISO 639-1 language code.")]
    conversation_artifact: Annotated[
        Optional[Path],
        Field(
            description="Stored conversation-judge artifact; source of the "
            "LLM-quality cells on corpora that store them there (see "
            "``CalibrationPacketOptions.conversation_artifact``)."
        ),
    ] = None
    per_factor_cap: Annotated[
        Optional[int],
        Field(gt=0, description="Per-factor cap override (default from config)."),
    ] = None
    decision_row_cap: Annotated[
        int,
        Field(
            ge=1,
            description="Per-factor cap on presented decision rows "
            "(``CalibrationAdjudicationOptions.decision_row_cap``).",
        ),
    ] = DEFAULT_DECISION_ROW_CAP
    seed: Annotated[int, Field(description="Draw seed.")] = DEFAULT_CALIBRATION_SEED
    batch_name: Annotated[Optional[str], Field(description="Packet label.")] = None
    out_dir: Annotated[Optional[Path], Field(description="Packet directory.")] = None
    emit_zip: Annotated[bool, Field(description="Emit the shipping ZIP.")] = True
    watermark: Annotated[
        Optional[str],
        Field(description="Banner override (default: the judge-only banner)."),
    ] = None

    @field_validator("language")
    @classmethod
    def _normalize_language(cls, value: str) -> str:
        return value.strip().lower()

    @property
    def effective_batch_name(self) -> str:
        return self.batch_name or f"judge_calibration_adjudication_{self.language}"

    @property
    def packet_dir(self) -> Path:
        return self.out_dir or (DEFAULT_OUTPUT_ROOT / self.effective_batch_name)

    @property
    def frame_path(self) -> Path:
        return self.packet_dir.parent / f"{self.packet_dir.name}_frame.json"

    @property
    def sidecar_path(self) -> Path:
        return self.packet_dir.parent / f"{self.packet_dir.name}_coverage.json"

    def draw_config(self) -> AdjudicationDrawConfig:
        overrides: dict = {"language": self.language, "seed": self.seed}
        if self.per_factor_cap is not None:
            overrides["per_factor_cap"] = self.per_factor_cap
        return AdjudicationDrawConfig(**overrides)


class CalibrationAdjudicationDrawBuild(BaseModel):
    """The per-factor draw's three artifacts, by path."""

    manifest_path: Path
    frame_path: Path
    sidecar_path: Path


def build_calibration_adjudication_draw(
    options: CalibrationAdjudicationDrawOptions,
) -> CalibrationAdjudicationDrawBuild:
    """Draw the per-factor cohort and build the judge-only adjudication page.

    Deterministic: same results + seed + cap → the same cohort, clip ids,
    and candidate ids. Per-factor counts land on the persisted frame's
    stratum accounting AND the coverage sidecar's ``per_factor_counts``.
    """
    config = options.draw_config()
    semantic_by_key = {}
    if options.conversation_artifact is not None:
        semantic_by_key = load_semantic_artifact(options.conversation_artifact).by_key()
    frame = build_adjudication_frame(
        iter_loaded_sims(options.results),
        config,
        results=options.results,
        conversation_artifact=options.conversation_artifact,
        semantic_by_key=semantic_by_key,
    )
    cohort = sorted(
        frame.drawn_calls(),
        key=lambda c: (c.results_path, c.task_id, c.trial, c.sim_id),
    )
    if not cohort:
        raise ValueError(
            "the per-factor draw selected no calls — the inputs hold no "
            f"judge FAIL for language '{config.language}' (see the frame "
            "drop counters and stratum table)"
        )

    # Second pass: load the drawn sims (integrity + judge version stamps).
    wanted = {call.key: call for call in cohort}
    loaded: dict[tuple[str, str], LoadedSim] = {}
    for item in iter_loaded_sims(options.results):
        key = (str((item.results_dir / "results.json").resolve()), item.sim.id)
        if key in wanted and key not in loaded:
            loaded[key] = item
    missing = sorted(set(wanted) - set(loaded))
    if missing:
        raise ValueError(f"drawn calls missing on the second pass: {missing[:5]}")

    selection = hashlib.sha256(
        json.dumps(
            {
                "sampler_version": CALIBRATION_SAMPLER_VERSION,
                "config": config.model_dump(mode="json"),
                "cohort": [
                    {"results_path": call.results_path, "sim_id": call.sim_id}
                    for call in cohort
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    batch_name = options.effective_batch_name
    batch_id = derive_batch_id(
        JUDGE_CALIBRATION_ADJUDICATION_KIND, batch_name, {"selection": selection}
    )

    frame_path = write_json_artifact(options.frame_path, frame)
    per_factor_counts = {
        row.stratum_id: row.covered
        for row in frame.strata
        if row.stratum_id != CLEAN_CONTROL_STRATUM
    }
    sidecar = CalibrationCoverageSidecar(
        created_at=datetime.now(timezone.utc).isoformat(),
        git_sha=git_sha(),
        batch_id=batch_id,
        batch_name=batch_name,
        language=config.language,
        frame_path=str(frame_path),
        conversation_artifact=(
            str(options.conversation_artifact)
            if options.conversation_artifact is not None
            else None
        ),
        cells_version=CALIBRATION_CELLS_VERSION,
        judge_versions=observed_judge_versions(
            [loaded[call.key].sim for call in cohort]
        ),
        per_factor_counts=per_factor_counts,
        rows=[
            CalibrationCoverageRow(
                clip_id=f"clip_{ordinal:03d}",
                sim_id=call.sim_id,
                task_id=call.task_id,
                trial=call.trial,
                results_path=call.results_path,
                provider=call.provider,
                experiment=call.experiment,
                judges_present=call.judges_present,
                strata=call.strata,
                drawn_for=call.drawn_for,
            )
            for ordinal, call in enumerate(cohort, 1)
        ],
    )
    sidecar_path = write_json_artifact(options.sidecar_path, sidecar)
    logger.info(
        f"per-factor adjudication draw '{batch_name}': {len(cohort)} calls "
        f"across {len(per_factor_counts)} defect strata (cap "
        f"{config.per_factor_cap}, seed {config.seed}) -> {sidecar_path}"
    )
    manifest_path = build_calibration_adjudication(
        CalibrationAdjudicationOptions(
            sidecar=sidecar_path,
            filled=[],
            batch_name=batch_name,
            out_dir=options.packet_dir,
            emit_zip=options.emit_zip,
            watermark=options.watermark,
            decision_row_cap=options.decision_row_cap,
        )
    )
    # The render may drop drawn calls whose decision candidates all capped
    # out (they would page as nothing-to-rate). This sidecar belongs to the
    # adjudication packet alone, so re-persist it without the dropped rows —
    # coverage counts included — to keep it consistent with what shipped.
    manifest = ArtifactManifest.model_validate_json(manifest_path.read_text())
    dropped = set(manifest.provenance.get("dropped_empty_clips") or [])
    if dropped:
        kept_rows = [row for row in sidecar.rows if row.clip_id not in dropped]
        sidecar = sidecar.model_copy(
            update={
                "rows": kept_rows,
                "per_factor_counts": {
                    stratum: sum(1 for row in kept_rows if stratum in row.strata)
                    for stratum in per_factor_counts
                },
            }
        )
        sidecar_path = write_json_artifact(options.sidecar_path, sidecar)
        logger.info(
            f"coverage sidecar rewritten without {len(dropped)} dropped "
            f"empty call(s): {sidecar_path}"
        )
    return CalibrationAdjudicationDrawBuild(
        manifest_path=manifest_path,
        frame_path=frame_path,
        sidecar_path=sidecar_path,
    )


__all__ = [
    "CALIBRATION_PACKET_VERSION",
    "JUDGE_CALIBRATION_ADJUDICATION_KIND",
    "JUDGE_CALIBRATION_KIND",
    "JUDGE_ONLY_BANNER",
    "JUDGE_VISIBLE_BANNER",
    "AdjudicationCallVM",
    "AdjudicationRowVM",
    "CalibrationAdjudicationDrawBuild",
    "CalibrationAdjudicationDrawOptions",
    "CalibrationAdjudicationOptions",
    "CalibrationDecisionCandidateVM",
    "CalibrationPacketBuild",
    "CalibrationPacketOptions",
    "build_calibration_adjudication",
    "build_calibration_adjudication_draw",
    "build_calibration_packet",
]
