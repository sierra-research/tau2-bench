"""Reproduce the tau-Elicitation v6 utterance-level speech judgments.

The runner reads the saved stereo ``both.wav`` (caller left, agent right),
slices agent utterances by the recorded tick timeline, sends each WAV clip and
its delivered reference text through ``tau2.utils.llm_utils.generate()``, and
stores the same ``delivery_info`` contract used by the frozen paper runs.
"""

import base64
import hashlib
import json
import math
import re
import wave
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from importlib.resources import files
from io import BytesIO
from pathlib import Path
from typing import Literal, Optional

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tau2.agent.base.streaming_utils import extract_delivered_text
from tau2.data_model.message import Message, SystemMessage, UserMessage
from tau2.data_model.simulation import SimulationRun
from tau2.evaluator.evaluator_communicate import FullDuplexCommunicateEvaluator
from tau2.utils.llm_utils import extract_json_from_llm_response, generate

PROMPT_VERSION = "v6"
FINDING_FILTER_VERSION = "v1"
FIDELITY_END_EXCLUSION_SECONDS = 1.0
DEFAULT_SEED = 670487
_STOP_TOKEN = "###STOP###"
_BARGE_IN_GRACE_SECONDS = 2.0
_TIME_TOKEN = re.compile(r"\d+:\d+(?:\.\d+)?|\d+(?:\.\d+)?")

DeliveryAxis = Literal["fidelity", "intonation"]


class JudgeOutcome(str, Enum):
    """Outcome of one utterance-level speech judgment."""

    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"


class SpeechJudgeConfig(BaseModel):
    """Recorded configuration for a speech-judge rerun."""

    model: str = "gemini/gemini-3.1-pro-preview"
    model_args: dict = Field(
        default_factory=lambda: {
            "temperature": 0.0,
            "max_tokens": 8192,
            "timeout": 120,
        }
    )
    sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    max_segments: int = Field(default=100, gt=0)
    concurrency: int = Field(default=8, gt=0)
    seed: int = DEFAULT_SEED


class SpeechJudgeFinding(BaseModel):
    """One issue parsed from the model response."""

    model_config = ConfigDict(extra="ignore")

    axis: DeliveryAxis
    category: str = "other"
    time_range: Optional[str] = None
    issue: Optional[str] = None
    severity: int = Field(default=1, ge=1, le=3)
    confidence: Optional[float] = None

    @model_validator(mode="before")
    @classmethod
    def infer_axis(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        if data.get("axis") not in {"fidelity", "intonation"}:
            intonation = {
                "unnatural_pause",
                "missing_pause",
                "phrase_internal_gap",
                "robotic_or_mechanical_delivery",
                "unnatural_pitch_contour",
                "misplaced_stress",
                "inappropriate_upinflection",
                "monotone_flat_delivery",
                "emotional_mismatch",
                "cadence_or_speed_shift",
                "uncanny_prosody",
            }
            data["axis"] = (
                "intonation" if data.get("category") in intonation else "fidelity"
            )
        return data

    @field_validator("category", mode="before")
    @classmethod
    def default_category(cls, value: object) -> object:
        return "other" if value in (None, "") else value

    @field_validator("severity", mode="before")
    @classmethod
    def clamp_severity(cls, value: object) -> int:
        try:
            severity = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            severity = 1
        return max(1, min(3, severity))

    @field_validator("time_range", "issue", mode="before")
    @classmethod
    def empty_to_none(cls, value: object) -> object:
        return None if value in (None, "") else value


class SpeechJudgeResponse(BaseModel):
    """Typed boundary for the v6 model response."""

    model_config = ConfigDict(extra="ignore")

    findings: list[SpeechJudgeFinding] = Field(default_factory=list)
    factor_checks: list[dict] = Field(default_factory=list)
    summary: Optional[str] = None
    confidence: Optional[float] = None


class FindingExclusion(BaseModel):
    """Raw fidelity finding excluded by the calibrated final-window rule."""

    finding: SpeechJudgeFinding
    reason: Literal["final_utterance_window"] = "final_utterance_window"
    clip_duration_seconds: float
    exclusion_window_seconds: float = FIDELITY_END_EXCLUSION_SECONDS
    span_start_seconds: float
    span_end_seconds: float


class UtteranceJudgment(BaseModel):
    """Stored v6 result for one agent utterance."""

    utterance_idx: int
    was_interrupted: bool = False
    expected_text: Optional[str] = None
    outcome: JudgeOutcome
    flag_for_review: bool = False
    severity: int = 0
    confidence: Optional[float] = None
    summary: Optional[str] = None
    findings: list[SpeechJudgeFinding] = Field(default_factory=list)
    excluded_findings: list[FindingExclusion] = Field(default_factory=list)
    factor_checks: list[dict] = Field(default_factory=list)


class DeliveryInfo(BaseModel):
    """Paper-compatible simulation-level speech-judge payload."""

    score: Optional[float] = None
    fidelity_score: Optional[float] = None
    intonation_score: Optional[float] = None
    num_judged: int = 0
    num_errors: int = 0
    num_flagged: int = 0
    utterance_results: list[UtteranceJudgment] = Field(default_factory=list)
    factor_score: Optional[float] = None
    factor_checks: list[dict] = Field(default_factory=list)
    rubric_source: Optional[str] = None
    language: Optional[str] = None
    locale: Optional[str] = None
    judge_model: Optional[str] = None
    judge_args: Optional[dict] = None
    judge_prompt_version: Optional[str] = None
    finding_filter_version: Optional[str] = None
    fidelity_end_exclusion_seconds: Optional[float] = None
    num_findings_excluded: int = 0
    sample_rate: Optional[float] = None
    max_segments: Optional[int] = None
    seed: Optional[int] = None


class RejudgeReport(BaseModel):
    """Summary returned by a results-tree rerun."""

    judged: int = 0
    reused: int = 0
    unsampled: int = 0
    unavailable: int = 0
    output_description: str


class PromptSpec(BaseModel):
    """Exact fixed prompt components parsed from the checked-in v6 asset."""

    system: str
    user_template: str
    output_shape: str


def load_prompt_spec() -> PromptSpec:
    """Load the exact archived v6 prompt rather than duplicating prompt text."""
    path = files("tau2.paper").joinpath("prompts/elicitation_speech_v6.md")
    text = path.read_text(encoding="utf-8")
    system = text.split("## System prompt\n\n", 1)[1].split(
        "\n\n## Per-utterance user-prompt template", 1
    )[0]
    user_template = text.split("## Per-utterance user-prompt template\n\n", 1)[1].split(
        "\n\n## Output shape", 1
    )[0]
    output_shape = text.split("## Output shape\n\n", 1)[1].strip()
    return PromptSpec(
        system=system.strip(),
        user_template=user_template.strip(),
        output_shape=output_shape,
    )


def build_user_prompt(
    expected_text: str, *, was_interrupted: bool, locale: str = "en"
) -> str:
    """Render the checked-in v6 per-utterance template."""
    spec = load_prompt_spec()
    return spec.user_template.format(
        ref=expected_text.strip() or "(not provided)",
        was_interrupted=str(was_interrupted).lower(),
        locale=locale,
        rubric_section="",
        output_shape=spec.output_shape,
    )


def _time_bounds(value: Optional[str]) -> Optional[tuple[float, float]]:
    points: list[float] = []
    for token in _TIME_TOKEN.findall(value or ""):
        if ":" in token:
            minutes, seconds = token.split(":", 1)
            points.append(60 * float(minutes) + float(seconds))
        else:
            points.append(float(token))
    return (points[0], points[-1]) if points else None


def _in_final_window(value: Optional[str], duration: float) -> bool:
    bounds = _time_bounds(value)
    if bounds is None:
        return False
    start, end = bounds
    return end >= max(0.0, duration - 1.0) and start <= duration + 1.0


def _wav_duration(wav_b64: str) -> float:
    with wave.open(BytesIO(base64.b64decode(wav_b64)), "rb") as wav_file:
        return wav_file.getnframes() / wav_file.getframerate()


def run_speech_judge(
    wav_b64: str,
    expected_text: str,
    *,
    utterance_idx: int,
    was_interrupted: bool,
    config: SpeechJudgeConfig,
) -> UtteranceJudgment:
    """Make one versioned multimodal judge call and validate its reply."""
    spec = load_prompt_spec()
    reply = generate(
        model=config.model,
        messages=[
            SystemMessage(role="system", content=spec.system),
            UserMessage(
                role="user",
                content=build_user_prompt(
                    expected_text, was_interrupted=was_interrupted
                ),
                audio_content=wav_b64,
            ),
        ],
        call_name="delivery_judge",
        **config.model_args,
    )
    payload = json.loads(extract_json_from_llm_response(reply.content or ""))
    parsed = SpeechJudgeResponse.model_validate(payload)
    duration = _wav_duration(wav_b64)
    findings: list[SpeechJudgeFinding] = []
    excluded: list[FindingExclusion] = []
    for finding in parsed.findings:
        if finding.axis == "fidelity" and _in_final_window(
            finding.time_range, duration
        ):
            bounds = _time_bounds(finding.time_range)
            assert bounds is not None
            excluded.append(
                FindingExclusion(
                    finding=finding,
                    clip_duration_seconds=duration,
                    span_start_seconds=bounds[0],
                    span_end_seconds=bounds[1],
                )
            )
        else:
            findings.append(finding)
    severity = max((finding.severity for finding in findings), default=0)
    return UtteranceJudgment(
        utterance_idx=utterance_idx,
        was_interrupted=was_interrupted,
        expected_text=expected_text[:200] or None,
        outcome=JudgeOutcome.FAIL if findings else JudgeOutcome.PASS,
        flag_for_review=severity >= 2,
        severity=severity,
        confidence=parsed.confidence,
        summary=parsed.summary,
        findings=findings,
        excluded_findings=excluded,
    )


def _utterance_tick_spans(sim: SimulationRun) -> list[tuple[int, int]]:
    chunks: list[tuple[int, frozenset[str]]] = []
    for tick in sim.ticks or []:
        chunk = tick.agent_chunk
        if chunk is not None and not chunk.is_tool_call():
            chunks.append((tick.tick_id, frozenset(chunk.utterance_ids or [])))
    if not chunks:
        return []
    spans: list[tuple[int, int]] = []
    start = end = chunks[0][0]
    current_ids = set(chunks[0][1])
    for tick_id, ids in chunks[1:]:
        if current_ids and ids and not current_ids.isdisjoint(ids):
            end = tick_id
            current_ids.update(ids)
        else:
            spans.append((start, end))
            start = end = tick_id
            current_ids = set(ids)
    spans.append((start, end))
    return spans


def _barge_in_indices(sim: SimulationRun) -> set[int]:
    if not sim.ticks:
        return set()
    tick_seconds = next(
        (
            tick.tick_duration_seconds
            for tick in sim.ticks
            if tick.tick_duration_seconds
        ),
        0.2,
    )
    grace_ticks = max(1, math.ceil(_BARGE_IN_GRACE_SECONDS / tick_seconds))
    agent_speech = {
        tick.tick_id
        for tick in sim.ticks
        if tick.agent_chunk is not None
        and not tick.agent_chunk.is_tool_call()
        and tick.agent_chunk.contains_speech
    }
    user_speech = {
        tick.tick_id
        for tick in sim.ticks
        if tick.user_chunk is not None and tick.user_chunk.contains_speech
    }
    result: set[int] = set()
    for idx, (start, end) in enumerate(_utterance_tick_spans(sim)):
        last = max((t for t in agent_speech if start <= t <= end), default=None)
        if last is not None and any(
            t in user_speech for t in range(max(start, last - grace_ticks), last + 1)
        ):
            result.add(idx)
    return result


def _find_both_wav(cell_root: Path, sim: SimulationRun) -> Optional[Path]:
    for parent in ("tasks", "artifacts"):
        for sub in ("audio", ""):
            path = (
                cell_root
                / parent
                / f"task_{sim.task_id}"
                / f"sim_{sim.id}"
                / sub
                / "both.wav"
            )
            if path.exists():
                return path
    return None


def _agent_channel(path: Path) -> tuple[bytes, int, int]:
    """Return right-channel PCM bytes, sample rate, and sample width."""
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getnchannels() != 2 or wav_file.getcomptype() != "NONE":
            raise ValueError(f"expected uncompressed stereo both.wav: {path}")
        rate = wav_file.getframerate()
        width = wav_file.getsampwidth()
        frames = wav_file.readframes(wav_file.getnframes())
    frame_width = width * 2
    right = b"".join(
        frames[offset + width : offset + frame_width]
        for offset in range(0, len(frames), frame_width)
    )
    return right, rate, width


def _slice_wav_b64(
    agent_pcm: bytes,
    rate: int,
    width: int,
    start_tick: int,
    end_tick: int,
    tick_seconds: float,
) -> str:
    samples = len(agent_pcm) // width
    start = round(start_tick * tick_seconds * rate)
    end = min(round((end_tick + 1) * tick_seconds * rate), samples)
    if start >= samples:
        raise ValueError("tick span starts beyond the stored audio timeline")
    out = BytesIO()
    with wave.open(out, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(width)
        wav_file.setframerate(rate)
        wav_file.writeframes(agent_pcm[start * width : end * width])
    return base64.b64encode(out.getvalue()).decode("ascii")


def _delivered_reference(message: Message) -> tuple[str, bool]:
    text, interrupted = extract_delivered_text(
        message.audio_script_gold or message.content or ""
    )
    return text.replace(_STOP_TOKEN, "").strip(), interrupted


def _sample_value(key: str) -> float:
    digest = hashlib.sha256(key.encode()).hexdigest()
    return int(digest[:8], 16) / 0x100000000


def _even_spread_indices(n: int, k: int) -> list[int]:
    if k >= n:
        return list(range(n))
    if k <= 1:
        return [0] if k == 1 else []
    return [round(i * (n - 1) / (k - 1)) for i in range(k)]


def _cleanliness(severity: int) -> float:
    return 1.0 - max(0, min(3, severity)) / 3.0


def _axis_score(
    judgments: list[UtteranceJudgment], axis: DeliveryAxis
) -> Optional[float]:
    scored = [item for item in judgments if item.outcome != JudgeOutcome.ERROR]
    if not scored:
        return None
    values = []
    for item in scored:
        worst = max(
            (finding.severity for finding in item.findings if finding.axis == axis),
            default=0,
        )
        values.append(_cleanliness(worst))
    return sum(values) / len(values)


def judge_simulation(
    sim: SimulationRun, cell_root: Path, config: SpeechJudgeConfig
) -> Optional[DeliveryInfo]:
    """Regenerate one simulation's paper-compatible ``delivery_info``."""
    if not sim.ticks:
        raise ValueError("simulation has no full-duplex tick timeline")
    if (
        config.sample_rate < 1.0
        and _sample_value(f"{sim.id}:{config.seed}") >= config.sample_rate
    ):
        return None
    wav_path = _find_both_wav(cell_root, sim)
    if wav_path is None:
        raise ValueError("simulation has no stored both.wav")
    spans = _utterance_tick_spans(sim)
    messages = FullDuplexCommunicateEvaluator.ticks_to_message_history(sim.ticks)
    if len(spans) != len(messages):
        raise ValueError("agent utterance spans do not align with merged messages")
    agent_pcm, rate, width = _agent_channel(wav_path)
    tick_seconds = next(
        (
            tick.tick_duration_seconds
            for tick in sim.ticks
            if tick.tick_duration_seconds
        ),
        None,
    )
    if tick_seconds is None:
        raise ValueError("simulation ticks do not record tick_duration_seconds")
    barged = _barge_in_indices(sim)
    last_tick = sim.ticks[-1].tick_id
    eligible: list[tuple[int, str, str, bool]] = []
    for idx, (message, span) in enumerate(zip(messages, spans)):
        reference, interrupted = _delivered_reference(message)
        if not reference:
            continue
        wav_b64 = _slice_wav_b64(agent_pcm, rate, width, span[0], span[1], tick_seconds)
        eligible.append(
            (
                idx,
                wav_b64,
                reference,
                interrupted or idx in barged or span[1] >= last_tick,
            )
        )
    selected = [
        eligible[index]
        for index in _even_spread_indices(len(eligible), config.max_segments)
    ]

    def run(item: tuple[int, str, str, bool]) -> UtteranceJudgment:
        idx, wav_b64, reference, interrupted = item
        try:
            return run_speech_judge(
                wav_b64,
                reference,
                utterance_idx=idx,
                was_interrupted=interrupted,
                config=config,
            )
        except Exception as exc:  # noqa: BLE001 - preserve per-unit failure
            logger.warning(
                f"speech judge failed for sim {sim.id} utterance {idx}: {exc}"
            )
            return UtteranceJudgment(
                utterance_idx=idx,
                was_interrupted=interrupted,
                expected_text=reference[:200] or None,
                outcome=JudgeOutcome.ERROR,
                summary=str(exc),
            )

    with ThreadPoolExecutor(max_workers=config.concurrency) as pool:
        judgments = list(pool.map(run, selected))
    scored = [item for item in judgments if item.outcome != JudgeOutcome.ERROR]
    score = (
        sum(_cleanliness(item.severity) for item in scored) / len(scored)
        if scored
        else None
    )
    return DeliveryInfo(
        score=score,
        fidelity_score=_axis_score(judgments, "fidelity"),
        intonation_score=_axis_score(judgments, "intonation"),
        num_judged=len(judgments),
        num_errors=len(judgments) - len(scored),
        num_flagged=sum(item.flag_for_review for item in scored),
        utterance_results=judgments,
        judge_model=config.model if selected else None,
        judge_args=dict(config.model_args) if selected else None,
        judge_prompt_version=PROMPT_VERSION if selected else None,
        finding_filter_version=FINDING_FILTER_VERSION if selected else None,
        fidelity_end_exclusion_seconds=(
            FIDELITY_END_EXCLUSION_SECONDS if selected else None
        ),
        num_findings_excluded=sum(len(item.excluded_findings) for item in judgments),
        sample_rate=config.sample_rate,
        max_segments=config.max_segments,
        seed=config.seed,
    )


def _simulation_paths(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    direct = source / "simulations"
    if direct.is_dir():
        return sorted(direct.glob("*.json"))
    return sorted(source.rglob("simulations/*.json"))


def _is_frozen(path: Path) -> bool:
    parts = path.resolve().parts
    return "paper_runs" in parts and "tau-elicit" in parts


def rejudge_results(
    source: Path,
    *,
    output: Optional[Path],
    replace_existing: bool,
    limit_sims: Optional[int],
    config: SpeechJudgeConfig,
) -> RejudgeReport:
    """Regenerate speech judgments over a results file/cell/tree.

    Frozen paper roots are read-only by contract. To rejudge them, provide an
    output directory; only mirrored simulation JSON is written there, while
    audio remains read from the detached source root.
    """
    source = source.resolve()
    if output is None and _is_frozen(source):
        raise ValueError(
            "refusing to modify a frozen tau-elicit paper root; pass --output"
        )
    paths = _simulation_paths(source)
    if limit_sims is not None:
        paths = paths[:limit_sims]
    report = RejudgeReport(
        output_description=str(output.resolve() if output else source)
    )
    for path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("delivery_info") and not replace_existing:
            report.reused += 1
            continue
        sim = SimulationRun.model_validate(raw)
        cell_root = path.parent.parent
        try:
            info = judge_simulation(sim, cell_root, config)
        except (OSError, ValueError, wave.Error) as exc:
            logger.warning(f"speech rejudge unavailable for {path}: {exc}")
            report.unavailable += 1
            continue
        if info is None:
            report.unsampled += 1
            continue
        raw["delivery_info"] = info.model_dump(mode="json")
        destination = path
        if output is not None:
            base = source.parent if source.is_file() else source
            destination = output.resolve() / path.relative_to(base)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        report.judged += 1
    return report
