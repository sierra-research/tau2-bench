"""Lazy saved-run evidence and authoritative identities for post-hoc judges."""

import base64
import hashlib
import io
import json
import wave
from collections.abc import Iterator
from functools import cached_property
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from tau2.data_model.judges import ContractModel, JudgeInput, UnitError
from tau2.data_model.simulation import Results, SimulationRun, TerminationReason
from tau2.data_model.tasks import Task
from tau2.judges.exclusions import has_control_marker, terminal_speech_tick
from tau2.orchestrator.orchestrator import DEFAULT_FIRST_AGENT_MESSAGE
from tau2.utils.transcript.audio_text import extract_delivered_text
from tau2.utils.transcript.live import live_metadata, live_utterance_groups
from tau2.utils.transcript.live_audio import recover_live_audio
from tau2.utils.transcript.views import (
    agent_messages,
    agent_utterance_tick_spans,
    barged_in_tick_spans,
    tick_span_barged_in,
    utterance_interrupted,
)

SAVED_INPUT_VERSION = "ordinary-results-v4-live-tick-padding"
_HISTORICAL_CONTEXT = {"historical_judge_read": True}


class AudioEvidence(ContractModel):
    """Verified input location/hash or a local malformed-artifact error."""

    path: Annotated[Path | None, Field(description="Located ordinary audio artifact.")]
    sha256: Annotated[str | None, Field(description="Exact audio bytes fingerprint.")]
    error: Annotated[
        str | None,
        Field(description="Malformed audio/timeline, distinct from missing input."),
    ]


class AudioClip(ContractModel):
    """One aligned delivered agent utterance supplied to the frozen judge."""

    index: Annotated[int, Field(description="ticks_to_message_history ordinal.")]
    segment_id: str | None = Field(default=None, exclude=True)
    wav_base64: Annotated[str, Field(description="Mono agent-channel WAV.")]
    duration_seconds: Annotated[
        float, Field(description="Actual sliced clip duration.")
    ]
    reference: Annotated[
        str, Field(description="Untruncated delivered reference text.")
    ]
    interrupted: Annotated[
        bool, Field(description="Chunk or tick-overlap barge-in signal.")
    ]
    has_control_marker: bool = Field(default=False, exclude=True)
    terminal_cutoff: bool = Field(default=False, exclude=True)


def _find_audio(results: Path, simulation: SimulationRun) -> Path | None:
    root = results if results.is_dir() else results.parent
    for parent in ("tasks", "artifacts"):
        for sub in ("audio", ""):
            candidate = (
                root
                / parent
                / f"task_{simulation.task_id}"
                / f"sim_{simulation.id}"
                / sub
                / "both.wav"
            )
            if not candidate.resolve().is_relative_to(root.resolve()):
                raise ValueError("Audio identifier escapes results directory")
            if candidate.exists():
                return candidate
    return None


# A call-level measure cannot judge a call with an unresolved Live group; the
# suite stores the raised ``ValueError`` as ``"ValueError: <this>: <reasons>"``.
# Saved error rows keep only that string, so this prefix is their identity.
LIVE_INPUT_CALL_ERROR = "Call-level judgment requires complete Live input"


def is_live_input_call_error(error: str) -> bool:
    """Whether a saved whole-call judge error is the Live-input check above."""
    return error.startswith(f"ValueError: {LIVE_INPUT_CALL_ERROR}: ")


def live_input_errors(simulation: SimulationRun) -> list[UnitError]:
    """Identify only the Live groups whose audio/reference association is unresolved."""
    errors = []
    for group in live_utterance_groups(simulation.ticks or []):
        segment = group.chunks[0].raw_data["live_segment"]
        if reason := segment.get("audio_alignment_error"):
            errors.append(
                UnitError(unit_id=segment["segment_id"], kind="input", error=reason)
            )
    return errors


def load_audio_clips(path: Path, simulation: SimulationRun) -> list[AudioClip]:
    """Extract unchanged tick-aligned agent-channel clips and delivered references."""
    if not simulation.ticks:
        raise ValueError("Audio requires recorded full-duplex ticks")
    durations = {tick.tick_duration_seconds for tick in simulation.ticks}
    if len(durations) != 1 or None in durations or next(iter(durations)) <= 0:
        raise ValueError("Audio requires one recorded positive tick duration")
    seconds = next(iter(durations))
    ids = [tick.tick_id for tick in simulation.ticks]
    if ids != list(range(len(ids))):
        raise ValueError("Audio timeline requires contiguous zero-based ticks")
    with wave.open(str(path), "rb") as audio:
        if audio.getnchannels() != 2 or audio.getcomptype() != "NONE":
            raise ValueError("Audio requires stereo linear PCM; agent is right channel")
        width = audio.getsampwidth()
        rate = audio.getframerate()
        frames = audio.getnframes()
        raw = audio.readframes(frames)
    if width not in (1, 2, 4) or len(raw) != frames * width * 2:
        raise ValueError("Malformed stereo PCM audio")
    is_live = any(live_metadata(tick) is not None for tick in simulation.ticks)
    spans = agent_utterance_tick_spans(simulation)
    messages = agent_messages(simulation.ticks)
    if len(spans) != len(messages):
        raise ValueError("Reference/audio utterance alignment differs")
    barged = barged_in_tick_spans(simulation)
    terminal_tick = terminal_speech_tick(simulation)
    result = []
    for index, ((start, end), message) in enumerate(zip(spans, messages, strict=True)):
        if is_live and message.contains_speech is False:
            continue
        message_raw = message.raw_data
        segment = (
            message_raw.get("live_segment") if isinstance(message_raw, dict) else None
        )
        if segment and segment.get("audio_alignment_error"):
            continue
        ref, interrupted = extract_delivered_text(
            message.audio_script_gold or message.content or ""
        )
        control_marker = has_control_marker(ref)
        ref = ref.replace("###STOP###", "").strip()
        if not ref:
            continue
        first = round(start * seconds * rate)
        last = round((end + 1) * seconds * rate)
        if segment and "playback_start_us" in segment:
            first = round(segment["playback_start_us"] * rate / 1_000_000)
            last = round(segment["playback_end_us"] * rate / 1_000_000)
            if last > frames:
                raise ValueError("Live speech extends beyond the recorded audio")
        last = min(last, frames)
        if first >= frames or last <= first:
            raise ValueError("Utterance starts outside recorded audio timeline")
        # Preserve PCM samples bit-for-bit, taking only channel1 from stereo.
        import numpy as np

        mono = (
            np.frombuffer(raw, dtype=f"V{width}")
            .reshape(-1, 2)[first:last, 1]
            .tobytes()
        )
        output = io.BytesIO()
        with wave.open(output, "wb") as clip:
            clip.setnchannels(1)
            clip.setsampwidth(width)
            clip.setframerate(rate)
            clip.writeframes(mono)
        result.append(
            AudioClip(
                index=index,
                segment_id=segment.get("segment_id") if segment else None,
                wav_base64=base64.b64encode(output.getvalue()).decode(),
                duration_seconds=(last - first) / rate,
                reference=ref,
                has_control_marker=control_marker,
                terminal_cutoff=(
                    terminal_tick is not None and start <= terminal_tick <= end
                ),
                interrupted=(
                    interrupted
                    or utterance_interrupted([message])
                    or (not segment and tick_span_barged_in(start, end, barged))
                ),
            )
        )
    return result


def inspect_audio(results: Path, simulation: SimulationRun) -> AudioEvidence:
    """Read-only verification; missing and corrupt artifacts remain distinguishable."""
    path = _find_audio(results, simulation)
    if path is None:
        return AudioEvidence(path=None, sha256=None, error=None)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        load_audio_clips(path, simulation)
    except Exception as exc:
        return AudioEvidence(
            path=path, sha256=digest, error=f"{type(exc).__name__}: {exc}"
        )
    return AudioEvidence(path=path, sha256=digest, error=None)


class RunEvidence(ContractModel):
    """Recorded call evidence; implementations select only caller-visible task fields.

    The full recorded task participates in input identity and deterministic literal
    exclusions. Evaluation criteria and rewards must never enter model prompts.
    """

    simulation: Annotated[
        SimulationRun, Field(description="Recorded call with delivered messages/ticks.")
    ]
    task: Annotated[
        Task,
        Field(description="Recorded task; evaluator goldens are not prompt context."),
    ]
    language: Annotated[
        str, Field(description="Recorded effective language, never inferred.")
    ]
    agent_gender: Annotated[
        Literal["male", "female", "unknown"],
        Field(default="unknown", description="Recorded agent voice gender."),
    ]
    caller_gender: Annotated[
        Literal["male", "female", "unknown"],
        Field(default="unknown", description="Recorded caller gender."),
    ]
    domain: Annotated[
        str, Field(default="", description="Recorded environment domain.")
    ]
    policy: Annotated[
        str, Field(default="", description="Actual saved environment policy.")
    ]
    opener: Annotated[
        str | None,
        Field(
            default=None,
            description="Recorded benchmark-injected first agent message; None when the call had none.",
        ),
    ]
    results_path: Annotated[
        Path | None,
        Field(
            default=None, description="Saved run root used to locate audio artifacts."
        ),
    ]

    audio: Annotated[
        AudioEvidence | None,
        Field(
            default=None,
            description="Verified delivered audio reference and fingerprint, when loaded from saved results.",
        ),
    ]


def progress_path(results: Path) -> Path:
    """Place diagnostics beside run data, never inside simulation files."""
    if results.is_dir():
        return results / "diagnostics" / "verdicts.json"
    return results.with_name(f"{results.stem}.diagnostics.json")


class SavedRun:
    """Load metadata once, inventory calls by streaming, and verify on-demand reads."""

    def __init__(self, results: Path):
        self.results_path = results.resolve()
        self._metadata_stamp: tuple[int, int] | None = None
        self._monolithic: dict[str, SimulationRun] | None = None

    @cached_property
    def metadata_path(self) -> Path:
        return Results._resolve_paths(self.results_path)[0]

    @cached_property
    def metadata(self) -> Results:
        stamp = self.metadata_path.stat()
        self._metadata_stamp = (stamp.st_size, stamp.st_mtime_ns)
        return Results.load_metadata(self.results_path, context=_HISTORICAL_CONTEXT)

    @cached_property
    def tasks(self) -> dict[str, Task]:
        tasks = {task.id: task for task in self.metadata.tasks}
        if len(tasks) != len(self.metadata.tasks):
            raise ValueError("Saved result metadata contains duplicate task IDs")
        return tasks

    @cached_property
    def language(self) -> str:
        language = getattr(self.metadata.info, "language", None)
        if self.metadata.info.benchmark is not None:
            recorded = self.metadata.info.benchmark.language
            if language is not None and language != recorded:
                raise ValueError("Recorded language conflicts with benchmark metadata")
            language = recorded
        # A run without a recorded language is an English run; recorded values
        # win and must agree.
        return language or "en"

    @cached_property
    def run_id(self) -> str:
        return hashlib.sha256(
            json.dumps(
                {
                    "timestamp": self.metadata.timestamp,
                    "git_commit": self.metadata.info.git_commit,
                    "domain": self.metadata.info.environment_info.domain_name,
                    "language": self.language,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()

    @cached_property
    def inputs(self) -> list[JudgeInput]:
        """Stream authoritative identities without retaining directory-format calls."""
        # Validate metadata even when there are no saved calls.
        tasks, language, run_id = self.tasks, self.language, self.run_id
        inputs = []
        seen = set()
        for simulation in Results.iter_simulations(
            self.results_path, context=_HISTORICAL_CONTEXT
        ):
            if simulation.termination_reason == TerminationReason.INFRASTRUCTURE_ERROR:
                continue
            simulation = self._prepare_simulation(simulation)
            if simulation.id in seen:
                raise ValueError("Duplicate saved simulation IDs")
            seen.add(simulation.id)
            if simulation.trial is None:
                raise ValueError(
                    f"Simulation {simulation.id} has no recorded trial index"
                )
            task = tasks.get(simulation.task_id)
            if task is None:
                raise ValueError(
                    f"Simulation {simulation.id} has no matching recorded task"
                )
            inputs.append(
                _input_for_simulation(
                    self.metadata,
                    simulation,
                    task,
                    language,
                    run_id,
                    self.results_path,
                )
            )
        return inputs

    def iter_transcripts(
        self, simulation_ids: set[str]
    ) -> Iterator[tuple[SimulationRun, str | None]]:
        """Stream selected prepared calls with their opener, without audio hashing."""
        for simulation in Results.iter_simulations(
            self.results_path, context=_HISTORICAL_CONTEXT
        ):
            if simulation.id in simulation_ids:
                simulation = self._prepare_simulation(simulation)
                task = self.tasks[simulation.task_id]
                yield simulation, recorded_opener(self.metadata, simulation, task)

    def _prepare_simulation(self, simulation: SimulationRun) -> SimulationRun:
        config = self.metadata.info.audio_native_config
        if config is None or config.provider != "openai_live":
            return simulation
        if simulation.ticks and any(
            live_metadata(tick) is not None for tick in simulation.ticks
        ):
            return simulation
        path = _find_audio(self.results_path, simulation)
        if path is None:
            raise ValueError(
                "Recorded Live audio is required to recover judge boundaries"
            )
        return recover_live_audio(simulation, path, provider=config.provider)

    def load(self, item: JudgeInput) -> RunEvidence:
        """Read one planned call and reject metadata, identity, or evidence drift."""
        metadata = self.metadata
        stamp = self.metadata_path.stat()
        if (stamp.st_size, stamp.st_mtime_ns) != self._metadata_stamp:
            raise ValueError("Saved run metadata changed after planning")
        if Results._detect_format(self.results_path) == "dir":
            _, directory = Results._resolve_paths(self.results_path)
            filename = (directory / f"{item.simulation_id}.json").resolve()
            if not filename.is_relative_to(directory.resolve()):
                raise ValueError("Simulation identifier escapes saved results")
            simulation = SimulationRun.model_validate_json(
                filename.read_text(), context=_HISTORICAL_CONTEXT
            )
        else:
            if self._monolithic is None:
                self._monolithic = {
                    sim.id: sim
                    for sim in Results.iter_simulations(
                        self.results_path, context=_HISTORICAL_CONTEXT
                    )
                }
            simulation = self._monolithic[item.simulation_id]
        if simulation.id != item.simulation_id:
            raise ValueError("Saved simulation identity changed after planning")
        simulation = self._prepare_simulation(simulation)
        task = self.tasks[simulation.task_id]
        audio = inspect_audio(self.results_path, simulation)
        current = _input_for_simulation(
            metadata,
            simulation,
            task,
            self.language,
            self.run_id,
            self.results_path,
            audio=audio,
        )
        if current != item:
            raise ValueError("Saved judge inputs changed after planning")
        return RunEvidence(
            simulation=simulation,
            task=task,
            language=current.language,
            agent_gender=current.agent_gender,
            caller_gender=current.caller_gender,
            domain=metadata.info.environment_info.domain_name,
            policy=metadata.info.environment_info.policy,
            opener=recorded_opener(metadata, simulation, task),
            results_path=self.results_path,
            audio=audio,
        )


def _language_context(metadata: Results, simulation: SimulationRun, task: Task):
    context = getattr(simulation, "language_context", None)
    if context is None:
        context = getattr(metadata.info, "language_tasks", {}).get(task.id)
    return context


def recorded_opener(
    metadata: Results, simulation: SimulationRun, task: Task
) -> str | None:
    """The first agent message the orchestrator injected, from recorded inputs.

    Mirrors the orchestrator's precedence: the task's ``agent_opener``, else the
    recorded language context's reviewed greeting, else the English default.
    The exclusion rule additionally requires the call's first agent message to
    carry exactly this text, so a call without an injected opener (solo mode)
    loses nothing.
    """
    if task.agent_opener is not None:
        return task.agent_opener
    context = _language_context(metadata, simulation, task)
    if context is not None:
        return context.agent_greeting
    return DEFAULT_FIRST_AGENT_MESSAGE.content


def _input_for_simulation(
    metadata: Results,
    simulation: SimulationRun,
    task: Task,
    language: str,
    run_id: str,
    results_path: Path | None = None,
    *,
    audio: AudioEvidence | None = None,
) -> JudgeInput:
    """Fingerprint the same evidence during planning and immediately before judging."""
    info = metadata.info.model_dump(mode="json")
    gender = getattr(simulation, "agent_voice_gender", None) or "unknown"
    if gender not in {"male", "female", "unknown"}:
        raise ValueError("Invalid recorded agent voice gender")
    # Deliberately omit ordinary reward and post-hoc review fields: judgments
    # consume conversation/task evidence, not the benchmark's answer or scores.
    evidence = simulation.model_dump(
        mode="json",
        exclude={
            "reward_info",
            "review",
            "user_only_review",
            "agent_cost",
            "user_cost",
            "agent_usage",
        },
    )
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "adapter_version": SAVED_INPUT_VERSION,
                "info": info,
                "task": task.model_dump(mode="json"),
                "simulation": evidence,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    context = _language_context(metadata, simulation, task)
    caller_gender = getattr(context, "caller_gender", None) or "unknown"
    if audio is None and results_path is not None:
        audio = inspect_audio(results_path, simulation)
    return JudgeInput(
        domain=metadata.info.environment_info.domain_name,
        run_id=run_id,
        simulation_id=simulation.id,
        trial=simulation.trial,
        language=language,
        agent_gender=gender,
        input_fingerprint=fingerprint,
        caller_gender=caller_gender,
        has_audio=audio is not None and audio.path is not None,
        audio_fingerprint=audio.sha256 if audio is not None else None,
        audio_error=audio.error if audio is not None else None,
    )
