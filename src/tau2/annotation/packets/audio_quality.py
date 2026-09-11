# Copyright Sierra
"""Agent-speech span helpers shared with the audio-quality judge.

The packet builders that once lived here shipped with the concluded
audio-quality annotation waves; what remains is the span logic that maps a
simulation's agent ticks to the utterance-bounded speech spans and their
transcript blocks. ``tau2.judges.audio_quality`` keys its per-utterance
evidence to these exact spans.
"""

from pathlib import Path
from typing import Annotated, Optional

from pydantic import BaseModel, Field

from tau2.data_model.audio import AudioData
from tau2.data_model.simulation import SimulationRun
from tau2.judges.delivery.disk_audio import (
    DiskAudioError,
    find_both_wav,
    sim_tick_duration_ms,
    slice_tick_span,
    split_agent_channel,
)
from tau2.voice.utils.audio_io import load_wav_file
from tau2.voice.utils.audio_preprocessing import merge_audio_datas


def agent_speech_tick_spans(sim: SimulationRun) -> list[tuple[int, int]]:
    """Agent utterance spans bounded by actual speech, not padded silent ticks.

    A shared utterance id keeps an internal pause inside the span. Silence
    between separate utterances is omitted when the spans are concatenated.
    Providers that record no utterance id fall back to contiguous speech ticks.
    """
    spans: list[tuple[int, int]] = []
    start: Optional[int] = None
    end: Optional[int] = None
    current_ids: set[str] = set()
    previous_speech_tick: Optional[int] = None

    for tick in sim.ticks or []:
        chunk = tick.agent_chunk
        if chunk is None or chunk.is_tool_call() or not chunk.contains_speech:
            continue
        ids = set(chunk.utterance_ids or [])
        if start is None:
            start = end = tick.tick_id
            current_ids = ids
        else:
            same_utterance = bool(current_ids and ids and current_ids & ids)
            contiguous_without_ids = (
                not current_ids
                and not ids
                and previous_speech_tick is not None
                and tick.tick_id == previous_speech_tick + 1
            )
            if same_utterance or contiguous_without_ids:
                end = tick.tick_id
                current_ids.update(ids)
            else:
                assert end is not None
                spans.append((start, end))
                start = end = tick.tick_id
                current_ids = ids
        previous_speech_tick = tick.tick_id

    if start is not None and end is not None:
        spans.append((start, end))
    return spans


def agent_speech_transcript(sim: SimulationRun) -> list[str]:
    """Return agent-only transcript blocks matching the exported speech spans."""
    ticks = sim.ticks or []
    transcript: list[str] = []
    for start, end in agent_speech_tick_spans(sim):
        text = "".join(
            chunk.content or ""
            for tick in ticks
            if start <= tick.tick_id <= end
            and (chunk := tick.agent_chunk) is not None
            and not chunk.is_tool_call()
            and chunk.contains_speech
        ).strip()
        if text:
            transcript.append(text)
    return transcript


AUDIO_QUALITY_RUBRIC_VERSION = "v3-fidelity-intonation"


class AudioQualityDimension(BaseModel):
    """One fixed audible phenomenon rated on every clip."""

    id: Annotated[str, Field(description="Stable CSV field id.")]
    title: Annotated[str, Field(description="Annotator-facing title.")]
    description: Annotated[str, Field(description="What the annotator listens for.")]


AUDIO_QUALITY_DIMENSIONS: tuple[AudioQualityDimension, ...] = (
    AudioQualityDimension(
        id="mispronunciation",
        title="Mispronunciation",
        description="A word is spoken incorrectly but remains the intended word.",
    ),
    AudioQualityDimension(
        id="word_substitution",
        title="Word substitution",
        description="The audio says a different word from the reference transcript.",
    ),
    AudioQualityDimension(
        id="missing_word",
        title="Missing word",
        description="A word or phrase in the reference transcript is not spoken.",
    ),
    AudioQualityDimension(
        id="extra_or_hallucinated_word",
        title="Extra or hallucinated word",
        description="The audio adds a word or phrase that is not in the reference.",
    ),
    AudioQualityDimension(
        id="number_date_currency",
        title="Number, date, or currency",
        description="A number, date, time, amount, or currency value is spoken incorrectly.",
    ),
    AudioQualityDimension(
        id="email_url_code",
        title="Email, URL, or code",
        description="An email address, URL, identifier, or code is spoken incorrectly.",
    ),
    AudioQualityDimension(
        id="punctuation_or_formatting",
        title="Punctuation and formatting",
        description="Punctuation or formatting causes an incorrect spoken realization.",
    ),
    AudioQualityDimension(
        id="acronym_brand_name",
        title="Acronym, brand, or name",
        description="An acronym, brand, product, person, or place name is spoken incorrectly.",
    ),
    AudioQualityDimension(
        id="clipped_or_garbled",
        title="Clipped or garbled",
        description="Speech is clipped, garbled, slurred, or cannot be understood.",
    ),
    AudioQualityDimension(
        id="other",
        title="Other fidelity issue",
        description="A spoken-output fidelity problem not covered above.",
    ),
    AudioQualityDimension(
        id="intonation",
        title="Intonation",
        description=(
            "Unnatural pauses, cadence, pitch, stress, timing, or delivery tone."
        ),
    ),
)


def concatenate_agent_speech(sim: SimulationRun, results_dir: Path) -> AudioData:
    """Losslessly split the agent channel and concatenate speech utterances."""
    wav = find_both_wav(results_dir, sim)
    if wav is None:
        raise DiskAudioError(f"sim {sim.id}: no both.wav under {results_dir}")
    agent = split_agent_channel(load_wav_file(wav))
    tick_ms = sim_tick_duration_ms(sim)
    spans = agent_speech_tick_spans(sim)
    if not spans:
        raise DiskAudioError(f"sim {sim.id}: no agent speech spans")
    pieces = [slice_tick_span(agent, start, end, tick_ms) for start, end in spans]
    return merge_audio_datas(pieces, silence_duration_ms=None)
