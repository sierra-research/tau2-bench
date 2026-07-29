# Copyright Sierra
"""Render a parsed phone-call transcript into a single call-recording WAV.

Pipeline per case: synthesize each spoken turn with the cast voice
(ElevenLabs via ``tau2.voice``), insert seeded silence gaps between turns —
longer "hold" pauses where the transcript records a support-console event —
then mix continuous background noise over the whole call and convert to
telephony bandwidth (μ-law 8kHz, saved as PCM16 WAV).

Support-console events are non-spoken and contribute only a pause; their
text never reaches TTS.

``mock=True`` replaces TTS with per-role sine tones so the full assembly
path can be exercised without an ELEVENLABS_API_KEY.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

import numpy as np
from elevenlabs import VoiceSettings as ElevenLabsVoiceSettings
from loguru import logger
from pydantic import BaseModel

from tau2.config import DEFAULT_PCM_SAMPLE_RATE
from tau2.data_model.audio import AudioData, AudioEncoding, AudioFormat
from tau2.data_model.audio_effects import SourceEffectsConfig
from tau2.data_model.voice import ElevenLabsTTSConfig
from tau2.hyper.call_audio.casting import CaseCasting
from tau2.hyper.call_audio.personas import CallVoice, get_call_voice
from tau2.hyper.call_audio.transcript_parser import (
    CallTranscript,
    ConsoleEvent,
    SpokenTurn,
)
from tau2.voice.synthesis.audio_effects.effects import convert_to_telephony
from tau2.voice.synthesis.audio_effects.noise_generator import (
    apply_background_noise,
    create_background_noise_generator,
)
from tau2.voice.synthesis.synthesize import synthesize_voice
from tau2.voice.utils.audio_io import load_wav_file, save_wav_file
from tau2.voice.utils.audio_preprocessing import merge_audio_datas
from tau2.voice_config import BACKGROUND_NOISE_CONTINUOUS_DIR

MOCK_TONE_HZ = {"agent": 170.0, "customer": 250.0}


def _silence(duration_ms: int) -> AudioData:
    # tau2's generate_silence_audio truncates float seconds to a byte count,
    # which can split a 16-bit sample in half; sizing in whole samples keeps
    # every segment mixable against the noise generator.
    num_samples = round(DEFAULT_PCM_SAMPLE_RATE * duration_ms / 1000)
    return AudioData(
        data=b"\x00\x00" * num_samples,
        format=AudioFormat(
            encoding=AudioEncoding.PCM_S16LE, sample_rate=DEFAULT_PCM_SAMPLE_RATE
        ),
    )


class RenderSettings(BaseModel):
    model_id: str = "eleven_v3"
    turn_gap_ms: tuple[int, int] = (350, 900)
    console_pause_ms: tuple[int, int] = (2000, 4500)
    background_noise: bool = True
    noise_snr_db: float = 20.0
    telephony: bool = True
    insert_audio_tags: bool = False
    mock: bool = False


def _mock_turn_audio(turn: SpokenTurn) -> AudioData:
    """Sine-tone stand-in for TTS, roughly matched to spoken duration."""
    duration_ms = 600 + 370 * len(turn.text.split())
    sample_rate = DEFAULT_PCM_SAMPLE_RATE
    num_samples = int(sample_rate * duration_ms / 1000)
    t = np.arange(num_samples) / sample_rate
    wave = 3000 * np.sin(2 * np.pi * MOCK_TONE_HZ[turn.role] * t)
    fade_samples = min(int(sample_rate * 0.06), num_samples // 2)
    envelope = np.ones(num_samples)
    envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
    envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
    samples = (wave * envelope).astype(np.int16)
    return AudioData(
        data=samples.tobytes(),
        format=AudioFormat(
            encoding=AudioEncoding.PCM_S16LE, sample_rate=sample_rate
        ),
    )


def _turn_cache_key(
    turn: SpokenTurn, voice: CallVoice, settings: RenderSettings, seed: int
) -> str:
    payload = "|".join(
        [
            turn.text,
            voice.elevenlabs_voice_id,
            str(voice.stability),
            str(voice.style),
            settings.model_id,
            str(settings.insert_audio_tags),
            str(seed),
        ]
    )
    return hashlib.md5(payload.encode()).hexdigest()[:12]


def _synthesize_turn(
    turn: SpokenTurn,
    voice: CallVoice,
    settings: RenderSettings,
    seed: int,
    cache_dir: Path | None,
) -> AudioData:
    if settings.mock:
        return _mock_turn_audio(turn)

    cache_path = None
    if cache_dir is not None:
        key = _turn_cache_key(turn, voice, settings, seed)
        cache_path = cache_dir / f"turn_{turn.turn:03d}_{turn.role}_{key}.wav"
        if cache_path.exists():
            logger.debug(f"Reusing cached audio for turn {turn.turn}")
            return load_wav_file(cache_path)

    config = ElevenLabsTTSConfig(
        model_id=settings.model_id,
        voice_id=voice.elevenlabs_voice_id,
        voice_settings=ElevenLabsVoiceSettings(
            stability=voice.stability,
            similarity_boost=0.75,
            style=voice.style,
            use_speaker_boost=False,
        ),
        insert_audio_tags=settings.insert_audio_tags,
        seed=seed,
    )
    audio = synthesize_voice(
        text=turn.text, provider="elevenlabs", provider_config=config
    )
    if cache_path is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        save_wav_file(audio, cache_path)
    return audio


def render_call(
    transcript: CallTranscript,
    casting: CaseCasting,
    settings: RenderSettings,
    output_path: Path,
    turn_cache_dir: Path | None = None,
) -> Path:
    """Render a call transcript to a single WAV file at output_path."""
    if not transcript.is_phone_call:
        raise ValueError(
            f"{transcript.source_path}: channel is {transcript.channel!r}, "
            "only 'phone call' records can be rendered to audio"
        )

    voices = {
        "agent": get_call_voice(casting.agent_voice),
        "customer": get_call_voice(casting.customer_voice),
    }
    rng = random.Random(casting.seed)

    segments: list[AudioData] = []
    pending_turn_gap = False
    for event in transcript.events:
        if isinstance(event, SpokenTurn):
            if pending_turn_gap:
                segments.append(_silence(rng.randint(*settings.turn_gap_ms)))
            segments.append(
                _synthesize_turn(
                    event,
                    voices[event.role],
                    settings,
                    seed=casting.seed + event.turn,
                    cache_dir=turn_cache_dir,
                )
            )
            pending_turn_gap = True
        elif isinstance(event, ConsoleEvent):
            # Non-spoken lookup/system event: the line goes quiet while the
            # agent works the console. This pause replaces the normal
            # inter-turn gap rather than stacking on top of it.
            segments.append(_silence(rng.randint(*settings.console_pause_ms)))
            pending_turn_gap = False

    call_audio = merge_audio_datas(segments, silence_duration_ms=None)

    if settings.background_noise:
        noise_files = sorted(BACKGROUND_NOISE_CONTINUOUS_DIR.glob("*.wav"))
        if not noise_files:
            logger.warning(
                f"No background noise files in {BACKGROUND_NOISE_CONTINUOUS_DIR}, "
                "rendering without noise"
            )
        else:
            noise_generator = create_background_noise_generator(
                config=SourceEffectsConfig(
                    enable_background_noise=True,
                    noise_snr_db=settings.noise_snr_db,
                    enable_burst_noise=False,
                ),
                sample_rate=call_audio.format.sample_rate,
                background_noise_file=rng.choice(noise_files),
            )
            call_audio = apply_background_noise(call_audio, noise_generator)

    if settings.telephony:
        call_audio = convert_to_telephony(call_audio)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # save_wav_file refuses to overwrite; the CLI has already decided
    # whether an existing output should be replaced.
    output_path.unlink(missing_ok=True)
    save_wav_file(call_audio, output_path)
    logger.info(
        f"Rendered {transcript.source_path.name} "
        f"({len(transcript.spoken_turns)} turns, {call_audio.duration:.1f}s) "
        f"-> {output_path}"
    )
    return output_path
