# Copyright Sierra
"""Audio effect processor mixins for batch and streaming modes."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
from pydantic import BaseModel

from tau2.data_model.audio import AudioData, AudioEncoding
from tau2.data_model.audio_effects import (
    ChannelEffectsConfig,
    ChannelEffectsResult,
    SourceEffectsResult,
    SpeechEffectsConfig,
    SpeechEffectsResult,
    UserSpeechInsert,
)
from tau2.voice.synthesis.audio_effects.effects import (
    apply_burst_noise,
    apply_dynamic_muffling,
    apply_frame_drops,
    convert_to_telephony,
)
from tau2.voice.synthesis.audio_effects.noise_generator import (
    BackgroundNoiseGenerator,
    apply_background_noise,
)
from tau2.voice.synthesis.audio_effects.scheduler import ScheduledEffect
from tau2.voice.synthesis.audio_effects.speech_generator import OutOfTurnSpeechGenerator
from tau2.voice.utils.audio_preprocessing import (
    AudioTracks,
    apply_fade_out,
    mix_audio_to_tracks,
    numpy_to_audio_data,
)


class PendingEffectState(BaseModel):
    """State for an effect that spans multiple chunks."""

    model_config = {"arbitrary_types_allowed": True}

    audio_bytes: bytes
    offset: int = 0
    info: Optional[dict] = None

    @property
    def remaining_bytes(self) -> int:
        return len(self.audio_bytes) - self.offset

    @property
    def is_complete(self) -> bool:
        return self.offset >= len(self.audio_bytes)


class BatchAudioEffectsMixin:
    """Mixin for batch audio effect processing (VoiceMixin)."""

    def apply_batch_effects(
        self,
        audio: AudioData,
        speech_effects: SpeechEffectsResult,
        source_effects: SourceEffectsResult,
        channel_effects: ChannelEffectsResult,
        speech_config: SpeechEffectsConfig,
        channel_config: ChannelEffectsConfig,
        noise_generator: Optional[BackgroundNoiseGenerator] = None,
        apply_speech_effects: bool = True,
        apply_source_effects: bool = True,
        apply_channel_effects: bool = True,
        apply_telephony: bool = True,
    ) -> AudioData:
        """Apply all batch effects: Speech → Noise → Burst → Telephony → Frame drops."""
        result = audio

        # 1. Speech effects (muffling)
        if apply_speech_effects:
            if speech_effects.dynamic_muffling_enabled:
                result = apply_dynamic_muffling(
                    result,
                    segment_count=speech_config.muffle_segment_count,
                    segment_duration_ms=speech_config.muffle_segment_duration_ms,
                    cutoff_freq=speech_config.muffle_cutoff_freq,
                    transition_ms=speech_config.muffle_transition_ms,
                )

        # 2. Background noise (continuous)
        if noise_generator is not None:
            result = apply_background_noise(result, noise_generator)

        # 3. Burst noise (source effect)
        if apply_source_effects and source_effects.burst_noise_file:
            result = apply_burst_noise(
                result,
                burst_noise_file=Path(source_effects.burst_noise_file),
            )

        # 4. Telephony conversion
        if apply_telephony:
            result = convert_to_telephony(result)

        # 5. Frame drops (channel effect)
        if apply_channel_effects and channel_effects.frame_drops_enabled:
            result = apply_frame_drops(
                result,
                drop_count=channel_config.frame_drop_count,
                drop_duration_ms=channel_config.frame_drop_duration_ms,
            )

        return result


@dataclass
class StreamingChunkResult:
    """Result of processing one streaming audio chunk.

    Contains individual audio tracks (for multitrack recording) plus
    metadata about effects that were triggered or completed.
    """

    tracks: AudioTracks
    out_of_turn_speech: np.ndarray
    pending_effect: Optional[PendingEffectState]
    source_effects: Optional[SourceEffectsResult]
    triggered_speech_insert: Optional[UserSpeechInsert]
    burst_noise_file: Optional[str]
    pending_effect_completed: bool
    pending_effect_discarded: bool

    def to_mixed_audio_data(self) -> AudioData:
        """Sum all tracks (including out-of-turn speech) into one AudioData."""
        mixed = (
            self.tracks.speech
            + self.tracks.background_noise
            + self.tracks.burst_noise
            + self.out_of_turn_speech
        )
        mixed = np.clip(mixed, -32768, 32767).astype(np.int16)
        return numpy_to_audio_data(
            mixed,
            encoding=AudioEncoding.PCM_S16LE,
            sample_rate=self.tracks.sample_rate,
            channels=1,
            dtype=np.int16,
        )


class StreamingAudioEffectsMixin:
    """Mixin for streaming audio effect processing (VoiceStreamingUserSimulator)."""

    def process_streaming_chunk(
        self,
        speech_audio: Optional[AudioData],
        noise_generator: BackgroundNoiseGenerator,
        num_samples: int,
        scheduled_effects: Optional[list[ScheduledEffect]] = None,
        out_of_turn_generator: Optional[OutOfTurnSpeechGenerator] = None,
        pending_effect: Optional[PendingEffectState] = None,
    ) -> StreamingChunkResult:
        """Process a single audio chunk, returning individual tracks.

        Returns a StreamingChunkResult containing:
        - Individual audio tracks (speech, background noise, burst noise,
          out-of-turn speech) for multitrack recording
        - Effect metadata (what was triggered/completed)
        - Updated pending effect state
        """
        is_speech = speech_audio is not None

        # If speech starts, discard any pending silence-only effect
        pending_effect_discarded = False
        if is_speech and pending_effect is not None:
            logger.debug(
                f"Speech started, discarding pending effect: {pending_effect.info}"
            )
            pending_effect = None
            pending_effect_discarded = True

        noise_data, snr_envelope = noise_generator.get_next_chunk(num_samples)
        burst_info = noise_generator.get_burst_chunk(num_samples)

        tracks = mix_audio_to_tracks(
            speech=speech_audio,
            noise=noise_data,
            snr_envelope_db=snr_envelope,
            burst_info=burst_info,
        )

        burst_noise_file = None
        triggered_speech_insert = None

        if scheduled_effects:
            for effect in scheduled_effects:
                if effect.timing == "cross_turn":
                    if (
                        effect.effect_type == "burst_noise_file"
                        and effect.burst_noise_file
                    ):
                        burst_noise_file = str(effect.burst_noise_file)
                        noise_generator.add_burst(effect.burst_noise_file)
                        logger.debug(f"Added burst noise: {burst_noise_file}")

                elif effect.timing == "out_of_turn" and not is_speech:
                    if (
                        effect.effect_type == "out_of_turn_speech"
                        and effect.out_of_turn_speech
                    ):
                        item = effect.out_of_turn_speech
                        if out_of_turn_generator and out_of_turn_generator.has_audio(
                            item
                        ):
                            audio = out_of_turn_generator.get_audio(item)
                            if audio:
                                pending_effect = PendingEffectState(
                                    audio_bytes=audio.data,
                                    offset=0,
                                    info={"type": item.type, "value": item.text},
                                )
                                triggered_speech_insert = item
                                logger.debug(
                                    f"Started out-of-turn {item.type}: {item.text}"
                                )

        # Build out-of-turn speech track (separate from the other tracks)
        oot_track = np.zeros(num_samples, dtype=np.float64)
        pending_effect_completed = False

        if (
            pending_effect is not None
            and not pending_effect.is_complete
            and not is_speech
        ):
            oot_track, pending_effect, pending_effect_completed = (
                self._get_pending_effect_track(pending_effect, num_samples)
            )

        source_effects = None
        if burst_noise_file or triggered_speech_insert:
            source_effects = SourceEffectsResult(
                burst_noise_file=burst_noise_file,
                speech_insert=triggered_speech_insert,
            )

        return StreamingChunkResult(
            tracks=tracks,
            out_of_turn_speech=oot_track,
            pending_effect=pending_effect,
            source_effects=source_effects,
            triggered_speech_insert=triggered_speech_insert,
            burst_noise_file=burst_noise_file,
            pending_effect_completed=pending_effect_completed,
            pending_effect_discarded=pending_effect_discarded,
        )

    def _get_pending_effect_track(
        self,
        pending_effect: PendingEffectState,
        num_samples: int,
    ) -> tuple[np.ndarray, Optional[PendingEffectState], bool]:
        """Extract the current chunk of a pending out-of-turn effect as a track.

        Returns:
            Tuple of (float64 track_samples, updated_pending_effect, completed).
        """
        bytes_per_sample = 2
        chunk_bytes = num_samples * bytes_per_sample

        remaining = pending_effect.remaining_bytes
        if remaining <= 0:
            return np.zeros(num_samples, dtype=np.float64), None, True

        portion_length = min(chunk_bytes, remaining)
        portion_bytes = pending_effect.audio_bytes[
            pending_effect.offset : pending_effect.offset + portion_length
        ]

        portion_np = (
            np.frombuffer(portion_bytes, dtype=np.int16).copy().astype(np.float64)
        )
        completed = False

        if len(portion_np) < num_samples:
            portion_np = apply_fade_out(portion_np)
            portion_np = np.pad(portion_np, (0, num_samples - len(portion_np)))

        new_offset = pending_effect.offset + portion_length

        if new_offset >= len(pending_effect.audio_bytes):
            logger.debug(f"Pending effect complete: {pending_effect.info}")
            completed = True
            return portion_np, None, completed
        else:
            return (
                portion_np,
                PendingEffectState(
                    audio_bytes=pending_effect.audio_bytes,
                    offset=new_offset,
                    info=pending_effect.info,
                ),
                completed,
            )

    def apply_streaming_frame_drop(
        self,
        audio_bytes: bytes,
        sample_rate: int,
        bytes_per_sample: int,
        drop_duration_ms: int,
        is_ulaw: bool = False,
    ) -> bytes:
        """Apply frame drop (packet loss simulation) by inserting silence."""
        samples_to_drop = int(drop_duration_ms * sample_rate / 1000)
        bytes_to_drop = samples_to_drop * bytes_per_sample

        # μ-law silence is 0xFF, PCM silence is 0x00
        silence_byte = b"\xff" if is_ulaw else b"\x00"

        if bytes_to_drop >= len(audio_bytes):
            return silence_byte * len(audio_bytes)
        else:
            return silence_byte * bytes_to_drop + audio_bytes[bytes_to_drop:]
