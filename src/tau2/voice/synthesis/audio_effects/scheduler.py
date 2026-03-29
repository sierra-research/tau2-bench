# Copyright Sierra
"""Audio effect scheduling and decision-making."""

import random
from pathlib import Path
from typing import Literal, Optional

from loguru import logger
from pydantic import BaseModel, Field, computed_field

from tau2.config import DEFAULT_TELEPHONY_RATE
from tau2.data_model.audio_effects import (
    ChannelEffectsConfig,
    ChannelEffectsResult,
    SourceEffectsConfig,
    SourceEffectsResult,
    SpeechEffectsConfig,
    SpeechEffectsResult,
    UserSpeechInsert,
)
from tau2.data_model.voice import SynthesisConfig
from tau2.voice.utils.probability import GilbertElliottModel, poisson_should_trigger
from tau2.voice.utils.utils import BURST_NOISE_FILES
from tau2.voice_config import ASSUMED_TURNS_PER_MINUTE

EffectTiming = Literal["cross_turn", "out_of_turn", "in_turn"]
"""When an effect can be triggered relative to speech turns."""


class ScheduledEffect(BaseModel):
    """A scheduled audio effect event."""

    model_config = {"arbitrary_types_allowed": True}

    effect_type: Literal["burst_noise_file", "out_of_turn_speech", "frame_drop"] = (
        Field(description="Type of audio effect")
    )
    timestamp_ms: int = Field(description="Timestamp when effect was triggered (ms)")
    duration_ms: int = Field(description="Expected duration of effect (ms)")
    timing: EffectTiming = Field(
        default="cross_turn",
        description="When this effect can occur relative to speech",
    )
    burst_noise_file: Optional[Path] = Field(
        default=None,
        description="Path to burst noise audio file",
    )
    out_of_turn_speech: Optional[UserSpeechInsert] = Field(
        default=None,
        description="Out-of-turn speech insert configuration",
    )
    frame_drop_duration_ms: Optional[int] = Field(
        default=None,
        description="Duration of frame drop in milliseconds",
    )


class EffectSchedulerState(BaseModel):
    """State for tracking effect scheduling during a conversation."""

    model_config = {"arbitrary_types_allowed": True}

    elapsed_samples: int = Field(
        default=0,
        description="Number of audio samples processed",
    )
    sample_rate: int = Field(
        default=DEFAULT_TELEPHONY_RATE,
        description="Audio sample rate in Hz",
    )
    pending_effects: list[ScheduledEffect] = Field(
        default_factory=list,
        description="Effects scheduled but not yet executed",
    )
    executed_effects: list[ScheduledEffect] = Field(
        default_factory=list,
        description="Effects that have been applied",
    )

    @computed_field
    @property
    def elapsed_ms(self) -> int:
        """Elapsed time in milliseconds."""
        return int(self.elapsed_samples * 1000 / self.sample_rate)


class EffectScheduler:
    """Effect scheduler for streaming audio effects."""

    def __init__(
        self,
        seed: int,
        source_config: SourceEffectsConfig,
        speech_config: SpeechEffectsConfig,
        channel_config: ChannelEffectsConfig,
        sample_rate: int = DEFAULT_TELEPHONY_RATE,
        burst_noise_events_per_minute: Optional[float] = None,
        speech_insert_events_per_minute: Optional[float] = None,
        frame_drop_rate: Optional[float] = None,
        frame_drop_burst_duration_ms: Optional[float] = None,
        burst_noise_files: Optional[list[Path]] = None,
    ):
        """Initialize the effect scheduler."""
        # Store configurations
        self.source_config = source_config
        self.speech_config = speech_config
        self.channel_config = channel_config
        self.sample_rate = sample_rate
        self.burst_noise_files = (
            burst_noise_files if burst_noise_files is not None else BURST_NOISE_FILES
        )

        # Initialize random number generators (separate streams for reproducibility)
        self._trigger_rng = random.Random(seed)
        self._burst_selection_rng = random.Random(seed + 1)
        self._speech_selection_rng = random.Random(seed + 2)
        self._frame_drop_rng = random.Random(seed + 3)

        # Configure Poisson rates for burst noise and speech inserts
        self.burst_rate = self._resolve_rate(
            burst_noise_events_per_minute,
            source_config.burst_noise_events_per_minute,
        )
        self.speech_insert_rate = self._resolve_rate(
            speech_insert_events_per_minute,
            speech_config.speech_insert_events_per_minute,
        )

        # Configure Gilbert-Elliott model for frame drops
        self.ge_model = self._create_ge_model(
            channel_config=channel_config,
            frame_drop_rate=frame_drop_rate,
            frame_drop_burst_duration_ms=frame_drop_burst_duration_ms,
        )

        # Log configuration
        target_loss_rate = (
            frame_drop_rate
            if frame_drop_rate is not None
            else channel_config.frame_drop_rate
        )
        burst_duration = (
            frame_drop_burst_duration_ms
            if frame_drop_burst_duration_ms is not None
            else channel_config.frame_drop_burst_duration_ms
        )
        logger.info(
            f"EffectScheduler initialized: "
            f"burst_rate={self.burst_rate:.4f}/s, "
            f"speech_insert_rate={self.speech_insert_rate:.4f}/s, "
            f"frame_drop_rate={target_loss_rate:.4f} (GE model), "
            f"frame_drop_burst_duration={burst_duration}ms, "
            f"burst_files={len(self.burst_noise_files)}"
        )

    @staticmethod
    def _resolve_rate(override: Optional[float], config_value: float) -> float:
        """Resolve rate from override or config, converting to per-second."""
        events_per_minute = override if override is not None else config_value
        return events_per_minute / 60.0

    def _create_ge_model(
        self,
        channel_config: ChannelEffectsConfig,
        frame_drop_rate: Optional[float],
        frame_drop_burst_duration_ms: Optional[float],
    ) -> Optional[GilbertElliottModel]:
        """Create Gilbert-Elliott model for frame drops if enabled."""
        if not channel_config.enable_frame_drops:
            return None

        target_loss_rate = (
            frame_drop_rate
            if frame_drop_rate is not None
            else channel_config.frame_drop_rate
        )

        if target_loss_rate <= 0:
            return None

        burst_duration = (
            frame_drop_burst_duration_ms
            if frame_drop_burst_duration_ms is not None
            else channel_config.frame_drop_burst_duration_ms
        )

        return GilbertElliottModel(
            target_loss_rate=target_loss_rate,
            avg_burst_duration_ms=burst_duration,
            rng=self._frame_drop_rng,
        )

    def check_for_effects(
        self,
        chunk_duration_ms: int,
        is_silence: bool,
        current_time_ms: int,
        has_active_burst: bool = False,
    ) -> list[ScheduledEffect]:
        """Check if any effects should be triggered for this audio chunk."""
        effects: list[ScheduledEffect] = []
        chunk_duration_sec = chunk_duration_ms / 1000.0

        # Check for burst noise (cross-turn, Poisson)
        burst_effect = self._check_burst_noise(
            chunk_duration_sec, current_time_ms, has_active_burst
        )
        if burst_effect:
            effects.append(burst_effect)

        # Check for out-of-turn speech (silence-only, Poisson)
        speech_effect = self._check_out_of_turn_speech(
            chunk_duration_sec, current_time_ms, is_silence
        )
        if speech_effect:
            effects.append(speech_effect)

        # Check for frame drops (cross-turn, Gilbert-Elliott)
        frame_drop_effect = self._check_frame_drop(chunk_duration_sec, current_time_ms)
        if frame_drop_effect:
            effects.append(frame_drop_effect)

        if effects:
            logger.info(
                f"Effects triggered at {current_time_ms}ms: "
                f"{[e.effect_type for e in effects]}"
            )

        return effects

    def _check_burst_noise(
        self,
        chunk_duration_sec: float,
        current_time_ms: int,
        has_active_burst: bool,
    ) -> Optional[ScheduledEffect]:
        """Check for burst noise trigger (Poisson, cross-turn)."""
        if not self.source_config.enable_burst_noise:
            return None
        if not self.burst_noise_files:
            return None
        if has_active_burst:
            return None

        if poisson_should_trigger(
            self.burst_rate, chunk_duration_sec, self._trigger_rng
        ):
            burst_file = self._burst_selection_rng.choice(self.burst_noise_files)
            return ScheduledEffect(
                effect_type="burst_noise_file",
                timestamp_ms=current_time_ms,
                duration_ms=500,  # Typical burst duration
                timing="cross_turn",
                burst_noise_file=burst_file,
            )

        return None

    def _check_out_of_turn_speech(
        self,
        chunk_duration_sec: float,
        current_time_ms: int,
        is_silence: bool,
    ) -> Optional[ScheduledEffect]:
        """Check for out-of-turn speech trigger (Poisson, silence-only)."""
        if not is_silence:
            return None

        items = self._get_out_of_turn_speech_inserts()
        if not items:
            return None

        if poisson_should_trigger(
            self.speech_insert_rate, chunk_duration_sec, self._trigger_rng
        ):
            item = self._speech_selection_rng.choice(items)
            duration_ms = 1500 if item.is_muffled else 1000
            return ScheduledEffect(
                effect_type="out_of_turn_speech",
                timestamp_ms=current_time_ms,
                duration_ms=duration_ms,
                timing="out_of_turn",
                out_of_turn_speech=item,
            )

        return None

    def _check_frame_drop(
        self,
        chunk_duration_sec: float,
        current_time_ms: int,
    ) -> Optional[ScheduledEffect]:
        """Check for frame drop trigger (Gilbert-Elliott, cross-turn)."""
        if not self.channel_config.enable_frame_drops:
            return None
        if self.ge_model is None:
            return None

        if self.ge_model.should_drop(chunk_duration_sec):
            return ScheduledEffect(
                effect_type="frame_drop",
                timestamp_ms=current_time_ms,
                duration_ms=self.channel_config.frame_drop_duration_ms,
                timing="cross_turn",
                frame_drop_duration_ms=self.channel_config.frame_drop_duration_ms,
            )

        return None

    def _get_out_of_turn_speech_inserts(self) -> list[UserSpeechInsert]:
        """Get combined list of out-of-turn speech inserts."""
        return self.speech_config.get_out_of_turn_speech_inserts()

    def get_burst_noise_file(self) -> Optional[Path]:
        """Get a random burst noise file."""
        if self.burst_noise_files:
            return self._burst_selection_rng.choice(self.burst_noise_files)
        return None

    def get_out_of_turn_speech_item(self) -> Optional[UserSpeechInsert]:
        """Get a random out-of-turn speech item."""
        items = self._get_out_of_turn_speech_inserts()
        if items:
            return self._speech_selection_rng.choice(items)
        return None


def generate_turn_effects(
    seed: int,
    turn_idx: int,
    synthesis_config: SynthesisConfig,
    burst_noise_files: Optional[list[Path]] = None,
) -> tuple[SpeechEffectsResult, SourceEffectsResult, ChannelEffectsResult]:
    """Generate per-turn audio effects using deterministic RNG (batch mode)."""
    rng = random.Random(seed + turn_idx)

    channel_config = synthesis_config.channel_effects_config
    source_config = synthesis_config.source_effects_config
    speech_config = synthesis_config.speech_effects_config
    available_burst_files = (
        burst_noise_files if burst_noise_files is not None else BURST_NOISE_FILES
    )

    # Read values directly from configs (complexity overrides already merged)
    muffle_prob = speech_config.muffle_probability

    # Convert rates to per-turn probability
    # TODO: Use actual turn duration for proper Poisson probability
    burst_prob = min(
        source_config.burst_noise_events_per_minute / ASSUMED_TURNS_PER_MINUTE, 0.5
    )
    # Use frame_drop_rate directly as probability for batch mode
    # Scale by 10 to get reasonable per-turn probability from loss rate
    frame_drop_prob = min(channel_config.frame_drop_rate * 10, 0.3)
    speech_insert_prob = min(
        speech_config.speech_insert_events_per_minute / ASSUMED_TURNS_PER_MINUTE, 0.5
    )

    # Determine which effects are enabled for this turn
    frame_drops_enabled = (
        channel_config.enable_frame_drops and rng.random() <= frame_drop_prob
    )

    dynamic_muffling_enabled = (
        speech_config.enable_dynamic_muffling and rng.random() <= muffle_prob
    )

    burst_noise_file: Optional[str] = None
    if source_config.enable_burst_noise and available_burst_files:
        if rng.random() <= burst_prob:
            burst_noise_file = str(rng.choice(available_burst_files))

    speech_insert: Optional[UserSpeechInsert] = None
    if speech_config.enable_vocal_tics and speech_config.vocal_tics:
        if rng.random() <= speech_insert_prob:
            speech_insert = rng.choice(speech_config.vocal_tics)

    # Build result objects
    speech_effects = SpeechEffectsResult(
        dynamic_muffling_enabled=dynamic_muffling_enabled,
        speech_insert=speech_insert,
    )

    source_effects = SourceEffectsResult(
        burst_noise_file=burst_noise_file,
    )

    channel_effects = ChannelEffectsResult(
        frame_drops_enabled=frame_drops_enabled,
    )

    return speech_effects, source_effects, channel_effects
