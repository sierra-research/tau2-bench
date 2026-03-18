# Copyright Sierra
"""Voice data models for synthesis, transcription, and audio effects."""

from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, computed_field

from tau2.config import DEFAULT_SEED
from tau2.data_model.audio import AudioEncoding, AudioFormat
from tau2.data_model.audio_effects import (
    ChannelEffectsConfig,
    SourceEffectsConfig,
    SpeechEffectsConfig,
)
from tau2.data_model.persona import PersonaConfig
from tau2.data_model.voice_personas import DEFAULT_PERSONA_NAME
from tau2.voice_config import (
    BURST_NOISE_EVENTS_PER_MINUTE,
    DEFAULT_TRANSCRIPTION_MODEL,
    DEFAULT_VOICE_SYNTHESIS_PROVIDER,
    ELEVENLABS_AUDIO_TAGS_PROBABILITY,
    ELEVENLABS_ENABLE_AUDIO_TAGS,
    ENABLE_BACKGROUND_NOISE,
    ENABLE_BURST_NOISE,
    SNR_SPEECH_REFERENCE_RMS,
)

try:
    from elevenlabs import VoiceSettings as ElevenLabsVoiceSettings
except ImportError:
    ElevenLabsVoiceSettings = None

# ============================================================================
# Transcription Models
# ============================================================================

TranscriptionModel = Literal[
    "nova-2", "nova-3", "whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe"
]


class TranscriptionConfig(BaseModel):
    """Configuration for audio transcription."""

    model: TranscriptionModel = Field(default=DEFAULT_TRANSCRIPTION_MODEL)
    language: Optional[str] = Field(default=None)
    deepgram_punctuate: bool = Field(default=True)
    deepgram_smart_format: bool = Field(default=False)
    openai_silence_duration_ms: int = Field(default=1000)
    extra_options: dict[str, Any] = Field(default_factory=dict)


class TranscriptionResult(BaseModel):
    """Result of a transcription operation."""

    transcript: str = Field(description="The transcribed text")
    confidence: Optional[float] = Field(default=None)
    error: Optional[str] = Field(default=None)


# ============================================================================
# ElevenLabs TTS Models
# ============================================================================

if ElevenLabsVoiceSettings is not None:
    DEFAULT_ELEVENLABS_VOICE_SETTINGS = ElevenLabsVoiceSettings(
        stability=0.5, similarity_boost=0.75, style=0.0, use_speaker_boost=False
    )
else:
    DEFAULT_ELEVENLABS_VOICE_SETTINGS = None

DEFAULT_ELEVENLABS_OUTPUT_FORMAT = "pcm_16000"
DEFAULT_ELEVEN_LAB_AUDIO_FORMAT = AudioFormat(
    encoding=AudioEncoding.PCM_S16LE,
    sample_rate=16000,
)


class ElevenLabsTTSConfig(BaseModel):
    """Configuration for ElevenLabs TTS."""

    model_id: str = Field(default="eleven_v3")
    voice_id: Optional[str] = Field(default=None)
    api_key: Optional[str] = Field(default=None)
    voice_settings: Any = Field(default=DEFAULT_ELEVENLABS_VOICE_SETTINGS)

    @property
    def output_format_name(self) -> str:
        return DEFAULT_ELEVENLABS_OUTPUT_FORMAT

    @property
    def output_audio_format(self) -> AudioFormat:
        return DEFAULT_ELEVEN_LAB_AUDIO_FORMAT

    insert_audio_tags: bool = Field(default=True)
    audio_tags_probability: float = Field(default=0.2, ge=0.0, le=1.0)
    seed: Optional[int] = Field(default=None)


ProviderConfig = ElevenLabsTTSConfig


# ============================================================================
# Synthesis Models
# ============================================================================


class SynthesisConfig(BaseModel):
    """Voice synthesis configuration with 3-tier effect taxonomy."""

    provider: str = Field(default="elevenlabs")
    provider_config: Optional[ProviderConfig] = Field(default=ElevenLabsTTSConfig())
    channel_effects_config: ChannelEffectsConfig = Field(
        default_factory=ChannelEffectsConfig
    )
    source_effects_config: SourceEffectsConfig = Field(
        default_factory=SourceEffectsConfig
    )
    speech_effects_config: SpeechEffectsConfig = Field(
        default_factory=SpeechEffectsConfig
    )


class SynthesisResult(BaseModel):
    """Result of a voice synthesis operation."""

    text_input: Optional[str] = None
    audio_data: Optional[object] = None
    error: Optional[str] = None


# ============================================================================
# Environment Models
# ============================================================================

# Speech complexity level type
# - control: Clean baseline (no effects, American accents, patient user)
# - regular: Full realistic conditions (all effects enabled)
# Single-feature ablations:
# - control_audio: Control + audio/transmission effects (noise, muffling, frame drops)
# - control_accents: Control + diverse accent personas
# - control_behavior: Control + user behavior effects (interrupts, backchannels, speech inserts)
# Pairwise ablations:
# - control_audio_accents: Control + audio effects + diverse accents
# - control_audio_behavior: Control + audio effects + user behavior
# - control_accents_behavior: Control + diverse accents + user behavior
SpeechComplexity = Literal[
    "control",
    "regular",
    "control_audio",
    "control_accents",
    "control_behavior",
    "control_audio_accents",
    "control_audio_behavior",
    "control_accents_behavior",
]


class SpeechEnvironment(BaseModel):
    """A sampled voice environment instance for a given complexity level.

    All fields are determined by sampling from the complexity preset.
    Stored on SimulationRun for reproducibility and analysis.
    """

    voice_seed: int = Field(default=DEFAULT_SEED)
    persona_name: str = Field(default=DEFAULT_PERSONA_NAME)
    background_noise_file: Optional[str] = Field(default=None)
    burst_noise_files: list[str] = Field(
        default_factory=list,
        description="List of burst noise files for this task's environment",
    )
    environment: Optional[str] = Field(
        default=None,
        description="Environment preset (indoor/outdoor), or None if control",
    )
    backchannel_min_threshold: Optional[int] = Field(
        default=None,
        description="Threshold for backchanneling (ticks of agent speech). None disables backchanneling.",
    )
    use_llm_backchannel: bool = Field(
        default=True,
        description="Whether to use LLM-based backchanneling. If False, uses Poisson-based policy.",
    )
    enable_interruptions: bool = Field(
        default=False,
        description="Whether user interruptions are enabled",
    )
    telephony_enabled: bool = Field(default=True)
    complexity: SpeechComplexity = Field(default="regular")
    snr_speech_reference_rms: float = Field(
        default=SNR_SPEECH_REFERENCE_RMS,
        description="Speech RMS level used as the reference for SNR-based noise scaling. "
        "Noise is scaled so that noise_level / this_value = 10^(-SNR/20). "
        "Currently a fixed estimate of typical TTS speech level in 16-bit PCM.",
    )

    source_effects_config: Optional[SourceEffectsConfig] = Field(
        default=None,
        description="Sampled source/acoustic effects config (noise, bursts) with complexity overrides applied.",
    )
    speech_effects_config: Optional[SpeechEffectsConfig] = Field(
        default=None,
        description="Sampled speech effects config (muffling, vocal tics) with complexity overrides applied.",
    )
    channel_effects_config: Optional[ChannelEffectsConfig] = Field(
        default=None,
        description="Sampled channel effects config (frame drops) with complexity overrides applied.",
    )


class SampledVoiceConfig(BaseModel):
    """Result of sampling voice configuration from complexity presets.

    Contains all instantiated configs with complexity overrides applied.
    This is the single source of truth for voice configuration at task level.
    """

    # Simulation-level settings (consistent throughout a task)
    persona_name: str = Field(description="Selected persona name for the user")
    background_noise_file: Optional[str] = Field(
        default=None, description="Selected background noise file, or None if disabled"
    )
    burst_noise_files: list[str] = Field(
        default_factory=list,
        description="List of burst noise files for this task's environment",
    )
    environment: Optional[str] = Field(
        default=None,
        description="Environment preset (indoor/outdoor), or None if control",
    )
    backchannel_min_threshold: Optional[int] = Field(
        default=None,
        description="Threshold for backchanneling (ticks of agent speech). None disables backchanneling.",
    )
    use_llm_backchannel: bool = Field(
        default=True,
        description="Whether to use LLM-based backchanneling. If False, uses Poisson-based policy.",
    )
    enable_interruptions: bool = Field(
        default=False, description="Whether user interruptions are enabled"
    )
    telephony_enabled: bool = Field(
        default=True,
        description="Whether telephony compression (G.711 μ-law 8kHz) is enabled",
    )

    # Merged effect configs with complexity overrides applied
    channel_effects_config: ChannelEffectsConfig = Field(
        description="Channel effects config with complexity overrides"
    )
    source_effects_config: SourceEffectsConfig = Field(
        description="Source effects config with complexity overrides"
    )
    speech_effects_config: SpeechEffectsConfig = Field(
        description="Speech effects config with complexity overrides"
    )

    # User persona config
    persona_config: PersonaConfig = Field(
        description="User persona configuration (verbosity, interrupt tendency)"
    )

    # Complexity level (stored for reference)
    complexity: SpeechComplexity = Field(description="The complexity level used")

    def to_speech_environment(self, seed: int) -> "SpeechEnvironment":
        """Create a SpeechEnvironment from this sampled config."""
        return SpeechEnvironment(
            voice_seed=seed,
            persona_name=self.persona_name,
            background_noise_file=self.background_noise_file,
            burst_noise_files=self.burst_noise_files,
            environment=self.environment,
            backchannel_min_threshold=self.backchannel_min_threshold,
            use_llm_backchannel=self.use_llm_backchannel,
            enable_interruptions=self.enable_interruptions,
            telephony_enabled=self.telephony_enabled,
            complexity=self.complexity,
            source_effects_config=self.source_effects_config,
            speech_effects_config=self.speech_effects_config,
            channel_effects_config=self.channel_effects_config,
        )


class VoiceSettings(BaseModel):
    """Voice settings for enabling synthesis and transcription."""

    synthesis_config: Optional[SynthesisConfig] = Field(default=SynthesisConfig())
    transcription_config: Optional[TranscriptionConfig] = Field(
        default=TranscriptionConfig()
    )
    output_dir: Optional[Path] = Field(default=None)
    speech_environment: Optional[SpeechEnvironment] = Field(
        default_factory=SpeechEnvironment
    )

    @computed_field
    @property
    def synthesis_enabled(self) -> bool:
        return self.synthesis_config is not None

    @computed_field
    @property
    def transcription_enabled(self) -> bool:
        return self.transcription_config is not None

    @classmethod
    def from_cli_args(
        cls,
        voice_synthesis_provider: str = DEFAULT_VOICE_SYNTHESIS_PROVIDER,
        voice_transcription_model: str = DEFAULT_TRANSCRIPTION_MODEL,
        output_dir: Optional[Path] = None,
        burst_noise_events_per_minute: float = BURST_NOISE_EVENTS_PER_MINUTE,
        audio_tags_prob: float = ELEVENLABS_AUDIO_TAGS_PROBABILITY,
        no_background_noise: bool = not ENABLE_BACKGROUND_NOISE,
        no_burst_noise: bool = not ENABLE_BURST_NOISE,
        no_audio_tags: bool = not ELEVENLABS_ENABLE_AUDIO_TAGS,
    ) -> "VoiceSettings":
        """Create a VoiceSettings instance from CLI arguments."""
        if voice_synthesis_provider != "elevenlabs":
            raise ValueError(
                f"Unsupported voice synthesis provider: {voice_synthesis_provider}"
            )

        provider_config = ElevenLabsTTSConfig()
        provider_config.insert_audio_tags = not no_audio_tags
        provider_config.audio_tags_probability = audio_tags_prob

        source_effects_config = SourceEffectsConfig(
            enable_background_noise=not no_background_noise,
            enable_burst_noise=not no_burst_noise,
            burst_noise_events_per_minute=burst_noise_events_per_minute,
        )

        synthesis_config = SynthesisConfig(
            provider=voice_synthesis_provider,
            provider_config=provider_config,
            source_effects_config=source_effects_config,
        )

        transcription_config = TranscriptionConfig(
            model=voice_transcription_model,
        )

        return cls(
            synthesis_config=synthesis_config,
            transcription_config=transcription_config,
            output_dir=output_dir,
        )
