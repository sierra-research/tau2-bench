# Copyright Sierra
"""User simulation voice complexity presets.

These presets control both:
1. Audio effects on the user simulator's synthesized speech (noise, dropouts, etc.)
2. User persona/behavior (verbosity, interrupt tendency)

The complexity level represents how challenging the user is for the agent to handle.
Used by run.py and audio effects scheduler.
"""

import json
import random
from pathlib import Path
from typing import Optional

from loguru import logger
from pydantic import BaseModel, Field

from tau2.data_model.audio_effects import (
    ChannelEffectsConfig,
    SourceEffectsConfig,
    SpeechEffectsConfig,
)
from tau2.data_model.persona import InterruptTendency, PersonaConfig, Verbosity
from tau2.data_model.voice import SampledVoiceConfig, SpeechComplexity, SynthesisConfig
from tau2.data_model.voice_personas import (
    CONTROL_PERSONA_NAMES,
    REGULAR_PERSONA_NAMES,
    get_persona_name_by_voice_id,
)
from tau2.voice_config import BACKGROUND_NOISE_CONTINUOUS_DIR, BURST_NOISE_DIR

# Seed offsets for each complexity level to ensure different voice selections
# Control and regular offsets are fixed for backward compatibility
COMPLEXITY_SEED_OFFSETS: dict[str, int] = {
    "control": 0,
    "regular": 1000000,
    "control_audio": 2000000,
    "control_accents": 3000000,
    "control_behavior": 4000000,
    "control_audio_accents": 5000000,
    "control_audio_behavior": 6000000,
    "control_accents_behavior": 7000000,
}

# ============================================================================
# Environment Presets (Indoor/Outdoor)
# ============================================================================

# Display names for background noise files (used in visualizations)
BACKGROUND_NOISE_DISPLAY_NAMES = {
    "people_talking.wav": "People Talking",
    "medium_size_room_tv_news_iphone_mic.wav": "TV News",
    "busy_street_iphone_mic.wav": "Busy Street",
    "street_and_metro_station_iphone_mic.wav": "Street & Metro",
}

# Display names for burst noise files (used in visualizations)
BURST_NOISE_DISPLAY_NAMES = {
    "ringing_phone.wav": "Ringing Phone",
    "dog_bark.wav": "Dog Bark",
    "car_horn.wav": "Car Horn",
    "engine_idling.wav": "Engine Idling",
    "siren.wav": "Siren",
}

# Environment presets define coherent combinations of background noise and burst sounds
ENVIRONMENT_PRESETS = {
    "indoor": {
        "background_noise_files": [
            "people_talking.wav",
            "medium_size_room_tv_news_iphone_mic.wav",
        ],
        "burst_noise_files": [
            "ringing_phone.wav",
            "dog_bark.wav",
        ],
    },
    "outdoor": {
        "background_noise_files": [
            "busy_street_iphone_mic.wav",
            "street_and_metro_station_iphone_mic.wav",
        ],
        "burst_noise_files": [
            "car_horn.wav",
            "engine_idling.wav",
            "siren.wav",
        ],
    },
}

ENVIRONMENT_NAMES = list(ENVIRONMENT_PRESETS.keys())

# ============================================================================
# Complexity Preset Definitions
# ============================================================================

# CONTROL complexity preset: Clean speech, no effects, American accents, patient user.
CONTROL_CONFIG = {
    # Persona selection
    "persona_names": CONTROL_PERSONA_NAMES,
    # Environment - none for control
    "environment": None,
    # Background noise - disabled
    "enable_background_noise": False,
    # Burst noise - disabled
    "enable_burst_noise": False,
    "burst_noise_events_per_minute": 0.0,
    # Frame drops - disabled (GE model params)
    "frame_drop_rate": 0.0,
    "frame_drop_burst_duration_ms": 200,
    # Telephony - always enabled (G.711 μ-law 8kHz compression)
    "telephony_enabled": True,
    # Speech inserts - disabled
    "speech_insert_events_per_minute": 0.0,
    "enable_vocal_tics": False,
    "enable_non_directed_phrases": False,
    # Muffling - disabled
    "enable_muffling": False,
    # Backchanneling and interruptions - disabled
    "backchannel_min_threshold": None,  # None = disabled
    "use_llm_backchannel": False,  # Disable LLM-based backchanneling
    "enable_interruptions": False,
    # User persona - patient, waits for agent to finish
    "verbosity": "minimal",
    "interrupt_tendency": "waits",
}

# REGULAR complexity preset: Realistic audio with fixed effects, diverse accents, interrupts.
REGULAR_CONFIG = {
    # Persona selection
    "persona_names": REGULAR_PERSONA_NAMES,
    # Environment - will be selected deterministically per task
    "environment": "auto",  # Selected based on task hash
    # Background noise - always on
    "enable_background_noise": True,
    # Burst noise - enabled with fixed rate
    "enable_burst_noise": True,
    "burst_noise_events_per_minute": 1.0,
    # Frame drops - enabled with GE model params (1% loss rate, 100ms bursts)
    "frame_drop_rate": 0.02,
    "frame_drop_burst_duration_ms": 100,
    # Telephony - enabled (G.711 μ-law 8kHz compression)
    "telephony_enabled": True,
    # Speech inserts - enabled with fixed rate
    "speech_insert_events_per_minute": 0.7,
    "enable_vocal_tics": True,
    "enable_non_directed_phrases": True,
    # Muffling - always on
    "enable_muffling": True,
    # Backchanneling and interruptions
    "backchannel_min_threshold": 3,
    "use_llm_backchannel": True,  # Enable LLM-based backchanneling
    "enable_interruptions": True,
    # User persona - may interrupt the agent
    "verbosity": "minimal",
    "interrupt_tendency": "interrupts",
}

# ============================================================================
# Ablation Presets (Control + specific feature groups)
# ============================================================================

# CONTROL_AUDIO: Control baseline + audio/transmission effects
# Adds: background noise, burst noise, muffling, frame drops
# Keeps: American accents, patient user behavior
CONTROL_AUDIO_CONFIG = {
    **CONTROL_CONFIG,
    # Environment - enables noise file selection
    "environment": "auto",
    # Background noise - enabled
    "enable_background_noise": True,
    # Burst noise - enabled with regular rate
    "enable_burst_noise": True,
    "burst_noise_events_per_minute": 1.0,
    # Frame drops - enabled with regular params
    "frame_drop_rate": 0.005,
    "frame_drop_burst_duration_ms": 100,
    # Muffling - enabled
    "enable_muffling": True,
}

# CONTROL_ACCENTS: Control baseline + diverse accent personas
# Adds: diverse accents (REGULAR_PERSONA_NAMES)
# Keeps: clean audio, patient user behavior
CONTROL_ACCENTS_CONFIG = {
    **CONTROL_CONFIG,
    # Use regular personas (diverse accents)
    "persona_names": REGULAR_PERSONA_NAMES,
}

# CONTROL_BEHAVIOR: Control baseline + user interaction patterns
# Adds: interruptions, backchannels, speech inserts (vocal tics, non-directed phrases)
# Keeps: American accents, clean audio
CONTROL_BEHAVIOR_CONFIG = {
    **CONTROL_CONFIG,
    # Speech inserts - enabled with regular rate
    "speech_insert_events_per_minute": 0.7,
    "enable_vocal_tics": True,
    "enable_non_directed_phrases": True,
    # Backchanneling - enabled with regular settings
    "backchannel_min_threshold": 3,
    "use_llm_backchannel": True,
    # Interruptions - enabled
    "enable_interruptions": True,
    # User persona - may interrupt the agent
    "interrupt_tendency": "interrupts",
}

# ============================================================================
# Pairwise Ablation Presets (Control + two feature groups)
# ============================================================================

# CONTROL_AUDIO_ACCENTS: Control baseline + audio effects + diverse accents
# Adds: background noise, burst noise, muffling, frame drops, diverse accents
# Keeps: patient user behavior
CONTROL_AUDIO_ACCENTS_CONFIG = {
    **CONTROL_AUDIO_CONFIG,
    # Use regular personas (diverse accents)
    "persona_names": REGULAR_PERSONA_NAMES,
}

# CONTROL_AUDIO_BEHAVIOR: Control baseline + audio effects + user behavior
# Adds: background noise, burst noise, muffling, frame drops, interruptions, backchannels
# Keeps: American accents
CONTROL_AUDIO_BEHAVIOR_CONFIG = {
    **CONTROL_AUDIO_CONFIG,
    # Speech inserts - enabled with regular rate
    "speech_insert_events_per_minute": 0.7,
    "enable_vocal_tics": True,
    "enable_non_directed_phrases": True,
    # Backchanneling - enabled with regular settings
    "backchannel_min_threshold": 3,
    "use_llm_backchannel": True,
    # Interruptions - enabled
    "enable_interruptions": True,
    # User persona - may interrupt the agent
    "interrupt_tendency": "interrupts",
}

# CONTROL_ACCENTS_BEHAVIOR: Control baseline + diverse accents + user behavior
# Adds: diverse accents, interruptions, backchannels, speech inserts
# Keeps: clean audio
CONTROL_ACCENTS_BEHAVIOR_CONFIG = {
    **CONTROL_BEHAVIOR_CONFIG,
    # Use regular personas (diverse accents)
    "persona_names": REGULAR_PERSONA_NAMES,
}


COMPLEXITY_CONFIGS: dict[SpeechComplexity, dict] = {
    "control": CONTROL_CONFIG,
    "regular": REGULAR_CONFIG,
    "control_audio": CONTROL_AUDIO_CONFIG,
    "control_accents": CONTROL_ACCENTS_CONFIG,
    "control_behavior": CONTROL_BEHAVIOR_CONFIG,
    "control_audio_accents": CONTROL_AUDIO_ACCENTS_CONFIG,
    "control_audio_behavior": CONTROL_AUDIO_BEHAVIOR_CONFIG,
    "control_accents_behavior": CONTROL_ACCENTS_BEHAVIOR_CONFIG,
}


# ============================================================================
# Main Sampling Function
# ============================================================================


def sample_voice_config(
    seed: int,
    synthesis_config: SynthesisConfig,
    complexity: SpeechComplexity = "regular",
) -> SampledVoiceConfig:
    """Sample a complete voice configuration from complexity presets.

    This function:
    1. Looks up the complexity preset
    2. Selects persona deterministically from the preset's persona list (unless provided
       via synthesis_config.provider_config.persona_name)
    3. For regular mode, selects environment (indoor/outdoor) and associated audio files
    4. Creates configs with complexity settings applied
    5. Creates PersonaConfig from complexity settings

    Each complexity level uses a different seed offset to ensure different persona
    selections across complexity levels for the same base seed.

    Args:
        seed: Random seed for reproducibility.
        synthesis_config: Base synthesis configuration with effect configs.
        complexity: Speech environment complexity level ("control" or "regular").

    Returns:
        SampledVoiceConfig with all configs instantiated and complexity settings applied.
    """
    # Apply complexity-specific seed offset to ensure different selections per complexity
    complexity_seed = seed + COMPLEXITY_SEED_OFFSETS.get(complexity, 0)
    rng = random.Random(complexity_seed)
    preset = COMPLEXITY_CONFIGS[complexity]

    # Get base configs from synthesis_config
    base_channel = synthesis_config.channel_effects_config
    base_source = synthesis_config.source_effects_config
    base_speech = synthesis_config.speech_effects_config

    # -------------------------------------------------------------------------
    # Sample persona (simulation-level, same speaker throughout)
    # -------------------------------------------------------------------------
    provider_config = synthesis_config.provider_config
    voice_id = provider_config.voice_id if provider_config else None
    persona_name = get_persona_name_by_voice_id(voice_id) if voice_id else None
    if not persona_name:
        persona_names = preset.get("persona_names", CONTROL_PERSONA_NAMES)
        persona_name = rng.choice(persona_names)

    # -------------------------------------------------------------------------
    # Select environment and audio files
    # -------------------------------------------------------------------------
    environment: Optional[str] = None
    background_noise_file: Optional[str] = None
    burst_noise_files: list[str] = []

    if preset.get("enable_background_noise") or preset.get("enable_burst_noise"):
        # Select environment deterministically based on seed
        env_setting = preset.get("environment")
        if env_setting == "auto":
            # Deterministic environment selection based on seed
            environment = ENVIRONMENT_NAMES[seed % len(ENVIRONMENT_NAMES)]
        elif env_setting in ENVIRONMENT_NAMES:
            environment = env_setting

        if environment:
            env_preset = ENVIRONMENT_PRESETS[environment]

            # Select background noise file
            if preset.get("enable_background_noise"):
                bg_files = env_preset.get("background_noise_files", [])
                if bg_files:
                    # Select one background noise file deterministically
                    bg_filename = rng.choice(bg_files)
                    bg_path = BACKGROUND_NOISE_CONTINUOUS_DIR / bg_filename
                    if bg_path.exists():
                        background_noise_file = bg_filename
                    else:
                        logger.warning(f"Background noise file not found: {bg_path}")

            # Get burst noise files (all files for the environment)
            if preset.get("enable_burst_noise"):
                burst_filenames = env_preset.get("burst_noise_files", [])
                for burst_filename in burst_filenames:
                    burst_path = BURST_NOISE_DIR / burst_filename
                    if burst_path.exists():
                        burst_noise_files.append(burst_filename)
                    else:
                        logger.warning(f"Burst noise file not found: {burst_path}")

    # -------------------------------------------------------------------------
    # Create merged ChannelEffectsConfig (uses GE model for frame drops)
    # -------------------------------------------------------------------------
    frame_drop_rate = preset.get("frame_drop_rate", base_channel.frame_drop_rate)
    merged_channel = ChannelEffectsConfig(
        enable_frame_drops=frame_drop_rate > 0,
        frame_drop_rate=frame_drop_rate,
        frame_drop_burst_duration_ms=preset.get(
            "frame_drop_burst_duration_ms", base_channel.frame_drop_burst_duration_ms
        ),
        frame_drop_count=base_channel.frame_drop_count,
        frame_drop_duration_ms=base_channel.frame_drop_duration_ms,
    )

    # -------------------------------------------------------------------------
    # Create merged SourceEffectsConfig
    # -------------------------------------------------------------------------
    merged_source = SourceEffectsConfig(
        enable_background_noise=preset.get("enable_background_noise", False),
        noise_snr_db=base_source.noise_snr_db,
        noise_snr_drift_db=base_source.noise_snr_drift_db,
        noise_variation_speed=base_source.noise_variation_speed,
        enable_burst_noise=preset.get("enable_burst_noise", False),
        burst_noise_events_per_minute=preset.get(
            "burst_noise_events_per_minute", base_source.burst_noise_events_per_minute
        ),
        burst_snr_range_db=base_source.burst_snr_range_db,
    )

    # -------------------------------------------------------------------------
    # Create merged SpeechEffectsConfig
    # -------------------------------------------------------------------------
    merged_speech = SpeechEffectsConfig(
        enable_dynamic_muffling=preset.get("enable_muffling", False),
        muffle_probability=base_speech.muffle_probability
        if preset.get("enable_muffling")
        else 0.0,
        muffle_segment_count=base_speech.muffle_segment_count,
        muffle_segment_duration_ms=base_speech.muffle_segment_duration_ms,
        muffle_cutoff_freq=base_speech.muffle_cutoff_freq,
        muffle_transition_ms=base_speech.muffle_transition_ms,
        enable_vocal_tics=preset.get(
            "enable_vocal_tics", base_speech.enable_vocal_tics
        ),
        vocal_tics=base_speech.vocal_tics,
        min_words_for_vocal_tics=base_speech.min_words_for_vocal_tics,
        enable_non_directed_phrases=preset.get(
            "enable_non_directed_phrases", base_speech.enable_non_directed_phrases
        ),
        non_directed_phrases=base_speech.non_directed_phrases,
        speech_insert_events_per_minute=preset.get(
            "speech_insert_events_per_minute",
            base_speech.speech_insert_events_per_minute,
        ),
    )

    # -------------------------------------------------------------------------
    # Create PersonaConfig from complexity settings
    # -------------------------------------------------------------------------
    persona_config = PersonaConfig(
        verbosity=Verbosity(preset["verbosity"]),
        interrupt_tendency=InterruptTendency(preset["interrupt_tendency"]),
    )

    return SampledVoiceConfig(
        persona_name=persona_name,
        background_noise_file=background_noise_file,
        burst_noise_files=burst_noise_files,
        environment=environment,
        backchannel_min_threshold=preset.get("backchannel_min_threshold"),
        use_llm_backchannel=preset.get("use_llm_backchannel", True),
        enable_interruptions=preset.get("enable_interruptions", False),
        telephony_enabled=preset.get("telephony_enabled", True),
        channel_effects_config=merged_channel,
        source_effects_config=merged_source,
        speech_effects_config=merged_speech,
        persona_config=persona_config,
        complexity=complexity,
    )


# ============================================================================
# Task Voice Config Generation and Loading
# ============================================================================

# Complexity levels to generate
# Control and regular are the main conditions; ablations are for analysis
COMPLEXITY_LEVELS: list[SpeechComplexity] = [
    "control",
    "regular",
    "control_audio",
    "control_accents",
    "control_behavior",
    "control_audio_accents",
    "control_audio_behavior",
    "control_accents_behavior",
]


class TaskVoiceConfigsByComplexity(BaseModel):
    """Voice configs for a single task across complexity levels.

    Uses a dict for extensibility - supports any complexity level defined in
    COMPLEXITY_LEVELS. Backward compatible with older JSON files that had
    explicit control/regular fields.
    """

    configs: dict[str, SampledVoiceConfig] = Field(
        default_factory=dict,
        description="Mapping from complexity level to voice config",
    )

    def get(self, complexity: SpeechComplexity) -> Optional[SampledVoiceConfig]:
        """Get config for a specific complexity level."""
        return self.configs.get(complexity)

    # Backward compatibility: support old format with explicit control/regular fields
    def __init__(self, **data):
        # Handle old format: convert explicit fields to configs dict
        if "configs" not in data and ("control" in data or "regular" in data):
            configs = {}
            if "control" in data:
                configs["control"] = data.pop("control")
            if "regular" in data:
                configs["regular"] = data.pop("regular")
            data["configs"] = configs
        super().__init__(**data)


class TaskVoiceConfigs(BaseModel):
    """Pre-sampled voice configurations for a set of tasks.

    Contains configs for all complexity levels (control, regular) for each task.
    This allows sampling once and reusing the same voice config for each task
    across multiple experiments for reproducibility.
    """

    # Metadata
    base_seed: int = Field(description="Base seed used for sampling")

    # Task ID -> configs for all complexity levels
    configs: dict[str, TaskVoiceConfigsByComplexity] = Field(
        description="Mapping from task ID to voice configs for all complexity levels"
    )

    def get_config(
        self, task_id: str, complexity: SpeechComplexity
    ) -> Optional[SampledVoiceConfig]:
        """Get the voice config for a specific task and complexity."""
        task_configs = self.configs.get(task_id)
        if task_configs is None:
            return None
        return task_configs.get(complexity)


def generate_task_voice_configs(
    task_set_name: str,
    base_seed: int,
    synthesis_config: Optional[SynthesisConfig] = None,
    task_split_name: Optional[str] = None,
) -> TaskVoiceConfigs:
    """Generate pre-sampled voice configs for all tasks in a task set.

    Generates configs for all complexity levels (control, regular) for each task.
    Each task gets a deterministic seed based on base_seed + hash(task.id).

    Args:
        task_set_name: Name of the task set (e.g., "telecom", "airline").
        base_seed: Base random seed. Each task gets seed = base_seed + hash(task.id) % 1000000.
        synthesis_config: Base synthesis config. If None, uses defaults.
        task_split_name: Optional task split to filter tasks.

    Returns:
        TaskVoiceConfigs with pre-sampled configs for all tasks and complexity levels.
    """
    # Import here to avoid circular imports
    from tau2.run import load_tasks

    if synthesis_config is None:
        synthesis_config = SynthesisConfig()

    tasks = load_tasks(task_set_name=task_set_name, task_split_name=task_split_name)

    configs: dict[str, TaskVoiceConfigsByComplexity] = {}
    for task in tasks:
        # Deterministic seed per task (same logic as run_task)
        task_seed = base_seed + hash(task.id) % 1000000

        # Sample for each complexity level
        task_configs: dict[str, SampledVoiceConfig] = {}
        for complexity in COMPLEXITY_LEVELS:
            sampled = sample_voice_config(
                seed=task_seed,
                synthesis_config=synthesis_config,
                complexity=complexity,
            )
            task_configs[complexity] = sampled
            logger.debug(
                f"Sampled voice config for task {task.id} ({complexity}): "
                f"persona={sampled.persona_name}, env={sampled.environment}"
            )

        configs[task.id] = TaskVoiceConfigsByComplexity(configs=task_configs)

    logger.info(
        f"Generated voice configs for {len(configs)} tasks "
        f"(all complexity levels, base_seed={base_seed})"
    )

    return TaskVoiceConfigs(
        base_seed=base_seed,
        configs=configs,
    )


def save_task_voice_configs(
    task_voice_configs: TaskVoiceConfigs,
    output_path: Path,
) -> None:
    """Save task voice configs to a JSON file.

    Args:
        task_voice_configs: The configs to save.
        output_path: Path to save the JSON file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(task_voice_configs.model_dump(mode="json"), f, indent=2)
    logger.info(f"Saved task voice configs to {output_path}")


def load_task_voice_configs(input_path: Path) -> TaskVoiceConfigs:
    """Load task voice configs from a JSON file.

    Args:
        input_path: Path to the JSON file.

    Returns:
        TaskVoiceConfigs loaded from the file.
    """
    with open(input_path) as f:
        data = json.load(f)
    configs = TaskVoiceConfigs.model_validate(data)
    logger.info(
        f"Loaded task voice configs from {input_path}: "
        f"{len(configs.configs)} tasks (all complexity levels)"
    )
    return configs


def get_task_voice_configs_path(domain: str) -> Path:
    """Get the default path for a domain's tasks_voice.json file.

    Args:
        domain: Domain name (e.g., "telecom", "airline").

    Returns:
        Path to the tasks_voice.json file.
    """
    from tau2.utils.utils import DATA_DIR

    return DATA_DIR / "tau2" / "domains" / domain / "tasks_voice.json"


def get_or_load_task_voice_config(
    domain: str,
    task_id: str,
    task_seed: int,
    complexity: SpeechComplexity,
    synthesis_config: SynthesisConfig,
) -> SampledVoiceConfig:
    """Get voice config for a task, loading from file if available.

    Args:
        domain: Domain name.
        task_id: Task ID to look up.
        task_seed: Seed to use if sampling is needed.
        complexity: Speech complexity level.
        synthesis_config: Base synthesis config for sampling.

    Returns:
        SampledVoiceConfig for the task.
    """
    config_path = get_task_voice_configs_path(domain)

    if config_path.exists():
        try:
            configs = load_task_voice_configs(config_path)
            config = configs.get_config(task_id, complexity)
            if config is not None:
                logger.info(
                    f"Using pre-sampled voice config for task {task_id} "
                    f"(complexity={complexity}) from {config_path}"
                )
                return config
            else:
                logger.warning(
                    f"Task {task_id} not found in {config_path}, sampling on the fly"
                )
        except Exception as e:
            logger.warning(
                f"Failed to load voice configs from {config_path}: {e}, "
                "sampling on the fly"
            )

    # No pre-sampled config available, sample on the fly
    logger.warning(
        f"No pre-sampled voice config for task {task_id}. "
        f"Sampling on the fly (seed={task_seed}, complexity={complexity}). "
        f"Run `python -m tau2.user_simulation_voice_presets {domain}` to generate."
    )
    return sample_voice_config(
        seed=task_seed,
        synthesis_config=synthesis_config,
        complexity=complexity,
    )


# ============================================================================
# CLI Entry Point
# ============================================================================


def generate_task_voice_configs_for_levels(
    task_set_name: str,
    base_seed: int,
    complexity_levels: list[SpeechComplexity],
    synthesis_config: Optional[SynthesisConfig] = None,
    task_split_name: Optional[str] = None,
) -> TaskVoiceConfigs:
    """Generate pre-sampled voice configs for specific complexity levels only.

    Args:
        task_set_name: Name of the task set (e.g., "telecom", "airline").
        base_seed: Base random seed.
        complexity_levels: List of complexity levels to generate.
        synthesis_config: Base synthesis config. If None, uses defaults.
        task_split_name: Optional task split to filter tasks.

    Returns:
        TaskVoiceConfigs with pre-sampled configs for specified complexity levels.
    """
    from tau2.run import load_tasks

    if synthesis_config is None:
        synthesis_config = SynthesisConfig()

    tasks = load_tasks(task_set_name=task_set_name, task_split_name=task_split_name)

    configs: dict[str, TaskVoiceConfigsByComplexity] = {}
    for task in tasks:
        task_seed = base_seed + hash(task.id) % 1000000

        task_configs: dict[str, SampledVoiceConfig] = {}
        for complexity in complexity_levels:
            sampled = sample_voice_config(
                seed=task_seed,
                synthesis_config=synthesis_config,
                complexity=complexity,
            )
            task_configs[complexity] = sampled
            logger.debug(
                f"Sampled voice config for task {task.id} ({complexity}): "
                f"persona={sampled.persona_name}, env={sampled.environment}"
            )

        configs[task.id] = TaskVoiceConfigsByComplexity(configs=task_configs)

    logger.info(
        f"Generated voice configs for {len(configs)} tasks "
        f"(levels: {complexity_levels}, base_seed={base_seed})"
    )

    return TaskVoiceConfigs(base_seed=base_seed, configs=configs)


def merge_task_voice_configs(
    existing: TaskVoiceConfigs,
    new_configs: TaskVoiceConfigs,
) -> TaskVoiceConfigs:
    """Merge new configs into existing configs without overwriting.

    For each task, adds new complexity levels while preserving existing ones.

    Args:
        existing: Existing task voice configs.
        new_configs: New configs to merge in.

    Returns:
        Merged TaskVoiceConfigs.
    """
    merged_configs: dict[str, TaskVoiceConfigsByComplexity] = {}

    # Get all task IDs from both
    all_task_ids = set(existing.configs.keys()) | set(new_configs.configs.keys())

    for task_id in all_task_ids:
        existing_task = existing.configs.get(task_id)
        new_task = new_configs.configs.get(task_id)

        merged_task_configs: dict[str, SampledVoiceConfig] = {}

        # Add existing configs first
        if existing_task:
            merged_task_configs.update(existing_task.configs)

        # Add new configs (only if not already present)
        if new_task:
            for level, config in new_task.configs.items():
                if level not in merged_task_configs:
                    merged_task_configs[level] = config
                    logger.debug(f"Added {level} config for task {task_id}")
                else:
                    logger.debug(
                        f"Skipping {level} for task {task_id} (already exists)"
                    )

        merged_configs[task_id] = TaskVoiceConfigsByComplexity(
            configs=merged_task_configs
        )

    logger.info(
        f"Merged configs: {len(merged_configs)} tasks, preserving existing levels"
    )

    return TaskVoiceConfigs(base_seed=existing.base_seed, configs=merged_configs)


# Ablation-only complexity levels (excludes control and regular)
# Single-feature ablations (one feature group added to control)
SINGLE_ABLATION_LEVELS: list[SpeechComplexity] = [
    "control_audio",
    "control_accents",
    "control_behavior",
]

# Pairwise ablations (two feature groups added to control)
PAIRWISE_ABLATION_LEVELS: list[SpeechComplexity] = [
    "control_audio_accents",
    "control_audio_behavior",
    "control_accents_behavior",
]

# All ablation levels combined
ABLATION_LEVELS: list[SpeechComplexity] = (
    SINGLE_ABLATION_LEVELS + PAIRWISE_ABLATION_LEVELS
)


def main():
    """CLI entry point for generating task voice configs."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate pre-sampled voice configs for task sets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate voice configs for telecom domain (all levels)
  python -m tau2.user_simulation_voice_presets telecom

  # Generate ONLY the 3 pairwise ablation configs (safest option for new ablations)
  # This will NOT touch control, regular, or single-feature ablations
  python -m tau2.user_simulation_voice_presets telecom --pairwise-only

  # Generate all ablation configs (single + pairwise), merging with existing control/regular
  python -m tau2.user_simulation_voice_presets telecom --ablations-only

  # Generate specific complexity levels only
  python -m tau2.user_simulation_voice_presets telecom --complexity control_audio control_behavior

  # Generate for multiple domains
  python -m tau2.user_simulation_voice_presets telecom airline retail

  # Generate with custom seed
  python -m tau2.user_simulation_voice_presets telecom --seed 123

Complexity levels:
  - control: Clean baseline (no effects, American accents, patient user)
  - regular: Full realistic conditions (all effects enabled)
  Single-feature ablations:
  - control_audio: Control + audio/transmission effects (noise, muffling, frame drops)
  - control_accents: Control + diverse accent personas
  - control_behavior: Control + user behavior effects (interrupts, backchannels, speech inserts)
  Pairwise ablations:
  - control_audio_accents: Control + audio effects + diverse accents
  - control_audio_behavior: Control + audio effects + user behavior
  - control_accents_behavior: Control + diverse accents + user behavior
""",
    )
    parser.add_argument(
        "domains",
        nargs="+",
        help="Domain names to generate voice configs for (e.g., telecom, airline)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (default: 42)",
    )
    parser.add_argument(
        "--task-split",
        type=str,
        default=None,
        help="Optional task split name to filter tasks",
    )
    parser.add_argument(
        "--complexity",
        nargs="+",
        choices=COMPLEXITY_LEVELS,
        default=None,
        help="Specific complexity levels to generate (default: all)",
    )
    parser.add_argument(
        "--ablations-only",
        action="store_true",
        help="Generate all ablation configs (single + pairwise) "
        "and merge with existing file, preserving control/regular",
    )
    parser.add_argument(
        "--pairwise-only",
        action="store_true",
        help="Generate ONLY the 3 pairwise ablation configs "
        "(control_audio_accents, control_audio_behavior, control_accents_behavior) "
        "and merge with existing file, preserving ALL other configs",
    )

    args = parser.parse_args()

    # Determine which complexity levels to generate
    if args.pairwise_only:
        levels_to_generate = PAIRWISE_ABLATION_LEVELS
        merge_mode = True
    elif args.ablations_only:
        levels_to_generate = ABLATION_LEVELS
        merge_mode = True
    elif args.complexity:
        levels_to_generate = args.complexity
        merge_mode = False
    else:
        levels_to_generate = COMPLEXITY_LEVELS
        merge_mode = False

    for domain in args.domains:
        print(f"\n{'=' * 60}")
        print(f"Generating voice configs for domain: {domain}")
        print(f"  Complexity levels: {', '.join(levels_to_generate)}")
        print(f"  Seed: {args.seed}")
        if merge_mode:
            print(f"  Mode: MERGE (preserving existing control/regular)")
        print(f"{'=' * 60}")

        try:
            output_path = get_task_voice_configs_path(domain)

            # Generate new configs for specified levels
            new_configs = generate_task_voice_configs_for_levels(
                task_set_name=domain,
                base_seed=args.seed,
                complexity_levels=levels_to_generate,
                task_split_name=args.task_split,
            )

            # If merge mode and file exists, merge with existing
            if merge_mode and output_path.exists():
                print(f"  Loading existing configs from {output_path}...")
                existing_configs = load_task_voice_configs(output_path)
                final_configs = merge_task_voice_configs(existing_configs, new_configs)
            else:
                final_configs = new_configs

            save_task_voice_configs(final_configs, output_path)

            # Count levels per task
            sample_task = next(iter(final_configs.configs.values()), None)
            levels_count = len(sample_task.configs) if sample_task else 0

            print(f"\n✓ Generated configs for {len(final_configs.configs)} tasks")
            print(f"✓ Complexity levels per task: {levels_count}")
            print(f"✓ Saved to: {output_path}")

        except Exception as e:
            print(f"\n✗ Error generating configs for {domain}: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
