# Copyright Sierra
"""Effects-mode overlays on the speech-complexity presets.

The channel/speech intensity modes (owner call 2026-08-25) layer fixed
in-code overrides over a complexity preset. These tests pin the three
contracts: 'regular' modes are a no-op (byte-identical config to a
pre-modes call), 'light'/'heavy' land in the merged effect configs, and
the applied modes are recorded on the sampled config for provenance.
"""

from tau2.data_model.voice import SynthesisConfig
from tau2.user_simulation_voice_presets import (
    CHANNEL_EFFECTS_MODES,
    SPEECH_EFFECTS_MODES,
    sample_voice_config,
)

SEED = 12345


def _sample(**kwargs):
    return sample_voice_config(
        seed=SEED,
        synthesis_config=SynthesisConfig(),
        complexity="regular",
        **kwargs,
    )


def test_regular_modes_are_a_noop():
    baseline = _sample()
    with_modes = _sample(channel_effects_mode="regular", speech_effects_mode="regular")
    assert with_modes.model_dump() == baseline.model_dump()
    assert baseline.channel_effects_mode == "regular"
    assert baseline.speech_effects_mode == "regular"


def test_channel_light_cleans_the_line():
    cfg = _sample(channel_effects_mode="light")
    assert cfg.channel_effects_config.frame_drop_rate == 0.0
    assert not cfg.channel_effects_config.enable_frame_drops
    assert not cfg.speech_effects_config.enable_dynamic_muffling
    assert cfg.speech_effects_config.muffle_probability == 0.0
    assert cfg.channel_effects_mode == "light"


def test_channel_heavy_mode_dict_pins_the_owner_values():
    """The heavy overlay is versioned experiment definition: freeze the
    exact params (2026-09-02 redefinition adds the heavy acoustics; toned
    down same day to 4 bursts/min + 3% drops) so a drive-by edit cannot
    silently change what the C realization means."""
    assert CHANNEL_EFFECTS_MODES["heavy"] == {
        "frame_drop_rate": 0.03,
        "frame_drop_burst_duration_ms": 300,
        "enable_muffling": True,
        "muffle_probability": 0.5,
        "noise_snr_db": 10.0,
        "burst_noise_events_per_minute": 4.0,
        "burst_snr_range_db": (-5.0, 0.0),
    }
    # light/regular carry NO acoustics keys: they leave acoustics at stock.
    for name in ("light", "regular"):
        assert not (
            {"noise_snr_db", "burst_noise_events_per_minute", "burst_snr_range_db"}
            & set(CHANNEL_EFFECTS_MODES[name])
        )
    # S keeps the stock floor by design: no acoustics keys in any speech mode.
    for mode in SPEECH_EFFECTS_MODES.values():
        assert not (
            {"noise_snr_db", "burst_noise_events_per_minute", "burst_snr_range_db"}
            & set(mode)
        )


def test_channel_heavy_degrades_the_line():
    cfg = _sample(channel_effects_mode="heavy")
    mode = CHANNEL_EFFECTS_MODES["heavy"]
    assert cfg.channel_effects_config.frame_drop_rate == mode["frame_drop_rate"]
    assert (
        cfg.channel_effects_config.frame_drop_burst_duration_ms
        == mode["frame_drop_burst_duration_ms"]
    )
    assert cfg.speech_effects_config.enable_dynamic_muffling
    assert cfg.speech_effects_config.muffle_probability == mode["muffle_probability"]


def test_channel_heavy_degrades_the_acoustics():
    """The 2026-09-02 redefinition: channel-heavy raises the noise floor
    and bursts, and the degraded params land in the sampled (and therefore
    stored, via SpeechEnvironment) source effects config — which is what
    distinguishes new-C from legacy frame-drops-only channel-heavy runs."""
    baseline = _sample()
    cfg = _sample(channel_effects_mode="heavy")
    assert cfg.source_effects_config.noise_snr_db == 10.0
    assert cfg.source_effects_config.burst_noise_events_per_minute == 4.0
    assert cfg.source_effects_config.burst_snr_range_db == (-5.0, 0.0)
    # The stock floor really is different (guards the legacy/new distinction).
    assert baseline.source_effects_config.noise_snr_db == 15.0
    assert baseline.source_effects_config.burst_noise_events_per_minute == 1.0
    assert baseline.source_effects_config.burst_snr_range_db != (-5.0, 0.0)


def test_channel_mode_leaves_speech_behavior_alone():
    baseline = _sample()
    cfg = _sample(channel_effects_mode="heavy")
    assert cfg.enable_interruptions == baseline.enable_interruptions
    assert cfg.use_llm_backchannel == baseline.use_llm_backchannel
    assert (
        cfg.speech_effects_config.speech_insert_events_per_minute
        == baseline.speech_effects_config.speech_insert_events_per_minute
    )


def test_speech_light_makes_a_patient_caller():
    cfg = _sample(speech_effects_mode="light")
    assert cfg.speech_effects_config.speech_insert_events_per_minute == 0.0
    assert not cfg.speech_effects_config.enable_vocal_tics
    assert not cfg.speech_effects_config.enable_non_directed_phrases
    assert not cfg.use_llm_backchannel
    assert not cfg.enable_interruptions
    assert cfg.persona_config.interrupt_tendency.value == "waits"
    assert cfg.speech_effects_mode == "light"


def test_speech_heavy_makes_a_chatty_caller():
    cfg = _sample(speech_effects_mode="heavy")
    mode = SPEECH_EFFECTS_MODES["heavy"]
    assert (
        cfg.speech_effects_config.speech_insert_events_per_minute
        == mode["speech_insert_events_per_minute"]
    )
    assert cfg.enable_interruptions
    assert cfg.persona_config.interrupt_tendency.value == "interrupts"


def test_speech_mode_leaves_the_line_alone():
    baseline = _sample()
    cfg = _sample(speech_effects_mode="light")
    assert (
        cfg.channel_effects_config.frame_drop_rate
        == baseline.channel_effects_config.frame_drop_rate
    )
    assert (
        cfg.speech_effects_config.enable_dynamic_muffling
        == baseline.speech_effects_config.enable_dynamic_muffling
    )


def test_run_info_records_the_effects_modes():
    from tau2.data_model.simulation import AudioNativeConfig, VoiceRunConfig
    from tau2.runner.helpers import get_info

    config = VoiceRunConfig(
        domain="mock",
        task_set_name="mock",
        audio_native_config=AudioNativeConfig(),
        channel_effects_mode="heavy",
        speech_effects_mode="light",
    )
    info = get_info(config)
    assert info.channel_effects_mode == "heavy"
    assert info.speech_effects_mode == "light"


def test_run_level_persona_config_respects_speech_mode():
    """The run-level PersonaConfig (which shadows the sampled one in
    build_voice_user for plain English runs) must carry the mode-adjusted
    interrupt tendency — it is what gates interruptions in the simulator."""
    from tau2.data_model.simulation import AudioNativeConfig, VoiceRunConfig
    from tau2.runner.batch import run_level_user_persona_config

    def cfg(**kw):
        return VoiceRunConfig(
            domain="mock",
            task_set_name="mock",
            audio_native_config=AudioNativeConfig(),
            **kw,
        )

    assert (
        run_level_user_persona_config(
            cfg(speech_effects_mode="light")
        ).interrupt_tendency.value
        == "waits"
    )
    assert (
        run_level_user_persona_config(
            cfg(speech_effects_mode="heavy")
        ).interrupt_tendency.value
        == "interrupts"
    )
    assert (
        run_level_user_persona_config(cfg()).interrupt_tendency.value
        == "interrupts"  # regular preset value, unchanged
    )


def test_modes_do_not_touch_persona_or_noise_selection():
    """Modes change effect INTENSITY, never the sampled identity: persona,
    environment, and noise-file selection are byte-identical under any mode
    combination. Channel-heavy changes acoustics LEVELS (since 2026-09-02)
    but never which files play."""
    baseline = _sample()
    cfg = _sample(channel_effects_mode="heavy", speech_effects_mode="heavy")
    assert cfg.persona_name == baseline.persona_name
    assert cfg.background_noise_file == baseline.background_noise_file
    assert cfg.burst_noise_files == baseline.burst_noise_files
    assert cfg.environment == baseline.environment
    # Exactly the three redefined acoustics params differ; nothing else.
    assert cfg.source_effects_config.model_dump() == {
        **baseline.source_effects_config.model_dump(),
        "noise_snr_db": 10.0,
        "burst_noise_events_per_minute": 4.0,
        "burst_snr_range_db": (-5.0, 0.0),
    }


def test_speech_and_light_modes_leave_acoustics_at_stock():
    """S keeps the stock floor by design, and channel-light/regular leave
    acoustics untouched — only channel-heavy is environment-heavy."""
    baseline = _sample()
    for kwargs in (
        {"speech_effects_mode": "heavy"},
        {"speech_effects_mode": "light"},
        {"channel_effects_mode": "light"},
    ):
        cfg = _sample(**kwargs)
        assert (
            cfg.source_effects_config.model_dump()
            == baseline.source_effects_config.model_dump()
        ), kwargs
