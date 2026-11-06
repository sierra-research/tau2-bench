"""Tests for food domain store/channel voice presets and upstream voice integration."""

import random
import wave

import numpy as np
import pytest

from tau2.data_model.run_config import AudioNativeConfig
from tau2.data_model.simulation import VoiceRunConfig
from tau2.data_model.tasks import Action, StructuredUserInstructions, UserScenario
from tau2.data_model.voice import SynthesisConfig
from tau2.domains.food.environment import get_environment, get_tasks
from tau2.domains.food.voice_presets import (
    FOOD_BACKGROUND_SPEECH_EVENTS_PER_MINUTE,
    FOOD_DRIVE_THRU_ACOUSTIC_BURSTS,
    FOOD_DRIVE_THRU_BG_FILES,
    FOOD_HEAVY_BACKGROUND_SPEECH_EVENTS_PER_MINUTE,
    FOOD_IN_DOMAIN_DISTRACTORS,
    FOOD_IN_DOMAIN_SPEECH_SLOTS,
    FOOD_MOBILE_ACOUSTIC_BURSTS,
    FOOD_MOBILE_BG_FILES,
    FOOD_OOD_SPEECH_SLOTS,
    FOOD_OUT_OF_DOMAIN_DRIVE_THRU_SPEECH,
    FOOD_OUT_OF_DOMAIN_MOBILE_SPEECH,
    _expand_slot_bucket,
    apply_food_voice_preset,
    build_food_background_speech_pool,
    get_food_in_domain_distractor_files,
    get_food_task_store_and_channel,
)
from tau2.multilingual.runtime import task_context
from tau2.registry import registry
from tau2.runner.build import build_voice_user
from tau2.user_simulation_voice_presets import (
    CHANNEL_EFFECTS_MODES,
    ENVIRONMENT_PRESETS,
    get_or_load_task_voice_config,
    get_task_voice_configs_path,
    load_task_voice_configs,
    sample_voice_config,
)
from tau2.voice.utils.utils import BURST_NOISE_FILES, CONTINUOUS_NOISE_FILES
from tau2.voice_config import (
    BACKGROUND_NOISE_CONTINUOUS_DIR,
    BACKGROUND_SPEECH_SNR_RANGE_DB,
    BURST_NOISE_DIR,
    BURST_NOISE_EVENTS_PER_MINUTE,
    BURST_SNR_RANGE_DB,
)


def test_food_wav_assets_format_and_default_discovery_isolation():
    continuous_beds = {
        "food/car_cabin_radio_music.wav",
        "food/car_cabin_radio_talk.wav",
        "food/living_room_tv_broadcast.wav",
    }
    for bed_name in continuous_beds:
        bed_path = BACKGROUND_NOISE_CONTINUOUS_DIR / bed_name
        assert bed_path.is_file(), bed_name
        with wave.open(str(bed_path), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000
            assert wf.getnframes() >= 19 * 16000
            samples = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
            assert int(np.sum(np.abs(samples) >= 32767)) == 0

    distractor_subpaths = [
        subpath for specs in FOOD_IN_DOMAIN_DISTRACTORS.values() for subpath, _ in specs
    ]
    assert len(distractor_subpaths) == 27
    food_ood_speech_subpaths = (
        *FOOD_OUT_OF_DOMAIN_DRIVE_THRU_SPEECH,
        *FOOD_OUT_OF_DOMAIN_MOBILE_SPEECH,
    )
    assert len(food_ood_speech_subpaths) == 7

    for subpath in (*distractor_subpaths, *food_ood_speech_subpaths):
        wav_path = BURST_NOISE_DIR / subpath
        assert wav_path.is_file(), subpath
        with wave.open(str(wav_path), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000
            samples = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
            assert len(samples) > 0
            assert int(np.sum(np.abs(samples) >= 32767)) == 0

    for burst_name in (*FOOD_DRIVE_THRU_ACOUSTIC_BURSTS, *FOOD_MOBILE_ACOUSTIC_BURSTS):
        assert not burst_name.startswith("food/"), burst_name
        assert (BURST_NOISE_DIR / burst_name).is_file(), burst_name

    assert sorted((BACKGROUND_NOISE_CONTINUOUS_DIR / "food").glob("*.wav")) == sorted(
        BACKGROUND_NOISE_CONTINUOUS_DIR / name for name in continuous_beds
    )
    assert sorted((BURST_NOISE_DIR / "food").glob("*.wav")) == []
    assert sorted((BURST_NOISE_DIR / "food/out_of_domain").glob("*.wav")) == sorted(
        BURST_NOISE_DIR / name for name in food_ood_speech_subpaths
    )
    assert sorted((BURST_NOISE_DIR / "food/in_domain").glob("*.wav")) == sorted(
        BURST_NOISE_DIR / name for name in distractor_subpaths
    )

    assert not any(
        "food" in p.parts for p in (*CONTINUOUS_NOISE_FILES, *BURST_NOISE_FILES)
    )
    assert not any(name.startswith("food") for name in ENVIRONMENT_PRESETS)


def test_food_tasks_partition_into_store_channel_presets():
    tasks = get_tasks("base")
    channels = [get_food_task_store_and_channel(t)[1] for t in tasks]
    assert channels.count("drive_thru") == 30
    assert channels.count("mobile") == 32

    loaded = load_task_voice_configs(get_task_voice_configs_path("food"))
    assert len(loaded.configs) == len(tasks)

    for idx, task in enumerate(tasks):
        store_id, channel = get_food_task_store_and_channel(task)
        expected_env = f"food_{store_id}_{channel}"
        distractors = get_food_in_domain_distractor_files(task)
        ood_speech = list(
            FOOD_OUT_OF_DOMAIN_DRIVE_THRU_SPEECH
            if channel == "drive_thru"
            else FOOD_OUT_OF_DOMAIN_MOBILE_SPEECH
        )
        acoustic_bursts = list(
            FOOD_DRIVE_THRU_ACOUSTIC_BURSTS
            if channel == "drive_thru"
            else FOOD_MOBILE_ACOUSTIC_BURSTS
        )
        bg_speech_files = build_food_background_speech_pool(task, seed=42 + idx)
        assert (
            len(bg_speech_files) == FOOD_OOD_SPEECH_SLOTS + FOOD_IN_DOMAIN_SPEECH_SLOTS
        )
        assert all(f in ood_speech for f in bg_speech_files[:FOOD_OOD_SPEECH_SLOTS])
        assert all(f in distractors for f in bg_speech_files[FOOD_OOD_SPEECH_SLOTS:])
        assert set(distractors).issubset(set(bg_speech_files[FOOD_OOD_SPEECH_SLOTS:]))
        if channel == "drive_thru":
            assert 2 <= len(distractors) <= 5, task.id
            assert all(
                d.startswith(f"food/in_domain/other_lane_{store_id}_")
                for d in distractors
            )
        else:
            assert 1 <= len(distractors) <= 4, task.id
            assert all(
                d.startswith(f"food/in_domain/household_{store_id}_")
                for d in distractors
            )
            actions = (
                task.evaluation_criteria.actions if task.evaluation_criteria else None
            ) or []
            if any("reservation" in a.name for a in actions):
                assert not any(d.endswith("_table.wav") for d in distractors), task.id

        reg = loaded.get_config(task.id, "regular")
        assert reg is not None
        assert reg.environment == expected_env
        expected_bgs = (
            set(FOOD_DRIVE_THRU_BG_FILES)
            if channel == "drive_thru"
            else set(FOOD_MOBILE_BG_FILES)
        )
        assert reg.background_noise_file in expected_bgs
        assert reg.burst_noise_files == acoustic_bursts
        assert len(reg.background_speech_files) == 10
        assert (
            sum(1 for f in reg.background_speech_files if f in ood_speech)
            == FOOD_OOD_SPEECH_SLOTS
        )
        assert (
            sum(1 for f in reg.background_speech_files if f in distractors)
            == FOOD_IN_DOMAIN_SPEECH_SLOTS
        )
        assert set(distractors).issubset(
            set(reg.background_speech_files[FOOD_OOD_SPEECH_SLOTS:])
        )
        assert reg.source_effects_config.burst_snr_range_db == BURST_SNR_RANGE_DB
        assert (
            reg.source_effects_config.burst_noise_events_per_minute
            == BURST_NOISE_EVENTS_PER_MINUTE
        )
        assert reg.source_effects_config.enable_background_speech is True
        assert (
            reg.source_effects_config.background_speech_events_per_minute
            == FOOD_BACKGROUND_SPEECH_EVENTS_PER_MINUTE
        )
        assert (
            reg.source_effects_config.background_speech_snr_range_db
            == BACKGROUND_SPEECH_SNR_RANGE_DB
        )

        for clean_level in (
            "control",
            "control_accents",
            "control_behavior",
            "control_accents_behavior",
        ):
            clean_cfg = loaded.get_config(task.id, clean_level)
            assert clean_cfg is not None
            assert clean_cfg.environment is None
            assert clean_cfg.background_speech_files == []
            assert clean_cfg.source_effects_config.enable_background_speech is False

        resampling = get_or_load_task_voice_config(
            domain="food",
            task_id=task.id,
            task_seed=42 + idx,
            complexity="regular",
            synthesis_config=SynthesisConfig(),
            force_sample=True,
        )
        assert resampling == get_or_load_task_voice_config(
            domain="food",
            task_id=task.id,
            task_seed=42 + idx,
            complexity="regular",
            synthesis_config=SynthesisConfig(),
            force_sample=True,
        )
        assert resampling.environment == expected_env
        assert resampling.background_noise_file in expected_bgs


def test_food_distractor_filter_handles_actions_and_scenarios():
    tasks_by_id = {t.id: t for t in get_tasks("base")}

    cafe_combo_files = get_food_in_domain_distractor_files(
        tasks_by_id["cafe_turkey_club_combo_upgrade"]
    )
    assert "food/in_domain/other_lane_cafe_panini.wav" not in cafe_combo_files
    assert "food/in_domain/other_lane_cafe_mocha.wav" in cafe_combo_files
    assert "food/in_domain/other_lane_cafe_window.wav" in cafe_combo_files

    burger_nugget_files = get_food_in_domain_distractor_files(
        tasks_by_id["burger_salad_side_dressing_and_nugget_swap"]
    )
    assert "food/in_domain/other_lane_burger_wrap.wav" not in burger_nugget_files
    assert "food/in_domain/other_lane_burger_shake.wav" not in burger_nugget_files
    assert "food/in_domain/other_lane_burger_pier.wav" in burger_nugget_files
    assert "food/in_domain/other_lane_burger_window.wav" in burger_nugget_files

    assert "food/in_domain/other_lane_cafe_matcha.wav" in (
        get_food_in_domain_distractor_files(
            tasks_by_id["cafe_pastry_combo_latte_to_cappuccino"]
        )
    )
    assert "food/in_domain/other_lane_pizza_classic.wav" in (
        get_food_in_domain_distractor_files(
            tasks_by_id["pizza_half_and_half_supreme_swap"]
        )
    )

    cafe_dt = tasks_by_id["cafe_pastry_combo_latte_to_cappuccino"].model_copy(deep=True)
    cafe_dt.user_scenario = None
    cafe_dt.evaluation_criteria.actions = [
        Action(
            action_id="a1",
            name="add_to_cart",
            arguments={
                "store_id": "cafe",
                "item_id": "White Chocolate Mocha",
                "modifiers": [{"group_id": "Size", "option_id": "Large"}],
            },
        )
    ]
    custom_files = get_food_in_domain_distractor_files(cafe_dt)
    assert "food/in_domain/other_lane_cafe_mocha.wav" not in custom_files
    assert "food/in_domain/other_lane_cafe_cold_brew.wav" in custom_files
    assert "food/in_domain/other_lane_cafe_panini.wav" in custom_files

    cafe_mob_res = cafe_dt.model_copy(deep=True)
    cafe_mob_res.initial_state.initialization_data.agent_data["cart"]["channel"] = (
        "mobile"
    )
    cafe_mob_res.evaluation_criteria.actions = [
        Action(
            action_id="r1",
            name="make_reservation",
            arguments={"store_id": "cafe", "party_size": 2},
        )
    ]
    mob_res_files = get_food_in_domain_distractor_files(cafe_mob_res)
    assert "food/in_domain/household_cafe_table.wav" not in mob_res_files
    assert "food/in_domain/household_cafe_mocha.wav" in mob_res_files

    cafe_mob_seating = cafe_mob_res.model_copy(deep=True)
    cafe_mob_seating.evaluation_criteria.actions = []
    cafe_mob_seating.user_scenario = UserScenario(
        instructions=StructuredUserInstructions(
            domain="food",
            reason_for_call="Ask if they have outdoor seating available tonight.",
            task_instructions="",
        )
    )
    assert (
        "food/in_domain/household_cafe_table.wav"
        not in get_food_in_domain_distractor_files(cafe_mob_seating)
    )


@pytest.mark.parametrize(
    ("task_id", "expected_env", "expected_burst_substr"),
    [
        (
            "cafe_pastry_combo_latte_to_cappuccino",
            "food_cafe_drive_thru",
            "food/in_domain/other_lane_cafe_",
        ),
        (
            "burger_mobile_pier_burger_combos_bacon_undo",
            "food_burger_mobile",
            "food/in_domain/household_burger_",
        ),
    ],
)
def test_build_voice_user_wires_food_presets_and_channel_modes(
    task_id, expected_env, expected_burst_substr
):
    pytest.importorskip("scipy")

    tasks = {t.id: t for t in get_tasks("base")}
    task = tasks[task_id]
    env = get_environment(task=task)

    reg_cfg = VoiceRunConfig(
        domain="food",
        speech_complexity="regular",
        audio_native_config=AudioNativeConfig(),
    )
    lang_ctx = task_context(reg_cfg, task)
    user = build_voice_user(
        env,
        task,
        reg_cfg.audio_native_config,
        language_context=lang_ctx,
        speech_complexity="regular",
        domain="food",
    )
    speech_env = user.voice_settings.speech_environment
    source_cfg = user.voice_settings.synthesis_config.source_effects_config

    assert speech_env.environment == expected_env
    assert all(not b.startswith("food/") for b in speech_env.burst_noise_files)
    assert any(expected_burst_substr in b for b in speech_env.background_speech_files)
    assert len(speech_env.background_speech_files) == 10
    if expected_env.endswith("_mobile"):
        assert speech_env.burst_noise_files == list(FOOD_MOBILE_ACOUSTIC_BURSTS)
        assert not any("other_lane_" in b for b in speech_env.background_speech_files)
        assert (
            sum(
                1
                for b in speech_env.background_speech_files
                if b in FOOD_OUT_OF_DOMAIN_MOBILE_SPEECH
            )
            == FOOD_OOD_SPEECH_SLOTS
        )
    else:
        assert speech_env.burst_noise_files == list(FOOD_DRIVE_THRU_ACOUSTIC_BURSTS)
        assert not any("household_" in b for b in speech_env.background_speech_files)
        assert speech_env.background_speech_files[:FOOD_OOD_SPEECH_SLOTS] == list(
            FOOD_OUT_OF_DOMAIN_DRIVE_THRU_SPEECH
        )
    assert source_cfg.burst_snr_range_db == BURST_SNR_RANGE_DB
    assert source_cfg.burst_noise_events_per_minute == BURST_NOISE_EVENTS_PER_MINUTE
    assert source_cfg.enable_background_speech is True
    assert (
        source_cfg.background_speech_events_per_minute
        == FOOD_BACKGROUND_SPEECH_EVENTS_PER_MINUTE
    )
    assert source_cfg.background_speech_snr_range_db == BACKGROUND_SPEECH_SNR_RANGE_DB

    for clean_complexity in ("control", "control_behavior"):
        clean_user = build_voice_user(
            env,
            task,
            reg_cfg.audio_native_config,
            language_context=lang_ctx,
            speech_complexity=clean_complexity,
            domain="food",
        )
        assert clean_user.voice_settings.speech_environment.environment is None
        assert clean_user.voice_settings.speech_environment.burst_noise_files == []
        assert (
            clean_user.voice_settings.speech_environment.background_speech_files == []
        )
        assert (
            clean_user.voice_settings.synthesis_config.source_effects_config.enable_background_speech
            is False
        )

    heavy_channel_user = build_voice_user(
        env,
        task,
        reg_cfg.audio_native_config,
        language_context=lang_ctx,
        speech_complexity="regular",
        channel_effects_mode="heavy",
        domain="food",
    )
    assert (
        heavy_channel_user.voice_settings.speech_environment.environment == expected_env
    )
    heavy_src = heavy_channel_user.voice_settings.synthesis_config.source_effects_config
    assert heavy_src.burst_noise_events_per_minute == 4.0
    assert heavy_src.burst_snr_range_db == (-5.0, 0.0)
    assert (
        heavy_src.background_speech_events_per_minute
        == FOOD_HEAVY_BACKGROUND_SPEECH_EVENTS_PER_MINUTE
    )
    assert heavy_src.background_speech_snr_range_db == BACKGROUND_SPEECH_SNR_RANGE_DB

    clean_heavy_channel = build_voice_user(
        env,
        task,
        reg_cfg.audio_native_config,
        language_context=lang_ctx,
        speech_complexity="control",
        channel_effects_mode="heavy",
        domain="food",
    )
    assert clean_heavy_channel.voice_settings.speech_environment.environment is None
    assert (
        clean_heavy_channel.voice_settings.synthesis_config.source_effects_config.enable_background_speech
        is False
    )


def test_registry_hook_and_slot_pool_determinism():
    assert registry.get_domain_voice_preset("food") is apply_food_voice_preset
    assert registry.get_domain_voice_preset("airline") is None
    assert "background_speech_events_per_minute" not in CHANNEL_EFFECTS_MODES["heavy"]

    non_food_heavy = sample_voice_config(
        seed=77,
        synthesis_config=SynthesisConfig(),
        complexity="regular",
        channel_effects_mode="heavy",
    )
    assert non_food_heavy.source_effects_config.enable_background_speech is False
    assert (
        non_food_heavy.source_effects_config.background_speech_events_per_minute == 0.0
    )

    expanded_4_to_3 = _expand_slot_bucket(
        ["a.wav", "b.wav", "c.wav", "d.wav"], 3, random.Random(9)
    )
    assert len(expanded_4_to_3) == 3
    assert len(set(expanded_4_to_3)) == 3

    expanded_2_to_7 = _expand_slot_bucket(["x.wav", "y.wav"], 7, random.Random(9))
    assert len(expanded_2_to_7) == 7
    assert sorted([expanded_2_to_7.count("x.wav"), expanded_2_to_7.count("y.wav")]) == [
        3,
        4,
    ]
    assert expanded_2_to_7 == _expand_slot_bucket(
        ["x.wav", "y.wav"], 7, random.Random(9)
    )
