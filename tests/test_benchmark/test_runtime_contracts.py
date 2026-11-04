"""Requested settings and persisted scoring conditions have separate contracts."""

import json

import pytest

from tau2.data_model.benchmark import BenchmarkPlan, VoiceBenchmarkConditions
from tau2.data_model.run_config import AudioNativeConfig, VoiceRunSettings
from tau2.data_model.simulation import VoiceRunConfig
from tau2.metrics.benchmark import (
    export_submission,
    load_submission,
    validate_submission,
)
from tau2.runner.presets import resolve_configs


def test_conditions_reuse_voice_settings_and_require_complete_record(
    published_frame, tmp_path
):
    plan = resolve_configs(
        published_frame,
        VoiceRunConfig(audio_native_config=AudioNativeConfig()),
        tmp_path,
    )
    record = plan.configs[0].benchmark
    assert isinstance(record.conditions, VoiceRunSettings)
    for change in ("missing", "unknown", "missing_audio", "unknown_audio"):
        payload = plan.model_dump(mode="json")
        condition = payload["configs"][0]["benchmark"]["conditions"]
        if change == "missing":
            condition.pop("max_errors")
        elif change == "unknown":
            condition["made_up_setting"] = True
        elif change == "missing_audio":
            condition["audio_native_config"].pop("max_steps_seconds")
        else:
            condition["audio_native_config"]["unknown_setting"] = True
        with pytest.raises(ValueError):
            BenchmarkPlan.model_validate(payload)


def test_historical_judge_read_accepts_only_preexisting_voice_condition_gaps():
    complete = VoiceBenchmarkConditions.model_validate(
        VoiceRunSettings(audio_native_config=AudioNativeConfig()).model_dump(
            mode="json"
        )
        | {"voice_instruction_prompt_version": "voice-instructions-v2"}
    ).model_dump(mode="json")
    legacy = complete.copy()
    for field in (
        "channel_effects_mode",
        "speech_effects_mode",
        "pass3_condition",
        "voice_sampling_seed",
    ):
        legacy.pop(field)
    with pytest.raises(ValueError, match="incomplete"):
        VoiceBenchmarkConditions.model_validate(legacy)
    assert (
        VoiceBenchmarkConditions.model_validate(
            legacy, context={"historical_judge_read": True}
        ).model_dump(mode="json")
        == complete
    )
    legacy.pop("max_errors")
    with pytest.raises(ValueError, match="incomplete"):
        VoiceBenchmarkConditions.model_validate(
            legacy, context={"historical_judge_read": True}
        )


def test_saved_livekit_template_keeps_request_separate_from_effective_settings(
    published_frame, tmp_path
):
    from tau2.data_model.cascaded import CascadedConfigRequest

    template = VoiceRunConfig(
        audio_native_config=AudioNativeConfig(
            provider="livekit",
            cascaded_request=CascadedConfigRequest(preset="openai-thinking"),
        )
    )
    # A serialized ordinary English-default template must still resolve all locales.
    template = VoiceRunConfig.model_validate_json(template.model_dump_json())
    before = template.model_dump_json()
    plan = resolve_configs(published_frame, template, tmp_path)
    assert len({c.benchmark.condition_fingerprint for c in plan.configs}) == 1
    for config in plan.configs:
        audio = config.audio_native_config
        assert audio.cascaded_request.preset == "openai-thinking"
        assert audio.cascaded_request.config is None
        assert audio.cascaded_config.tts.language == config.language
        assert audio.cascaded_config.llm.model == "gpt-5.2"
        assert VoiceRunConfig.model_validate_json(config.model_dump_json()) == config
    assert template.model_dump_json() == before


def test_submission_readers_recompute_both_summaries(published_frame, tmp_path):
    plan = resolve_configs(
        published_frame,
        VoiceRunConfig(num_tasks=1, audio_native_config=AudioNativeConfig()),
        tmp_path,
    )
    (tmp_path / "benchmark-plan.json").write_text(plan.model_dump_json())
    submission = export_submission(tmp_path, published_frame)
    validate_submission(submission)
    for field in ("summary", "requested"):
        payload = submission.model_dump(mode="json")
        payload[field]["score"] = 1
        path = tmp_path / "benchmark-submission.json"
        path.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="summary"):
            load_submission(path)


def test_livekit_saved_effective_settings_cannot_disagree_with_request():
    run = VoiceRunConfig(
        language="es", audio_native_config=AudioNativeConfig(provider="livekit")
    )
    payload = run.model_dump(mode="json")
    payload["audio_native_config"]["cascaded_config"]["tts"]["voice_id"] = (
        "different-voice"
    )
    with pytest.raises(ValueError, match="Recorded LiveKit settings"):
        VoiceRunConfig.model_validate(payload)
