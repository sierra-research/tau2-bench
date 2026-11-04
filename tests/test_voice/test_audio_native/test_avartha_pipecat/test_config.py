from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tau2.data_model.simulation import AudioNativeConfig, VoiceRunConfig
from tau2.evaluator.evaluator import EvaluationType
from tau2.runner.controller import ControllerRun
from tau2.voice.audio_native.avartha_pipecat.config import AvarthaPipecatConfig

AGENT_URL = "https://platform.example/inference/serverless/pipecat/v1/public/agent-1"


def test_the_hosted_agent_url_is_required():
    with pytest.raises(ValidationError):
        AvarthaPipecatConfig()
    assert AvarthaPipecatConfig(base_url=AGENT_URL + "/").base_url == AGENT_URL


def test_the_agent_url_must_be_http():
    with pytest.raises(ValidationError, match="http\\(s\\) URL"):
        AvarthaPipecatConfig(base_url="ws://localhost:7860")


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        AvarthaPipecatConfig(base_url=AGENT_URL, unknown_option="value")


def test_audio_native_config_needs_the_agent_settings():
    with pytest.raises(ValidationError, match="hosted agent's URL"):
        AudioNativeConfig(provider="avartha_pipecat")


def test_audio_native_config_records_the_avartha_label():
    config = AudioNativeConfig(
        provider="avartha_pipecat",
        avartha_config=AvarthaPipecatConfig(base_url=AGENT_URL),
    )

    assert config.model == "avartha"


def test_the_user_is_generated_without_blocking_ticks_by_default():
    settings = AvarthaPipecatConfig(base_url=AGENT_URL)

    config = AudioNativeConfig(provider="avartha_pipecat", avartha_config=settings)
    assert config.realtime_generation_enabled is True

    config = AudioNativeConfig(
        provider="avartha_pipecat",
        avartha_config=settings,
        realtime_generation=False,
    )
    assert config.realtime_generation_enabled is False


def test_avartha_config_is_rejected_for_other_providers():
    with pytest.raises(ValidationError, match="only valid"):
        AudioNativeConfig(
            provider="openai",
            avartha_config=AvarthaPipecatConfig(base_url=AGENT_URL),
        )


def test_config_survives_controller_worker_serialization():
    config = VoiceRunConfig(
        domain="mock",
        audio_native_config=AudioNativeConfig(
            provider="avartha_pipecat",
            avartha_config=AvarthaPipecatConfig(
                base_url=AGENT_URL, boot_timeout_seconds=30.0
            ),
        ),
    )

    controller_run = object.__new__(ControllerRun)
    controller_run.config = config
    controller_run.tasks_by_id = {
        "task-1": SimpleNamespace(model_dump=lambda **_kwargs: {"id": "task-1"})
    }
    controller_run.evaluation_type = EvaluationType.ALL
    controller_run.save_dir = None
    controller_run.llm_log_mode = "latest"
    payload = controller_run.lease_payload(SimpleNamespace(task_id="task-1"))
    restored = VoiceRunConfig.model_validate(payload["config"])

    assert restored.audio_native_config.avartha_config == (
        config.audio_native_config.avartha_config
    )
