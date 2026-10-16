from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tau2.data_model.simulation import AudioNativeConfig, VoiceRunConfig
from tau2.evaluator.evaluator import EvaluationType
from tau2.runner.controller import ControllerRun
from tau2.voice.audio_native.ava_zero_pipecat.config import AvaZeroPipecatConfig


def test_defaults_target_a_local_server_and_default_agent():
    config = AvaZeroPipecatConfig()

    assert config.base_url == "ws://localhost:7860"
    assert config.agent == "conductor-v2"
    assert config.websocket_token_env is None


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        AvaZeroPipecatConfig(unknown_option="value")


def test_agent_id_must_be_a_plain_identifier():
    with pytest.raises(ValidationError):
        AvaZeroPipecatConfig(agent="agent?token=1")


def test_audio_native_config_resolves_ava_zero_defaults():
    config = AudioNativeConfig(provider="ava_zero_pipecat")

    assert config.model == "conductor-v2"
    assert config.ava_zero_config.agent == "conductor-v2"


def test_audio_native_config_takes_model_from_supplied_config():
    config = AudioNativeConfig(
        provider="ava_zero_pipecat",
        ava_zero_config=AvaZeroPipecatConfig(agent="custom-agent"),
    )

    assert config.model == "custom-agent"


def test_audio_native_config_rejects_conflicting_agent_ids():
    with pytest.raises(ValidationError, match="must match"):
        AudioNativeConfig(
            provider="ava_zero_pipecat",
            model="one",
            ava_zero_config=AvaZeroPipecatConfig(agent="two"),
        )


def test_ava_zero_config_is_rejected_for_other_providers():
    with pytest.raises(ValidationError, match="only valid"):
        AudioNativeConfig(provider="openai", ava_zero_config=AvaZeroPipecatConfig())


def test_config_survives_controller_worker_serialization():
    config = VoiceRunConfig(
        domain="mock",
        audio_native_config=AudioNativeConfig(
            provider="ava_zero_pipecat",
            ava_zero_config=AvaZeroPipecatConfig(
                base_url="wss://ava-zero.example",
                websocket_token_env="AVA_ZERO_TOKEN",
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

    assert restored.audio_native_config.ava_zero_config == (
        config.audio_native_config.ava_zero_config
    )
