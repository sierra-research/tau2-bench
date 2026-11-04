"""Intake pilot bindings stay separate from banking and the released benchmark."""

import pytest

from tau2.data_model.run_config import AudioNativeConfig
from tau2.data_model.simulation import VoiceRunConfig
from tau2.multilingual.runtime import load_language, resolve_task
from tau2.runner.batch import _load_run_tasks
from tau2.runner.helpers import get_info
from tau2.runner.presets import load_frame


@pytest.mark.parametrize("language", ["ko", "zh", "pt", "es"])
def test_intake_requires_its_own_explicit_binding(language):
    with pytest.raises(ValueError, match="does not support"):
        load_language(language, "intake", mode="voice")
    with pytest.raises(ValueError, match="new declared domain"):
        load_language(language, "intake", mode="voice", pack_extension="banking")

    loaded = load_language(language, "intake", mode="voice", pack_extension="intake")
    assert loaded.scenario_mode == "regional"
    assert len(loaded.tasks) == len(loaded.metadata) == 200
    assert loaded.scenario.localization_map_sha256 == loaded.localization_map_sha256
    context = resolve_task(loaded, loaded.tasks[0], seed=300)
    assert context.pack_extension == "intake"
    assert context.artifact_pins == loaded.artifact_pins

    banking = load_language(
        language, "banking_knowledge", mode="voice", pack_extension="banking"
    )
    assert banking.artifact_pins.pack_sha256 != loaded.artifact_pins.pack_sha256
    assert all(cell.domain != "intake" for cell in load_frame().cells)


def test_intake_voice_run_resolves_and_records_its_binding():
    config = VoiceRunConfig(
        domain="intake",
        language="ko",
        num_tasks=1,
        audio_native_config=AudioNativeConfig(),
    )
    tasks = _load_run_tasks(config)
    assert config.pack_extension == "intake"
    info = get_info(config, tasks=tasks)
    assert len(tasks) == len(info.language_tasks) == 1
    assert info.pack_extension == "intake"
    assert info.language_artifacts == info.language_tasks[tasks[0].id].artifact_pins


def test_hindi_intake_pilot_keeps_full_easy_and_hard_frame():
    config = VoiceRunConfig(
        domain="intake",
        language="hi",
        num_tasks=1,
        audio_native_config=AudioNativeConfig(),
    )
    tasks = _load_run_tasks(config)
    assert len(tasks) == 1
    assert config.pack_extension == "intake"
    loaded = load_language("hi", "intake", mode="voice", pack_extension="intake")
    assert len(loaded.tasks) == len(loaded.metadata) == 200
    assert sum("_easy_" in task.id for task in loaded.tasks) == 100
    assert sum("_hard_" in task.id for task in loaded.tasks) == 100
    assert loaded.scenario.localization_map_sha256 == loaded.localization_map_sha256
    context = resolve_task(loaded, loaded.tasks[0], seed=300)
    assert context.pack_extension == "intake"
    assert context.persona.gender == loaded.metadata[loaded.tasks[0].id].caller_gender
