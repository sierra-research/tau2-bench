"""Separate-run regional bindings leave the released V2 benchmark untouched."""

import pytest

from tau2.data_model.run_config import AudioNativeConfig
from tau2.data_model.simulation import VoiceRunConfig
from tau2.environment.environment import EnvironmentInfo
from tau2.multilingual.runtime import load_language, resolve_task
from tau2.runner.batch import _load_run_tasks
from tau2.runner.helpers import get_info
from tau2.runner.presets import binding_revision, load_frame


@pytest.mark.parametrize("language", ["ko", "zh", "pt", "es"])
def test_banking_requires_explicit_extension(language):
    with pytest.raises(ValueError, match="does not support"):
        load_language(language, "banking_knowledge", mode="voice")
    loaded = load_language(
        language, "banking_knowledge", mode="voice", pack_extension="banking"
    )
    assert loaded.scenario_mode == "regional"
    assert len(loaded.tasks) == len(loaded.metadata) == 97
    assert loaded.scenario.localization_map_sha256 == loaded.localization_map_sha256
    assert loaded.pack_extension == "banking"
    context = resolve_task(loaded, loaded.tasks[0], seed=0)
    assert context.pack_extension == "banking"
    assert context.artifact_pins == loaded.artifact_pins


@pytest.mark.parametrize("language", ["ko", "zh", "pt", "es"])
def test_extension_does_not_change_released_bindings(language):
    frame = load_frame()
    released = load_language(language, "retail", mode="voice")
    retail_cell = next(
        cell
        for cell in frame.cells
        if cell.language == language and cell.domain == "retail"
    )
    assert binding_revision(released) == retail_cell.binding_revision
    assert all(
        cell.domain not in {"banking_knowledge", "intake"} for cell in frame.cells
    )


def test_extension_rejects_unlocalized_runs():
    with pytest.raises(ValueError, match="localized regional"):
        load_language(
            "ko",
            "banking_knowledge",
            mode="voice",
            localize_db=False,
            pack_extension="banking",
        )


def test_separate_voice_run_resolves_and_records_extension(monkeypatch):
    # Core CI omits banking retrieval dependencies; this checks run metadata.
    monkeypatch.setattr(
        "tau2.runner.helpers.get_environment_info",
        lambda domain, **kwargs: EnvironmentInfo(domain_name=domain, policy="test"),
    )
    config = VoiceRunConfig(
        domain="banking_knowledge",
        language="ko",
        num_tasks=1,
        retrieval_config="bm25",
        audio_native_config=AudioNativeConfig(),
    )
    tasks = _load_run_tasks(config)
    assert config.pack_extension == "banking-dates-v1"
    info = get_info(config, tasks=tasks)
    assert len(tasks) == len(info.language_tasks) == 1
    assert info.pack_extension == "banking-dates-v1"
    assert (
        info.language_artifacts
        == next(iter(info.language_tasks.values())).artifact_pins
    )


@pytest.mark.parametrize(
    "language,canonical_id,gender",
    [
        ("ko", "task_007", "female"),
        ("ko", "task_008", "male"),
        ("ko", "task_010", "male"),
        ("zh", "task_034", "male"),
        ("zh", "task_059", "male"),
    ],
)
def test_banking_voice_matches_approved_name_gender(language, canonical_id, gender):
    loaded = load_language(
        language, "banking_knowledge", mode="voice", pack_extension="banking"
    )
    task_id = f"{canonical_id}_{language}"
    assert loaded.metadata[task_id].caller_gender == gender
    context = resolve_task(
        loaded, next(task for task in loaded.tasks if task.id == task_id), seed=0
    )
    assert context.persona.gender == gender


def test_hindi_banking_pilot_resolves_reviewed_names_and_contacts():
    config = VoiceRunConfig(
        domain="banking_knowledge",
        language="hi",
        num_tasks=1,
        retrieval_config="bm25",
        audio_native_config=AudioNativeConfig(),
    )
    tasks = _load_run_tasks(config)
    assert len(tasks) == 1
    assert config.pack_extension == "banking-dates-v1"
    loaded = load_language(
        "hi", "banking_knowledge", mode="voice", pack_extension="banking-dates-v1"
    )
    assert len(loaded.tasks) == 97
    for task_id, gender in (
        ("task_020_hi", "female"),
        ("task_026_hi", "female"),
        ("task_072_hi", "male"),
        ("task_077_hi", "male"),
        ("task_091_hi", "male"),
    ):
        assert loaded.metadata[task_id].caller_gender == gender
        context = resolve_task(loaded, loaded.task_index[task_id], seed=0)
        assert context.persona.gender == gender


@pytest.mark.parametrize("language", ["hi", "ko", "zh", "pt", "es"])
def test_banking_referral_criterion_uses_reachable_localized_evidence(language):
    loaded = load_language(
        language, "banking_knowledge", mode="voice", pack_extension="banking"
    )
    task = loaded.task_index[f"task_102_{language}"]
    instructions = task.user_scenario.instructions
    criterion = task.evaluation_criteria.nl_assertions[0]

    assert "Let me check the incorporation record" in instructions
    assert "It says November 2020" in instructions
    assert "TechFlow Labs" not in criterion
    assert "Ember Analytics" not in criterion
    assert "newer, eligible startup" in criterion
    assert "older startup is ineligible" in criterion
