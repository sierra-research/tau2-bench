"""Prompt implementation provenance cannot silently change on voice resume."""

import pytest

from tau2.config import VOICE_INSTRUCTION_PROMPT_VERSION
from tau2.data_model.run_config import AudioNativeConfig
from tau2.data_model.simulation import Results, TextRunConfig, VoiceRunConfig
from tau2.data_model.tasks import Task
from tau2.runner.checkpoint import try_resume
from tau2.runner.helpers import get_info


@pytest.mark.parametrize("kind", ["text", "voice", "scaffolded", "agent_directed"])
def test_effective_shared_prompt_revision_is_recorded(kind):
    if kind == "text":
        config = TextRunConfig(domain="retail")
    elif kind == "voice":
        config = VoiceRunConfig(
            domain="retail", audio_native_config=AudioNativeConfig()
        )
    else:
        config = VoiceRunConfig(
            domain="intake",
            elicitation={"strategy": kind},
            audio_native_config=AudioNativeConfig(),
        )
    info = get_info(config)
    expected = (
        VOICE_INSTRUCTION_PROMPT_VERSION if kind in {"voice", "scaffolded"} else None
    )
    assert info.voice_instruction_prompt_version == expected
    assert "voice_instruction_version" not in AudioNativeConfig.model_fields


@pytest.mark.parametrize("previous_version", [None, "voice-instructions-v1"])
def test_voice_resume_refuses_legacy_or_changed_shared_prompt(
    previous_version, tmp_path
):
    current = get_info(
        VoiceRunConfig(domain="retail", audio_native_config=AudioNativeConfig())
    )
    old = current.model_copy(
        update={"voice_instruction_prompt_version": previous_version}, deep=True
    )
    task = Task(id="fixture", user_scenario={"instructions": "Fixture."})
    path = tmp_path / "results.json"
    Results(info=old, tasks=[task]).save(path)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Voice instruction prompt revision changed"):
        try_resume(
            path, Results(info=current, tasks=[task]), [task], 1, auto_resume=True
        )
    assert path.read_bytes() == before


def test_voice_resume_accepts_same_shared_prompt_revision(tmp_path):
    info = get_info(
        VoiceRunConfig(domain="retail", audio_native_config=AudioNativeConfig())
    )
    task = Task(id="fixture", user_scenario={"instructions": "Fixture."})
    path = tmp_path / "results.json"
    results = Results(info=info, tasks=[task])
    results.save(path)
    resumed, _, _ = try_resume(path, results, [task], 1, auto_resume=True)
    assert (
        resumed.info.voice_instruction_prompt_version
        == VOICE_INSTRUCTION_PROMPT_VERSION
    )


@pytest.mark.parametrize("previous_version", [None, "exact-identity-v1"])
def test_resume_rejects_changed_domain_lookup_contract(previous_version, tmp_path):
    from tau2.config import DOMAIN_TOOLS_VERSION

    current = get_info(TextRunConfig(domain="retail"))
    assert current.domain_tools_version == DOMAIN_TOOLS_VERSION
    old = current.model_copy(update={"domain_tools_version": previous_version})
    task = Task(id="fixture", user_scenario={"instructions": "Fixture."})
    path = tmp_path / "results.json"
    Results(info=old, tasks=[task]).save(path)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="[Dd]omain.tools.(revision|version)"):
        try_resume(
            path, Results(info=current, tasks=[task]), [task], 1, auto_resume=True
        )
    assert path.read_bytes() == before
