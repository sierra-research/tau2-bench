"""Regressions for lean authoring inputs and bounded instruction checks."""

import pytest

from tau2.multilingual.factory import voice_assets
from tau2.multilingual.factory.prompt_rules import validate_authoring_instruction
from tests.test_multilingual import test_voice_asset_storage as fixtures

Provider = fixtures.Provider
asset_request = fixtures.asset_request
draft_request = fixtures.draft_request


@pytest.mark.parametrize(
    "text",
    [
        "Do not add pronunciation guidance.",
        "Do not add pronunciation instructions.",
        "Never add authentication tactics.",
        "Avoid enunciation instructions.",
        "Do not always spell your name.",
        "Do not always spell names.",
        'Say "Could you speak slowly?" when you need clarification.',
    ],
)
def test_instruction_guard_accepts_prohibitions_and_customer_questions(text):
    validate_authoring_instruction(text, "fixture")


@pytest.mark.parametrize(
    "text",
    [
        '"Use a rising pitch."',
        'Use the instruction "Always spell your name."',
        "Follow this instruction: “Always spell your name.”",
        '"If lookup fails, ask the customer to spell their name."',
        "Do not change vocabulary, but speak slowly.",
        "Never change vocabulary; use a rising pitch.",
    ],
)
def test_instruction_guard_rejects_quoted_and_contrasting_directives(text):
    with pytest.raises(ValueError, match="authoring instruction"):
        validate_authoring_instruction(text, "fixture")


def test_existing_voice_input_needs_only_voice_model_and_audition(
    asset_request, tmp_path
):
    value = asset_request.model_dump()
    value["config"] = {
        "mode": "existing",
        "model": "fixture-tts",
        "voice_ids": {name: f"existing-{i}" for i, name in enumerate(value["specs"])},
    }
    value["specs"] = {
        name: {"audition_text": spec["audition_text"]}
        for name, spec in value["specs"].items()
    }
    request = voice_assets.VoiceAssetsRequest.model_validate(value)
    provider = Provider()
    result = voice_assets.prepare_voices(request, tmp_path / "existing", provider)
    assert (provider.previews, provider.creates, provider.auditions) == (0, 0, 2)
    assert result.config == request.config
    assert {
        name: checkpoint.voice.voice_id for name, checkpoint in result.voices.items()
    } == request.config.voice_ids
    assert all(
        set(spec.model_dump()) == {"audition_text"} for spec in result.specs.values()
    )
    value["config"]["seed"] = 123
    with pytest.raises(ValueError):
        voice_assets.VoiceAssetsRequest.model_validate(value)
