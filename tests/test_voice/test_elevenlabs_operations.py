"""Voice setup and reviewed provisioning preserve their SDK request semantics."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tau2.data_model.voice import VoiceCreateRequest, VoiceDesignRequest
from tau2.voice.elevenlabs_operations import create_voice, design_voice


@pytest.fixture
def client():
    client = Mock()
    client.text_to_voice.design.return_value = SimpleNamespace(
        previews=[
            SimpleNamespace(
                generated_voice_id="first",
                audio_base_64="YQ==",
                media_type="audio/mpeg",
            ),
            SimpleNamespace(
                generated_voice_id="second",
                audio_base_64="Yg==",
                media_type="audio/mpeg",
            ),
        ]
    )
    client.text_to_voice.create.return_value = SimpleNamespace(voice_id="saved")
    client.voices.get_all.return_value = SimpleNamespace(voices=[])
    return client


def test_shared_operations_preserve_parameters_and_never_retry_creation(client):
    previews = design_voice(
        client,
        VoiceDesignRequest(
            description="full brief",
            text="reviewed text",
            model="design-model",
            seed=31,
            loudness=0.4,
            guidance_scale=29,
        ),
    )
    assert [p.generated_voice_id for p in previews] == ["first", "second"]
    client.text_to_voice.design.assert_called_once_with(
        voice_description="full brief",
        text="reviewed text",
        model_id="design-model",
        seed=31,
        loudness=0.4,
        guidance_scale=29,
        auto_generate_text=False,
    )
    client.text_to_voice.create.side_effect = TimeoutError("uncertain creation")
    with pytest.raises(TimeoutError):
        create_voice(
            client,
            VoiceCreateRequest(
                name="account-name",
                description="short label",
                generated_voice_id="first",
            ),
        )
    client.text_to_voice.create.assert_called_once_with(
        voice_name="account-name",
        voice_description="short label",
        generated_voice_id="first",
    )


def test_setup_designs_from_the_language_prompts_and_replaces_existing_ids(
    client, monkeypatch
):
    import elevenlabs

    from tau2.data_model.voice import PersonaVoiceDesign
    from tau2.voice.scripts import setup_voices

    design = PersonaVoiceDesign(
        gender="female",
        model="persona-model",
        prompt="long acoustic brief",
        audition_text="native audition " * 10,
    )
    target = setup_voices.PersonaDesignTarget(
        language="pt-br", persona_id="example", design=design
    )
    requested = []

    def fake_targets(languages):
        requested.append(languages)
        return [target]

    monkeypatch.setattr(setup_voices, "design_targets", fake_targets)
    monkeypatch.setattr(elevenlabs, "ElevenLabs", lambda: client)
    client.voices.get_all.return_value = SimpleNamespace(
        voices=[SimpleNamespace(name="tau2_pt-br_example", voice_id="old")]
    )
    result = setup_voices.setup_voices(languages=["pt-br"], seed=17, force=True)
    assert requested == [["pt-br"]]
    assert result == {"TAU2_VOICE_ID_PT_BR_EXAMPLE": "saved"}
    client.voices.delete.assert_called_once_with(voice_id="old")
    client.text_to_voice.design.assert_called_once_with(
        voice_description="long acoustic brief",
        text="native audition " * 10,
        model_id="persona-model",
        seed=17,
        loudness=setup_voices.VOICE_DESIGN_LOUDNESS,
        guidance_scale=setup_voices.VOICE_DESIGN_GUIDANCE_SCALE,
        auto_generate_text=False,
    )
    client.text_to_voice.create.assert_called_once_with(
        voice_name="tau2_pt-br_example",
        voice_description="tau2 pt-br caller example (female)",
        generated_voice_id="first",
    )


def test_setup_model_override_and_skip_keep_existing_voices(client, monkeypatch):
    import elevenlabs

    from tau2.voice.scripts import setup_voices

    targets = setup_voices.design_targets(["en", "es"])
    monkeypatch.setattr(setup_voices.time, "sleep", lambda _: None)
    monkeypatch.setattr(elevenlabs, "ElevenLabs", lambda: client)
    monkeypatch.setattr("builtins.input", lambda _: "s")
    existing = targets[0].voice_name
    client.voices.get_all.return_value = SimpleNamespace(
        voices=[
            SimpleNamespace(name=existing, voice_id="kept"),
            # The pre-language naming never shadows a language-prefixed voice.
            SimpleNamespace(name="tau2_mamadou_diallo", voice_id="legacy"),
        ]
    )
    result = setup_voices.setup_voices(
        languages=["en", "es"], model="eleven_multilingual_ttv_v2"
    )
    assert list(result) == [t.env_key for t in targets[1:]]
    client.voices.delete.assert_not_called()
    calls = client.text_to_voice.design.call_args_list
    assert len(calls) == len(targets) - 1
    assert {c.kwargs["model_id"] for c in calls} == {"eleven_multilingual_ttv_v2"}
    assert [c.kwargs["text"] for c in calls] == [
        t.design.audition_text for t in targets[1:]
    ]


def test_dry_run_makes_no_design_calls(client, monkeypatch, capsys):
    import elevenlabs

    from tau2.voice.scripts import setup_voices

    monkeypatch.setattr(elevenlabs, "ElevenLabs", lambda: client)
    client.voices.get_all.return_value = SimpleNamespace(voices=[])
    assert setup_voices.setup_voices(languages=["hi"], dry_run=True) == {}
    client.text_to_voice.design.assert_not_called()
    client.text_to_voice.create.assert_not_called()
    out = capsys.readouterr().out
    assert "TAU2_VOICE_ID_HI_RISHIKA_HINDI_V1" in out
    assert "tau2_hi_imran_hindi_v1" in out


def test_env_block_prints_runtime_override_lines(capsys):
    from tau2.voice.scripts import setup_voices

    setup_voices.print_env_block({"TAU2_VOICE_ID_KO_JIHUN_KO_V1": "abc"})
    assert "TAU2_VOICE_ID_KO_JIHUN_KO_V1=abc" in capsys.readouterr().out.splitlines()


def test_factory_keeps_reviewed_text_and_selected_preview(client):
    from tau2.multilingual.factory.voice_assets import (
        ElevenLabsVoiceProvider,
        VoiceDesignConfig,
        VoiceSpec,
    )

    provider = ElevenLabsVoiceProvider.__new__(ElevenLabsVoiceProvider)
    provider.client = client
    preview = provider.preview(
        VoiceSpec(
            description="reviewed brief", audition_text="reviewed audition " * 10
        ),
        VoiceDesignConfig(
            model="reviewed-model", seed=8, loudness=0.1, guidance_scale=40
        ),
    )
    assert preview.generated_voice_id == "first"
    client.text_to_voice.design.assert_called_once_with(
        voice_description="reviewed brief",
        text="reviewed audition " * 10,
        model_id="reviewed-model",
        seed=8,
        loudness=0.1,
        guidance_scale=40,
        auto_generate_text=False,
    )
