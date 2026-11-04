"""Offline fixed-prompt authoring, validation and resume coverage."""

import json
from types import SimpleNamespace

import pytest

from tau2.multilingual.factory import author_form
from tau2.multilingual.factory.author_form import (
    run_autoform,
)
from tau2.multilingual.factory.schema import (
    AuthorForm,
    AutoformRequest,
    FactoryModelConfig,
    LanguageIdentity,
    PersonaSketch,
)


@pytest.fixture
def autoform_request():
    return AutoformRequest(
        identity=LanguageIdentity(language="en", display_name="English", script="latn"),
        generation=FactoryModelConfig(
            model="fixture-model", reasoning_effort="minimal"
        ),
        hints="Offline test only.",
    )


@pytest.fixture
def recorded_reply(autoform_request):
    return AuthorForm(
        identity=autoform_request.identity,
        locale_notes="Fixture locale notes.",
        preferences="Fixture preferences.",
        personas=[
            PersonaSketch(
                name="Fixture Male", gender="male", sketch="Fixture register A."
            ),
            PersonaSketch(
                name="Fixture Female", gender="female", sketch="Fixture register B."
            ),
        ],
    )


def test_validated_autoform_records_provenance_and_resumes_without_generation(
    monkeypatch, autoform_request, recorded_reply, tmp_path
):
    autoform_request.persona_names = {
        persona.gender: persona.name for persona in recorded_reply.personas
    }
    calls = []
    fixtures = {"factory_autoform": recorded_reply.model_dump_json()}

    def generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(content=fixtures[kwargs["call_name"]])

    monkeypatch.setattr(author_form, "generate", generate)
    output = tmp_path / "author-form.json"
    first = run_autoform(autoform_request, output)
    assert run_autoform(autoform_request, output) == first
    assert len(calls) == 1
    assert calls[0]["model"] == "fixture-model"
    assert calls[0]["reasoning_effort"] == "minimal"
    assert author_form.AUTHORING_RULES in calls[0]["messages"][1].content
    assert first.prompt_version == author_form.AUTOFORM_PROMPT_VERSION
    assert first.request == autoform_request
    assert first.request.hints == "Offline test only."
    assert first.input_sha256 == author_form.content_digest(autoform_request)
    assert first.form == recorded_reply
    changed = autoform_request.model_copy(update={"hints": "Changed authoring brief"})
    with pytest.raises(FileExistsError, match="new output path"):
        run_autoform(changed, output)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "defect", ["identity", "gender_pair", "unknown_field", "not_json", "reviewed_names"]
)
def test_bad_llm_reply_never_publishes_a_form(
    monkeypatch, autoform_request, recorded_reply, tmp_path, defect
):
    reply = recorded_reply.model_dump(mode="json")
    if defect == "identity":
        reply["identity"]["language"] = "es"
    elif defect == "gender_pair":
        reply["personas"][1]["gender"] = "male"
    elif defect == "unknown_field":
        reply["voice_id"] = "invented"
    elif defect == "reviewed_names":
        autoform_request.persona_names = {
            "male": "Mamadou Diallo",
            "female": "Priya Patil",
        }
    content = "invalid JSON" if defect == "not_json" else json.dumps(reply)
    monkeypatch.setattr(
        author_form, "generate", lambda **kwargs: SimpleNamespace(content=content)
    )
    output = tmp_path / "author-form.json"
    with pytest.raises(ValueError):
        run_autoform(autoform_request, output)
    assert not output.exists()


def test_prior_tool_value_authoring_rules_cannot_resume(
    monkeypatch, autoform_request, recorded_reply, tmp_path
):
    monkeypatch.setattr(
        author_form,
        "generate",
        lambda **kwargs: SimpleNamespace(content=recorded_reply.model_dump_json()),
    )
    output = tmp_path / "author-form.json"
    previous = run_autoform(autoform_request, output)
    previous.prompt_version = "public-author-form-v3"
    output.write_text(previous.model_dump_json())
    monkeypatch.setattr(
        author_form, "generate", lambda **kwargs: pytest.fail("stale replay called LLM")
    )
    with pytest.raises(FileExistsError, match="new output path"):
        run_autoform(autoform_request, output)


@pytest.mark.parametrize(
    "field,guidance",
    [
        ("locale_notes", "Use a rising pitch and speak at a slower rate."),
        ("preferences", "If lookup fails, ask the customer to spell their name."),
        ("sketch", "Authenticate customers by requiring their name and date of birth."),
    ],
)
def test_retired_instructions_never_persist_in_generated_form(
    monkeypatch, autoform_request, recorded_reply, tmp_path, field, guidance
):
    value = recorded_reply.model_dump()
    if field == "sketch":
        value["personas"][0]["sketch"] = guidance
    else:
        value[field] = guidance
    monkeypatch.setattr(
        author_form, "generate", lambda **kw: SimpleNamespace(content=json.dumps(value))
    )
    output = tmp_path / "bad-form.json"
    with pytest.raises(ValueError, match="authoring instruction"):
        run_autoform(autoform_request, output)
    assert not output.exists()
