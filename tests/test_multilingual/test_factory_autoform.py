# Copyright Sierra
"""Tests for autoform generation (FakeFactoryLLM only).

``generate_author_form`` is one ``json_call`` (``factory_autoform``) that the
deterministic ``AuthorForm`` schema then gates. The model's language /
display_name / script are NOT trusted — the caller's identity is forced onto
the result so the form can never disagree with --lang.
"""

import pytest

from tau2.multilingual.factory.author_form import (
    AUTOFORM_CALL_NAME,
    AuthorForm,
    FactoryDraftError,
    generate_author_form,
)
from test_multilingual.factory_testing.fake_llm import FakeFactoryLLM


def valid_form_dict() -> dict:
    return {
        "language": "ro",
        "display_name": "Romanian",
        "script": "latn",
        "locale_notes": "Standard Romanian; numbers read in pairs.",
        "personas": [
            {
                "name": "Andrei",
                "sketch": "A formal man in his 50s, careful Romanian, no slang.",
            },
            {
                "name": "Ioana",
                "sketch": "A casual woman in her 20s with heavy English insertions.",
            },
        ],
        "preferences": "Keep it natural.",
    }


class TestGenerateAuthorForm:
    def test_returns_validated_form(self):
        llm = FakeFactoryLLM({AUTOFORM_CALL_NAME: [valid_form_dict()]})
        form = generate_author_form("ro", "Romanian", "latn", llm=llm)
        assert isinstance(form, AuthorForm)
        assert form.language == "ro"
        assert [p.name for p in form.personas] == ["Andrei", "Ioana"]
        llm.assert_called(AUTOFORM_CALL_NAME, times=1)

    def test_identity_fields_are_forced(self):
        # The model returns a different identity; the caller's args win.
        rogue = valid_form_dict()
        rogue["language"] = "zz"
        rogue["display_name"] = "Wrongese"
        rogue["script"] = "deva"
        llm = FakeFactoryLLM({AUTOFORM_CALL_NAME: [rogue]})
        form = generate_author_form("ro", "Romanian", "latn", llm=llm)
        assert form.language == "ro"
        assert form.display_name == "Romanian"
        assert form.script == "latn"

    def test_reasoning_effort_defaults_to_medium(self):
        llm = FakeFactoryLLM({AUTOFORM_CALL_NAME: [valid_form_dict()]})
        generate_author_form("ro", "Romanian", "latn", llm=llm)
        assert llm.reasonings_for(AUTOFORM_CALL_NAME) == ["medium"]

    def test_hints_are_embedded_in_prompt(self):
        llm = FakeFactoryLLM({AUTOFORM_CALL_NAME: [valid_form_dict()]})
        generate_author_form(
            "ro", "Romanian", "latn", hints="Prefer Transylvanian accent.", llm=llm
        )
        assert "Transylvanian accent" in llm.prompts_for(AUTOFORM_CALL_NAME)[0]

    def test_invalid_form_raises_readable_error(self):
        # Only one persona violates AuthorForm.personas min_length=2.
        bad = valid_form_dict()
        bad["personas"] = [bad["personas"][0]]
        llm = FakeFactoryLLM({AUTOFORM_CALL_NAME: [bad]})
        with pytest.raises(FactoryDraftError) as excinfo:
            generate_author_form("ro", "Romanian", "latn", llm=llm)
        assert "personas" in str(excinfo.value)

    def test_prompt_instructs_plain_string_freeform_fields(self):
        # Prompt hardening: the autoform prompt should tell the model that the
        # freeform fields must be plain strings (reduces structured-output rate).
        llm = FakeFactoryLLM({AUTOFORM_CALL_NAME: [valid_form_dict()]})
        generate_author_form("ro", "Romanian", "latn", llm=llm)
        prompt = llm.prompts_for(AUTOFORM_CALL_NAME)[0]
        assert "locale_notes" in prompt and "preferences" in prompt
        assert "plain JSON string" in prompt


class TestFreeformStructuredValueCoercion:
    """The model intermittently returns the freeform string fields
    (``preferences``, ``locale_notes``) as a structured object/array. Coercion
    at the validation seam must turn those into readable strings instead of
    hard-failing — without losing information and without touching structured
    fields like ``personas``.
    """

    def test_dict_preferences_is_coerced_to_string(self):
        form_dict = valid_form_dict()
        form_dict["preferences"] = {
            "backchannel_density": "medium",
            "acoustic_environments": ["street", "cafe"],
            "out_of_turn": "interjects politely",
        }
        llm = FakeFactoryLLM({AUTOFORM_CALL_NAME: [form_dict]})
        form = generate_author_form("ro", "Romanian", "latn", llm=llm)
        assert isinstance(form, AuthorForm)
        assert isinstance(form.preferences, str)
        # No information is lost: keys and values survive into the string.
        assert "backchannel density" in form.preferences
        assert "medium" in form.preferences
        assert "street" in form.preferences and "cafe" in form.preferences
        assert "interjects politely" in form.preferences

    def test_dict_locale_notes_is_coerced_to_string(self):
        form_dict = valid_form_dict()
        form_dict["locale_notes"] = {
            "dialects": ["Moldavian", "Transylvanian"],
            "numbers": "read in pairs",
        }
        llm = FakeFactoryLLM({AUTOFORM_CALL_NAME: [form_dict]})
        form = generate_author_form("ro", "Romanian", "latn", llm=llm)
        assert isinstance(form.locale_notes, str)
        assert "Moldavian" in form.locale_notes
        assert "Transylvanian" in form.locale_notes
        assert "read in pairs" in form.locale_notes

    def test_list_preferences_is_coerced_to_string(self):
        form_dict = valid_form_dict()
        form_dict["preferences"] = [
            "Keep backchannels sparse.",
            "Prefer outdoor acoustics.",
        ]
        llm = FakeFactoryLLM({AUTOFORM_CALL_NAME: [form_dict]})
        form = generate_author_form("ro", "Romanian", "latn", llm=llm)
        assert isinstance(form.preferences, str)
        assert "Keep backchannels sparse." in form.preferences
        assert "Prefer outdoor acoustics." in form.preferences

    def test_plain_string_freeform_fields_pass_through_unchanged(self):
        form_dict = valid_form_dict()
        form_dict["preferences"] = "Keep it natural and warm."
        form_dict["locale_notes"] = "Standard Romanian."
        llm = FakeFactoryLLM({AUTOFORM_CALL_NAME: [form_dict]})
        form = generate_author_form("ro", "Romanian", "latn", llm=llm)
        assert form.preferences == "Keep it natural and warm."
        assert form.locale_notes == "Standard Romanian."

    def test_structured_personas_field_is_not_coerced(self):
        # personas is a list of objects and MUST stay validated as structure —
        # coercion must never flatten it into a string. A malformed persona
        # (missing required 'sketch') must still raise, not be stringified.
        form_dict = valid_form_dict()
        form_dict["personas"] = [
            {"name": "Andrei"},  # missing required 'sketch'
            {"name": "Ioana", "sketch": "Casual woman, heavy English insertions."},
        ]
        llm = FakeFactoryLLM({AUTOFORM_CALL_NAME: [form_dict]})
        with pytest.raises(FactoryDraftError) as excinfo:
            generate_author_form("ro", "Romanian", "latn", llm=llm)
        assert "personas" in str(excinfo.value)
