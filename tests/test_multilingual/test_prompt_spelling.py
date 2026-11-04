"""Reviewed prompt parity and strict language-owned template contracts."""

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tau2.data_model.language import (
    LANGUAGE_PROMPT_VERSION,
    LanguageDirective,
    LanguagePack,
    PersonaVoice,
    RuntimePersona,
    VoiceExamples,
)
from tau2.multilingual.runtime import agent_language_guidance
from tau2.user.guidelines import (
    GUIDELINE_TEMPLATE_VERSION,
    finalize_system_prompt,
    render_simulation_guidelines,
    simulation_guidelines,
)
from tau2.user.user_simulator import UserSimulator

# Words that presuppose a spoken phone call; text-mode scaffolds never use them.
# ("tool call", "call a tool" and "call for" are not the telephony sense.)
SPOKEN_CALL_WORDS = re.compile(
    r"(?<!tool )\b(call|calls|phone|speak|spoken|caller)\b(?! for| a tool)", re.I
)


@pytest.fixture
def reviewed():
    return json.loads(
        (Path(__file__).parent / "fixtures/reviewed_prompt_content.json").read_text()
    )


def persona(reviewed, mode="voice", *, directive=False):
    """English reviewed content; ``directive`` makes it a cross-lingual caller."""
    content = reviewed["en"]
    return RuntimePersona(
        persona_id="fixture",
        mode=mode,
        gender="male",
        locale="Fixture",
        variety="Fixture variety",
        language="xx" if directive else "en",
        display_language="Fixture language" if directive else "English",
        gender_guidance=content["prompt_content"]["gender_guidance"],
        language_directive=(
            LanguageDirective(origin="Fixture region") if directive else None
        ),
        spellout_section=content["spellout_section"] if mode == "voice" else "",
        voice=(
            PersonaVoice(
                voice_id="fixture",
                backchannel_phrases=["yes"],
                voice_examples=VoiceExamples.model_validate(
                    content["prompt_content"]["voice_examples"]
                ),
            )
            if mode == "voice"
            else None
        ),
    )


@pytest.mark.parametrize("mode", ["voice", "text"])
@pytest.mark.parametrize("use_tools", [False, True])
def test_english_guidelines_match_reviewed_scaffold(reviewed, mode, use_tools):
    assert GUIDELINE_TEMPLATE_VERSION == "language-guidelines-v4"
    assert (
        render_simulation_guidelines(persona(reviewed, mode), use_tools=use_tools)
        == reviewed["en"]["guidelines"][f"{mode}:{use_tools}"]
    )


@pytest.mark.parametrize("mode", ["voice", "text"])
@pytest.mark.parametrize("use_tools", [False, True])
def test_cross_lingual_directive_is_the_only_difference(reviewed, mode, use_tools):
    base = render_simulation_guidelines(persona(reviewed, mode), use_tools=use_tools)
    cross = render_simulation_guidelines(
        persona(reviewed, mode, directive=True), use_tools=use_tools
    )
    assert cross != base
    heading = "## LANGUAGE OF THE " + ("CALL" if mode == "voice" else "CONVERSATION")
    start = cross.index(heading)
    end = cross.index("\n\n## ", start) + 2
    assert cross[:start] + cross[end:] == base
    assert "Fixture region" in cross[start:end]
    assert "Fixture variety" in cross[start:end]


@pytest.mark.parametrize("use_tools", [False, True])
def test_text_mode_never_presupposes_a_phone_call(reviewed, use_tools):
    subject = persona(reviewed, "text", directive=True).model_copy(
        update={"gender_guidance": "You are a male customer (he/him)."}
    )
    rendered = finalize_system_prompt(
        subject, render_simulation_guidelines(subject, use_tools=use_tools)
    )
    rendered += "\n" + (subject.to_guidelines_text("text") or "")
    assert not SPOKEN_CALL_WORDS.findall(rendered)
    assert "message you write" in rendered


def test_voice_directive_keeps_spoken_wording(reviewed):
    subject = persona(reviewed, "voice", directive=True)
    rendered = render_simulation_guidelines(subject, use_tools=False)
    assert "## LANGUAGE OF THE CALL (MANDATORY)" in rendered
    assert "every word you speak to the agent" in rendered
    assert "Every word you speak to the agent" in finalize_system_prompt(subject, "")


@pytest.mark.parametrize("mutation", ["missing_example", "unknown", "whitespace"])
def test_voice_examples_reject_incomplete_or_unknown_content(reviewed, mutation):
    content = reviewed["en"]["prompt_content"]["voice_examples"]
    if mutation == "missing_example":
        del content["restart_example"]
    elif mutation == "unknown":
        content["made_up_example"] = "unreviewed"
    else:
        content["restart_example"] = " \n\t"
    with pytest.raises(ValidationError):
        VoiceExamples.model_validate(content)


def test_every_example_has_a_template_slot(reviewed):
    subject = persona(reviewed)
    values = {name: f"UNIQUE_{name}_VALUE" for name in VoiceExamples.model_fields}
    subject.voice.voice_examples = VoiceExamples.model_validate(values)
    rendered = "\n".join(
        render_simulation_guidelines(subject, use_tools=tools)
        for tools in (False, True)
    )
    assert all(value in rendered for value in values.values())
    assert "${" not in rendered


@pytest.mark.parametrize("mode", ["text", "voice"])
def test_simulators_render_the_shared_scaffold_or_an_override(reviewed, mode, request):
    subject = persona(reviewed, mode)
    user = SimpleNamespace(
        persona_config=subject,
        tools=None,
        global_guidelines_override=None,
        simulation_mode=mode,
    )
    if mode == "voice":
        request.getfixturevalue("voice_extra")
        from tau2.user.user_simulator_streaming import VoiceStreamingUserSimulator

        cls = VoiceStreamingUserSimulator
    else:
        cls = UserSimulator
    assert cls.global_simulation_guidelines.fget(user) == simulation_guidelines(
        subject, mode=mode, use_tools=False
    )
    user.global_guidelines_override = "Explicit elicitation override"
    assert (
        cls.global_simulation_guidelines.fget(user) == user.global_guidelines_override
    )


def _pack_data(reviewed, *, voice: bool):
    voice_examples = reviewed["en"]["prompt_content"]["voice_examples"]
    persona_data = {
        "gender": "male",
        "locale": "Fixture",
        "variety": "Fixture",
        "gender_guidance": "You are a male customer.",
    }
    data = dict(
        schema_version=3,
        language="en",
        revision="fixture",
        display_name="English",
        agent_greeting="Hello.",
        agent_language_clause="Use English.",
        personas={"male": persona_data},
        domains={
            "mock": {
                "revision": "v1",
                "tasks_file": "tasks.json",
                "task_metadata_file": "metadata.json",
            }
        },
    )
    if voice:
        persona_data["voice"] = {
            "voice_id": "fixture",
            "backchannel_phrases": ["yes"],
            "voice_examples": voice_examples,
        }
        data["voice"] = {
            "spellout_section": reviewed["en"]["spellout_section"],
            "speech_insert_events_per_minute": 1,
            "backchannel_level": "medium",
        }
    return data


@pytest.mark.parametrize("empty", ["", " \n\t"])
def test_pack_requires_explicit_greeting_language_and_spelling(reviewed, empty):
    data = _pack_data(reviewed, voice=True)
    LanguagePack.model_validate(data)
    for key in ("agent_greeting", "agent_language_clause"):
        with pytest.raises(ValidationError):
            LanguagePack.model_validate({**data, key: empty})
    with pytest.raises(ValidationError):
        LanguagePack.model_validate(
            {**data, "voice": {**data["voice"], "spellout_section": empty}}
        )
    with pytest.raises(ValidationError):
        LanguagePack.model_validate({**data, "schema_version": 2})


def test_text_only_pack_validates_and_voice_sections_are_all_or_nothing(reviewed):
    pack = LanguagePack.model_validate(_pack_data(reviewed, voice=False))
    assert pack.voice is None
    assert all(p.voice is None for p in pack.personas.values())
    half = _pack_data(reviewed, voice=True)
    del half["voice"]
    with pytest.raises(ValidationError, match="voice section"):
        LanguagePack.model_validate(half)


def test_cross_lingual_packs_require_directive_facts(reviewed):
    data = _pack_data(reviewed, voice=False)
    with pytest.raises(ValidationError, match="language directive"):
        LanguagePack.model_validate({**data, "language": "xx"})
    data["language"] = "xx"
    data["personas"]["male"]["language_directive"] = {"origin": "Fixture region"}
    LanguagePack.model_validate(data)


@pytest.mark.parametrize("gender", ["female", "male", None])
@pytest.mark.parametrize("cross_lingual", [False, True])
def test_agent_guidance_is_short_and_uses_actual_voice(gender, cross_lingual):
    context = SimpleNamespace(
        cross_lingual=cross_lingual,
        scenario_mode="pack",
        persona=SimpleNamespace(agent_background_clause=""),
        agent_language_clause="Use English.",
        domain_agent_guidance=None,
    )
    text = agent_language_guidance(context, agent_voice_gender=gender)
    assert text.startswith("Use English.")
    assert ("Tool names" in text) == cross_lingual
    assert text.endswith(f"\nYour gender is {gender}.") == (gender is not None)
    assert LANGUAGE_PROMPT_VERSION == "language-runtime-v8"


@pytest.mark.parametrize("mode", ["text", "voice"])
def test_persona_rendering_rejects_the_wrong_mode(reviewed, mode):
    subject = persona(reviewed, mode)
    other = "voice" if mode == "text" else "text"
    assert subject.to_guidelines_text(mode)
    with pytest.raises(ValueError, match="resolved for"):
        subject.to_guidelines_text(other)


@pytest.mark.parametrize("use_tools", [False, True])
def test_language_text_prompt_contains_complete_persona_once(
    reviewed, use_tools, monkeypatch
):
    monkeypatch.setattr(
        "tau2.user.user_simulator.generate",
        lambda *args, **kwargs: pytest.fail("Prompt assembly must not call a model"),
    )
    subject = persona(reviewed, "text", directive=True).model_copy(
        update={
            "pragmatics_clauses": ["REVIEWED_REGISTER_BEHAVIOR"],
            "mode_guidelines": "REVIEWED_TEXT_DOMAIN_GUIDANCE",
        }
    )
    user = UserSimulator(
        llm="offline-fixture",
        instructions="An offline scenario.",
        tools=[] if use_tools else None,
        persona_config=subject,
    )
    rendered = user.system_prompt
    for clause in (
        subject.gender_guidance,
        "REVIEWED_REGISTER_BEHAVIOR",
        "REVIEWED_TEXT_DOMAIN_GUIDANCE",
    ):
        assert rendered.count(clause) == 1
        assert rendered.index(clause) < rendered.index("## Task Completion")
    assert "<PERSONA_GUIDELINES>" not in rendered
    assert rendered.rstrip().endswith(
        "Every message you write to the agent is in Fixture language — never in "
        "English.\nSpecial tokens (###STOP###, ###TRANSFER###, ###OUT-OF-SCOPE###) "
        "stay verbatim, in ASCII, exactly as written."
    )
