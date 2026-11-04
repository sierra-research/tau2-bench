"""Text runs are first class: no shipped pack leaks phone-call wording into them.

Every text caller prompt (scaffold, persona behavior, shared and domain
guidance, final reminder) and the agent's language guidance is rendered for
every shipped pack, domain binding and persona, with and without user tools.
English text is checked everywhere; in other packs only the English-authored
guidance is checked, so reviewed native utterances inside quotes are skipped.
Reviewed spoken-call-only behavior lives in each persona's
``voice.pragmatics_clauses`` and must reach voice prompts only.
"""

import re

import pytest

from tau2.multilingual.runtime import list_languages, load_pack, resolve_persona
from tau2.user.guidelines import render_simulation_guidelines
from tau2.user.user_simulator import build_user_system_prompt

# Words that only make sense on a phone call. "speak(s)" is deliberately
# absent: it describes language use ("speaks Spanish", "if the agent speaks
# only English"), which is mode-neutral.
CALL_ONLY = re.compile(
    r"\b(calls?|calling|called|callers?|phones?|telephone|hang(?:s|ing)?[- ]up|"
    r"hung up|on hold|dial(?:s|ing|ed)?|voicemail|listen(?:s|ing|ed)?|"
    r"spoken|speech|ringing)\b",
    re.IGNORECASE,
)
# Legitimate uses of the same words: tool-call mechanics, "call for"
# (require), and telecom product terms.
ALLOWED = re.compile(
    r"\btool calls?\b|\bcall a tool\b|\bcalls? for\b|\bWi-?Fi calling\b|"
    r"\bphone numbers?\b",
    re.IGNORECASE,
)
# Reviewed native examples: “…” spans and standalone '…' spans (an apostrophe
# inside a word, as in "can't", never opens one).
NATIVE_QUOTED = re.compile(r"“[^”]*”|(?<![A-Za-z])'[^'\n]*'(?![A-Za-z])")


def _renders(mode="text", languages=None):
    for language in languages or list_languages():
        pack = load_pack(language)
        if mode == "voice" and pack.voice is None:
            continue
        bindings = dict(pack.domains)
        if pack.default_binding is not None:
            bindings["<default>"] = pack.default_binding
        for domain, binding in sorted(bindings.items()):
            for persona_id in sorted(pack.personas):
                persona = resolve_persona(pack, persona_id, mode=mode, binding=binding)
                for use_tools in (False, True):
                    prompt = build_user_system_prompt(
                        "<scenario>",
                        persona_config=persona,
                        mode=mode,
                        guidelines=render_simulation_guidelines(
                            persona, use_tools=use_tools
                        ),
                    )
                    agent = "\n".join(
                        (pack.agent_language_clause, persona.agent_background_clause)
                    )
                    yield (
                        f"{language}/{domain}/{persona_id}/tools={use_tools}",
                        language,
                        prompt + "\n" + agent,
                    )


def _text_renders():
    return _renders("text")


def _call_only_terms(text: str, language: str) -> list[str]:
    hits = []
    for line in text.splitlines():
        checked = line if language == "en" else NATIVE_QUOTED.sub(" ", line)
        checked = ALLOWED.sub(" ", checked)
        hits.extend(
            f"{m.group(0)!r} in: {line.strip()}" for m in CALL_ONLY.finditer(checked)
        )
    return hits


def test_text_prompts_carry_no_phone_call_wording():
    renders = list(_text_renders())
    assert {language for _, language, _ in renders} == set(list_languages())
    problems = {
        case: hits
        for case, language, text in renders
        if (hits := _call_only_terms(text, language))
    }
    assert not problems, problems


def _voice_only_clauses():
    for language in list_languages():
        for persona_id, persona in load_pack(language).personas.items():
            if persona.voice is not None:
                for clause in persona.voice.pragmatics_clauses:
                    yield language, persona_id, clause


def test_text_prompts_carry_no_voice_only_clause():
    clauses = list(_voice_only_clauses())
    assert clauses
    problems = [
        (case, clause)
        for case, _, text in _text_renders()
        for _, _, clause in clauses
        if clause in text
    ]
    assert not problems, problems


def test_voice_prompts_render_voice_only_clauses_after_mode_neutral_ones():
    for language, persona_id, clause in _voice_only_clauses():
        pack = load_pack(language)
        persona = resolve_persona(pack, persona_id, mode="voice")
        behavior = persona.to_guidelines_text("voice")
        neutral = pack.personas[persona_id].pragmatics_clauses
        assert behavior.count(clause) == 1
        assert all(behavior.index(n) < behavior.index(clause) for n in neutral)


# Reviewed native utterances that only make sense on a spoken call: phone
# openings and closings, listening idioms and continuers while the agent talks.
VOICE_ONLY_UTTERANCES = {
    ("es", "alejandro_es_v1"): ["le llamo"],
    ("es", "camila_es_v1"): ["te llamo"],
    ("ko", "soyeon_ko_v1"): ["여보세요", "끊을게요"],
    ("zh", "xiaomeng_zh_v1"): ["喂，你好", "你有没有听我说话"],
    ("zh", "jianguo_zh_v1"): ["仔细听一下"],
    ("pt", "juliana_pt_v1"): ["'uhum', 'tá', 'sei', 'aham'"],
}


@pytest.mark.parametrize(
    ("language", "persona_id"), sorted(VOICE_ONLY_UTTERANCES), ids="/".join
)
def test_call_only_native_utterances_reach_voice_prompts_only(language, persona_id):
    utterances = VOICE_ONLY_UTTERANCES[(language, persona_id)]
    for case, _, text in _renders("text", [language]):
        assert not [u for u in utterances if u in text], case
    voice = [
        (case, text)
        for case, _, text in _renders("voice", [language])
        if f"/{persona_id}/" in case
    ]
    assert voice
    for case, text in voice:
        assert all(u in text for u in utterances), case


@pytest.mark.parametrize(
    ("text", "language", "flagged"),
    [
        ("Yeah, hi, I'm calling about an order.", "en", True),
        ("A wrap-up, then you hang up.", "en", True),
        ("I've been on hold forever.", "en", True),
        ("rather than an abrupt hang-up: '好的'", "zh", True),
        ("sounds stilted on the phone", "pt", True),
        ("Make a tool call; the customer speaks Spanish.", "en", False),
        ("- Wi-Fi calling → “Wi-Fi calling”; phone number", "hi", False),
        ("examples: 'call me back', “speech”", "es", False),
        ("situations that genuinely call for softening", "zh", False),
    ],
)
def test_call_only_detector(text, language, flagged):
    assert bool(_call_only_terms(text, language)) is flagged
