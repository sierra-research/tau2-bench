"""Shared authoring contract and targeted checks for retired prompt instructions.

These checks are intentionally not a multilingual semantic judge. Native review
of the exact artifact is still required. Callers select instruction fields;
task data and TTS asset specifications are not inputs.
"""

import re

AUTHORING_RULES_VERSION = "text-and-policy-boundaries-v2"
AUTHORING_RULES = """The CUSTOMER simulator produces TEXT, not acoustic output.
Author vocabulary, register, native orthography and ordinary number, date and
symbol words. Preserve tool-value formatting facts and task values. Do not coach
accent, pronunciation, phonetic realization, pitch, volume, speech rate, pauses
or enunciation. Acoustic briefs belong only in VoiceSpec at the separate TTS
asset stage, never in author-form locale notes, sketches or pack instructions.
Spelling is allowed only when explicitly asked to spell. Repeat requests and
failed lookups are not spelling requests. Local word anchors require a separate
explicit request for anchors; a spelling request alone does not request anchors.
Authentication belongs to the shared agent wrapper, which already says:
"First, authenticate the customer according to the domain policy."
The domain policy remains unchanged. Do not duplicate that instruction or add
lookup procedures, new identity requirements, fallback/retry tactics, or hints
to spell on authentication/lookup failure. Task authentication and evaluation
remain unchanged. A locale pack describes language, not authentication strategy.
The agent language clause controls conversation language and register only. Do not
require English tool values or unaccented database values: the runtime supplies
tool-value language and formatting instructions from the selected database mode.
"""

# Factory-owned drafting seed, not a fallback injected into published runtime packs.
REQUESTED_SPELLING_SECTION = """## Speaking Special Characters and Numbers
Provide names, email addresses, user IDs, numbers, and codes normally.
Spell out a value only if the agent explicitly asks you to spell it.
If the agent asks you to repeat a value, repeat it normally; a repeat request is not a spelling request.
Use letter-name anchors (such as "D as in dog") only if the agent explicitly asks for anchors.
"""

# Match quoted examples without treating every quote as a safe instruction.
# Single quotes match only at word boundaries, preserving contractions.
_QUOTED_EXAMPLE = re.compile(r'"[^"\n]*"|“[^”\n]*”|(?<!\w)\'[^\'\n]*\'(?!\w)')
_SHARED_AUTH = "First, authenticate the customer according to the domain policy."
_ACOUSTIC = re.compile(
    r"\[pause\]|\b(?:pronunciation|phonetic realization|enunciation)\b"
    r"|\b(?:pronounce|enunciate|articulate)\s+(?:each|every|all)\b"
    r"|\b(?:speak|talk|read)\s+(?:clearly|slowly|softly|loudly)\b"
    r"|\b(?:speak|talk|voice|sound)\b[^.!?\n]{0,70}\baccent\b"
    r"|\b(?:flat|Midwestern|non-native)\s+(?:\w+\s+)?accent\b"
    r"|\b(?:use|adopt|maintain)\b[^.!?\n]{0,45}\baccent\b"
    r"|\b(?:rising|falling|high|low|higher|lower)\s+(?:vocal\s+)?pitch\b"
    r"|\b(?:raise|lower|adjust|control|maintain)\b[^.!?\n]{0,30}\b(?:pitch|volume)\b"
    r"|\b(?:speech|speaking|vocal)\s+(?:rate|pace|pitch|volume)\b"
    r"|\b(?:speak|talk)\b[^.!?\n]{0,30}\b(?:slower|faster)\s+(?:rate|pace)\b"
    r"|\b(?:insert|add|use)\b[^.!?\n]{0,30}\bpauses\b",
    re.I,
)
_AUTH_PROCEDURE = re.compile(
    r"\b(?:authenticate|authentication)\b"
    r"|\b(?:ask for|require|collect|request|verify)\b[^.!?\n]{0,60}"
    r"\b(?:name|date of birth|DOB)\b"
    r"|\blook up\s+(?:the\s+)?(?:customer|account|user)\b"
    r"|\bfind\s+(?:the\s+)?(?:customer|account|user)\s+by\b"
    r"|\b(?:look\s?up|lookup)\b[^.!?\n]{0,40}\b(?:fails?|failed|unsuccessful)\b"
    r"[^.!?\n]{0,50}\b(?:ask|offer|retry|try|spell|fall back)\b"
    r"|\b(?:fall back|retry|try|use)\b[^.!?\n]{0,40}\b(?:name|phone|email) lookup\b",
    re.I,
)
_UNREQUESTED_SPELLING = re.compile(
    r"\balways\s+spell\b"
    r"|\b(?:mishears?|repeat|failed lookup)\b[^.!?\n]{0,65}"
    r"\b(?:offer to spell|spell it out|spell their|spell your)\b"
    r"|\b(?:always|automatically)\s+(?:add|use)\b[^.!?\n]{0,30}\banchors\b",
    re.I,
)


def validate_authoring_instruction(text: str, field: str) -> None:
    """Reject known retired instruction patterns in an explicitly owned field.

    Ordinary quoted conversational examples and the shared policy-only sentence
    are allowed. This guard neither edits content nor changes domain policy.
    """

    # Preserve ordinary customer utterances, but inspect quoted directives too.
    # This is a small English authoring guard, not a semantic language classifier.
    def quoted(match: re.Match[str]) -> str:
        utterance = match.group()[1:-1].strip()
        if re.search(
            r"\b(?:say|reply|respond)(?: with)?\s*$", text[: match.start()], re.I
        ):
            return ""
        if re.match(r"(?:I|we|my|our|can you|could you|would you)\b", utterance, re.I):
            return ""
        return utterance

    instruction = _QUOTED_EXAMPLE.sub(quoted, text).replace(_SHARED_AUTH, "")
    # Only direct prohibitions are exempt. Contrast/new clauses are checked on
    # their own so a negative prefix cannot hide a later positive instruction.
    clauses = re.split(r"[.!?;\n]+|,?\s+but\s+", instruction, flags=re.I)
    for clause in clauses:
        clause = clause.strip().lstrip("-* ")
        if re.match(r"(?:do not|don't|never|avoid)\b", clause, re.I):
            continue
        for rule, pattern in (
            ("acoustic coaching", _ACOUSTIC),
            ("authentication procedure", _AUTH_PROCEDURE),
            ("unsolicited spelling or anchors", _UNREQUESTED_SPELLING),
        ):
            if pattern.search(clause):
                raise ValueError(f"{field}: authoring instruction contains {rule}")
