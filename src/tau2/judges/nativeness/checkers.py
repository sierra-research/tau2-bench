# Copyright Sierra
"""Deterministic nativeness checkers.

Each checker is a pure function ``(CheckerContext) -> (JudgeOutcome, evidence)``.
Adding a checker means adding a function + a ``CHECKER_REGISTRY`` entry; the
aggregator never changes. Checkers lean toward precision: when in doubt they
return NO_OPPORTUNITY rather than a false FAIL.
"""

import re
from typing import Annotated, Callable, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from tau2.data_model.simulation import JudgeOutcome
from tau2.judges.nativeness.caller_identity import CallerNameContext
from tau2.judges.nativeness.factors import ENGLISH_SYMBOL_LEAKS


class CheckerContext(BaseModel):
    """Everything a checker needs about one simulation."""

    agent_text: str = Field(
        description="Concatenated agent speech (gold transcript / content)"
    )
    has_email: bool = Field(description="Whether the task carries an email value")
    language: str = Field(description="Resolved run language (lower-cased ISO 639-1)")
    agent_turns: list[str] = Field(
        default_factory=list,
        description="Delivered agent utterances, kept separate for turn-level checks.",
    )
    agent_turn_interruptions: list[bool] = Field(
        default_factory=list,
        description="Whether each corresponding delivered agent utterance was "
        "interrupted; empty means no interruption metadata was available.",
    )
    user_turns: list[str] = Field(
        default_factory=list,
        description="Delivered user utterances, kept separate for opportunity gates.",
    )
    user_speech_seconds: Optional[float] = Field(
        default=None,
        ge=0,
        description="Tick-aligned user speech duration, or None when unavailable.",
    )
    is_voice: bool = Field(
        default=False,
        description="Whether the transcript came from a voice/full-duplex run.",
    )
    allowed_literals: list[str] = Field(
        default_factory=list,
        description="Task values that must not be treated as agent-authored script.",
    )
    agent_gender: Optional[Literal["male", "female"]] = Field(
        default=None,
        description="Known agent voice gender; never inferred from a name.",
    )
    caller_gender: Optional[Literal["male", "female"]] = Field(
        default=None,
        description="Known caller persona gender; never inferred from a name.",
    )
    caller_name: Optional[CallerNameContext] = Field(
        default=None,
        description="Known localized Korean caller name from typed identity metadata.",
    )
    email_symbols: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Native email-symbol verbalizations for this language "
        "(from the pack); empty makes the email checker a no-op",
    )
    pinned_greeting: Optional[str] = Field(
        default=None,
        description="The harness-injected agent opening greeting, extracted "
        "structurally from the simulation (never keyed on current pack text). "
        "Pinned pack content, not model output: the gender checkers skip any "
        "agent turn equal to it.",
    )

    @model_validator(mode="after")
    def _interruptions_align_with_agent_turns(self) -> "CheckerContext":
        """Keep the per-turn interruption signal structurally aligned."""
        if self.agent_turn_interruptions and len(self.agent_turn_interruptions) != len(
            self.agent_turns
        ):
            raise ValueError(
                "agent_turn_interruptions must align one-to-one with agent_turns"
            )
        return self


CheckerFn = Callable[[CheckerContext], tuple[JudgeOutcome, Optional[str]]]


class BackchannelFrequencySpec(BaseModel):
    """Fixed deterministic rubric for one language's agent acknowledgments."""

    acknowledgments: Annotated[
        tuple[str, ...],
        Field(
            description="Common acknowledgment phrases counted at most once per turn."
        ),
    ]
    min_user_speech_seconds: Annotated[
        float,
        Field(
            default=30.0,
            gt=0,
            description="Minimum user speech duration required for an opportunity.",
        ),
    ]
    min_substantive_user_turns: Annotated[
        int,
        Field(
            default=2,
            ge=1,
            description="Minimum non-acknowledgment user turns required.",
        ),
    ]
    pass_min_per_minute: Annotated[
        float,
        Field(default=1.0, ge=0, description="Inclusive lower passing rate."),
    ]
    pass_max_per_minute: Annotated[
        float,
        Field(default=6.0, gt=0, description="Inclusive upper passing rate."),
    ]


BACKCHANNEL_FREQUENCY_VERSION = "v1"

# Fixed, reviewed lexical examples. Matching is deliberately permissive: an
# acknowledgment counts whether it stands alone or opens a substantive reply.
BACKCHANNEL_FREQUENCY_SPECS: dict[str, BackchannelFrequencySpec] = {
    "en": BackchannelFrequencySpec(
        acknowledgments=(
            "mm-hmm",
            "mhm",
            "uh-huh",
            "yes",
            "yeah",
            "okay",
            "ok",
            "right",
            "got it",
            "I see",
            "understood",
            "sure",
        )
    ),
    "hi": BackchannelFrequencySpec(
        acknowledgments=(
            "जी",
            "हाँ",
            "हां",
            "ठीक है",
            "समझ गया",
            "समझ गई",
            "अच्छा",
            "बिल्कुल",
        )
    ),
    "ko": BackchannelFrequencySpec(
        acknowledgments=(
            "네",
            "예",
            "알겠습니다",
            "그렇군요",
            "좋아요",
            "맞습니다",
        )
    ),
    "zh": BackchannelFrequencySpec(
        acknowledgments=("好的", "明白", "明白了", "嗯", "是的", "可以"),
    ),
    "pt": BackchannelFrequencySpec(
        acknowledgments=(
            "sim",
            "certo",
            "tá",
            "está bem",
            "entendi",
            "claro",
            "perfeito",
        )
    ),
    "es": BackchannelFrequencySpec(
        acknowledgments=(
            "sí",
            "claro",
            "vale",
            "de acuerdo",
            "entiendo",
            "entendido",
            "perfecto",
            "bien",
        )
    ),
}


def _word_present(token: str, text_lower: str) -> bool:
    """Whole-word match for ASCII tokens; substring match for non-ASCII tokens."""
    token = token.lower()
    if token.isascii():
        return re.search(rf"\b{re.escape(token)}\b", text_lower) is not None
    return token in text_lower


def _phrase_present(phrase: str, text: str) -> bool:
    """Match a phrase without letting short tokens fire inside larger words."""
    return (
        re.search(
            rf"(?<!\w){re.escape(phrase.casefold())}(?!\w)",
            text.casefold(),
        )
        is not None
    )


def _is_acknowledgment_only(text: str, spec: BackchannelFrequencySpec) -> bool:
    """Whether a user turn contains only acknowledgments and punctuation."""
    remainder = text.casefold()
    for phrase in sorted(spec.acknowledgments, key=len, reverse=True):
        remainder = re.sub(
            rf"(?<!\w){re.escape(phrase.casefold())}(?!\w)", " ", remainder
        )
    return re.sub(r"[^\w]+", "", remainder, flags=re.UNICODE) == ""


def check_backchannel_frequency(
    ctx: CheckerContext,
) -> tuple[JudgeOutcome, Optional[str]]:
    """Score agent acknowledgment frequency per minute of user speech.

    Opportunity requires at least 30 seconds of tick-aligned user speech and at
    least two user turns that are more than acknowledgments. Each agent turn
    contributes at most one count, even if it contains several acknowledgment
    words or continues with a substantive response.
    """
    spec = BACKCHANNEL_FREQUENCY_SPECS.get(ctx.language)
    if spec is None or ctx.user_speech_seconds is None:
        return JudgeOutcome.NO_OPPORTUNITY, None

    substantive_turns = sum(
        bool(turn.strip()) and not _is_acknowledgment_only(turn, spec)
        for turn in ctx.user_turns
    )
    if (
        ctx.user_speech_seconds < spec.min_user_speech_seconds
        or substantive_turns < spec.min_substantive_user_turns
    ):
        return JudgeOutcome.NO_OPPORTUNITY, None

    count = sum(
        any(_phrase_present(phrase, turn) for phrase in spec.acknowledgments)
        for turn in ctx.agent_turns
    )
    user_minutes = ctx.user_speech_seconds / 60.0
    rate = count / user_minutes
    evidence = (
        f"backchannel-frequency-{BACKCHANNEL_FREQUENCY_VERSION}: "
        f"{count} acknowledgment turn(s) / {user_minutes:.2f} user-speech "
        f"minute(s) = {rate:.2f}/min; {substantive_turns} substantive user turn(s); "
        f"pass range [{spec.pass_min_per_minute:.0f}, "
        f"{spec.pass_max_per_minute:.0f}]/min"
    )
    if rate < spec.pass_min_per_minute:
        return JudgeOutcome.FAIL, f"fail-low: {evidence}"
    if rate > spec.pass_max_per_minute:
        return JudgeOutcome.FAIL, f"fail-high: {evidence}"
    return JudgeOutcome.PASS, evidence


def check_email_symbol_verbalization(
    ctx: CheckerContext,
) -> tuple[JudgeOutcome, Optional[str]]:
    """Email symbols (@ .) should be read with native words, not English ones.

    Gated on the task carrying an email (so "at"/"dot" elsewhere can't fire) and
    on an accept-list existing for the language (from the pack). FAILs only on an
    English leak word that is NOT native for this language.
    """
    if not ctx.is_voice or not ctx.has_email:
        return JudgeOutcome.NO_OPPORTUNITY, None
    accept = ctx.email_symbols
    if not accept:
        return JudgeOutcome.NO_OPPORTUNITY, None

    text = ctx.agent_text.lower()
    accept_tokens = {t.lower() for tokens in accept.values() for t in tokens}
    saw_native = False

    for symbol, leak_words in ENGLISH_SYMBOL_LEAKS.items():
        native_for_symbol = [t.lower() for t in accept.get(symbol, [])]
        if any(_word_present(t, text) for t in native_for_symbol):
            saw_native = True
            continue  # this symbol was read natively
        # Symbol not read natively — did the agent fall back to an English word
        # that is not itself native for this language?
        for word in leak_words:
            if word in accept_tokens:
                continue
            if _word_present(word, text):
                return (
                    JudgeOutcome.FAIL,
                    f"English '{word}' used for '{symbol}' instead of a native token",
                )

    if saw_native:
        return JudgeOutcome.PASS, None
    # Task had an email but the agent never verbalized a symbol we recognise.
    return JudgeOutcome.NO_OPPORTUNITY, None


# v5: ko honorific_levels retired (owner ruling 2026-09-01) — zero failures
# in the recall_20 calibration corpus and every historical firing was a
# fragment-split false positive. es/hi register checkers unchanged in text.
REGISTER_CHECKER_VERSION = "v5"

_QUOTED_SPAN_RE = re.compile(
    r'"[^"\n]*"|“[^”\n]*”|‘[^’\n]*’|«[^»\n]*»|「[^」\n]*」|『[^』\n]*』'
)

_HI_FORMAL_ADDRESS = frozenset({"आप", "आपको", "आपका", "आपकी", "आपके", "आपसे", "आपने"})
_HI_FAMILIAR_ADDRESS = frozenset(
    {
        "तुम",
        "तुमको",
        "तुम्हें",
        "तुम्हे",
        "तुम्हारा",
        "तुम्हारी",
        "तुम्हारे",
        "तुमसे",
        "तुमने",
        "तू",
        "तुझे",
        "तुझको",
        "तुझसे",
        "तूने",
        "तेरा",
        "तेरी",
        "तेरे",
    }
)
_DEVANAGARI_TOKEN_RE = re.compile(r"[\u0900-\u097f]+")


def _without_allowed_literals(text: str, allowed_literals: list[str]) -> str:
    """Remove exact task-supplied values before attributing wording to the agent."""
    for literal in sorted(
        (value for value in allowed_literals if value), key=len, reverse=True
    ):
        text = re.sub(re.escape(literal), "", text, flags=re.IGNORECASE)
    return text


def _register_text(ctx: CheckerContext) -> str:
    """Agent-authored register evidence after literal and quotation stripping."""
    text = _without_allowed_literals(ctx.agent_text, ctx.allowed_literals)
    return _QUOTED_SPAN_RE.sub("", text)


def check_hi_register_formality(
    ctx: CheckerContext,
) -> tuple[JudgeOutcome, Optional[str]]:
    """Require polite Hindi customer address when a pronoun is present."""
    if ctx.language != "hi":
        return JudgeOutcome.NO_OPPORTUNITY, None

    text = _register_text(ctx)
    tokens = set(_DEVANAGARI_TOKEN_RE.findall(text))
    familiar = sorted(tokens & _HI_FAMILIAR_ADDRESS)
    if familiar:
        return (
            JudgeOutcome.FAIL,
            f"register-{REGISTER_CHECKER_VERSION}: familiar customer address "
            f"used ({', '.join(familiar)})",
        )
    formal = sorted(tokens & _HI_FORMAL_ADDRESS)
    if formal:
        return (
            JudgeOutcome.PASS,
            f"register-{REGISTER_CHECKER_VERSION}: polite आप-family address used",
        )
    return JudgeOutcome.NO_OPPORTUNITY, None


_ES_FORMAL_ADDRESS = frozenset({"usted", "ustedes"})
_ES_INFORMAL_ADDRESS = frozenset(
    {"tú", "tu", "tus", "te", "ti", "contigo", "vosotros", "vosotras", "os"}
)
_LETTER_TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def check_es_register_formality(
    ctx: CheckerContext,
) -> tuple[JudgeOutcome, Optional[str]]:
    """Require one consistent strong Spanish customer-address family."""
    if ctx.language != "es":
        return JudgeOutcome.NO_OPPORTUNITY, None

    text = _register_text(ctx)
    tokens = set(_LETTER_TOKEN_RE.findall(text.casefold()))
    formal = sorted(tokens & _ES_FORMAL_ADDRESS)
    informal = sorted(tokens & _ES_INFORMAL_ADDRESS)
    if formal and informal:
        return (
            JudgeOutcome.FAIL,
            f"register-{REGISTER_CHECKER_VERSION}: mixed usted-family "
            f"({', '.join(formal)}) and tú-family ({', '.join(informal)}) address",
        )
    if formal:
        return (
            JudgeOutcome.PASS,
            f"register-{REGISTER_CHECKER_VERSION}: consistent usted-family address",
        )
    if informal:
        return (
            JudgeOutcome.PASS,
            f"register-{REGISTER_CHECKER_VERSION}: consistent tú-family address",
        )
    return JudgeOutcome.NO_OPPORTUNITY, None


def check_zh_register_formality(
    ctx: CheckerContext,
) -> tuple[JudgeOutcome, Optional[str]]:
    """Retired: Mandarin register is never scored.

    v4: owner ruling from recall_20 calibration. Native raters accepted
    most informal 你-address service calls (including 53 uses with no 您
    outside the canned greeting); the checker's confirmed catches (2) did
    not justify its false-positive profile (3), so 你/您 choice is not
    scored in Mandarin.
    """
    return JudgeOutcome.NO_OPPORTUNITY, None


# _KO_SENTENCE_RE also serves the ko honorific_agreement hybrid precheck.
_KO_SENTENCE_RE = re.compile(r"[^.!?。！？\n]+")


def check_ko_honorific_levels(
    ctx: CheckerContext,
) -> tuple[JudgeOutcome, Optional[str]]:
    """Retired: Korean honorific level is never scored.

    v5: owner ruling 2026-09-01. Zero failures in the recall_20 calibration
    corpus (19 pass / 1 no-opportunity), so the factor carried no validated
    signal, and every historical firing (2 in airline_v1, 4 in xai_v2) was a
    fragment-split false positive — polite sentences cut mid-clause at
    transcript breaks, with the 해$ banmal ending matching truncations like
    "이해".
    """
    return JudgeOutcome.NO_OPPORTUNITY, None


# v2: the harness-injected opening greeting (``CheckerContext.pinned_greeting``)
# is never scored — it is pinned pack content, not model output. The historical
# hi pack greeting was masculine ("… कर सकता हूँ") while every pinned agent
# voice is female, so v1 contradicted the known agent gender with our own
# pinned line. The es/pt greetings are gender-neutral; the guard applies to all
# three checkers on principle.
# v3 (owner ruling 2026-09-03): with honorific आप, masculine plural is the
# unmarked polite form for ANY addressee, so आप + masculine plural toward a
# known female caller is no longer a contradiction and that rule is removed.
# Feminine plural toward a known male caller stays a failure — the feminine
# form is marked, never a neutral default. Spoken slash-alternatives such as
# "मैं सुन रहा/रही हूँ" are now a deterministic failure regardless of gender:
# a speaker commits to one form.
GENDER_AGREEMENT_CHECKER_VERSION = "v3"

_HI_AGENT_MASCULINE_RE = re.compile(r"मैं\s+[^।.!?\n]{0,48}?सकता\s+ह(?:ूँ|ूं)")
_HI_AGENT_FEMININE_RE = re.compile(r"मैं\s+[^।.!?\n]{0,48}?सकती\s+ह(?:ूँ|ूं)")
_HI_CALLER_FEMININE_RE = re.compile(r"आप\s+[^।.!?\n]{0,48}?रही\s+हैं")
# Both gender alternatives voiced in one first-person form, e.g. "रहा/रही",
# "देता/देती", "सकूँगा/सकूँगी". Slash or pipe between the two variants.
# NOTE: no \b — Devanagari matras are combining marks, not word chars to
# Python's re, so \b misfires after them; an explicit lookahead ends the match.
_HI_SLASH_ALTERNATIVE_RE = re.compile(
    r"(?:[क-ह][ा-ौँ-ः]*)?(?:ता|ती|रहा|रही|गा|गी|ूँगा|ूँगी)\s*[/|]\s*"
    r"(?:[क-ह][ा-ौँ-ः]*)?(?:ता|ती|रहा|रही|गा|गी|ूँगा|ूँगी)(?![क-हा-ौ])"
)
_ES_AGENT_MASCULINE_RE = re.compile(r"(?<!\w)estoy\s+(?:listo|seguro)(?!\w)")
_ES_AGENT_FEMININE_RE = re.compile(r"(?<!\w)estoy\s+(?:lista|segura)(?!\w)")
_PT_AGENT_MASCULINE_RE = re.compile(r"(?<!\w)estou\s+pronto(?!\w)")
_PT_AGENT_FEMININE_RE = re.compile(r"(?<!\w)estou\s+pronta(?!\w)")


def _agent_turns_without_literals(ctx: CheckerContext) -> list[str]:
    """Return separate agent turns after removing exact task-provided values."""
    return [
        _without_allowed_literals(turn, ctx.allowed_literals)
        for turn in (ctx.agent_turns or [ctx.agent_text])
    ]


def _gender_agent_turns(ctx: CheckerContext) -> list[str]:
    """Agent turns eligible for gender scoring, minus the pinned greeting.

    The harness-injected opening greeting is pack content, not model output;
    a known participant gender must never be contradicted by our own pinned
    line. The greeting is matched by exact stripped-text equality against the
    structurally extracted ``ctx.pinned_greeting``, so frozen sims carrying an
    older pack greeting are handled without keying on current pack text.
    Literal stripping happens after the exclusion, as in
    ``_agent_turns_without_literals``.
    """
    turns = ctx.agent_turns or [ctx.agent_text]
    pinned = (ctx.pinned_greeting or "").strip()
    if pinned:
        turns = [turn for turn in turns if turn.strip() != pinned]
    return [_without_allowed_literals(turn, ctx.allowed_literals) for turn in turns]


def _direct_title_pattern(title: str) -> re.Pattern[str]:
    """Match a title only in a punctuated vocative/direct-address position."""
    return re.compile(
        rf"(?:^|[,;:!?¿¡]\s*){re.escape(title)}\s*[,;:!?.]",
        flags=re.IGNORECASE,
    )


def _gender_failure(
    *, role: str, expected: str, phrase: str, utterance: str
) -> tuple[JudgeOutcome, Optional[str]]:
    return (
        JudgeOutcome.FAIL,
        f"gender-{GENDER_AGREEMENT_CHECKER_VERSION}: known {expected} {role} "
        f"contradicted by {phrase!r} — in: {_evidence_snippet(utterance)!r}",
    )


def _evidence_snippet(utterance: str, limit: int = 200) -> str:
    text = " ".join(utterance.replace("###STOP###", " ").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def check_hi_gender_agreement(
    ctx: CheckerContext,
) -> tuple[JudgeOutcome, Optional[str]]:
    """Fail only unambiguous Hindi agreement contradictions tied to a participant."""
    if ctx.language != "hi":
        return JudgeOutcome.NO_OPPORTUNITY, None

    for turn in _gender_agent_turns(ctx):
        if ctx.agent_gender == "female" and _HI_AGENT_MASCULINE_RE.search(turn):
            return _gender_failure(
                role="agent", expected="female", phrase="मैं … सकता हूँ", utterance=turn
            )
        if ctx.agent_gender == "male" and _HI_AGENT_FEMININE_RE.search(turn):
            return _gender_failure(
                role="agent", expected="male", phrase="मैं … सकती हूँ", utterance=turn
            )
        if ctx.caller_gender == "male" and _HI_CALLER_FEMININE_RE.search(turn):
            return _gender_failure(
                role="caller", expected="male", phrase="आप … रही हैं", utterance=turn
            )
        if _HI_SLASH_ALTERNATIVE_RE.search(turn):
            return (
                JudgeOutcome.FAIL,
                f"gender-{GENDER_AGREEMENT_CHECKER_VERSION}: spoken slash-alternative "
                "gender forms (e.g. 'रहा/रही') — the agent must commit to one form"
                f" — in: {_evidence_snippet(turn)!r}",
            )
    return JudgeOutcome.NO_OPPORTUNITY, None


def check_es_gender_agreement(
    ctx: CheckerContext,
) -> tuple[JudgeOutcome, Optional[str]]:
    """Fail only unambiguous Spanish agreement contradictions."""
    if ctx.language != "es":
        return JudgeOutcome.NO_OPPORTUNITY, None

    masculine_title = _direct_title_pattern("señor")
    feminine_title = _direct_title_pattern("señora")
    for turn in _gender_agent_turns(ctx):
        folded = turn.casefold()
        if ctx.agent_gender == "female" and _ES_AGENT_MASCULINE_RE.search(folded):
            return _gender_failure(
                role="agent",
                expected="female",
                phrase="estoy listo/seguro",
                utterance=turn,
            )
        if ctx.agent_gender == "male" and _ES_AGENT_FEMININE_RE.search(folded):
            return _gender_failure(
                role="agent",
                expected="male",
                phrase="estoy lista/segura",
                utterance=turn,
            )
        if ctx.caller_gender == "female" and masculine_title.search(turn):
            return _gender_failure(
                role="caller",
                expected="female",
                phrase="vocative señor",
                utterance=turn,
            )
        if ctx.caller_gender == "male" and feminine_title.search(turn):
            return _gender_failure(
                role="caller",
                expected="male",
                phrase="vocative señora",
                utterance=turn,
            )
    return JudgeOutcome.NO_OPPORTUNITY, None


def check_pt_gender_agreement(
    ctx: CheckerContext,
) -> tuple[JudgeOutcome, Optional[str]]:
    """Fail only unambiguous Portuguese agreement contradictions.

    ``obrigado``/``obrigada`` are deliberately not anchors: Brazilian usage
    permits invariant, interjection-like ``obrigado``, and the reverse direction
    has not been calibrated precisely enough for deterministic scoring.
    """
    if ctx.language != "pt":
        return JudgeOutcome.NO_OPPORTUNITY, None

    masculine_title = _direct_title_pattern("senhor")
    feminine_title = _direct_title_pattern("senhora")
    for turn in _gender_agent_turns(ctx):
        folded = turn.casefold()
        if ctx.agent_gender == "female" and _PT_AGENT_MASCULINE_RE.search(folded):
            return _gender_failure(
                role="agent", expected="female", phrase="estou pronto", utterance=turn
            )
        if ctx.agent_gender == "male" and _PT_AGENT_FEMININE_RE.search(folded):
            return _gender_failure(
                role="agent", expected="male", phrase="estou pronta", utterance=turn
            )
        if ctx.caller_gender == "female" and masculine_title.search(turn):
            return _gender_failure(
                role="caller",
                expected="female",
                phrase="vocative senhor",
                utterance=turn,
            )
        if ctx.caller_gender == "male" and feminine_title.search(turn):
            return _gender_failure(
                role="caller",
                expected="male",
                phrase="vocative senhora",
                utterance=turn,
            )
    return JudgeOutcome.NO_OPPORTUNITY, None


# v2: an interrupted final Hindi ``बता`` is an ambiguous prefix of the
# formal ``बताइए``, not a definite familiar imperative. The raw delivered
# transcript remains intact; only this truncation-sensitive precheck match is
# suppressed so the interruption-aware LLM judge handles the unresolved case.
HONORIFIC_AGREEMENT_CHECKER_VERSION = "v2"


class HonorificAgreementMarkerSpec(BaseModel):
    """One locally reviewed honorific-agreement contradiction."""

    id: Annotated[str, Field(description="Stable marker identifier for evidence.")]
    pattern: Annotated[
        str,
        Field(description="Unicode-aware same-clause contradiction pattern."),
    ]
    provenance: Annotated[
        str,
        Field(description="Reviewed pack or calibration source for the marker."),
    ]


_HI_FAMILIAR_IMPERATIVE = r"(?:बताओ|करो|लो|दो)"
_HI_CLAUSE_RE = re.compile(r"[^।.!?;\n]+")
_HI_DIRECT_CUSTOMER_ANCHOR_RE = re.compile(r"(?<!\w)(?:आप|आपको)(?!\w)")
_HI_FORMAL_CUSTOMER_PREDICATE_RE = re.compile(
    r"(?:हैं|सकते\s+हैं|चाहिए|होगा|होगी|दीजिए|लीजिए|कीजिए|बताइए|"
    r"कहिए|देखिए|लें|दें|करें)\s*$"
)
_HI_BARE_FAMILIAR_FRAGMENT_RE = re.compile(
    rf"^\s*(?:(?:अब|फिर|तो|कृपया)\s+)?{_HI_FAMILIAR_IMPERATIVE}\s*$"
)

_KO_SELF_HONORIFIC_ACTIONS = (
    r"(?:확인하시겠습니다|확인하실게요|처리하시겠습니다|처리하실게요|"
    r"안내하시겠습니다|안내하실게요|조회하시겠습니다|조회하실게요|"
    r"변경하시겠습니다|변경하실게요|취소하시겠습니다|취소하실게요|"
    r"환불하시겠습니다|환불하실게요|보내시겠습니다|도와주시겠습니다|"
    r"말씀하시겠습니다)"
)

# Sources are the current reviewed hi/ko pack rubrics and the Korean cold-eval
# calibration rows documented in ROUND3_JUDGE_CALIBRATION.md. The catalogs are
# intentionally bounded: omitted subjects, ordinary speech-level endings, and
# unreviewed Korean predicates remain for the LLM.
HONORIFIC_AGREEMENT_MARKERS: dict[str, tuple[HonorificAgreementMarkerSpec, ...]] = {
    "hi": (
        HonorificAgreementMarkerSpec(
            id="familiar_imperative_after_aap",
            pattern=(
                rf"(?<!\w)(?:आप|आपको)(?!\w)"
                rf"(?:(?!(?:कि|वह|वे|ग्राहक|एजेंट)(?:\s|$))"
                rf"[^।.!?;\n]){{0,48}}?(?<!\w){_HI_FAMILIAR_IMPERATIVE}\s*$"
            ),
            provenance="hi_pack_honorific_agreement_rubric",
        ),
        HonorificAgreementMarkerSpec(
            id="singular_ready_after_aap",
            pattern=(
                r"(?<!\w)आप(?!\w)\s+"
                r"(?:(?:अब|अभी|तो|भी|कृपया)\s+){0,3}तैयार\s+है(?!ं)"
            ),
            provenance="hi_pack_honorific_agreement_rubric",
        ),
        HonorificAgreementMarkerSpec(
            id="singular_modal_after_aap",
            pattern=(
                r"(?<!\w)आप(?!\w)\s+"
                r"(?:(?:यह|इसे|अब|अभी|तो|भी)\s+){0,3}"
                r"कर\s+सकत(?:ा|ी)\s+है(?!ं)"
            ),
            provenance="hi_pack_honorific_agreement_rubric",
        ),
        HonorificAgreementMarkerSpec(
            id="bare_bata_after_aap",
            pattern=r"(?<!\w)आप(?!\w)\s+(?:अब\s+)?बता\s*$",
            provenance="hi_pack_honorific_agreement_rubric",
        ),
    ),
    "ko": (
        HonorificAgreementMarkerSpec(
            id="self_honorification",
            pattern=(
                rf"(?<!\w)(?:제가|저희가)(?!\w)"
                rf"(?:(?!고객님(?:께서|이))[^.!?。！？\n]){{0,48}}?"
                rf"{_KO_SELF_HONORIFIC_ACTIONS}"
            ),
            provenance="ko_pack_honorific_agreement_rubric",
        ),
        HonorificAgreementMarkerSpec(
            id="customer_application_without_si",
            pattern=r"고객님께서\s+신청한\s+예약",
            provenance="ko_round3_reviewed_honorific_agreement_example",
        ),
    ),
}


def _honorific_failure(
    marker: HonorificAgreementMarkerSpec, *, turn_index: int
) -> tuple[JudgeOutcome, Optional[str]]:
    return (
        JudgeOutcome.FAIL,
        f"honorific-agreement-{HONORIFIC_AGREEMENT_CHECKER_VERSION}: agent turn "
        f"{turn_index + 1} used {marker.id}; source={marker.provenance}",
    )


def _hi_real_customer_anchor(clause: str) -> bool:
    """Whether a Hindi clause unmistakably establishes the customer as addressee."""
    return bool(
        _HI_DIRECT_CUSTOMER_ANCHOR_RE.search(clause)
        and (
            clause.lstrip().startswith("क्या आप")
            or _HI_FORMAL_CUSTOMER_PREDICATE_RE.search(clause)
        )
    )


def check_hi_honorific_agreement(
    ctx: CheckerContext,
) -> tuple[JudgeOutcome, Optional[str]]:
    """Fail bounded आप-agreement contradictions; defer every other Hindi case."""
    if ctx.language != "hi":
        return JudgeOutcome.NO_OPPORTUNITY, None

    markers = HONORIFIC_AGREEMENT_MARKERS["hi"]
    previous_last_clause: Optional[str] = None
    for turn_index, turn in enumerate(_agent_turns_without_literals(ctx)):
        turn_interrupted = bool(
            ctx.agent_turn_interruptions and ctx.agent_turn_interruptions[turn_index]
        )
        unquoted = _QUOTED_SPAN_RE.sub("", turn)
        cut_tail_is_bare_bata = bool(
            turn_interrupted and re.search(r"(?<!\w)बता\s*$", unquoted)
        )
        clauses = [clause.strip() for clause in _HI_CLAUSE_RE.findall(unquoted)]
        clauses = [clause for clause in clauses if clause]
        for clause_index, clause in enumerate(clauses):
            for marker in markers:
                if (
                    cut_tail_is_bare_bata
                    and clause_index == len(clauses) - 1
                    and marker.id == "bare_bata_after_aap"
                ):
                    continue
                if re.search(marker.pattern, clause):
                    return _honorific_failure(marker, turn_index=turn_index)

        # One immediately adjacent bare familiar command may inherit a real
        # customer anchor. Nothing is carried beyond that single boundary.
        for first, second in zip(clauses, clauses[1:]):
            if _hi_real_customer_anchor(first) and _HI_BARE_FAMILIAR_FRAGMENT_RE.search(
                second
            ):
                return (
                    JudgeOutcome.FAIL,
                    f"honorific-agreement-{HONORIFIC_AGREEMENT_CHECKER_VERSION}: "
                    f"agent turn {turn_index + 1} followed explicit आप-address "
                    "with a familiar imperative; "
                    "source=hi_pack_honorific_agreement_rubric",
                )
        if (
            previous_last_clause is not None
            and clauses
            and _hi_real_customer_anchor(previous_last_clause)
            and _HI_BARE_FAMILIAR_FRAGMENT_RE.search(clauses[0])
        ):
            return (
                JudgeOutcome.FAIL,
                f"honorific-agreement-{HONORIFIC_AGREEMENT_CHECKER_VERSION}: "
                f"agent turn {turn_index + 1} immediately followed explicit "
                "आप-address with a familiar imperative; "
                "source=hi_pack_honorific_agreement_rubric",
            )
        previous_last_clause = clauses[-1] if clauses else None

    return JudgeOutcome.NO_OPPORTUNITY, None


def check_ko_honorific_agreement(
    ctx: CheckerContext,
) -> tuple[JudgeOutcome, Optional[str]]:
    """Fail only reviewed same-clause Korean subject-honorific contradictions."""
    if ctx.language != "ko":
        return JudgeOutcome.NO_OPPORTUNITY, None

    markers = HONORIFIC_AGREEMENT_MARKERS["ko"]
    for turn_index, turn in enumerate(_agent_turns_without_literals(ctx)):
        unquoted = _QUOTED_SPAN_RE.sub("", turn)
        for clause in _KO_SENTENCE_RE.findall(unquoted):
            for marker in markers:
                if re.search(marker.pattern, clause):
                    return _honorific_failure(marker, turn_index=turn_index)
    return JudgeOutcome.NO_OPPORTUNITY, None


REGIONAL_CONSISTENCY_CHECKER_VERSION = "v1"


class RegionalMarkerSpec(BaseModel):
    """One high-precision non-target regional marker and its provenance."""

    id: Annotated[str, Field(description="Stable marker identifier for evidence.")]
    category: Annotated[
        Literal["grammar", "lexicon"],
        Field(description="Linguistic feature family represented by the marker."),
    ]
    pattern: Annotated[
        str,
        Field(description="Unicode-aware regular expression for agent-turn text."),
    ]
    provenance: Annotated[
        str,
        Field(description="Reviewed local or authoritative source for the marker."),
    ]


class RegionalConsistencySpec(BaseModel):
    """Precision-first repeated-marker rule for one benchmark target variety."""

    language: Annotated[str, Field(description="ISO 639-1 language code.")]
    target_variety: Annotated[
        str, Field(description="Variety required by the language pack.")
    ]
    non_target_variety: Annotated[
        str, Field(description="Variety family represented by the markers.")
    ]
    markers: Annotated[
        tuple[RegionalMarkerSpec, ...],
        Field(description="Closed, provenance-bearing high-precision marker catalog."),
    ]
    min_occurrences: Annotated[
        int,
        Field(
            default=2,
            ge=2,
            description="Minimum marker-turn hits required for a definite failure.",
        ),
    ] = 2
    min_agent_turns: Annotated[
        int,
        Field(
            default=2,
            ge=2,
            description="Minimum distinct delivered agent turns carrying markers.",
        ),
    ] = 2


# Sources:
# - pt/zh pack.yaml regional rubrics and translation guidance;
# - Taiwan NCC usage for 門號/簡訊/訊號 and Hong Kong OFCA usage for
#   流動電話/手提電話/短訊. Script-conversion-only forms are intentionally absent.
REGIONAL_CONSISTENCY_SPECS: dict[str, RegionalConsistencySpec] = {
    "pt": RegionalConsistencySpec(
        language="pt",
        target_variety="Brazilian Portuguese",
        non_target_variety="European Portuguese",
        markers=(
            RegionalMarkerSpec(
                id="ep_progressive",
                category="grammar",
                pattern=(
                    r"(?<!\w)(?:estou|estás|está|estamos|estais|estão)\s+a\s+"
                    r"(?:verificar|confirmar|consultar|procurar|tentar|processar|"
                    r"analisar|resolver|alterar|cancelar|reembolsar|enviar|receber|"
                    r"pagar|fazer|dizer|informar|aguardar|remarcar|reservar|"
                    r"devolver|trocar|acompanhar|localizar|ativar|desativar|"
                    r"reiniciar|ligar|configurar|abrir|falar|pesquisar|tratar|"
                    r"ajudar)(?!\w)"
                ),
                provenance="pt_pack_regional_rubric",
            ),
            RegionalMarkerSpec(
                id="ep_telemovel",
                category="lexicon",
                pattern=r"(?<!\w)(?:telemóvel|telemóveis)(?!\w)",
                provenance="pt_pack_translation_guidance",
            ),
            RegionalMarkerSpec(
                id="ep_ficheiro",
                category="lexicon",
                pattern=r"(?<!\w)ficheiros?(?!\w)",
                provenance="pt_pack_regional_rubric",
            ),
            RegionalMarkerSpec(
                id="ep_autocarro",
                category="lexicon",
                pattern=r"(?<!\w)autocarros?(?!\w)",
                provenance="pt_factory_nuance_audit",
            ),
            RegionalMarkerSpec(
                id="ep_comboio",
                category="lexicon",
                pattern=r"(?<!\w)comboios?(?!\w)",
                provenance="pt_factory_nuance_audit",
            ),
            RegionalMarkerSpec(
                id="ep_ecra",
                category="lexicon",
                pattern=r"(?<!\w)ecrãs?(?!\w)",
                provenance="pt_pack_translation_guidance",
            ),
            RegionalMarkerSpec(
                id="ep_equipa",
                category="lexicon",
                pattern=(
                    r"(?<!\w)(?:(?:a|uma|nossa|sua|esta|essa)\s+equipas?"
                    r"|equipas?\s+(?:de|do|da))(?!\w)"
                ),
                provenance="pt_pack_translation_guidance",
            ),
            RegionalMarkerSpec(
                id="ep_utilizador",
                category="lexicon",
                pattern=r"(?<!\w)utilizadores?(?!\w)",
                provenance="pt_domain_regional_inventory",
            ),
            RegionalMarkerSpec(
                id="ep_hiperligacao",
                category="lexicon",
                pattern=r"(?<!\w)hiperligaç(?:ão|ões)(?!\w)",
                provenance="pt_domain_regional_inventory",
            ),
            RegionalMarkerSpec(
                id="ep_morada",
                category="lexicon",
                pattern=r"(?<!\w)moradas?(?!\w)",
                provenance="pt_domain_regional_inventory",
            ),
            RegionalMarkerSpec(
                id="ep_nif",
                category="lexicon",
                pattern=r"(?<!\w)NIF(?!\w)",
                provenance="pt_pack_translation_guidance",
            ),
            RegionalMarkerSpec(
                id="ep_checked_baggage",
                category="lexicon",
                pattern=r"(?<!\w)bagagem\s+de\s+porão(?!\w)",
                provenance="pt_domain_regional_inventory",
            ),
            RegionalMarkerSpec(
                id="ep_receipt",
                category="lexicon",
                pattern=(
                    r"(?<!\w)tal(?:ão|ões)\s+(?:de\s+compra|da\s+compra|"
                    r"do\s+pedido|do\s+pagamento|fiscal)(?!\w)"
                ),
                provenance="pt_domain_regional_inventory",
            ),
        ),
    ),
    "zh": RegionalConsistencySpec(
        language="zh",
        target_variety="Mainland Mandarin",
        non_target_variety="Taiwan or Hong Kong Mandarin",
        markers=(
            RegionalMarkerSpec(
                id="tw_mobile_phone",
                category="lexicon",
                pattern=r"(?:行動電話|行动电话)",
                provenance="zh_pack_regional_rubric",
            ),
            RegionalMarkerSpec(
                id="hk_mobile_phone",
                category="lexicon",
                pattern=r"(?:流動電話|流动电话|手提電話|手提电话)",
                provenance="hk_ofca_official_telecom_usage",
            ),
            RegionalMarkerSpec(
                id="tw_software",
                category="lexicon",
                pattern=r"(?:軟體|软体)",
                provenance="zh_pack_regional_rubric",
            ),
            RegionalMarkerSpec(
                id="tw_application",
                category="lexicon",
                pattern=r"(?:應用程式|应用程式)",
                provenance="tw_government_software_usage",
            ),
            RegionalMarkerSpec(
                id="tw_server",
                category="lexicon",
                pattern=r"伺服器",
                provenance="tw_government_software_usage",
            ),
            RegionalMarkerSpec(
                id="tw_mobile_line",
                category="lexicon",
                pattern=r"(?:門號|门号)",
                provenance="tw_ncc_official_telecom_usage",
            ),
            RegionalMarkerSpec(
                id="tw_sms",
                category="lexicon",
                pattern=r"(?:簡訊|简讯)",
                provenance="tw_ncc_official_telecom_usage",
            ),
            RegionalMarkerSpec(
                id="tw_signal",
                category="lexicon",
                pattern=r"(?:訊號|讯号)",
                provenance="tw_ncc_official_telecom_usage",
            ),
            RegionalMarkerSpec(
                id="hk_sms",
                category="lexicon",
                pattern=r"(?:短訊|短讯)",
                provenance="hk_ofca_official_telecom_usage",
            ),
            RegionalMarkerSpec(
                id="hk_mobile_data",
                category="lexicon",
                pattern=r"(?:流動數據|流动数据)",
                provenance="hk_ofca_official_telecom_usage",
            ),
            RegionalMarkerSpec(
                id="tw_mobile_data",
                category="lexicon",
                pattern=r"(?:行動數據|行动数据|行動資料|行动资料)",
                provenance="tw_ncc_official_telecom_usage",
            ),
            RegionalMarkerSpec(
                id="tw_telecom_provider",
                category="lexicon",
                pattern=r"(?:電信業者|电信业者)",
                provenance="tw_ncc_official_telecom_usage",
            ),
            RegionalMarkerSpec(
                id="tw_base_station",
                category="lexicon",
                pattern=r"基地台",
                provenance="tw_ncc_official_telecom_usage",
            ),
            RegionalMarkerSpec(
                id="tw_airplane_mode",
                category="lexicon",
                pattern=r"(?:飛航模式|飞航模式)",
                provenance="tw_domain_official_usage",
            ),
            RegionalMarkerSpec(
                id="tw_reception",
                category="lexicon",
                pattern=r"(?:收訊|收讯|吃到飽|吃到饱)",
                provenance="tw_domain_official_usage",
            ),
            RegionalMarkerSpec(
                id="tw_monthly_plan",
                category="lexicon",
                pattern=r"(?:月費計劃|月费计划)",
                provenance="tw_domain_official_usage",
            ),
            RegionalMarkerSpec(
                id="hk_sim_card",
                category="lexicon",
                pattern=r"(?:SIM|電話|电话|儲值|储值)\s*咭",
                provenance="hk_ofca_official_telecom_usage",
            ),
            RegionalMarkerSpec(
                id="tw_retail_delivery",
                category="lexicon",
                pattern=r"(?:宅配|超商取貨|超商取货)",
                provenance="tw_post_official_retail_usage",
            ),
            RegionalMarkerSpec(
                id="tw_retail_terms",
                category="lexicon",
                pattern=r"(?:折價券|折价券|賣場|卖场|郵遞區號|邮递区号|列印)",
                provenance="tw_post_official_retail_usage",
            ),
            RegionalMarkerSpec(
                id="tw_airline_terms",
                category="lexicon",
                pattern=r"(?:訂位|订位|航廈|航厦|登機門|登机门)",
                provenance="tw_caa_official_airline_usage",
            ),
            RegionalMarkerSpec(
                id="hk_email",
                category="lexicon",
                pattern=r"(?:電郵|电邮)",
                provenance="hk_government_service_usage",
            ),
        ),
    ),
}


def _regional_consistency_precheck(
    ctx: CheckerContext, spec: RegionalConsistencySpec
) -> tuple[JudgeOutcome, Optional[str]]:
    """Fail only repeated, agent-authored, non-target markers across turns."""
    if ctx.language != spec.language:
        return JudgeOutcome.NO_OPPORTUNITY, None

    user_text = "\n".join(ctx.user_turns)
    mirrored_marker_ids = {
        marker.id
        for marker in spec.markers
        if re.search(marker.pattern, user_text, flags=re.IGNORECASE)
    }
    hits: list[tuple[int, RegionalMarkerSpec, str]] = []
    turns = _agent_turns_without_literals(ctx)
    for turn_index, turn in enumerate(turns):
        unquoted = _QUOTED_SPAN_RE.sub("", turn)
        for marker in spec.markers:
            if marker.id in mirrored_marker_ids:
                continue
            match = re.search(marker.pattern, unquoted, flags=re.IGNORECASE)
            if match is not None:
                hits.append((turn_index, marker, match.group(0)))

    hit_turns = {turn_index for turn_index, _, _ in hits}
    if len(hits) < spec.min_occurrences or len(hit_turns) < spec.min_agent_turns:
        return JudgeOutcome.NO_OPPORTUNITY, None

    examples = ", ".join(
        f"turn {turn_index + 1}: {marker.id}={matched!r}"
        for turn_index, marker, matched in hits[:4]
    )
    return (
        JudgeOutcome.FAIL,
        f"regional-{REGIONAL_CONSISTENCY_CHECKER_VERSION}: repeated "
        f"{spec.non_target_variety} markers across {len(hit_turns)} delivered "
        f"agent turns ({examples}); target={spec.target_variety}",
    )


def check_pt_regional_consistency(
    ctx: CheckerContext,
) -> tuple[JudgeOutcome, Optional[str]]:
    """Fail repeated high-confidence European Portuguese usage."""
    return _regional_consistency_precheck(ctx, REGIONAL_CONSISTENCY_SPECS["pt"])


def check_zh_regional_consistency(
    ctx: CheckerContext,
) -> tuple[JudgeOutcome, Optional[str]]:
    """Fail repeated high-confidence Taiwan/Hong Kong Mandarin vocabulary."""
    return _regional_consistency_precheck(ctx, REGIONAL_CONSISTENCY_SPECS["zh"])


KOREAN_NAME_ADDRESS_CHECKER_VERSION = "v1"
_HANGUL_CLASS = r"가-힣"


def _ko_name_checker_turns(ctx: CheckerContext) -> list[str]:
    """Strip non-name task literals and quoted reports from delivered agent turns."""
    if ctx.caller_name is None:
        return []
    protected = {value.casefold() for value in ctx.caller_name.protected_name_literals}
    removable = [
        value
        for value in ctx.allowed_literals
        if value and value.casefold() not in protected
    ]
    return [
        _QUOTED_SPAN_RE.sub("", _without_allowed_literals(turn, removable))
        for turn in (ctx.agent_turns or [ctx.agent_text])
    ]


def check_ko_name_address_conventions(
    ctx: CheckerContext,
) -> tuple[JudgeOutcome, Optional[str]]:
    """Fail only a known caller's reversed name or strict family-name ``씨`` vocative."""
    name = ctx.caller_name
    if ctx.language != "ko" or name is None:
        return JudgeOutcome.NO_OPPORTUNITY, None

    reversed_name = (
        rf"(?<![{_HANGUL_CLASS}]){re.escape(name.given_spoken)}\s*"
        rf"{re.escape(name.family_spoken)}(?![{_HANGUL_CLASS}])"
    )
    family_ssi = (
        rf"(?<![{_HANGUL_CLASS}]){re.escape(name.family_spoken)}\s*씨"
        rf"(?![{_HANGUL_CLASS}])"
    )
    # A customer may explicitly prefer or introduce an otherwise unusual form.
    # User speech is an exemption only and never contributes a failure hit.
    user_text = "\n".join(ctx.user_turns)
    suppress_reversal = (
        re.search(
            rf"{re.escape(name.given_spoken)}\s*{re.escape(name.family_spoken)}",
            user_text,
        )
        is not None
    )
    suppress_family_ssi = (
        re.search(rf"{re.escape(name.family_spoken)}\s*씨", user_text) is not None
    )

    boundary = r"(?:^|[.!?。！？\n]\s*)"
    reversed_vocative = re.compile(
        rf"{boundary}(?P<name>{reversed_name})\s*(?:고객)?님\s*[,，:：!?！？]"
    )
    reversed_confirmation = re.compile(
        rf"고객님(?:의)?\s*성함(?:은|이)?\s*(?P<name>{reversed_name})"
        rf"(?=\s*(?:맞으|이시))"
    )
    family_vocative = re.compile(rf"{boundary}(?P<name>{family_ssi})\s*[,，:：!?！？]")

    for turn_index, turn in enumerate(_ko_name_checker_turns(ctx), 1):
        if not suppress_reversal:
            match = reversed_vocative.search(turn) or reversed_confirmation.search(turn)
            if match is not None:
                return (
                    JudgeOutcome.FAIL,
                    f"name-address-{KOREAN_NAME_ADDRESS_CHECKER_VERSION}: known "
                    f"caller {name.full_spoken} addressed with reversed name "
                    f"{match.group('name')!r} in delivered agent turn {turn_index}",
                )
        if not suppress_family_ssi and (match := family_vocative.search(turn)):
            return (
                JudgeOutcome.FAIL,
                f"name-address-{KOREAN_NAME_ADDRESS_CHECKER_VERSION}: known "
                f"caller {name.full_spoken} directly addressed as family-name-only "
                f"{match.group('name')!r} in delivered agent turn {turn_index}",
            )
    return JudgeOutcome.NO_OPPORTUNITY, None


COUNTING_UNIT_CHECKER_VERSION = "v1"


class CountingUnitMarkerSpec(BaseModel):
    """One reviewed number-form or counting-word contradiction."""

    id: Annotated[str, Field(description="Stable marker identifier for evidence.")]
    pattern: Annotated[
        str,
        Field(description="Unicode-aware regular expression for agent-turn text."),
    ]
    provenance: Annotated[
        str,
        Field(description="Reviewed pack or human-calibration source."),
    ]


class CountingUnitSpec(BaseModel):
    """Precision-first counting-unit precheck for one benchmark language."""

    language: Annotated[str, Field(description="ISO 639-1 language code.")]
    markers: Annotated[
        tuple[CountingUnitMarkerSpec, ...],
        Field(description="Closed, provenance-bearing contradiction inventory."),
    ]


COUNTING_UNIT_SPECS: dict[str, CountingUnitSpec] = {
    "ko": CountingUnitSpec(
        language="ko",
        markers=(
            CountingUnitMarkerSpec(
                id="sino_number_with_baggage_gae",
                pattern=(
                    r"(?<!\w)(?:수하물|짐|가방)(?:은|는|이|가|을|를)?\s*"
                    r"(?:일|이|삼|사)\s*개(?:가|는|를|만|씩)?(?=$|[^\w])"
                ),
                provenance="ko_pack_counting_units_rubric_and_round1_review",
            ),
            CountingUnitMarkerSpec(
                id="sino_number_with_hours",
                pattern=(
                    r"(?<![\w제])삼\s*시간"
                    r"(?:이|은|을|만|씩)?(?=$|[^\w])"
                ),
                provenance="ko_pack_counting_units_rubric_and_nuance_map",
            ),
            CountingUnitMarkerSpec(
                id="people_counted_with_gae",
                pattern=(
                    r"(?<!\w)(?:사람|승객)\s*"
                    r"(?:한|두|세|네)\s*개(?:가|는|를|만|씩)?(?=$|[^\w])"
                ),
                provenance="ko_pack_counting_units_rubric_and_nuance_map",
            ),
            CountingUnitMarkerSpec(
                id="gae_followed_by_person_noun",
                pattern=(
                    r"(?<!\w)(?:한|두|세|네)\s*개(?:의)?\s*"
                    r"(?:사람|승객)(?:이|은|을|만)?(?=$|[^\w])"
                ),
                provenance="ko_pack_counting_units_rubric_and_nuance_map",
            ),
        ),
    ),
    "zh": CountingUnitSpec(
        language="zh",
        markers=(
            CountingUnitMarkerSpec(
                id="human_measure_word_for_letters",
                pattern=(
                    r"(?:一|两|二|三|四|五|六|七|八|九|几)\s*位\s*"
                    r"字母(?![\u3400-\u4dbf\u4e00-\u9fff])"
                ),
                provenance="zh_pack_counting_units_rubric",
            ),
            CountingUnitMarkerSpec(
                id="er_ge_order",
                pattern=r"(?<!第)(?<!第\s)二\s*个\s*(?:订单|訂單|预订|預訂)",
                provenance="zh_pack_counting_units_rubric",
            ),
            CountingUnitMarkerSpec(
                id="er_zhang_ticket",
                pattern=r"(?<!第)(?<!第\s)二\s*张\s*(?:票|机票|機票)",
                provenance="zh_pack_counting_units_rubric",
            ),
        ),
    ),
}


def _counting_unit_precheck(
    ctx: CheckerContext, spec: CountingUnitSpec
) -> tuple[JudgeOutcome, Optional[str]]:
    """Fail only reviewed contradictions in delivered agent speech.

    Task-provided values and quoted spans are excluded. Arabic digits are not
    markers because a transcript such as ``2个`` does not reveal whether the
    agent pronounced 二 or 两. Absence and every unlisted case fall through to
    the LLM; this checker never returns PASS.
    """
    if ctx.language != spec.language:
        return JudgeOutcome.NO_OPPORTUNITY, None

    for turn_index, turn in enumerate(_agent_turns_without_literals(ctx)):
        unquoted = _QUOTED_SPAN_RE.sub("", turn)
        for marker in spec.markers:
            match = re.search(marker.pattern, unquoted)
            if match is not None:
                return (
                    JudgeOutcome.FAIL,
                    f"counting-units-{COUNTING_UNIT_CHECKER_VERSION}: agent turn "
                    f"{turn_index + 1} used {marker.id}={match.group(0)!r}; "
                    f"source={marker.provenance}",
                )
    return JudgeOutcome.NO_OPPORTUNITY, None


def check_ko_counting_units(
    ctx: CheckerContext,
) -> tuple[JudgeOutcome, Optional[str]]:
    """Fail definite Korean number-system or counter contradictions."""
    return _counting_unit_precheck(ctx, COUNTING_UNIT_SPECS["ko"])


def check_zh_counting_units(
    ctx: CheckerContext,
) -> tuple[JudgeOutcome, Optional[str]]:
    """Fail definite Mandarin numeral or measure-word contradictions."""
    return _counting_unit_precheck(ctx, COUNTING_UNIT_SPECS["zh"])


# High-precision Traditional-only forms that have a distinct Simplified form.
# Shared Han characters are intentionally absent: they provide no script signal.
_TRADITIONAL_TO_SIMPLIFIED = {
    traditional: simplified
    for traditional, simplified in zip(
        "與為這個們來說請問處訂單號電話機軟體證讓幫確認後裡還會應該開關時間轉換資聯繫錢費飛預約續務優價選擇謝對錯無訊實際經驗國業發現歡頁網絡寫讀點萬億兩長張種樣據從將過進達運門風區縣市東廣辦壞買賣稱學習總終結線數據庫",
        "与为这个们来说请问处订单号电话机软体证让帮确认后里还会应该开关时间转换资联系钱费飞预约续务优价选择谢对错无讯实际经验国业发现欢页网络写读点万亿两长张种样据从将过进达运门风区县市东广办坏买卖称学习总终结线数据库",
        strict=True,
    )
    if traditional != simplified
}
_HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def check_zh_script_consistency(
    ctx: CheckerContext,
) -> tuple[JudgeOutcome, Optional[str]]:
    """Require Simplified Chinese in Chinese text runs only.

    Latin borrowings, emails, codes, and product names contain no Han-script
    signal and are naturally ignored. Exact task literals are stripped first so
    a supplied Traditional-script name or identifier is not blamed on the agent.
    """
    if ctx.language != "zh" or ctx.is_voice:
        return JudgeOutcome.NO_OPPORTUNITY, None

    text = ctx.agent_text
    for literal in sorted(
        (value for value in ctx.allowed_literals if value),
        key=len,
        reverse=True,
    ):
        text = text.replace(literal, "")
    if not _HAN_RE.search(text):
        return JudgeOutcome.NO_OPPORTUNITY, None

    found = [
        (char, _TRADITIONAL_TO_SIMPLIFIED[char])
        for char in text
        if char in _TRADITIONAL_TO_SIMPLIFIED
    ]
    if not found:
        return JudgeOutcome.PASS, "Simplified Chinese script used consistently"

    unique = list(dict.fromkeys(found))
    examples = ", ".join(
        f"{traditional}→{simplified}" for traditional, simplified in unique[:8]
    )
    return (
        JudgeOutcome.FAIL,
        f"Traditional forms in a Simplified-Chinese text run: {examples}",
    )


CHECKER_REGISTRY: dict[str, CheckerFn] = {
    "backchannel_frequency": check_backchannel_frequency,
    "email_symbol_verbalization": check_email_symbol_verbalization,
    "es_register_formality": check_es_register_formality,
    "es_gender_agreement": check_es_gender_agreement,
    "hi_gender_agreement": check_hi_gender_agreement,
    "hi_honorific_agreement": check_hi_honorific_agreement,
    "hi_register_formality": check_hi_register_formality,
    "ko_counting_units": check_ko_counting_units,
    "ko_honorific_levels": check_ko_honorific_levels,
    "ko_honorific_agreement": check_ko_honorific_agreement,
    "ko_name_address_conventions": check_ko_name_address_conventions,
    "pt_gender_agreement": check_pt_gender_agreement,
    "pt_regional_consistency": check_pt_regional_consistency,
    "zh_counting_units": check_zh_counting_units,
    "zh_regional_consistency": check_zh_regional_consistency,
    "zh_register_formality": check_zh_register_formality,
    "zh_script_consistency": check_zh_script_consistency,
}
