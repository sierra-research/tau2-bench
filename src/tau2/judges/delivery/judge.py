# Copyright Sierra
"""Combined multimodal AUDIO judge for delivery: fidelity + intonation + factors.

One Gemini call per agent utterance, given the synthesized audio (inline WAV),
the expected synthesis text, and the language's delivery rubric. It evaluates
everything in a single pass (never extra calls per factor):

- Fidelity   — did the audio faithfully realize the script (right words, numbers,
               codes; not garbled)? Prompt ported from sierra's TTS mispronunciation
               judge (``server/api/tts_mispronunciation_judge.go``).
- Intonation — does the delivery sound natural (prosody, pauses, cadence, pitch,
               stress, tone)? Ported from sierra's TTS intonation judge
               (``server/api/tts_intonation_judge.go``).
- Language-specific factors — the pack's delivery rubric (closed catalog in
               ``tau2.multilingual.delivery_catalog``), scored per-factor like
               nativeness factors. A language with no pack rubric still gets a
               language-SPECIFIC instruction: the fixed native-listener fallback
               template, parameterized only on the language name/code (see
               ``tau2.judges.delivery.factors.build_delivery_rubric``).

Findings are tagged with ``axis`` so the harness can split them into per-axis
sub-scores. The reply is validated into ``DeliveryJudgeResponse``; known model
quirks (missing axis, out-of-range severity, stringly booleans) are coerced by
validators, and a finding that STILL fails validation fails the whole response —
the harness records the utterance as a loud ERROR instead of silently dropping
the finding.
"""

from typing import Annotated, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tau2.config import (
    DEFAULT_LLM_DELIVERY_JUDGE,
    DEFAULT_LLM_DELIVERY_JUDGE_ARGS,
)
from tau2.data_model.message import SystemMessage, UserMessage
from tau2.data_model.simulation import (
    DeliveryAxis,
    DeliveryFactorCheck,
    DeliveryFinding,
    DeliveryUtteranceResult,
    JudgeOutcome,
)
from tau2.judges.base import VerdictReplyBase, judge_structured, verdict_outcome
from tau2.judges.delivery.factors import DeliveryFactorConfig, DeliveryRubric

_FIDELITY_CATEGORIES = {
    "mispronunciation",
    "word_substitution",
    "missing_word",
    "extra_or_hallucinated_word",
    "number_date_currency",
    "email_url_code",
    "punctuation_or_formatting",
    "acronym_brand_name",
    "clipped_or_garbled",
    "other",
}
_INTONATION_CATEGORIES = {
    "unnatural_pause",
    "missing_pause",
    "phrase_internal_gap",
    "robotic_or_mechanical_delivery",
    "unnatural_pitch_contour",
    "misplaced_stress",
    "inappropriate_upinflection",
    "monotone_flat_delivery",
    "emotional_mismatch",
    "cadence_or_speed_shift",
    "uncanny_prosody",
    "other",
}


class DeliveryJudgeFinding(BaseModel):
    """One issue the judge flagged, as parsed from the model reply.

    Validators absorb the judge's known quirks: a missing/mislabeled ``axis``
    is inferred from the category sets, and ``severity`` is clamped into
    [1, 3]. Anything the coercions cannot repair raises — failing the whole
    response — so a malformed finding is a loud ERROR, never a silent drop.
    """

    model_config = ConfigDict(extra="ignore")

    axis: Annotated[
        DeliveryAxis,
        Field(
            description="'fidelity' (right words) or 'intonation' (natural "
            "delivery); inferred from the category when omitted/mislabeled."
        ),
    ]
    category: Annotated[
        str, Field(description="Judge finding category, e.g. 'mispronunciation'.")
    ] = "other"
    time_range: Annotated[
        Optional[str],
        Field(description="Approx. time span of the issue in the clip, if given."),
    ] = None
    issue: Annotated[
        Optional[str],
        Field(description="Description of the problem (PII-redacted by the judge)."),
    ] = None
    severity: Annotated[
        int, Field(description="1 (minor) .. 3 (critical).", ge=1, le=3)
    ] = 1
    confidence: Optional[float] = None

    @model_validator(mode="before")
    @classmethod
    def _infer_axis(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        if data.get("axis") not in ("fidelity", "intonation"):
            category = str(data.get("category", "")).strip()
            if (
                category in _INTONATION_CATEGORIES
                and category not in _FIDELITY_CATEGORIES
            ):
                data["axis"] = "intonation"
            else:
                data["axis"] = "fidelity"
        return data

    @field_validator("category", mode="before")
    @classmethod
    def _default_category(cls, value: object) -> object:
        return "other" if value in (None, "") else value

    @field_validator("severity", mode="before")
    @classmethod
    def _clamp_severity(cls, value: object) -> object:
        try:
            severity = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            severity = 1
        return max(1, min(3, severity))

    @field_validator("time_range", "issue", mode="before")
    @classmethod
    def _empty_to_none(cls, value: object) -> object:
        return None if value in (None, "") else value


class DeliveryFactorCheckReply(VerdictReplyBase):
    """One language-specific factor verdict, as parsed from the model reply.

    The nativeness-shaped opportunity/violated/reasoning/quote fields (and the
    quirk-absorbing validators) live on the shared ``VerdictReplyBase``; this
    adds only the ``factor_id`` the verdict is for.
    """

    factor_id: Annotated[
        str, Field(description="The rubric factor this verdict is for.")
    ]


class DeliveryJudgeResponse(BaseModel):
    """The judge's structured verdict for one utterance's audio clip.

    Only ``findings`` drive the fidelity/intonation verdict — the reply's own
    ``has_issue`` / ``severity`` / ``flag_for_review`` fields are ignored (an
    issue the judge cannot articulate as an axis-tagged finding is not
    actionable, and would desync the overall severity from the per-axis
    sub-scores). ``factor_checks`` carry the language-specific rubric verdicts
    (pack mode only).
    """

    model_config = ConfigDict(extra="ignore")

    findings: Annotated[
        list[DeliveryJudgeFinding],
        Field(
            default_factory=list,
            description="Axis-tagged issues found in the clip; empty = clean.",
        ),
    ]
    factor_checks: Annotated[
        list[DeliveryFactorCheckReply],
        Field(
            default_factory=list,
            description="Per-factor verdicts against the language rubric; "
            "empty in fallback/generic mode.",
        ),
    ]
    summary: Annotated[
        Optional[str], Field(description="One-line summary of the clip verdict.")
    ] = None
    confidence: Optional[float] = None


_SYSTEM_PROMPT = """\
You are a careful evaluator of synthesized customer-support voice agents. You judge \
the AGENT's spoken audio on TWO independent axes and report issues for both in a \
single response. Every finding must be tagged with its axis: "fidelity" or "intonation".

=== AXIS A: FIDELITY (did it say the right words?) ===
Compare the audio against the reference transcript when provided, and identify clear \
audible ways the synthesized speech fails to faithfully realize the script:
- Mispronounced words, names, brands, acronyms, or phrases.
- Word or phrase substitutions that change the intended wording or meaning.
- Missing words, dropped syllables, extra words, or hallucinated words.
- Numbers, dates, times, percentages, currency, emails, URLs, codes, account numbers, \
or digit strings spoken incorrectly or confusingly.
- Audible punctuation/formatting mistakes that change what the listener hears (saying \
"comma" aloud, dropping a needed separator in an email/URL/code, adding a strong break \
that changes how structured text parses, or removing a needed pause between tokens).
- Garbled, clipped, swallowed, or unintelligible speech that changes a word, drops a \
phoneme/syllable, or prevents the intended phrase from being understood.
Fidelity categories: mispronunciation, word_substitution, missing_word, \
extra_or_hallucinated_word, number_date_currency, email_url_code, \
punctuation_or_formatting, acronym_brand_name, clipped_or_garbled, other.

Be tolerant of harmless spoken-language normalization. For fidelity, do NOT flag:
- Tiny breath pauses, normal phrase grouping, or ordinary emphasis differences.
- Equivalent contractions/expansions ("I'm" vs "I am"); "OK" vs "okay".
- "zero" vs "oh" unless it makes a number/code confusing.
- Accent variation that does not change intelligibility.
- Slight rephrasing that preserves meaning and is acceptable in support speech.
- Lightly unreleased/softened final consonants when the word is still clear.
- Pure acoustic-quality artifacts (vocal fry, stretched vowels, robotic tone, odd \
warmth, uncanny texture, telephony band-limiting) when the script token is correct.
- Localized interruption-recovery artifacts (a brief gap, cutoff, clipped partial word, \
repeated syllable, restart, momentary garble at a stop/resume boundary) if the agent \
resumes the expected script and meaning is recoverable.

=== AXIS B: INTONATION (did it sound natural?) ===
Intonation means audible delivery patterns: prosody, pauses, cadence, pitch movement, \
lexical stress, and delivery tone. It is NOT about whether the script was faithfully \
spoken (that is fidelity). Identify clear audible ways the delivery sounds objectively \
unnatural, mechanical, inhuman, tonally inappropriate, or hard to parse:
- Unnatural pauses, missing pauses, overlong punctuation pauses, phrase-internal gaps, \
or stitched/segmented timing.
- Odd pitch movement, misplaced stress, inappropriate up-inflection, abrupt pitch \
resets, or emphasis that makes ordinary words sound unnatural.
- Robotic, mechanical, monotone, sing-song, overly uniform, or uncanny cadence.
- Support-tone mismatch caused by delivery: sounding scary, stern, sarcastic, \
performative, overly authoritative, or oddly upbeat in a sensitive moment.
- Cadence or speed changes that are objectively inhuman, jarring, rushed, dragging, or \
hard to parse.
Intonation categories: unnatural_pause, missing_pause, phrase_internal_gap, \
robotic_or_mechanical_delivery, unnatural_pitch_contour, misplaced_stress, \
inappropriate_upinflection, monotone_flat_delivery, emotional_mismatch, \
cadence_or_speed_shift, uncanny_prosody, other.

Be tolerant of normal human variation. For intonation, do NOT flag:
- Merely bland, less warm, less expressive, faster, or slower delivery if it still \
sounds like plausible human speech.
- Normal punctuation pauses, breath pauses, phrase grouping, regional accent, or style.
- Subjective preference or voice-identity mismatch.
- Recording/telephony quality unless it directly causes a distinct prosody/pause/pitch/ \
stress/delivery-tone issue.
- Localized interruption-recovery artifacts as described above.

=== LANGUAGE-SPECIFIC DELIVERY RUBRIC ===
The user message may include a language-specific delivery rubric, in one of two forms:
- A list of FACTOR CHECKS, each with a factor_id and a "listen for" description. For \
EACH listed factor decide: did this clip give the factor an OPPORTUNITY to surface? If \
it did, was it VIOLATED (the delivered audio audibly exhibits the described problem) \
or handled natively? Report EVERY listed factor in "factor_checks" using its exact \
factor_id, judging each factor only by its own "listen for" description. When \
uncertain whether a native listener would object, do NOT flag it (prefer \
false-negative over false-positive). A factor violation that is also a fidelity or \
intonation issue should ALSO be reported as a normal axis-tagged finding.
- A NATIVE-LISTENER instruction naming a language, with no factor list. Then evaluate \
the clip as a native listener of that language — attending to its phonology, its \
tone/pitch-accent/lexical-stress system, its native letter-name and digit/number/date \
readout conventions, and its typical text-to-speech failure modes — and report what \
you hear through the ordinary fidelity/intonation findings. Return an empty \
"factor_checks" list.
When the user message contains no rubric, return an empty "factor_checks" list.

=== GENERAL ===
Prefer precision over recall on both axes and on every factor check. If you are not \
sure, leave it out of findings and use lower confidence. If a transcript is provided, \
use it as the reference.

Do not include PII, customer secrets, or sensitive literal values in the response. When \
describing a problem involving a name, email, URL, phone number, address, \
account/card/identity number, date of birth, code, or token, describe the value by its \
class and the structural error rather than copying the literal value (e.g. "the final \
digit of a 4-digit account-number suffix was omitted", or "a customer name has an \
unnatural stress pattern").

Severity (applies to each finding and the overall clip):
- 0: no clear issue.
- 1: minor; listener can still recover the intended content / it still sounds mostly human.
- 2: clear issue that may confuse a listener, or makes the voice sound mechanical/wrong.
- 3: critical — sensitive data/money/date/identity fidelity error or meaning-changing \
hallucination; or delivery that would damage trust (scary/mean/sarcastic, extreme \
roboticness, very disruptive pauses/cadence).
Set flag_for_review to true whenever any finding has severity 2 or 3.

Return only valid JSON. Do not wrap it in Markdown."""


_OUTPUT_SHAPE = """\
Return this exact JSON shape:
{
  "has_issue": boolean,
  "flag_for_review": boolean,
  "severity": 0 | 1 | 2 | 3,
  "confidence": number,
  "summary": string,
  "findings": [
    {
      "axis": "fidelity" | "intonation",
      "category": string,
      "time_range": string,
      "issue": string,
      "severity": 1 | 2 | 3,
      "confidence": number
    }
  ],
  "factor_checks": [
    {
      "factor_id": string,
      "opportunity": boolean,
      "violated": boolean,
      "reasoning": string,
      "quote": string
    }
  ]
}"""


# The two fixed per-language rubric templates. Rendering is pure interpolation
# of typed inputs (language name/code and the pack's rubric text); the judge
# NEVER receives agent-improvised instructions. The pack template lists the
# language's catalog factors; the fallback template interpolates ONLY the
# language name/code and instructs the judge to construct language-appropriate
# criteria itself as a native listener.
_RUBRIC_PACK_TEMPLATE = """\
Language-specific delivery rubric for {display} — source: language pack.
Judge each factor below against the delivered audio and report EVERY one in \
"factor_checks" by its exact factor_id:
{factor_lines}"""

_RUBRIC_PACK_FACTOR_LINE = """\
- factor_id: {id} (severity {severity})
  Listen for: {listen_for}"""

_RUBRIC_FALLBACK_TEMPLATE = """\
Language-specific delivery rubric for {display} — source: fallback (no curated \
delivery factors for this language yet).
Evaluate this clip AS A NATIVE LISTENER of {name}. Attend to {name}'s phonology \
and phoneme inventory; its lexical tone, pitch-accent, or stress system; its \
native letter-name and digit/number/date readout conventions; and the typical \
text-to-speech failure modes for {name}. Construct the language-appropriate \
criteria yourself and report any problems as ordinary fidelity/intonation findings. \
Return an empty "factor_checks" list."""

# The fixed user-prompt template (pure interpolation of typed inputs). Part of
# the calibrated prompt surface, so it is covered by the sha below.
_USER_PROMPT_TEMPLATE = """\
Evaluate the attached AGENT AUDIO clip on BOTH axes (fidelity and intonation).

Reference transcript (expected_synthesis_text):
{ref}

Conversation context:
- was_interrupted: {was_interrupted}
- If was_interrupted is true, evaluate only the delivered audio; do not penalize \
trailing content that was cut off when the caller barged in or the call recording \
ended, nor a partial word or clipped sound at the very end of the clip (a cut, \
not a synthesis defect).

Target locale/language:
{locale}
{rubric_section}
{output_shape}"""

# The judge prompt is calibrated material: bump the version on ANY retune so
# a prompt change between reruns is visible instead of silently invalidating
# calibration.
# v2: language-specific rubric (pack factor checks + native-listener fallback).
# v3: user-prompt template retune.
# v4: zh/ko code, calendar, date, and money readout rubric retune.
# v5: interrupted-utterance tolerance widened to trailing partial/clipped
#     sounds (barge-in cut artifacts, paired with tick-level barge-in
#     detection in the harness).
# v6: the tolerance also covers a recording that ends mid-speech (the call
#     hit its ceiling or hung up while the agent was still talking, so the
#     audio clamps at the tail with no barge-in anywhere) — the harness now
#     sets was_interrupted for the utterance whose tick span reaches the
#     final tick. Measured on recall_20 gold: this truncation class was the
#     largest fidelity false-positive source (6 of 21 FP cells).
# v7: prompt text unchanged; delivery reference construction now also carries
# the shared legacy ``raw_data.was_truncated`` signal into ``was_interrupted``.
DELIVERY_JUDGE_PROMPT_VERSION = "v7"


def _display(rubric: DeliveryRubric) -> str:
    """Human-readable language label, e.g. 'Polish (pl)' (code-only if unnamed)."""
    suffix = f", locale {rubric.locale}" if rubric.locale else ""
    if rubric.display_name:
        return f"{rubric.display_name} ({rubric.language}{suffix})"
    return f"{rubric.language}{suffix}"


def render_language_rubric(rubric: DeliveryRubric) -> Optional[str]:
    """Render the per-language rubric block of the user prompt.

    Pure interpolation of the fixed templates: pack mode renders the factor
    list, fallback mode interpolates only the language name/code. Returns None
    when the run has no language (the judge stays language-generic).
    """
    if rubric.source == "pack":
        factor_lines = "\n".join(
            _RUBRIC_PACK_FACTOR_LINE.format(
                id=f.id, severity=f.severity, listen_for=f.listen_for
            )
            for f in rubric.enabled_factors
        )
        return _RUBRIC_PACK_TEMPLATE.format(
            display=_display(rubric), factor_lines=factor_lines
        )
    if rubric.source == "fallback":
        return _RUBRIC_FALLBACK_TEMPLATE.format(
            display=_display(rubric),
            name=rubric.display_name or rubric.language,
        )
    return None


def build_user_prompt(
    expected_text: str,
    rubric: DeliveryRubric,
    was_interrupted: bool,
) -> str:
    """Render the judge's user prompt for one clip (pure interpolation).

    Public: annotation packet rendering re-uses this exact seam so reviewers
    see byte-identical prompts to what the judge received.
    """
    rubric_block = render_language_rubric(rubric)
    return _USER_PROMPT_TEMPLATE.format(
        ref=expected_text.strip() or "(not provided)",
        was_interrupted=str(was_interrupted).lower(),
        locale=rubric.locale or rubric.language or "(not provided)",
        rubric_section=f"\n{rubric_block}\n" if rubric_block else "",
        output_shape=_OUTPUT_SHAPE,
    )


def _factor_checks_from_reply(
    reply: DeliveryJudgeResponse,
    factors: list[DeliveryFactorConfig],
) -> list[DeliveryFactorCheck]:
    """Map the reply's factor verdicts onto the REQUESTED factors, one check each.

    The request is the contract: every enabled rubric factor gets exactly one
    check. A factor the judge did not answer is recorded as a loud ERROR (never
    silently missing — that would read as "clean" downstream); reply rows for
    factor_ids that were never requested are dropped (a hallucinated id must not
    introduce an un-reviewed axis).
    """
    by_id = {c.factor_id: c for c in reply.factor_checks}
    checks: list[DeliveryFactorCheck] = []
    for factor in factors:
        verdict = by_id.get(factor.id)
        if verdict is None:
            outcome: JudgeOutcome = JudgeOutcome.ERROR
            evidence: Optional[str] = "judge returned no verdict for this factor"
            quote: Optional[str] = None
        else:
            outcome, evidence, quote = verdict_outcome(verdict)
        checks.append(
            DeliveryFactorCheck(
                id=factor.id,
                category=factor.category,
                severity=factor.severity,
                outcome=outcome,
                evidence=evidence,
                quote=quote,
                shadow=factor.shadow,
            )
        )
    return checks


def run_delivery_judge(
    audio_wav_b64: str,
    expected_text: str,
    *,
    utterance_idx: int,
    rubric: Optional[DeliveryRubric] = None,
    was_interrupted: bool = False,
    model: str = DEFAULT_LLM_DELIVERY_JUDGE,
    model_args: Optional[dict] = None,
) -> DeliveryUtteranceResult:
    """Judge one agent utterance's audio: fidelity + intonation + rubric factors.

    ONE call per utterance regardless of how many rubric factors the language
    defines. ``rubric`` (built once per sim via ``build_delivery_rubric``)
    selects the prompt mode: pack factors, native-listener fallback, or
    language-generic when None / language-less. The audio rides a
    ``UserMessage.audio_content`` multipart through the single ``generate()``
    seam (the WAV container is sniffed from the base64 magic).

    Deliberately propagates exceptions (API failures and invalid replies
    alike) so the harness records the utterance as ERROR without masking it
    as "clean".
    """
    rubric = rubric if rubric is not None else DeliveryRubric()
    messages = [
        SystemMessage(role="system", content=_SYSTEM_PROMPT),
        UserMessage(
            role="user",
            content=build_user_prompt(expected_text, rubric, was_interrupted),
            audio_content=audio_wav_b64,
        ),
    ]
    reply = judge_structured(
        model=model,
        messages=messages,
        response_model=DeliveryJudgeResponse,
        call_name="delivery_judge",
        model_args=(
            model_args if model_args is not None else DEFAULT_LLM_DELIVERY_JUDGE_ARGS
        ),
    )

    # Findings are the single source of truth for the fidelity/intonation
    # verdict: an "issue" the judge cannot articulate as an axis-tagged finding
    # is not actionable, and would make the overall severity disagree with both
    # axis sub-scores (which are computed from findings). outcome/severity/flag
    # are all derived. Factor checks are a decoupled sibling channel (scored
    # into DeliveryInfo.factor_score, mirroring nativeness) and never alter the
    # axis verdict.
    findings = [DeliveryFinding(**f.model_dump()) for f in reply.findings]
    severity = max((f.severity for f in findings), default=0)
    flag_for_review = any(f.severity >= 2 for f in findings)
    factor_checks = (
        _factor_checks_from_reply(reply, rubric.enabled_factors)
        if rubric.source == "pack"
        else []
    )
    return DeliveryUtteranceResult(
        utterance_idx=utterance_idx,
        was_interrupted=was_interrupted,
        expected_text=(expected_text[:200] or None),
        outcome=JudgeOutcome.FAIL if findings else JudgeOutcome.PASS,
        flag_for_review=flag_for_review,
        severity=severity,
        confidence=reply.confidence,
        summary=reply.summary or None,
        findings=findings,
        factor_checks=factor_checks,
    )
