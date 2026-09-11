# Copyright Sierra
"""Combined interaction, nativeness, and audio-quality annotation packet.

One packet owns the complete review surface. Annotators can switch the evidence
pane between the full caller+agent conversation and the agent-only view, while
all three expandable rubric sections remain available beside it. The audio
section reuses the canonical audio-quality dimensions rather than copying a
second rubric catalog.
"""

import hashlib
import json
import shutil
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal, Optional

from jinja2 import Environment, FileSystemLoader
from loguru import logger
from markupsafe import Markup
from pydantic import BaseModel, Field, model_validator

from tau2.annotation.artifacts import (
    ArtifactManifest,
    AudioQualityEntry,
    NativenessAgentTurn,
    RubricPacketEntry,
    derive_batch_id,
    git_sha,
    write_json_artifact,
)
from tau2.annotation.loading import LoadedSim, iter_loaded_sims
from tau2.annotation.models import (
    OVERALL_INTERACTION_QUALITY_FACTOR_ID,
    OVERALL_NATIVENESS_FACTOR_ID,
    NativenessHumanLabelRow,
    RubricAnnotationRow,
)
from tau2.annotation.packets.audio_quality import (
    AUDIO_QUALITY_DIMENSIONS,
    AUDIO_QUALITY_RUBRIC_VERSION,
    agent_speech_transcript,
)
from tau2.annotation.packets.builder import (
    DEFAULT_OUTPUT_ROOT,
    PACKET_MANIFEST_NAME,
    sync_packet_zip,
)
from tau2.annotation.packets.forms import AUDIO_QUALITY_KIND, RUBRIC_PACKET_KIND
from tau2.annotation.packets.transcript import generate_tick_rows
from tau2.judges.delivery.disk_audio import find_both_wav
from tau2.judges.nativeness.factors import judge_factors_for
from tau2.judges.nativeness.harness import build_agent_turns
from tau2.judges.quality.factors import QUALITY_FACTORS, QUALITY_RUBRIC_VERSION
from tau2.multilingual.factory.entity_localization import caller_gender_for_task
from tau2.voice.voice_gender import resolve_agent_gender

RUBRIC_PACKET_VERSION = "combined-rubric-packet-v16"

#: The template + static-asset sources that BECOME the rendered rubric
#: instrument. Fingerprinted into packet provenance so a rebuild with an
#: edited instrument (template, CSS, or JS) can never silently keep a stale
#: index.html behind an unchanged draw.
RUBRIC_INSTRUMENT_FILES = (
    "rubric.html.j2",
    "base.html.j2",
    "static/rubric.css",
    "static/rubric.js",
)


def rubric_instrument_sha256() -> str:
    """Fingerprint of the rubric instrument's template and asset sources."""
    root = Path(__file__).parent / "templates"
    digest = hashlib.sha256()
    for name in RUBRIC_INSTRUMENT_FILES:
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update((root / name).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


# Fixed annotator-facing question set for the semantic (progression /
# conciseness) judges; bump on any retune of the question text or answer
# contract below (v2: utterance-level turn selection + minor/major severity).
SEMANTIC_QUESTIONS_VERSION = "calibration-semantic-questions-v2"
RubricSection = Literal["interaction", "semantic", "nativeness", "audio"]
AnswerKind = Literal["binary", "likert4", "nativeness", "audio_severity"]


class RubricScoreAnchor(BaseModel):
    """One labeled answer on an overall 1-4 packet scale."""

    value: Annotated[Literal["1", "2", "3", "4"], Field(description="Score.")]
    label: Annotated[str, Field(description="Short score label.")]
    description: Annotated[str, Field(description="Behavioral anchor.")]


class RubricQuestion(BaseModel):
    """One fixed question rendered in the combined packet."""

    id: Annotated[str, Field(description="Stable factor identifier.")]
    section: Annotated[RubricSection, Field(description="Packet section.")]
    answer_kind: Annotated[AnswerKind, Field(description="Answer scale to render.")]
    question: Annotated[str, Field(description="Annotator-facing question.")]
    guidance: Annotated[str, Field(description="Short boundary guidance.")]
    evaluation_level: Annotated[
        Literal["call", "utterance"],
        Field(
            description="Unit labeled for this question. Every "
            "utterance-level question (any answer kind) gets the agent-turn "
            "selection flow: a violation answer must pin the offending "
            "agent turns."
        ),
    ] = "call"
    severity_scale: Annotated[
        Literal["none", "minor_major"],
        Field(
            description="'minor_major' on the semantic-source questions: a "
            "YES answer must carry the two-level severity the semantic "
            "judges grade on (major = rating 1, minor = rating 2)."
        ),
    ] = "none"
    score_anchors: Annotated[
        list[RubricScoreAnchor],
        Field(description="Ordered anchors for a 1-4 score question."),
    ] = Field(default_factory=list)

    @model_validator(mode="after")
    def _anchors_match_answer_kind(self) -> "RubricQuestion":
        values = [anchor.value for anchor in self.score_anchors]
        if self.answer_kind == "likert4" and values != ["1", "2", "3", "4"]:
            raise ValueError("likert4 questions need ordered 1-4 score anchors")
        if self.answer_kind != "likert4" and values:
            raise ValueError("only likert4 questions may define score anchors")
        if self.severity_scale == "minor_major" and self.answer_kind != "binary":
            raise ValueError("minor/major severity is a binary-question contract")
        return self


class RubricSectionVM(BaseModel):
    """One expandable annotation section and its ordered questions."""

    id: Annotated[RubricSection, Field(description="Stable section id.")]
    title: Annotated[str, Field(description="Annotator-facing heading.")]
    note: Annotated[str, Field(description="Evidence boundary for this section.")]
    questions: Annotated[list[RubricQuestion], Field(description="Ordered questions.")]


class JudgeAnnotationVM(BaseModel):
    """One stored judge verdict rendered inline beneath its calibrating
    question — judge-visible (``--show-judge``) calibration builds only.

    Display style follows the audio-quality adjudication packet's LLM-finding
    cards: a verdict chip plus judge/factor meta line, then the evidence and
    the pinned quote.
    """

    judge: Annotated[str, Field(description="Judge family display name.")]
    factor_id: Annotated[str, Field(description="Factor / axis / dimension id.")]
    verdict: Annotated[str, Field(description="Verdict display label.")]
    flagged: Annotated[
        bool, Field(description="Whether the verdict is a positive (a flag).")
    ] = False
    evidence: Annotated[str, Field(description="Judge rationale.")] = ""
    quote: Annotated[str, Field(description="Judge's pinned span.")] = ""
    turn_index: Annotated[
        Optional[int],
        Field(description="Flagged agent-turn index (utterance-level only)."),
    ] = None
    turn_html: Annotated[
        str,
        Field(
            description="Escaped flagged-turn text with the pinned quote "
            "<mark>-highlighted; empty for call-level findings."
        ),
    ] = ""
    context_text: Annotated[
        str, Field(description="Immediately preceding caller turn, when any.")
    ] = ""


class RubricCallVM(BaseModel):
    """Blind dual-evidence call passed to the combined template."""

    clip_id: Annotated[str, Field(description="Blind call id.")]
    ordinal: Annotated[int, Field(description="One-based display position.")]
    full_audio_file: Annotated[str, Field(description="Full-conversation WAV.")]
    agent_audio_file: Annotated[str, Field(description="Agent-only WAV.")]
    full_duration: Annotated[str, Field(description="Full audio duration.")]
    agent_duration: Annotated[str, Field(description="Agent-only audio duration.")]
    transcript_rows: Annotated[str, Field(description="Escaped full trace rows.")]
    agent_transcript: Annotated[list[str], Field(description="Agent utterances.")]
    sim_id: Annotated[str, Field(description="Source simulation identity.")]
    task_id: Annotated[str, Field(description="Source task identity.")]
    agent_turns: Annotated[
        list[NativenessAgentTurn],
        Field(description="Stable agent turns available for nativeness selection."),
    ]
    agent_gender: Annotated[Literal["male", "female", "unknown"], Field()]
    caller_gender: Annotated[Literal["male", "female", "unknown"], Field()]


class RubricPacketOptions(BaseModel):
    """Everything needed to build one combined rubric packet."""

    source_manifest: Annotated[
        Path, Field(description="Audio-quality manifest pinning the call cohort.")
    ]
    batch_name: Annotated[Optional[str], Field(description="Packet label.")] = None
    out_dir: Annotated[Optional[Path], Field(description="Packet directory.")] = None
    emit_zip: Annotated[bool, Field(description="Emit the shipping ZIP.")] = True

    @property
    def effective_batch_name(self) -> str:
        stem = self.source_manifest.parent.name or self.source_manifest.stem
        return self.batch_name or f"rubric_{stem}"

    @property
    def packet_dir(self) -> Path:
        return self.out_dir or (DEFAULT_OUTPUT_ROOT / self.effective_batch_name)


_QUALITY_GUIDANCE = {
    "unnecessary_repetition": (
        "Example issue: the agent gives the same instructions again after the caller "
        "already understood them. A needed clarification or correction is fine."
    ),
    "agent_caused_tool_error": (
        "Count this only when the tool returned an error and the agent caused it—for "
        "example, by omitting a required field. A system outage or bad information "
        "from the caller does not count."
    ),
    "auth_arg_mismatch": (
        "Count transcription or carryover errors only: the caller says “Smyth” but the "
        "agent looks up “Smith.” If the caller gave the wrong name, do not count it. "
        "Judge the written arguments in the agent's tool calls against what the "
        "caller said; how a name, letter, or number was PRONOUNCED belongs to Audio "
        "quality, not here."
    ),
    "incorrect_tool_parameters": (
        "This is the one for a tool call that can succeed but contains wrong details. "
        "Example: the caller asks for tomorrow but the agent submits today. Judge the "
        "written tool arguments only; pronunciation issues belong to Audio quality."
    ),
    "unnecessary_tool_call": (
        "Count it only if the call could reach the same correct outcome without that "
        "tool call. Repeating a lookup with no new purpose is an issue; making several "
        "legitimate attempts to find the caller’s order is not. A call that returned "
        "an error belongs to “avoidable tool error”, not here — and retrying after a "
        "failed lookup is fine."
    ),
}

_LLM_QUALITY_FACTORS = tuple(
    factor for factor in QUALITY_FACTORS if factor.evaluator in {"llm", "hybrid"}
)
if set(_QUALITY_GUIDANCE) != {factor.id for factor in _LLM_QUALITY_FACTORS}:
    raise ValueError("rubric packet guidance does not match the LLM quality catalog")

#: Violation-phrased instrument text for the DETERMINISTIC quality checkers —
#: calibration packets only (the standing rubric packet keeps them excluded).
#: The catalog's own ``question`` fields are judge-spec (pass-)phrased, so
#: they cannot be reused verbatim in a yes=violation binary instrument; each
#: entry is (question, guidance), keyed by quality factor id and validated
#: against the catalog below.
_DETERMINISTIC_QUALITY_QUESTIONS: dict[str, tuple[str, str]] = {
    "responsiveness": (
        "Did the agent leave the caller waiting — long silences or missing replies?",
        "Count clear dead air where a response was due: the caller finishes and "
        "the agent stays silent long enough to be awkward, or never answers. "
        "Ordinary thinking pauses and caller-side delays do not count.",
    ),
    "yielding": (
        "Did the agent keep talking after the caller interrupted?",
        "When the caller starts speaking over the agent to take the turn, the "
        "agent should stop. Count it when the agent talks through the "
        "interruption; finishing a short word is fine.",
    ),
    "inappropriate_interruption": (
        "Did the agent interrupt or talk over the caller?",
        "Count it when the agent starts speaking while the caller is "
        "mid-sentence and clearly not done. Overlapping a short caller "
        "acknowledgement is not an interruption.",
    ),
    "backchannel_selectivity": (
        "Did the agent wrongly stop or restart because of a caller backchannel?",
        "Backchannels are short acknowledgements (like 'uh-huh' or 'okay') "
        "that do not take the turn. Count it when such a sound derails the "
        "agent — it stops, restarts, or answers it as if it were a question.",
    ),
    "vocal_tic_selectivity": (
        "Did the agent wrongly react to a caller vocal tic (a cough, filler, "
        "or throat-clear)?",
        "Count it when a non-speech sound or bare filler makes the agent stop, "
        "restart, or respond as if it were speech.",
    ),
    "non_directed_selectivity": (
        "Did the agent wrongly respond to speech that was not directed at it?",
        "Background voices, or the caller talking to someone else, should be "
        "ignored. Count it when the agent answers or reacts to such speech.",
    ),
    "monologue": (
        "Did the agent deliver an overlong monologue instead of interacting?",
        "Count a single uninterrupted agent speech so long a caller would "
        "struggle to get a word in — reading everything out at once instead of "
        "checking in. An ordinary complete answer is fine.",
    ),
}
if set(_DETERMINISTIC_QUALITY_QUESTIONS) != {
    factor.id for factor in QUALITY_FACTORS if factor.evaluator == "deterministic"
}:
    raise ValueError(
        "calibration deterministic questions do not match the quality catalog"
    )

#: Violation-phrased instrument text for the DETERMINISTIC nativeness
#: checkers, keyed by (language, factor id) — calibration packets only. Every
#: enabled non-LLM factor of a drawn language MUST have an entry; the
#: builder below fails loudly otherwise, so a new checker cannot silently
#: ship without human calibration text.
_DETERMINISTIC_NATIVENESS_QUESTIONS: dict[tuple[str, str], tuple[str, str]] = {
    ("es", "register_formality"): (
        "Did the agent mix usted-family and tú-family address to the caller?",
        "One consistent address family is required across the call. Issue: "
        "switching between usted/ustedes and tú/te/ti (or vosotros) forms "
        "toward the caller. Either family used consistently is fine; quoted or "
        "caller-supplied wording does not count.",
    ),
    ("hi", "register_formality"): (
        "Did the agent address the caller with familiar तुम/तू forms?",
        "Polite आप-family customer address is required. Issue: any familiar "
        "तुम/तू-family address toward the caller. Turns with no customer "
        "pronoun at all are N/A; quoted or caller-supplied wording does not "
        "count.",
    ),
    ("zh", "register_formality"): (
        "Did the agent address the caller as 你/你们 instead of 您?",
        "Polite 您 is required toward the customer. Issue: informal 你-family "
        "customer address anywhere in the call. Quoted or caller-supplied "
        "wording does not count.",
    ),
    ("ko", "honorific_levels"): (
        "Did the agent repeatedly use 반말 (plain speech) toward the caller?",
        "Polite 합쇼체/해요체 levels are required. Issue: two or more "
        "customer-directed 반말 sentence endings across the call; one clipped "
        "plain fragment or bit of self-talk is tolerated.",
    ),
}


INTERACTION_QUALITY_SCORE_ANCHORS = [
    RubricScoreAnchor(
        value="1",
        label="Incredibly frustrating",
        description=(
            "The agent makes the call extremely difficult; a real caller would likely "
            "end it angry or give up."
        ),
    ),
    RubricScoreAnchor(
        value="2",
        label="Frustrating",
        description=(
            "The call has substantial avoidable friction, even if the caller eventually "
            "gets through it."
        ),
    ),
    RubricScoreAnchor(
        value="3",
        label="Acceptable",
        description=(
            "The interaction works with at most minor friction; it is not meaningfully "
            "unpleasant."
        ),
    ),
    RubricScoreAnchor(
        value="4",
        label="Pleasant",
        description=(
            "The interaction is smooth, responsive, and easy for a real caller."
        ),
    ),
]

NATIVENESS_SCORE_ANCHORS = [
    RubricScoreAnchor(
        value="1",
        label="Clearly non-native",
        description=(
            "The agent's language is persistently unnatural in wording, grammar, "
            "register, or conversational flow."
        ),
    ),
    RubricScoreAnchor(
        value="2",
        label="Noticeably non-native",
        description=(
            "The agent is understandable, but several choices sound unnatural to a "
            "native speaker."
        ),
    ),
    RubricScoreAnchor(
        value="3",
        label="Mostly native-like",
        description=(
            "The language is generally natural, with only small or occasional oddities."
        ),
    ),
    RubricScoreAnchor(
        value="4",
        label="Native-speaker-like",
        description=(
            "The agent's wording, grammar, register, and conversational flow are "
            "consistently natural."
        ),
    ),
]


def interaction_sections() -> list[RubricSectionVM]:
    """Non-deterministic interaction factors requiring full-call context."""
    questions = [
        RubricQuestion(
            id=OVERALL_INTERACTION_QUALITY_FACTOR_ID,
            section="interaction",
            answer_kind="likert4",
            question="Overall, how was the interaction quality?",
            guidance=(
                "Rate the experience created by the agent across the whole call. "
                "Consider responsiveness, turn-taking, comprehension, repetition, and "
                "workflow; do not penalize the agent for caller or infrastructure faults."
            ),
            score_anchors=INTERACTION_QUALITY_SCORE_ANCHORS,
        )
    ]
    questions.extend(
        RubricQuestion(
            id=factor.id,
            section="interaction",
            answer_kind="binary",
            question=factor.question,
            guidance=_QUALITY_GUIDANCE[factor.id],
            evaluation_level=factor.evaluation_level,
            severity_scale=_severity_scale_for("interaction", factor.id),
        )
        for factor in _LLM_QUALITY_FACTORS
    )
    return [
        RubricSectionVM(
            id="interaction",
            title="Interaction and workflow quality",
            note=(
                "Use the full conversation and tool trace. Deterministic timing, "
                "overlap, and exact duplicate-tool features are excluded."
            ),
            questions=questions,
        )
    ]


def nativeness_sections_for(language: str) -> list[RubricSectionVM]:
    """The selected language pack's transcript-judge nativeness factors.

    A pack that declares no enabled nativeness factors is the native
    control (en): there is no nativeness surface to judge, so the section —
    including the overall likert — is omitted entirely."""
    lang = language.strip().lower()
    from tau2.multilingual.registry import get_language_pack

    if not any(factor.enabled for factor in judge_factors_for(lang)):
        return []
    pack = get_language_pack(lang)
    language_name = pack.display_name if pack is not None else lang.upper()
    questions: list[RubricQuestion] = [
        RubricQuestion(
            id=OVERALL_NATIVENESS_FACTOR_ID,
            section="nativeness",
            answer_kind="likert4",
            question=f"Overall, how native-like was the agent's {language_name}?",
            guidance=(
                "Rate the agent's language across the whole call. Judge wording, grammar, "
                "register, and conversational flow; do not penalize speech-synthesis or "
                "acoustic issues covered by Audio quality."
            ),
            score_anchors=NATIVENESS_SCORE_ANCHORS,
        )
    ]
    for factor in judge_factors_for(lang):
        if not factor.enabled or factor.type != "judge":
            continue
        params = factor.params
        if params:
            question = params.language_question or params.question
            guidance = (
                f"{params.language_criterion}. Good: {params.language_allowed} "
                f"Issue: {params.language_violation}"
            )
        else:
            question = f"Did the agent violate the {factor.id} nativeness standard?"
            guidance = "Judge only this factor."
        questions.append(
            RubricQuestion(
                id=factor.id,
                section="nativeness",
                answer_kind="nativeness",
                question=question,
                guidance=guidance,
                evaluation_level=factor.evaluation_level,
            )
        )
    return [
        RubricSectionVM(
            id="nativeness",
            title=f"{language_name} language nativeness",
            note=(
                "Use the agent-only transcript. Whole-call factors get one label; "
                "for utterance factors, mark every offending agent turn. Audio is optional."
            ),
            questions=questions,
        )
    ]


#: Which human answer calibrates each semantic (EVA progression /
#: conciseness) judge dimension. Two progression dimensions are near-twins of
#: quality factors the interaction section already asks about — asking them
#: twice would confuse raters, so those dimensions reuse the interaction
#: answer (one human label serving two judges' calibration); the remaining
#: dimensions get their own questions in the semantic section below.
SEMANTIC_DIMENSION_SOURCES: dict[str, tuple[str, str]] = {
    "unnecessary_tool_calls": ("interaction", "unnecessary_tool_call"),
    "redundant_statements": ("interaction", "unnecessary_repetition"),
    "information_loss": ("semantic", "information_loss"),
    "question_quality": ("semantic", "question_quality"),
    "conciseness": ("semantic", "conciseness"),
}

#: The (section, factor_id) questions whose YES answers carry the two-level
#: minor/major severity: exactly the human sources of the semantic judge
#: dimensions, which grade major (rating 1) vs minor (rating 2).
SEMANTIC_SEVERITY_QUESTIONS: frozenset[tuple[str, str]] = frozenset(
    SEMANTIC_DIMENSION_SOURCES.values()
)


def _severity_scale_for(section: str, factor_id: str) -> Literal["none", "minor_major"]:
    """Severity contract of one question: minor/major on the semantic-source
    questions, plain answers everywhere else."""
    if (section, factor_id) in SEMANTIC_SEVERITY_QUESTIONS:
        return "minor_major"
    return "none"


def semantic_sections() -> list[RubricSectionVM]:
    """Conversation-progression and conciseness questions (fixed in-code).

    The question ids match the semantic judges' dimension names exactly so
    calibration joins need no mapping table beyond
    ``SEMANTIC_DIMENSION_SOURCES``.
    """
    questions = [
        RubricQuestion(
            id="information_loss",
            section="semantic",
            answer_kind="binary",
            question=(
                "Did the agent lose or ignore information the caller had "
                "already provided?"
            ),
            guidance=(
                "Count it when the agent asks again for something the caller "
                "already gave, or acts as if it was never said. A legitimate "
                "re-confirmation of a critical detail (an ID before a payment) "
                "does not count."
            ),
            evaluation_level="utterance",
            severity_scale="minor_major",
        ),
        RubricQuestion(
            id="question_quality",
            section="semantic",
            answer_kind="binary",
            question=(
                "Did the agent ask vague, redundant, or poorly targeted "
                "questions that slowed the conversation down?"
            ),
            guidance=(
                "Count questions that a competent human agent would not need "
                "to ask at that point — too broad, already answered, or "
                "missing the caller's stated goal. A precise clarifying "
                "question is fine."
            ),
            evaluation_level="utterance",
            severity_scale="minor_major",
        ),
        RubricQuestion(
            id="conciseness",
            section="semantic",
            answer_kind="binary",
            question=(
                "Were the agent's spoken turns noticeably longer or denser "
                "than the caller needed?"
            ),
            guidance=(
                "Count sustained verbosity: filler-heavy preambles, reading "
                "out exhaustive lists, or detail disproportionate to what the "
                "caller asked. One long turn that the caller explicitly "
                "requested does not count."
            ),
            evaluation_level="utterance",
            severity_scale="minor_major",
        ),
    ]
    return [
        RubricSectionVM(
            id="semantic",
            title="Conversation progression and conciseness",
            note=(
                "Use the full conversation. Judge whether the conversation "
                "moved forward efficiently; do not re-penalize issues you "
                "already marked in the interaction section."
            ),
            questions=questions,
        )
    ]


def delivery_pack_audio_questions_for(language: str) -> list[RubricQuestion]:
    """The language pack's delivery-factor questions, as audio-severity items.

    Question guidance is the pack's calibrated ``listen_for`` text — the same
    rubric text the delivery judge is prompted with — so human labels and
    judge verdicts answer the same question. Empty for languages whose pack
    declares no delivery factors.
    """
    from tau2.judges.delivery.factors import delivery_factors_for

    return [
        RubricQuestion(
            id=factor.id,
            section="audio",
            answer_kind="audio_severity",
            question=factor.id.replace("_", " ").capitalize(),
            guidance=factor.listen_for,
        )
        for factor in delivery_factors_for(language.strip().lower())
        if factor.enabled
    ]


def fidelity_dimension_ids() -> frozenset[str]:
    """The generic audio dimensions that calibrate the delivery ``fidelity``
    axis — every canonical dimension except ``intonation``, which is its own
    axis. Shared by the agreement report (label folding) and the judge-visible
    annotate build (which delivery cell renders beneath which question)."""
    return frozenset(
        dimension.id
        for dimension in AUDIO_QUALITY_DIMENSIONS
        if dimension.id != "intonation"
    )


def audio_sections() -> list[RubricSectionVM]:
    """Canonical spoken-fidelity and intonation dimensions."""
    dimensions = sorted(
        AUDIO_QUALITY_DIMENSIONS,
        key=lambda dimension: dimension.id != "intonation",
    )
    return [
        RubricSectionVM(
            id="audio",
            title="Audio quality",
            note=(
                "Listen to the agent-only audio. Rate 0 = no issue, 1 = minor, "
                "2 = clear, 3 = severe, or NA = no opportunity. Where a turn "
                "carries the [truncated] badge, the caller interrupted "
                "the agent: the shown text is an estimate that can stop "
                "mid-word, and the audio may finish the word or say slightly "
                "more — that boundary mismatch (and the abrupt stop itself) is "
                "NOT a delivery defect; everything before the cut counts "
                "normally."
            ),
            questions=[
                RubricQuestion(
                    id=dimension.id,
                    section="audio",
                    answer_kind="audio_severity",
                    question=dimension.title,
                    guidance=dimension.description,
                )
                for dimension in dimensions
            ],
        )
    ]


def rubric_sections_for(language: str) -> list[RubricSectionVM]:
    """All sections in annotator order."""
    return [
        *interaction_sections(),
        *nativeness_sections_for(language),
        *audio_sections(),
    ]


#: The generic accent surface's question id (calibration instrument only).
ACCENT_QUESTION_ID = "accent"

#: The pack delivery factor with its own accent surface; a pack that
#: declares it needs no generic accent question.
ACCENT_PACK_FACTOR_ID = "regional_accent_consistency"

#: Boundary line appended to every accent surface (generic and pack) —
#: added after the pt calibration wave filed mispronounced names, numbers,
#: and spelled letters under accent instead of the fidelity question.
ACCENT_BOUNDARY_NOTE = (
    "Mispronounced words, names, numbers, or spelled letters belong to the "
    "spoken-delivery fidelity question above — this question is ONLY about a "
    "sustained or switching accent."
)


#: The single audio fidelity question's id — matches the delivery judge's
#: folded cell (``calibration_instrument_cells``) so calibration joins need
#: no mapping.
FIDELITY_QUESTION_ID = "fidelity"


def fidelity_audio_question() -> RubricQuestion:
    """ONE audio question covering every generic spoken-delivery FIDELITY
    dimension (calibration instrument only).

    The instrument collapses the per-dimension fidelity questions into a
    single severity rating: raters flag ANY delivery defect and name the kind
    in the evidence box; the dimension taxonomy lives in the guidance so the
    closed catalog stays single-sourced. Intonation is its OWN axis and its
    own question (``intonation_audio_question``, cells-v3 unfold); pack
    delivery factors (and the generic accent question) also remain their own
    questions."""
    dimensions = ", ".join(
        dimension.title.lower()
        for dimension in AUDIO_QUALITY_DIMENSIONS
        if dimension.id in fidelity_dimension_ids()
    )
    return RubricQuestion(
        id=FIDELITY_QUESTION_ID,
        section="audio",
        answer_kind="audio_severity",
        question="Spoken-delivery fidelity issue",
        guidance=(
            "Any defect in WHAT the agent's audio says relative to what it "
            f"meant to say. It can be any of: {dimensions}. Rate the worst "
            "issue you hear across the call and name which kind(s) in the "
            "evidence box. Pauses, cadence, pitch, and delivery tone belong "
            "to the intonation question below. EXCEPTION: on a turn badged "
            "[truncated], the text/audio mismatch at the cut (a completed or "
            "extra word, or missing trailing words) is caller-caused "
            "truncation, not a delivery defect — never rate it."
        ),
    )


#: The intonation question's id — matches the delivery judge's intonation
#: axis cell so calibration joins need no mapping.
INTONATION_QUESTION_ID = "intonation"


def intonation_audio_question() -> RubricQuestion:
    """The intonation axis's own question (calibration instrument only,
    cells-v3 unfold): delivery HOW-it-sounds defects, separate from the
    fidelity WHAT-it-says question so each axis earns its own
    precision/recall numbers."""
    dimension = next(
        dimension
        for dimension in AUDIO_QUALITY_DIMENSIONS
        if dimension.id == INTONATION_QUESTION_ID
    )
    return RubricQuestion(
        id=INTONATION_QUESTION_ID,
        section="audio",
        answer_kind="audio_severity",
        question=dimension.title,
        guidance=(
            f"{dimension.description} Rate the worst issue you hear across "
            "the call. Wrong or garbled words belong to the fidelity "
            "question above. EXCEPTION: on a turn badged [truncated], the "
            "abrupt stop at the cut is the caller interrupting, not an "
            "intonation defect — never rate it; intonation issues before "
            "the cut still count."
        ),
    )


def accent_audio_question() -> RubricQuestion:
    """The generic accent surface (calibration instrument only).

    There is no stored accent judge yet — this question collects the human
    labels the future audio accent judge (pinned-variety pronunciation) will
    be cold-validated against. Folded away when the pack already asks it via
    ``regional_accent_consistency``."""
    return RubricQuestion(
        id=ACCENT_QUESTION_ID,
        section="audio",
        answer_kind="audio_severity",
        question="Wrong accent for the language variety",
        guidance=(
            "The agent's accent does not match the language's pinned native "
            "variety — e.g. an American- or British-English accent bleeding "
            "into the target language, or the wrong regional variety (a "
            "Mexican-Spanish accent on a Spain-pinned Spanish call). Rate the "
            "mismatch severity; NA only if there is too little speech to "
            f"tell. {ACCENT_BOUNDARY_NOTE}"
        ),
    )


def deterministic_interaction_questions() -> list[RubricQuestion]:
    """Binary questions for the deterministic quality checkers, catalog order
    (calibration instrument only)."""
    return [
        RubricQuestion(
            id=factor.id,
            section="interaction",
            answer_kind="binary",
            question=_DETERMINISTIC_QUALITY_QUESTIONS[factor.id][0],
            guidance=_DETERMINISTIC_QUALITY_QUESTIONS[factor.id][1],
            evaluation_level=factor.evaluation_level,
            severity_scale=_severity_scale_for("interaction", factor.id),
        )
        for factor in QUALITY_FACTORS
        if factor.evaluator == "deterministic"
    ]


def deterministic_nativeness_questions_for(language: str) -> list[RubricQuestion]:
    """Questions for the language pack's deterministic nativeness checkers
    (calibration instrument only); raises when an enabled checker has no
    fixed instrument text."""
    lang = language.strip().lower()
    questions: list[RubricQuestion] = []
    for factor in judge_factors_for(lang):
        if not factor.enabled or factor.type == "judge":
            continue
        entry = _DETERMINISTIC_NATIVENESS_QUESTIONS.get((lang, factor.id))
        if entry is None:
            raise ValueError(
                f"deterministic nativeness factor {lang}:{factor.id} has no "
                "calibration instrument question"
            )
        questions.append(
            RubricQuestion(
                id=factor.id,
                section="nativeness",
                answer_kind="nativeness",
                question=entry[0],
                guidance=entry[1],
                evaluation_level="call",
            )
        )
    return questions


def calibration_sections_for(language: str) -> list[RubricSectionVM]:
    """The judge-calibration instrument: every judged LLM surface in one
    packet (``calibration-cells-v3``).

    The interaction section's LLM quality factors, the language pack's
    nativeness factors (LLM AND deterministic checkers), and an audio section
    of one generic fidelity question (the fidelity dimensions folded; see
    ``fidelity_audio_question``) plus its own intonation question (cells-v3
    unfold; see ``intonation_audio_question``) plus the pack's delivery
    factors.

    Deliberately OUT of the instrument (2026-08-25 cut): the deterministic
    EVA turn-taking checkers — metric threshold events, not judged defects,
    with nothing to calibrate — and the semantic progression/conciseness
    suite.

    Packs declare only delivery factors the generic fidelity taxonomy does
    not cover (e.g. ``tone_meaning_flip``, ``regional_accent_consistency``),
    so pack questions ride along verbatim, and a generic accent question is
    added for packs without their own ``regional_accent_consistency`` factor
    (labels-first: the future accent judge is validated against this wave).

    The native control (a pack with no nativeness factors, i.e. en) gets
    neither a nativeness section nor the generic accent question — accent
    mismatch is a nativeness-adjacent surface with nothing to calibrate
    against in the control language.
    """
    interaction = [
        section.model_copy(
            update={
                "note": (
                    "Use the full conversation and tool trace, including "
                    "timing and overlap."
                ),
            }
        )
        for section in interaction_sections()
    ]
    nativeness = nativeness_sections_for(language)
    checker_questions = deterministic_nativeness_questions_for(language)
    if checker_questions:
        nativeness = [
            section.model_copy(
                update={"questions": [*section.questions, *checker_questions]}
            )
            for section in nativeness
        ]
    pack_questions = [
        question
        if question.id != ACCENT_PACK_FACTOR_ID
        else question.model_copy(
            update={"guidance": f"{question.guidance} {ACCENT_BOUNDARY_NOTE}"}
        )
        for question in delivery_pack_audio_questions_for(language)
    ]
    has_accent_factor = any(
        question.id == ACCENT_PACK_FACTOR_ID for question in pack_questions
    )
    native_control = not nativeness
    audio = [
        section.model_copy(
            update={
                "questions": [
                    fidelity_audio_question(),
                    intonation_audio_question(),
                    *pack_questions,
                    *(
                        []
                        if has_accent_factor or native_control
                        else [accent_audio_question()]
                    ),
                ]
            }
        )
        for section in audio_sections()
    ]
    return [
        *interaction,
        *nativeness,
        *audio,
    ]


def _template_environment() -> Environment:
    template_dir = Path(__file__).parent / "templates"
    return Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _resolve_results_path(raw: str) -> Path:
    path = Path(raw)
    resolved = path if path.is_absolute() else Path.cwd() / path
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"source results not found: {raw} (resolved {resolved})"
        )
    return resolved


def _load_selected_sims(
    entries: list[AudioQualityEntry],
) -> tuple[dict[tuple[str, str], LoadedSim], dict[str, Path]]:
    resolved_by_raw = {
        entry.results_path: _resolve_results_path(entry.results_path)
        for entry in entries
    }
    wanted = {
        (str(resolved_by_raw[entry.results_path]), entry.sim_id) for entry in entries
    }
    loaded: dict[tuple[str, str], LoadedSim] = {}
    for item in iter_loaded_sims(sorted(set(resolved_by_raw.values()))):
        key = (str((item.results_dir / "results.json").resolve()), item.sim.id)
        if key in wanted:
            loaded[key] = item
    missing = sorted(wanted - set(loaded))
    if missing:
        raise ValueError(f"selected simulations missing from results: {missing}")
    return loaded, resolved_by_raw


def _duration_label(path: Path) -> str:
    with wave.open(str(path), "rb") as audio:
        seconds = audio.getnframes() / audio.getframerate()
    minutes, remainder = divmod(round(seconds), 60)
    return f"{minutes:02d}:{remainder:02d}"


def _selection_hash(
    source: ArtifactManifest,
    sections: list[RubricSectionVM],
    contexts: list[dict[str, str]],
) -> str:
    payload = {
        "source_batch_id": source.batch_id,
        "packet_version": RUBRIC_PACKET_VERSION,
        "entries": [
            {
                "clip_id": entry.clip_id,
                "results_path": entry.results_path,
                "sim_id": entry.sim_id,
            }
            for entry in (source.audio_quality_entries or [])
        ],
        "sections": [section.model_dump() for section in sections],
        "contexts": contexts,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def render_rubric_page(
    *,
    batch_id: str,
    batch_name: str,
    language: str,
    calls: list[RubricCallVM],
    sections: list[RubricSectionVM],
    packet_version: str = RUBRIC_PACKET_VERSION,
    judge_annotations: Optional[dict[str, dict[str, list[JudgeAnnotationVM]]]] = None,
    banner: Optional[str] = None,
) -> str:
    """Render the single-page dual-evidence annotation UI.

    Shared by the combined rubric packet and the judge-calibration packet —
    both are blind instruments over the same call/section view models; only
    the section composition and version tag differ.

    ``judge_annotations`` (clip_id → "section:question_id" → annotations) is
    the calibration packet's OPT-IN judge-visible mode: when given, every
    listed annotation renders inline beneath its question and the page marks
    itself judge-visible in ``PACKET_CONFIG`` (the export JS stamps that into
    each nativeness row's provenance). The DEFAULT (None) stays blind by
    construction — the template receives no judge material at all. ``banner``
    renders a fixed sticky warning strip across the page.
    """
    template_root = Path(__file__).parent / "templates"
    # Display numbering ("3.6" = section 3, question 6) is pure presentation,
    # derived from section order at render time: it rides on PACKET_CONFIG so
    # the page's blocker messages and exports can cite it, but never enters
    # the question models or the instrument hash.
    questions = [
        {**question.model_dump(), "number": f"{section_index}.{question_index}"}
        for section_index, section in enumerate(sections, 1)
        for question_index, question in enumerate(section.questions, 1)
    ]
    config = {
        "batch_id": batch_id,
        "batch_name": batch_name,
        "packet_version": packet_version,
        "language": language,
        "judge_visible": judge_annotations is not None,
        "rubric_csv_headers": RubricAnnotationRow.headers(),
        "nativeness_csv_headers": NativenessHumanLabelRow.headers(),
        "clip_ids": [call.clip_id for call in calls],
        "questions": questions,
        "contexts": {
            call.clip_id: {
                "agent_gender": call.agent_gender,
                "caller_gender": call.caller_gender,
            }
            for call in calls
        },
        "calls": {
            call.clip_id: {
                "simulation_id": call.sim_id,
                "task_id": call.task_id,
                "agent_turns": [turn.model_dump() for turn in call.agent_turns],
            }
            for call in calls
        },
    }
    config_json = json.dumps(config, ensure_ascii=False).replace("</", "<\\/")
    return (
        _template_environment()
        .get_template("rubric.html.j2")
        .render(
            packet_config=config_json,
            batch_name=batch_name,
            language=language,
            calls=calls,
            sections=sections,
            question_count=len(questions),
            judge_annotations=judge_annotations,
            banner=banner,
            css=(template_root / "static" / "rubric.css").read_text(),
            js=(template_root / "static" / "rubric.js").read_text(),
        )
    )


def _copy_audio(source: Path, output: Path) -> None:
    """Atomically copy one lossless packet audio file."""
    if not source.is_file():
        raise FileNotFoundError(f"audio evidence not found: {source}")
    staging = output.with_name(f".{output.stem}.tmp{output.suffix}")
    staging.unlink(missing_ok=True)
    shutil.copy2(source, staging)
    staging.replace(output)


def _source_agent_clip(source_manifest: Path, audio_file: str) -> Path:
    """Resolve a source packet clip without allowing path traversal."""
    root = source_manifest.parent.resolve()
    path = (root / audio_file).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"agent audio path escapes source packet: {audio_file}")
    return path


def build_rubric_packet(options: RubricPacketOptions) -> Path:
    """Build the combined dual-evidence annotation packet."""
    source = ArtifactManifest.model_validate_json(options.source_manifest.read_text())
    if source.kind != AUDIO_QUALITY_KIND or not source.audio_quality_entries:
        raise ValueError(
            "rubric-packet requires a packet_audio_quality manifest with entries"
        )
    language = (source.language or "").strip().lower()
    if not language:
        raise ValueError("source manifest must declare one language")

    sections = rubric_sections_for(language)
    questions = [question for section in sections for question in section.questions]
    if not questions:
        raise ValueError(f"language '{language}' has no rubric questions")
    loaded, resolved_by_raw = _load_selected_sims(source.audio_quality_entries)

    packet_dir = options.packet_dir
    calls_dir = packet_dir / "calls"
    calls_dir.mkdir(parents=True, exist_ok=True)
    calls: list[RubricCallVM] = []
    entries: list[RubricPacketEntry] = []

    for ordinal, entry in enumerate(source.audio_quality_entries, 1):
        key = (str(resolved_by_raw[entry.results_path]), entry.sim_id)
        item = loaded[key]
        full_source = find_both_wav(item.results_dir, item.sim)
        if full_source is None:
            raise FileNotFoundError(
                f"sim {entry.sim_id}: no both.wav under {item.results_dir}"
            )
        agent_source = _source_agent_clip(options.source_manifest, entry.audio_file)
        full_relative = f"calls/{entry.clip_id}_full.wav"
        agent_relative = f"calls/{entry.clip_id}_agent.wav"
        full_output = packet_dir / full_relative
        agent_output = packet_dir / agent_relative
        _copy_audio(full_source, full_output)
        _copy_audio(agent_source, agent_output)

        provider = item.sim.agent_provider or entry.provider
        agent_gender = resolve_agent_gender(provider, item.sim.agent_voice) or "unknown"
        caller_gender = (
            caller_gender_for_task(entry.task_id, language, entry.domain) or "unknown"
        )
        agent_turns = [
            NativenessAgentTurn(
                index=turn.index,
                turn_id=f"agent-turn-{turn.index:03d}",
                text=turn.text,
                preceding_customer_text=turn.preceding_user_text,
                interrupted=turn.interrupted,
            )
            for turn in build_agent_turns(item.sim)
        ]
        calls.append(
            RubricCallVM(
                clip_id=entry.clip_id,
                ordinal=ordinal,
                full_audio_file=full_relative,
                agent_audio_file=agent_relative,
                full_duration=_duration_label(full_output),
                agent_duration=_duration_label(agent_output),
                transcript_rows=str(Markup(generate_tick_rows(item.sim))),
                agent_transcript=agent_speech_transcript(item.sim),
                sim_id=entry.sim_id,
                task_id=entry.task_id,
                agent_turns=agent_turns,
                agent_gender=agent_gender,
                caller_gender=caller_gender,
            )
        )
        entries.append(
            RubricPacketEntry(
                **entry.model_dump(exclude={"audio_file"}),
                full_audio_file=full_relative,
                agent_audio_file=agent_relative,
                agent_gender=agent_gender,
                caller_gender=caller_gender,
                agent_turns=agent_turns,
            )
        )
        logger.info(
            f"rubric {entry.clip_id}: full + agent-only {entry.domain} "
            f"task {entry.task_id} / sim {entry.sim_id} "
            f"(agent={agent_gender}, caller={caller_gender})"
        )

    contexts = [
        {
            "clip_id": call.clip_id,
            "agent_gender": call.agent_gender,
            "caller_gender": call.caller_gender,
        }
        for call in calls
    ]
    selection_sha = _selection_hash(source, sections, contexts)
    batch_id = derive_batch_id(
        RUBRIC_PACKET_KIND,
        options.effective_batch_name,
        {"selection": selection_sha},
    )
    html = render_rubric_page(
        batch_id=batch_id,
        batch_name=options.effective_batch_name,
        language=language,
        calls=calls,
        sections=sections,
    )
    (packet_dir / "index.html").write_text(html)
    manifest = ArtifactManifest(
        kind=RUBRIC_PACKET_KIND,
        batch_id=batch_id,
        batch_name=options.effective_batch_name,
        created_at=datetime.now(timezone.utc).isoformat(),
        git_sha=git_sha(),
        language=language,
        form="combined_rubric",
        provenance={
            "source_manifest": str(options.source_manifest),
            "source_batch_id": source.batch_id,
            "selection_sha256": selection_sha,
            "rubric_packet_version": RUBRIC_PACKET_VERSION,
            "quality_rubric_version": QUALITY_RUBRIC_VERSION,
            "audio_quality_rubric_version": AUDIO_QUALITY_RUBRIC_VERSION,
            "question_count": len(questions),
            "sections": [section.model_dump() for section in sections],
            "evidence_modes": {
                "full_conversation": "caller + agent audio, transcript, and tool trace",
                "agent_only": "complete ordered agent speech and transcript",
            },
            "context_policy": {
                "agent_gender": "resolved from recorded provider and voice",
                "caller_gender": "resolved from the task caller-gender sidecar",
                "unknown_policy": "unknown; never infer from name or audio",
            },
            "review_policy": (
                "Audio is required only for the audio-quality section. Text should be "
                "enough for interaction and nativeness unless evidence is ambiguous."
            ),
            "judge_policy": "No machine judge verdicts are rendered in this packet.",
        },
        files=[],
        rubric_entries=entries,
    )
    manifest_path = write_json_artifact(packet_dir / PACKET_MANIFEST_NAME, manifest)
    sync_packet_zip(packet_dir, emit=options.emit_zip)
    logger.info(
        f"combined rubric packet '{options.effective_batch_name}' ({batch_id}): "
        f"{len(entries)} calls x {len(questions)} questions -> {packet_dir}"
    )
    return manifest_path


__all__ = [
    "RUBRIC_INSTRUMENT_FILES",
    "RUBRIC_PACKET_VERSION",
    "SEMANTIC_DIMENSION_SOURCES",
    "SEMANTIC_QUESTIONS_VERSION",
    "SEMANTIC_SEVERITY_QUESTIONS",
    "JudgeAnnotationVM",
    "RubricCallVM",
    "RubricPacketOptions",
    "RubricQuestion",
    "RubricSectionVM",
    "audio_sections",
    "build_rubric_packet",
    "calibration_sections_for",
    "delivery_pack_audio_questions_for",
    "fidelity_dimension_ids",
    "interaction_sections",
    "nativeness_sections_for",
    "render_rubric_page",
    "rubric_instrument_sha256",
    "rubric_sections_for",
    "semantic_sections",
]
