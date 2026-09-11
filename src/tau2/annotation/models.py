# Copyright Sierra
"""Single source of truth for every annotation sheet contract.

Every human-facing sheet (CSV → workbook → filled CSV) is defined ONCE here as a
pydantic row model. Headers, dropdown option lists, hidden trace columns, and
column widths are all **generated from the model** via ``Col(...)`` metadata —
the CSV writer, the analyzer, and the xlsx workbook builder read the same
``ColumnSpec``s, so the annotator contract can never drift between them.

Verdict enums carry lenient ``BeforeValidator``s absorbing every legacy string
form annotators have produced (``"✅ yes"``, ``"n/a"``, ``"no-opp"`` …); an
unrecognized non-empty label is a LOUD validation error, never a silently
dropped row.

The one non-static shape is the WIDE cold sheet (one row per call, one dropdown
column per judge factor): ``ColdSheet`` owns its melt/unmelt against the
``FactorKeyRow`` sidecar. Long-format cold sheets no longer exist.
"""

import json
import re
from enum import Enum
from typing import (
    Annotated,
    Any,
    Iterator,
    Literal,
    Mapping,
    Optional,
    Union,
)

from pydantic import BaseModel, BeforeValidator, Field, model_validator

from tau2.data_model.simulation import JudgeOutcome

# ============================================================================
# Column metadata: Col() → ColumnSpec
# ============================================================================


class ColumnSpec(BaseModel):
    """One sheet column, generated from a row model field's ``Col`` metadata."""

    field: Annotated[str, Field(description="The row-model field name.")]
    label: Annotated[str, Field(description="The header shown to annotators.")]
    options: Annotated[
        Optional[list[str]],
        Field(description="Dropdown option list (data validation), if any."),
    ] = None
    hidden: Annotated[
        bool,
        Field(description="Trace column hidden when built into a workbook."),
    ] = False
    width: Annotated[
        Optional[int], Field(description="Column width in characters, if pinned.")
    ] = None
    description: Annotated[str, Field(description="What the column holds.")] = ""


def Col(
    label: str,
    *,
    options: Optional[list[str]] = None,
    hidden: bool = False,
    width: Optional[int] = None,
    description: str = "",
) -> Any:
    """Field metadata declaring a sheet column on a ``SheetRow`` field."""
    return Field(
        description=description or label,
        json_schema_extra={
            "sheet": {
                "label": label,
                "options": options,
                "hidden": hidden,
                "width": width,
                "description": description,
            }
        },
    )


def _render_cell(value: Any) -> str:
    """Render one field value to its sheet cell (enums show display labels).

    Lists (multi-select cells) render as comma-joined RAW values, not display
    labels — displays may themselves contain commas, which would break the
    cell's round-trip split.
    """
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(str(getattr(v, "value", v)) for v in value)
    if isinstance(value, DisplayEnum):
        return value.display
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


class SheetRow(BaseModel):
    """Base for every fixed-shape sheet row: columns generated from the model."""

    @classmethod
    def columns(cls) -> list[ColumnSpec]:
        """The sheet columns, in field-declaration (= reading) order."""
        specs: list[ColumnSpec] = []
        for name, finfo in cls.model_fields.items():
            extra = finfo.json_schema_extra
            sheet = extra.get("sheet") if isinstance(extra, dict) else None
            if sheet is None:
                raise TypeError(
                    f"{cls.__name__}.{name} lacks Col(...) metadata — every "
                    "SheetRow field must declare its sheet column"
                )
            specs.append(ColumnSpec(field=name, **sheet))
        return specs

    @classmethod
    def headers(cls) -> list[str]:
        return [spec.label for spec in cls.columns()]

    @classmethod
    def dropdowns(cls) -> dict[str, list[str]]:
        """{header -> dropdown options} for columns carrying data validation."""
        return {spec.label: spec.options for spec in cls.columns() if spec.options}

    @classmethod
    def hidden_labels(cls) -> set[str]:
        return {spec.label for spec in cls.columns() if spec.hidden}

    @classmethod
    def widths(cls) -> dict[str, int]:
        """{header -> pinned width in chars} for columns that declare one."""
        return {
            spec.label: spec.width for spec in cls.columns() if spec.width is not None
        }

    def to_cells(self) -> dict[str, str]:
        """{header -> rendered cell} (enums render their display labels)."""
        return {
            spec.label: _render_cell(getattr(self, spec.field))
            for spec in self.columns()
        }

    @classmethod
    def from_cells(cls, cells: Mapping[str, str]) -> "SheetRow":
        """Validate one CSV row (header-keyed) back into a typed row."""
        data = {spec.field: cells.get(spec.label, "") for spec in cls.columns()}
        return cls.model_validate(data)


# ============================================================================
# Verdict enums (display labels + lenient legacy-form normalizers)
# ============================================================================


class DisplayEnum(str, Enum):
    """A str enum whose members carry an annotator-facing display label."""

    display: str

    def __new__(cls, value: str, display: str):
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.display = display
        return obj

    @classmethod
    def options(cls) -> list[str]:
        """Dropdown option list = the display labels, in member order."""
        return [member.display for member in cls]


class YesNo(DisplayEnum):
    """A binary annotator verdict (rendered Y / N)."""

    YES = ("yes", "Y")
    NO = ("no", "N")


class PrecisionVerdict(DisplayEnum):
    """Adjudication of one judge FAIL: was the flagged violation real?"""

    REAL = ("real", "✅ yes")
    FALSE_ALARM = ("false_alarm", "❌ no")
    UNSURE = ("unsure", "🤔 unsure")


class LocalizationVerdict(DisplayEnum):
    """A native reviewer's call on one task's localized scenario text."""

    APPROVE = ("approve", "✅ approve")
    DENY = ("deny", "❌ deny")
    UNSURE = ("unsure", "🤔 unsure")


class ColdLabel(DisplayEnum):
    """A blind cold-sheet label for one (call, factor) cell."""

    YES = ("yes", "yes")
    NO = ("no", "no")
    NO_OPPORTUNITY = ("no_opportunity", "no-opportunity")
    UNSURE = ("unsure", "unsure")


class PipelineVerdict(DisplayEnum):
    """The translation pipeline's own verdict for a row (hidden trace column)."""

    VERIFIED = ("verified", "verified")
    FLAGGED = ("flagged", "flagged")
    UNTRANSLATED = ("untranslated", "untranslated")


class AuditStatus(DisplayEnum):
    """Verification status of one audit nuance row (Round 2 is verify-only)."""

    AUTO = ("auto", "🔵 Auto — needs review")
    CONFIRMED = ("confirmed", "✅ Confirmed")
    NOT_TRUE = ("not_true", "❌ Not true")
    UNSURE = ("unsure", "🤔 Unsure")


class AudioIssueRating(str, Enum):
    """Severity of one audible delivery issue in an agent-only clip."""

    NO_OPPORTUNITY = "NA"
    NO_ISSUE = "0"
    MINOR = "1"
    CLEAR = "2"
    SEVERE = "3"


class RubricAnswer(DisplayEnum):
    """Answer vocabulary for the combined human rubric packet."""

    YES = ("yes", "Yes")
    NO = ("no", "No")
    NO_OPPORTUNITY = ("no_opportunity", "N/A")
    AUDIO_NO_ISSUE = ("0", "0")
    AUDIO_MINOR = ("1", "1")
    AUDIO_CLEAR = ("2", "2")
    AUDIO_SEVERE = ("3", "3")
    SCORE_FOUR = ("4", "4")
    AUDIO_NO_OPPORTUNITY = ("NA", "NA")


class RubricAnnotationSection(DisplayEnum):
    """Closed section vocabulary for long-format rubric annotation rows."""

    INTERACTION = ("interaction", "interaction")
    SEMANTIC = ("semantic", "semantic")
    NATIVENESS = ("nativeness", "nativeness")
    AUDIO = ("audio", "audio")
    CUSTOM = ("custom", "custom")


class SemanticSeverity(DisplayEnum):
    """Human severity of a YES on a semantic-source question — the two-level
    scale the semantic judges grade on (major = rating 1, minor = rating 2),
    NOT the 1-4 nativeness severity."""

    MINOR = ("minor", "minor")
    MAJOR = ("major", "major")


class RubricPhase(DisplayEnum):
    """Which layer of a two-phase (blind-then-reveal) packet a row records.

    Blank on rows from a blind packet (single layer). A judge-visible
    two-phase packet exports BOTH layers for every question: the pre-reveal
    row is the rater's locked blind answer (routed to the blind/recall pool)
    and the post-reveal row is their answer after the judge findings revealed
    (routed to the precision pool).
    """

    PRE_REVEAL = ("pre_reveal", "pre_reveal")
    POST_REVEAL = ("post_reveal", "post_reveal")


OVERALL_INTERACTION_QUALITY_FACTOR_ID = "overall_interaction_quality"
OVERALL_NATIVENESS_FACTOR_ID = "overall_nativeness"
OVERALL_SCORE_FACTOR_IDS = frozenset(
    {
        OVERALL_INTERACTION_QUALITY_FACTOR_ID,
        OVERALL_NATIVENESS_FACTOR_ID,
    }
)


class ErrorSource(DisplayEnum):
    """Who caused the first critical error in a reviewed voice call."""

    AGENT = ("agent", "Agent Error")
    USER = ("user", "User Simulator Error")
    SYSTEM = ("system", "System Error (framework/infrastructure)")


class ErrorType(DisplayEnum):
    """What kind of failure the first critical error was."""

    TRANSCRIPTION = ("transcription", "Transcription (ASR/speech-to-text)")
    VAD = ("vad", "VAD (turn-taking/interruption)")
    LOGICAL = ("logical", "Logical (reasoning/tool call/instruction)")
    HALLUCINATION = ("hallucination", "Hallucination (made up info)")
    UNRESPONSIVE = ("unresponsive", "Unresponsive (no response/latency)")
    EARLY_TERMINATION = ("early_termination", "Early Termination (ended prematurely)")


class ExperienceFactor(DisplayEnum):
    """What made the call unpleasant — the caller-experience factor taxonomy.

    Annotators multi-select every contributing factor and then name exactly
    one PRIMARY factor among them; we are mining what actually makes people
    hang up. ``SIM_MULTILINGUAL_BROKEN`` and ``OTHER`` require elaboration in
    the experience notes.

    ``REPETITION`` is deliberately a SYMPTOM factor and overlaps with
    ``TRANSCRIPTION`` and ``COMPREHENSION_LOGIC``, its two usual causes. The
    multi-select carries the overlap — an annotator selects the symptom and
    the cause, and the PRIMARY choice says which one dominated the call — so
    the taxonomy must never be read as mutually exclusive.
    """

    TRANSCRIPTION = (
        "transcription",
        "Heard/said wrong (mishearings, wrong name or number)",
    )
    LATENCY = ("latency", "Latency (dead air, long pauses, unresponsive stretches)")
    COMPREHENSION_LOGIC = (
        "comprehension_logic",
        "Didn't understand or follow (ignored/forgot what was said, illogical)",
    )
    REPETITION = (
        "repetition",
        "Had to repeat or re-explain (said the same thing twice, spelled a name again)",
    )
    TURN_TAKING = (
        "turn_taking",
        "Turn-taking (talked over the caller, interrupted, cut off)",
    )
    UNNATURAL_LANGUAGE = (
        "unnatural_language",
        "Unnatural or non-native wording (translationese, odd phrasing)",
    )
    SOCIAL_OFFENSE = (
        "social_offense",
        "Impolite or socially off (rude, misgendering, wrong register)",
    )
    INTONATION_DELIVERY = (
        "intonation_delivery",
        "Robotic or off intonation/delivery",
    )
    SIM_MULTILINGUAL_BROKEN = (
        "sim_multilingual_broken",
        "User-sim multilingual breakage — the CALLER side is broken (e.g. "
        "mispronounces its own name), not the agent's fault; elaborate in notes",
    )
    OTHER = ("other", "Other (explain in notes)")


# Factors whose selection demands elaboration in the experience notes.
EXPERIENCE_ELABORATE_FACTORS = [
    ExperienceFactor.SIM_MULTILINGUAL_BROKEN,
    ExperienceFactor.OTHER,
]


class BreakingPoint(DisplayEnum):
    """For a bad call: was the damage one moment, or the whole experience?"""

    OVERALL = ("overall", "Overall experience (no single moment)")
    TICK = ("tick", "A specific moment (give the tick)")


class IssueType(str, Enum):
    """The closed translation-review issue taxonomy (the dropdown options)."""

    MEANING = "meaning"
    NATURALNESS = "naturalness"
    REGISTER = "register"
    TERMINOLOGY = "terminology"
    ENTITY_FORMAT = "entity/format"
    OTHER = "other"


class PromptBedItemKind(DisplayEnum):
    """Which section of a prompt+bed review packet a long-format row is from."""

    PROMPT = ("prompt", "prompt")
    BED = ("bed", "bed")
    VOICE = ("voice", "voice")


class PromptBedVerdict(DisplayEnum):
    """A reviewed prompt section or background bed: fine, or flagged."""

    OK = ("ok", "OK")
    ISSUE = ("issue", "issue")


def _prep(value: Any) -> Optional[str]:
    """Common normalizer front half: None/enum passthrough, blank → None."""
    if value is None or isinstance(value, Enum):
        return value  # type: ignore[return-value]
    s = str(value).strip().lower()
    return s if s else None


def norm_yes_no(value: Any) -> Any:
    """Absorb Y/N/yes/no in any case; blank → None; junk is a loud error."""
    s = _prep(value)
    if s is None or isinstance(s, Enum):
        return s
    if s in {"y", "yes"} or "✅" in s:
        return YesNo.YES
    if s in {"n", "no"} or "❌" in s:
        return YesNo.NO
    raise ValueError(f"unrecognized Y/N verdict: {value!r}")


def norm_precision(value: Any) -> Any:
    """Absorb every legacy precision form: '✅ yes' / '❌ no' / '🤔 unsure',
    bare yes/no, and the canonical enum values. Blank → None (unlabeled)."""
    s = _prep(value)
    if s is None or isinstance(s, Enum):
        return s
    if s in {"real", "false_alarm"}:
        return PrecisionVerdict(s)
    if "unsure" in s or "🤔" in s:
        return PrecisionVerdict.UNSURE
    if s.startswith("y") or "✅" in s:
        return PrecisionVerdict.REAL
    if s.startswith("n") or "❌" in s or s == "false":
        return PrecisionVerdict.FALSE_ALARM
    raise ValueError(f"unrecognized precision verdict: {value!r}")


def norm_cold(value: Any) -> Any:
    """Absorb every legacy cold-label form ('n/a', 'no-opp', opportunity
    substrings, emoji forms). Blank → None (unlabeled)."""
    s = _prep(value)
    if s is None or isinstance(s, Enum):
        return s
    if "opportunity" in s or s in {"n/a", "na", "noopp", "no-opp", "none"}:
        return ColdLabel.NO_OPPORTUNITY
    if s.startswith("y") or "✅" in s:
        return ColdLabel.YES
    if "unsure" in s or "🤔" in s:
        return ColdLabel.UNSURE
    if s.startswith("n") or "❌" in s:
        return ColdLabel.NO
    raise ValueError(f"unrecognized cold label: {value!r}")


def norm_judge_outcome(value: Any) -> Any:
    """Stored judge outcome cell → Optional[JudgeOutcome].

    Blank and the legacy ``"missing"`` sentinel both mean "no stored verdict"
    → None (the runtime enum deliberately has no MISSING member).
    """
    s = _prep(value)
    if s is None or isinstance(s, Enum):
        return s
    if s == "missing":
        return None
    return JudgeOutcome(s)


def _norm_display_enum(enum_cls: type[DisplayEnum], value: Any) -> Any:
    """Absorb the raw value OR the display label of a DisplayEnum member."""
    s = _prep(value)
    if s is None or isinstance(s, Enum):
        return s
    for member in enum_cls:
        if s == member.value or s == member.display.lower():
            return member
    raise ValueError(f"unrecognized {enum_cls.__name__} cell: {value!r}")


def _display_enum_cell(enum_cls: type[DisplayEnum]) -> Any:
    """``Optional[enum_cls]`` cell type absorbing raw values or display labels."""
    return Annotated[
        Optional[enum_cls], BeforeValidator(lambda v: _norm_display_enum(enum_cls, v))
    ]


def norm_likert(value: Any) -> Any:
    """Blank → None (unrated); numeric strings become ints; junk is loud."""
    s = _prep(value)
    if s is None or not isinstance(s, str):
        return s
    return int(s)


def norm_likert_na(value: Any) -> Any:
    """The 1-4 rubric cell contract: blank → None (not yet rated);
    'NA'/'N/A' → the NA sentinel (unable to evaluate — NOT a low score,
    excluded from aggregation); numeric strings become ints; junk is loud."""
    s = _prep(value)
    if s is None or not isinstance(s, str):
        return s
    if s in {"na", "n/a"}:
        return NA_SCORE
    return int(s)


def norm_bool(value: Any) -> Any:
    """Browser booleans ('true'/'false'/'1'/'0'); blank → False."""
    if isinstance(value, bool):
        return value
    s = _prep(value)
    if s is None:
        return False
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"unrecognized boolean cell: {value!r}")


def norm_int(value: Any) -> Any:
    """Blank → 0; numeric strings become ints."""
    if isinstance(value, int):
        return value
    s = _prep(value)
    if s is None:
        return 0
    return int(float(s)) if "." in s else int(s)


def norm_issue_type(value: Any) -> Any:
    """Blank → None (no issue named); anything else must be in the taxonomy."""
    s = _prep(value)
    if s is None or isinstance(s, Enum):
        return s
    return IssueType(s)


def norm_audit_status(value: Any) -> Any:
    s = _prep(value)
    if s is None or isinstance(s, Enum):
        return s
    for member in AuditStatus:
        if s == member.value or s == member.display.lower():
            return member
    if "auto" in s or "🔵" in s:
        return AuditStatus.AUTO
    if "confirm" in s or "✅" in s:
        return AuditStatus.CONFIRMED
    if "not true" in s or "❌" in s:
        return AuditStatus.NOT_TRUE
    if "unsure" in s or "🤔" in s:
        return AuditStatus.UNSURE
    raise ValueError(f"unrecognized audit status: {value!r}")


def norm_prompt_bed_verdict(value: Any) -> Any:
    """Absorb the packet's per-kind radio labels — 'OK', 'Issues found'
    (prompt sections), 'Sounds off' (beds) — plus emoji/legacy yes-no forms.
    Blank → None (unreviewed)."""
    s = _prep(value)
    if s is None or isinstance(s, Enum):
        return s
    if s in {"ok", "okay"} or s.startswith("y") or "✅" in s:
        return PromptBedVerdict.OK
    if "issue" in s or "off" in s or "❌" in s or s.startswith("n"):
        return PromptBedVerdict.ISSUE
    raise ValueError(f"unrecognized prompt/bed verdict: {value!r}")


def norm_opt_int(value: Any) -> Any:
    """Blank → None (NOT 0 — a blank tick anchor must never become tick 0);
    numeric strings become ints."""
    if isinstance(value, int):
        return value
    s = _prep(value)
    if s is None:
        return None
    return int(float(s)) if "." in s else int(s)


def norm_opt_float(value: Any) -> Any:
    """Blank → None; numeric strings become floats."""
    if isinstance(value, (int, float)):
        return float(value)
    s = _prep(value)
    if s is None:
        return None
    return float(s)


def norm_int_list(value: Any) -> Any:
    """A multi-int cell (selected agent-turn indices): blank → []; a
    comma/semicolon-joined string (bracketed JSON-array form tolerated)
    becomes the sorted, deduplicated int list; junk is a loud error."""
    if isinstance(value, (list, tuple)):
        return sorted({int(v) for v in value})
    s = _prep(value)
    if s is None:
        return []
    s = s.strip("[]").strip()
    if not s:
        return []
    return sorted({int(token) for token in re.split(r"[,;]", s) if token.strip()})


def norm_experience_factors(value: Any) -> Any:
    """The multi-select factor cell: blank → []; a comma/semicolon-joined
    string of factor ids (or display labels) becomes the member list."""
    if isinstance(value, (list, tuple)):
        return [_norm_display_enum(ExperienceFactor, v) for v in value]
    s = _prep(value)
    if s is None:
        return []
    return [
        _norm_display_enum(ExperienceFactor, token)
        for token in re.split(r"[,;]", s)
        if token.strip()
    ]


YesNoCell = Annotated[Optional[YesNo], BeforeValidator(norm_yes_no)]
PrecisionCell = Annotated[Optional[PrecisionVerdict], BeforeValidator(norm_precision)]
ColdCell = Annotated[Optional[ColdLabel], BeforeValidator(norm_cold)]
JudgeOutcomeCell = Annotated[
    Optional[JudgeOutcome], BeforeValidator(norm_judge_outcome)
]
PipelineVerdictCell = _display_enum_cell(PipelineVerdict)
IssueTypeCell = Annotated[Optional[IssueType], BeforeValidator(norm_issue_type)]
AuditStatusCell = Annotated[AuditStatus, BeforeValidator(norm_audit_status)]
ErrorSourceCell = _display_enum_cell(ErrorSource)
ErrorTypeCell = _display_enum_cell(ErrorType)
LikertCell = Annotated[
    Optional[Union[Annotated[int, Field(ge=1, le=4)], Literal["NA"]]],
    BeforeValidator(norm_likert_na),
]
Likert4Cell = Annotated[
    Optional[Annotated[int, Field(ge=1, le=4)]], BeforeValidator(norm_likert)
]
ExperienceFactorCell = _display_enum_cell(ExperienceFactor)
ExperienceFactorsCell = Annotated[
    list[ExperienceFactor], BeforeValidator(norm_experience_factors)
]
BreakingPointCell = _display_enum_cell(BreakingPoint)
BoolCell = Annotated[bool, BeforeValidator(norm_bool)]
IntCell = Annotated[int, BeforeValidator(norm_int)]
PromptBedItemKindCell = _display_enum_cell(PromptBedItemKind)
PromptBedVerdictCell = Annotated[
    Optional[PromptBedVerdict], BeforeValidator(norm_prompt_bed_verdict)
]
OptIntCell = Annotated[Optional[int], BeforeValidator(norm_opt_int)]
OptFloatCell = Annotated[Optional[float], BeforeValidator(norm_opt_float)]


def norm_audio_issue_rating(value: Any) -> Any:
    """Blank → None; otherwise accept the stable NA/0/1/2/3 CSV values."""
    if isinstance(value, AudioIssueRating):
        return value
    s = _prep(value)
    if s is None:
        return None
    normalized = str(s).upper()
    aliases = {
        "N/A": "NA",
        "NO OPPORTUNITY": "NA",
        "NO_ISSUE": "0",
        "MINOR": "1",
        "CLEAR": "2",
        "SEVERE": "3",
    }
    return AudioIssueRating(aliases.get(normalized, normalized))


AudioIssueRatingCell = Annotated[
    Optional[AudioIssueRating], BeforeValidator(norm_audio_issue_rating)
]
RubricAnswerCell = _display_enum_cell(RubricAnswer)
RubricAnnotationSectionCell = _display_enum_cell(RubricAnnotationSection)
SemanticSeverityCell = _display_enum_cell(SemanticSeverity)
RubricPhaseCell = _display_enum_cell(RubricPhase)
IntListCell = Annotated[list[int], BeforeValidator(norm_int_list)]


Likert5Cell = Annotated[
    Optional[Annotated[int, Field(ge=1, le=5)]], BeforeValidator(norm_likert)
]


# ============================================================================
# Row models (generated-headers-first: the model IS the contract)
# ============================================================================


class TranslationReviewRow(SheetRow):
    """One translation-review row: native annotator vs the pipeline's verdict.

    The pipeline verdict rides two typed HIDDEN columns (``pipeline_verdict`` +
    ``pipeline_detail``) instead of the old ``"pipeline: ..."`` comment-prefix
    hack, so the comparison joins on data, not string parsing.
    """

    field: Annotated[str, Col("field")] = ""
    english_original: Annotated[str, Col("english_original", width=50)] = ""
    translation: Annotated[str, Col("translation", width=50)] = ""
    meaning_ok: Annotated[
        YesNoCell, Col("meaning_ok (Y/N)", options=YesNo.options())
    ] = None
    natural_ok: Annotated[
        YesNoCell, Col("natural_ok (Y/N)", options=YesNo.options())
    ] = None
    issue_type: Annotated[
        IssueTypeCell,
        Col("issue_type", options=[m.value for m in IssueType]),
    ] = None
    suggested_rewrite: Annotated[str, Col("suggested_rewrite (only if N)")] = ""
    comments: Annotated[str, Col("comments")] = ""
    pipeline_verdict: Annotated[
        PipelineVerdictCell,
        Col(
            "pipeline_verdict",
            hidden=True,
            description="The pipeline's own verdict (None for verdict-less "
            "--from-existing exports).",
        ),
    ] = None
    pipeline_detail: Annotated[
        str,
        Col(
            "pipeline_detail",
            hidden=True,
            description="Attempt count / rejection reasons behind the verdict.",
        ),
    ] = ""
    row_id: Annotated[int, Col("row_id", hidden=True)] = 0
    task_id: Annotated[str, Col("task_id", hidden=True)] = ""

    @property
    def native_ok(self) -> Optional[bool]:
        """The native's overall verdict; None when the row is unlabeled."""
        if self.meaning_ok is None and self.natural_ok is None:
            return None
        return self.meaning_ok is YesNo.YES and self.natural_ok is not YesNo.NO

    @property
    def pipeline_ok(self) -> Optional[bool]:
        """The pipeline's verdict as a bool; None for verdict-less exports."""
        if self.pipeline_verdict is None:
            return None
        return self.pipeline_verdict is PipelineVerdict.VERIFIED


class JudgePrecisionRow(SheetRow):
    """One judge FAIL to adjudicate: annotator marks it real or a false alarm.

    Reading-order columns first (what we check, the exact phrase, why, context,
    verdict + notes); trace ids trail and are hidden when built into a sheet.
    The annotator never sees the rubric — spelling it out would bias them
    toward confirming the judge.
    """

    what_checked: Annotated[str, Col("What we're checking", width=32)] = ""
    quote: Annotated[str, Col("Exact phrase flagged", width=32)] = ""
    why_flagged: Annotated[str, Col("Why the judge flagged it", width=60)] = ""
    transcript: Annotated[str, Col("Full transcript (context)", width=80)] = ""
    verdict: Annotated[
        PrecisionCell, Col("Real violation?", options=PrecisionVerdict.options())
    ] = None
    notes: Annotated[str, Col("Notes")] = ""
    factor_id: Annotated[str, Col("factor_id", hidden=True)] = ""
    sim_id: Annotated[str, Col("sim_id", hidden=True)] = ""
    task_id: Annotated[str, Col("task_id", hidden=True)] = ""
    language: Annotated[str, Col("language", hidden=True)] = ""


class CommunicateJudgeRow(SheetRow):
    """One sampled communicate_info criterion: native verdict vs the LLM judge."""

    criterion: Annotated[str, Col("criterion", width=40)] = ""
    agent_turn_excerpt: Annotated[str, Col("agent_turn_excerpt", width=60)] = ""
    judge_verdict: Annotated[
        YesNoCell, Col("judge_verdict (Y/N)", options=YesNo.options())
    ] = None
    native_verdict: Annotated[
        YesNoCell, Col("native_verdict (Y/N)", options=YesNo.options())
    ] = None
    comments: Annotated[str, Col("comments")] = ""
    llm_judged: Annotated[
        YesNoCell,
        Col(
            "llm_judged",
            hidden=True,
            description="Whether the justification carries the LLM-judge "
            "prefix (vs a deterministic substring match).",
        ),
    ] = None
    task_id: Annotated[str, Col("task_id", hidden=True)] = ""
    sim_id: Annotated[str, Col("sim_id", hidden=True)] = ""


class ColdSidecarRow(SheetRow):
    """The HIDDEN judge verdict for one (sim, factor) — the cold sheet's answer
    key, kept out of the annotator's sight for the confusion matrix."""

    sim_id: Annotated[str, Col("sim_id")] = ""
    factor_id: Annotated[str, Col("factor_id")] = ""
    language: Annotated[str, Col("language")] = ""
    judge_outcome: Annotated[
        JudgeOutcomeCell,
        Col(
            "judge_outcome",
            description="Stored judge outcome; None (blank cell) when the sim "
            "carries no verdict for this factor.",
        ),
    ] = None
    judge_reasoning: Annotated[str, Col("judge_reasoning")] = ""
    judge_quote: Annotated[str, Col("judge_quote")] = ""


class FactorKeyRow(SheetRow):
    """One language's rubric for one judge factor (the cold sheet's column key).

    ``non_native_ref`` is reference-only: it names the expected error, which
    would prime blind labels — the workbook rubric tab must never show it.
    """

    language: Annotated[str, Col("language")] = ""
    factor_id: Annotated[str, Col("factor_id")] = ""
    category: Annotated[str, Col("category")] = ""
    nuance: Annotated[str, Col("nuance", width=40)] = ""
    native_does: Annotated[str, Col("native_does", width=70)] = ""
    non_native_ref: Annotated[
        str,
        Col(
            "common non-native error (reference)",
            description="Priming risk — hidden from blind annotators.",
        ),
    ] = ""


class AuditNuanceRow(SheetRow):
    """One candidate nuance for the Nativeness Audit sheet (AI-suggested;
    the annotator verifies via ``status`` and adds a real example)."""

    category: Annotated[str, Col("Category")] = ""
    nuance: Annotated[str, Col("Nuance (short title)", width=40)] = ""
    ai_does: Annotated[str, Col("How the AI caller sounds non-native", width=60)] = ""
    native_does: Annotated[str, Col("What a real speaker does", width=60)] = ""
    severity: Annotated[
        int, Col("Severity", options=["1", "2", "3"]), Field(ge=1, le=3)
    ] = 2
    status: Annotated[AuditStatusCell, Col("Status", options=AuditStatus.options())] = (
        AuditStatus.AUTO
    )


# ============================================================================
# HTML-packet rows (the browser CSV contract, one model per FormType)
# ============================================================================
#
# Packet pages inject ``model.headers()`` as ``PACKET_CONFIG.csv_headers``;
# the in-browser JS builds its export/import from that list, so the annotator
# CSV can never drift from these models. Ingest is header-set match (client
# JS cannot write manifests — the ONE sanctioned manifest-less path).

# The "unable to evaluate" sentinel for rubric dimension cells. Distinct from
# blank (= not yet rated) and from any 1-4 score; consumers computing means
# MUST exclude it (round-1 annotators had no such option and entered 1,
# silently dragging dimension means down).
NA_SCORE = "NA"

LIKERT_OPTIONS = ["1", "2", "3", "4", NA_SCORE]

# Rubric dimension ids, in rubric (= column) order. A guard test asserts these
# stay in lockstep with packets/assets/voice_user_simulator_rubric.json.
REALISM_DIMENSION_IDS = [
    "voice_prosody_quality",
    "audio_environment_realism",
    "turn_taking_naturalness",
    "backchannel_naturalness",
    "interruption_behavior",
    "behavioral_plausibility",
    "speech_accuracy",
    "phrasing_naturalness",
]


# The three packet row models share their column blocks. Pydantic collects
# fields base-most-first, so a row model lists its blocks in REVERSE reading
# order (the LAST base contributes the FIRST columns); the header-order guard
# tests in test_packets.py pin the resulting layout.


class _PacketTraceFields(SheetRow):
    """The call-identity block every packet row starts with."""

    batch: Annotated[str, Col("batch")] = ""
    rater: Annotated[str, Col("rater")] = ""
    task_id: Annotated[str, Col("task_id")] = ""
    simulation_id: Annotated[str, Col("simulation_id")] = ""
    trial: Annotated[IntCell, Col("trial")] = 0


class _ErrorFindingFields(BaseModel):
    """The error-analysis block: the first critical error's source/type plus
    free-text notes (blank = no error found)."""

    error_source: Annotated[
        ErrorSourceCell, Col("error_source", options=[m.value for m in ErrorSource])
    ] = None
    error_type: Annotated[
        ErrorTypeCell, Col("error_type", options=[m.value for m in ErrorType])
    ] = None
    notes: Annotated[str, Col("notes")] = ""


class _RealismScoreFields(BaseModel):
    """The realism block: 1-4 Likert scores (or ``NA`` = unable to evaluate)
    per rubric dimension plus per-dimension notes (required for scores 1-2
    and NA) and free-text comments.

    Field order (scores in ``REALISM_DIMENSION_IDS`` order, then the notes) is
    the column contract; the guard test ties it to the rubric JSON.
    """

    voice_prosody_quality: Annotated[
        LikertCell, Col("voice_prosody_quality", options=LIKERT_OPTIONS)
    ] = None
    audio_environment_realism: Annotated[
        LikertCell, Col("audio_environment_realism", options=LIKERT_OPTIONS)
    ] = None
    turn_taking_naturalness: Annotated[
        LikertCell, Col("turn_taking_naturalness", options=LIKERT_OPTIONS)
    ] = None
    backchannel_naturalness: Annotated[
        LikertCell, Col("backchannel_naturalness", options=LIKERT_OPTIONS)
    ] = None
    interruption_behavior: Annotated[
        LikertCell, Col("interruption_behavior", options=LIKERT_OPTIONS)
    ] = None
    behavioral_plausibility: Annotated[
        LikertCell, Col("behavioral_plausibility", options=LIKERT_OPTIONS)
    ] = None
    speech_accuracy: Annotated[
        LikertCell, Col("speech_accuracy", options=LIKERT_OPTIONS)
    ] = None
    phrasing_naturalness: Annotated[
        LikertCell, Col("phrasing_naturalness", options=LIKERT_OPTIONS)
    ] = None
    voice_prosody_quality_notes: Annotated[str, Col("voice_prosody_quality_notes")] = ""
    audio_environment_realism_notes: Annotated[
        str, Col("audio_environment_realism_notes")
    ] = ""
    turn_taking_naturalness_notes: Annotated[
        str, Col("turn_taking_naturalness_notes")
    ] = ""
    backchannel_naturalness_notes: Annotated[
        str, Col("backchannel_naturalness_notes")
    ] = ""
    interruption_behavior_notes: Annotated[str, Col("interruption_behavior_notes")] = ""
    behavioral_plausibility_notes: Annotated[
        str, Col("behavioral_plausibility_notes")
    ] = ""
    speech_accuracy_notes: Annotated[str, Col("speech_accuracy_notes")] = ""
    phrasing_naturalness_notes: Annotated[str, Col("phrasing_naturalness_notes")] = ""
    free_text_comments: Annotated[str, Col("free_text_comments")] = ""


class _CallerExperienceFields(BaseModel):
    """The caller-experience block: would a real caller have tolerated this
    call? ``caller_experience`` anchors: 1 = wouldn't call again (a real
    caller would hang up), 2 = harsh and frustrating, 3 = acceptable,
    4 = pleasant. For scores 1-2 the annotator locates the damage
    (``experience_breaking_point``: the overall experience, or one specific
    tick), multi-selects every contributing ``experience_factors`` entry, and
    names exactly ONE ``primary_factor`` among them. Elaboration-required
    factors (sim breakage / other) demand ``experience_notes`` at any score.
    """

    caller_experience: Annotated[
        Likert4Cell, Col("caller_experience", options=["1", "2", "3", "4"])
    ] = None
    experience_breaking_point: Annotated[
        BreakingPointCell,
        Col(
            "experience_breaking_point",
            options=[m.value for m in BreakingPoint],
        ),
    ] = None
    experience_breaking_tick: Annotated[OptIntCell, Col("experience_breaking_tick")] = (
        None
    )
    experience_factors: Annotated[
        ExperienceFactorsCell,
        Col(
            "experience_factors",
            description="Comma-joined factor ids, every contributing factor.",
        ),
    ] = []
    primary_factor: Annotated[
        ExperienceFactorCell,
        Col("primary_factor", options=[m.value for m in ExperienceFactor]),
    ] = None
    experience_notes: Annotated[str, Col("experience_notes")] = ""

    @model_validator(mode="after")
    def _primary_among_selected(self) -> "_CallerExperienceFields":
        if self.primary_factor is not None and self.primary_factor not in (
            self.experience_factors or []
        ):
            raise ValueError(
                f"primary_factor {self.primary_factor.value!r} is not among "
                "the selected experience_factors"
            )
        return self


class _PacketStateFields(BaseModel):
    """The client-state block every packet row ends with."""

    completed: Annotated[BoolCell, Col("completed")] = False
    created_at: Annotated[str, Col("created_at")] = ""


class ErrorAnalysisRow(_PacketStateFields, _ErrorFindingFields, _PacketTraceFields):
    """One reviewed call in an ``error_analysis`` packet: the first critical
    error's source/type plus free-text notes (blank = no error found)."""


class RealismRow(_PacketStateFields, _RealismScoreFields, _PacketTraceFields):
    """One reviewed call in a ``user_realism`` packet: 1-4 Likert scores (or
    ``NA`` = unable to evaluate) per rubric dimension plus per-dimension notes
    (required for scores 1-2 and NA)."""


class VoiceReviewRow(
    _PacketStateFields,
    _CallerExperienceFields,
    _RealismScoreFields,
    _ErrorFindingFields,
    _PacketTraceFields,
):
    """One reviewed call in a ``voice_review`` packet: the combined contract —
    the error-finding block, the realism Likert block, then the
    caller-experience block, filled in one listening pass."""


class PromptBedRow(SheetRow):
    """One reviewed item of a ``prompt_bed_review`` packet CSV (long format).

    ``row_kind`` discriminates: PROMPT rows carry a rendered runtime prompt
    section (``item_id`` like ``system_prompt_english``); BED rows carry one
    background-bed audio file (``locale`` + ``bed_type``); VOICE rows carry
    one persona's rendered voice sample (``item_id`` like
    ``voice_es_camila_es_v1``). The packet JS
    exports EVERY item — unreviewed rows ride with a blank verdict, so
    coverage gaps are visible at ingest, never silently absent.
    """

    row_kind: Annotated[
        PromptBedItemKindCell, Col("row_kind", options=PromptBedItemKind.options())
    ] = None
    batch: Annotated[str, Col("batch")] = ""
    rater: Annotated[str, Col("rater")] = ""
    language: Annotated[
        str,
        Col(
            "language",
            description="The packet's language (prompt sections are rendered "
            "for it; bed rows cover every locale regardless).",
        ),
    ] = ""
    item_id: Annotated[
        str,
        Col(
            "item_id",
            description="Stable item id: a prompt section id (e.g. "
            "'system_prompt_english'), a bed id (e.g. 'bed_es_ES_outdoor'), "
            "or a voice id (e.g. 'voice_es_camila_es_v1').",
        ),
    ] = ""
    locale: Annotated[
        str,
        Col(
            "locale",
            description="Bed rows only: the bed's locale dir (e.g. 'es_ES'), "
            "or 'shared' for the office bed.",
        ),
    ] = ""
    bed_type: Annotated[
        str,
        Col(
            "bed_type",
            description="Bed rows only: outdoor / indoor_tv / shared_office.",
        ),
    ] = ""
    verdict: Annotated[
        PromptBedVerdictCell,
        Col(
            "verdict",
            options=PromptBedVerdict.options(),
            description="OK, or issue (shown as 'Issues found' on prompt "
            "sections and 'Sounds off' on beds and voice samples).",
        ),
    ] = None
    notes: Annotated[str, Col("notes")] = ""
    completed: Annotated[BoolCell, Col("completed")] = False
    created_at: Annotated[str, Col("created_at")] = ""


AUDIO_ISSUE_RATING_OPTIONS = [member.value for member in AudioIssueRating]


class AudioQualityRow(SheetRow):
    """One agent-only clip's spoken-fidelity and intonation annotation.

    Each fixed dimension records issue severity plus optional evidence. ``NA``
    means the clip offered no opportunity to judge that dimension; blank means
    the annotator has not rated it yet. Adjudication packets use this same row
    contract and additionally record the disposition of every LLM finding.
    """

    batch: Annotated[str, Col("batch")] = ""
    rater: Annotated[str, Col("rater")] = ""
    clip_id: Annotated[str, Col("clip_id")] = ""
    listened_fully: Annotated[BoolCell, Col("listened_fully")] = False
    mispronunciation: Annotated[
        AudioIssueRatingCell,
        Col("mispronunciation", options=AUDIO_ISSUE_RATING_OPTIONS),
    ] = None
    mispronunciation_notes: Annotated[str, Col("mispronunciation_notes")] = ""
    word_substitution: Annotated[
        AudioIssueRatingCell,
        Col("word_substitution", options=AUDIO_ISSUE_RATING_OPTIONS),
    ] = None
    word_substitution_notes: Annotated[str, Col("word_substitution_notes")] = ""
    missing_word: Annotated[
        AudioIssueRatingCell,
        Col("missing_word", options=AUDIO_ISSUE_RATING_OPTIONS),
    ] = None
    missing_word_notes: Annotated[str, Col("missing_word_notes")] = ""
    extra_or_hallucinated_word: Annotated[
        AudioIssueRatingCell,
        Col("extra_or_hallucinated_word", options=AUDIO_ISSUE_RATING_OPTIONS),
    ] = None
    extra_or_hallucinated_word_notes: Annotated[
        str, Col("extra_or_hallucinated_word_notes")
    ] = ""
    number_date_currency: Annotated[
        AudioIssueRatingCell,
        Col("number_date_currency", options=AUDIO_ISSUE_RATING_OPTIONS),
    ] = None
    number_date_currency_notes: Annotated[str, Col("number_date_currency_notes")] = ""
    email_url_code: Annotated[
        AudioIssueRatingCell,
        Col("email_url_code", options=AUDIO_ISSUE_RATING_OPTIONS),
    ] = None
    email_url_code_notes: Annotated[str, Col("email_url_code_notes")] = ""
    punctuation_or_formatting: Annotated[
        AudioIssueRatingCell,
        Col("punctuation_or_formatting", options=AUDIO_ISSUE_RATING_OPTIONS),
    ] = None
    punctuation_or_formatting_notes: Annotated[
        str, Col("punctuation_or_formatting_notes")
    ] = ""
    acronym_brand_name: Annotated[
        AudioIssueRatingCell,
        Col("acronym_brand_name", options=AUDIO_ISSUE_RATING_OPTIONS),
    ] = None
    acronym_brand_name_notes: Annotated[str, Col("acronym_brand_name_notes")] = ""
    clipped_or_garbled: Annotated[
        AudioIssueRatingCell,
        Col("clipped_or_garbled", options=AUDIO_ISSUE_RATING_OPTIONS),
    ] = None
    clipped_or_garbled_notes: Annotated[str, Col("clipped_or_garbled_notes")] = ""
    other: Annotated[
        AudioIssueRatingCell,
        Col("other", options=AUDIO_ISSUE_RATING_OPTIONS),
    ] = None
    other_notes: Annotated[str, Col("other_notes")] = ""
    intonation: Annotated[
        AudioIssueRatingCell,
        Col("intonation", options=AUDIO_ISSUE_RATING_OPTIONS),
    ] = None
    intonation_notes: Annotated[str, Col("intonation_notes")] = ""
    overall_notes: Annotated[str, Col("overall_notes")] = ""
    previously_completed: Annotated[BoolCell, Col("previously_completed")] = False
    adjudication_required: Annotated[BoolCell, Col("adjudication_required")] = False
    llm_total_findings: Annotated[int, Col("llm_total_findings")] = 0
    llm_reviewed_findings: Annotated[int, Col("llm_reviewed_findings")] = 0
    llm_approved_findings: Annotated[int, Col("llm_approved_findings")] = 0
    llm_decisions_json: Annotated[str, Col("llm_decisions_json")] = ""
    human_annotation_verdict: Annotated[str, Col("human_annotation_verdict")] = ""
    adjudication_notes: Annotated[str, Col("adjudication_notes")] = ""
    completed: Annotated[BoolCell, Col("completed")] = False
    created_at: Annotated[str, Col("created_at")] = ""

    @model_validator(mode="after")
    def _completed_rows_are_complete(self) -> "AudioQualityRow":
        if not self.completed:
            return self
        if not self.listened_fully:
            raise ValueError("completed audio-quality row was not listened to fully")
        fixed = (
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
            "intonation",
        )
        missing = [name for name in fixed if getattr(self, name) is None]
        if missing:
            raise ValueError(
                "completed audio-quality row has unrated dimensions: "
                + ", ".join(missing)
            )
        if self.adjudication_required:
            if self.llm_reviewed_findings != self.llm_total_findings:
                raise ValueError(
                    "completed adjudication row has unreviewed LLM findings"
                )
            if self.human_annotation_verdict not in {"correct", "needs_correction"}:
                raise ValueError(
                    "completed adjudication row needs a human annotation verdict"
                )
        return self


class RubricAnnotationRow(SheetRow):
    """One overall score or factor answer from the combined rubric packet.

    ``severity`` and ``selected_turn_indices`` ride YES (violation) answers
    only: severity is the two-level minor/major scale of the semantic-source
    questions, and the turn indices are the agent turns an utterance-level
    question's violation was pinned to. ``phase`` is blank on blind-packet
    rows; two-phase (judge-visible) packets export every question twice, as a
    ``pre_reveal`` and a ``post_reveal`` row. Returns exported before these
    columns existed still ingest (see ``LEGACY_ABSENT_LABELS``): the absent
    cells default — on an old utterance-level YES row the turns are simply
    unrecorded, never fabricated.
    """

    batch: Annotated[str, Col("batch")] = ""
    rater: Annotated[str, Col("rater")] = ""
    clip_id: Annotated[str, Col("clip_id")] = ""
    language: Annotated[str, Col("language")] = ""
    agent_gender: Annotated[
        str, Col("agent_gender", options=["male", "female", "unknown"])
    ] = "unknown"
    caller_gender: Annotated[
        str, Col("caller_gender", options=["male", "female", "unknown"])
    ] = "unknown"
    section: Annotated[
        RubricAnnotationSectionCell,
        Col("section", options=RubricAnnotationSection.options()),
    ] = None
    factor_id: Annotated[str, Col("factor_id")] = ""
    question: Annotated[str, Col("question")] = ""
    answer: Annotated[
        RubricAnswerCell, Col("answer", options=RubricAnswer.options())
    ] = None
    severity: Annotated[
        SemanticSeverityCell,
        Col(
            "severity",
            options=SemanticSeverity.options(),
            description="Semantic-source questions only: minor/major on a "
            "YES answer (the semantic judges' two-level scale).",
        ),
    ] = None
    evidence: Annotated[str, Col("evidence")] = ""
    selected_turn_indices: Annotated[
        IntListCell,
        Col(
            "selected_turn_indices",
            description="Utterance-level questions only: the agent-turn "
            "indices a YES answer was pinned to, comma-joined.",
        ),
    ] = []
    is_custom: Annotated[BoolCell, Col("is_custom")] = False
    evidence_mode: Annotated[
        str,
        Col("evidence_mode", options=["full_conversation", "agent_only"]),
    ] = "full_conversation"
    audio_consulted: Annotated[BoolCell, Col("audio_consulted")] = False
    notes: Annotated[str, Col("notes")] = ""
    phase: Annotated[
        RubricPhaseCell,
        Col(
            "phase",
            options=RubricPhase.options(),
            description="Blank for blind packets; pre_reveal/post_reveal "
            "layers of a two-phase judge-visible packet.",
        ),
    ] = None
    completed: Annotated[BoolCell, Col("completed")] = False
    created_at: Annotated[str, Col("created_at")] = ""

    @model_validator(mode="after")
    def _completed_rows_have_valid_answers(self) -> "RubricAnnotationRow":
        if self.answer is not RubricAnswer.YES:
            if self.severity is not None:
                raise ValueError("rubric severity requires a yes (violation) answer")
            if self.selected_turn_indices:
                raise ValueError(
                    "selected agent turns require a yes (violation) answer"
                )
        if any(index < 0 for index in self.selected_turn_indices):
            raise ValueError("selected agent-turn indices must be non-negative")
        if not self.completed:
            return self
        if self.answer is None:
            raise ValueError("completed rubric row has no answer")
        if not self.question.strip():
            raise ValueError("completed rubric row has no question")
        binary = {
            RubricAnswer.YES,
            RubricAnswer.NO,
            RubricAnswer.NO_OPPORTUNITY,
        }
        audio = {
            RubricAnswer.AUDIO_NO_ISSUE,
            RubricAnswer.AUDIO_MINOR,
            RubricAnswer.AUDIO_CLEAR,
            RubricAnswer.AUDIO_SEVERE,
            RubricAnswer.AUDIO_NO_OPPORTUNITY,
        }
        overall = {
            RubricAnswer.AUDIO_MINOR,
            RubricAnswer.AUDIO_CLEAR,
            RubricAnswer.AUDIO_SEVERE,
            RubricAnswer.SCORE_FOUR,
        }
        if self.factor_id in OVERALL_SCORE_FACTOR_IDS:
            expected_section = (
                RubricAnnotationSection.INTERACTION
                if self.factor_id == OVERALL_INTERACTION_QUALITY_FACTOR_ID
                else RubricAnnotationSection.NATIVENESS
            )
            if self.section != expected_section:
                raise ValueError(
                    f"{self.factor_id} belongs to the {expected_section.value} section"
                )
            if self.answer not in overall:
                raise ValueError("completed overall-score row needs a 1-4 answer")
        elif self.section == RubricAnnotationSection.AUDIO:
            if self.answer not in audio:
                raise ValueError("completed audio rubric row has a non-audio answer")
            if not self.audio_consulted:
                raise ValueError("completed audio rubric row has not consulted audio")
        elif self.answer not in binary:
            raise ValueError("completed text rubric row has a non-binary answer")
        return self


class NativenessAnnotationLabel(DisplayEnum):
    """Blind human label for one nativeness factor and evaluation unit."""

    NOT_APPLICABLE = ("not_applicable", "Not applicable")
    PASS = ("pass", "Pass")
    VIOLATION = ("violation", "Violation")


class NativenessAdjudicationDecision(DisplayEnum):
    """Human disposition of one machine-proposed nativeness violation."""

    CONFIRMED = ("confirmed", "Confirmed")
    REJECTED = ("rejected", "Rejected")


NativenessAnnotationLabelCell = Annotated[
    Optional[NativenessAnnotationLabel],
    BeforeValidator(lambda value: _norm_display_enum(NativenessAnnotationLabel, value)),
]
NativenessAdjudicationDecisionCell = Annotated[
    Optional[NativenessAdjudicationDecision],
    BeforeValidator(
        lambda value: _norm_display_enum(NativenessAdjudicationDecision, value)
    ),
]


class NativenessHumanLabelRow(SheetRow):
    """One canonical blind-annotation or adjudication nativeness label.

    Utterance violations pin the exact rendered agent turn. A factor-level
    pass/no-opportunity label deliberately has no turn. Adjudication uses the
    same contract and fills ``candidate_id`` plus ``adjudication_decision``.
    """

    batch: Annotated[str, Col("batch")] = ""
    rater: Annotated[str, Col("rater")] = ""
    clip_id: Annotated[str, Col("clip_id")] = ""
    simulation_id: Annotated[str, Col("simulation_id")] = ""
    task_id: Annotated[str, Col("task_id")] = ""
    language: Annotated[str, Col("language")] = ""
    factor_id: Annotated[str, Col("factor_id")] = ""
    evaluation_level: Annotated[
        Literal["call", "utterance"], Col("evaluation_level")
    ] = "call"
    annotation_label: Annotated[
        NativenessAnnotationLabelCell,
        Col("annotation_label", options=NativenessAnnotationLabel.options()),
    ] = None
    candidate_id: Annotated[str, Col("candidate_id")] = ""
    adjudication_decision: Annotated[
        NativenessAdjudicationDecisionCell,
        Col(
            "adjudication_decision",
            options=NativenessAdjudicationDecision.options(),
        ),
    ] = None
    agent_turn_index: Annotated[OptIntCell, Col("agent_turn_index")] = None
    agent_turn_id: Annotated[str, Col("agent_turn_id")] = ""
    agent_text: Annotated[str, Col("agent_text")] = ""
    preceding_customer_text: Annotated[str, Col("preceding_customer_text")] = ""
    severity: Annotated[OptIntCell, Col("severity")] = None
    note: Annotated[str, Col("note")] = ""
    provenance_json: Annotated[str, Col("provenance_json")] = ""
    phase: Annotated[
        RubricPhaseCell,
        Col(
            "phase",
            options=RubricPhase.options(),
            description="Blank for blind packets; pre_reveal/post_reveal "
            "layers of a two-phase judge-visible packet.",
        ),
    ] = None
    completed: Annotated[BoolCell, Col("completed")] = False
    created_at: Annotated[str, Col("created_at")] = ""

    @model_validator(mode="after")
    def _valid_nativeness_label(self) -> "NativenessHumanLabelRow":
        if not self.completed:
            return self
        if self.severity is not None and not 1 <= self.severity <= 4:
            raise ValueError("nativeness severity must be 1..4")
        if bool(self.annotation_label) == bool(self.adjudication_decision):
            raise ValueError(
                "completed nativeness row needs exactly one annotation label "
                "or adjudication decision"
            )
        if not self.provenance_json:
            raise ValueError("completed nativeness row needs provenance")
        try:
            provenance = json.loads(self.provenance_json)
        except json.JSONDecodeError as exc:
            raise ValueError("nativeness provenance_json is not valid JSON") from exc
        if not isinstance(provenance, dict) or not provenance.get("packet_batch_id"):
            raise ValueError("nativeness provenance needs packet_batch_id")
        if self.annotation_label is NativenessAnnotationLabel.VIOLATION:
            if self.evaluation_level == "call" and self.severity is None:
                raise ValueError("call-level nativeness violation needs severity")
            if self.evaluation_level == "utterance" and self.severity is not None:
                raise ValueError("utterance-level nativeness violation has no severity")
            if self.evaluation_level == "utterance" and (
                self.agent_turn_index is None
                or not self.agent_turn_id
                or not self.agent_text
            ):
                raise ValueError(
                    "utterance violation needs the exact selected agent turn"
                )
        if (
            self.annotation_label
            in {
                NativenessAnnotationLabel.PASS,
                NativenessAnnotationLabel.NOT_APPLICABLE,
            }
            and self.agent_turn_index is not None
        ):
            raise ValueError("factor-level pass/not-applicable cannot pin a turn")
        if (
            self.annotation_label
            in {
                NativenessAnnotationLabel.PASS,
                NativenessAnnotationLabel.NOT_APPLICABLE,
            }
            and self.severity is not None
        ):
            raise ValueError("pass/not-applicable nativeness label has no severity")
        if self.adjudication_decision is not None and not self.candidate_id:
            raise ValueError("adjudication decision needs a candidate id")
        if (
            self.adjudication_decision is NativenessAdjudicationDecision.CONFIRMED
            and self.evaluation_level == "call"
            and self.severity is None
        ):
            raise ValueError("confirmed call-level adjudication needs severity")
        if (
            self.adjudication_decision is not None
            and self.evaluation_level == "utterance"
        ):
            if self.severity is not None:
                raise ValueError("utterance-level adjudication has no severity")
            if (
                self.agent_turn_index is None
                or not self.agent_turn_id
                or not self.agent_text
            ):
                raise ValueError(
                    "utterance-level adjudication needs the exact candidate turn"
                )
        if (
            self.adjudication_decision is NativenessAdjudicationDecision.REJECTED
            and self.severity is not None
        ):
            raise ValueError("rejected adjudication has no severity")
        return self


#: The judge families the calibration adjudication pages minted decisions
#: for, kept as a plain closed vocabulary so returned decisions CSVs stay
#: ingestible.
CALIBRATION_JUDGE_FAMILIES: tuple[str, ...] = (
    "delivery",
    "nativeness",
    "quality",
    "semantic",
)


class CalibrationDecisionRow(SheetRow):
    """One owner adjudication of a judge-positive calibration candidate.

    The decisions CSV of the judge-calibration adjudicate page. This row
    EXTENDS ``NativenessHumanLabelRow``'s candidate_id + adjudication_decision
    contract to all four calibrated judge families rather than inventing a
    parallel one: the decision vocabulary (``NativenessAdjudicationDecision``),
    the candidate-id requirement, the exact-flagged-turn pinning rules for
    utterance-level candidates, and the provenance requirement are identical.
    The minimal extension — because delivery / quality / semantic positives
    are not nativeness rows and cannot reuse that model as-is — is the
    ``judge`` family column and the stored ``judge_verdict`` being decided.
    Severity is deliberately absent: the calibration wave measures verdict
    correctness only; severity capture stays owned by the nativeness
    adjudication packets.
    """

    batch: Annotated[str, Col("batch")] = ""
    rater: Annotated[str, Col("rater")] = ""
    clip_id: Annotated[str, Col("clip_id")] = ""
    simulation_id: Annotated[str, Col("simulation_id")] = ""
    task_id: Annotated[str, Col("task_id")] = ""
    language: Annotated[str, Col("language")] = ""
    judge: Annotated[str, Col("judge", options=list(CALIBRATION_JUDGE_FAMILIES))] = ""
    factor_id: Annotated[str, Col("factor_id")] = ""
    evaluation_level: Annotated[
        Literal["call", "utterance"], Col("evaluation_level")
    ] = "call"
    judge_verdict: Annotated[str, Col("judge_verdict")] = ""
    candidate_id: Annotated[str, Col("candidate_id")] = ""
    adjudication_decision: Annotated[
        NativenessAdjudicationDecisionCell,
        Col(
            "adjudication_decision",
            options=NativenessAdjudicationDecision.options(),
        ),
    ] = None
    reclassified_factor: Annotated[
        str,
        Col("reclassified_factor"),
        Field(
            description="Set only on a REJECTED decision: the "
            "``judge:factor_id`` this finding actually belongs to, from the "
            "instrument's closed catalog — records that the finding is real "
            "but mis-filed (cross-factor attribution). Empty otherwise."
        ),
    ] = ""
    agent_turn_index: Annotated[OptIntCell, Col("agent_turn_index")] = None
    agent_turn_id: Annotated[str, Col("agent_turn_id")] = ""
    agent_text: Annotated[str, Col("agent_text")] = ""
    preceding_customer_text: Annotated[str, Col("preceding_customer_text")] = ""
    note: Annotated[str, Col("note")] = ""
    provenance_json: Annotated[str, Col("provenance_json")] = ""
    completed: Annotated[BoolCell, Col("completed")] = False
    created_at: Annotated[str, Col("created_at")] = ""

    @model_validator(mode="after")
    def _valid_calibration_decision(self) -> "CalibrationDecisionRow":
        if self.judge and self.judge not in CALIBRATION_JUDGE_FAMILIES:
            raise ValueError(
                f"unknown calibration judge family {self.judge!r} (expected "
                f"one of {', '.join(CALIBRATION_JUDGE_FAMILIES)})"
            )
        if not self.completed:
            return self
        if self.adjudication_decision is None:
            raise ValueError("completed calibration decision needs a decision")
        if not self.candidate_id:
            raise ValueError("calibration decision needs a candidate id")
        if not self.judge:
            raise ValueError("completed calibration decision needs a judge family")
        if self.judge_verdict != "fail":
            raise ValueError(
                "calibration decisions adjudicate judge POSITIVES only — "
                f"judge_verdict must be 'fail', got {self.judge_verdict!r}"
            )
        if self.evaluation_level == "utterance" and (
            self.agent_turn_index is None
            or not self.agent_turn_id
            or not self.agent_text
        ):
            raise ValueError(
                "utterance-level calibration decision needs the exact candidate turn"
            )
        if self.evaluation_level == "call" and self.agent_turn_index is not None:
            raise ValueError("call-level calibration decision cannot pin a turn")
        if not self.provenance_json:
            raise ValueError("completed calibration decision needs provenance")
        try:
            provenance = json.loads(self.provenance_json)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "calibration decision provenance_json is not valid JSON"
            ) from exc
        if not isinstance(provenance, dict) or not provenance.get("packet_batch_id"):
            raise ValueError("calibration decision provenance needs packet_batch_id")
        return self


# ============================================================================
# The WIDE cold sheet (the one non-static shape)
# ============================================================================

COLD_TRANSCRIPT_LABEL = "Full transcript (context)"
COLD_NOTES_LABEL = "Notes"
COLD_TRACE_LABELS = ("sim_id", "task_id", "language")
COLD_META_LABELS = frozenset(
    {COLD_TRANSCRIPT_LABEL, COLD_NOTES_LABEL, *COLD_TRACE_LABELS}
)
COLD_TRANSCRIPT_WIDTH = 80


class ColdSheetRow(BaseModel):
    """One call in the wide blind sheet: the annotator reads the transcript
    once and labels every factor column in that row."""

    transcript: Annotated[str, Field(description="Agent transcript, truncated.")]
    notes: Annotated[str, Field(description="Annotator free-text notes.")] = ""
    sim_id: str
    task_id: str = ""
    language: str = ""
    labels: Annotated[
        dict[str, ColdCell],
        Field(
            default_factory=dict,
            description="Annotator label per factor column, keyed by the "
            "factor's readable nuance header.",
        ),
    ]


class ColdSheet(BaseModel):
    """The wide cold sheet + its factor key, with melt/unmelt in one place."""

    rows: Annotated[list[ColdSheetRow], Field(default_factory=list)]
    factors: Annotated[
        list[FactorKeyRow],
        Field(default_factory=list, description="Column key: nuance ↔ factor."),
    ]

    def factor_headers(self) -> list[str]:
        """Factor column headers in first-seen row order (wide rows may differ
        across languages), falling back to the factor key for empty sheets."""
        headers: list[str] = []
        for row in self.rows:
            for header in row.labels:
                if header not in headers:
                    headers.append(header)
        for key in self.factors:
            header = key.nuance or key.factor_id
            if header not in headers:
                headers.append(header)
        return headers

    def headers(self) -> list[str]:
        """Transcript leads; factor dropdowns next; notes; trace ids trail."""
        return [
            COLD_TRANSCRIPT_LABEL,
            *self.factor_headers(),
            COLD_NOTES_LABEL,
            *COLD_TRACE_LABELS,
        ]

    def header_to_factor(self) -> dict[tuple[str, str], str]:
        """{(language, nuance-header) -> factor_id} so readable column headers
        join back to the sidecar."""
        return {
            (key.language, key.nuance): key.factor_id
            for key in self.factors
            if key.nuance
        }

    def to_cells(self) -> list[dict[str, str]]:
        """Render the wide grid (header-keyed cells, blank annotator cells)."""
        headers = self.headers()
        out: list[dict[str, str]] = []
        for row in self.rows:
            cells = dict.fromkeys(headers, "")
            cells[COLD_TRANSCRIPT_LABEL] = row.transcript
            cells[COLD_NOTES_LABEL] = row.notes
            cells["sim_id"] = row.sim_id
            cells["task_id"] = row.task_id
            cells["language"] = row.language
            for header, label in row.labels.items():
                cells[header] = _render_cell(label)
            out.append(cells)
        return out

    @classmethod
    def from_cells(
        cls, rows: list[Mapping[str, str]], factors: list[FactorKeyRow]
    ) -> "ColdSheet":
        """Unmelt a filled wide CSV: every non-meta column is a factor column."""
        sheet_rows: list[ColdSheetRow] = []
        for raw in rows:
            labels = {
                header: value
                for header, value in raw.items()
                if header not in COLD_META_LABELS
            }
            sheet_rows.append(
                ColdSheetRow(
                    transcript=raw.get(COLD_TRANSCRIPT_LABEL, ""),
                    notes=raw.get(COLD_NOTES_LABEL, ""),
                    sim_id=raw.get("sim_id", ""),
                    task_id=raw.get("task_id", ""),
                    language=raw.get("language", ""),
                    labels=labels,
                )
            )
        return cls(rows=sheet_rows, factors=factors)

    def melt(self) -> Iterator[tuple[str, str, str, Optional[ColdLabel]]]:
        """Yield (language, sim_id, factor_id, label) per labeled cell.

        Readable headers map back to factor ids via the factor key; a header
        with no mapping falls back to itself.
        """
        mapping = self.header_to_factor()
        for row in self.rows:
            for header, label in row.labels.items():
                factor_id = mapping.get((row.language, header), header)
                yield row.language, row.sim_id, factor_id, label


# ============================================================================
# Sheet-family registry: artifact kind -> {role -> row model}
# ============================================================================

# Sentinel for the wide cold grid (dynamic factor columns; not a SheetRow).
COLD_WIDE = "cold_wide"

SheetShape = Union[type[SheetRow], str]

SHEET_FAMILIES: dict[str, dict[str, SheetShape]] = {
    "translation_review": {"sheet": TranslationReviewRow},
    "communicate_judge": {"sheet": CommunicateJudgeRow},
    "nativeness_precision": {"sheet": JudgePrecisionRow},
    "nativeness_cold": {
        "sheet": COLD_WIDE,
        "judge_sidecar": ColdSidecarRow,
        "factors_key": FactorKeyRow,
        # Present only for `--mode both` families (precision derived from the
        # same judge pass); ingest dispatches on (kind, role).
        "precision": JudgePrecisionRow,
    },
    # HTML annotation packets: the "sheet" is the browser-exported CSV (built
    # by the packet JS from PACKET_CONFIG.csv_headers = model.headers()).
    "packet_error_analysis": {"sheet": ErrorAnalysisRow},
    "packet_user_realism": {"sheet": RealismRow},
    "packet_voice_review": {"sheet": VoiceReviewRow},
    "packet_prompt_bed_review": {"sheet": PromptBedRow},
    "packet_audio_quality": {"sheet": AudioQualityRow},
    "packet_rubric": {"sheet": RubricAnnotationRow},
    "packet_nativeness_labels": {"sheet": NativenessHumanLabelRow},
    "packet_judge_calibration_decisions": {"sheet": CalibrationDecisionRow},
}
