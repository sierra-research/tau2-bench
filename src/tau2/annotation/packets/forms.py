# Copyright Sierra
"""Packet form types: ONE parameterized form instead of three near-copies.

``FormType`` selects which sections a packet page renders (findings and/or the
realism Likert dimensions) and which ``SheetRow`` model defines its browser
CSV contract (``PACKET_CONFIG.csv_headers`` = ``model.headers()``). The
``voice_review`` form composes the other two sections, exactly as before.

The rubric (dimension names, descriptions, 1-4 score anchors) is calibrated
annotator-facing material living in ``assets/voice_user_simulator_rubric.json``;
it is validated into typed models here. Guidelines HTML is embedded verbatim
into the page's modal.

Browser-exported CSVs are the ONE sanctioned manifest-less ingest path (client
JS cannot write manifests): ``match_browser_kind`` dispatches on the exact
header set of the registered row models — every ``FORM_ROW_MODELS`` contract
plus the page-less prompt+bed review packet — and rows are ``model_validate``d
per row.
"""

import csv
import re
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Optional

from pydantic import BaseModel, Field

from tau2.annotation.artifacts import CSV_ENCODING, ArtifactKind, dearmor_cell
from tau2.annotation.models import (
    EXPERIENCE_ELABORATE_FACTORS,
    REALISM_DIMENSION_IDS,
    AudioQualityRow,
    BreakingPoint,
    CalibrationDecisionRow,
    ErrorAnalysisRow,
    ExperienceFactor,
    NativenessHumanLabelRow,
    PromptBedRow,
    RealismRow,
    RubricAnnotationRow,
    SheetRow,
    VoiceReviewRow,
)

ASSETS_DIR = Path(__file__).parent / "assets"
RUBRIC_PATH = ASSETS_DIR / "voice_user_simulator_rubric.json"
ERROR_GUIDELINES_PATH = ASSETS_DIR / "ANNOTATION_GUIDELINES.html"
REALISM_GUIDELINES_PATH = ASSETS_DIR / "REALISM_GUIDELINES.html"


class FormType(str, Enum):
    """The packet form types (what the annotator fills per call)."""

    ERROR_ANALYSIS = "error_analysis"
    USER_REALISM = "user_realism"
    VOICE_REVIEW = "voice_review"


FORM_ROW_MODELS: dict[FormType, type[SheetRow]] = {
    FormType.ERROR_ANALYSIS: ErrorAnalysisRow,
    FormType.USER_REALISM: RealismRow,
    FormType.VOICE_REVIEW: VoiceReviewRow,
}

FORM_LABELS: dict[FormType, str] = {
    FormType.ERROR_ANALYSIS: "Error Analysis",
    FormType.USER_REALISM: "User Simulator Realism Evaluation",
    FormType.VOICE_REVIEW: "Voice Call Review",
}


def packet_kind_for(form: FormType) -> ArtifactKind:
    """The artifact-manifest kind for a packet of this form."""
    return f"packet_{form.value}"  # type: ignore[return-value]


# The prompt+bed review packet has no FormType: it is a single-page packet
# (rendered prompts + bed audio table), not a per-conversation form, so none
# of the FORM_SPECS/sim-page machinery applies. Its browser CSV contract is
# registered here alongside the form-backed ones.
PROMPT_BED_KIND: ArtifactKind = "packet_prompt_bed_review"

# Agent-only pronunciation/prosody packet. Like prompt+bed it has its own
# single-page UI rather than a generic per-simulation FormType.
AUDIO_QUALITY_KIND: ArtifactKind = "packet_audio_quality"

# Combined interaction, nativeness, and audio-quality packet.
RUBRIC_PACKET_KIND: ArtifactKind = "packet_rubric"

# Browser-return kind for the shared nativeness annotation/adjudication row.
NATIVENESS_LABEL_KIND: ArtifactKind = "packet_nativeness_labels"

# Owner decisions exported from the judge-calibration adjudication page (one
# row per judge-positive candidate; see CalibrationDecisionRow).
CALIBRATION_DECISIONS_KIND: ArtifactKind = "packet_judge_calibration_decisions"

# Every browser-exported packet CSV contract, by artifact kind. THE dispatch
# table for manifest-less ingest.
BROWSER_CSV_MODELS: dict[ArtifactKind, type[SheetRow]] = {
    **{packet_kind_for(form): model for form, model in FORM_ROW_MODELS.items()},
    PROMPT_BED_KIND: PromptBedRow,
    AUDIO_QUALITY_KIND: AudioQualityRow,
    RUBRIC_PACKET_KIND: RubricAnnotationRow,
    NATIVENESS_LABEL_KIND: NativenessHumanLabelRow,
    CALIBRATION_DECISIONS_KIND: CalibrationDecisionRow,
}

# Columns added to a shipped contract AFTER returns already existed in the
# field. A CSV whose header set equals the model's minus (a subset of) these
# still dispatches to the kind; ``from_cells`` fills the model defaults for
# the absent cells. POSTURE for the rubric contract: an old row's missing
# ``severity`` / ``selected_turn_indices`` / ``phase`` mean "not captured" —
# on an old utterance-level YES row the turns are unrecorded, never inferred,
# and a blank ``phase`` reads as a single-layer (blind-packet) row. Every
# other header mismatch stays a loud error.
LEGACY_ABSENT_LABELS: dict[ArtifactKind, frozenset[str]] = {
    RUBRIC_PACKET_KIND: frozenset({"severity", "selected_turn_indices", "phase"}),
    NATIVENESS_LABEL_KIND: frozenset({"phase"}),
}


# ---------------------------------------------------------------------------
# Rubric (typed, from the calibrated JSON asset)
# ---------------------------------------------------------------------------


class RubricScore(BaseModel):
    """One 1-4 anchor of a rubric dimension."""

    label: Annotated[str, Field(description="Short anchor label, e.g. 'Poor'.")]
    description: Annotated[str, Field(description="What earns this score.")]


class RubricDimension(BaseModel):
    """One realism dimension the annotator rates 1-4."""

    id: Annotated[str, Field(description="Stable dimension id (a CSV column).")]
    name: Annotated[str, Field(description="Display name.")]
    short_description: Annotated[str, Field(description="One-line card subtitle.")]
    description: Annotated[str, Field(description="Full 'More' description.")]
    scores: Annotated[
        dict[str, RubricScore],
        Field(description="Score value ('1'..'4') -> anchor."),
    ]

    def sorted_scores(self) -> list[tuple[str, RubricScore]]:
        return sorted(self.scores.items(), key=lambda kv: int(kv[0]))


class Rubric(BaseModel):
    """The voice user-simulator realism rubric (calibrated annotator text)."""

    rubric_name: str
    version: str
    dimensions: list[RubricDimension]


@lru_cache(maxsize=1)
def load_rubric() -> Rubric:
    rubric = Rubric.model_validate_json(RUBRIC_PATH.read_text())
    rubric_ids = [d.id for d in rubric.dimensions]
    if rubric_ids != REALISM_DIMENSION_IDS:
        raise ValueError(
            "rubric dimensions drifted from the RealismRow contract: "
            f"rubric {rubric_ids} vs model {REALISM_DIMENSION_IDS}"
        )
    return rubric


# ---------------------------------------------------------------------------
# Form spec (drives the single form template)
# ---------------------------------------------------------------------------


class FormSpec(BaseModel):
    """Everything the ONE form template needs to render a FormType."""

    form: FormType
    heading: Annotated[str, Field(description="The form's <h2> heading.")]
    show_findings: Annotated[
        bool, Field(description="Render the error source/type/notes section.")
    ]
    show_audio: Annotated[
        bool, Field(description="Render the realism Likert dimension cards.")
    ]
    show_experience: Annotated[
        bool,
        Field(
            description="Render the caller-experience section (CSAT-like "
            "1-4 score, breaking point, contributing factors + primary)."
        ),
    ] = False
    show_judge_review: Annotated[
        bool,
        Field(
            description="Render outcome/judge context (LLM judge review, "
            "reward details, reward cells). False is a BLIND-SAFETY guarantee: "
            "the renderer structurally never builds those sections, so no "
            "judge or outcome trace can reach the annotator."
        ),
    ] = True
    complete_hint: Annotated[
        str, Field(description="Text after the Mark as Complete checkbox.")
    ]


FORM_SPECS: dict[FormType, FormSpec] = {
    FormType.ERROR_ANALYSIS: FormSpec(
        form=FormType.ERROR_ANALYSIS,
        heading="📝 Your Annotation",
        show_findings=True,
        show_audio=False,
        complete_hint="Check this when you've finished annotating this simulation",
    ),
    FormType.USER_REALISM: FormSpec(
        form=FormType.USER_REALISM,
        heading="📝 Your Annotation — User Simulator Realism",
        show_findings=False,
        show_audio=True,
        complete_hint="Check this when you've finished rating this conversation",
    ),
    FormType.VOICE_REVIEW: FormSpec(
        form=FormType.VOICE_REVIEW,
        heading="📝 Your Annotation — Voice Call Review",
        show_findings=True,
        show_audio=True,
        show_experience=True,
        complete_hint="Check this when you've finished reviewing this call",
    ),
}


# ---------------------------------------------------------------------------
# Caller-experience section (fixed in-code annotator text, like the rubric)
# ---------------------------------------------------------------------------


class ExperienceAnchor(BaseModel):
    """One 1-4 anchor of the caller-experience scale."""

    value: str
    label: str
    description: str


# Customer service is a captive setting — callers rarely hang up even on bad
# calls — so the scale is anchored on tolerance, not satisfaction: would a
# real caller have been annoyed enough to hang up or end the call frustrated?
CALLER_EXPERIENCE_ANCHORS: list[ExperienceAnchor] = [
    ExperienceAnchor(
        value="1",
        label="Wouldn't call again",
        description="Terrible. A real caller would hang up on this call, or "
        "finish it furious and avoid this line in the future.",
    ),
    ExperienceAnchor(
        value="2",
        label="Harsh and frustrating",
        description="A real caller would grit their teeth to the end, but the "
        "call was genuinely aggravating.",
    ),
    ExperienceAnchor(
        value="3",
        label="Acceptable",
        description="No real complaints. Nothing memorable, nothing that "
        "would seriously annoy a real caller.",
    ),
    ExperienceAnchor(
        value="4",
        label="Pleasant",
        description="A genuinely good service call. A real caller would come "
        "away satisfied.",
    ),
]


def experience_config(form: FormType) -> dict:
    """The caller-experience block for PACKET_CONFIG (empty when the form
    doesn't render the section)."""
    if not FORM_SPECS[form].show_experience:
        return {}
    return {
        "factor_ids": [m.value for m in ExperienceFactor],
        "elaborate_ids": [m.value for m in EXPERIENCE_ELABORATE_FACTORS],
        "breaking_points": [m.value for m in BreakingPoint],
    }


# ---------------------------------------------------------------------------
# Guidelines (embedded into the page's modal)
# ---------------------------------------------------------------------------


def _guidelines_body(html_file: Path) -> str:
    """Extract <style> + <body> content from a full guidelines document so it
    embeds cleanly inside the modal."""
    content = html_file.read_text()
    style_match = re.search(r"<style>(.*?)</style>", content, re.DOTALL)
    style_content = style_match.group(1) if style_match else ""
    body_match = re.search(r"<body>(.*?)</body>", content, re.DOTALL)
    body_content = body_match.group(1) if body_match else content
    if style_content:
        return f"<style>{style_content}</style>\n{body_content}"
    return body_content


def guidelines_html(form: FormType) -> str:
    """The modal guidelines for a form; voice_review shows both documents."""
    if form is FormType.USER_REALISM:
        return _guidelines_body(REALISM_GUIDELINES_PATH)
    if form is FormType.VOICE_REVIEW:
        return (
            _guidelines_body(ERROR_GUIDELINES_PATH)
            + "\n<hr>\n"
            + _guidelines_body(REALISM_GUIDELINES_PATH)
        )
    return _guidelines_body(ERROR_GUIDELINES_PATH)


def guidelines_anchors(form: FormType) -> list[str]:
    """Which guideline documents this form embeds (recorded in PACKET_CONFIG)."""
    if form is FormType.USER_REALISM:
        return ["user_realism"]
    if form is FormType.VOICE_REVIEW:
        return ["error_analysis", "user_realism"]
    return ["error_analysis"]


# ---------------------------------------------------------------------------
# Browser-CSV ingest (the one sanctioned header-match dispatch)
# ---------------------------------------------------------------------------


class BrowserCsv(BaseModel):
    """A validated browser-exported packet CSV."""

    form: Annotated[
        Optional[FormType],
        Field(
            description="The packet's form type; None for page-less packets "
            "with no FormType (prompt_bed_review)."
        ),
    ]
    kind: Annotated[str, Field(description="The matching packet artifact kind.")]
    rows: list[SheetRow]

    @property
    def completed(self) -> int:
        return sum(1 for row in self.rows if getattr(row, "completed", False))


def match_browser_kind(headers: set[str]) -> Optional[ArtifactKind]:
    """Which packet kind's row model has exactly this header set (None if none).

    Exact match, with one sanctioned relaxation: a header set equal to a
    model's minus a subset of its ``LEGACY_ABSENT_LABELS`` (columns added
    after returns shipped) still matches, so old returns keep ingesting.
    """
    for kind, model in BROWSER_CSV_MODELS.items():
        expected = set(model.headers())
        if headers == expected:
            return kind
        legacy_absent = LEGACY_ABSENT_LABELS.get(kind)
        if legacy_absent and headers < expected and expected - headers <= legacy_absent:
            return kind
    return None


def ingest_browser_csv(path: Path) -> BrowserCsv:
    """Validate a browser-exported packet CSV into typed rows.

    Dispatch is exact header-set match against the registered packet row
    models; a row that fails validation is a loud error, never silently
    dropped.
    """
    path = Path(path)
    with open(path, newline="", encoding=CSV_ENCODING) as fp:
        reader = csv.DictReader(fp)
        headers = set(reader.fieldnames or [])
        raw_rows = list(reader)
    kind = match_browser_kind(headers)
    if kind is None:
        known = " / ".join(sorted(BROWSER_CSV_MODELS))
        raise ValueError(
            f"{path.name} matches no packet's CSV contract (headers do not "
            f"exactly match any of {known})"
        )
    model = BROWSER_CSV_MODELS[kind]
    # The packet JS armors formula-leading cells on export (escapeCSV); strip
    # exactly one armoring apostrophe back off so round-trips are lossless.
    rows = [
        model.from_cells(
            {h: dearmor_cell(v) if isinstance(v, str) else v for h, v in raw.items()}
        )
        for raw in raw_rows
    ]
    form = next(
        (f for f in FORM_ROW_MODELS if packet_kind_for(f) == kind),
        None,
    )
    return BrowserCsv(form=form, kind=kind, rows=rows)


def dims_config(form: FormType) -> list[dict[str, str]]:
    """`[{id, name}]` for PACKET_CONFIG (the JS iterates the audio dimensions)."""
    if not FORM_SPECS[form].show_audio:
        return []
    return [{"id": d.id, "name": d.name} for d in load_rubric().dimensions]


def csv_headers(form: FormType) -> list[str]:
    """The browser CSV headers, generated from the form's row model."""
    return FORM_ROW_MODELS[form].headers()


def rubric_json_for_provenance() -> dict[str, str]:
    rubric = load_rubric()
    return {"rubric_name": rubric.rubric_name, "rubric_version": rubric.version}
