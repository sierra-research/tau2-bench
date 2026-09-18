"""Render model and language summaries for the tau-multilingual manuscript.

Task completion is read from the frozen trial-0 benchmark summaries;
Interaction and Generation are read from the typed frozen analysis artifact.
Turn-taking latency is read from the frozen reproduction artifact.
The primary heatmaps use higher-is-better scores, while the metric breakdowns
report lower-is-better failure rates.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from subprocess import run
from typing import Annotated, Sequence

from pydantic import BaseModel, ConfigDict, Field
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

from experiments.tau_multilingual.experience_without_fluency import (
    UtteranceExperienceArtifact,
)
from experiments.tau_multilingual.task_success_significance import (
    TaskSuccessAnalysisArtifact,
)

FIGURE_DIR = Path(__file__).parent
MODEL_OUTPUT = FIGURE_DIR / "model_summary.pdf"
HEATMAP_OUTPUT = FIGURE_DIR / "language_system_heatmaps.pdf"
METRIC_OUTPUT = FIGURE_DIR / "model_metric_heatmap.pdf"
REPO_ROOT = Path(__file__).resolve().parents[4]
TASK_SUCCESS_PATH = (
    REPO_ROOT
    / "data/analysis/tau_multilingual_task_success_significance_2026-09-18.json"
)
ANALYSIS_PATH = (
    REPO_ROOT
    / "data/analysis/tau_multilingual_experience_without_fluency_2026-09-18.json"
)
LATENCY_PATH = REPO_ROOT / "papers/tau-multilingual/reproduction/experience.json"


class RenderConfig(BaseModel):
    """Validated input and output paths for one figure render."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_success_artifact: Annotated[
        Path,
        Field(description="Typed task-success analysis artifact."),
    ] = TASK_SUCCESS_PATH
    experience_artifact: Annotated[
        Path,
        Field(description="Typed interaction/generation analysis artifact."),
    ] = ANALYSIS_PATH
    latency_artifact: Annotated[
        Path,
        Field(description="Typed turn-taking latency analysis artifact."),
    ] = LATENCY_PATH
    output_dir: Annotated[
        Path,
        Field(description="Directory receiving the three fixed figure filenames."),
    ] = FIGURE_DIR


FONT = "FigureRoman"
BOLD_FONT = "FigureRoman-Bold"
INK = colors.HexColor("#20252B")
GRID = colors.HexColor("#D9DEE5")
HEAT_COLOR = colors.HexColor("#197052")


def _tex_font_path(filename: str) -> Path:
    """Resolve a font shipped with the TeX installation used for the paper."""
    result = run(
        ["kpsewhich", filename],
        check=True,
        capture_output=True,
        text=True,
    )
    path = Path(result.stdout.strip())
    if not path.is_file():
        raise FileNotFoundError(f"TeX font not found: {filename}")
    return path


def _register_type1_font(name: str, stem: str) -> None:
    """Register an embedded Nimbus Roman face under a stable figure name."""
    face = pdfmetrics.EmbeddedType1Face(
        str(_tex_font_path(f"{stem}.afm")),
        str(_tex_font_path(f"{stem}.pfb")),
    )
    pdfmetrics.registerTypeFace(face)
    pdfmetrics.registerFont(pdfmetrics.Font(name, face.name, "WinAnsiEncoding"))


_FONTS_REGISTERED = False


def _ensure_fonts_registered() -> None:
    """Register publication fonts lazily so input validation needs no TeX install."""
    global _FONTS_REGISTERED
    if _FONTS_REGISTERED:
        return
    _register_type1_font(FONT, "utmr8a")
    _register_type1_font(BOLD_FONT, "utmb8a")
    _FONTS_REGISTERED = True


# Use the ICASSP minimum figure-label size without visually overpowering the
# surrounding 9.5-point body text.
MODEL_WIDTH = 244
MODEL_HEIGHT = 124
MODEL_FONT_SIZE = 9.2
HEATMAP_FONT_SIZE = 9.2
HEATMAP_HEIGHT = 117
HEATMAP_TITLE_Y = 108
HEATMAP_LANGUAGE_Y = 93
HEATMAP_TOP = 89
HEATMAP_CELL_HEIGHT = 14

LANGUAGES = ("English", "Spanish", "Portuguese", "Hindi", "Korean", "Mandarin")
LOCALIZED_LANGUAGES = LANGUAGES[1:]
LANGUAGE_LABELS = {
    "English": "EN",
    "Spanish": "ES",
    "Portuguese": "PT",
    "Hindi": "HI",
    "Korean": "KO",
    "Mandarin": "ZH",
}
SYSTEMS = (
    "OpenAI minimal",
    "OpenAI xhigh",
    "Gemini minimal",
    "Gemini high",
    "xAI",
)
SYSTEM_LABELS = {
    "OpenAI minimal": "GPT min.",
    "OpenAI xhigh": "GPT xhigh",
    "Gemini minimal": "Gem min.",
    "Gemini high": "Gem high",
    "xAI": "Grok",
}
TASK_LANGUAGE_KEYS = {
    "English": "en",
    "Spanish": "es",
    "Portuguese": "pt",
    "Hindi": "hi",
    "Korean": "ko",
    "Mandarin": "zh",
}
TASK_SYSTEM_KEYS = {
    "OpenAI minimal": "openai_minimal",
    "OpenAI xhigh": "openai_xhigh",
    "Gemini minimal": "gemini_minimal",
    "Gemini high": "gemini_high",
    "xAI": "xai_provider_default",
}


def _load_experience_artifact(path: Path) -> UtteranceExperienceArtifact:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"experience artifact not found: {resolved}")
    return UtteranceExperienceArtifact.model_validate_json(resolved.read_text())


class _LatencyCell(BaseModel):
    """One provider's validated turn-taking latency."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    latency_seconds: Annotated[
        float,
        Field(ge=0, description="Mean turn-taking latency in seconds."),
    ]


class _LatencyCohort(BaseModel):
    """Provider cells in the frozen complete cohort."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    provider: Annotated[
        dict[str, _LatencyCell],
        Field(description="Turn-taking latency keyed by displayed system."),
    ]


class _LatencyArtifact(BaseModel):
    """Minimal typed projection of the frozen latency artifact."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    descriptive_complete_cohort: Annotated[
        _LatencyCohort,
        Field(description="Complete-cohort provider summaries."),
    ]


def _load_latency_scores(path: Path) -> dict[str, float]:
    """Load the exact displayed systems from the typed latency artifact."""
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"latency artifact not found: {resolved}")
    artifact = _LatencyArtifact.model_validate_json(resolved.read_text())
    provider = artifact.descriptive_complete_cohort.provider
    _require_exact_keys(set(provider), set(SYSTEMS), label="latency provider")
    return {system: provider[system].latency_seconds for system in SYSTEMS}


def _require_exact_keys(actual: set[str], expected: set[str], *, label: str) -> None:
    if actual == expected:
        return
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    raise ValueError(
        f"{label} keys drifted: missing={missing}, unexpected={unexpected}"
    )


def _percentage(value: float, *, label: str) -> float:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} must be a finite proportion in [0, 1]: {value}")
    return 100.0 * value


def _load_task_scores(
    path: Path,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Load task-success scores from the frozen typed artifact."""
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"task-success artifact not found: {resolved}")
    artifact = TaskSuccessAnalysisArtifact.model_validate_json(resolved.read_text())
    descriptive = artifact.descriptive_trial_0
    expected_languages = set(TASK_LANGUAGE_KEYS.values())
    expected_systems = set(TASK_SYSTEM_KEYS.values())
    _require_exact_keys(
        set(descriptive.voice_language_system),
        expected_languages,
        label="task-success language",
    )
    _require_exact_keys(
        set(descriptive.voice_system_mean),
        expected_systems,
        label="task-success provider",
    )

    language_system: dict[str, dict[str, float]] = {}
    for language, language_key in TASK_LANGUAGE_KEYS.items():
        raw_scores = descriptive.voice_language_system[language_key]
        _require_exact_keys(
            set(raw_scores),
            expected_systems,
            label=f"task-success {language_key} system",
        )
        language_system[language] = {
            system: _percentage(
                raw_scores[system_key],
                label=f"task-success {language_key}/{system_key}",
            )
            for system, system_key in TASK_SYSTEM_KEYS.items()
        }
    provider = {
        system: _percentage(
            descriptive.voice_system_mean[system_key],
            label=f"task-success provider/{system_key}",
        )
        for system, system_key in TASK_SYSTEM_KEYS.items()
    }
    return language_system, provider


def _draw_mean_marker(
    pdf: canvas.Canvas,
    x: float,
    y: float,
    marker: str,
    *,
    size: float = 4.5,
) -> None:
    """Draw a monochrome marker for a provider-level mean."""
    pdf.setStrokeColor(INK)
    pdf.setFillColor(INK if marker == "circle" else colors.white)
    pdf.setLineWidth(0.8)
    radius = size / 2
    if marker == "circle":
        pdf.circle(x, y, radius, fill=1, stroke=1)
    elif marker == "square":
        pdf.rect(x - radius, y - radius, size, size, fill=1, stroke=1)
    elif marker == "diamond":
        path = pdf.beginPath()
        path.moveTo(x, y + radius)
        path.lineTo(x + radius, y)
        path.lineTo(x, y - radius)
        path.lineTo(x - radius, y)
        path.close()
        pdf.drawPath(path, fill=1, stroke=1)
    else:
        raise ValueError(f"Unknown marker: {marker}")


def _write_marker_legend(
    pdf: canvas.Canvas,
    entries: list[tuple[str, str]],
    *,
    y: float,
    width: float,
) -> None:
    size = MODEL_FONT_SIZE
    swatch = 7
    gap = 10
    item_widths = [swatch + 3 + stringWidth(label, FONT, size) for label, _ in entries]
    x = (width - sum(item_widths) - gap * (len(entries) - 1)) / 2
    for (label, marker), item_width in zip(entries, item_widths, strict=True):
        _draw_mean_marker(pdf, x + swatch / 2, y + 3, marker)
        pdf.setFillColor(INK)
        pdf.setFont(FONT, size)
        pdf.drawString(x + swatch + 3, y, label)
        x += item_width + gap


def _draw_patterned_rect(
    pdf: canvas.Canvas,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    pattern: str,
) -> None:
    """Draw a monochrome rectangle with a print-safe system pattern."""
    pdf.setStrokeColor(INK)
    pdf.setLineWidth(0.4)
    if pattern == "solid":
        pdf.setFillColor(INK)
        pdf.rect(x, y, width, height, fill=1, stroke=1)
        return

    pdf.setFillColor(colors.white)
    pdf.rect(x, y, width, height, fill=1, stroke=0)
    pdf.saveState()
    clip = pdf.beginPath()
    clip.rect(x, y, width, height)
    pdf.clipPath(clip, stroke=0, fill=0)
    pdf.setStrokeColor(INK)
    pdf.setLineWidth(0.45)
    spacing = 3.0

    if pattern in {"horizontal", "crosshatch"}:
        line_y = y + spacing / 2
        while line_y < y + height:
            pdf.line(x, line_y, x + width, line_y)
            line_y += spacing
    if pattern == "vertical":
        line_x = x + spacing / 2
        while line_x < x + width:
            pdf.line(line_x, y, line_x, y + height)
            line_x += spacing
    if pattern in {"diagonal", "crosshatch"}:
        line_x = x - height
        while line_x < x + width:
            pdf.line(line_x, y, line_x + height, y + height)
            line_x += spacing
    pdf.restoreState()
    pdf.setStrokeColor(INK)
    pdf.setLineWidth(0.4)
    pdf.rect(x, y, width, height, fill=0, stroke=1)


def render_model_summary(
    artifact: UtteranceExperienceArtifact,
    *,
    pass_at_1: dict[str, dict[str, float]],
    pass_provider: dict[str, float],
    output: Path = MODEL_OUTPUT,
) -> None:
    """Render the transposed model overview with across-language ranges."""
    _ensure_fonts_registered()
    width = MODEL_WIDTH
    height = MODEL_HEIGHT
    pdf = canvas.Canvas(
        str(output),
        pagesize=(width, height),
        pageCompression=1,
        initialFontName=FONT,
        invariant=1,
    )
    pdf.setTitle("Model-level task, interaction, and generation summary")
    metric_markers = {
        "Task completion": "circle",
        "Interaction": "square",
        "Generation": "diamond",
    }
    _write_marker_legend(
        pdf,
        [(label, marker) for label, marker in metric_markers.items()],
        y=height - 11,
        width=width,
    )

    left = 27
    right = 4
    bottom = 25
    top = height - 28
    plot_height = top - bottom
    metric_gap = 9.5
    interaction_by_language = {
        language: {
            system: 100.0
            - artifact.descriptive_complete_cohort.language_system[language][
                system
            ].interaction_failure
            for system in SYSTEMS
        }
        for language in LANGUAGES
    }
    experience_by_language = {
        language: {
            system: artifact.descriptive_complete_cohort.language_system[language][
                system
            ].experience
            for system in SYSTEMS
        }
        for language in LOCALIZED_LANGUAGES
    }
    metrics = (
        ("Task completion", pass_provider, pass_at_1),
        (
            "Interaction",
            {
                system: 100.0
                - artifact.descriptive_complete_cohort.provider[
                    system
                ].interaction_failure
                for system in SYSTEMS
            },
            interaction_by_language,
        ),
        (
            "Generation",
            {
                system: artifact.descriptive_complete_cohort.provider[system].experience
                for system in SYSTEMS
            },
            experience_by_language,
        ),
    )
    group_width = (width - left - right) / len(SYSTEMS)

    for tick in (0, 25, 50, 75, 100):
        y = bottom + plot_height * tick / 100
        pdf.setStrokeColor(GRID)
        pdf.setLineWidth(0.35)
        pdf.line(left, y, width - right, y)
        pdf.setFillColor(colors.HexColor("#5D6670"))
        pdf.setFont(FONT, MODEL_FONT_SIZE)
        pdf.drawRightString(left - 3, y - 3, str(tick))

    for system_index, system in enumerate(SYSTEMS):
        center = left + (system_index + 0.5) * group_width
        for metric_index, (metric_label, values, language_values) in enumerate(metrics):
            value = values[system]
            language_range = [cells[system] for cells in language_values.values()]
            range_min = bottom + plot_height * min(language_range) / 100
            range_max = bottom + plot_height * max(language_range) / 100
            mean_y = bottom + plot_height * value / 100
            center_x = center + (metric_index - 1) * metric_gap
            pdf.setStrokeColor(INK)
            pdf.setLineWidth(0.65)
            pdf.line(center_x, range_min, center_x, range_max)
            pdf.line(center_x - 2.5, range_min, center_x + 2.5, range_min)
            pdf.line(center_x - 2.5, range_max, center_x + 2.5, range_max)
            _draw_mean_marker(pdf, center_x, mean_y, metric_markers[metric_label])

            pdf.setFillColor(INK)
            pdf.setFont(BOLD_FONT, MODEL_FONT_SIZE)
            pdf.drawCentredString(
                center_x + (metric_index - 1) * 1.5,
                range_max + 2,
                f"{value:.0f}",
            )

        pdf.setFillColor(INK)
        label_parts = SYSTEM_LABELS[system].split(" ", maxsplit=1)
        pdf.setFont(BOLD_FONT, MODEL_FONT_SIZE)
        pdf.drawCentredString(center, 11.5, label_parts[0])
        if len(label_parts) == 2:
            pdf.setFont(FONT, MODEL_FONT_SIZE)
            pdf.drawCentredString(center, 2.5, label_parts[1])

    pdf.save()


def _heat_fill(value: float) -> colors.Color:
    intensity = 0.12 + 0.88 * value / 100
    return colors.Color(
        1.0 - (1.0 - HEAT_COLOR.red) * intensity,
        1.0 - (1.0 - HEAT_COLOR.green) * intensity,
        1.0 - (1.0 - HEAT_COLOR.blue) * intensity,
    )


def render_model_metric_heatmap(
    artifact: UtteranceExperienceArtifact,
    *,
    pass_provider: dict[str, float],
    latency_provider: dict[str, float],
    output: Path = METRIC_OUTPUT,
) -> None:
    """Render the metric breakdown at its final, one-column print size."""
    _ensure_fonts_registered()
    provider = artifact.descriptive_complete_cohort.provider
    interaction_component = {
        component: {
            system: getattr(provider[system].interaction_components, component)
            for system in SYSTEMS
        }
        for component in (
            "nonresponse",
            "interruption",
            "selectivity",
            "monologue",
            "tool_use",
        )
    }
    model_rows = (
        (
            "Task failure",
            "Call",
            {system: 100.0 - pass_provider[system] for system in SYSTEMS},
            False,
        ),
        ("Non-response", "Call", interaction_component["nonresponse"], False),
        ("Interruption", "Call", interaction_component["interruption"], False),
        (
            "Selectivity error",
            "Call",
            interaction_component["selectivity"],
            False,
        ),
        ("Monologue", "Call", interaction_component["monologue"], False),
        ("Tool misuse", "Call", interaction_component["tool_use"], False),
        (
            "Naturalness failure",
            "Utt.",
            {system: provider[system].fluency_failure for system in SYSTEMS},
            False,
        ),
        (
            "Speech-fidelity failure",
            "Utt.",
            {system: provider[system].speech_fidelity_failure for system in SYSTEMS},
            False,
        ),
        ("Turn-taking latency", "Turn", latency_provider, True),
    )

    width = 244
    model_x = 110
    model_cell_width = (width - model_x) / len(SYSTEMS)
    row_heights = [14] * len(model_rows)
    height = sum(row_heights) + 27
    top = height - 25
    pdf = canvas.Canvas(
        str(output),
        pagesize=(width, height),
        pageCompression=1,
        initialFontName=FONT,
        invariant=1,
    )
    pdf.setTitle("Main-metric breakdown by system")

    pdf.setFillColor(INK)
    pdf.setFont(BOLD_FONT, HEATMAP_FONT_SIZE)
    header_lines = (
        ("GPT", "min."),
        ("GPT", "xhigh"),
        ("Gem", "min."),
        ("Gem", "high"),
        ("Grok",),
    )
    for system_index, lines in enumerate(header_lines):
        center_x = model_x + (system_index + 0.5) * model_cell_width
        if len(lines) == 1:
            pdf.drawCentredString(center_x, height - 15, lines[0])
        else:
            for line_index, line in enumerate(lines):
                pdf.drawCentredString(center_x, height - 9 - line_index * 11, line)
    grid_right_x = model_x + len(SYSTEMS) * model_cell_width

    pdf.drawString(1, height - 15, "Metric")

    pdf.saveState()
    pdf.setStrokeColor(GRID)
    pdf.setLineWidth(0.35)
    row_y = top
    for cell_height in row_heights:
        row_y -= cell_height
        for column_index in range(len(SYSTEMS)):
            pdf.rect(
                model_x + column_index * model_cell_width,
                row_y,
                model_cell_width,
                cell_height,
                fill=0,
                stroke=1,
            )
    pdf.restoreState()

    row_y = top
    for (label, level, values, is_latency), cell_height in zip(
        model_rows, row_heights, strict=True
    ):
        row_y -= cell_height
        text_y = row_y + (cell_height - HEATMAP_FONT_SIZE) / 2 + 1.7
        pdf.setFillColor(INK)
        pdf.setFont(FONT, HEATMAP_FONT_SIZE)
        label_x = 1
        pdf.drawString(label_x, text_y, label)
        label_x += stringWidth(label, FONT, HEATMAP_FONT_SIZE)
        # A lowered baseline makes the level a subscript without reducing the
        # type below the ICASSP figure-label minimum. Units stay on the baseline.
        level_label = {"Call": "c", "Utt.": "u", "Turn": "t"}[level]
        level_x = label_x + 0.9
        pdf.drawString(level_x, text_y - 2.2, level_label)
        unit_x = level_x + stringWidth(level_label, FONT, HEATMAP_FONT_SIZE) + 2
        unit = "(s)" if is_latency else "(%)"
        pdf.drawString(unit_x, text_y, unit)
        if unit_x + stringWidth(unit, FONT, HEATMAP_FONT_SIZE) > model_x - 2:
            raise ValueError(f"Metric label exceeds the one-column layout: {label}")

        row_min = min(values.values())
        for system_index, system in enumerate(SYSTEMS):
            value = values[system]
            cell_x = model_x + system_index * model_cell_width
            pdf.setFillColor(INK)
            pdf.setFont(
                BOLD_FONT if value == row_min else FONT,
                HEATMAP_FONT_SIZE,
            )
            display = f"{value:.2f}" if is_latency else f"{value:.0f}"
            pdf.drawCentredString(cell_x + model_cell_width / 2, text_y, display)

    pdf.saveState()
    pdf.setStrokeColor(colors.HexColor("#AEB6BF"))
    pdf.setLineWidth(0.6)
    for boundary_after in (1, 6, 8):
        boundary_y = top - sum(row_heights[:boundary_after])
        pdf.line(1, boundary_y, grid_right_x, boundary_y)
    pdf.restoreState()

    pdf.save()


def _draw_heatmap_panel(
    pdf: canvas.Canvas,
    *,
    x: float,
    width: float,
    title: str,
    languages: tuple[str, ...],
    values: dict[str, dict[str, float]],
) -> None:
    top = HEATMAP_TOP
    cell_height = HEATMAP_CELL_HEIGHT
    cell_width = width / len(languages)
    panel_values = [
        values[language][system] for language in languages for system in SYSTEMS
    ]
    panel_min = min(panel_values)
    panel_max = max(panel_values)
    pdf.setFillColor(INK)
    pdf.setFont(BOLD_FONT, HEATMAP_FONT_SIZE)
    pdf.drawCentredString(x + width / 2, HEATMAP_TITLE_Y, title)
    pdf.setFont(BOLD_FONT, HEATMAP_FONT_SIZE)
    for language_index, language in enumerate(languages):
        pdf.drawCentredString(
            x + (language_index + 0.5) * cell_width,
            HEATMAP_LANGUAGE_Y,
            LANGUAGE_LABELS[language],
        )

    for system_index, system in enumerate(SYSTEMS):
        row_y = top - (system_index + 1) * cell_height
        for language_index, language in enumerate(languages):
            value = values[language][system]
            cell_x = x + language_index * cell_width
            color_value = (
                50.0
                if panel_max == panel_min
                else 100.0 * (value - panel_min) / (panel_max - panel_min)
            )
            pdf.setFillColor(_heat_fill(color_value))
            pdf.rect(cell_x, row_y, cell_width, cell_height, fill=1, stroke=0)
            pdf.setFillColor(colors.white if color_value >= 61 else INK)
            pdf.setFont(BOLD_FONT, HEATMAP_FONT_SIZE)
            pdf.drawCentredString(
                cell_x + cell_width / 2,
                row_y + 4.7,
                f"{value:.0f}",
            )
            best = max(values[language][candidate] for candidate in SYSTEMS)
            if value == best:
                pdf.setStrokeColor(INK)
                pdf.setLineWidth(0.7)
                pdf.rect(
                    cell_x + 0.35,
                    row_y + 0.35,
                    cell_width - 0.7,
                    cell_height - 0.7,
                    fill=0,
                    stroke=1,
                )

    overall_y = top - (len(SYSTEMS) + 1) * cell_height
    for language_index, language in enumerate(languages):
        value = sum(values[language].values()) / len(SYSTEMS)
        cell_x = x + language_index * cell_width
        pdf.setFillColor(colors.HexColor("#F0F2F4"))
        pdf.rect(cell_x, overall_y, cell_width, cell_height, fill=1, stroke=0)
        pdf.setFillColor(INK)
        pdf.setFont(BOLD_FONT, HEATMAP_FONT_SIZE)
        pdf.drawCentredString(
            cell_x + cell_width / 2,
            overall_y + 4.7,
            f"{value:.0f}",
        )
    pdf.setStrokeColor(INK)
    pdf.setLineWidth(0.65)
    pdf.line(x, overall_y + cell_height, x + width, overall_y + cell_height)


def render_heatmaps(
    artifact: UtteranceExperienceArtifact,
    *,
    pass_at_1: dict[str, dict[str, float]],
    output: Path = HEATMAP_OUTPUT,
) -> None:
    """Render three higher-is-better system-language heatmaps."""
    _ensure_fonts_registered()
    width = 500
    height = HEATMAP_HEIGHT
    pdf = canvas.Canvas(
        str(output),
        pagesize=(width, height),
        pageCompression=1,
        initialFontName=FONT,
        invariant=1,
    )
    pdf.setTitle("System-language task, interaction, and generation heatmaps")
    pdf.setFillColor(INK)
    pdf.setFont(FONT, HEATMAP_FONT_SIZE)
    for system_index, system in enumerate(SYSTEMS):
        y = HEATMAP_TOP - (system_index + 0.67) * HEATMAP_CELL_HEIGHT
        pdf.drawRightString(50, y, SYSTEM_LABELS[system])
    pdf.setFont(BOLD_FONT, HEATMAP_FONT_SIZE)
    overall_y = HEATMAP_TOP - (len(SYSTEMS) + 0.67) * HEATMAP_CELL_HEIGHT
    pdf.drawRightString(50, overall_y, "Overall")

    interaction = {
        language: {
            system: 100.0
            - artifact.descriptive_complete_cohort.language_system[language][
                system
            ].interaction_failure
            for system in SYSTEMS
        }
        for language in LANGUAGES
    }
    experience = {
        language: {
            system: artifact.descriptive_complete_cohort.language_system[language][
                system
            ].experience
            for system in SYSTEMS
        }
        for language in LOCALIZED_LANGUAGES
    }

    _draw_heatmap_panel(
        pdf,
        x=56,
        width=139,
        title="(a) Task completion",
        languages=LANGUAGES,
        values=pass_at_1,
    )
    _draw_heatmap_panel(
        pdf,
        x=207,
        width=139,
        title="(b) Interaction",
        languages=LANGUAGES,
        values=interaction,
    )
    _draw_heatmap_panel(
        pdf,
        x=358,
        width=137,
        title="(c) Generation",
        languages=LOCALIZED_LANGUAGES,
        values=experience,
    )
    pdf.save()


def _parse_args(argv: Sequence[str] | None = None) -> RenderConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-success-artifact",
        type=Path,
        default=TASK_SUCCESS_PATH,
        help=f"Typed task-success JSON artifact (default: {TASK_SUCCESS_PATH}).",
    )
    parser.add_argument(
        "--experience-artifact",
        type=Path,
        default=ANALYSIS_PATH,
        help=f"Typed experience JSON artifact (default: {ANALYSIS_PATH}).",
    )
    parser.add_argument(
        "--latency-artifact",
        type=Path,
        default=LATENCY_PATH,
        help=f"Typed latency JSON artifact (default: {LATENCY_PATH}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FIGURE_DIR,
        help=f"Output directory for all three PDFs (default: {FIGURE_DIR}).",
    )
    return RenderConfig.model_validate(vars(parser.parse_args(argv)))


def main(argv: Sequence[str] | None = None) -> None:
    """Write the publication-ready vector figures."""
    config = _parse_args(argv)
    artifact = _load_experience_artifact(config.experience_artifact)
    pass_at_1, pass_provider = _load_task_scores(config.task_success_artifact)
    latency_provider = _load_latency_scores(config.latency_artifact)
    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    render_model_summary(
        artifact,
        pass_at_1=pass_at_1,
        pass_provider=pass_provider,
        output=output_dir / MODEL_OUTPUT.name,
    )
    render_model_metric_heatmap(
        artifact,
        pass_provider=pass_provider,
        latency_provider=latency_provider,
        output=output_dir / METRIC_OUTPUT.name,
    )
    render_heatmaps(
        artifact,
        pass_at_1=pass_at_1,
        output=output_dir / HEATMAP_OUTPUT.name,
    )


if __name__ == "__main__":
    main()
