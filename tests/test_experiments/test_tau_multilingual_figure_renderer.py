"""Focused tests for the τ-Multilingual figure-renderer input boundary."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RENDERER_PATH = (
    REPO_ROOT / "papers/tau-multilingual/v3/figures/render_language_system_metrics.py"
)


def _load_renderer(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    def hex_color(value: str) -> str:
        return value

    def string_width(*_args: object, **_kwargs: object) -> float:
        return 0.0

    reportlab = ModuleType("reportlab")
    reportlab_lib = ModuleType("reportlab.lib")
    colors = ModuleType("reportlab.lib.colors")
    colors.HexColor = hex_color
    reportlab_pdfbase = ModuleType("reportlab.pdfbase")
    pdfmetrics = ModuleType("reportlab.pdfbase.pdfmetrics")
    pdfmetrics.stringWidth = string_width
    reportlab_pdfgen = ModuleType("reportlab.pdfgen")
    canvas = ModuleType("reportlab.pdfgen.canvas")
    reportlab_lib.colors = colors
    reportlab_pdfbase.pdfmetrics = pdfmetrics
    reportlab_pdfgen.canvas = canvas
    for name, module in {
        "reportlab": reportlab,
        "reportlab.lib": reportlab_lib,
        "reportlab.lib.colors": colors,
        "reportlab.pdfbase": reportlab_pdfbase,
        "reportlab.pdfbase.pdfmetrics": pdfmetrics,
        "reportlab.pdfgen": reportlab_pdfgen,
        "reportlab.pdfgen.canvas": canvas,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location(
        "tau_multilingual_figure_renderer_test", RENDERER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def renderer(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import the renderer without requiring its optional PDF dependency."""
    return _load_renderer(monkeypatch)


def _task_success_payload() -> dict[str, object]:
    languages = ("en", "es", "pt", "hi", "ko", "zh")
    voice_systems = (
        "openai_minimal",
        "openai_xhigh",
        "gemini_minimal",
        "gemini_high",
        "xai_provider_default",
    )
    text_systems = ("gpt55_xhigh", "gemini31pro_high")
    return {
        "schema_version": "tau-multi-task-success-significance-v1",
        "seed": 42,
        "permutations": 100,
        "bootstrap_resamples": 100,
        "analysis_unit": "test",
        "test": "test",
        "sources_fingerprint_sha256": "0" * 64,
        "sources": [],
        "descriptive_trial_0": {
            "voice_language_system": {
                language: {
                    system: 0.40 + 0.01 * system_index
                    for system_index, system in enumerate(voice_systems)
                }
                for language in languages
            },
            "voice_language_mean": {language: 0.42 for language in languages},
            "voice_system_mean": {
                system: 0.50 + 0.01 * system_index
                for system_index, system in enumerate(voice_systems)
            },
            "voice_domain_mean": {
                "airline": 0.4,
                "retail": 0.5,
                "telecom": 0.6,
            },
            "text_language_system": {
                language: {system: 0.8 for system in text_systems}
                for language in languages
            },
            "text_language_mean": {language: 0.8 for language in languages},
            "text_voice_gap_points": {language: 38.0 for language in languages},
        },
        "per_system_significance": [],
        "fixed_system_panel": {"system": "fixed", "comparisons": []},
        "trial_stability": [],
        "overall_trial_stability": {
            "n_clusters": 1,
            "trial_0_rate": 0.5,
            "trial_1_rate": 0.5,
            "delta_points": 0.0,
            "raw_p": 1.0,
        },
    }


def test_load_task_scores_validates_and_maps_typed_artifact(
    tmp_path: Path, renderer: ModuleType
) -> None:
    artifact_path = tmp_path / "task-success.json"
    artifact_path.write_text(json.dumps(_task_success_payload()))

    language_system, provider = renderer._load_task_scores(artifact_path)

    assert language_system["English"]["OpenAI minimal"] == pytest.approx(40.0)
    assert language_system["Mandarin"]["xAI"] == pytest.approx(44.0)
    assert provider["OpenAI minimal"] == pytest.approx(50.0)
    assert provider["xAI"] == pytest.approx(54.0)


def test_load_task_scores_rejects_cohort_key_drift(
    tmp_path: Path, renderer: ModuleType
) -> None:
    payload = _task_success_payload()
    descriptive = payload["descriptive_trial_0"]
    assert isinstance(descriptive, dict)
    language_system = descriptive["voice_language_system"]
    assert isinstance(language_system, dict)
    language_system.pop("zh")
    artifact_path = tmp_path / "task-success.json"
    artifact_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="task-success language keys drifted"):
        renderer._load_task_scores(artifact_path)


def test_load_latency_scores_validates_and_maps_typed_artifact(
    tmp_path: Path, renderer: ModuleType
) -> None:
    artifact_path = tmp_path / "latency.json"
    artifact_path.write_text(
        json.dumps(
            {
                "descriptive_complete_cohort": {
                    "provider": {
                        system: {"latency_seconds": 1.0 + 0.1 * index}
                        for index, system in enumerate(renderer.SYSTEMS)
                    }
                }
            }
        )
    )

    scores = renderer._load_latency_scores(artifact_path)

    assert scores["OpenAI minimal"] == pytest.approx(1.0)
    assert scores["xAI"] == pytest.approx(1.4)


def test_default_config_uses_checked_in_frozen_artifacts(renderer: ModuleType) -> None:
    config = renderer._parse_args([])

    assert config.task_success_artifact == renderer.TASK_SUCCESS_PATH
    assert config.experience_artifact == renderer.ANALYSIS_PATH
    assert config.latency_artifact == renderer.LATENCY_PATH


def test_main_routes_explicit_inputs_to_selected_output_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, renderer: ModuleType
) -> None:
    experience_path = tmp_path / "experience.json"
    task_path = tmp_path / "task-success.json"
    latency_path = tmp_path / "latency.json"
    output_dir = tmp_path / "figures"
    experience_sentinel = object()
    task_scores = ({"task": {"scores": 1.0}}, {"provider": 2.0})
    latency_scores = {"provider": 3.0}
    calls: dict[str, object] = {}

    def load_experience(path: Path) -> object:
        calls["experience_path"] = path
        return experience_sentinel

    def load_task(path: Path) -> tuple[dict[str, Any], dict[str, float]]:
        calls["task_path"] = path
        return task_scores

    def load_latency(path: Path) -> dict[str, float]:
        calls["latency_path"] = path
        return latency_scores

    def render_summary(artifact: object, **kwargs: object) -> None:
        calls["summary"] = (artifact, kwargs)

    def render_metric(artifact: object, **kwargs: object) -> None:
        calls["metric"] = (artifact, kwargs)

    def render_heatmaps(artifact: object, **kwargs: object) -> None:
        calls["heatmaps"] = (artifact, kwargs)

    monkeypatch.setattr(renderer, "_load_experience_artifact", load_experience)
    monkeypatch.setattr(renderer, "_load_task_scores", load_task)
    monkeypatch.setattr(renderer, "_load_latency_scores", load_latency)
    monkeypatch.setattr(renderer, "render_model_summary", render_summary)
    monkeypatch.setattr(renderer, "render_model_metric_heatmap", render_metric)
    monkeypatch.setattr(renderer, "render_heatmaps", render_heatmaps)

    renderer.main(
        [
            "--task-success-artifact",
            str(task_path),
            "--experience-artifact",
            str(experience_path),
            "--latency-artifact",
            str(latency_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert calls["task_path"] == task_path
    assert calls["experience_path"] == experience_path
    assert calls["latency_path"] == latency_path
    assert output_dir.is_dir()
    summary_kwargs = calls["summary"][1]
    metric_kwargs = calls["metric"][1]
    heatmap_kwargs = calls["heatmaps"][1]
    assert summary_kwargs["output"] == output_dir / "model_summary.pdf"
    assert metric_kwargs["output"] == output_dir / "model_metric_heatmap.pdf"
    assert metric_kwargs["latency_provider"] == latency_scores
    assert heatmap_kwargs["output"] == output_dir / "language_system_heatmaps.pdf"
    assert list(output_dir.iterdir()) == []
