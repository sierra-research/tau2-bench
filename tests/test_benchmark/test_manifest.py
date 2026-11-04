"""Benchmark membership is data; shared scoring models validate its structure."""

import hashlib
import json
import shutil

import pytest

from tau2.data_model.benchmark import BenchmarkFrame, CellFrame
from tau2.runner.presets import load_frame


@pytest.mark.parametrize("name", ("voice", "text"))
def test_relocated_manifest_preserves_reviewed_population(name):
    """Pins (digests, binding revisions) regenerate; the reviewed population does not."""
    frame = load_frame(name)
    population = [
        [cell.language, cell.domain, cell.canonical_task_ids] for cell in frame.cells
    ]
    assert (
        hashlib.sha256(
            json.dumps(population, separators=(",", ":")).encode()
        ).hexdigest()
        == "cb66a374fd6b334a641a0198ee04499e7068f6eba47db57623610a3e89cb740b"
    )
    assert sum(len(cell.canonical_task_ids) for cell in frame.cells) == 900


def test_structural_frame_accepts_a_different_release():
    frame = BenchmarkFrame(
        benchmark_version="fixture-v3",
        revision="fixture",
        mode="voice",
        trials=[0],
        cells=[
            CellFrame(
                language="fr",
                domain="mock",
                canonical_task_ids=["a", "b"],
                binding_revision="fixture",
            )
        ],
    )
    assert len(frame.cells) == 1
    with pytest.raises(ValueError, match="unique|distinct"):
        frame.cells = frame.cells * 2


def test_manifest_rejects_changed_subset_bytes(tmp_path, monkeypatch):
    import tau2.utils

    source = tau2.utils.DATA_DIR / "tau2"
    root = tmp_path / "tau2"
    shutil.copytree(source / "benchmarks", root / "benchmarks")
    shutil.copytree(source / "task_subsets", root / "task_subsets")
    subset = root / "task_subsets" / "airline_50.json"
    subset.write_text(subset.read_text() + "\n")
    monkeypatch.setattr(tau2.utils, "DATA_DIR", tmp_path)
    with pytest.raises(ValueError, match="changed|digest"):
        load_frame()


def test_manifest_name_cannot_escape_data_directory():
    with pytest.raises(ValueError, match="name"):
        load_frame("../voice")
