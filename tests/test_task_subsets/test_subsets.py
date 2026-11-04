"""Frozen membership, frame verification, and safe redraws without providers."""

import argparse
import json
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from tau2.task_subsets import sampler, store
from tau2.task_subsets.cli import add_tasks_args
from tau2.task_subsets.schema import DrawRequest, TaskSubset


@pytest.fixture
def subset_dir(tmp_path, monkeypatch):
    root = tmp_path / "subsets"
    shutil.copytree(Path(__file__).parent / "fixtures", root)
    monkeypatch.setattr(store, "task_subset_dir", lambda: root)
    return root


def parser():
    result = argparse.ArgumentParser()
    add_tasks_args(result, default_size=50, default_seed=42)
    return result


@pytest.mark.parametrize("domain", ("airline", "retail", "telecom"))
def test_shipped_subset_reproduces_frame_and_strata(subset_dir, domain):
    subset = store.load_subset(domain + "_50")
    report = store.verify_subset(subset)
    assert report.selected_size == 50
    assert report.frame_size == (50 if domain == "airline" else 114)
    assert report.frame_digest == subset.frame.digest


def test_official_membership_cannot_change_even_with_same_counts(subset_dir):
    path = subset_dir / "retail_50.json"
    content = json.loads(path.read_text())
    content["task_ids"][0], content["task_ids"][1] = (
        content["task_ids"][1],
        content["task_ids"][0],
    )
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="frozen"):
        store.load_subset("retail_50")


def test_new_draw_is_deterministic_and_cannot_overwrite(subset_dir):
    request = DrawRequest(name="retail_20_review", domain="retail", size=20, seed=17)
    drawn = sampler.build_subset(request, store.load_frame("retail"))
    assert drawn == sampler.build_subset(request, store.load_frame("retail"))
    assert drawn.task_ids != store.load_frame("retail").task_ids[:20]
    path = store.save_subset(drawn)
    original = path.read_bytes()
    assert store.save_subset(drawn) == path
    assert path.read_bytes() == original
    altered = sampler.build_subset(
        request.model_copy(update={"seed": 18}), store.load_frame("retail")
    )
    with pytest.raises(FileExistsError, match="new name"):
        store.save_subset(altered)
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="reserved"):
        store.save_subset(drawn.model_copy(update={"name": "retail_50"}))


@pytest.mark.parametrize("size", (1, 9, 114, 200))
def test_small_budget_and_census_have_exact_capacity(size):
    frame = store.load_frame("telecom")
    subset = sampler.build_subset(
        DrawRequest(name=f"telecom_review_{size}", domain="telecom", size=size, seed=7),
        frame,
    )
    assert subset.size == min(size, len(frame.tasks))
    assert all(row.drawn <= row.frame_size for row in subset.strata)
    if size == 9:
        assert all(row.drawn == 1 for row in subset.strata)
    if size >= len(frame.tasks):
        assert subset.strategy == "census"
        assert subset.task_ids == frame.task_ids
    store.verify_subset(subset, frame=frame)


def test_frame_and_stratum_tampering_fail_before_publication(subset_dir):
    request = DrawRequest(name="telecom_review", domain="telecom", size=30, seed=9)
    frame = store.load_frame("telecom")
    subset = sampler.build_subset(request, frame)
    drifted = frame.model_copy(update={"tasks": list(reversed(frame.tasks))})
    with pytest.raises(ValueError, match="frame"):
        store.verify_subset(subset, frame=drifted)
    changed = subset.model_dump(mode="json")
    changed["strata"][0]["key"] = "invented"
    with pytest.raises(ValueError, match="strata"):
        store.verify_subset(TaskSubset.model_validate(changed), frame=frame)
    changed = subset.model_copy(update={"task_ids": list(reversed(subset.task_ids))})
    with pytest.raises(ValueError, match="order|draw"):
        store.save_subset(changed)
    assert not (subset_dir / "telecom_review.json").exists()


def test_strict_schema_and_safe_artifact_names(subset_dir):
    source = store.load_subset("airline_50").model_dump(mode="json")
    for update in (
        {"unexpected": True},
        {"task_ids": ["0"] * 50},
        {"name": "../../escape"},
        {"seed": "42"},
        {"domain": "elicitation"},
    ):
        with pytest.raises(ValidationError):
            TaskSubset.model_validate(source | update)
    with pytest.raises(ValueError, match="name"):
        store.load_subset("../retail_50")
    with pytest.raises(FileNotFoundError):
        store.load_subset("missing")


def test_cli_reads_and_dry_run_have_no_writes(subset_dir, capsys):
    cli = parser()
    original = {p.name: p.read_bytes() for p in subset_dir.iterdir()}
    for args in (["list"], ["show", "retail_50", "--ids"], ["verify"]):
        parsed = cli.parse_args(args)
        parsed.func(parsed)
        assert capsys.readouterr().out
    parsed = cli.parse_args(
        ["subset", "--domain", "retail", "--name", "retail_new", "--dry-run"]
    )
    parsed.func(parsed)
    assert json.loads(capsys.readouterr().out)["name"] == "retail_new"
    assert {p.name: p.read_bytes() for p in subset_dir.iterdir()} == original
    parsed = cli.parse_args(
        ["subset", "--domain", "retail", "--name", "retail_new", "--size", "20"]
    )
    parsed.func(parsed)
    assert store.load_subset("retail_new").size == 20
    with pytest.raises(SystemExit):
        cli.parse_args(["subset", "--domain", "retail", "--overwrite"])
