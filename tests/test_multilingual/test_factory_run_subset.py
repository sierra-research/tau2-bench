# Copyright Sierra
"""`tau2 factory subset-run` — seeded subset of a dir-format run directory.

Fixture: a fake voice run (results.json with a populated simulation_index,
one simulations/<id>.json per entry, one artifacts/task_<tid>/sim_<id>/audio/
both.wav per entry). Asserts determinism across invocations, pruned-index
integrity (Results.load round-trip), exact artifact mapping, provenance
stamping, source immutability, and every loud failure mode.
"""

import argparse
import json
from pathlib import Path

import pytest

from tau2.data_model.simulation import Results
from tau2.multilingual.factory.author_form import FactoryDraftError
from tau2.multilingual.factory.cli import add_factory_args
from tau2.multilingual.factory.run_subset import (
    RUN_SUBSET_DRAW_RULE,
    SUBSET_PROVENANCE_KEY,
    build_run_subset,
    draw_run_subset,
)

N_SIMS = 8


def _index_entry(i: int) -> dict:
    return {
        "id": f"sim-{i:02d}",
        "task_id": f"{i % 4}_es_identity",  # 4 tasks x 2 trials
        "trial": i // 4,
        "reward": 1.0 if i % 2 == 0 else 0.0,
        "quality": None,
        "nativeness": 0.5,
        "fidelity": None,
        "intonation": None,
        "termination_reason": "agent_stop",
        "agent_cost": None,
        "duration": 100.0 + i,
    }


@pytest.fixture
def source_run(tmp_path: Path) -> Path:
    run = tmp_path / "es_airline_openai_xhigh"
    (run / "simulations").mkdir(parents=True)
    index = [_index_entry(i) for i in range(N_SIMS)]
    for entry in index:
        sim = {
            "id": entry["id"],
            "task_id": entry["task_id"],
            "trial": entry["trial"],
            "payload": f"conversation for {entry['id']}",
        }
        (run / "simulations" / f"{entry['id']}.json").write_text(json.dumps(sim))
        audio = (
            run
            / "artifacts"
            / f"task_{entry['task_id']}"
            / f"sim_{entry['id']}"
            / "audio"
        )
        audio.mkdir(parents=True)
        (audio / "both.wav").write_bytes(b"RIFF" + entry["id"].encode())
    meta = {
        "timestamp": "2026-08-19 10:00:00",
        "info": {"git_commit": "abc123", "num_trials": 2, "custom_key": "kept"},
        "tasks": [{"id": f"{i}_es_identity"} for i in range(4)],
        "simulation_index": index,
    }
    (run / "results.json").write_text(json.dumps(meta, indent=2))
    return run


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_subset_is_deterministic_and_complete(source_run: Path, tmp_path: Path):
    out_a = tmp_path / "subset_a"
    out_b = tmp_path / "subset_b"
    outcome_a = build_run_subset(source_run, out_a, seed=7, count=3)
    outcome_b = build_run_subset(source_run, out_b, seed=7, count=3)

    ids_a = [e.id for e in outcome_a.drawn]
    ids_b = [e.id for e in outcome_b.drawn]
    assert ids_a == ids_b
    assert len(ids_a) == 3

    meta_a = json.loads((out_a / "results.json").read_text())
    meta_b = json.loads((out_b / "results.json").read_text())
    # Identical apart from wall-clock provenance fields.
    for meta in (meta_a, meta_b):
        meta["info"][SUBSET_PROVENANCE_KEY].pop("created_at")
        meta["info"][SUBSET_PROVENANCE_KEY].pop("git_sha")
    assert meta_a == meta_b

    # A different seed draws a different cohort (statistically certain on
    # C(8,3) with these fixed seeds).
    outcome_c = build_run_subset(source_run, tmp_path / "subset_c", seed=8, count=3)
    assert [e.id for e in outcome_c.drawn] != ids_a


def test_pruned_index_and_files_are_self_consistent(source_run: Path, tmp_path: Path):
    out = tmp_path / "subset"
    outcome = build_run_subset(source_run, out, seed=1, count=4)
    drawn_ids = {e.id for e in outcome.drawn}

    meta = json.loads((out / "results.json").read_text())
    assert {e["id"] for e in meta["simulation_index"]} == drawn_ids
    assert {p.stem for p in (out / "simulations").glob("*.json")} == drawn_ids

    # timestamp/info/tasks pass through byte-faithfully (info gains only the
    # provenance key — unknown keys like custom_key survive).
    src_meta = json.loads((source_run / "results.json").read_text())
    assert meta["timestamp"] == src_meta["timestamp"]
    assert meta["tasks"] == src_meta["tasks"]
    info = dict(meta["info"])
    provenance = info.pop(SUBSET_PROVENANCE_KEY)
    assert info == src_meta["info"]

    assert provenance["draw_rule"] == RUN_SUBSET_DRAW_RULE
    assert provenance["source"] == str(source_run.resolve())
    assert provenance["seed"] == 1
    assert provenance["count"] == 4
    assert provenance["source_index_size"] == N_SIMS

    # Index entries preserve source order and full field content.
    src_by_id = {e["id"]: e for e in src_meta["simulation_index"]}
    src_order = [e["id"] for e in src_meta["simulation_index"] if e["id"] in drawn_ids]
    assert [e["id"] for e in meta["simulation_index"]] == src_order
    for entry in meta["simulation_index"]:
        assert entry == src_by_id[entry["id"]]


def test_artifacts_map_exactly_to_drawn_sims(source_run: Path, tmp_path: Path):
    out = tmp_path / "subset"
    outcome = build_run_subset(source_run, out, seed=3, count=3)
    expected = {
        f"task_{e.task_id}/sim_{e.id}/audio/both.wav": e.id.encode()
        for e in outcome.drawn
    }
    actual = {
        str(p.relative_to(out / "artifacts")): p.read_bytes().removeprefix(b"RIFF")
        for p in (out / "artifacts").rglob("*")
        if p.is_file()
    }
    assert actual == expected
    assert outcome.artifact_dirs_copied == 3


def test_source_is_never_modified(source_run: Path, tmp_path: Path):
    before = _snapshot(source_run)
    build_run_subset(source_run, tmp_path / "subset", seed=5, count=2)
    assert _snapshot(source_run) == before


def test_subset_of_a_real_run_round_trips_through_results(tmp_path: Path):
    """A subset of a REAL dir-format run (written via Results.save) must be
    consumable by Results.load — pruned index and on-disk sim files agree,
    and the provenance stamped into info survives model validation."""
    import fixtures_runs

    sims = [
        fixtures_runs.hi_sim(
            f"s{i}",
            f"t{i % 2}",
            trial=i // 2,
            checks=fixtures_runs.hi_factor_checks(),
            judge_model="fake-judge",
        )
        for i in range(4)
    ]
    run_dir = fixtures_runs.make_hi_results(tmp_path, sims, name="real_run")
    for sim in sims:
        audio = (
            run_dir / "artifacts" / f"task_{sim.task_id}" / f"sim_{sim.id}" / "audio"
        )
        audio.mkdir(parents=True)
        (audio / "both.wav").write_bytes(b"RIFFfake")

    out = tmp_path / "subset"
    outcome = build_run_subset(run_dir, out, seed=2, count=2)
    results = Results.load(out)
    assert results.simulation_index is not None
    assert len(results.simulation_index) == 2
    assert {s.id for s in results.simulations} == {e.id for e in outcome.drawn}


def test_count_over_index_size_fails(source_run: Path, tmp_path: Path):
    with pytest.raises(FactoryDraftError, match="exceeds the source index size"):
        build_run_subset(source_run, tmp_path / "subset", seed=1, count=N_SIMS + 1)
    assert not (tmp_path / "subset").exists()


def test_existing_out_dir_fails(source_run: Path, tmp_path: Path):
    out = tmp_path / "subset"
    out.mkdir()
    with pytest.raises(FactoryDraftError, match="already exists"):
        build_run_subset(source_run, out, seed=1, count=2)


def test_out_inside_source_fails(source_run: Path):
    with pytest.raises(FactoryDraftError, match="inside the source"):
        build_run_subset(source_run, source_run / "subset", seed=1, count=2)


def test_missing_sim_file_fails_before_writing(source_run: Path, tmp_path: Path):
    # Remove one sim file; any full draw must trip over it.
    (source_run / "simulations" / "sim-03.json").unlink()
    with pytest.raises(FactoryDraftError, match="sim file"):
        build_run_subset(source_run, tmp_path / "subset", seed=1, count=N_SIMS)
    assert not (tmp_path / "subset").exists()


def test_missing_artifacts_fail_before_writing(source_run: Path, tmp_path: Path):
    import shutil

    shutil.rmtree(source_run / "artifacts" / "task_2_es_identity" / "sim_sim-02")
    with pytest.raises(FactoryDraftError, match="no artifact subtree"):
        build_run_subset(source_run, tmp_path / "subset", seed=1, count=N_SIMS)
    assert not (tmp_path / "subset").exists()


def test_corrupt_sim_id_fails(source_run: Path, tmp_path: Path):
    sim_path = source_run / "simulations" / "sim-01.json"
    sim = json.loads(sim_path.read_text())
    sim["id"] = "someone-else"
    sim_path.write_text(json.dumps(sim))
    with pytest.raises(FactoryDraftError, match="records id"):
        build_run_subset(source_run, tmp_path / "subset", seed=1, count=N_SIMS)
    assert not (tmp_path / "subset").exists()


def test_missing_index_fails(source_run: Path, tmp_path: Path):
    meta = json.loads((source_run / "results.json").read_text())
    del meta["simulation_index"]
    (source_run / "results.json").write_text(json.dumps(meta))
    with pytest.raises(FactoryDraftError, match="no simulation_index"):
        build_run_subset(source_run, tmp_path / "subset", seed=1, count=2)


def test_draw_rule_sorts_before_sampling():
    """run-subset-draw-v1: the draw depends on (task_id, trial, id) order,
    not on the index's file order."""
    from tau2.data_model.simulation import SimulationIndexEntry

    entries = [
        SimulationIndexEntry.model_validate(_index_entry(i)) for i in range(N_SIMS)
    ]
    shuffled = list(reversed(entries))
    assert draw_run_subset(entries, seed=11, count=3) == draw_run_subset(
        shuffled, seed=11, count=3
    )


def test_cli_registration_parses():
    parser = argparse.ArgumentParser(prog="tau2 factory")
    add_factory_args(parser)
    args = parser.parse_args(
        [
            "subset-run",
            "--results",
            "/tmp/src",
            "--seed",
            "7",
            "--count",
            "30",
            "--out",
            "/tmp/out",
        ]
    )
    assert args.func.__name__ == "run_factory_subset_run"
    assert args.seed == 7
    assert args.count == 30
