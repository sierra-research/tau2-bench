"""Tests for the visualizer's split trajectory layout (viewer_data)."""

import gzip
import json

import pytest

from tau2.scripts.leaderboard.viewer_data import (
    check_staging,
    split_results,
    write_viewer_cell,
)


def _results() -> dict:
    return {
        "timestamp": "2026-10-01T00:00:00",
        "info": {"num_trials": 2},
        "tasks": [{"id": "0"}, {"id": "1"}],
        "simulations": [
            {
                "id": "sim-a",
                "task_id": "0",
                "trial": 0,
                "reward_info": {"reward": 1.0},
                "termination_reason": "user_stop",
                "agent_cost": 0.12,
                "duration": 30.5,
                "messages": [{"role": "user", "content": "hi"}],
            },
            {
                "id": "sim-b",
                "task_id": "1",
                "trial": 1,
                "reward_info": None,
                "termination_reason": "max_steps",
                "messages": [],
            },
        ],
    }


def test_split_results_indexes_without_messages():
    meta, simulations = split_results(_results())

    assert "simulations" not in meta
    assert meta["tasks"] == [{"id": "0"}, {"id": "1"}]
    assert meta["simulation_index"] == [
        {
            "id": "sim-a",
            "task_id": "0",
            "trial": 0,
            "reward": 1.0,
            "termination_reason": "user_stop",
            "agent_cost": 0.12,
            "duration": 30.5,
        },
        {
            "id": "sim-b",
            "task_id": "1",
            "trial": 1,
            "reward": None,
            "termination_reason": "max_steps",
            "agent_cost": None,
            "duration": None,
        },
    ]
    assert simulations == _results()["simulations"]


def test_write_viewer_cell_writes_gzipped_index_and_simulations(tmp_path):
    cell = tmp_path / "trajectories" / "retail"
    (cell / "simulations").mkdir(parents=True)
    (cell / "simulations" / "stale.json").write_bytes(b"old")

    assert write_viewer_cell(_results(), cell) == 2

    read = lambda path: json.loads(gzip.decompress(path.read_bytes()))  # noqa: E731
    assert [e["id"] for e in read(cell / "results.json")["simulation_index"]] == [
        "sim-a",
        "sim-b",
    ]
    assert sorted(p.name for p in (cell / "simulations").iterdir()) == [
        "sim-a.json",
        "sim-b.json",
    ]
    assert read(cell / "simulations" / "sim-a.json")["messages"][0]["content"] == "hi"


def _submissions(tmp_path):
    root = tmp_path / "submissions"
    (root / "text-model").mkdir(parents=True)
    (root / "voice-model").mkdir()
    (root / "manifest.json").write_text(
        json.dumps(
            {"submissions": ["text-model"], "voice_submissions": ["voice-model"]}
        )
    )
    (root / "text-model" / "submission.json").write_text(
        json.dumps({"trajectory_files": {"retail": "retail_results.json"}})
    )
    (root / "voice-model" / "submission.json").write_text(
        json.dumps({"trajectory_files": {"retail": "retail"}})
    )
    return root


def test_check_staging_accepts_text_viewer_cells(tmp_path):
    staging = tmp_path / "staging"
    write_viewer_cell(_results(), staging / "text-model" / "trajectories" / "retail")

    check_staging(staging, _submissions(tmp_path))


@pytest.mark.parametrize(
    "stray",
    [
        # A voice cell: re-uploading it gzipped would overwrite the original.
        "voice-model/trajectories/retail/results.json",
        # A text submission's monolithic file or an unlisted domain.
        "text-model/trajectories/retail_results.json",
        "text-model/trajectories/airline/results.json",
        "text-model/submission.json",
    ],
)
def test_check_staging_refuses_anything_else(tmp_path, stray):
    staging = tmp_path / "staging"
    write_viewer_cell(_results(), staging / "text-model" / "trajectories" / "retail")
    (staging / stray).parent.mkdir(parents=True, exist_ok=True)
    (staging / stray).write_bytes(b"{}")

    with pytest.raises(SystemExit, match="refusing to upload"):
        check_staging(staging, _submissions(tmp_path))


def test_failed_viewer_cells_still_upload_submission_json(tmp_path, monkeypatch):
    from rich.console import Console

    from tau2.scripts.leaderboard import review_submission

    aws_calls, synced = [], []
    monkeypatch.setattr(
        review_submission,
        "_run_aws",
        lambda cmd, profile, console: aws_calls.append(cmd) or True,
    )
    monkeypatch.setattr(
        review_submission, "sync_viewer_tree", lambda *args, **kw: synced.append(1)
    )
    submission_dir = tmp_path / "text-model"
    submission_dir.mkdir()
    (submission_dir / "submission.json").write_text("{}")
    broken = tmp_path / "retail_results.json"
    broken.write_text("not json")

    review_submission._upload(
        submission_dir,
        [str(broken)],
        {"retail": "retail_results.json"},
        is_voice=False,
        profile="test",
        console=Console(quiet=True),
    )

    assert synced == []
    assert aws_calls[-1][-1].endswith("/text-model/submission.json")
