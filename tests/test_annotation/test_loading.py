# Copyright Sierra
"""The one results loader: streaming, filters, task attachment."""

import pytest

from tau2.annotation.loading import iter_loaded_sims, parse_reward_filter
from test_annotation.conftest import make_hi_results_dir


def test_iter_loaded_sims_streams_with_tasks(hi_results_dir):
    loaded = list(iter_loaded_sims([hi_results_dir]))
    assert [entry.sim.id for entry in loaded] == ["s1", "s2"]
    assert all(entry.domain == "airline" for entry in loaded)
    assert all(entry.results_dir == hi_results_dir for entry in loaded)
    assert all(entry.experiment_label == hi_results_dir.name for entry in loaded)
    # The run's OWN task definitions ride along.
    assert loaded[0].task is not None and str(loaded[0].task.id) == "t1"


def test_iter_loaded_sims_discovers_nested_runs(tmp_path):
    make_hi_results_dir(tmp_path / "parent", name="run_a")
    make_hi_results_dir(tmp_path / "parent", name="run_b")
    loaded = list(iter_loaded_sims([tmp_path / "parent"]))
    assert len(loaded) == 4
    assert {entry.experiment_label for entry in loaded} == {"run_a", "run_b"}


def test_reward_and_task_filters(hi_results_dir):
    failed = list(iter_loaded_sims([hi_results_dir], reward_filter="< 1"))
    assert [entry.sim.id for entry in failed] == ["s2"]
    only_t1 = list(iter_loaded_sims([hi_results_dir], task_ids=["t1"]))
    assert [entry.sim.id for entry in only_t1] == ["s1"]
    trial0 = list(iter_loaded_sims([hi_results_dir], trials=[0]))
    assert len(trial0) == 2
    assert list(iter_loaded_sims([hi_results_dir], trials=[7])) == []


def test_domain_override(hi_results_dir):
    loaded = list(iter_loaded_sims([hi_results_dir], domain_override="retail"))
    assert all(entry.domain == "retail" for entry in loaded)


def test_missing_path_and_bad_filter_are_loud(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(iter_loaded_sims([tmp_path / "nope"]))
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="No results.json"):
        list(iter_loaded_sims([tmp_path / "empty"]))
    with pytest.raises(ValueError, match="Invalid reward filter"):
        parse_reward_filter("about half")


def test_parse_reward_filter_ops():
    op, threshold = parse_reward_filter("< 1")
    assert threshold == 1.0 and op(0.5, 1.0) and not op(1.0, 1.0)
    op, _ = parse_reward_filter(">= 0.5")
    assert op(0.5, 0.5)
