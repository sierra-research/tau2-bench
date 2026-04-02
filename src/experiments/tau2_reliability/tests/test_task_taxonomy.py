"""Tests for reliability-driven task classification."""


from tau2_reliability.analysis.task_taxonomy import (
    classify_tasks,
    compute_taxonomy_summary,
)
from tau2_reliability.models import (
    ConsistencyMetrics,
    TaskReliabilityClass,
    TaskTrialData,
)


def _make_consistency(per_task: dict[str, dict[str, float]]) -> ConsistencyMetrics:
    """Helper to create ConsistencyMetrics with per-task data."""
    return ConsistencyMetrics(
        c_out=0.5, c_traj_d=0.5, c_traj_s=0.5, c_res=0.5, r_con=0.5,
        per_task=per_task,
    )


def _make_td(task_id, pass_rate, n=4):
    """Helper to create TaskTrialData with a given pass rate."""
    n_pass = int(pass_rate * n)
    outcomes = [True] * n_pass + [False] * (n - n_pass)
    return TaskTrialData(
        task_id=task_id, outcomes=outcomes,
        action_sequences=[["a"]] * n, costs=[0.1] * n,
        durations=[30] * n, num_actions=[1] * n,
    )


class TestClassifyTasks:
    def test_stable_pass(self):
        consistency = _make_consistency({
            "t1": {"c_out": 0.95, "c_traj_s": 0.9, "c_traj_d": 0.9, "c_res": 0.9},
        })
        td = [_make_td("t1", 1.0)]
        result = classify_tasks(td, consistency)
        assert result["t1"] == TaskReliabilityClass.STABLE_PASS

    def test_stable_fail(self):
        consistency = _make_consistency({
            "t1": {"c_out": 0.95, "c_traj_s": 0.9, "c_traj_d": 0.9, "c_res": 0.9},
        })
        td = [_make_td("t1", 0.0)]
        result = classify_tasks(td, consistency)
        assert result["t1"] == TaskReliabilityClass.STABLE_FAIL

    def test_bimodal(self):
        consistency = _make_consistency({
            "t1": {"c_out": 0.1, "c_traj_s": 0.5, "c_traj_d": 0.5, "c_res": 0.5},
        })
        td = [_make_td("t1", 0.5)]
        result = classify_tasks(td, consistency)
        assert result["t1"] == TaskReliabilityClass.BIMODAL

    def test_fragile(self):
        consistency = _make_consistency({
            "t1": {"c_out": 0.9, "c_traj_s": 0.3, "c_traj_d": 0.8, "c_res": 0.9},
        })
        td = [_make_td("t1", 1.0)]
        result = classify_tasks(td, consistency)
        assert result["t1"] == TaskReliabilityClass.FRAGILE

    def test_mixed_tasks(self):
        consistency = _make_consistency({
            "stable": {"c_out": 0.95, "c_traj_s": 0.9, "c_traj_d": 0.9, "c_res": 0.9},
            "bimodal": {"c_out": 0.05, "c_traj_s": 0.4, "c_traj_d": 0.5, "c_res": 0.6},
            "fail": {"c_out": 0.9, "c_traj_s": 0.8, "c_traj_d": 0.8, "c_res": 0.8},
        })
        td = [_make_td("stable", 1.0), _make_td("bimodal", 0.5), _make_td("fail", 0.0)]
        result = classify_tasks(td, consistency)
        assert result["stable"] == TaskReliabilityClass.STABLE_PASS
        assert result["bimodal"] == TaskReliabilityClass.BIMODAL
        assert result["fail"] == TaskReliabilityClass.STABLE_FAIL


class TestTaxonomySummary:
    def test_counts(self):
        classifications = {
            "t1": TaskReliabilityClass.STABLE_PASS,
            "t2": TaskReliabilityClass.BIMODAL,
            "t3": TaskReliabilityClass.BIMODAL,
            "t4": TaskReliabilityClass.STABLE_FAIL,
        }
        summary = compute_taxonomy_summary(classifications)
        assert summary.counts["stable_pass"] == 1
        assert summary.counts["bimodal"] == 2
        assert summary.counts["stable_fail"] == 1

    def test_bimodal_tasks_list(self):
        classifications = {
            "t1": TaskReliabilityClass.STABLE_PASS,
            "t2": TaskReliabilityClass.BIMODAL,
            "t3": TaskReliabilityClass.BIMODAL,
        }
        summary = compute_taxonomy_summary(classifications)
        assert summary.bimodal_tasks == ["t2", "t3"]

    def test_empty(self):
        summary = compute_taxonomy_summary({})
        assert summary.bimodal_tasks == []
        assert all(v == 0 for v in summary.counts.values())
