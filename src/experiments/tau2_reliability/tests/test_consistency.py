"""Tests for consistency metrics."""

import pytest

from tau2_reliability.metrics.consistency import (
    _edit_distance,
    compute_all_consistency,
    compute_c_out,
    compute_c_res,
    compute_c_traj_d,
    compute_c_traj_s,
    compute_r_con,
)

# ---------------------------------------------------------------------------
# C_out tests
# ---------------------------------------------------------------------------

class TestCOut:
    def test_all_success(self, make_task_trial_data):
        td = make_task_trial_data(outcomes=[True, True, True, True, True])
        c_out, per = compute_c_out([td])
        assert c_out == pytest.approx(1.0, abs=0.01)

    def test_all_failure(self, make_task_trial_data):
        td = make_task_trial_data(outcomes=[False, False, False, False, False])
        c_out, per = compute_c_out([td])
        assert c_out == pytest.approx(1.0, abs=0.01)

    def test_maximally_inconsistent(self, make_task_trial_data):
        td = make_task_trial_data(outcomes=[True, False, True, False])
        c_out, per = compute_c_out([td])
        assert c_out < 0.15  # Near 0

    def test_single_trial(self, make_task_trial_data):
        td = make_task_trial_data(outcomes=[True])
        c_out, per = compute_c_out([td])
        assert c_out == 1.0

    def test_multiple_tasks_averaged(self, make_task_trial_data):
        td1 = make_task_trial_data(task_id="t1", outcomes=[True, True, True])
        td2 = make_task_trial_data(task_id="t2", outcomes=[True, False, True, False])
        c_out, per = compute_c_out([td1, td2])
        assert per["t1"] > per["t2"]
        assert c_out == pytest.approx((per["t1"] + per["t2"]) / 2)

    def test_per_task_keys(self, make_task_trial_data):
        td = make_task_trial_data(task_id="task_42")
        _, per = compute_c_out([td])
        assert "task_42" in per

    def test_bounded_0_1(self, make_task_trial_data):
        for outcomes in [
            [True], [False], [True, False], [True, True, False],
            [True, False, True, False, True],
        ]:
            td = make_task_trial_data(outcomes=outcomes)
            c_out, _ = compute_c_out([td])
            assert 0.0 <= c_out <= 1.0


# ---------------------------------------------------------------------------
# C_traj_d tests
# ---------------------------------------------------------------------------

class TestCTrajD:
    def test_identical_sequences(self, make_task_trial_data):
        seqs = [["search", "book"]] * 5
        td = make_task_trial_data(action_sequences=seqs)
        c_traj_d, _ = compute_c_traj_d([td])
        assert c_traj_d == pytest.approx(1.0, abs=0.01)

    def test_completely_different(self, make_task_trial_data):
        seqs = [["action_a"], ["action_b"], ["action_c"]]
        td = make_task_trial_data(
            outcomes=[True, True, True], action_sequences=seqs
        )
        c_traj_d, _ = compute_c_traj_d([td])
        assert c_traj_d < 0.5

    def test_partially_overlapping(self, make_task_trial_data):
        seqs = [["search", "book"], ["search", "cancel"], ["search", "book"]]
        td = make_task_trial_data(
            outcomes=[True, True, True], action_sequences=seqs
        )
        c_traj_d, _ = compute_c_traj_d([td])
        assert 0.3 < c_traj_d < 0.95

    def test_empty_sequences(self, make_task_trial_data):
        seqs = [[], [], []]
        td = make_task_trial_data(outcomes=[True, True, True], action_sequences=seqs)
        c_traj_d, _ = compute_c_traj_d([td])
        assert c_traj_d == 1.0

    def test_single_trial(self, make_task_trial_data):
        td = make_task_trial_data(
            outcomes=[True], action_sequences=[["search", "book"]]
        )
        c_traj_d, _ = compute_c_traj_d([td])
        assert c_traj_d == 1.0


# ---------------------------------------------------------------------------
# C_traj_s tests
# ---------------------------------------------------------------------------

class TestCTrajS:
    def test_identical_order(self, make_task_trial_data):
        seqs = [["a", "b", "c"]] * 4
        td = make_task_trial_data(outcomes=[True] * 4, action_sequences=seqs)
        c_traj_s, _ = compute_c_traj_s([td])
        assert c_traj_s == pytest.approx(1.0, abs=0.01)

    def test_reversed_order(self, make_task_trial_data):
        seqs = [["a", "b", "c"], ["c", "b", "a"]]
        td = make_task_trial_data(outcomes=[True, True], action_sequences=seqs)
        c_traj_s, _ = compute_c_traj_s([td])
        assert c_traj_s < 0.7

    def test_different_lengths(self, make_task_trial_data):
        seqs = [["a", "b"], ["a", "b", "c", "d"]]
        td = make_task_trial_data(outcomes=[True, True], action_sequences=seqs)
        c_traj_s, _ = compute_c_traj_s([td])
        assert 0.0 < c_traj_s < 1.0

    def test_empty_sequences(self, make_task_trial_data):
        seqs = [[], []]
        td = make_task_trial_data(outcomes=[True, True], action_sequences=seqs)
        c_traj_s, _ = compute_c_traj_s([td])
        assert c_traj_s == 1.0


# ---------------------------------------------------------------------------
# Edit distance tests
# ---------------------------------------------------------------------------

class TestEditDistance:
    def test_identical(self):
        assert _edit_distance(["a", "b", "c"], ["a", "b", "c"]) == 0

    def test_empty_both(self):
        assert _edit_distance([], []) == 0

    def test_one_empty(self):
        assert _edit_distance([], ["a", "b", "c"]) == 3
        assert _edit_distance(["a", "b"], []) == 2

    def test_substitution(self):
        assert _edit_distance(["a", "b", "c"], ["a", "x", "c"]) == 1

    def test_insertion(self):
        assert _edit_distance(["a", "c"], ["a", "b", "c"]) == 1

    def test_deletion(self):
        assert _edit_distance(["a", "b", "c"], ["a", "c"]) == 1


# ---------------------------------------------------------------------------
# C_res tests
# ---------------------------------------------------------------------------

class TestCRes:
    def test_identical_resources(self, make_task_trial_data):
        td = make_task_trial_data(
            costs=[0.1, 0.1, 0.1], durations=[30, 30, 30],
            action_sequences=[["a", "b"]] * 3, outcomes=[True] * 3,
        )
        c_res, _ = compute_c_res([td])
        assert c_res == pytest.approx(1.0, abs=0.01)

    def test_high_variance(self, make_task_trial_data):
        td = make_task_trial_data(
            costs=[0.01, 1.0, 0.05], durations=[10, 200, 15],
            action_sequences=[["a"], ["a"] * 20, ["a", "b"]], outcomes=[True] * 3,
        )
        c_res, _ = compute_c_res([td])
        assert c_res < 0.5

    def test_zero_cost(self, make_task_trial_data):
        td = make_task_trial_data(
            costs=[0.0, 0.0, 0.0], durations=[30, 30, 30],
            action_sequences=[["a"]] * 3, outcomes=[True] * 3,
        )
        c_res, _ = compute_c_res([td])
        # Should not crash, CV of zero-mean is 0
        assert 0.0 <= c_res <= 1.0

    def test_single_trial(self, make_task_trial_data):
        td = make_task_trial_data(
            outcomes=[True], costs=[0.1], durations=[30.0],
            action_sequences=[["a"]],
        )
        c_res, _ = compute_c_res([td])
        assert c_res == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# R_con aggregate tests
# ---------------------------------------------------------------------------

class TestRCon:
    def test_perfect_consistency(self):
        r = compute_r_con(1.0, 1.0, 1.0, 1.0)
        assert r == pytest.approx(1.0, abs=0.01)

    def test_zero_consistency(self):
        r = compute_r_con(0.0, 0.0, 0.0, 0.0)
        assert r == pytest.approx(0.0, abs=0.01)

    def test_mixed(self):
        r = compute_r_con(0.8, 0.6, 0.4, 0.9)
        assert 0.0 < r < 1.0


# ---------------------------------------------------------------------------
# compute_all_consistency integration test
# ---------------------------------------------------------------------------

class TestComputeAll:
    def test_returns_consistency_metrics(self, make_task_trial_data):
        td1 = make_task_trial_data(task_id="t1", outcomes=[True, True, True])
        td2 = make_task_trial_data(task_id="t2", outcomes=[True, False, True])
        result = compute_all_consistency([td1, td2])
        assert 0.0 <= result.c_out <= 1.0
        assert 0.0 <= result.c_traj_d <= 1.0
        assert 0.0 <= result.c_traj_s <= 1.0
        assert 0.0 <= result.c_res <= 1.0
        assert 0.0 <= result.r_con <= 1.0
        assert "t1" in result.per_task
        assert "t2" in result.per_task
