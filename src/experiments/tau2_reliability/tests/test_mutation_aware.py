"""Tests for mutation-aware failure attribution."""

import pytest

from tau2_reliability.analysis.mutation_aware import (
    _compute_verification_rates,
    _find_decisive_mutation,
    compute_mutation_analysis,
)
from tau2_reliability.models import TaskTrialData


def _make_td(task_id, outcomes, sequences, types_per_trial):
    """Helper to create TaskTrialData with tool types."""
    return TaskTrialData(
        task_id=task_id,
        outcomes=outcomes,
        action_sequences=sequences,
        costs=[0.1] * len(outcomes),
        durations=[30.0] * len(outcomes),
        num_actions=[len(s) for s in sequences],
        tool_types_per_action=types_per_trial,
    )


class TestComputeMutationAnalysis:
    def test_basic_write_fraction(self):
        td = _make_td(
            "t1",
            outcomes=[True, False],
            sequences=[["read_user", "update_user"], ["read_user", "read_user"]],
            types_per_trial=[["READ", "WRITE"], ["READ", "READ"]],
        )
        result = compute_mutation_analysis([td])
        assert result.write_action_fraction == pytest.approx(0.25)  # 1 WRITE out of 4

    def test_all_reads(self):
        td = _make_td(
            "t1",
            outcomes=[True, True],
            sequences=[["read_a", "read_b"], ["read_a", "read_b"]],
            types_per_trial=[["READ", "READ"], ["READ", "READ"]],
        )
        result = compute_mutation_analysis([td])
        assert result.write_action_fraction == 0.0

    def test_all_writes(self):
        td = _make_td(
            "t1",
            outcomes=[True, False],
            sequences=[["write_a"], ["write_b"]],
            types_per_trial=[["WRITE"], ["WRITE"]],
        )
        result = compute_mutation_analysis([td])
        assert result.write_action_fraction == 1.0

    def test_per_task_breakdown(self):
        td1 = _make_td("t1", [True], [["r", "w"]], [["READ", "WRITE"]])
        td2 = _make_td("t2", [True], [["r", "r"]], [["READ", "READ"]])
        result = compute_mutation_analysis([td1, td2])
        assert result.per_task["t1"]["write_fraction"] == pytest.approx(0.5)
        assert result.per_task["t2"]["write_fraction"] == pytest.approx(0.0)

    def test_with_tool_type_map(self):
        td = TaskTrialData(
            task_id="t1",
            outcomes=[True, False],
            action_sequences=[["get_user", "cancel_order"], ["get_user", "get_order"]],
            costs=[0.1, 0.1],
            durations=[30, 30],
            num_actions=[2, 2],
        )
        tool_map = {"get_user": "READ", "get_order": "READ", "cancel_order": "WRITE"}
        result = compute_mutation_analysis([td], tool_type_map=tool_map)
        assert result.write_action_fraction == pytest.approx(0.25)

    def test_empty_data(self):
        result = compute_mutation_analysis([])
        assert result.write_action_fraction == 0.0


class TestDecisiveMutation:
    def test_clear_decisive_mutation(self):
        td = _make_td(
            "t1",
            outcomes=[True, True, False, False],
            sequences=[
                ["read", "update"],  # success
                ["read", "update"],  # success
                ["read", "cancel"],  # fail
                ["read", "cancel"],  # fail
            ],
            types_per_trial=[
                ["READ", "WRITE"],
                ["READ", "WRITE"],
                ["READ", "WRITE"],
                ["READ", "WRITE"],
            ],
        )
        decisive = _find_decisive_mutation(td, None)
        # "update" appears in 100% of successes, 0% of failures
        # "cancel" appears in 0% of successes, 100% of failures
        assert decisive in ["update", "cancel"]

    def test_no_decisive_when_all_same(self):
        td = _make_td(
            "t1",
            outcomes=[True, False],
            sequences=[["read", "update"], ["read", "update"]],
            types_per_trial=[["READ", "WRITE"], ["READ", "WRITE"]],
        )
        # Same write action in both — difference < threshold
        decisive = _find_decisive_mutation(td, None)
        assert decisive is None

    def test_no_decisive_all_success(self):
        td = _make_td(
            "t1",
            outcomes=[True, True],
            sequences=[["read", "update"], ["read", "update"]],
            types_per_trial=[["READ", "WRITE"], ["READ", "WRITE"]],
        )
        decisive = _find_decisive_mutation(td, None)
        assert decisive is None


class TestVerificationRates:
    def test_all_verified(self):
        td = _make_td(
            "t1",
            outcomes=[True, True],
            sequences=[["read", "write"], ["read", "write"]],
            types_per_trial=[["READ", "WRITE"], ["READ", "WRITE"]],
        )
        rate_s, rate_f = _compute_verification_rates([td], None)
        assert rate_s == pytest.approx(1.0)
        assert rate_f is None  # No failures

    def test_unverified_writes(self):
        td = _make_td(
            "t1",
            outcomes=[False, False],
            sequences=[["write_a", "write_b"], ["write_a", "write_b"]],
            types_per_trial=[["WRITE", "WRITE"], ["WRITE", "WRITE"]],
        )
        rate_s, rate_f = _compute_verification_rates([td], None)
        assert rate_s is None  # No successes
        assert rate_f == pytest.approx(0.0)  # No reads before any write

    def test_mixed_verification(self):
        td = _make_td(
            "t1",
            outcomes=[True, False],
            sequences=[["read", "write", "read", "write"], ["write", "write"]],
            types_per_trial=[["READ", "WRITE", "READ", "WRITE"], ["WRITE", "WRITE"]],
        )
        rate_s, rate_f = _compute_verification_rates([td], None)
        assert rate_s == pytest.approx(1.0)  # Both writes preceded by read
        assert rate_f == pytest.approx(0.0)  # Neither write preceded by read
