"""Tests for cross-trial trajectory divergence analysis."""


from tau2_reliability.analysis.divergence import (
    _classify_divergence,
    _find_decision_points,
    _longest_common_prefix,
    _most_common_sequence,
    compute_all_divergence_profiles,
    compute_divergence_profile,
)
from tau2_reliability.models import DivergenceType


class TestLongestCommonPrefix:
    def test_identical_sequences(self):
        seqs = [["a", "b", "c"], ["a", "b", "c"], ["a", "b", "c"]]
        assert _longest_common_prefix(seqs) == ["a", "b", "c"]

    def test_diverge_at_second(self):
        seqs = [["a", "b", "c"], ["a", "x", "c"], ["a", "b", "d"]]
        assert _longest_common_prefix(seqs) == ["a"]

    def test_no_common_prefix(self):
        seqs = [["a", "b"], ["x", "y"], ["p", "q"]]
        assert _longest_common_prefix(seqs) == []

    def test_empty_sequences(self):
        assert _longest_common_prefix([]) == []
        assert _longest_common_prefix([[], []]) == []

    def test_different_lengths(self):
        seqs = [["a", "b", "c", "d"], ["a", "b"]]
        assert _longest_common_prefix(seqs) == ["a", "b"]

    def test_single_sequence(self):
        assert _longest_common_prefix([["a", "b"]]) == ["a", "b"]


class TestClassifyDivergence:
    def test_tool_choice(self):
        seqs = [["a", "b"], ["a", "c"]]
        assert _classify_divergence(seqs, 1) == DivergenceType.TOOL_CHOICE

    def test_tool_args_same_action(self):
        seqs = [["a", "b", "c"], ["a", "b", "d"]]
        # Both pick "b" at index 1, diverge at 2
        assert _classify_divergence(seqs, 2) == DivergenceType.TOOL_CHOICE

    def test_none_divergence_idx(self):
        seqs = [["a", "b"], ["a", "c"]]
        assert _classify_divergence(seqs, None) is None

    def test_sequence_ended_early(self):
        seqs = [["a", "b", "c"], ["a", "b"]]
        # At index 2, second sequence has ended
        result = _classify_divergence(seqs, 2)
        assert result == DivergenceType.TOOL_CHOICE


class TestMostCommonSequence:
    def test_majority_wins(self):
        seqs = [["a", "b"], ["a", "b"], ["x", "y"]]
        assert _most_common_sequence(seqs) == ["a", "b"]

    def test_single_sequence(self):
        assert _most_common_sequence([["a"]]) == ["a"]

    def test_empty(self):
        assert _most_common_sequence([]) == []

    def test_tie_returns_one(self):
        seqs = [["a"], ["b"]]
        result = _most_common_sequence(seqs)
        assert result in [["a"], ["b"]]


class TestFindDecisionPoints:
    def test_no_divergence(self):
        seqs = [["a", "b"], ["a", "b"]]
        outcomes = [True, True]
        points = _find_decision_points(seqs, outcomes)
        assert len(points) == 0

    def test_single_divergence(self):
        seqs = [["a", "b"], ["a", "c"]]
        outcomes = [True, False]
        points = _find_decision_points(seqs, outcomes)
        assert len(points) == 1
        assert points[0].action_index == 1
        assert "b" in points[0].actions_observed
        assert "c" in points[0].actions_observed

    def test_multiple_divergence_points(self):
        seqs = [["a", "b", "c"], ["x", "b", "d"], ["a", "y", "c"]]
        outcomes = [True, False, True]
        points = _find_decision_points(seqs, outcomes)
        # Index 0: a vs x, Index 1: b vs y, Index 2: c vs d
        assert len(points) == 3

    def test_outcome_correlation_computed(self):
        seqs = [["a", "b"], ["a", "c"], ["a", "b"], ["a", "c"]]
        outcomes = [True, False, True, False]
        points = _find_decision_points(seqs, outcomes)
        assert len(points) == 1
        assert points[0].outcome_correlation is not None


class TestComputeDivergenceProfile:
    def test_single_trial(self, make_task_trial_data):
        td = make_task_trial_data(
            outcomes=[True],
            action_sequences=[["a", "b"]],
        )
        profile = compute_divergence_profile(td)
        assert profile.task_id == "task_0"
        assert profile.divergence_turn is None

    def test_identical_trials(self, make_task_trial_data):
        td = make_task_trial_data(
            outcomes=[True, True, True],
            action_sequences=[["a", "b", "c"]] * 3,
        )
        profile = compute_divergence_profile(td)
        assert profile.consensus_prefix == ["a", "b", "c"]
        assert profile.divergence_turn is None

    def test_diverging_trials(self, make_task_trial_data):
        td = make_task_trial_data(
            outcomes=[True, False, True],
            action_sequences=[
                ["search", "book", "confirm"],
                ["search", "cancel"],
                ["search", "book", "confirm"],
            ],
        )
        profile = compute_divergence_profile(td)
        assert profile.consensus_prefix == ["search"]
        assert profile.divergence_turn == 1
        assert profile.divergence_type == DivergenceType.TOOL_CHOICE
        assert profile.success_path == ["search", "book", "confirm"]
        assert profile.failure_path == ["search", "cancel"]

    def test_bimodal_task_has_decision_points(self, make_task_trial_data):
        td = make_task_trial_data(
            outcomes=[True, False, True, False],
            action_sequences=[
                ["get", "update", "confirm"],
                ["get", "get", "get"],
                ["get", "update", "confirm"],
                ["get", "cancel"],
            ],
        )
        profile = compute_divergence_profile(td)
        assert len(profile.decision_points) > 0
        # First decision point at index 1 (update vs get vs cancel)
        assert profile.decision_points[0].action_index == 1


class TestComputeAllProfiles:
    def test_multiple_tasks(self, make_task_trial_data):
        td1 = make_task_trial_data(task_id="t1", outcomes=[True, True])
        td2 = make_task_trial_data(task_id="t2", outcomes=[True, False],
                                    action_sequences=[["a", "b"], ["a", "c"]])
        profiles = compute_all_divergence_profiles([td1, td2])
        assert len(profiles) == 2
        assert profiles[0].task_id == "t1"
        assert profiles[1].task_id == "t2"
