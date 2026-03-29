"""Tests for robustness metrics and prompt variation."""

import pytest

from tau2_reliability.metrics.robustness import (
    bootstrap_robustness_ratio,
    compute_robustness_ratio,
)


class TestRobustnessRatio:
    def test_equal_accuracy(self):
        assert compute_robustness_ratio(0.8, 0.8) == pytest.approx(1.0)

    def test_half_accuracy(self):
        assert compute_robustness_ratio(0.8, 0.4) == pytest.approx(0.5)

    def test_capped_at_one(self):
        assert compute_robustness_ratio(0.5, 0.8) == pytest.approx(1.0)

    def test_zero_baseline(self):
        assert compute_robustness_ratio(0.0, 0.5) == pytest.approx(1.0)

    def test_zero_both(self):
        assert compute_robustness_ratio(0.0, 0.0) == pytest.approx(1.0)

    def test_full_degradation(self):
        assert compute_robustness_ratio(0.8, 0.0) == pytest.approx(0.0)

    def test_bounded(self):
        for base in [0.0, 0.1, 0.5, 0.9, 1.0]:
            for pert in [0.0, 0.1, 0.5, 0.9, 1.0]:
                r = compute_robustness_ratio(base, pert)
                assert 0.0 <= r <= 1.0


class TestBootstrapRobustness:
    def test_basic(self):
        baseline = [True, True, True, False, False]
        perturbed = [True, True, False, False, False]
        result = bootstrap_robustness_ratio(baseline, perturbed)
        assert 0.0 <= result.point_estimate <= 1.0
        assert result.standard_error >= 0.0
        assert result.ci_lower <= result.point_estimate <= result.ci_upper

    def test_identical_outcomes(self):
        outcomes = [True, True, False, False]
        result = bootstrap_robustness_ratio(outcomes, outcomes)
        assert result.point_estimate == pytest.approx(1.0)

    def test_empty_outcomes(self):
        result = bootstrap_robustness_ratio([], [])
        assert result.point_estimate == 1.0
        assert result.standard_error == 0.0
