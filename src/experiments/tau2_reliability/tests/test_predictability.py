"""Tests for predictability metrics."""

import math

import pytest

from tau2_reliability.metrics.predictability import (
    compute_calibration_bins,
    compute_p_auroc,
    compute_p_brier,
    compute_p_cal,
)


class TestPCal:
    def test_perfect_calibration(self):
        # Confidence matches outcome exactly
        confs = [1.0, 1.0, 0.0, 0.0, 1.0]
        outs = [True, True, False, False, True]
        assert compute_p_cal(confs, outs) == pytest.approx(1.0, abs=0.01)

    def test_worst_calibration(self):
        # Always confident but always wrong
        confs = [0.95, 0.95, 0.95, 0.95]
        outs = [False, False, False, False]
        p_cal = compute_p_cal(confs, outs)
        assert p_cal < 0.15

    def test_empty_input(self):
        assert math.isnan(compute_p_cal([], []))

    def test_mismatched_lengths(self):
        assert math.isnan(compute_p_cal([0.5, 0.5], [True]))

    def test_bounded(self):
        import random
        random.seed(42)
        confs = [random.random() for _ in range(100)]
        outs = [random.choice([True, False]) for _ in range(100)]
        p_cal = compute_p_cal(confs, outs)
        assert 0.0 <= p_cal <= 1.0


class TestPAuroc:
    def test_perfect_discrimination(self):
        # All successes have higher confidence than all failures
        confs = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
        outs = [True, True, True, False, False, False]
        assert compute_p_auroc(confs, outs) == pytest.approx(1.0, abs=0.01)

    def test_anti_discrimination(self):
        # All failures have higher confidence (inverted)
        confs = [0.1, 0.2, 0.9, 0.8]
        outs = [True, True, False, False]
        assert compute_p_auroc(confs, outs) == pytest.approx(0.0, abs=0.01)

    def test_random_discrimination(self):
        # Interleaved — should be near 0.5
        confs = [0.5, 0.5, 0.5, 0.5]
        outs = [True, False, True, False]
        assert compute_p_auroc(confs, outs) == pytest.approx(0.5, abs=0.01)

    def test_single_class_nan(self):
        assert math.isnan(compute_p_auroc([0.8, 0.9], [True, True]))
        assert math.isnan(compute_p_auroc([0.3, 0.2], [False, False]))

    def test_empty_nan(self):
        assert math.isnan(compute_p_auroc([], []))


class TestPBrier:
    def test_perfect_predictions(self):
        confs = [1.0, 0.0, 1.0, 0.0]
        outs = [True, False, True, False]
        assert compute_p_brier(confs, outs) == pytest.approx(1.0, abs=0.01)

    def test_worst_predictions(self):
        confs = [1.0, 1.0, 0.0, 0.0]
        outs = [False, False, True, True]
        assert compute_p_brier(confs, outs) == pytest.approx(0.0, abs=0.01)

    def test_half_confidence(self):
        confs = [0.5, 0.5, 0.5, 0.5]
        outs = [True, False, True, False]
        assert compute_p_brier(confs, outs) == pytest.approx(0.75, abs=0.01)

    def test_empty_nan(self):
        assert math.isnan(compute_p_brier([], []))

    def test_bounded(self):
        import random
        random.seed(42)
        confs = [random.random() for _ in range(50)]
        outs = [random.choice([True, False]) for _ in range(50)]
        p_brier = compute_p_brier(confs, outs)
        assert 0.0 <= p_brier <= 1.0


class TestCalibrationBins:
    def test_returns_correct_bins(self):
        confs = [0.05, 0.15, 0.95]
        outs = [False, True, True]
        bins = compute_calibration_bins(confs, outs, n_bins=10)
        assert len(bins) == 10
        assert bins[0]["count"] == 1  # 0.05 in bin 0
        assert bins[1]["count"] == 1  # 0.15 in bin 1
        assert bins[9]["count"] == 1  # 0.95 in bin 9

    def test_empty_returns_empty(self):
        assert compute_calibration_bins([], []) == []
