"""Tests for 3-dimension AVER metrics computation."""

import pytest

from tau_robustness.metrics import (
    WEIGHT_DETECTION,
    WEIGHT_DIAGNOSIS,
    WEIGHT_RECOVERY,
    AggregateRobustness,
    RobustnessMetrics,
    _compute_pass_hat_k,
    compute_robustness_metrics,
)


class TestRobustnessMetrics:
    def test_compute_aver(self):
        m = RobustnessMetrics(
            num_injections=1,
            detection_score=0.8,
            diagnosis_score=0.5,
            recovery_score=0.6,
        )
        m.compute_aver()
        expected = (0.8 * 0.4 + 0.5 * 0.2 + 0.6 * 0.4) * 100
        assert abs(m.aver_score - round(expected, 1)) < 0.1

    def test_compute_aver_no_detection(self):
        m = RobustnessMetrics(
            num_injections=0,
            detection_score=None,
            diagnosis_score=None,
            recovery_score=1.0,
        )
        m.compute_aver()
        assert m.aver_score is None

    def test_compute_aver_perfect(self):
        m = RobustnessMetrics(
            num_injections=1,
            detection_score=1.0,
            diagnosis_score=1.0,
            recovery_score=1.0,
        )
        m.compute_aver()
        assert m.aver_score == 100.0

    def test_compute_aver_zero(self):
        m = RobustnessMetrics(
            num_injections=1,
            detection_score=0.0,
            diagnosis_score=0.0,
            recovery_score=0.0,
        )
        m.compute_aver()
        assert m.aver_score == 0.0

    def test_negative_control_flag(self):
        m = RobustnessMetrics(
            num_injections=0,
            recovery_score=1.0,
            is_negative_control=True,
        )
        assert m.is_negative_control is True

    def test_temporal_pattern(self):
        m = RobustnessMetrics(
            num_injections=1,
            detection_score=0.8,
            diagnosis_score=0.5,
            recovery_score=0.6,
            temporal_pattern="proactive",
        )
        assert m.temporal_pattern == "proactive"


class TestPassHatK:
    def test_all_pass(self):
        task_trials = {"t1": [1.0, 1.0, 1.0, 1.0]}
        assert _compute_pass_hat_k(task_trials, k=1) == 1.0
        assert _compute_pass_hat_k(task_trials, k=4) == 1.0

    def test_all_fail(self):
        task_trials = {"t1": [0.0, 0.0, 0.0, 0.0]}
        assert _compute_pass_hat_k(task_trials, k=1) == 0.0

    def test_half_pass(self):
        task_trials = {"t1": [1.0, 0.0, 1.0, 0.0]}
        assert abs(_compute_pass_hat_k(task_trials, k=1) - 0.5) < 1e-6

    def test_k_exceeds_trials(self):
        task_trials = {"t1": [1.0, 1.0]}
        assert _compute_pass_hat_k(task_trials, k=4) == 0.0

    def test_empty_tasks(self):
        assert _compute_pass_hat_k({}, k=1) == 0.0

    def test_multiple_tasks(self):
        task_trials = {
            "t1": [1.0, 1.0, 0.0, 0.0],
            "t2": [1.0, 1.0, 1.0, 1.0],
        }
        assert abs(_compute_pass_hat_k(task_trials, k=1) - 0.75) < 1e-6


class TestComputeRobustnessMetrics:
    def test_basic_aggregate(self):
        per_task = {
            "t1": [
                RobustnessMetrics(
                    num_injections=1,
                    detection_score=0.8,
                    diagnosis_score=0.5,
                    recovery_score=0.5,
                    temporal_pattern="proactive",
                ),
                RobustnessMetrics(
                    num_injections=1,
                    detection_score=0.4,
                    diagnosis_score=0.25,
                    recovery_score=0.0,
                    temporal_pattern="reactive",
                ),
            ],
        }

        result = compute_robustness_metrics(per_task, ks=[1, 2])
        assert result.num_tasks == 1
        assert result.num_trials == 2
        assert result.total_injections == 2
        assert abs(result.avg_detection - 0.6) < 1e-6
        assert abs(result.avg_diagnosis - 0.375) < 1e-6
        assert abs(result.avg_recovery - 0.25) < 1e-6

    def test_aver_score_formula(self):
        per_task = {
            "t1": [
                RobustnessMetrics(
                    num_injections=1,
                    detection_score=1.0,
                    diagnosis_score=1.0,
                    recovery_score=1.0,
                ),
            ],
        }
        result = compute_robustness_metrics(per_task, ks=[1])
        assert result.aver_score == 100.0

    def test_aver_score_zero(self):
        per_task = {
            "t1": [
                RobustnessMetrics(
                    num_injections=1,
                    detection_score=0.0,
                    diagnosis_score=0.0,
                    recovery_score=0.0,
                ),
            ],
        }
        result = compute_robustness_metrics(per_task, ks=[1])
        assert result.aver_score == 0.0

    def test_no_injections(self):
        per_task = {
            "t1": [
                RobustnessMetrics(
                    num_injections=0,
                    detection_score=None,
                    diagnosis_score=None,
                    recovery_score=1.0,
                ),
            ],
        }
        result = compute_robustness_metrics(per_task, ks=[1])
        assert result.total_injections == 0
        assert result.avg_detection == 0.0

    def test_negative_controls_excluded_from_averages(self):
        per_task = {
            "t1": [
                RobustnessMetrics(
                    num_injections=1,
                    detection_score=0.8,
                    diagnosis_score=0.5,
                    recovery_score=0.6,
                ),
                RobustnessMetrics(
                    num_injections=0,
                    detection_score=0.3,
                    diagnosis_score=None,
                    recovery_score=1.0,
                    is_negative_control=True,
                ),
            ],
        }
        result = compute_robustness_metrics(per_task, ks=[1])
        # Only the injected run should count
        assert result.total_injections == 1
        assert abs(result.avg_detection - 0.8) < 1e-6

    def test_false_positive_rate(self):
        per_task = {
            "t1": [
                RobustnessMetrics(
                    num_injections=0,
                    detection_score=0.5,
                    recovery_score=1.0,
                    is_negative_control=True,
                ),
                RobustnessMetrics(
                    num_injections=0,
                    detection_score=0.0,
                    recovery_score=1.0,
                    is_negative_control=True,
                ),
            ],
        }
        result = compute_robustness_metrics(per_task, ks=[1])
        assert result.false_positive_rate == 0.5  # 1 out of 2

    def test_temporal_breakdown(self):
        per_task = {
            "t1": [
                RobustnessMetrics(
                    num_injections=1,
                    detection_score=0.8,
                    diagnosis_score=0.5,
                    recovery_score=0.5,
                    temporal_pattern="proactive",
                ),
                RobustnessMetrics(
                    num_injections=1,
                    detection_score=0.4,
                    diagnosis_score=0.25,
                    recovery_score=0.0,
                    temporal_pattern="proactive",
                ),
                RobustnessMetrics(
                    num_injections=1,
                    detection_score=0.2,
                    diagnosis_score=0.0,
                    recovery_score=0.0,
                    temporal_pattern="reactive",
                ),
            ],
        }
        result = compute_robustness_metrics(per_task, ks=[1])
        assert result.temporal_breakdown["proactive"] == pytest.approx(2 / 3, abs=0.01)
        assert result.temporal_breakdown["reactive"] == pytest.approx(1 / 3, abs=0.01)
