"""Tests for Pydantic data models."""

import json

import pytest

from tau2_reliability.models import (
    BootstrapResult,
    ConsistencyMetrics,
    ReliabilityReport,
    TaskReliabilityClass,
    TaskTrialData,
)


class TestTaskTrialData:
    def test_num_trials(self):
        td = TaskTrialData(
            task_id="t1", outcomes=[True, False, True],
            action_sequences=[[], [], []], costs=[0.1, 0.2, 0.1],
            durations=[30, 40, 35], num_actions=[0, 0, 0],
        )
        assert td.num_trials == 3

    def test_pass_rate(self):
        td = TaskTrialData(
            task_id="t1", outcomes=[True, False, True, True],
            action_sequences=[[] for _ in range(4)],
            costs=[0] * 4, durations=[0] * 4, num_actions=[0] * 4,
        )
        assert td.pass_rate == pytest.approx(0.75)

    def test_pass_rate_empty(self):
        td = TaskTrialData(
            task_id="t1", outcomes=[],
            action_sequences=[], costs=[], durations=[], num_actions=[],
        )
        assert td.pass_rate == 0.0


class TestConsistencyMetrics:
    def test_serialization_roundtrip(self):
        cm = ConsistencyMetrics(
            c_out=0.8, c_traj_d=0.7, c_traj_s=0.6, c_res=0.9, r_con=0.75,
            per_task={"t1": {"c_out": 0.8}},
        )
        data = json.loads(cm.model_dump_json())
        cm2 = ConsistencyMetrics.model_validate(data)
        assert cm2.c_out == cm.c_out
        assert cm2.per_task == cm.per_task


class TestReliabilityReport:
    def test_compute_overall_all_present(self):
        report = ReliabilityReport(r_con=0.8, r_pred=0.7, r_rob=0.6)
        r = report.compute_overall()
        assert r == pytest.approx(0.7, abs=0.01)

    def test_compute_overall_partial(self):
        report = ReliabilityReport(r_con=0.8, r_pred=None, r_rob=None)
        r = report.compute_overall()
        assert r == pytest.approx(0.8, abs=0.01)

    def test_compute_overall_none(self):
        report = ReliabilityReport()
        assert report.compute_overall() is None

    def test_compute_overall_ignores_nan(self):
        report = ReliabilityReport(r_con=0.8, r_pred=float("nan"), r_rob=0.6)
        r = report.compute_overall()
        assert r == pytest.approx(0.7, abs=0.01)

    def test_serialization(self):
        report = ReliabilityReport(
            domain="airline", agent_model="gpt-4o",
            num_tasks=26, num_trials=5, accuracy=0.65,
            r_con=0.71, bootstrap_se={"c_out": 0.03},
        )
        data = json.loads(report.model_dump_json())
        report2 = ReliabilityReport.model_validate(data)
        assert report2.domain == "airline"
        assert report2.r_con == pytest.approx(0.71)
        assert report2.bootstrap_se["c_out"] == pytest.approx(0.03)


class TestBootstrapResult:
    def test_ci_ordering(self):
        br = BootstrapResult(
            point_estimate=0.75, standard_error=0.05,
            ci_lower=0.65, ci_upper=0.85,
        )
        assert br.ci_lower < br.point_estimate < br.ci_upper


class TestEnums:
    def test_task_reliability_class_values(self):
        assert TaskReliabilityClass.BIMODAL.value == "bimodal"
        assert TaskReliabilityClass.STABLE_PASS.value == "stable_pass"
