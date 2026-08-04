"""Tests for cost/time breakdown metrics in tau2.metrics.agent_metrics."""

import pytest

from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage
from tau2.data_model.simulation import (
    Info,
    Results,
    RewardInfo,
    SimulationRun,
    TerminationReason,
    UserInfo,
)
from tau2.data_model.tasks import EvaluationCriteria, Task, UserScenario
from tau2.environment.environment import EnvironmentInfo
from tau2.metrics.agent_metrics import compute_metrics, compute_sim_time_breakdown


def _ts(seconds: float) -> str:
    """ISO timestamp `seconds` after a fixed epoch."""
    whole = int(seconds)
    frac = int(round((seconds - whole) * 1e6))
    return f"2026-01-01T00:{whole // 60:02d}:{whole % 60:02d}.{frac:06d}"


def _make_messages():
    """A conversation with known gap attribution.

    t=0   assistant greeting (first message: excluded from attribution)
    t=10  user reply           -> 10s user
    t=13  assistant tool call  -> 3s agent
    t=15  tool result          -> 2s tool
    t=21  assistant reply      -> 6s agent
    t=30  user reply           -> 9s user
    """
    return [
        AssistantMessage(role="assistant", content="Hi!", timestamp=_ts(0)),
        UserMessage(role="user", content="I need help.", timestamp=_ts(10)),
        AssistantMessage(role="assistant", content=None, timestamp=_ts(13)),
        ToolMessage(id="tc1", role="tool", content="ok", timestamp=_ts(15)),
        AssistantMessage(role="assistant", content="Done.", timestamp=_ts(21)),
        UserMessage(role="user", content="Thanks!", timestamp=_ts(30)),
    ]


def _make_sim(task_id="t0", trial=0, **kwargs) -> SimulationRun:
    defaults = dict(
        id=f"sim-{task_id}-{trial}",
        task_id=task_id,
        start_time="2026-01-01T00:00:00",
        end_time="2026-01-01T00:01:00",
        duration=60.0,
        termination_reason=TerminationReason.USER_STOP,
        messages=[],
        trial=trial,
        reward_info=RewardInfo(reward=1.0),
    )
    defaults.update(kwargs)
    return SimulationRun(**defaults)


def _make_results(sims) -> Results:
    task_ids = sorted({s.task_id for s in sims})
    return Results(
        info=Info(
            git_commit="abc123",
            num_trials=1,
            max_steps=100,
            max_errors=10,
            user_info=UserInfo(implementation="user_simulator"),
            agent_info={"implementation": "llm_agent"},
            environment_info=EnvironmentInfo(domain_name="mock", policy="p"),
        ),
        tasks=[
            Task(
                id=tid,
                user_scenario=UserScenario(instructions="i"),
                evaluation_criteria=EvaluationCriteria(),
            )
            for tid in task_ids
        ],
        simulations=sims,
    )


class TestSimTimeBreakdown:
    def test_gap_attribution(self):
        sim = _make_sim(messages=_make_messages())
        breakdown = compute_sim_time_breakdown(sim)
        assert breakdown is not None
        assert breakdown["agent_time"] == pytest.approx(9.0)  # 3 + 6
        assert breakdown["user_time"] == pytest.approx(19.0)  # 10 + 9
        assert breakdown["tool_time"] == pytest.approx(2.0)

    def test_no_messages_returns_none(self):
        assert compute_sim_time_breakdown(_make_sim(messages=[])) is None
        assert compute_sim_time_breakdown(_make_sim(messages=None)) is None

    def test_single_message_returns_none(self):
        msgs = [AssistantMessage(role="assistant", content="Hi!", timestamp=_ts(0))]
        assert compute_sim_time_breakdown(_make_sim(messages=msgs)) is None

    def test_missing_timestamp_returns_none(self):
        msgs = _make_messages()
        msgs[2].timestamp = None
        assert compute_sim_time_breakdown(_make_sim(messages=msgs)) is None

    def test_negative_gap_clamped_to_zero(self):
        msgs = [
            AssistantMessage(role="assistant", content="Hi!", timestamp=_ts(10)),
            UserMessage(role="user", content="Hello.", timestamp=_ts(5)),
        ]
        breakdown = compute_sim_time_breakdown(_make_sim(messages=msgs))
        assert breakdown["user_time"] == 0.0


class TestComputeMetricsBreakdown:
    def test_cost_and_time_aggregation(self):
        sims = [
            _make_sim(
                task_id="t0",
                messages=_make_messages(),
                agent_cost=0.10,
                user_cost=0.04,
                duration=30.0,
            ),
            _make_sim(
                task_id="t1",
                messages=_make_messages(),
                agent_cost=0.20,
                user_cost=0.06,
                duration=50.0,
            ),
        ]
        metrics = compute_metrics(_make_results(sims))
        assert metrics.avg_agent_cost == pytest.approx(0.15)
        assert metrics.avg_user_cost == pytest.approx(0.05)
        assert metrics.avg_total_cost == pytest.approx(0.20)
        assert metrics.avg_duration == pytest.approx(40.0)
        assert metrics.avg_agent_time == pytest.approx(9.0)
        assert metrics.avg_user_time == pytest.approx(19.0)
        assert metrics.avg_tool_time == pytest.approx(2.0)

    def test_missing_costs_and_timestamps_yield_none(self):
        # No costs recorded, no messages: breakdown fields must be None,
        # duration still averages (it's a required field).
        sims = [_make_sim(task_id="t0", duration=20.0)]
        metrics = compute_metrics(_make_results(sims))
        assert metrics.avg_user_cost is None
        assert metrics.avg_total_cost is None
        assert metrics.avg_agent_time is None
        assert metrics.avg_user_time is None
        assert metrics.avg_tool_time is None
        assert metrics.avg_duration == pytest.approx(20.0)

    def test_partial_coverage_averages_available_sims(self):
        # One sim with costs+messages, one without: averages use what exists.
        sims = [
            _make_sim(
                task_id="t0",
                messages=_make_messages(),
                agent_cost=0.10,
                user_cost=0.04,
                duration=30.0,
            ),
            _make_sim(task_id="t1", duration=50.0),
        ]
        metrics = compute_metrics(_make_results(sims))
        assert metrics.avg_user_cost == pytest.approx(0.04)
        assert metrics.avg_total_cost == pytest.approx(0.14)
        assert metrics.avg_agent_time == pytest.approx(9.0)
        assert metrics.avg_duration == pytest.approx(40.0)
