"""Shared fixtures for tau2-reliability tests.

Builds synthetic SimulationRun and Results objects without any API calls.
"""

from __future__ import annotations

import pytest
from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
from tau2.data_model.simulation import (
    AgentInfo,
    Info,
    Results,
    RewardInfo,
    SimulationRun,
    TerminationReason,
    UserInfo,
)
from tau2.data_model.tasks import Task, UserScenario
from tau2.environment.environment import EnvironmentInfo

from tau2_reliability.models import TaskTrialData


def _make_messages(action_names: list[str]) -> list:
    """Build a realistic message trajectory from action names."""
    messages = []
    messages.append(UserMessage(role="user", content="I need help with my account."))
    for i, action in enumerate(action_names):
        tc = ToolCall(
            id=f"tc_{i}",
            name=action,
            arguments={"id": "test_123"},
            requestor="assistant",
        )
        messages.append(AssistantMessage(role="assistant", content=None, tool_calls=[tc]))
        messages.append(ToolMessage(role="tool", id=f"tc_{i}", content='{"status": "ok"}'))
    messages.append(AssistantMessage(role="assistant", content="Done. Is there anything else?"))
    return messages


@pytest.fixture
def make_sim():
    """Factory fixture for creating synthetic SimulationRun objects."""

    def _make(
        task_id: str = "task_0",
        trial: int = 0,
        reward: float = 1.0,
        action_names: list[str] | None = None,
        cost: float = 0.1,
        duration: float = 30.0,
        seed: int = 42,
        termination: TerminationReason = TerminationReason.AGENT_STOP,
    ) -> SimulationRun:
        if action_names is None:
            action_names = ["get_details", "update_record"]
        return SimulationRun(
            id=f"sim_{task_id}_t{trial}",
            task_id=task_id,
            start_time="2026-01-01T00:00:00",
            end_time="2026-01-01T00:01:00",
            duration=duration,
            termination_reason=termination,
            agent_cost=cost,
            reward_info=RewardInfo(reward=reward),
            messages=_make_messages(action_names),
            trial=trial,
            seed=seed + trial,
        )

    return _make


@pytest.fixture
def make_results(make_sim):
    """Factory fixture for wrapping simulations into a Results object."""

    def _make(
        simulations: list[SimulationRun] | None = None,
        domain: str = "airline",
    ) -> Results:
        if simulations is None:
            simulations = [make_sim()]
        info = Info(
            agent_info=AgentInfo(implementation="llm_agent", llm="gpt-4o", llm_args={}),
            user_info=UserInfo(implementation="user_simulator", llm="gpt-4o", llm_args={}),
            environment_info=EnvironmentInfo(domain_name=domain, policy="test policy"),
            num_trials=max((s.trial or 0) for s in simulations) + 1 if simulations else 1,
            git_commit="test",
            max_steps=30,
            max_errors=5,
        )
        tasks = []
        seen_tasks = set()
        for sim in simulations:
            if sim.task_id not in seen_tasks:
                seen_tasks.add(sim.task_id)
                tasks.append(
                    Task(
                        id=sim.task_id,
                        user_scenario=UserScenario(instructions="Test scenario"),
                    )
                )
        return Results(info=info, tasks=tasks, simulations=simulations)

    return _make


@pytest.fixture
def make_task_trial_data():
    """Factory fixture for creating TaskTrialData directly."""

    def _make(
        task_id: str = "task_0",
        outcomes: list[bool] | None = None,
        action_sequences: list[list[str]] | None = None,
        costs: list[float] | None = None,
        durations: list[float] | None = None,
    ) -> TaskTrialData:
        if outcomes is None:
            outcomes = [True, True, False, True, True]
        n = len(outcomes)
        if action_sequences is None:
            action_sequences = [["search", "book"] for _ in range(n)]
        if costs is None:
            costs = [0.1] * n
        if durations is None:
            durations = [30.0] * n
        return TaskTrialData(
            task_id=task_id,
            outcomes=outcomes,
            action_sequences=action_sequences,
            costs=costs,
            durations=durations,
            num_actions=[len(s) for s in action_sequences],
        )

    return _make
