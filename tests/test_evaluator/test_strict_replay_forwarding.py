"""Regression tests for ``strict_replay`` forwarding in ``evaluate_simulation``.

A lenient replay is the documented mode for re-grading historical
trajectories (``docs/evaluation.md``), so ``strict_replay=False`` must reach
``Environment.set_state`` on every evaluation type. The ignore-basis branches
used to drop the argument and fall back to the strict default, aborting a
re-grade that the ``all`` branch completes.
"""

import json
from pathlib import Path

import pytest

from tau2.data_model.message import (
    AssistantMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.semantic_communication import CommunicateScoring
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.data_model.tasks import Task
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation

MOCK_DATA_DIR = Path(__file__).resolve().parents[2] / "data/tau2/domains/mock"


def _mock_task() -> Task:
    tasks = json.loads((MOCK_DATA_DIR / "tasks.json").read_text())
    return Task.model_validate(next(t for t in tasks if t["id"] == "create_task_1"))


def _drifted_simulation(task_id: str) -> SimulationRun:
    """Trajectory whose recorded tool output cosmetically predates current code
    (``description`` was recorded as the title instead of ``null``)."""
    return SimulationRun(
        id="sim-drift",
        task_id=task_id,
        start_time="2026-01-01T00:00:00",
        end_time="2026-01-01T00:01:00",
        duration=60.0,
        termination_reason=TerminationReason.USER_STOP,
        messages=[
            UserMessage(id="1", role="user", content="Create a task for user_1"),
            AssistantMessage(
                id="2",
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(
                        id="3",
                        name="create_task",
                        arguments={"user_id": "user_1", "title": "Important Meeting"},
                    )
                ],
            ),
            ToolMessage(
                id="3",
                role="tool",
                content='{"task_id": "task_2", "title": "Important Meeting", '
                '"description": "Important Meeting", "status": "pending"}',
            ),
            AssistantMessage(id="4", role="assistant", content="Done."),
        ],
    )


@pytest.mark.parametrize(
    "evaluation_type",
    [EvaluationType.ALL, EvaluationType.ALL_IGNORE_BASIS],
)
def test_lenient_replay_reaches_environment_evaluator(evaluation_type):
    task = _mock_task()
    info = evaluate_simulation(
        simulation=_drifted_simulation(task.id),
        task=task,
        evaluation_type=evaluation_type,
        solo_mode=False,
        domain="mock",
        strict_replay=False,
        communicate_scoring=CommunicateScoring(),
    )
    assert info.reward == 1.0
