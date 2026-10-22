"""Tests for run_with_retry termination-reason classification."""

import litellm

from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.data_model.tasks import EvaluationCriteria, Task, UserScenario
from tau2.runner.progress import run_with_retry


def _make_task(task_id: str = "t0") -> Task:
    return Task(
        id=task_id,
        user_scenario=UserScenario(instructions="test instruction"),
        evaluation_criteria=EvaluationCriteria(),
    )


def _context_window_error() -> litellm.ContextWindowExceededError:
    return litellm.ContextWindowExceededError(
        message="This model's maximum context length is 8192 tokens.",
        model="gpt-4o",
        llm_provider="openai",
    )


def test_context_window_error_is_not_retried_and_labeled():
    """A context-window error is deterministic, so it should not be retried and
    should be surfaced as CONTEXT_WINDOW_EXCEEDED rather than INFRASTRUCTURE_ERROR."""
    calls = {"n": 0}

    def run_fn() -> SimulationRun:
        calls["n"] += 1
        raise _context_window_error()

    sim = run_with_retry(
        run_fn,
        _make_task(),
        trial=0,
        seed=42,
        max_retries=3,
        retry_delay=0.0,
        console_display=False,
    )

    assert sim.termination_reason == TerminationReason.CONTEXT_WINDOW_EXCEEDED
    assert calls["n"] == 1  # deterministic error: stopped after the first attempt
    assert sim.info["error_type"] == "ContextWindowExceededError"
    assert sim.info["failed_after_attempts"] == 1


def test_generic_error_is_retried_and_labeled_infrastructure():
    """Unchanged behavior: a generic error is retried up to max_attempts and then
    labeled INFRASTRUCTURE_ERROR."""
    calls = {"n": 0}

    def run_fn() -> SimulationRun:
        calls["n"] += 1
        raise RuntimeError("transient boom")

    sim = run_with_retry(
        run_fn,
        _make_task(),
        trial=0,
        seed=42,
        max_retries=2,
        retry_delay=0.0,
        console_display=False,
    )

    assert sim.termination_reason == TerminationReason.INFRASTRUCTURE_ERROR
    assert calls["n"] == 3  # initial attempt + 2 retries
    assert sim.info["failed_after_attempts"] == 3
