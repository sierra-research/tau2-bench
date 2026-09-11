"""Tests for hallucination-check error handling in the batch runner.

The hallucination check is a post-hoc reviewer LLM call that runs after a
simulation has already completed. A reviewer failure (e.g. provider
overloaded after all litellm retries) must not crash the whole batch: the
completed simulation is kept and its hallucination check is recorded as
errored instead.
"""

from tau2.data_model.message import Tick
from tau2.data_model.simulation import (
    HallucinationCheck,
    SimulationRun,
    TerminationReason,
    TextRunConfig,
)
from tau2.data_model.tasks import Task
from tau2.runner.batch import run_tasks


def _make_config(hallucination_retries: int = 2) -> TextRunConfig:
    return TextRunConfig(
        domain="mock",
        agent="llm_agent",
        user="user_simulator",
        num_trials=1,
        max_concurrency=1,
        hallucination_retries=hallucination_retries,
        # In-process execution: the monkeypatched check_hallucination must be
        # visible (a worker fleet would run the real one in subprocesses).
        workers=0,
    )


def _make_full_duplex_sim(task: Task) -> SimulationRun:
    return SimulationRun(
        id="sim-test-1",
        task_id=task.id,
        start_time="2026-01-01T00:00:00",
        end_time="2026-01-01T00:01:00",
        duration=60.0,
        termination_reason=TerminationReason.USER_STOP,
        messages=[],
        ticks=[Tick(tick_id=0, timestamp="2026-01-01T00:00:00")],
        trial=0,
    )


def _patch_run_with_retry(monkeypatch, task: Task) -> None:
    def fake_run_with_retry(run_fn, *args, **kwargs) -> SimulationRun:
        return _make_full_duplex_sim(task)

    monkeypatch.setattr("tau2.runner.batch.run_with_retry", fake_run_with_retry)


def test_reviewer_exception_does_not_fail_batch(monkeypatch, base_task: Task):
    """A reviewer LLM failure in check_hallucination must not crash the batch."""
    _patch_run_with_retry(monkeypatch, base_task)

    def failing_check(simulation, task):
        raise RuntimeError("AnthropicError overloaded_error")

    monkeypatch.setattr("tau2.runner.batch.check_hallucination", failing_check)

    results = run_tasks(
        config=_make_config(),
        tasks=[base_task],
        console_display=False,
    )

    assert len(results.simulations) == 1
    sim = results.simulations[0]
    check = sim.hallucination_check
    assert check is not None
    assert check.check_error is not None
    assert "overloaded_error" in check.check_error
    assert check.hallucination_found is False
    assert sim.hallucination_retries_used == 0


def test_clean_hallucination_check_still_recorded(monkeypatch, base_task: Task):
    """Sanity check: a successful check is recorded unchanged (check_error unset)."""
    _patch_run_with_retry(monkeypatch, base_task)

    clean_check = HallucinationCheck(
        reasoning="No fabricated information.",
        hallucination_found=False,
        summary="Clean.",
    )
    monkeypatch.setattr(
        "tau2.runner.batch.check_hallucination",
        lambda simulation, task: clean_check,
    )

    results = run_tasks(
        config=_make_config(),
        tasks=[base_task],
        console_display=False,
    )

    assert len(results.simulations) == 1
    check = results.simulations[0].hallucination_check
    assert check is not None
    assert check.check_error is None
    assert check.summary == "Clean."
