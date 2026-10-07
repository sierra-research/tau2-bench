"""Regression tests for evaluate_simulation's graceful-degradation guard (issue #387).

Replay-based grading can raise when an environment tool is non-deterministic
(e.g. a dispute status that advances with wall-clock time), and that exception
used to propagate out of ``evaluate_simulation`` and abort an entire batch run.
These tests pin the contract of the guard without needing the retrieval/LLM
stack: the inner implementation is monkeypatched to raise or return, and the
public wrapper's behavior is asserted.
"""

from types import SimpleNamespace

import pytest

from tau2.data_model.simulation import RewardInfo
from tau2.evaluator import evaluator as evaluator_module
from tau2.evaluator.evaluator import (
    EvaluationCriteriaError,
    EvaluationType,
    evaluate_simulation,
)


def _call():
    return evaluate_simulation(
        simulation=None,
        task=SimpleNamespace(id="test_task", evaluation_criteria=None),
        evaluation_type=EvaluationType.ALL,
        solo_mode=False,
        domain="mock",
    )


def test_unexpected_error_is_graded_as_failure(monkeypatch):
    """A raising evaluator yields reward=0.0 with the error recorded, not a crash."""

    def boom(**kwargs):
        # Mimics environment.set_state diverging on a non-deterministic tool
        # during replay-based grading (issue #387).
        raise ValueError("Returned RESOLVED, expected SUBMITTED")

    monkeypatch.setattr(evaluator_module, "_evaluate_simulation", boom)

    reward_info = _call()

    assert isinstance(reward_info, RewardInfo)
    assert reward_info.reward == 0.0
    assert reward_info.info is not None
    assert "ValueError" in reward_info.info["error"]


def test_configuration_error_is_reraised(monkeypatch):
    """A genuine config error must still surface loudly, not be swallowed."""

    def bad_config(**kwargs):
        raise EvaluationCriteriaError("reward_basis includes {RewardType.DB}")

    monkeypatch.setattr(evaluator_module, "_evaluate_simulation", bad_config)

    with pytest.raises(EvaluationCriteriaError):
        _call()


def test_successful_result_passes_through(monkeypatch):
    """The guard is transparent when the inner evaluation succeeds."""
    sentinel = RewardInfo(reward=1.0)

    monkeypatch.setattr(
        evaluator_module, "_evaluate_simulation", lambda **kwargs: sentinel
    )

    assert _call() is sentinel
