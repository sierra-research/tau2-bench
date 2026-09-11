# Copyright Sierra
"""get_response_cost returns None (unknown) when a model isn't in litellm's
cost map.

An unmapped model (e.g. a freshly released or vendor-prefixed id) must not
flood logs with ERROR on every call nor masquerade as a real $0.00 cost in
sums/averages — it returns None and logs a WARNING once per model. Genuinely
unexpected failures still log at ERROR (and the cost is None: unknown).
"""

from types import SimpleNamespace

import pytest

import tau2.utils.llm_utils as llm_utils


def _response(model: str) -> SimpleNamespace:
    return SimpleNamespace(model=model)


def test_unmapped_model_returns_none_and_warns_once(monkeypatch):
    def boom(completion_response):
        raise Exception("This model isn't mapped yet. model=us/claude-opus-4-8")

    monkeypatch.setattr(llm_utils, "completion_cost", boom)
    llm_utils._UNMAPPED_COST_MODELS_SEEN.clear()
    errors: list[object] = []
    warnings: list[str] = []
    monkeypatch.setattr(llm_utils.logger, "error", lambda m: errors.append(m))
    monkeypatch.setattr(llm_utils.logger, "warning", lambda m: warnings.append(m))

    assert llm_utils.get_response_cost(_response("us/claude-opus-4-8")) is None
    assert llm_utils.get_response_cost(_response("us/claude-opus-4-8")) is None
    # No ERROR spam; the once-per-model WARNING fired exactly once.
    assert errors == []
    assert len(warnings) == 1


def test_real_cost_for_mapped_model(monkeypatch):
    monkeypatch.setattr(llm_utils, "completion_cost", lambda completion_response: 0.42)
    assert llm_utils.get_response_cost(_response("gpt-4.1-mini")) == pytest.approx(0.42)


def test_unexpected_error_logs_error_and_returns_none(monkeypatch):
    def boom(completion_response):
        raise ValueError("totally unexpected")

    monkeypatch.setattr(llm_utils, "completion_cost", boom)
    errors: list[object] = []
    monkeypatch.setattr(llm_utils.logger, "error", lambda m: errors.append(m))
    assert llm_utils.get_response_cost(_response("gpt-4.1-mini")) is None
    assert len(errors) == 1
