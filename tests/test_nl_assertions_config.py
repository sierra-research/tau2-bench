"""Tests for configuring the natural-language assertion judge."""

import json
from types import SimpleNamespace

import pytest

from tau2.config import DEFAULT_LLM_NL_ASSERTIONS, DEFAULT_LLM_NL_ASSERTIONS_ARGS
from tau2.data_model.message import UserMessage
from tau2.evaluator import evaluator_nl_assertions
from tau2.evaluator.evaluator_nl_assertions import (
    NLAssertionsEvaluator,
    _get_nl_assertions_llm_config,
)


def test_nl_assertion_judge_uses_official_defaults(monkeypatch):
    monkeypatch.delenv("TAU2_NL_ASSERTIONS_MODEL", raising=False)
    monkeypatch.delenv("TAU2_NL_ASSERTIONS_ARGS", raising=False)

    model, args = _get_nl_assertions_llm_config()

    assert model == DEFAULT_LLM_NL_ASSERTIONS
    assert args == DEFAULT_LLM_NL_ASSERTIONS_ARGS
    assert args is not DEFAULT_LLM_NL_ASSERTIONS_ARGS


def test_nl_assertion_judge_accepts_openrouter_override(monkeypatch):
    monkeypatch.setenv(
        "TAU2_NL_ASSERTIONS_MODEL", "openrouter/openai/gpt-4.1-2025-04-14"
    )
    monkeypatch.setenv(
        "TAU2_NL_ASSERTIONS_ARGS",
        '{"temperature":0,"extra_body":{"provider":'
        '{"order":["OpenAI"],"allow_fallbacks":false}}}',
    )

    model, args = _get_nl_assertions_llm_config()

    assert model == "openrouter/openai/gpt-4.1-2025-04-14"
    assert args["temperature"] == 0
    assert args["extra_body"]["provider"]["order"] == ["OpenAI"]
    assert args["extra_body"]["provider"]["allow_fallbacks"] is False


@pytest.mark.parametrize("value", ["not-json", "[]", "null"])
def test_nl_assertion_judge_rejects_invalid_args(monkeypatch, value):
    monkeypatch.setenv("TAU2_NL_ASSERTIONS_ARGS", value)

    with pytest.raises(ValueError, match="TAU2_NL_ASSERTIONS_ARGS"):
        _get_nl_assertions_llm_config()


def test_nl_assertion_judge_retains_route_provenance(monkeypatch):
    monkeypatch.delenv("TAU2_NL_ASSERTIONS_MODEL", raising=False)
    monkeypatch.delenv("TAU2_NL_ASSERTIONS_ARGS", raising=False)
    response = {
        "results": [
            {
                "expectedOutcome": "The agent explains the fee.",
                "reasoning": "The fee was explained.",
                "metExpectation": True,
            }
        ]
    }
    monkeypatch.setattr(
        evaluator_nl_assertions,
        "generate",
        lambda **kwargs: SimpleNamespace(
            content=json.dumps(response),
            raw_data={
                "id": "generation-id",
                "model": "openai/gpt-4.1-2025-04-14",
                "provider": "OpenAI",
                "service_tier": "default",
            },
        ),
    )

    checks, provenance = NLAssertionsEvaluator._evaluate_nl_assertions_with_provenance(
        [UserMessage(role="user", content="What is the fee?")],
        ["The agent explains the fee."],
    )

    assert checks[0].met is True
    assert provenance == {
        "requested_model": DEFAULT_LLM_NL_ASSERTIONS,
        "resolved_model": "openai/gpt-4.1-2025-04-14",
        "provider": "OpenAI",
        "service_tier": "default",
        "response_id": "generation-id",
    }
