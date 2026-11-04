# Copyright Sierra

import pytest
from litellm import ModelResponse

from tau2.utils.llm_utils import get_response_cost


@pytest.mark.parametrize(
    "model",
    [
        "gpt-5.5",
        "gpt-5.5-2026-04-23",
        "openai/gpt-5.5",
        "openai/responses/gpt-5.5-2026-04-23",
    ],
)
@pytest.mark.parametrize("cached,expected", [(0, 0.008), (200, 0.0071)])
def test_gpt_5_5_standard_cost(model, cached, expected):
    response = ModelResponse(
        model=model,
        usage={
            "prompt_tokens": 1000,
            "completion_tokens": 100,
            "prompt_tokens_details": {"cached_tokens": cached},
            "completion_tokens_details": {"reasoning_tokens": 40},
        },
    )
    assert get_response_cost(response) == pytest.approx(expected)


@pytest.mark.parametrize(
    "prompt_tokens,expected", [(272_000, 1.3621), (272_001, 2.72271)]
)
def test_gpt_5_5_long_context_boundary(prompt_tokens, expected):
    response = ModelResponse(
        model="gpt-5.5",
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 100,
            "prompt_tokens_details": {"cached_tokens": 200},
        },
    )
    assert get_response_cost(response) == pytest.approx(expected)


def test_gpt_5_5_usage_without_details():
    response = ModelResponse(
        model="gpt-5.5",
        usage={"prompt_tokens": 1000, "completion_tokens": 100},
    )
    assert get_response_cost(response) == pytest.approx(0.008)


@pytest.mark.parametrize(
    "model,tier",
    [("gpt-5.5-pro", None), ("azure/gpt-5.5", None), ("gpt-5.5", "priority")],
)
def test_other_models_and_tiers_retain_litellm_pricing(monkeypatch, model, tier):
    monkeypatch.setattr("tau2.utils.llm_utils.completion_cost", lambda **kwargs: 0.25)
    response = ModelResponse(
        model=model,
        service_tier=tier,
        usage={"prompt_tokens": 1000, "completion_tokens": 100},
    )
    assert get_response_cost(response) == 0.25
