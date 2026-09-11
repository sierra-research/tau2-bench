"""The adapter consumes the resolved reasoning effort; it never re-derives it.

Re-deriving inside ``create_adapter`` is what left every results.json claiming
``reasoning_effort: null`` while gemini connected at ``thinking_level=HIGH``.
"""

import pytest

from tau2.agent.discrete_time_audio_native_agent import DiscreteTimeAudioNativeAgent
from tau2.config import ReasoningEffort
from tau2.voice.audio_native.adapter import create_adapter


@pytest.fixture(autouse=True)
def _fake_keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")


def test_create_adapter_refuses_an_unresolved_effort():
    with pytest.raises(ValueError, match="already-resolved reasoning_effort"):
        create_adapter(provider="openai", tick_duration_ms=1000, reasoning_effort=None)


def test_provider_default_sends_nothing():
    """'provider_default' is not a level: nothing goes on the wire."""
    adapter, _ = create_adapter(
        provider="openai",
        tick_duration_ms=1000,
        reasoning_effort=ReasoningEffort.PROVIDER_DEFAULT,
    )
    assert adapter.provider.reasoning_effort is None


def test_a_pinned_level_reaches_the_provider():
    adapter, _ = create_adapter(
        provider="openai",
        tick_duration_ms=1000,
        reasoning_effort=ReasoningEffort.MEDIUM,
    )
    assert adapter.provider.reasoning_effort == "medium"


def test_gemini_gets_the_high_it_records():
    adapter, _ = create_adapter(
        provider="gemini",
        tick_duration_ms=1000,
        reasoning_effort=ReasoningEffort.HIGH,
    )
    assert adapter.reasoning_effort == "high"


def test_a_pin_a_provider_cannot_honor_is_still_loud():
    """xai/nova/qwen reject a level outright — that guard must survive the
    PROVIDER_DEFAULT -> None translation."""
    with pytest.raises(ValueError, match="does not support reasoning_effort"):
        create_adapter(
            provider="xai",
            tick_duration_ms=1000,
            reasoning_effort=ReasoningEffort.HIGH,
        )


def test_agent_resolves_when_constructed_directly():
    """The manual construction path still cannot reach a provider with None."""
    agent = DiscreteTimeAudioNativeAgent(
        tools=[], domain_policy="policy", provider="gemini"
    )
    assert agent.reasoning_effort is ReasoningEffort.HIGH
