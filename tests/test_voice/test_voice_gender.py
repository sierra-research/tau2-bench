# Copyright Sierra
"""Provider voice -> perceived gender resolution."""

from tau2.config import DEFAULT_XAI_VOICE
from tau2.voice.voice_gender import resolve_agent_gender


def test_resolve_gender_known_voice():
    assert resolve_agent_gender("xai", "Ara") == "female"
    assert resolve_agent_gender("xai", "Rex") == "male"
    assert resolve_agent_gender("nova", "matthew") == "male"


def test_resolve_gender_provider_default_voice():
    # Voice omitted -> falls back to the provider's default voice.
    assert resolve_agent_gender("xai") == resolve_agent_gender("xai", DEFAULT_XAI_VOICE)
    assert resolve_agent_gender("xai") == "female"  # default Ara


def test_every_provider_default_voice_has_a_gender():
    # The OpenAI default is pinned gendered so the judge knows every agent's gender.
    for provider in ("openai", "gemini", "xai", "nova", "qwen"):
        assert resolve_agent_gender(provider) in ("male", "female"), provider


def test_resolve_gender_neutral_or_unknown_is_none():
    assert resolve_agent_gender("openai", "alloy") is None  # alloy itself is neutral
    assert resolve_agent_gender("bogus", "whoever") is None
    assert resolve_agent_gender(None) is None
