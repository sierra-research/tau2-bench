"""Phonic records each call's conversation ID as the provider session ID."""

from types import SimpleNamespace

from tau2.voice.audio_native.phonic import DiscreteTimePhonicAdapter, PhonicProvider


def test_conversation_created_sets_session_id():
    provider = PhonicProvider(api_key="test")
    adapter = DiscreteTimePhonicAdapter(tick_duration_ms=200, provider=provider)
    assert provider.session_id is None

    event = SimpleNamespace(type="conversation_created", conversation_id="conv_123")
    adapter._process_event(SimpleNamespace(events=[]), event)

    assert provider.session_id == "conv_123"
