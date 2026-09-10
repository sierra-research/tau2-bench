"""Provider failures must not silently turn into benchmark task failures."""

import pytest

from tau2.voice.audio_native.livekit.discrete_time_adapter import LiveKitCascadedAdapter
from tau2.voice.audio_native.livekit.provider import CascadedEvent, CascadedEventType


def test_payment_error_aborts_tick():
    adapter = LiveKitCascadedAdapter(tick_duration_ms=200)
    event = CascadedEvent(
        type=CascadedEventType.ERROR,
        data={"stage": "tts", "error": "402 Payment Required"},
    )
    with pytest.raises(RuntimeError, match="402 Payment Required"):
        adapter._handle_event(event, [], [], [])
