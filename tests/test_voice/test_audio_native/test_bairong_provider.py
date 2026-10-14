import asyncio

from tau2.config import (
    DEFAULT_BAIRONG_REALTIME_BASE_URL,
    DEFAULT_BAIRONG_REALTIME_MODEL,
)
from tau2.voice.audio_native.adapter import create_adapter
from tau2.voice.audio_native.bairong import BaiRongRealtimeProvider
from tau2.voice.audio_native.bairong.discrete_time_adapter import (
    DiscreteTimeBaiRongRealtimeAdapter,
)
from tau2.voice.audio_native.openai.events import ResponseDoneEvent
from tau2.voice.audio_native.tick_result import TickResult


def test_provider_uses_hosted_service_without_credentials(monkeypatch):
    monkeypatch.setenv("BAIRONG_REALTIME_BASE_URL", "ws://127.0.0.1:8765")
    monkeypatch.setenv("BAIRONG_REALTIME_API_KEY", "should-not-be-used")

    provider = BaiRongRealtimeProvider()

    assert provider.base_url == DEFAULT_BAIRONG_REALTIME_BASE_URL
    assert provider.model == DEFAULT_BAIRONG_REALTIME_MODEL
    assert provider.api_key == "EMPTY"
    assert provider.base_url.startswith("wss://")


def test_factory_builds_bairong_adapter():
    adapter, model = create_adapter("bairong", tick_duration_ms=200)

    assert isinstance(adapter, DiscreteTimeBaiRongRealtimeAdapter)
    assert model == DEFAULT_BAIRONG_REALTIME_MODEL
    adapter.disconnect()


def test_usage_records_are_attributed_to_bairong():
    adapter = DiscreteTimeBaiRongRealtimeAdapter(
        tick_duration_ms=200,
        model=DEFAULT_BAIRONG_REALTIME_MODEL,
    )
    result = TickResult(
        tick_number=1,
        audio_sent_bytes=0,
        audio_sent_duration_ms=0,
    )
    event = ResponseDoneEvent(
        type="response.done",
        response_id="response-1",
        usage={"input_tokens": 10, "output_tokens": 5},
    )

    asyncio.run(adapter._process_event(result, event))

    records = adapter.get_usage_records()
    assert len(records) == 1
    assert records[0].provider == "bairong"
