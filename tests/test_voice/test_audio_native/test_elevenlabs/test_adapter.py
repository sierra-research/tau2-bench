from collections import deque

import pytest

from tau2.data_model.simulation import AudioNativeConfig
from tau2.environment.tool import Tool
from tau2.voice.audio_native.adapter import create_adapter
from tau2.voice.audio_native.elevenlabs.discrete_time_adapter import (
    DiscreteTimeElevenLabsAdapter,
)
from tau2.voice.audio_native.elevenlabs.events import (
    ElevenLabsAgentResponseEvent,
    ElevenLabsAudioEvent,
    ElevenLabsClientToolCallEvent,
    ElevenLabsInterruptionEvent,
)


class FakeProvider:
    def __init__(
        self,
        events_by_tick,
        input_format="pcm_16000",
        output_format="pcm_24000",
    ):
        self.agent_id = "agent_test"
        self.session_id = "conv_test"
        self.user_input_audio_format = input_format
        self.agent_output_audio_format = output_format
        self.events_by_tick = deque(events_by_tick)
        self.sent_audio = []
        self.tool_results = []
        self.system_prompt = None

    async def connect(self, system_prompt):
        self.system_prompt = system_prompt

    async def send_audio(self, audio):
        self.sent_audio.append(audio)

    async def send_tool_result(self, call_id, result, is_error=False):
        self.tool_results.append((call_id, result, is_error))

    async def receive_events_for_duration(self, _duration):
        return self.events_by_tick.popleft() if self.events_by_tick else []


def make_order_tool():
    def get_order_status(order_id: str) -> str:
        """Look up an order.

        Args:
            order_id: Order identifier.
        """
        return order_id

    return Tool(get_order_status)


def test_audio_native_config_requires_explicit_elevenlabs_agent_id():
    with pytest.raises(ValueError, match="agent ID"):
        AudioNativeConfig(provider="elevenlabs")

    config = AudioNativeConfig(provider="elevenlabs", model="agent_test")
    assert config.model == "agent_test"


def test_factory_uses_audio_native_model_as_elevenlabs_agent_id():
    adapter, model = create_adapter(
        "elevenlabs",
        tick_duration_ms=20,
        model="agent_123",
    )

    assert isinstance(adapter, DiscreteTimeElevenLabsAdapter)
    assert adapter.agent_id == "agent_123"
    assert model == "agent_123"


def test_elevenlabs_requires_agent_id_and_hosted_reasoning_config():
    with pytest.raises(ValueError, match="agent ID"):
        DiscreteTimeElevenLabsAdapter(tick_duration_ms=20)
    with pytest.raises(ValueError, match="reasoning.*hosted agent"):
        DiscreteTimeElevenLabsAdapter(
            tick_duration_ms=20,
            model="agent_123",
            reasoning_effort="high",
        )


def test_pcm_audio_tool_call_and_result_round_trip():
    provider = FakeProvider(
        [
            [
                ElevenLabsAgentResponseEvent(
                    event_id=7,
                    transcript="Order status",
                ),
                ElevenLabsAudioEvent(
                    event_id=7,
                    audio_data=b"\x00\x00" * 480,
                    alignment={"chars": list("Order status")},
                    is_final=True,
                ),
                ElevenLabsClientToolCallEvent(
                    event_id=8,
                    tool_name="get_order_status",
                    tool_call_id="call_7",
                    parameters={"order_id": "A-1"},
                    expects_response=True,
                ),
            ],
            [],
        ]
    )
    adapter = DiscreteTimeElevenLabsAdapter(
        tick_duration_ms=20,
        model="agent_test",
        provider=provider,
    )
    adapter.connect("system prompt", [make_order_tool()])

    try:
        tick = adapter.run_tick(b"\xff" * 160, tick_number=1)
        assert provider.system_prompt == "system prompt"
        assert len(provider.sent_audio[0]) > 160
        assert tick.agent_audio_bytes == 160
        assert tick.proportional_transcript == "Order status"
        assert tick.tool_calls[0].model_dump() == {
            "id": "call_7",
            "name": "get_order_status",
            "arguments": {"order_id": "A-1"},
            "requestor": "assistant",
        }

        adapter.send_tool_result("call_7", "shipped", is_error=False)
        adapter.run_tick(b"\xff" * 160, tick_number=2)
        assert provider.tool_results == [("call_7", "shipped", False)]
    finally:
        adapter.disconnect()


def test_transcript_fallback_handles_audio_before_response_event():
    provider = FakeProvider(
        [
            [
                ElevenLabsAudioEvent(
                    event_id=11,
                    audio_data=b"\x7f" * 160,
                ),
                ElevenLabsAgentResponseEvent(
                    event_id=11,
                    transcript="Late transcript",
                ),
            ]
        ],
        input_format="ulaw_8000",
        output_format="ulaw_8000",
    )
    adapter = DiscreteTimeElevenLabsAdapter(
        tick_duration_ms=20,
        model="agent_test",
        provider=provider,
    )
    adapter.connect("system prompt", [])

    try:
        tick = adapter.run_tick(b"\xff" * 160, tick_number=1)
        assert tick.proportional_transcript == "Late transcript"
    finally:
        adapter.disconnect()


def test_ulaw_audio_is_passed_through_and_interruption_clears_buffer():
    provider = FakeProvider(
        [
            [
                ElevenLabsAgentResponseEvent(event_id=9, transcript="Long reply"),
                ElevenLabsAudioEvent(
                    event_id=9,
                    audio_data=b"\x7f" * 320,
                    alignment={"chars": list("Long reply")},
                ),
            ],
            [
                ElevenLabsInterruptionEvent(event_id=9, reason="user speech"),
                # ElevenLabs may deliver stale chunks after the interruption.
                ElevenLabsAudioEvent(event_id=8, audio_data=b"\x7f" * 10),
            ],
        ],
        input_format="ulaw_8000",
        output_format="ulaw_8000",
    )
    adapter = DiscreteTimeElevenLabsAdapter(
        tick_duration_ms=20,
        model="agent_test",
        provider=provider,
    )
    adapter.connect("system prompt", [])

    try:
        first = adapter.run_tick(b"\xff" * 160, tick_number=1)
        assert provider.sent_audio[0] == b"\xff" * 160
        assert first.agent_audio_bytes == 160

        interrupted = adapter.run_tick(b"\xff" * 160, tick_number=2)
        assert interrupted.was_truncated is True
        assert interrupted.skip_item_id == "9"
        assert interrupted.agent_audio_bytes == 0
        assert interrupted.truncated_audio_bytes == 170
    finally:
        adapter.disconnect()
