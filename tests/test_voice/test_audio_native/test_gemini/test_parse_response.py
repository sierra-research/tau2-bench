"""Unit tests for GeminiLiveProvider._parse_response.

Tests verify that events are parsed from the correct paths only,
after removing duplicate parsing paths per Gemini team guidance:
- Audio: only from response.data (not inline_data)
- Output transcript: only from server_content.output_transcription (not response.text)
- Function calls: only from response.tool_call (not model_turn.parts)

Uses SimpleNamespace mocks -- no real API calls needed.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from tau2.voice.audio_native.gemini.events import (
    GeminiAudioDeltaEvent,
    GeminiAudioDoneEvent,
    GeminiFunctionCallDoneEvent,
    GeminiGoAwayEvent,
    GeminiInputTranscriptionEvent,
    GeminiInterruptionEvent,
    GeminiSessionResumptionEvent,
    GeminiTextDeltaEvent,
    GeminiTurnCompleteEvent,
    GeminiUnknownEvent,
)
from tau2.voice.audio_native.gemini.provider import GeminiLiveProvider, GeminiVADConfig

# =============================================================================
# Helpers
# =============================================================================


def _make_provider() -> GeminiLiveProvider:
    """Create a GeminiLiveProvider without connecting (for unit-testing _parse_response)."""
    # Bypass __init__ auth detection by constructing minimally
    provider = object.__new__(GeminiLiveProvider)
    # Set the minimal attributes _parse_response relies on
    provider._current_item_id = "test_item_1"
    provider.model = "gemini-3.1-flash-live-preview"
    provider._go_away_received = False
    provider._reconnect_at_turn_boundary = False
    provider._reconnect_deadline = None
    provider._resumption_handle = None
    return provider


def _empty_response(**overrides) -> SimpleNamespace:
    """Create a minimal mock response with no data by default."""
    defaults = {
        "data": None,
        "inline_data": None,
        "text": None,
        "tool_call": None,
        "go_away": None,
        "session_resumption_update": None,
        "server_content": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_server_content(**overrides) -> SimpleNamespace:
    """Create a mock server_content object."""
    defaults = {
        "interrupted": None,
        "turn_complete": None,
        "output_transcription": None,
        "input_transcription": None,
        "model_turn": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.mark.parametrize("raw_schema", [True, False])
def test_extended_thinking_tools_are_non_blocking(raw_schema):
    from google.genai import types

    provider = _make_provider()
    provider.model = "gemini-3.8-live-extended-thinking"
    provider._use_raw_json_schema = raw_schema
    tool = SimpleNamespace(
        openai_schema={
            "function": {
                "name": "get_user_details",
                "description": "Find a customer",
                "parameters": {"type": "object", "properties": {}},
            }
        }
    )

    declaration = provider._format_tools_for_api([tool])[0]

    assert declaration.behavior == types.Behavior.NON_BLOCKING


def test_existing_gemini_tools_keep_default_behavior():
    provider = _make_provider()
    provider._use_raw_json_schema = True
    tool = SimpleNamespace(
        openai_schema={
            "function": {
                "name": "get_user_details",
                "description": "Find a customer",
                "parameters": {"type": "object", "properties": {}},
            }
        }
    )

    declaration = provider._format_tools_for_api([tool])[0]

    assert declaration.behavior is None


def test_sdk_receive_waits_for_extended_thinking_idle():
    from google.genai import types
    from google.genai.live import AsyncSession

    responses = iter(
        [
            types.LiveServerMessage(
                server_content=types.LiveServerContent(
                    turn_complete=True,
                    interaction_status=types.InteractionStatus.IN_PROGRESS,
                )
            ),
            types.LiveServerMessage(
                server_content=types.LiveServerContent(
                    turn_complete=True,
                    interaction_status=types.InteractionStatus.IDLE,
                )
            ),
        ]
    )
    session = object.__new__(AsyncSession)

    async def receive_one():
        return next(responses)

    session._receive = receive_one

    async def collect():
        return [response async for response in session.receive()]

    assert len(asyncio.run(collect())) == 2


def test_receive_loop_rotates_audio_item_at_intermediate_turn():
    """Later speech gets its own item before the interaction becomes idle."""
    from google.genai import types

    async def check():
        provider = _make_provider()
        provider.model = "gemini-3.8-live-extended-thinking"
        provider._stop_receive = False
        provider._event_queue = asyncio.Queue()
        provider._generate_item_id = Mock(return_value="test_item_2")

        async def responses():
            yield _empty_response(data=b"first utterance")
            yield _empty_response(
                server_content=_make_server_content(
                    turn_complete=True,
                    interaction_status=types.InteractionStatus.IN_PROGRESS,
                )
            )
            yield _empty_response(data=b"second utterance")
            provider._stop_receive = True

        provider._session = Mock()
        provider._session.receive.return_value = responses()

        await provider._receive_loop_coro()
        events = []
        while not provider._event_queue.empty():
            events.append(provider._event_queue.get_nowait())

        audio_items = [
            event.item_id
            for event in events
            if isinstance(event, GeminiAudioDeltaEvent)
        ]
        completed_items = [
            event.item_id for event in events if isinstance(event, GeminiAudioDoneEvent)
        ]
        assert audio_items == ["test_item_1", "test_item_2"]
        assert completed_items == ["test_item_1"]
        assert (
            len(
                [
                    event
                    for event in events
                    if isinstance(event, GeminiTurnCompleteEvent)
                ]
            )
            == 1
        )

    asyncio.run(check())


def test_receive_loop_does_not_duplicate_idle_completion_without_turn_complete():
    """An IDLE-only response completes the interaction once at iterator end."""
    from google.genai import types

    async def check():
        provider = _make_provider()
        provider.model = "gemini-3.8-live-extended-thinking"
        provider._stop_receive = False
        provider._event_queue = asyncio.Queue()
        provider._reconnect_at_turn_boundary = True
        provider._reconnect_deadline = None
        provider.session_id = "test-session"
        provider._generate_item_id = Mock(return_value="test_item_2")

        async def responses():
            yield _empty_response(data=b"spoken answer")
            yield _empty_response(
                server_content=_make_server_content(
                    turn_complete=False,
                    interaction_status=types.InteractionStatus.IDLE,
                )
            )

        provider._session = Mock()
        provider._session.receive.return_value = responses()

        await provider._receive_loop_coro()
        events = []
        while not provider._event_queue.empty():
            events.append(provider._event_queue.get_nowait())

        assert sum(isinstance(e, GeminiTurnCompleteEvent) for e in events) == 1
        assert [e.item_id for e in events if isinstance(e, GeminiAudioDoneEvent)] == [
            "test_item_1"
        ]

    asyncio.run(check())


# =============================================================================
# Correct parsing paths
# =============================================================================


class TestCorrectParsingPaths:
    """Each event type is parsed from its correct source."""

    def test_audio_from_data(self):
        """response.data produces GeminiAudioDeltaEvent."""
        provider = _make_provider()
        response = _empty_response(data=b"\x00\x01\x02\x03")

        events = provider._parse_response(response)

        audio_events = [e for e in events if isinstance(e, GeminiAudioDeltaEvent)]
        assert len(audio_events) == 1
        assert audio_events[0].data == b"\x00\x01\x02\x03"
        assert audio_events[0].item_id == "test_item_1"

    def test_transcript_from_output_transcription(self):
        """server_content.output_transcription produces GeminiTextDeltaEvent."""
        provider = _make_provider()
        transcription = SimpleNamespace(text="Hello from Gemini")
        sc = _make_server_content(output_transcription=transcription)
        response = _empty_response(server_content=sc)

        events = provider._parse_response(response)

        text_events = [e for e in events if isinstance(e, GeminiTextDeltaEvent)]
        assert len(text_events) == 1
        assert text_events[0].text == "Hello from Gemini"

    def test_function_call_from_tool_call(self):
        """response.tool_call produces GeminiFunctionCallDoneEvent."""
        provider = _make_provider()
        func_call = SimpleNamespace(
            id="fc_123", name="get_weather", args={"location": "NYC"}
        )
        tool_call = SimpleNamespace(function_calls=[func_call])
        response = _empty_response(tool_call=tool_call)

        events = provider._parse_response(response)

        fc_events = [e for e in events if isinstance(e, GeminiFunctionCallDoneEvent)]
        assert len(fc_events) == 1
        assert fc_events[0].call_id == "fc_123"
        assert fc_events[0].name == "get_weather"
        assert fc_events[0].arguments == {"location": "NYC"}

    def test_multiple_function_calls(self):
        """Multiple function calls in tool_call are all parsed."""
        provider = _make_provider()
        fc1 = SimpleNamespace(id="fc_1", name="func_a", args={"x": 1})
        fc2 = SimpleNamespace(id="fc_2", name="func_b", args={"y": 2})
        tool_call = SimpleNamespace(function_calls=[fc1, fc2])
        response = _empty_response(tool_call=tool_call)

        events = provider._parse_response(response)

        fc_events = [e for e in events if isinstance(e, GeminiFunctionCallDoneEvent)]
        assert len(fc_events) == 2
        assert fc_events[0].name == "func_a"
        assert fc_events[1].name == "func_b"

    def test_interruption_event(self):
        """server_content.interrupted=True produces GeminiInterruptionEvent."""
        provider = _make_provider()
        sc = _make_server_content(interrupted=True)
        response = _empty_response(server_content=sc)

        events = provider._parse_response(response)

        assert len([e for e in events if isinstance(e, GeminiInterruptionEvent)]) == 1

    def test_turn_complete_event(self):
        """server_content.turn_complete=True produces GeminiTurnCompleteEvent."""
        provider = _make_provider()
        sc = _make_server_content(turn_complete=True)
        response = _empty_response(server_content=sc)

        events = provider._parse_response(response)

        assert len([e for e in events if isinstance(e, GeminiTurnCompleteEvent)]) == 1

    def test_extended_thinking_utterance_is_not_interaction_complete(self):
        provider = _make_provider()
        provider.model = "gemini-3.8-live-extended-thinking"
        sc = _make_server_content(turn_complete=True, interaction_status="IN_PROGRESS")

        events = provider._parse_response(_empty_response(server_content=sc))

        assert events == []

    def test_extended_thinking_idle_completes_interaction(self):
        from google.genai import types

        provider = _make_provider()
        provider.model = "gemini-3.8-live-extended-thinking"
        sc = _make_server_content(
            turn_complete=True,
            interaction_status=types.InteractionStatus.IDLE,
        )

        events = provider._parse_response(_empty_response(server_content=sc))

        assert len([e for e in events if isinstance(e, GeminiTurnCompleteEvent)]) == 1

    def test_extended_thinking_status_only_is_not_unknown(self):
        provider = _make_provider()
        provider.model = "gemini-3.8-live-extended-thinking"
        sc = _make_server_content(interaction_status="IN_PROGRESS")

        events = provider._parse_response(_empty_response(server_content=sc))

        assert events == []

    def test_input_transcription(self):
        """server_content.input_transcription produces GeminiInputTranscriptionEvent."""
        provider = _make_provider()
        transcription = SimpleNamespace(text="User said hello")
        sc = _make_server_content(input_transcription=transcription)
        response = _empty_response(server_content=sc)

        events = provider._parse_response(response)

        input_events = [
            e for e in events if isinstance(e, GeminiInputTranscriptionEvent)
        ]
        assert len(input_events) == 1
        assert input_events[0].transcript == "User said hello"

    def test_go_away_event(self):
        """response.go_away produces GeminiGoAwayEvent and sets reconnection state."""
        provider = _make_provider()
        go_away = SimpleNamespace(time_left=SimpleNamespace(seconds=30))
        response = _empty_response(go_away=go_away)

        async def _parse():
            return provider._parse_response(response)

        events = asyncio.run(_parse())

        go_events = [e for e in events if isinstance(e, GeminiGoAwayEvent)]
        assert len(go_events) == 1
        assert go_events[0].time_left_seconds == 30.0
        assert provider._go_away_received is True
        assert provider._reconnect_at_turn_boundary is True
        assert provider._reconnect_deadline is not None

    def test_session_resumption_event(self):
        """response.session_resumption_update produces GeminiSessionResumptionEvent."""
        provider = _make_provider()
        update = SimpleNamespace(new_handle="handle_abc123", resumable=True)
        response = _empty_response(session_resumption_update=update)

        events = provider._parse_response(response)

        sr_events = [e for e in events if isinstance(e, GeminiSessionResumptionEvent)]
        assert len(sr_events) == 1
        assert sr_events[0].new_handle == "handle_abc123"
        assert sr_events[0].resumable is True
        assert provider._resumption_handle == "handle_abc123"

    def test_empty_response_produces_unknown(self):
        """Response with no recognized fields produces GeminiUnknownEvent."""
        provider = _make_provider()
        response = _empty_response()

        events = provider._parse_response(response)

        assert len([e for e in events if isinstance(e, GeminiUnknownEvent)]) == 1


# =============================================================================
# No double-emit (regression tests for removed duplicate paths)
# =============================================================================


class TestNoDoubleEmit:
    """Removed duplicate paths must not re-emit events."""

    def test_audio_not_double_emitted(self):
        """When both data and inline_data are present, only one audio event."""
        provider = _make_provider()
        response = _empty_response(
            data=b"\x00\x01\x02\x03",
            inline_data=SimpleNamespace(data=b"\x04\x05\x06\x07"),
        )

        events = provider._parse_response(response)

        audio_events = [e for e in events if isinstance(e, GeminiAudioDeltaEvent)]
        assert len(audio_events) == 1
        assert audio_events[0].data == b"\x00\x01\x02\x03"

    def test_function_call_not_double_emitted(self):
        """When tool_call and model_turn.parts both have FCs, only tool_call emits."""
        provider = _make_provider()
        fc_tool = SimpleNamespace(id="fc_1", name="get_order", args={})
        tool_call = SimpleNamespace(function_calls=[fc_tool])
        fc_part = SimpleNamespace(id="fc_2", name="get_order", args={})
        part = SimpleNamespace(function_call=fc_part)
        sc = _make_server_content(model_turn=SimpleNamespace(parts=[part]))
        response = _empty_response(tool_call=tool_call, server_content=sc)

        events = provider._parse_response(response)

        fc_events = [e for e in events if isinstance(e, GeminiFunctionCallDoneEvent)]
        assert len(fc_events) == 1
        assert fc_events[0].call_id == "fc_1"

    def test_transcript_not_double_emitted(self):
        """When both response.text and output_transcription exist, only transcription emits."""
        provider = _make_provider()
        transcription = SimpleNamespace(text="Correct transcript")
        sc = _make_server_content(output_transcription=transcription)
        response = _empty_response(text="Wrong text path", server_content=sc)

        events = provider._parse_response(response)

        text_events = [e for e in events if isinstance(e, GeminiTextDeltaEvent)]
        assert len(text_events) == 1
        assert text_events[0].text == "Correct transcript"


# =============================================================================
# Combined response
# =============================================================================


class TestCombinedResponse:
    """Test a response with multiple fields set simultaneously."""

    def test_audio_and_transcription_and_tool_call(self):
        """Response with audio, output transcription, and tool call."""
        provider = _make_provider()
        transcription = SimpleNamespace(text="I found the order")
        sc = _make_server_content(output_transcription=transcription)
        fc = SimpleNamespace(id="fc_1", name="lookup_order", args={"id": "123"})
        tool_call = SimpleNamespace(function_calls=[fc])
        response = _empty_response(
            data=b"\x00" * 100, tool_call=tool_call, server_content=sc
        )

        events = provider._parse_response(response)

        assert len([e for e in events if isinstance(e, GeminiAudioDeltaEvent)]) == 1
        assert len([e for e in events if isinstance(e, GeminiTextDeltaEvent)]) == 1
        assert (
            len([e for e in events if isinstance(e, GeminiFunctionCallDoneEvent)]) == 1
        )


class TestInputAudioTranscriptionConfig:
    """Verify input_audio_transcription language_codes defaults to en-US and serializes."""

    def test_default_language_codes(self):
        """DEFAULT_GEMINI_TRANSCRIPTION_LANGUAGE_CODES defaults to en-US."""
        from tau2.config import DEFAULT_GEMINI_TRANSCRIPTION_LANGUAGE_CODES

        assert DEFAULT_GEMINI_TRANSCRIPTION_LANGUAGE_CODES == ["en-US"]

    def test_live_converter_serialization(self):
        """types.AudioTranscriptionConfig(language_codes=['en-US']) serializes in Live setup."""
        from google.genai import _live_converters, types

        class _DummyClient:
            vertexai = False

        provider = GeminiLiveProvider(api_key="test-key")
        assert provider.transcription_language_codes == ["en-US"]
        cfg = types.AudioTranscriptionConfig(
            language_codes=provider.transcription_language_codes
        )
        params = types.LiveConnectParameters(
            model="models/gemini-3.8-live-extended-thinking",
            config=types.LiveConnectConfig(input_audio_transcription=cfg),
        ).model_dump(exclude_none=True)
        res = _live_converters._LiveConnectParameters_to_mldev(_DummyClient(), params)
        assert res["setup"]["inputAudioTranscription"]["language_codes"] == ["en-US"]

    @pytest.mark.parametrize("vertexai", [False, True])
    @pytest.mark.parametrize(
        "model,language_codes,enabled,expected",
        [
            ("gemini-2.5-flash-native-audio-preview-12-2025", None, True, ["en-US"]),
            (
                "gemini-2.5-flash-native-audio-preview-12-2025",
                ["fr-FR"],
                True,
                ["fr-FR"],
            ),
            ("gemini-2.5-flash-native-audio-preview-12-2025", [], True, []),
            ("gemini-2.5-flash-native-audio-preview-12-2025", None, False, None),
            ("gemini-3.1-flash-live-preview", None, True, ["en-US"]),
        ],
        ids=["default", "override", "auto-detect", "disabled", "gemini-3.1"],
    )
    def test_connect_transcription_config(
        self, model, language_codes, enabled, expected, vertexai
    ):
        """Check the actual connection config and both SDK serialization paths."""
        from google.genai import _live_converters, types

        provider = GeminiLiveProvider(
            api_key="test-key",
            model=model,
            transcription_language_codes=language_codes,
        )
        session_manager = AsyncMock()
        connect = Mock(return_value=session_manager)
        provider._client = SimpleNamespace(
            aio=SimpleNamespace(live=SimpleNamespace(connect=connect))
        )

        async def connect_and_disconnect():
            await provider.connect(
                system_prompt="Test instructions",
                tools=[],
                vad_config=GeminiVADConfig(enable_input_transcription=enabled),
            )
            await provider.disconnect()

        asyncio.run(connect_and_disconnect())
        config = connect.call_args.kwargs["config"]
        converter = (
            _live_converters._LiveConnectParameters_to_vertex
            if vertexai
            else _live_converters._LiveConnectParameters_to_mldev
        )
        params = types.LiveConnectParameters(model=model, config=config).model_dump(
            exclude_none=True
        )
        setup = converter(SimpleNamespace(vertexai=vertexai), params)["setup"]
        if expected is None:
            assert config.input_audio_transcription is None
            assert "inputAudioTranscription" not in setup
        else:
            assert (config.input_audio_transcription.language_codes or []) == expected
            assert (
                setup["inputAudioTranscription"].get("language_codes", []) == expected
            )


@pytest.mark.parametrize(
    "model,expected",
    [
        ("gemini-2.5-flash-native-audio-preview-12-2025", True),
        ("gemini-3.1-flash-live-preview", False),
        ("gemini-3.8-live-extended-thinking", False),
    ],
)
def test_connect_proactivity_config(model, expected):
    from google.genai import _live_converters, types

    provider = GeminiLiveProvider(api_key="test-key", model=model)
    session_manager = AsyncMock()
    connect = Mock(return_value=session_manager)
    provider._client = SimpleNamespace(
        aio=SimpleNamespace(live=SimpleNamespace(connect=connect))
    )

    async def connect_and_disconnect():
        await provider.connect(system_prompt="Test instructions", tools=[])
        await provider.disconnect()

    asyncio.run(connect_and_disconnect())
    config = connect.call_args.kwargs["config"]
    params = types.LiveConnectParameters(model=model, config=config).model_dump(
        exclude_none=True
    )
    setup = _live_converters._LiveConnectParameters_to_mldev(
        SimpleNamespace(vertexai=False), params
    )["setup"]
    assert (config.proactivity is not None) is expected
    assert ("proactivity" in setup) is expected


class TestGeminiInputSampleRate:
    """Test Gemini input sample rate configuration."""

    def test_adapter_default_8khz(self):
        """DiscreteTimeGeminiAdapter defaults to 8000 Hz input sample rate."""
        from tau2.voice.audio_native.gemini.discrete_time_adapter import (
            DiscreteTimeGeminiAdapter,
        )

        adapter = DiscreteTimeGeminiAdapter(tick_duration_ms=200)
        assert adapter.input_sample_rate == 8000

    def test_adapter_explicit_16khz(self):
        """DiscreteTimeGeminiAdapter accepts an explicit 16000 Hz input sample rate."""
        from tau2.voice.audio_native.gemini.discrete_time_adapter import (
            DiscreteTimeGeminiAdapter,
        )

        adapter = DiscreteTimeGeminiAdapter(
            tick_duration_ms=200, input_sample_rate=16000
        )
        assert adapter.input_sample_rate == 16000
