"""The smallest adapter against a fake Hydra server on 127.0.0.1 (no network).

These run the public sync API (connect, run_tick, send_tool_result,
disconnect) over a real WebSocket, so the handshake, framing and the
background event loop are exercised as in a benchmark run. Events are pushed
before the tick that should receive them.
"""

import asyncio
import base64

import pytest

from tau2.data_model.message import ToolCall
from tau2.environment.tool import Tool
from tau2.voice.audio_native.adapter import create_adapter
from tau2.voice.audio_native.smallest import (
    DiscreteTimeSmallestAdapter,
    SmallestProvider,
)
from tau2.voice.audio_native.smallest.provider import redact_url, validate_ws_url

from .fake_hydra import (
    SESSION_ID,
    SILENCE_TICK,
    TICK_BYTES,
    FakeHydra,
    audio,
    audio_done,
    barge_in,
    caption,
    created,
    done,
    speech_started,
    tool_call,
)

PROMPT = "You are a customer service agent.\n  exact whitespace\n"
SENTENCE = "I can change that flight for you."


def get_order_status(order_id: str) -> str:
    """Get the status of an order.

    Args:
        order_id: The order ID.
    """
    return order_id


@pytest.fixture
def hydra():
    server = FakeHydra(output_rate=24000).start()
    yield server
    server.stop()


@pytest.fixture
def adapter(hydra, monkeypatch):
    monkeypatch.setenv("SMALLEST_WS_URL", hydra.url)
    monkeypatch.delenv("SMALLEST_API_KEY", raising=False)
    monkeypatch.setenv("SMALLEST_VOICE", "sophie")
    adapter, _ = create_adapter(provider="smallest", tick_duration_ms=200)
    adapter.connect(PROMPT, [Tool(get_order_status)], vad_config=None)
    yield adapter
    adapter.disconnect()


def test_connect_sends_the_benchmark_prompt_tools_and_voice(adapter, hydra):
    (configure,) = hydra.messages("session.configure")
    session = configure["session"]
    assert session["instructions"] == PROMPT
    assert session["voice"] == "sophie"
    (tool,) = session["tools"]
    assert tool["type"] == "function" and tool["name"] == "get_order_status"
    assert tool["parameters"]["required"] == ["order_id"]
    assert adapter.is_connected
    assert adapter.provider.session_id == SESSION_ID
    assert adapter.provider.output_sample_rate == 24000
    assert hydra.auth_headers == [None]  # no key configured, no credential sent


def test_api_key_goes_in_the_authorization_header(hydra, monkeypatch):
    monkeypatch.setenv("SMALLEST_WS_URL", hydra.url)
    monkeypatch.setenv("SMALLEST_API_KEY", "test-key")
    adapter, _ = create_adapter(provider="smallest", tick_duration_ms=200)
    adapter.connect(PROMPT, [])
    adapter.disconnect()
    assert hydra.auth_headers == ["Bearer test-key"]


def run_until(adapter, condition, max_ticks: int = 5) -> list:
    """Ticks until ``condition(tick)`` holds (a busy machine may delay an event)."""
    ticks = []
    for _ in range(max_ticks):
        ticks.append(adapter.run_tick(SILENCE_TICK))
        if condition(ticks[-1]):
            return ticks
    raise AssertionError(f"condition not met within {max_ticks} ticks")


def test_audio_in_and_out_per_tick(adapter, hydra):
    hydra.send(created("r1"), caption("r1", SENTENCE), audio("r1", 1000, rate=24000))
    ticks = run_until(adapter, lambda t: t.agent_audio_bytes > 0)
    ticks += [adapter.run_tick(SILENCE_TICK) for _ in range(5)]
    # 1 s of 24 kHz audio = five ticks of 8 kHz u-law, played back to back.
    playing = [t for t in ticks if t.agent_audio_bytes]
    assert [t.agent_audio_bytes for t in playing] == [TICK_BYTES] * 5
    assert all(t.item_ids == ["r1"] for t in playing)
    assert all(len(t.get_played_agent_audio()) == TICK_BYTES for t in ticks)
    assert ticks[-1].agent_audio_bytes == 0
    # Caller audio: 20 ms frames of 16 kHz PCM16, 200 ms per tick.
    count = len(ticks)
    appends = lambda: hydra.messages("input_audio_buffer.append")  # noqa: E731
    hydra.wait_for(lambda: len(appends()) == 10 * count)
    frames = [base64.b64decode(m["audio"]) for m in appends()]
    assert all(len(f) <= 640 for f in frames)
    assert count * 6400 - 8 <= sum(len(f) for f in frames) <= count * 6400


def test_text_is_never_ahead_of_the_audio_played(adapter, hydra):
    hydra.send(created("r1"), caption("r1", SENTENCE))
    shown, played = "", 0
    for _ in range(5):
        hydra.send(audio("r1", 200, rate=24000))
        tick = adapter.run_tick(SILENCE_TICK)
        played += tick.agent_audio_bytes
        shown += tick.proportional_transcript
        assert SENTENCE.startswith(shown)
        assert len(shown) <= 15 * played / 8000
    hydra.send(audio_done("r1"), done("r1"))
    for _ in range(5):  # completed: the rest shows once all audio has played
        shown += adapter.run_tick(SILENCE_TICK).proportional_transcript
        if shown == SENTENCE:
            break
    assert shown == SENTENCE


def test_tool_call_relayed_and_result_returned(adapter, hydra):
    hydra.send(
        created("r1"),
        tool_call("r1", "call_1", "get_order_status", '{"order_id": "#W1"}'),
    )
    tick = run_until(adapter, lambda t: bool(t.tool_calls))[-1]
    assert tick.tool_calls == [
        ToolCall(id="call_1", name="get_order_status", arguments={"order_id": "#W1"})
    ]
    adapter.send_tool_result("call_1", "Order #W1 shipped", request_response=True)
    adapter.run_tick(SILENCE_TICK)
    hydra.wait_for(lambda: hydra.messages("response.create"))
    replies = [m for m in hydra.received if m["type"] != "input_audio_buffer.append"]
    assert replies[-2:] == [
        {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "Order #W1 shipped",
            },
        },
        {"type": "response.create", "response": None},
    ]


def test_barge_in_clears_queued_agent_audio(adapter, hydra):
    hydra.send(created("r1"), caption("r1", SENTENCE), audio("r1", 2000, rate=24000))
    hydra.send(audio_done("r1"), done("r1"))
    ticks = run_until(adapter, lambda t: t.agent_audio_bytes > 0)
    # Hydra's order: the onset, then its committed barge-in.
    hydra.send(speech_started(), barge_in())
    ticks += run_until(adapter, lambda t: t.was_truncated)
    cut = ticks[-1]
    assert "interrupted" in cut.vad_events
    assert any("speech_started" in t.vad_events for t in ticks)
    # 2 s = 10 ticks of audio: what played before and in the cut tick, plus
    # what the cut dropped, is all of it.
    played = sum(t.agent_audio_bytes for t in ticks)
    assert TICK_BYTES <= played < 10 * TICK_BYTES
    assert played + cut.truncated_audio_bytes == 10 * TICK_BYTES
    after = [adapter.run_tick(SILENCE_TICK) for _ in range(2)]
    assert all(
        t.agent_audio_bytes == 0 and not t.proportional_transcript for t in after
    )
    assert adapter._buffered_agent_audio == []


def test_clean_disconnect_keeps_the_session_and_records_no_usage(adapter, hydra):
    usage = {"input_tokens": 50, "output_tokens": 5, "total_tokens": 55}
    hydra.send(created("r1"), done("r1", usage=usage))
    adapter.run_tick(SILENCE_TICK)
    adapter.disconnect()
    assert not adapter.is_connected
    hydra.wait_for(lambda: hydra.close_codes == [1000])
    assert adapter.provider.session_id == SESSION_ID
    assert adapter.get_usage_records() == []  # the provider reports no usage
    adapter.disconnect()  # idempotent


def test_server_closing_mid_call_fails_the_tick(adapter, hydra):
    hydra.close_client(1011, "worker crashed")
    with pytest.raises(RuntimeError, match="1011"):
        for _ in range(3):
            adapter.run_tick(SILENCE_TICK)


@pytest.mark.parametrize(
    "reply, message",
    [("error", "rejected the session"), ("no_rate", "output_audio_sample_rate")],
)
def test_bad_handshake_fails_connect(monkeypatch, reply, message):
    server = FakeHydra(configure_reply=reply).start()
    try:
        monkeypatch.setenv("SMALLEST_WS_URL", server.url)
        adapter, _ = create_adapter(provider="smallest", tick_duration_ms=200)
        with pytest.raises(RuntimeError, match=message):
            adapter.connect(PROMPT, [])
        assert not adapter.is_connected
        server.wait_for(lambda: len(server.close_codes) == 1)
    finally:
        server.stop()


def test_silent_handshake_times_out():
    server = FakeHydra(configure_reply="silent").start()
    try:
        provider = SmallestProvider(ws_url=server.url, handshake_timeout_s=0.3)
        with pytest.raises(RuntimeError, match="did not confirm"):
            asyncio.run(provider.connect(PROMPT, []))
        assert provider.ws is None
    finally:
        server.stop()


def test_missing_url_fails_connect_with_a_clear_message(monkeypatch):
    monkeypatch.delenv("SMALLEST_WS_URL", raising=False)
    adapter = DiscreteTimeSmallestAdapter(tick_duration_ms=200)
    with pytest.raises(RuntimeError, match="SMALLEST_WS_URL"):
        adapter.connect(PROMPT, [])
    assert not adapter.is_connected


def test_urls_and_credentials_are_checked_and_redacted():
    url = "wss://hydra.example:8443/v1/hydra?api_key=secret"
    assert redact_url(url) == "wss://hydra.example:8443/v1/hydra"
    assert validate_ws_url("ws://localhost:7871/v1/hydra?api_key=k", False)
    assert validate_ws_url("wss://hydra.example/v1/hydra", True)
    for bad, has_key in [
        ("", False),
        ("http://hydra.example/v1/hydra", False),
        ("wss://user:pw@hydra.example/v1/hydra", False),
        ("ws://hydra.example/v1/hydra", True),
        ("ws://hydra.example/v1/hydra?api_key=k", False),
        ("wss://hydra.example:99999/v1/hydra?api_key=secret", False),
    ]:
        with pytest.raises(ValueError) as error:
            validate_ws_url(bad, has_key)
        assert "secret" not in str(error.value)
