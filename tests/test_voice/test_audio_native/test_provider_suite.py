"""Horizontal provider suite: smoke tests for DiscreteTimeAdapter implementations.

Goal: answer "will new issues arise when we run e2e simulations via tau2 run?"
Each test progressively exercises one more capability the eval framework needs.

Architecture
------------
- Parameterized by provider name. Every test automatically runs against all
  providers whose credentials are available (GEMINI_API_KEY, XAI_API_KEY, etc.).
  Providers without credentials are skipped.

- Tests at the DiscreteTimeAdapter interface. Uses create_adapter() -- the same
  factory the eval framework uses -- and only interacts through connect(),
  run_tick(), send_tool_result(), and disconnect(). No provider-specific types.

- Pre-converted telephony audio. Test audio files (.ulaw, 8kHz mu-law) are
  loaded from testdata/ and split into per-tick chunks with trailing silence.
  This matches exactly how the eval framework feeds audio to adapters.

- Progressive test classes. Each class tests one capability: connection
  lifecycle, single-turn reply, multi-turn, tool call round-trip, barge-in.
  Earlier failures tell you where things broke.

- Invariants checked on every tick. Audio capping (agent_audio_chunks <=
  bytes_per_tick) and played audio length (get_played_agent_audio() ==
  bytes_per_tick) are asserted on every tick across all tests.

- Audio length parameterization. Single-turn and barge-in tests run with both
  short (720ms) and medium (1120ms) speech to surface provider VAD thresholds.

Parallelism
-----------
Tests are safe to run in parallel (e.g. pytest-xdist -n 4). Each test creates
its own adapter with an independent connection. No shared state between tests.

Usage
-----
Run for a specific provider:
    pytest tests/test_voice/test_audio_native/test_provider_suite.py -v -s -k gemini

Run for all available providers:
    pytest tests/test_voice/test_audio_native/test_provider_suite.py -v -s

Run in parallel:
    pytest tests/test_voice/test_audio_native/test_provider_suite.py -n 4

Save a glanceable results summary (writes provider_suite_results.txt):
    uv run tests/test_voice/test_audio_native/run_provider_suite.py
"""

import os
import time
from pathlib import Path
from typing import List, Optional

import pytest

from tau2.config import TELEPHONY_ULAW_SILENCE
from tau2.environment.tool import Tool
from tau2.voice.audio_native.adapter import DiscreteTimeAdapter, create_adapter
from tau2.voice.audio_native.tick_result import TickResult

pytestmark = pytest.mark.full_duplex_integration

# =============================================================================
# Provider parameterization
# =============================================================================

TICK_DURATION_MS = 200
TESTDATA_DIR = Path(__file__).parent / "testdata"

# Each provider is gated by its env var so the suite only runs providers you
# have credentials for.
PROVIDERS = [
    pytest.param(
        "gemini",
        marks=pytest.mark.skipif(
            not (
                os.environ.get("GEMINI_API_KEY")
                or os.environ.get("GOOGLE_SERVICE_ACCOUNT_KEY")
                or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            ),
            reason="No Gemini credentials (GEMINI_API_KEY, GOOGLE_SERVICE_ACCOUNT_KEY, or GOOGLE_APPLICATION_CREDENTIALS)",
        ),
    ),
    pytest.param(
        "livekit",
        marks=pytest.mark.skipif(
            not os.environ.get("LIVEKIT_TEST_ENABLED"),
            reason="LIVEKIT_TEST_ENABLED not set",
        ),
    ),
    pytest.param(
        "livekit-thinking",
        marks=pytest.mark.skipif(
            not os.environ.get("LIVEKIT_TEST_ENABLED"),
            reason="LIVEKIT_TEST_ENABLED not set",
        ),
    ),
    pytest.param(
        "nova",
        marks=pytest.mark.skipif(
            not os.environ.get("NOVA_TEST_ENABLED"),
            reason="NOVA_TEST_ENABLED not set",
        ),
    ),
    pytest.param(
        "openai",
        marks=pytest.mark.skipif(
            not os.environ.get("OPENAI_API_KEY"),
            reason="OPENAI_API_KEY not set",
        ),
    ),
    pytest.param(
        "qwen",
        marks=pytest.mark.skipif(
            not os.environ.get("DASHSCOPE_API_KEY"),
            reason="DASHSCOPE_API_KEY not set",
        ),
    ),
    pytest.param(
        "xai",
        marks=pytest.mark.skipif(
            not os.environ.get("XAI_API_KEY"),
            reason="XAI_API_KEY not set",
        ),
    ),
]

SYSTEM_PROMPT = "You are a helpful assistant. Keep responses brief."
TOOL_SYSTEM_PROMPT = (
    "You are a customer service agent. When the user asks about an order, "
    "you MUST use the get_order_status tool. Never make up order information."
)
MAX_RESPONSE_TICKS = 75  # 15 seconds at 200ms ticks

# Speech audio samples of different lengths. Some provider VADs require a
# minimum speech duration to trigger (e.g., xAI needs ~1s).
SPEECH_AUDIO = [
    pytest.param("hello.ulaw", id="short-720ms"),
    pytest.param("hi_how_are_you.ulaw", id="medium-1120ms"),
]


# =============================================================================
# Audio helpers
# =============================================================================


def load_telephony_audio(filename: str) -> bytes:
    """Load a pre-converted telephony audio file (8kHz mu-law).

    Files are raw mu-law bytes at 8kHz (no header). Generated from WAV source
    files by generate_test_audio.py.
    """
    filepath = TESTDATA_DIR / filename
    if not filepath.exists():
        pytest.skip(f"Test audio not found: {filepath}. Run generate_test_audio.py.")
    return filepath.read_bytes()


def chunk_audio(
    audio: bytes, chunk_size: int, trailing_silence_chunks: int = 5
) -> List[bytes]:
    """Split audio into fixed-size chunks with trailing silence for VAD.

    Args:
        audio: Raw audio bytes.
        chunk_size: Size of each chunk in bytes.
        trailing_silence_chunks: Number of silence chunks to append after speech.
            Helps VAD detect end-of-utterance.
    """
    silence_byte = TELEPHONY_ULAW_SILENCE
    chunks = []
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i : i + chunk_size]
        if len(chunk) < chunk_size:
            chunk = chunk + silence_byte * (chunk_size - len(chunk))
        chunks.append(chunk)
    for _ in range(trailing_silence_chunks):
        chunks.append(silence_byte * chunk_size)
    return chunks


def make_silence(tick_duration_ms: int = TICK_DURATION_MS) -> bytes:
    """Generate one tick of mu-law silence at 8kHz."""
    num_bytes = int(8000 * tick_duration_ms / 1000)
    return TELEPHONY_ULAW_SILENCE * num_bytes


def _make_order_tool() -> Tool:
    """Create the get_order_status tool used in tool-call tests."""

    def get_order_status(order_id: str) -> str:
        """Get the status of a customer order by order ID.

        Use this whenever the user asks about an order.

        Args:
            order_id: The order ID to look up.
        """
        return f"Order {order_id} is shipped and arriving tomorrow."

    return Tool(get_order_status)


# =============================================================================
# Assertion helpers
# =============================================================================


def assert_audio_capping(result: TickResult, adapter: DiscreteTimeAdapter) -> None:
    """Assert that agent audio is capped to bytes_per_tick."""
    total = sum(len(data) for data, _ in result.agent_audio_chunks)
    assert total <= adapter.bytes_per_tick, (
        f"Audio capping violated: {total} bytes > {adapter.bytes_per_tick} bytes_per_tick"
    )


def assert_played_audio_length(
    result: TickResult, adapter: DiscreteTimeAdapter
) -> None:
    """Assert that get_played_agent_audio returns exactly bytes_per_tick."""
    assert result.bytes_per_tick > 0, (
        f"TickResult.bytes_per_tick is {result.bytes_per_tick} — "
        "adapter.run_tick() returned a TickResult without setting bytes_per_tick"
    )
    played = result.get_played_agent_audio()
    assert len(played) == adapter.bytes_per_tick, (
        f"get_played_agent_audio returned {len(played)} bytes, "
        f"expected {adapter.bytes_per_tick}"
    )


# Upper bound for tick wall-clock duration (same as test_tick_duration_bounds)
TICK_DURATION_MAX_FACTOR = 1.5


class TickTimer:
    """Collects tick wall-clock durations and asserts the timing invariant."""

    def __init__(self):
        self.timings: List[float] = []

    def run_tick(
        self,
        adapter: DiscreteTimeAdapter,
        user_audio: bytes,
        tick_number: int,
    ) -> TickResult:
        start = time.time()
        result = adapter.run_tick(user_audio, tick_number=tick_number)
        elapsed_ms = (time.time() - start) * 1000
        self.timings.append(elapsed_ms)
        max_ms = TICK_DURATION_MS * TICK_DURATION_MAX_FACTOR
        assert elapsed_ms <= max_ms, (
            f"Tick {tick_number} took {elapsed_ms:.0f}ms, "
            f"expected <= {max_ms:.0f}ms (tick_duration={TICK_DURATION_MS}ms × "
            f"{TICK_DURATION_MAX_FACTOR})"
        )
        return result

    def print_diagnostics(self, label: str = "") -> None:
        if not self.timings:
            return
        sorted_t = sorted(self.timings)
        p95_idx = int(len(sorted_t) * 0.95)
        p95 = sorted_t[min(p95_idx, len(sorted_t) - 1)]
        over = sum(1 for t in self.timings if t > TICK_DURATION_MS * TICK_DURATION_MAX_FACTOR)
        print(
            f"\n  [{label}] {len(self.timings)} ticks: "
            f"min={min(self.timings):.0f}ms, "
            f"avg={sum(self.timings)/len(self.timings):.0f}ms, "
            f"p95={p95:.0f}ms, max={max(self.timings):.0f}ms, "
            f"over-budget={over}"
        )


def run_ticks_until(
    adapter: DiscreteTimeAdapter,
    audio_chunks: List[bytes],
    timer: TickTimer,
    *,
    max_ticks: int = MAX_RESPONSE_TICKS,
    stop_when: Optional[str] = None,
) -> List[TickResult]:
    """Send audio chunks then silence ticks, collecting results.

    Args:
        adapter: The adapter to run ticks on.
        audio_chunks: Audio chunks to send (one per tick).
        timer: TickTimer that records durations and asserts timing invariant.
        max_ticks: Maximum total ticks to run.
        stop_when: Stop condition -- "agent_audio" stops when agent produces audio,
            "tool_call" stops when a tool call is detected.

    Returns:
        List of all TickResults collected.
    """
    silence = make_silence()
    results: List[TickResult] = []

    for tick in range(max_ticks):
        user_audio = audio_chunks[tick] if tick < len(audio_chunks) else silence
        result = timer.run_tick(adapter, user_audio, tick + 1)
        results.append(result)

        assert_audio_capping(result, adapter)
        assert_played_audio_length(result, adapter)

        if stop_when == "agent_audio" and result.agent_audio_bytes > 0:
            return results
        if stop_when == "tool_call" and result.tool_calls:
            return results

    return results


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(params=PROVIDERS)
def provider_name(request) -> str:
    """Parameterized provider name."""
    return request.param


CASCADED_CONFIG_ALIASES = {
    "livekit-thinking": ("livekit", "openai-thinking"),
}


@pytest.fixture
def adapter(provider_name: str):
    """Create, yield, and teardown a DiscreteTimeAdapter."""
    real_provider = provider_name
    cascaded_config = None

    if provider_name in CASCADED_CONFIG_ALIASES:
        from tau2.voice.audio_native.livekit.config import CASCADED_CONFIGS

        real_provider, config_name = CASCADED_CONFIG_ALIASES[provider_name]
        cascaded_config = CASCADED_CONFIGS[config_name]

    adapter, _model = create_adapter(
        real_provider,
        tick_duration_ms=TICK_DURATION_MS,
        cascaded_config=cascaded_config,
    )
    yield adapter
    if adapter.is_connected:
        adapter.disconnect()


@pytest.fixture
def connected_adapter(adapter: DiscreteTimeAdapter):
    """An adapter that is already connected with a basic system prompt."""
    adapter.connect(
        system_prompt=SYSTEM_PROMPT,
        tools=[],
        vad_config=None,
        modality="audio",
    )
    yield adapter


@pytest.fixture
def timer(request, provider_name: str):
    """Tick timer that records durations and prints diagnostics at teardown."""
    t = TickTimer()
    yield t
    t.print_diagnostics(f"{provider_name}/{request.node.name}")


# =============================================================================
# Tests: Connection lifecycle
# =============================================================================


class TestConnection:
    """Verify the adapter can connect and disconnect cleanly."""

    def test_connect_disconnect(self, adapter: DiscreteTimeAdapter):
        """Connect, verify connected, run a silence tick, disconnect."""
        assert not adapter.is_connected

        adapter.connect(
            system_prompt=SYSTEM_PROMPT,
            tools=[],
            vad_config=None,
            modality="audio",
        )
        assert adapter.is_connected

        result = adapter.run_tick(make_silence(), tick_number=1)
        assert result.tick_number == 1
        assert_played_audio_length(result, adapter)

        adapter.disconnect()
        assert not adapter.is_connected

    def test_reconnect_after_disconnect(self, adapter: DiscreteTimeAdapter):
        """Connect, disconnect, then connect again and run a tick."""
        adapter.connect(
            system_prompt=SYSTEM_PROMPT,
            tools=[],
            vad_config=None,
            modality="audio",
        )
        assert adapter.is_connected
        adapter.disconnect()
        assert not adapter.is_connected

        adapter.connect(
            system_prompt=SYSTEM_PROMPT,
            tools=[],
            vad_config=None,
            modality="audio",
        )
        assert adapter.is_connected

        result = adapter.run_tick(make_silence(), tick_number=1)
        assert result.tick_number == 1
        assert_played_audio_length(result, adapter)


# =============================================================================
# Tests: Tick timing
# =============================================================================


class TestTickTiming:
    """Verify ticks respect wall-clock timing bounds."""

    def test_tick_duration_bounds(self, connected_adapter: DiscreteTimeAdapter):
        """Silence ticks should take ~tick_duration_ms: not too fast, not too slow."""
        silence = make_silence()
        times = []
        for tick in range(5):
            start = time.time()
            connected_adapter.run_tick(silence, tick_number=tick + 1)
            elapsed_ms = (time.time() - start) * 1000
            times.append(elapsed_ms)

        for i, t in enumerate(times):
            assert t >= TICK_DURATION_MS * 0.9, (
                f"Tick {i + 1} too fast: {t:.0f}ms, "
                f"expected >= {TICK_DURATION_MS * 0.9:.0f}ms"
            )
            assert t <= TICK_DURATION_MS * 1.5, (
                f"Tick {i + 1} too slow: {t:.0f}ms, "
                f"expected <= {TICK_DURATION_MS * 1.5:.0f}ms"
            )


# =============================================================================
# Tests: Single turn reply
# =============================================================================


class TestSingleTurn:
    """Verify the adapter can produce a single-turn audio response."""

    @pytest.mark.parametrize("audio_file", SPEECH_AUDIO)
    def test_single_turn_reply(
        self, connected_adapter: DiscreteTimeAdapter, audio_file: str, timer: TickTimer
    ):
        """Send speech audio, verify agent responds with audio and transcript."""
        audio = load_telephony_audio(audio_file)
        chunks = chunk_audio(audio, connected_adapter.bytes_per_tick)

        results = run_ticks_until(
            connected_adapter, chunks, timer, stop_when="agent_audio"
        )

        got_audio = any(r.agent_audio_bytes > 0 for r in results)
        assert got_audio, (
            f"Agent did not produce audio within {len(results)} ticks "
            f"({len(results) * TICK_DURATION_MS}ms) for {audio_file}"
        )

        # Drain a few more ticks to let transcript arrive (may lag behind audio)
        silence = make_silence()
        for tick in range(10):
            result = timer.run_tick(
                connected_adapter, silence, len(results) + tick + 1
            )
            results.append(result)
            assert_audio_capping(result, connected_adapter)

        got_transcript = any(r.proportional_transcript for r in results)
        assert got_transcript, (
            f"Agent produced audio but no transcript within {len(results)} ticks "
            f"for {audio_file}"
        )


# =============================================================================
# Tests: Multi turn reply
# =============================================================================


class TestMultiTurn:
    """Verify the adapter handles multiple conversation turns."""

    def test_multi_turn_reply(
        self, connected_adapter: DiscreteTimeAdapter, timer: TickTimer
    ):
        """Two consecutive exchanges, both produce audio responses."""
        t1_audio = load_telephony_audio("hi_how_are_you.ulaw")
        t1_chunks = chunk_audio(t1_audio, connected_adapter.bytes_per_tick)

        # Turn 1: send speech, wait for response
        results_t1 = run_ticks_until(
            connected_adapter, t1_chunks, timer, stop_when="agent_audio"
        )
        got_audio_t1 = any(r.agent_audio_bytes > 0 for r in results_t1)
        assert got_audio_t1, "Turn 1: agent did not produce audio"

        # Let the agent finish responding (drain remaining audio)
        silence = make_silence()
        for tick in range(20):
            result = timer.run_tick(
                connected_adapter, silence, len(results_t1) + tick + 1
            )
            assert_audio_capping(result, connected_adapter)
            assert_played_audio_length(result, connected_adapter)

        # Turn 2: send different audio, wait for response
        t2_audio = load_telephony_audio("help_me.ulaw")
        t2_chunks = chunk_audio(t2_audio, connected_adapter.bytes_per_tick)

        tick_offset = len(results_t1) + 20
        results_t2: List[TickResult] = []
        for tick in range(MAX_RESPONSE_TICKS):
            user_audio = t2_chunks[tick] if tick < len(t2_chunks) else silence
            result = timer.run_tick(
                connected_adapter, user_audio, tick_offset + tick + 1
            )
            results_t2.append(result)
            assert_audio_capping(result, connected_adapter)
            assert_played_audio_length(result, connected_adapter)
            if result.agent_audio_bytes > 0:
                break

        got_audio_t2 = any(r.agent_audio_bytes > 0 for r in results_t2)
        assert got_audio_t2, "Turn 2: agent did not produce audio"


# =============================================================================
# Tests: Tool call round-trip
# =============================================================================


class TestToolCall:
    """Verify tool calls work end-to-end."""

    def test_tool_call_round_trip(self, adapter: DiscreteTimeAdapter, timer: TickTimer):
        """Send order status audio with tool configured, verify round-trip."""
        tool = _make_order_tool()

        adapter.connect(
            system_prompt=TOOL_SYSTEM_PROMPT,
            tools=[tool],
            vad_config=None,
            modality="audio",
        )

        audio = load_telephony_audio("check_order_12345.ulaw")
        chunks = chunk_audio(audio, adapter.bytes_per_tick)

        # Phase 1: send audio and wait for tool call
        results = run_ticks_until(adapter, chunks, timer, stop_when="tool_call")

        tool_call_results = [r for r in results if r.tool_calls]
        assert tool_call_results, (
            f"No tool call received within {len(results)} ticks "
            f"({len(results) * TICK_DURATION_MS}ms)"
        )

        tc = tool_call_results[0].tool_calls[0]
        assert tc.name == "get_order_status", (
            f"Expected get_order_status, got {tc.name}"
        )

        # Phase 2: send tool result, verify agent responds with audio
        adapter.send_tool_result(
            call_id=tc.id,
            result='{"status": "shipped", "estimated_delivery": "tomorrow"}',
        )

        silence = make_silence()
        tick_offset = len(results)
        got_response_audio = False

        for tick in range(MAX_RESPONSE_TICKS):
            result = timer.run_tick(adapter, silence, tick_offset + tick + 1)
            assert_audio_capping(result, adapter)
            assert_played_audio_length(result, adapter)
            if result.agent_audio_bytes > 0:
                got_response_audio = True
                break

        assert got_response_audio, "Agent did not produce audio after tool result"


# =============================================================================
# Tests: Barge-in / interruption
# =============================================================================


BARGE_IN_SYSTEM_PROMPT = (
    "You are a helpful assistant. When asked anything, give a very long, "
    "detailed response. Explain thoroughly with multiple paragraphs. "
    "Never give short answers."
)

# Minimum agent audio ticks to confirm sustained speech before interrupting
MIN_AGENT_AUDIO_TICKS = 25  # 5 seconds at 200ms ticks
# How many ticks of agent audio to let play before sending interrupt
INTERRUPT_AFTER_TICKS = 5  # 1 second at 200ms ticks
# Consecutive silence ticks required to confirm agent yielded
SILENCE_TICKS_REQUIRED = 3  # 0.6 seconds
# Max trailing audio ticks allowed after interruption event
MAX_TRAILING_AUDIO_TICKS = 3  # ~0.6 seconds grace for pipeline flush


class TestBargeIn:
    """Verify the adapter handles user interruptions and actually yields."""

    @pytest.mark.parametrize("audio_file", SPEECH_AUDIO)
    def test_barge_in_detected(
        self, adapter: DiscreteTimeAdapter, audio_file: str, timer: TickTimer
    ):
        """Basic check: interruption event fires when user speaks over agent."""
        adapter.connect(
            system_prompt=BARGE_IN_SYSTEM_PROMPT,
            tools=[],
            vad_config=None,
            modality="audio",
        )

        trigger_audio = load_telephony_audio(audio_file)
        trigger_chunks = chunk_audio(trigger_audio, adapter.bytes_per_tick)

        results = run_ticks_until(
            adapter, trigger_chunks, timer, stop_when="agent_audio"
        )
        assert any(r.agent_audio_bytes > 0 for r in results), (
            f"Agent never started speaking for {audio_file}"
        )

        interrupt_audio = load_telephony_audio("help_me.ulaw")
        interrupt_chunks = chunk_audio(interrupt_audio, adapter.bytes_per_tick)

        tick_offset = len(results)
        truncation_detected = False

        for tick in range(MAX_RESPONSE_TICKS):
            user_audio = (
                interrupt_chunks[tick]
                if tick < len(interrupt_chunks)
                else make_silence()
            )
            result = timer.run_tick(adapter, user_audio, tick_offset + tick + 1)
            assert_audio_capping(result, adapter)

            if result.was_truncated:
                truncation_detected = True
                break
            if any(v in ("speech_started", "interrupted") for v in result.vad_events):
                truncation_detected = True
                break

        assert truncation_detected, (
            "Barge-in not detected: no was_truncated or speech_started/interrupted "
            f"event within {MAX_RESPONSE_TICKS} ticks after sending interrupting speech"
        )

    def test_barge_in_baseline(self, adapter: DiscreteTimeAdapter, timer: TickTimer):
        """Verify agent produces sustained audio (5s+) with the barge-in prompt.

        This establishes that the prompt reliably triggers a long response,
        so test_barge_in_agent_yields can trust the agent would have kept
        speaking if not interrupted.
        """
        adapter.connect(
            system_prompt=BARGE_IN_SYSTEM_PROMPT,
            tools=[],
            vad_config=None,
            modality="audio",
        )

        trigger_audio = load_telephony_audio("hi_how_are_you.ulaw")
        trigger_chunks = chunk_audio(trigger_audio, adapter.bytes_per_tick)

        results = run_ticks_until(
            adapter, trigger_chunks, timer, stop_when="agent_audio"
        )
        assert any(r.agent_audio_bytes > 0 for r in results), (
            "Agent never started speaking"
        )

        tick_num = len(results)
        silence = make_silence()
        agent_audio_ticks = 0

        for _ in range(MAX_RESPONSE_TICKS):
            tick_num += 1
            result = timer.run_tick(adapter, silence, tick_num)
            assert_audio_capping(result, adapter)
            if result.agent_audio_bytes > 0:
                agent_audio_ticks += 1
            if agent_audio_ticks >= MIN_AGENT_AUDIO_TICKS:
                break

        assert agent_audio_ticks >= MIN_AGENT_AUDIO_TICKS, (
            f"Agent only produced {agent_audio_ticks} ticks of audio "
            f"({agent_audio_ticks * TICK_DURATION_MS}ms), "
            f"need at least {MIN_AGENT_AUDIO_TICKS} ticks "
            f"({MIN_AGENT_AUDIO_TICKS * TICK_DURATION_MS}ms)"
        )

    def test_barge_in_agent_yields(
        self, adapter: DiscreteTimeAdapter, timer: TickTimer
    ):
        """Full interruption lifecycle: agent speaks, user interrupts, agent yields.

        Relies on test_barge_in_baseline confirming the prompt produces 5s+
        of agent audio. This test interrupts after 1 second and verifies
        the agent actually stops.
        """
        adapter.connect(
            system_prompt=BARGE_IN_SYSTEM_PROMPT,
            tools=[],
            vad_config=None,
            modality="audio",
        )

        # Phase 1: trigger long agent response
        trigger_audio = load_telephony_audio("hi_how_are_you.ulaw")
        trigger_chunks = chunk_audio(trigger_audio, adapter.bytes_per_tick)

        results = run_ticks_until(
            adapter, trigger_chunks, timer, stop_when="agent_audio"
        )
        assert any(r.agent_audio_bytes > 0 for r in results), (
            "Agent never started speaking"
        )

        # Phase 2: let agent speak for INTERRUPT_AFTER_TICKS (~1 second)
        tick_num = len(results)
        silence = make_silence()
        agent_audio_ticks = 0

        for _ in range(INTERRUPT_AFTER_TICKS + 5):
            tick_num += 1
            result = timer.run_tick(adapter, silence, tick_num)
            assert_audio_capping(result, adapter)
            if result.agent_audio_bytes > 0:
                agent_audio_ticks += 1
            if agent_audio_ticks >= INTERRUPT_AFTER_TICKS:
                break

        assert agent_audio_ticks >= INTERRUPT_AFTER_TICKS, (
            f"Agent only produced {agent_audio_ticks} ticks of audio, "
            f"need at least {INTERRUPT_AFTER_TICKS} before interrupting"
        )

        # Phase 3: send interrupt speech (~3 seconds)
        interrupt_audio = load_telephony_audio("check_order_12345.ulaw")
        interrupt_chunks = chunk_audio(interrupt_audio, adapter.bytes_per_tick)

        interruption_tick = None
        trailing_audio_ticks = 0
        consecutive_silence = 0

        for tick in range(MAX_RESPONSE_TICKS):
            user_audio = (
                interrupt_chunks[tick] if tick < len(interrupt_chunks) else silence
            )
            tick_num += 1
            result = timer.run_tick(adapter, user_audio, tick_num)
            assert_audio_capping(result, adapter)

            # Check for interruption event
            if interruption_tick is None:
                if result.was_truncated or any(
                    v in ("speech_started", "interrupted") for v in result.vad_events
                ):
                    interruption_tick = tick

            # After interruption, track silence
            if interruption_tick is not None:
                if result.agent_audio_bytes > 0:
                    trailing_audio_ticks += 1
                    consecutive_silence = 0
                else:
                    consecutive_silence += 1

                if consecutive_silence >= SILENCE_TICKS_REQUIRED:
                    break

        # Assert interruption was detected
        assert interruption_tick is not None, (
            "Barge-in not detected: no interruption event within "
            f"{MAX_RESPONSE_TICKS} ticks"
        )

        # Assert agent actually stopped (silence follows)
        assert consecutive_silence >= SILENCE_TICKS_REQUIRED, (
            f"Agent did not yield after interruption: only {consecutive_silence} "
            f"consecutive silence ticks (need {SILENCE_TICKS_REQUIRED}). "
            f"Trailing audio ticks after event: {trailing_audio_ticks}"
        )

        # Report timing for diagnostic insight
        trailing_ms = trailing_audio_ticks * TICK_DURATION_MS
        interrupt_delay_ms = interruption_tick * TICK_DURATION_MS
        print(
            f"\n  Interruption detected at tick +{interruption_tick} "
            f"({interrupt_delay_ms}ms after interrupt speech started)"
            f"\n  Trailing agent audio after event: {trailing_audio_ticks} ticks "
            f"({trailing_ms}ms)"
        )
