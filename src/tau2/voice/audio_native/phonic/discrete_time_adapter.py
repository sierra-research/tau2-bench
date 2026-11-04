"""Discrete-time adapter for the Phonic speech-to-speech API.

Phonic accepts and returns G.711 mu-law at 8 kHz, which matches tau2's
telephony format, so audio passes through without conversion.

Phonic processes audio as a live call, so it expects a continuous real-time
stream. Between ticks the orchestrator can block for seconds (user simulator
LLM and TTS), so a background pump streams queued user audio, or mu-law
silence when none is queued, at a steady realtime cadence and reads agent
events continuously. Each tick hands its audio to the pump and processes
the events that arrived during the tick.

Phonic also streams silence frames in real time between utterances. The pump
receives them while the orchestrator is blocked, so playing them would queue
up silence and delay every later reply by the length of each stall. The
adapter therefore plays only frames between assistant_started_speaking and
assistant_finished_speaking, and classifies each tick as speech from the audio
actually played, as in the OpenAI Live adapter.
"""

import asyncio
import audioop
import base64
import time
from typing import Any, List, Optional

from loguru import logger

from tau2.config import (
    DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT,
    DEFAULT_AUDIO_NATIVE_DISCONNECT_TIMEOUT,
    DEFAULT_AUDIO_NATIVE_TICK_TIMEOUT_BUFFER,
    DEFAULT_PHONIC_MODEL,
    TELEPHONY_ULAW_SILENCE,
)
from tau2.data_model.message import ToolCall
from tau2.data_model.usage import UsageRecord
from tau2.environment.tool import Tool
from tau2.voice.audio_native.adapter import DiscreteTimeAdapter
from tau2.voice.audio_native.async_loop import BackgroundAsyncLoop
from tau2.voice.audio_native.phonic import events as ev
from tau2.voice.audio_native.phonic.provider import PhonicProvider
from tau2.voice.audio_native.tick_result import TickResult, UtteranceTranscript

# Peak PCM16 amplitude above which delivered agent audio counts as speech
# (same test as the OpenAI Live adapter).
_SPEECH_PEAK_THRESHOLD = 32


def mark_played_speech(result: TickResult) -> None:
    """Classify the tick as speech from the agent audio actually played."""
    pcm = audioop.ulaw2lin(result.agent_audio_data, 2)
    result.contains_speech = bool(pcm) and audioop.max(pcm, 2) > _SPEECH_PEAK_THRESHOLD


# tau2 reasoning effort -> Phonic intelligence level.
_INTELLIGENCE_LEVELS = {
    "minimal": "standard",
    "low": "standard",
    "medium": "standard",
    "high": "high",
    "xhigh": "high",
}


class DiscreteTimePhonicAdapter(DiscreteTimeAdapter):
    """Tick-based adapter for Phonic speech-to-speech with a realtime audio pump."""

    # How long to wait for Phonic to confirm the conversation, and how many
    # times to retry a connection Phonic rejects before the call starts
    # (for example when it is at capacity).
    _READY_TIMEOUT_S = 20.0
    _CONNECT_ATTEMPTS = 5

    def __init__(
        self,
        tick_duration_ms: int,
        send_audio_instant: bool = False,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        provider: Optional[PhonicProvider] = None,
        language: Optional[str] = None,
    ):
        """Initialize the discrete-time Phonic adapter.

        Args:
            tick_duration_ms: Duration of each tick in milliseconds. Must be > 0.
            send_audio_instant: Accepted for interface compatibility; the pump
                always streams in real time.
            model: Phonic model, sent as phonic_model (e.g. phonic-1).
            reasoning_effort: tau2 reasoning effort, mapped to Phonic's
                intelligence level (high/xhigh -> high, lower -> standard).
            provider: Optional provider instance. Created lazily if not provided.
            language: The call's language (e.g. "hi"), sent as Phonic's default_language.
        """
        if (
            reasoning_effort is not None
            and reasoning_effort not in _INTELLIGENCE_LEVELS
        ):
            raise ValueError(
                f"Unsupported reasoning_effort for Phonic: {reasoning_effort}"
            )
        super().__init__(tick_duration_ms, send_audio_instant=send_audio_instant)
        self._chunk_size = int(
            self.audio_format.bytes_per_second * self._voip_interval_ms / 1000
        )
        self._model = model
        self._language = language
        self._intelligence_level = _INTELLIGENCE_LEVELS.get(reasoning_effort or "")
        self._provider = provider
        self._owns_provider = provider is None
        self._bg_loop = BackgroundAsyncLoop()
        self._connected = False
        self._session_counter = 0
        self._item_counter = 0
        # True between assistant_started_speaking and assistant_finished_speaking.
        self._assistant_speaking = False

        self._pumping = False
        self._pump_gen = 0
        self._send_buf = bytearray()
        self._event_buf: List[Any] = []
        self._pump_tasks: List[asyncio.Task] = []
        self._pump_ready: Optional[asyncio.Event] = None
        self._pump_error: Optional[BaseException] = None

    @property
    def provider(self) -> PhonicProvider:
        if self._provider is None:
            self._provider = PhonicProvider(
                phonic_model=self._model,
                intelligence_level=self._intelligence_level,
                language=self._language,
            )
        return self._provider

    @property
    def is_connected(self) -> bool:
        return self._connected and self._bg_loop.is_running

    def connect(
        self,
        system_prompt: str,
        tools: List[Tool],
        vad_config: Any = None,
        modality: str = "audio",
    ) -> None:
        """Connect, then wait until Phonic confirms the conversation.

        Retries with backoff only if the conversation never starts.
        """
        if self._connected:
            logger.warning("Already connected, disconnecting first")
            self.disconnect()

        self._bg_loop.start()
        try:
            for attempt in range(1, self._CONNECT_ATTEMPTS + 1):
                self._bg_loop.run_coroutine(
                    self.provider.connect(system_prompt, tools),
                    timeout=DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT,
                )
                self._connected = True
                self._pumping = True
                self._send_buf = bytearray()
                self._event_buf = []
                self._pump_error = None
                self._bg_loop.run_coroutine(self._start_pump())
                err = self._bg_loop.run_coroutine(
                    self._await_conversation_ready(),
                    timeout=self._READY_TIMEOUT_S + 5,
                )
                if err is None:
                    break
                self._teardown_socket()
                if attempt == self._CONNECT_ATTEMPTS:
                    raise RuntimeError(
                        f"Phonic conversation not established after "
                        f"{attempt} attempts: {err}"
                    )
                delay = min(5 * 2 ** (attempt - 1), 30)
                logger.warning(
                    f"Phonic conversation not established "
                    f"(attempt {attempt}/{self._CONNECT_ATTEMPTS}): {err}; "
                    f"retrying in {delay}s"
                )
                time.sleep(delay)
            self._session_counter += 1
            logger.info(
                f"DiscreteTimePhonicAdapter connected (tick={self.tick_duration_ms}ms)"
            )
        except Exception as e:
            logger.error(f"DiscreteTimePhonicAdapter failed to connect: {e}")
            self._bg_loop.stop()
            raise RuntimeError(f"Failed to connect to Phonic API: {e}") from e

    def _make_audio_usage_record(self) -> Optional[UsageRecord]:
        """Session-audio meter: Phonic bills per minute of conversation.

        In the discrete-time simulation, cumulative user audio equals elapsed
        session audio time. Cumulative semantics per session keep re-reads and
        the final flush at disconnect from double counting.
        """
        if self._cumulative_user_audio_ms <= 0:
            return None
        return UsageRecord(
            provider="phonic",
            model=self._model or DEFAULT_PHONIC_MODEL,
            component="realtime",
            semantics="cumulative",
            scope_id=f"audio-session-{self._session_counter}",
            audio_input_seconds=self._cumulative_user_audio_ms / 1000,
        )

    def get_usage_records(self) -> List[UsageRecord]:
        """Ledger records plus the live session-audio meter (if connected)."""
        records = super().get_usage_records()
        live_audio = self._make_audio_usage_record()
        if live_audio is not None:
            records.append(live_audio)
        return records

    def disconnect(self) -> None:
        if not self._connected:
            return
        # Flush the session-audio meter before the counter is reset below.
        audio_record = self._make_audio_usage_record()
        if audio_record is not None:
            self._usage_records.append(audio_record)
        self._pumping = False
        if self._bg_loop.is_running:
            try:
                self._bg_loop.run_coroutine(
                    self._async_disconnect(),
                    timeout=DEFAULT_AUDIO_NATIVE_DISCONNECT_TIMEOUT,
                )
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")
        self._bg_loop.stop()
        self._connected = False
        self._tick_count = 0
        self._cumulative_user_audio_ms = 0
        self._assistant_speaking = False
        self.clear_buffers()
        logger.info("DiscreteTimePhonicAdapter disconnected")

    async def _async_disconnect(self) -> None:
        await self._stop_pump()
        if self._owns_provider and self._provider is not None:
            await self.provider.disconnect()

    def _teardown_socket(self) -> None:
        """Drop a rejected connection but keep the background loop for a retry."""
        self._pumping = False
        try:
            self._bg_loop.run_coroutine(self._stop_pump())
            self._bg_loop.run_coroutine(
                self.provider.disconnect(),
                timeout=DEFAULT_AUDIO_NATIVE_DISCONNECT_TIMEOUT,
            )
        except Exception as e:
            logger.debug(f"Phonic teardown after rejection: {e}")
        self._connected = False

    def run_tick(
        self, user_audio: bytes, tick_number: Optional[int] = None
    ) -> TickResult:
        if not self.is_connected:
            raise RuntimeError("Not connected to Phonic API. Call connect() first.")
        if tick_number is None:
            tick_number = self._tick_count
        self._tick_count = tick_number + 1
        try:
            result = self._bg_loop.run_coroutine(
                self._async_run_tick(user_audio, tick_number),
                timeout=self.tick_duration_ms / 1000
                + DEFAULT_AUDIO_NATIVE_TICK_TIMEOUT_BUFFER,
            )
        except Exception as e:
            logger.error(f"Error in run_tick (tick={tick_number}): {e}")
            raise
        mark_played_speech(result)
        return result

    async def _flush_pending_tool_results(self) -> None:
        for (
            call_id,
            result_str,
            _request_response,
            _is_error,
        ) in self._pending_tool_results:
            await self.provider.send_tool_result(call_id, result_str)
        self._pending_tool_results.clear()

    async def _execute_tick(
        self,
        user_audio: bytes,
        tick_number: int,
        result: TickResult,
        tick_start: float,
    ) -> None:
        if self._pump_error is not None:
            # An established call dropped. Raising marks the simulation as an
            # infrastructure error, as the other providers' adapters do.
            raise RuntimeError(f"Phonic connection lost: {self._pump_error}")
        self._send_buf.extend(user_audio)
        await asyncio.sleep(self.tick_duration_ms / 1000)
        events, self._event_buf = self._event_buf, []
        for event in events:
            self._process_event(result, event)

    # -----------------------------------------------------------------------
    # Realtime pump
    # -----------------------------------------------------------------------

    async def _start_pump(self) -> None:
        self._pump_gen += 1
        self._pump_ready = asyncio.Event()
        self._pump_tasks = [
            asyncio.create_task(self._pump_send(self._pump_gen)),
            asyncio.create_task(self._pump_recv(self._pump_gen)),
        ]

    async def _stop_pump(self) -> None:
        tasks, self._pump_tasks = self._pump_tasks, []
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _pump_fail(self, gen: int, e: BaseException) -> None:
        # Ignore failures from a superseded (rejected and retried) connection.
        if gen != self._pump_gen or not self._pumping:
            return
        if self._pump_error is None:
            self._pump_error = e
        if self._pump_ready is not None:
            self._pump_ready.set()

    async def _await_conversation_ready(self) -> Optional[str]:
        """Return None once conversation_created arrives, else a reason."""
        try:
            await asyncio.wait_for(self._pump_ready.wait(), self._READY_TIMEOUT_S)
        except asyncio.TimeoutError:
            return f"no conversation_created within {self._READY_TIMEOUT_S:.0f}s"
        return None if self._pump_error is None else str(self._pump_error)

    async def _pump_send(self, gen: int) -> None:
        """Stream queued user audio, or silence, on a realtime schedule."""
        interval = self._voip_interval_ms / 1000
        n = self._chunk_size
        silence = TELEPHONY_ULAW_SILENCE * n
        loop = asyncio.get_running_loop()
        next_t = loop.time()
        while self._pumping and gen == self._pump_gen:
            buf = self._send_buf
            if len(buf) >= n:
                chunk = bytes(buf[:n])
                del buf[:n]
            elif buf:
                chunk = bytes(buf) + silence[len(buf) :]
                buf.clear()
            else:
                chunk = silence
            try:
                await self.provider.send_audio(chunk)
            except Exception as e:
                self._pump_fail(gen, e)
                return
            next_t += interval
            delay = next_t - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            elif delay < -0.2:
                next_t = loop.time()  # fell far behind: resync rather than burst

    async def _pump_recv(self, gen: int) -> None:
        """Collect events continuously, including while the orchestrator blocks."""
        while self._pumping and gen == self._pump_gen:
            try:
                msg = await self.provider.recv_message(timeout=0.1)
            except Exception as e:
                self._pump_fail(gen, e)
                return
            if msg is None or isinstance(msg, (bytes, bytearray)):
                continue
            if getattr(msg, "type", None) == ev.CONVERSATION_CREATED:
                self._pump_ready.set()
            self._event_buf.append(msg)

    # -----------------------------------------------------------------------
    # Events
    # -----------------------------------------------------------------------

    def _next_item_id(self) -> str:
        self._item_counter += 1
        return f"phonic_item_{self._item_counter}"

    def _process_event(self, result: TickResult, event: Any) -> None:
        result.events.append(event)
        etype = getattr(event, "type", None)

        if etype == ev.AUDIO_CHUNK:
            if not self._assistant_speaking:
                # Silence between utterances (see module docstring).
                return
            audio_bytes = base64.b64decode(event.audio)
            if self._current_item_id is None:
                self._current_item_id = self._next_item_id()
            item_id = self._current_item_id
            if result.skip_item_id is not None and item_id == result.skip_item_id:
                result.truncated_audio_bytes += len(audio_bytes)
                return
            if audio_bytes:
                result.agent_audio_chunks.append((audio_bytes, item_id))
            utt = self._utterance_transcripts.setdefault(
                item_id, UtteranceTranscript(item_id=item_id)
            )
            if audio_bytes:
                utt.add_audio(len(audio_bytes))
            if event.text:
                utt.add_transcript(event.text)

        elif etype == ev.TOOL_CALL:
            result.tool_calls.append(
                ToolCall(
                    id=event.tool_call_id,
                    name=event.tool_name,
                    arguments=event.parameters,
                )
            )

        elif etype == ev.USER_STARTED_SPEAKING:
            # Barge-in: drop buffered and in-progress agent audio.
            result.vad_events.append("speech_started")
            if self._buffered_agent_audio:
                result.truncated_audio_bytes += sum(
                    len(c[0]) for c in self._buffered_agent_audio
                )
                self._buffered_agent_audio.clear()
            result.was_truncated = True
            result.skip_item_id = self._current_item_id
            self._assistant_speaking = False

        elif etype == ev.USER_FINISHED_SPEAKING:
            result.vad_events.append("speech_stopped")

        elif etype == ev.ASSISTANT_STARTED_SPEAKING:
            self._assistant_speaking = True
            self._current_item_id = self._next_item_id()

        elif etype == ev.ASSISTANT_FINISHED_SPEAKING:
            self._assistant_speaking = False

        elif etype == ev.CONVERSATION_CREATED:
            self.provider.session_id = event.conversation_id
            logger.info(f"Phonic conversation_id={event.conversation_id}")

        elif etype == ev.ERROR:
            logger.error(f"Phonic error event: {event}")
