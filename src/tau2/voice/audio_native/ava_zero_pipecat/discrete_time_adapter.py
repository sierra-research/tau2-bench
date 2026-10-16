"""Discrete-time adapter for the Ava Zero voice agent over Pipecat RTVI."""

from __future__ import annotations

import asyncio
import html
from bisect import bisect_right
from collections import deque
from typing import Any, List, Optional

from loguru import logger

from tau2.config import (
    DEFAULT_AUDIO_NATIVE_DISCONNECT_TIMEOUT,
    DEFAULT_AUDIO_NATIVE_TICK_TIMEOUT_BUFFER,
    DEFAULT_AVA_ZERO_INPUT_SAMPLE_RATE,
    DEFAULT_AVA_ZERO_MODEL,
)
from tau2.data_model.audio import TELEPHONY_AUDIO_FORMAT, AudioFormat
from tau2.data_model.message import ToolCall
from tau2.environment.tool import Tool
from tau2.voice.audio_native.adapter import DiscreteTimeAdapter
from tau2.voice.audio_native.async_loop import BackgroundAsyncLoop
from tau2.voice.audio_native.audio_converter import StreamingTelephonyConverter
from tau2.voice.audio_native.ava_zero_pipecat.config import AvaZeroPipecatConfig
from tau2.voice.audio_native.ava_zero_pipecat.provider import (
    AgentAudioEvent,
    AgentLLMStartedEvent,
    AgentLLMTextEvent,
    AgentPlaybackEvent,
    AgentStartedSpeakingEvent,
    AgentWordEvent,
    AvaZeroBootError,
    AvaZeroPipecatProvider,
    AvaZeroProtocolError,
    ProviderErrorEvent,
    SpeechEvent,
    ToolCallEvent,
    TranscriptEvent,
)
from tau2.voice.audio_native.tick_result import TickResult

_ITEM_ID = "ava_zero"
# How far past the last matched word a spoken word may land. It skips a word
# or two the TTS alignment dropped. A word never matches past the end of the
# sentence being spoken: that would mark unaired text as heard.
_WORD_MATCH_SLACK_CHARS = 16
# Shorter words ("I", "a", "to") match only the very next word, never skip.
_WORD_SKIP_MIN_CHARS = 3


def _unescape(text: str) -> str:
    """Ava Zero HTML-escapes some text, sometimes twice ("I&amp;#x27;m")."""

    for _ in range(3):
        unescaped = html.unescape(text)
        if unescaped == text:
            break
        text = unescaped
    return text


class _SpokenTranscript:
    """Reveal the agent's sentence text only as far as it has been aired.

    Ava Zero sends each sentence (``TranscriptEvent``) as soon as the LLM
    writes it, ahead of the audio, and streams the audio at about real time.
    Releasing text in proportion to audio played / audio received over the
    whole call therefore showed every sentence on its first audio chunk, and
    the simulated caller answered questions it had not heard yet.

    Text is released only on evidence that it aired, once the audio that
    carried the evidence has been played:

    - a spoken word (``AgentWordEvent``, sent by the TTS clock as it plays),
      matched in order within the sentence being spoken;
    - the played caption (``AgentPlaybackEvent``) Ava Zero reports when an
      utterance ends, which also covers words the alignment dropped.

    Without evidence text waits; it is never guessed. The text shown is always
    the sentence text, the words only mark progress through it.

    Implements the ``get_transcript_for_audio`` protocol that
    ``get_proportional_transcript`` calls on the audio played each tick.
    """

    def __init__(self, due_slack_bytes: int = 0) -> None:
        # Audio the adapter may be holding back at an utterance's end; a mark
        # this close to playback counts as played (see _hold_utterance_end).
        self._due_slack = due_slack_bytes
        self.item_id = _ITEM_ID
        self.text = ""
        self._sentence_ends: list[int] = []
        self._audio_received = 0
        self._audio_played = 0
        self._match_cursor = 0
        # Where the utterance now playing starts in ``text``.
        self._utterance_start = 0
        self._spoken_chars = 0
        self._shown_chars = 0
        # (audio position, text length aired once playback reaches it)
        self._marks: deque[tuple[int, int]] = deque()

    def add_text(self, text: str) -> None:
        """Append one bot-transcription sentence, keeping its own spacing.

        Ava Zero's splitter also breaks at decimal points ("$13." then
        "46"), and marks real sentence starts with a leading space. Only the
        first sentence of a response lacks one.
        """

        text = _unescape(text)
        if not self.text.strip():
            text = text.lstrip()
        elif not self.text[-1].isspace() and not text[0].isspace():
            if not (text[0].isdigit() and self.text[-1] in ".,"):
                text = " " + text
        self.text += text
        self._sentence_ends.append(len(self.text))

    def add_audio(self, num_bytes: int) -> None:
        self._audio_received += num_bytes

    def add_word(self, word: str) -> None:
        needle = _unescape(word).strip().lower()
        if not needle:
            return
        sentence = bisect_right(self._sentence_ends, self._match_cursor)
        if sentence == len(self._sentence_ends):
            return  # its sentence has not arrived yet
        haystack = self.text.lower()
        limit = min(
            self._match_cursor + len(needle) + _WORD_MATCH_SLACK_CHARS,
            self._sentence_ends[sentence],
        )
        cursor = self._match_cursor
        if 0 < cursor < len(haystack) and (
            haystack[cursor - 1].isalnum() and haystack[cursor].isalnum()
        ):
            # Mid-word: only the rest of that word may follow ("Wha" + "t").
            if haystack.startswith(needle, cursor):
                self._advance(cursor + len(needle))
            return
        # Otherwise a whole word. A fragment ("Wha") is indistinguishable from
        # a short word inside a longer one ("I" in "If"), so it waits for the
        # next whole word rather than risk jumping ahead of the audio.
        if len(needle.strip(".,!?;:—-")) < _WORD_SKIP_MIN_CHARS:
            # Too common to skip ahead for: only the very next word.
            nxt = cursor
            while nxt < limit and not haystack[nxt].isalnum():
                nxt += 1
            limit = min(limit, nxt + len(needle))
        index = haystack.find(needle, cursor, limit)
        while index >= 0 and not self._is_whole_word(haystack, index, len(needle)):
            index = haystack.find(needle, index + 1, limit)
        if index >= 0:
            self._advance(index + len(needle))

    @staticmethod
    def _is_whole_word(text: str, index: int, length: int) -> bool:
        end = index + length
        return (index == 0 or not text[index - 1].isalnum()) and (
            end == len(text) or not text[end].isalnum()
        )

    def add_playback(self, caption: str) -> None:
        """Ava Zero's caption of what this utterance actually played.

        Text in the caption that the sentences lack (speech no LLM sentence
        produced, or a reply's unpunctuated tail) is inserted where it aired.
        """

        caption = " ".join(_unescape(caption).split())
        if not caption:
            return
        start = self._utterance_start
        while start < len(self.text) and self.text[start].isspace():
            start += 1
        index = self.text.lower().find(caption.lower(), start)
        if index >= 0:
            self._advance(index + len(caption))
        else:
            aired = self.text[start:].lower()
            common = 0
            while (
                common < len(caption)
                and common < len(aired)
                and aired[common] == caption[common].lower()
            ):
                common += 1
            at = start + common
            missing = caption[common:]
            if at > 0 and not self.text[at - 1].isspace() and not missing[0].isspace():
                missing = " " + missing if common == 0 else missing
            if self.text[at:].strip() and not missing[-1].isspace():
                missing += " "
            logger.debug(f"Ava Zero playback adds unsent text: {missing!r}")
            self.text = self.text[:at] + missing + self.text[at:]
            self._sentence_ends = sorted(
                [e if e <= at else e + len(missing) for e in self._sentence_ends]
                + [at + len(missing.rstrip())]
            )
            self._advance(at + len(missing.rstrip()))
        self._utterance_start = self._match_cursor

    def _advance(self, cursor: int) -> None:
        if cursor <= self._match_cursor:
            return
        self._match_cursor = cursor
        self._marks.append((self._audio_received, cursor))

    def interrupt(self) -> None:
        """Forget everything the caller has not heard yet."""

        self.text = self.text[: self._shown_chars]
        self._sentence_ends = [e for e in self._sentence_ends if e < len(self.text)]
        self._sentence_ends.append(len(self.text))
        self._audio_received = self._audio_played
        self._match_cursor = self._spoken_chars = self._shown_chars
        self._utterance_start = self._match_cursor
        self._marks.clear()

    def get_transcript_for_audio(self, audio_bytes_to_play: int) -> str:
        self._audio_played += audio_bytes_to_play
        while self._marks and self._marks[0][0] <= self._audio_played + self._due_slack:
            self._spoken_chars = max(self._spoken_chars, self._marks.popleft()[1])
        end = min(self._spoken_chars, len(self.text))
        text = self.text[self._shown_chars : end]
        self._shown_chars = max(self._shown_chars, end)
        return text


def _mcp_tool_schema(tool: Tool) -> dict[str, Any]:
    """Translate Tau2's OpenAI-shaped schema into the harness MCP profile."""

    function = tool.openai_schema["function"]
    return {
        "name": function["name"],
        "description": function.get("description") or function["name"],
        "inputSchema": function["parameters"],
    }


class DiscreteTimeAvaZeroPipecatAdapter(DiscreteTimeAdapter):
    """Bridge tau2 telephony ticks to Ava Zero's RTVI WebSocket."""

    def __init__(
        self,
        tick_duration_ms: int,
        send_audio_instant: bool = True,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        audio_format: Optional[AudioFormat] = None,
        ava_zero_config: Optional[AvaZeroPipecatConfig] = None,
        provider: Optional[AvaZeroPipecatProvider] = None,
    ):
        if reasoning_effort is not None:
            raise ValueError(
                "Ava Zero reasoning is configured on the Ava Zero server; "
                "--reasoning-effort is not supported"
            )

        config = ava_zero_config or AvaZeroPipecatConfig(
            agent=model or DEFAULT_AVA_ZERO_MODEL
        )
        if model is not None and model != config.agent:
            raise ValueError(
                f"audio-native model {model!r} conflicts with "
                f"Ava Zero agent {config.agent!r}"
            )

        if audio_format is not None and audio_format != TELEPHONY_AUDIO_FORMAT:
            # Output is always converted to 8 kHz u-law and input is read as
            # it; any other format would be mislabelled and mistimed.
            raise ValueError(
                "Ava Zero supports only telephony audio (8 kHz u-law), "
                f"got {audio_format}"
            )

        super().__init__(
            tick_duration_ms,
            audio_format=audio_format,
            send_audio_instant=send_audio_instant,
        )
        self.model = config.agent
        self.reasoning_effort = reasoning_effort
        self.ava_zero_config = config

        self._input_converter = StreamingTelephonyConverter(
            input_sample_rate=DEFAULT_AVA_ZERO_INPUT_SAMPLE_RATE,
        )
        self._output_converters: dict[int, StreamingTelephonyConverter] = {}
        self._chunk_size = int(
            DEFAULT_AVA_ZERO_INPUT_SAMPLE_RATE * 2 * self._voip_interval_ms / 1000
        )

        self._provider = provider
        self._owns_provider = provider is None
        self._bg_loop = BackgroundAsyncLoop()
        self._connected = False
        self._end_hold_bytes = self.audio_format.bytes_per_second // 1000
        self._reset_interruption_state()

    def _reset_interruption_state(self) -> None:
        # After a cut, text and audio of the cut-off response are still in
        # flight. Text is stale until the next response starts generating,
        # audio until its first utterance starts playing.
        self._stale_text = False
        self._stale_audio = False
        # Replacement audio that arrived in the same tick as the cut. The
        # base class discards a cut tick's overflow, so it is replayed from
        # the next tick instead.
        self._audio_after_cut: list[tuple[bytes, Optional[str]]] = []
        # True from an utterance's first audio until its played caption.
        self._awaiting_playback = False
        # See _hold_utterance_end.
        self._held_end: list[tuple[bytes, Optional[str]]] = []
        # Pipecat's RTVI observer builds bot-transcription from bot-llm-text
        # and only empties its buffer when it sends a sentence, so a reply's
        # unsent tail (cut off, or unpunctuated) opens the next reply's first
        # sentence. This mirrors that buffer to strip it.
        self._llm_unsent = ""
        self._llm_carryover = ""

    @property
    def provider(self) -> AvaZeroPipecatProvider:
        """Get the provider, creating it lazily."""

        if self._provider is None:
            self._provider = AvaZeroPipecatProvider(self.ava_zero_config)
        return self._provider

    @property
    def is_connected(self) -> bool:
        """Whether both the adapter and background loop are active."""

        return self._connected and self._bg_loop.is_running

    def connect(
        self,
        system_prompt: str,
        tools: List[Tool],
        vad_config: Any = None,
        modality: str = "audio",
    ) -> None:
        """Connect to Ava Zero and establish a ready, tool-aware session."""

        if modality != "audio":
            raise ValueError("Ava Zero currently supports audio modality only")
        if self._connected:
            self.disconnect()

        try:
            self._bg_loop.start()
            timeout = (
                self.ava_zero_config.boot_attempts
                * (
                    self.ava_zero_config.connect_timeout_seconds
                    + self.ava_zero_config.boot_timeout_seconds
                )
                + self.ava_zero_config.tool_registration_timeout_seconds
                + 5
            )
            self._bg_loop.run_coroutine(
                self._async_connect(system_prompt, tools),
                timeout=timeout,
            )
            self._connected = True
            logger.info(
                "DiscreteTimeAvaZeroPipecatAdapter connected "
                f"(agent={self.model}, tick={self.tick_duration_ms}ms)"
            )
        except Exception as exc:
            if (
                self._bg_loop.is_running
                and self._provider is not None
                and self._provider.is_connected
            ):
                try:
                    self._bg_loop.run_coroutine(
                        self._provider.disconnect(request_cost_report=False),
                        timeout=DEFAULT_AUDIO_NATIVE_DISCONNECT_TIMEOUT,
                    )
                except Exception as cleanup_exc:
                    logger.debug(f"Ava Zero connect cleanup failed: {cleanup_exc}")
            self._bg_loop.stop()
            if isinstance(exc, AvaZeroProtocolError):
                raise
            raise RuntimeError(f"Failed to connect to Ava Zero: {exc}") from exc

    async def _async_connect(
        self,
        system_prompt: str,
        tools: List[Tool],
    ) -> None:
        for attempt in range(1, self.ava_zero_config.boot_attempts + 1):
            try:
                await self.provider.connect(system_prompt=system_prompt)
                break
            except AvaZeroBootError as exc:
                await self.provider.disconnect(request_cost_report=False)
                if attempt == self.ava_zero_config.boot_attempts:
                    logger.error(
                        f"Ava Zero boot failed {attempt} time(s), giving up: {exc}"
                    )
                    raise
                logger.warning(
                    "Ava Zero boot failed "
                    f"(attempt {attempt}/{self.ava_zero_config.boot_attempts}): "
                    f"{exc}; retrying"
                )
                self.provider.reset_for_retry()

        schemas = [_mcp_tool_schema(tool) for tool in tools]
        await self.provider.register_tools(schemas)

    def disconnect(self) -> None:
        """Close the call and reset tick/audio state."""

        if self._connected and self._bg_loop.is_running:
            try:
                self._bg_loop.run_coroutine(
                    self._async_disconnect(),
                    timeout=(
                        DEFAULT_AUDIO_NATIVE_DISCONNECT_TIMEOUT
                        + self.ava_zero_config.cost_report_timeout_seconds
                    ),
                )
            except Exception as exc:
                logger.warning(f"Error disconnecting Ava Zero: {exc}")

        self._bg_loop.stop()
        self._connected = False
        self._tick_count = 0
        self._cumulative_user_audio_ms = 0
        self.clear_buffers()
        self._input_converter.reset()
        for converter in self._output_converters.values():
            converter.reset()
        self._output_converters.clear()
        self._reset_interruption_state()
        logger.info("DiscreteTimeAvaZeroPipecatAdapter disconnected")

    async def _async_disconnect(self) -> None:
        if self._provider is not None and self._provider.is_connected:
            # Results of tools tau2 already executed are still queued for the
            # next tick, which will not come. Deliver them before hanging up.
            try:
                await self._flush_pending_tool_results()
            except Exception as exc:
                logger.warning(f"Ava Zero tool results lost at shutdown: {exc}")
        if self._owns_provider and self._provider is not None:
            await self.provider.disconnect()

    def run_tick(
        self,
        user_audio: bytes,
        tick_number: Optional[int] = None,
    ) -> TickResult:
        """Exchange one simulation tick of audio and events."""

        if not self.is_connected:
            raise RuntimeError("Not connected to Ava Zero. Call connect() first.")
        if tick_number is None:
            tick_number = self._tick_count
        self._tick_count = tick_number + 1
        return self._bg_loop.run_coroutine(
            self._async_run_tick(user_audio, tick_number),
            timeout=(
                self.tick_duration_ms / 1000 + DEFAULT_AUDIO_NATIVE_TICK_TIMEOUT_BUFFER
            ),
        )

    async def _flush_pending_tool_results(self) -> None:
        """Return all queued tau2 tool results before sending tick audio."""

        for (
            call_id,
            result,
            request_response,
            is_error,
        ) in self._pending_tool_results:
            await self.provider.send_tool_result(
                call_id,
                result,
                request_response=request_response,
                is_error=is_error,
            )
        self._pending_tool_results.clear()

    async def _async_run_tick(self, user_audio: bytes, tick_number: int) -> TickResult:
        """Retain post-interruption audio and utterance tails after tick buffering."""

        result = await super()._async_run_tick(user_audio, tick_number)
        if self._audio_after_cut:
            self._buffered_agent_audio[:0] = self._audio_after_cut
            self._audio_after_cut = []
        self._buffered_agent_audio.extend(self._held_end)
        self._held_end = []
        return result

    async def _execute_tick(
        self,
        user_audio: bytes,
        tick_number: int,
        result: TickResult,
        tick_start: float,
    ) -> None:
        """Convert audio, exchange provider events, and populate the tick."""

        provider_audio = self._input_converter.convert_input(user_audio)

        async def send_audio(chunk: bytes) -> None:
            await self.provider.send_audio(
                chunk,
                sample_rate=DEFAULT_AVA_ZERO_INPUT_SAMPLE_RATE,
            )

        async def receive_events() -> list[Any]:
            elapsed = asyncio.get_running_loop().time() - tick_start
            remaining = max(0.01, self.tick_duration_ms / 1000 - elapsed)
            return await self.provider.receive_events_for_duration(remaining)

        tasks = [
            asyncio.create_task(
                self._send_audio_chunked(
                    provider_audio,
                    send_audio,
                    self._chunk_size,
                )
            ),
            asyncio.create_task(receive_events()),
        ]
        try:
            _, events = await asyncio.gather(*tasks)
        finally:
            # gather does not cancel its sibling when audio sending fails.
            # Drain both before disconnect can stop their background loop.
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        for event in events:
            self._process_event(result, event)
        self._hold_utterance_end(result)

    def _hold_utterance_end(self, result: TickResult) -> None:
        """Keep the last millisecond of an unfinished utterance for later.

        The end of a line's text is released by its played caption. If that
        arrives after the line's audio has all been played, its text would
        land on a tick with no audio, and the user simulator drops text from
        silent ticks. Holding back 1 ms until the caption arrives makes the
        last text always ride on audio.
        """

        if not self._awaiting_playback or result.was_truncated:
            return
        available = sum(len(chunk) for chunk, _ in result.agent_audio_chunks)
        if available == 0 or available > self.bytes_per_tick:
            return  # nothing to play, or the tail is buffered anyway
        hold = min(available, self._end_hold_bytes)
        data, item_id = result.agent_audio_chunks.pop()
        while len(data) < hold:
            previous, item_id = result.agent_audio_chunks.pop()
            data = previous + data
        if len(data) > hold:
            result.agent_audio_chunks.append((data[:-hold], item_id))
        self._held_end = [(data[-hold:], item_id)]

    def _transcript(self) -> _SpokenTranscript:
        if _ITEM_ID not in self._utterance_transcripts:
            # Duck-typed: the base class only calls get_transcript_for_audio.
            self._utterance_transcripts[_ITEM_ID] = _SpokenTranscript(  # type: ignore[assignment]
                due_slack_bytes=self._end_hold_bytes
            )
        return self._utterance_transcripts[_ITEM_ID]  # type: ignore[return-value]

    def _process_event(self, result: TickResult, event: Any) -> None:
        result.events.append(event)
        if not isinstance(event, AgentAudioEvent):
            logger.debug(
                f"Ava Zero event: {event!r} stale_text={self._stale_text} "
                f"stale_audio={self._stale_audio}"
            )

        if isinstance(event, AgentAudioEvent):
            converter = self._output_converters.setdefault(
                event.sample_rate,
                StreamingTelephonyConverter(
                    input_sample_rate=DEFAULT_AVA_ZERO_INPUT_SAMPLE_RATE,
                    output_sample_rate=event.sample_rate,
                ),
            )
            telephony_audio = converter.convert_output(event.audio)
            if telephony_audio:
                if self._stale_audio:
                    result.truncated_audio_bytes += len(telephony_audio)
                    return
                chunk = (telephony_audio, _ITEM_ID)
                self._awaiting_playback = True
                if result.was_truncated:
                    self._audio_after_cut.append(chunk)
                else:
                    result.agent_audio_chunks.append(chunk)
                self._transcript().add_audio(len(telephony_audio))
        elif isinstance(event, ToolCallEvent):
            result.tool_calls.append(
                ToolCall(
                    id=event.id,
                    name=event.name,
                    arguments=event.arguments,
                    requestor="assistant",
                )
            )
            logger.info(f"Ava Zero tool call: {event.name}({event.arguments})")
        elif isinstance(event, AgentLLMTextEvent):
            self._llm_unsent += event.text
        elif isinstance(event, TranscriptEvent):
            text = event.text
            carryover, self._llm_carryover, self._llm_unsent = (
                self._llm_carryover,
                "",
                "",
            )
            if carryover and text.startswith(carryover):
                text = text[len(carryover) :]
            if self._stale_text:
                return
            if text.strip():
                self._transcript().add_text(text)
                logger.debug(f"Ava Zero transcript: {text!r}")
        elif isinstance(event, AgentWordEvent):
            if not self._stale_audio:
                self._transcript().add_word(event.text)
        elif isinstance(event, AgentPlaybackEvent):
            if not self._stale_audio:
                self._awaiting_playback = False
                self._transcript().add_playback(event.text)
        elif isinstance(event, AgentLLMStartedEvent):
            self._stale_text = False
            self._llm_carryover = self._llm_unsent
        elif isinstance(event, AgentStartedSpeakingEvent):
            if self._stale_audio:
                self._stale_audio = False
                for converter in self._output_converters.values():
                    converter.reset()
        elif isinstance(event, SpeechEvent):
            result.vad_events.append(event.kind)
            if event.kind in {"speech_started", "interrupted"}:
                result.was_truncated = True
                result.truncated_audio_bytes += sum(
                    len(audio) for audio, _item_id in result.agent_audio_chunks
                )
                result.agent_audio_chunks.clear()
                self._buffered_agent_audio.clear()
                self._audio_after_cut.clear()
                self._held_end = []
                self._awaiting_playback = False
                self._transcript().interrupt()
                self._stale_text = self._stale_audio = True
                for converter in self._output_converters.values():
                    converter.reset()
        elif isinstance(event, ProviderErrorEvent):
            raise RuntimeError(f"Ava Zero reader failed: {event.message}")
