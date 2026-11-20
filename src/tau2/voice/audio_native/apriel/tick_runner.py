"""Apriel event bookkeeping; the adapter uses Tau's shared tick template.

Audio remains in FIFO order as (bytes, item_id) pairs for the shared interface.
The item ownership table fences responses independently of their tool lifecycle.
Limits fail closed instead of evicting tombstones and accepting stale events.
"""

import asyncio
import base64
import json
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from tau2.data_model.audio import TELEPHONY_AUDIO_FORMAT, AudioFormat
from tau2.data_model.message import ToolCall
from tau2.voice.audio_native.apriel.events import (
    APRIEL_RESPONSE_EVENTS,
    AprielCapacityError,
    diagnostic_summary,
)
from tau2.voice.audio_native.apriel.provider import AprielRealtimeProvider
from tau2.voice.audio_native.tick_result import (
    TickResult,
    UtteranceTranscript,
    buffer_excess_audio,
    get_proportional_transcript,
)


@dataclass
class PendingUtterance:
    item_id: str
    audio_chunks: list[bytes] = field(default_factory=list)
    is_audio_complete: bool = False
    is_transcript_complete: bool = False

    @property
    def total_audio_bytes(self) -> int:
        return sum(map(len, self.audio_chunks))

    @property
    def is_complete(self) -> bool:
        return self.is_audio_complete and self.is_transcript_complete


@dataclass
class ResponseState:
    created: bool = False
    audio_cleared: bool = False
    terminal: Optional[str] = None
    published_call_ids: dict[str, None] = field(default_factory=dict)
    calls: dict[str, ToolCall] = field(default_factory=dict)
    invalid_calls: set[str] = field(default_factory=set)
    dispatched: bool = False
    results: dict[str, tuple[str, bool]] = field(default_factory=dict)
    sent_results: set[str] = field(default_factory=set)
    continued: bool = False


class TickRunner:
    def __init__(
        self,
        provider: AprielRealtimeProvider,
        tick_duration_ms: int,
        bytes_per_tick: int,
        send_audio_instant: bool,
        buffer_until_complete: bool,
        chunk_size: int = 160,
        voip_packet_interval_ms: int = 20,
        audio_format: Optional[AudioFormat] = None,
        fast_forward_mode: bool = False,
        manual_commit: bool = False,
    ):
        if tick_duration_ms <= 0 or bytes_per_tick <= 0:
            raise ValueError("tick duration and bytes_per_tick must be positive")
        self.provider = provider
        self.tick_duration_ms = tick_duration_ms
        self.bytes_per_tick = bytes_per_tick
        self.send_audio_instant = send_audio_instant
        self.buffer_until_complete = buffer_until_complete
        self.chunk_size = chunk_size
        self.voip_packet_interval_ms = voip_packet_interval_ms
        self.audio_format = audio_format or TELEPHONY_AUDIO_FORMAT
        self.fast_forward_mode = fast_forward_mode
        self.manual_commit = manual_commit
        self.max_buffer_bytes = int(self.audio_format.bytes_per_second * 120)
        self.max_state_entries = 4096
        self.max_text_chars = 65536
        self.max_event_ids = 262144
        self.reset()

    def reset(self) -> None:
        self.buffered_agent_audio: list[tuple[bytes, Optional[str]]] = []
        self.deferred_agent_audio: list[tuple[bytes, Optional[str]]] = []
        self.utterance_transcripts: dict[str, UtteranceTranscript] = {}
        self.pending_utterances: dict[str, PendingUtterance] = {}
        self.responses: dict[str, ResponseState] = {}
        self.item_responses: dict[str, str] = {}
        self.item_content_indices: dict[str, int] = {}
        self.audio_done_items: set[str] = set()
        self.transcript_done_items: set[str] = set()
        self.event_ids: set[str] = set()
        self.active_response_id: Optional[str] = None
        self.cumulative_user_audio_ms = 0
        self.skip_item_id = None

    @property
    def response_active(self) -> bool:
        return any(
            state.created and state.terminal is None
            for state in self.responses.values()
        )

    def _response(self, response_id: str) -> ResponseState:
        if response_id not in self.responses:
            if len(self.responses) >= self.max_state_entries:
                raise RuntimeError("Apriel response state limit exceeded")
            self.responses[response_id] = ResponseState()
        return self.responses[response_id]

    def _owner(self, event) -> str:
        item_id = getattr(event, "item_id", None)
        response_id = getattr(event, "response_id", None)
        owner = self.item_responses.get(item_id)
        if response_id and owner and response_id != owner:
            raise RuntimeError("Apriel item changed response ownership")
        response_id = response_id or owner or self.active_response_id or "_unscoped"
        if item_id and event.type.startswith("response."):
            if (
                item_id not in self.item_responses
                and len(self.item_responses) >= self.max_state_entries
            ):
                raise RuntimeError("Apriel item state limit exceeded")
            self.item_responses[item_id] = response_id
            self.item_content_indices[item_id] = getattr(event, "content_index", 0)
        return response_id

    async def execute_tick(
        self, user_audio: bytes, result: TickResult, tick_start: float
    ) -> None:
        result.agent_audio_chunks.extend(self.deferred_agent_audio)
        self.deferred_agent_audio.clear()
        tick_end = tick_start + self.tick_duration_ms / 1000

        async def send_audio():
            if self.send_audio_instant:
                await self.provider.send_audio(user_audio)
            else:
                for offset in range(0, len(user_audio), self.chunk_size):
                    await self.provider.send_audio(
                        user_audio[offset : offset + self.chunk_size]
                    )
                    await asyncio.sleep(self.voip_packet_interval_ms / 1000)
            if self.manual_commit and hasattr(self.provider, "commit_audio"):
                await self.provider.commit_audio()

        async def receive_events():
            async for event in self.provider.receive_events():
                await self._process_event(result, event)
                if asyncio.get_running_loop().time() >= tick_end:
                    break
                if (
                    self.fast_forward_mode
                    and result.agent_audio_bytes >= self.bytes_per_tick
                ):
                    break

        sending = asyncio.create_task(send_audio())
        receiving = asyncio.create_task(receive_events())
        try:
            await asyncio.gather(sending, receiving)
        finally:
            for task in (sending, receiving):
                if not task.done():
                    task.cancel()
            await asyncio.gather(sending, receiving, return_exceptions=True)

    async def run_tick(self, user_audio: bytes, tick_number: int) -> TickResult:
        """Standalone compatibility entry point; adapters call execute_tick instead."""
        start = asyncio.get_running_loop().time()
        result = TickResult(
            tick_number=tick_number,
            audio_sent_bytes=len(user_audio),
            audio_sent_duration_ms=len(user_audio)
            * 1000
            / self.audio_format.bytes_per_second,
            user_audio_data=user_audio,
            cumulative_user_audio_at_tick_start_ms=self.cumulative_user_audio_ms,
            bytes_per_tick=self.bytes_per_tick,
            bytes_per_second=self.audio_format.bytes_per_second,
        )
        result.agent_audio_chunks.extend(self.buffered_agent_audio)
        self.buffered_agent_audio.clear()
        await self.execute_tick(user_audio, result, start)
        result.tick_sim_duration_ms = result.audio_sent_duration_ms
        self._buffer_excess_audio(result)
        result.proportional_transcript = self._get_proportional_transcript(result)
        self.cumulative_user_audio_ms += int(result.audio_sent_duration_ms)
        if not self.fast_forward_mode:
            await asyncio.sleep(
                max(
                    0,
                    self.tick_duration_ms / 1000
                    - (asyncio.get_running_loop().time() - start),
                )
            )
        return result

    def _prefix(self, chunks, byte_limit):
        kept = []
        for data, item_id in chunks:
            if byte_limit <= 0:
                break
            kept.append((data[:byte_limit], item_id))
            byte_limit -= len(kept[-1][0])
        return kept

    def _tick_played_bytes(self, result: TickResult, audio_start_ms: int) -> int:
        offset = max(0, audio_start_ms - result.cumulative_user_audio_at_tick_start_ms)
        return min(
            self.bytes_per_tick, int(offset * self.audio_format.bytes_per_second / 1000)
        )

    def _calculate_item_audio_played_ms(self, *, item_id, result, audio_start_ms):
        transcript = self.utterance_transcripts.get(item_id)
        previous = transcript.audio_bytes_played if transcript else 0
        prefix = self._prefix(
            result.agent_audio_chunks, self._tick_played_bytes(result, audio_start_ms)
        )
        current = sum(len(data) for data, owner in prefix if owner == item_id)
        return int((previous + current) * 1000 / self.audio_format.bytes_per_second)

    def _clear_audio(self, result, response_ids, keep_prefix=None):
        def belongs(item):
            return self.item_responses.get(item, "_unscoped") in response_ids

        for rid in response_ids:
            self._response(rid).audio_cleared = True
        before = result.agent_audio_bytes + sum(
            len(a) for a, _ in self.buffered_agent_audio + self.deferred_agent_audio
        )
        if keep_prefix is None and result.was_truncated:
            keep_prefix = result.agent_audio_chunks[:]
        self.deferred_agent_audio[:] = [
            chunk for chunk in self.deferred_agent_audio if not belongs(chunk[1])
        ]
        result.agent_audio_chunks = [
            chunk for chunk in result.agent_audio_chunks if not belongs(chunk[1])
        ]
        if keep_prefix is not None:
            result.agent_audio_chunks = keep_prefix
        self.buffered_agent_audio[:] = [
            chunk for chunk in self.buffered_agent_audio if not belongs(chunk[1])
        ]
        kept_items = {item for _, item in result.agent_audio_chunks}
        for item in list(self.pending_utterances):
            if belongs(item):
                before += self.pending_utterances.pop(item).total_audio_bytes
        for item in list(self.utterance_transcripts):
            if belongs(item) and item not in kept_items:
                del self.utterance_transcripts[item]
        after = result.agent_audio_bytes + sum(
            len(a) for a, _ in self.buffered_agent_audio + self.deferred_agent_audio
        )
        result.truncated_audio_bytes += max(0, before - after)
        # Removing an incomplete head can expose an already-completed successor.
        # Count discarded bytes before moving that surviving audio into playback.
        if self.pending_utterances:
            self._check_and_release_utterance(
                next(iter(self.pending_utterances)), result
            )

    async def _process_event(self, result, event) -> None:
        if event.event_id:
            if event.event_id in self.event_ids:
                return
            if len(self.event_ids) >= self.max_event_ids:
                raise RuntimeError("Apriel event identity limit exceeded")
            self.event_ids.add(event.event_id)
        if len(result.events) >= 65536:
            raise RuntimeError("Apriel per-tick event limit exceeded")
        result.events.append(event)
        kind = event.type
        if kind in ("error", "conversation.item.input_audio_transcription.failed"):
            # TickResult.events is excluded from saved trajectories. Log the
            # bounded metadata, but never turn generic model/history/TTS
            # errors into Tau's automatic infrastructure retry path.
            logger.warning(f"Apriel diagnostic {diagnostic_summary(event)}")
            if (
                kind == "error"
                and event.code == "rate_limit_exceeded"
                and event.error.get("type") == "server_error"
            ):
                # The ordinary RPM gate has this pair in both backend versions.
                # Legacy private preparation failures incorrectly used the same
                # code with type=invalid_request_error; those stay observable.
                raise AprielCapacityError(event)
            return
        if kind == "conversation.item.input_audio_transcription.completed":
            if not event.transcript.strip():
                logger.warning(
                    f"Apriel empty input transcription {diagnostic_summary(event)}"
                )
            return
        if kind == "input_audio_buffer.speech_started":
            result.vad_events.append("speech_started")
            onset = event.audio_start_ms
            if onset is None:
                onset = result.cumulative_user_audio_at_tick_start_ms
            played = self._tick_played_bytes(result, onset)
            target = None
            remaining = played
            for data, item in result.agent_audio_chunks:
                target = item
                if remaining < len(data):
                    break
                remaining -= len(data)
            if target is None and self.pending_utterances:
                target = next(iter(self.pending_utterances))
            cursor = self._calculate_item_audio_played_ms(
                item_id=target, result=result, audio_start_ms=onset
            )
            owners = {
                self.item_responses.get(item, "_unscoped")
                for _, item in result.agent_audio_chunks + self.buffered_agent_audio
            }
            owners.update(
                self.item_responses.get(item, "_unscoped")
                for item in self.pending_utterances
            )
            owners.update(
                rid
                for rid, state in self.responses.items()
                if state.created and state.terminal is None
            )
            prefix = self._prefix(result.agent_audio_chunks, played)
            self._clear_audio(result, owners, keep_prefix=prefix)
            if target is not None:
                result.was_truncated = True
                result.interruption_audio_start_ms = onset
                await self.provider.truncate_item(
                    item_id=target,
                    content_index=self.item_content_indices.get(target, 0),
                    audio_end_ms=cursor,
                )
            # Backend owns VAD cancellation. Local fencing must not depend on a
            # remote cancel method, nor cancel a newer response by accident.
            return
        if kind == "input_audio_buffer.speech_stopped":
            result.vad_events.append("speech_stopped")
            return
        if kind == "output_audio_buffer.cleared":
            rid = getattr(event, "response_id", None) or self.item_responses.get(
                getattr(event, "item_id", None)
            )
            if rid is None:
                chunks = (
                    result.agent_audio_chunks
                    + self.buffered_agent_audio
                    + self.deferred_agent_audio
                )
                owners = {
                    self.item_responses.get(item, "_unscoped") for _, item in chunks
                }
                owners.update(
                    self.item_responses.get(item, "_unscoped")
                    for item in self.pending_utterances
                )
                if self.active_response_id:
                    owners.add(self.active_response_id)
                if len(owners) > 1:
                    raise RuntimeError("Apriel unscoped output clear is ambiguous")
                rid = next(iter(owners), None)
            if rid is not None:
                self._clear_audio(result, {rid})
            return
        if kind not in APRIEL_RESPONSE_EVENTS:
            return
        rid = self._owner(event)
        state = self._response(rid)
        if kind == "response.created":
            if not state.created and state.terminal is None and not state.audio_cleared:
                self.active_response_id = rid
            state.created = True
            return
        if kind in ("response.done", "response.cancelled"):
            if state.terminal is not None:
                return
            state.terminal = (
                "cancelled"
                if kind == "response.cancelled"
                else (event.status or "incomplete")
            )
            if state.terminal in ("failed", "incomplete"):
                logger.warning(
                    f"Apriel response terminal {diagnostic_summary(event, response_id=rid, status=state.terminal)}"
                )
            if self.active_response_id == rid:
                self.active_response_id = None
            if state.terminal == "completed":
                calls = [
                    call
                    for cid, call in state.calls.items()
                    if cid not in state.invalid_calls
                ]
                result.tool_calls.extend(calls)
                state.calls = {call.id: call for call in calls}
                state.dispatched = bool(calls)
            else:
                state.calls.clear()
                self._clear_audio(result, {rid})
            # Published declarations remain pending in server history even when
            # Tau must not execute them. Settle those calls honestly, separately
            # from executable calls and their eventual externally supplied results.
            for cid in state.published_call_ids:
                if cid not in state.calls:
                    completed = state.terminal == "completed"
                    reason = (
                        "its completed arguments were invalid or conflicting"
                        if completed
                        else f"its response was {state.terminal}"
                    )
                    output = json.dumps(
                        {
                            "status": "rejected" if completed else state.terminal,
                            "executed": False,
                            "error": f"Tool was not executed because {reason}.",
                        }
                    )
                    state.results[cid] = (output, completed)
            return
        if kind == "response.function_call_arguments.done":
            if state.terminal is not None:
                return
            cid = event.call_id
            if not cid:
                return
            total_calls = sum(
                len(s.published_call_ids) for s in self.responses.values()
            )
            if (
                total_calls >= self.max_state_entries
                and cid not in state.published_call_ids
            ):
                raise RuntimeError("Apriel tool call limit exceeded")
            state.published_call_ids[cid] = None
            if not event.name:
                state.invalid_calls.add(cid)
                return
            try:
                if len(event.arguments) > self.max_text_chars:
                    raise ValueError("arguments too long")

                def reject_constant(value):
                    raise ValueError(f"Invalid JSON constant: {value}")

                arguments = json.loads(event.arguments, parse_constant=reject_constant)
                if not isinstance(arguments, dict):
                    raise ValueError("arguments must be an object")
                call = ToolCall(id=cid, name=event.name, arguments=arguments)
                if cid in state.calls and state.calls[cid] != call:
                    state.invalid_calls.add(cid)
                else:
                    state.calls[cid] = call
            except (ValueError, TypeError):
                state.invalid_calls.add(cid)
            return
        item = getattr(event, "item_id", None)
        if state.audio_cleared or state.terminal is not None:
            if kind == "response.output_audio.delta":
                result.truncated_audio_bytes += len(base64.b64decode(event.delta))
            return
        if not item:
            return
        if kind == "response.output_audio.delta":
            if item in self.audio_done_items:
                return
            queued = result.agent_audio_bytes + sum(
                len(a) for a, _ in self.buffered_agent_audio + self.deferred_agent_audio
            )
            queued += sum(p.total_audio_bytes for p in self.pending_utterances.values())
            if len(event.delta) > (self.max_buffer_bytes - queued + 2) // 3 * 4:
                raise RuntimeError("Apriel audio buffer limit exceeded")
            decoded = base64.b64decode(event.delta, validate=True)
            if not decoded:
                return
            if queued + len(decoded) > self.max_buffer_bytes:
                raise RuntimeError("Apriel audio buffer limit exceeded")
            if self.buffer_until_complete:
                pending = self.pending_utterances.setdefault(
                    item, PendingUtterance(item)
                )
                pending.audio_chunks.append(decoded)
            else:
                destination = (
                    self.deferred_agent_audio
                    if result.was_truncated
                    else result.agent_audio_chunks
                )
                destination.append((decoded, item))
            self.utterance_transcripts.setdefault(
                item, UtteranceTranscript(item_id=item)
            ).add_audio(len(decoded))
        elif kind in (
            "response.output_audio_transcript.delta",
            "response.output_audio_transcript.done",
        ):
            if item in self.transcript_done_items:
                return
            if kind.endswith(".done"):
                self.transcript_done_items.add(item)
            ut = self.utterance_transcripts.setdefault(
                item, UtteranceTranscript(item_id=item)
            )
            text = event.delta if kind.endswith(".delta") else event.transcript
            size = len(text) + (
                len(ut.transcript_received) if kind.endswith(".delta") else 0
            )
            if size > self.max_text_chars:
                raise RuntimeError("Apriel transcript limit exceeded")
            if kind.endswith(".delta"):
                ut.add_transcript(text)
            else:
                ut.transcript_received = text
            if self.buffer_until_complete:
                pending = self.pending_utterances.setdefault(
                    item, PendingUtterance(item)
                )
                if kind.endswith(".done"):
                    pending.is_transcript_complete = True
                    self._check_and_release_utterance(item, result)
        elif kind == "response.output_audio.done":
            if item in self.audio_done_items:
                return
            self.audio_done_items.add(item)
            if self.buffer_until_complete:
                pending = self.pending_utterances.setdefault(
                    item, PendingUtterance(item)
                )
                pending.is_audio_complete = True
                self._check_and_release_utterance(item, result)

    def _check_and_release_utterance(self, item_id, result):
        # Dict insertion order is audio arrival order. A later item finishing
        # first must not overtake an incomplete earlier utterance.
        while self.pending_utterances:
            first = next(iter(self.pending_utterances))
            pending = self.pending_utterances[first]
            if not pending.is_complete:
                break
            destination = (
                self.deferred_agent_audio
                if result.was_truncated
                else result.agent_audio_chunks
            )
            destination.extend((data, first) for data in pending.audio_chunks)
            del self.pending_utterances[first]

    async def flush_tool_results(self, results) -> None:
        for call_id, output, request_response, _is_error in results:
            matches = [
                state
                for state in self.responses.values()
                if state.dispatched and call_id in state.calls
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    "Apriel tool result has unknown or ambiguous call ID"
                )
            state = matches[0]
            if not state.continued:
                value = (output, request_response)
                if call_id in state.results and state.results[call_id][0] != output:
                    raise RuntimeError("Apriel conflicting duplicate tool result")
                state.results[call_id] = value
        results.clear()
        for state in self.responses.values():
            if (
                not state.published_call_ids
                or state.terminal is None
                or state.continued
                or state.results.keys() != state.published_call_ids.keys()
            ):
                continue
            automatic_ids = state.published_call_ids.keys() - state.calls.keys()
            if any(
                sum(
                    call_id in owner.published_call_ids
                    for owner in self.responses.values()
                )
                > 1
                for call_id in automatic_ids
            ):
                # The wire result has only call_id, so preflight the complete
                # batch before sending any rejection to an ambiguous owner.
                raise RuntimeError(
                    "Apriel automatic tool settlement has ambiguous call ID across responses"
                )
            for call_id in state.published_call_ids:
                if call_id not in state.sent_results:
                    await self.provider.send_tool_result(
                        call_id, state.results[call_id][0], False
                    )
                    state.sent_results.add(call_id)
            if any(request for _, request in state.results.values()):
                await self.provider.request_response()
            state.continued = True

    def _buffer_excess_audio(self, result):
        self.buffered_agent_audio = buffer_excess_audio(result, self.bytes_per_tick)

    def _get_proportional_transcript(self, result):
        return get_proportional_transcript(
            result.agent_audio_chunks, self.utterance_transcripts
        )
