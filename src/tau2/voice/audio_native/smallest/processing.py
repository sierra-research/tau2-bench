"""Hydra events into tick results: agent audio, captions, cuts, tool calls.

``HydraEventProcessing`` is mixed into ``DiscreteTimeSmallestAdapter``; it uses
the adapter's reply ledger (``_responses``), output converter and audio format.

Barge-in: Hydra decides it. ``input_audio_buffer.speech_started`` marks the
first sound of a caller turn and cuts nothing (Hydra may keep the floor). Its
``response.barge_in``, sent when it commits a barge-in, truncates the queued
agent audio, as the other adapters do on speech onset, and skips the rest of
the reply that is still streaming. The reply's unspoken text is never revealed.
"""

import json
from typing import Any, Optional

from loguru import logger

from tau2.config import DEFAULT_SMALLEST_INPUT_SAMPLE_RATE
from tau2.data_model.audio import AudioFormat
from tau2.data_model.message import ToolCall
from tau2.voice.audio_native.audio_converter import StreamingTelephonyConverter
from tau2.voice.audio_native.smallest import events as ev
from tau2.voice.audio_native.smallest.responses import HydraResponses
from tau2.voice.audio_native.tick_result import TickResult


def parse_tool_arguments(arguments: Any) -> dict:
    """Tool arguments as a dict; anything unparseable becomes {} (logged)."""
    if isinstance(arguments, dict):
        return arguments
    if not arguments:
        return {}
    try:
        parsed = json.loads(arguments) if isinstance(arguments, str) else None
    except ValueError:
        parsed = None
    if not isinstance(parsed, dict):
        logger.warning(f"Smallest: tool arguments are not a JSON object: {arguments!r}")
        return {}
    return parsed


class HydraEventProcessing:
    """Applies Hydra events to the tick being built."""

    # Provided by DiscreteTimeSmallestAdapter / DiscreteTimeAdapter.
    _responses: HydraResponses
    _output_converter: Optional[StreamingTelephonyConverter]
    audio_format: AudioFormat

    def _process_event(self, result: TickResult, event: ev.SmallestEvent) -> None:
        result.events.append(event)
        responses = self._responses
        if isinstance(event, ev.SmallestAudioDelta):
            self._on_audio(result, event)
        elif isinstance(event, ev.SmallestTranscriptDelta):
            if event.text:
                key = responses.key(event.response_id)
                responses.transcript(key).add_transcript(event.text)
        elif isinstance(event, ev.SmallestResponseCreated):
            responses.open(event.response_id)
        elif isinstance(event, ev.SmallestAudioDone):
            responses.audio_done(event.response_id)
        elif isinstance(event, ev.SmallestResponseDone):
            responses.close(event.response_id, event.status)
        elif isinstance(event, ev.SmallestSpeech):
            self._on_speech(result, event)
        elif isinstance(event, ev.SmallestToolCall):
            self._on_tool_call(result, event)
        elif isinstance(event, ev.SmallestError):
            logger.error(f"Hydra error: {event.code}: {event.message}")
        else:
            logger.debug(f"Smallest event ignored: {event.type}")

    def _on_audio(self, result: TickResult, event: ev.SmallestAudioDelta) -> None:
        key = self._responses.key(event.response_id)
        self._responses.streaming = key
        if result.skip_item_id is not None and key == result.skip_item_id:
            # Stale audio of a reply the caller cut off: count it, never play it.
            seconds = len(event.audio) / 2 / event.sample_rate
            result.truncated_audio_bytes += int(
                seconds * self.audio_format.bytes_per_second
            )
            return
        converter = self._output_converter
        if converter is None or converter.output_sample_rate != event.sample_rate:
            converter = self._output_converter = StreamingTelephonyConverter(
                input_sample_rate=DEFAULT_SMALLEST_INPUT_SAMPLE_RATE,
                output_sample_rate=event.sample_rate,
            )
        audio = converter.convert_output(event.audio)
        if audio:
            result.agent_audio_chunks.append((audio, key))
            self._responses.transcript(key).add_audio(len(audio))

    def _on_speech(self, result: TickResult, event: ev.SmallestSpeech) -> None:
        if event.type == ev.SPEECH_STOPPED:
            result.vad_events.append("speech_stopped")
            return
        if event.type == ev.SPEECH_STARTED:
            # A caller turn began; Hydra alone decides whether it is a barge-in
            # (response.barge_in) or whether it keeps talking.
            result.vad_events.append("speech_started")
            return
        result.vad_events.append("interrupted")
        if not result.agent_audio_chunks:
            return  # nothing queued to play: nothing to cut
        result.was_truncated = True
        result.skip_item_id = self._responses.cut()
        self._output_converter = None  # the next reply starts a fresh stream

    def _on_tool_call(self, result: TickResult, event: ev.SmallestToolCall) -> None:
        if not (event.call_id and event.name):
            logger.warning(f"Smallest: tool call without call_id or name: {event}")
            return
        arguments = parse_tool_arguments(event.arguments)
        result.tool_calls.append(
            ToolCall(id=event.call_id, name=event.name, arguments=arguments)
        )
        logger.debug(f"Smallest tool call: {event.name}({event.call_id}) {arguments}")
