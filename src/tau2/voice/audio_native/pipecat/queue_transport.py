"""In-memory queue-backed Pipecat transport for tick-based simulation.

This module provides a custom Pipecat ``BaseTransport`` whose input and
output sides are driven by ``asyncio.Queue`` objects rather than a network
connection. The discrete-time adapter feeds user audio through the input
side and drains generated agent audio (and other event-bearing frames)
from the output side.

The transport is provider-agnostic: it knows nothing about STT/LLM/TTS
services. Those are wired into a normal Pipecat ``Pipeline`` in
``provider.py``.

Lazy imports
------------
Pipecat is an optional dependency of tau2-bench (installed via the
``voice`` extra). All ``pipecat.*`` imports happen inside the constructors
so importing this module never fails when Pipecat is missing.
"""

import asyncio
from typing import Any, Callable, List, Optional


class QueueInputTransport:
    """Marker / stub. Real class is built lazily in ``QueueTransport``.

    Pipecat's ``BaseInputTransport`` lives in an optional dependency so we
    cannot subclass it at module import time. ``QueueTransport.input()``
    constructs the concrete subclass on first use.
    """


class QueueOutputTransport:
    """Marker / stub. Real class is built lazily in ``QueueTransport``."""


class QueueTransport:
    """Pipecat transport that bridges a Pipeline to in-memory queues.

    Usage::

        transport = QueueTransport(
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
            on_output_audio=lambda data, rate: ...,
            on_event=lambda evt: ...,
        )

        pipeline = Pipeline([
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ])

        # Inside the running pipeline:
        await transport.push_user_audio(pcm16_bytes)

    The transport defers ``pipecat`` imports until ``input()`` /
    ``output()`` is called so importing this module without Pipecat
    installed does not fail.
    """

    def __init__(
        self,
        audio_in_sample_rate: int = 16000,
        audio_out_sample_rate: int = 24000,
        audio_in_channels: int = 1,
        audio_out_channels: int = 1,
        on_output_audio: Optional[Callable[[bytes, int], None]] = None,
        on_event: Optional[Callable[[Any], None]] = None,
        vad_analyzer: Any = None,
    ) -> None:
        self.audio_in_sample_rate = audio_in_sample_rate
        self.audio_out_sample_rate = audio_out_sample_rate
        self.audio_in_channels = audio_in_channels
        self.audio_out_channels = audio_out_channels
        self.on_output_audio = on_output_audio
        self.on_event = on_event
        self.vad_analyzer = vad_analyzer

        self._input: Optional[Any] = None
        self._output: Optional[Any] = None

    # ------------------------------------------------------------------
    # Pipecat transport API
    # ------------------------------------------------------------------

    def input(self) -> Any:
        """Return the input frame processor (lazily constructed)."""
        if self._input is None:
            self._input = self._build_input()
        return self._input

    def output(self) -> Any:
        """Return the output frame processor (lazily constructed)."""
        if self._output is None:
            self._output = self._build_output()
        return self._output

    # ------------------------------------------------------------------
    # External I/O API (called by the provider / adapter)
    # ------------------------------------------------------------------

    async def push_user_audio(self, audio: bytes) -> None:
        """Push user PCM16 audio bytes into the pipeline.

        The audio must already be at ``audio_in_sample_rate`` and mono
        16-bit signed PCM. Conversion from telephony format happens in
        ``provider.py`` before calling this method.
        """
        from pipecat.frames.frames import InputAudioRawFrame

        inp = self.input()
        if not getattr(inp, "_started", False):
            return

        if len(audio) == 0:
            return

        frame = InputAudioRawFrame(
            audio=audio,
            sample_rate=self.audio_in_sample_rate,
            num_channels=self.audio_in_channels,
        )
        await inp.push_audio_frame(frame)

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------

    def _build_input(self) -> Any:
        from pipecat.frames.frames import (
            BotStartedSpeakingFrame,
            BotStoppedSpeakingFrame,
            CancelFrame,
            EndFrame,
            ErrorFrame,
            Frame,
            FunctionCallInProgressFrame,
            InterruptionFrame,
            LLMFullResponseEndFrame,
            LLMFullResponseStartFrame,
            LLMTextFrame,
            StartFrame,
            TranscriptionFrame,
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
        )
        from pipecat.processors.frame_processor import FrameDirection
        from pipecat.transports.base_input import BaseInputTransport
        from pipecat.transports.base_transport import TransportParams

        on_event = self.on_event

        # ``vad_analyzer`` may live on TransportParams in newer Pipecat
        # versions; if not, we set it as an attribute on the input below.
        try:
            params = TransportParams(
                audio_in_enabled=True,
                audio_in_sample_rate=self.audio_in_sample_rate,
                audio_in_channels=self.audio_in_channels,
                audio_in_passthrough=True,
                vad_analyzer=self.vad_analyzer,
            )
        except TypeError:
            params = TransportParams(
                audio_in_enabled=True,
                audio_in_sample_rate=self.audio_in_sample_rate,
                audio_in_channels=self.audio_in_channels,
                audio_in_passthrough=True,
            )

        # Some Pipecat versions use ``vad_analyzer`` on TransportParams,
        # others expect it on the input transport directly. We pass it
        # through to TransportParams above and also stash it on the input
        # below to be safe across versions.
        vad_analyzer = self.vad_analyzer

        EVENT_FRAME_TYPES = (
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
            BotStartedSpeakingFrame,
            BotStoppedSpeakingFrame,
            InterruptionFrame,
            TranscriptionFrame,
            LLMFullResponseStartFrame,
            LLMFullResponseEndFrame,
            LLMTextFrame,
            FunctionCallInProgressFrame,
            ErrorFrame,
        )

        class _QueueInput(BaseInputTransport):
            def __init__(self):
                # Ensure VAD is set both on params and as an attribute.
                try:
                    setattr(params, "vad_analyzer", vad_analyzer)
                except Exception:
                    pass
                super().__init__(params=params)
                self._started = False

            async def start(self, frame: StartFrame):
                await super().start(frame)
                await self.set_transport_ready(frame)
                self._started = True

            async def stop(self, frame: EndFrame):
                self._started = False
                await super().stop(frame)

            async def cancel(self, frame: CancelFrame):
                self._started = False
                await super().cancel(frame)

            async def process_frame(self, frame: Frame, direction: FrameDirection):
                # Forward event-bearing frames to the on_event sink BEFORE
                # the base class processes them (so the sink is notified
                # even if downstream filtering would drop them).
                if on_event is not None and isinstance(frame, EVENT_FRAME_TYPES):
                    try:
                        on_event(frame)
                    except Exception:
                        pass
                await super().process_frame(frame, direction)

        return _QueueInput()

    def _build_output(self) -> Any:
        from pipecat.frames.frames import (
            BotStartedSpeakingFrame,
            BotStoppedSpeakingFrame,
            CancelFrame,
            EndFrame,
            ErrorFrame,
            Frame,
            FunctionCallInProgressFrame,
            InterruptionFrame,
            LLMFullResponseEndFrame,
            LLMFullResponseStartFrame,
            LLMTextFrame,
            OutputAudioRawFrame,
            StartFrame,
            TranscriptionFrame,
            TTSStartedFrame,
            TTSStoppedFrame,
            TTSTextFrame,
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
        )
        from pipecat.processors.frame_processor import FrameDirection
        from pipecat.transports.base_output import BaseOutputTransport
        from pipecat.transports.base_transport import TransportParams

        on_event = self.on_event
        on_output_audio = self.on_output_audio

        params = TransportParams(
            audio_out_enabled=True,
            audio_out_sample_rate=self.audio_out_sample_rate,
            audio_out_channels=self.audio_out_channels,
            # We handle silence padding ourselves in the adapter.
            audio_out_auto_silence=False,
        )

        # Frames we want to surface as PipecatEvents through on_event.
        EVENT_FRAME_TYPES = (
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
            BotStartedSpeakingFrame,
            BotStoppedSpeakingFrame,
            InterruptionFrame,
            TranscriptionFrame,
            LLMFullResponseStartFrame,
            LLMFullResponseEndFrame,
            LLMTextFrame,
            TTSStartedFrame,
            TTSStoppedFrame,
            TTSTextFrame,
            FunctionCallInProgressFrame,
            ErrorFrame,
        )

        class _QueueOutput(BaseOutputTransport):
            def __init__(self):
                super().__init__(params=params)

            async def start(self, frame: StartFrame):
                await super().start(frame)
                await self.set_transport_ready(frame)

            async def stop(self, frame: EndFrame):
                await super().stop(frame)

            async def cancel(self, frame: CancelFrame):
                await super().cancel(frame)

            async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
                if on_output_audio is not None and frame.audio:
                    try:
                        on_output_audio(frame.audio, frame.sample_rate)
                    except Exception:
                        pass
                return True

            async def process_frame(self, frame: Frame, direction: FrameDirection):
                if on_event is not None and isinstance(frame, EVENT_FRAME_TYPES):
                    try:
                        on_event(frame)
                    except Exception:
                        pass
                await super().process_frame(frame, direction)

        return _QueueOutput()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def collect_pending(queue: "asyncio.Queue[Any]") -> List[Any]:
    """Drain *all* currently-available items from an asyncio.Queue."""
    items: List[Any] = []
    while True:
        try:
            items.append(queue.get_nowait())
        except asyncio.QueueEmpty:
            break
    return items
