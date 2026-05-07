"""Core Pipecat-based cascaded voice provider.

Wraps a Pipecat ``Pipeline`` (STT → context → LLM → TTS) driven by the
in-memory ``QueueTransport``. Exposes a small async API consumed by
``DiscreteTimePipecatAdapter``:

- ``connect(system_prompt, tools)`` builds and starts the pipeline.
- ``process_audio(audio_bytes)`` pushes user audio through the pipeline
  and yields ``PipecatEvent`` objects describing what happened.
- ``send_tool_result(call_id, result)`` queues a tool result back into
  the LLM context and asks the LLM to continue.
- ``disconnect()`` stops the pipeline and cleans up.

All Pipecat plugin imports are deferred to ``connect()`` so importing
this module does not pull Pipecat into processes that don't use it.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional

from loguru import logger

from tau2.data_model.message import ToolCall
from tau2.environment.tool import Tool
from tau2.voice.audio_native.pipecat.config import (
    AnthropicLLMConfig,
    CartesiaTTSConfig,
    DeepgramSTTConfig,
    DeepgramTTSConfig,
    ElevenLabsTTSConfig,
    OpenAILLMConfig,
    OpenAISTTConfig,
    OpenAITTSConfig,
    PipecatConfig,
)
from tau2.voice.audio_native.pipecat.events import PipecatEvent, PipecatEventType
from tau2.voice.audio_native.pipecat.queue_transport import QueueTransport


class ProviderState(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"


# ---------------------------------------------------------------------------
# Tool helpers
# ---------------------------------------------------------------------------


def _tools_to_pipecat_schemas(tools: List[Tool]) -> List[Any]:
    """Convert tau2 ``Tool`` objects to Pipecat ``FunctionSchema`` list."""
    if not tools:
        return []

    from pipecat.adapters.schemas.function_schema import FunctionSchema
    from pipecat.adapters.schemas.tools_schema import ToolsSchema

    schemas: List[FunctionSchema] = []
    for tool in tools:
        try:
            full_schema = tool.openai_schema
            fn = full_schema.get("function", full_schema)
            params = fn.get("parameters") or {"type": "object", "properties": {}}
            schemas.append(
                FunctionSchema(
                    name=fn.get("name", tool.name),
                    description=fn.get("description", "") or "",
                    properties=params.get("properties", {}),
                    required=params.get("required", []),
                )
            )
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"Failed to convert tool {tool.name} for Pipecat: {e}")

    return ToolsSchema(standard_tools=schemas)


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class PipecatVoiceProvider:
    """Pipecat-based cascaded provider.

    Lifecycle:
        provider = PipecatVoiceProvider(config)
        await provider.connect(system_prompt, tools)
        async for evt in provider.process_audio(audio_bytes):
            ...
        await provider.disconnect()
    """

    def __init__(
        self,
        config: Optional[PipecatConfig] = None,
    ) -> None:
        self.config = config or PipecatConfig()

        self._state = ProviderState.DISCONNECTED
        self._tools: List[Tool] = []
        self._system_prompt: str = ""

        # Pipeline pieces (constructed in connect())
        self._task: Optional[Any] = None
        self._runner: Optional[Any] = None
        self._runner_task: Optional[asyncio.Task] = None
        self._transport: Optional[QueueTransport] = None
        self._context: Optional[Any] = None
        self._context_aggregator: Optional[Any] = None

        # Event sink (drained by process_audio / continue_after_tool)
        self._event_queue: asyncio.Queue[PipecatEvent] = asyncio.Queue()

        # In-flight tool calls awaiting results
        self._pending_function_calls: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._state != ProviderState.DISCONNECTED

    @property
    def state(self) -> ProviderState:
        return self._state

    @property
    def tts_sample_rate(self) -> int:
        tts = self.config.tts
        return getattr(tts, "sample_rate", 24000)

    async def connect(self, system_prompt: str, tools: List[Tool]) -> None:
        if self.is_connected:
            logger.warning("PipecatVoiceProvider already connected; reconnecting")
            await self.disconnect()

        self._system_prompt = system_prompt
        self._tools = tools
        # Drain any leftover events from a previous run
        while not self._event_queue.empty():
            self._event_queue.get_nowait()

        # Lazy imports - happen inside connect so the module imports cleanly
        # without Pipecat installed.
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.pipeline.task import PipelineParams, PipelineTask

        # VADProcessor was added in Pipecat 1.x. In Pipecat 1.x the input
        # transport no longer runs VAD inline on the audio task — VAD has
        # to be a real FrameProcessor in the pipeline. Without it,
        # ``VADUserStartedSpeakingFrame`` / ``VADUserStoppedSpeakingFrame``
        # are never emitted, OpenAI STT (which defaults to
        # ``turn_detection=False``) never commits the audio buffer, and the
        # entire downstream chain (LLM → TTS → output queue) sits idle.
        # The same frames are required by ``SpeechTimeoutUserTurnStopStrategy``
        # to advance the user turn.
        try:
            from pipecat.processors.audio.vad_processor import VADProcessor

            _has_vad_processor = True
        except ImportError:  # pragma: no cover - older Pipecat
            VADProcessor = None  # type: ignore[assignment]
            _has_vad_processor = False

        # In Pipecat 1.x, OpenAILLMContext was replaced by the provider-neutral
        # LLMContext + LLMContextAggregatorPair (in
        # pipecat.processors.aggregators.llm_response_universal). Older 0.x
        # builds still expose OpenAILLMContext; we try the new path first.
        #
        # Pipecat 1.1's default user-turn-stop strategy is
        # ``TurnAnalyzerUserTurnStopStrategy(LocalSmartTurnAnalyzerV3)``,
        # which imports ``transformers`` eagerly and requires ``torch`` at
        # inference time to score whether the user has finished speaking.
        # Without torch the analyzer silently fails to advance the turn,
        # the LLM is never invoked, and every tick eventually hits the
        # adapter's tick timeout. We don't want that hard dependency here
        # (we already have VAD + transcription), so we swap in the
        # timer-based ``SpeechTimeoutUserTurnStopStrategy`` which only
        # depends on VAD/STT signals already produced by the pipeline.
        try:
            from pipecat.processors.aggregators.llm_context import LLMContext
            from pipecat.processors.aggregators.llm_response_universal import (
                LLMContextAggregatorPair,
                LLMUserAggregatorParams,
            )
            from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
            from pipecat.turns.user_turn_strategies import UserTurnStrategies

            _use_universal_context = True
        except ImportError:  # pragma: no cover - older Pipecat
            from pipecat.processors.aggregators.openai_llm_context import (
                OpenAILLMContext as LLMContext,
            )

            LLMContextAggregatorPair = None  # type: ignore[assignment]
            LLMUserAggregatorParams = None  # type: ignore[assignment]
            UserTurnStrategies = None  # type: ignore[assignment]
            SpeechTimeoutUserTurnStopStrategy = None  # type: ignore[assignment]
            _use_universal_context = False

        # Build component services
        stt_service = self._build_stt()
        llm_service = self._build_llm()
        tts_service = self._build_tts()

        # Tool registration on the LLM service (so the LLM emits function calls)
        tools_schema = _tools_to_pipecat_schemas(tools)

        # Build context with system prompt. The new LLMContext only accepts a
        # ToolsSchema or NOT_GIVEN; passing None blows up validation.
        context_kwargs: Dict[str, Any] = {
            "messages": [{"role": "system", "content": system_prompt}],
        }
        if tools_schema:
            context_kwargs["tools"] = tools_schema
        context = LLMContext(**context_kwargs)

        if _use_universal_context:
            user_aggregator_params = LLMUserAggregatorParams(
                user_turn_strategies=UserTurnStrategies(
                    stop=[SpeechTimeoutUserTurnStopStrategy()],
                ),
            )
            context_aggregator = LLMContextAggregatorPair(
                context,
                user_params=user_aggregator_params,
            )
        else:
            # Older Pipecat: the LLM service wraps the context for us.
            context_aggregator = llm_service.create_context_aggregator(context)

        # Register all tau2 tools with a shared placeholder handler that
        # surfaces function calls as TOOL_CALL events. The actual tool
        # execution happens externally (in the simulation framework), and
        # results are fed back via send_tool_result().
        self._wire_tool_handlers(llm_service)

        # Build transport
        vad_analyzer = self._build_vad_analyzer() if self.config.enable_vad else None
        self._transport = QueueTransport(
            audio_in_sample_rate=16000,
            audio_out_sample_rate=self.tts_sample_rate,
            on_event=self._on_pipecat_event,
            on_output_audio=self._on_output_audio,
            vad_analyzer=vad_analyzer,
        )
        self._context = context
        self._context_aggregator = context_aggregator
        self._llm_service = llm_service

        # Build pipeline. ``VADProcessor`` sits right after the input
        # transport and turns raw audio frames from the simulator into
        # ``VADUserStartedSpeakingFrame`` / ``VADUserStoppedSpeakingFrame``
        # events that the STT and turn-stop strategy depend on. We skip
        # the processor only if VAD is explicitly disabled in the config
        # or the (older) Pipecat build doesn't have it.
        pipeline_components: List[Any] = [self._transport.input()]
        if _has_vad_processor and vad_analyzer is not None:
            pipeline_components.append(VADProcessor(vad_analyzer=vad_analyzer))
        else:
            logger.warning(
                "PipecatVoiceProvider building pipeline without VADProcessor "
                "(vad_analyzer=%s, has_vad_processor=%s). STT services that "
                "rely on VAD frames to commit audio (e.g. OpenAI's "
                "gpt-4o-transcribe with turn_detection=False) will not "
                "produce transcripts, and SpeechTimeoutUserTurnStopStrategy "
                "will not advance the user turn.",
                "set" if vad_analyzer is not None else "None",
                _has_vad_processor,
            )
        pipeline_components.extend(
            [
                stt_service,
                context_aggregator.user(),
                llm_service,
                tts_service,
                self._transport.output(),
                context_aggregator.assistant(),
            ]
        )
        pipeline = Pipeline(pipeline_components)

        self._task = PipelineTask(
            pipeline,
            params=PipelineParams(
                audio_in_sample_rate=16000,
                audio_out_sample_rate=self.tts_sample_rate,
                allow_interruptions=self.config.allow_interruptions,
                enable_metrics=False,
                enable_usage_metrics=False,
            ),
        )

        self._runner = PipelineRunner(handle_sigint=False)

        # Spawn the runner as a background task; PipelineRunner.run() blocks
        # until the pipeline ends.
        self._runner_task = asyncio.create_task(self._runner.run(self._task))

        # Give the pipeline a moment to start (StartFrame propagation).
        await asyncio.sleep(0.05)

        self._state = ProviderState.LISTENING
        logger.info(
            f"PipecatVoiceProvider connected "
            f"(STT={self.config.stt.provider}, "
            f"LLM={self.config.llm.provider}/{self.config.llm.model}, "
            f"TTS={self.config.tts.provider})"
        )

    async def disconnect(self) -> None:
        if not self.is_connected:
            return

        try:
            if self._task is not None:
                await self._task.cancel()
        except Exception as e:
            logger.debug(f"PipecatVoiceProvider: error cancelling task: {e}")

        if self._runner_task is not None:
            try:
                await asyncio.wait_for(self._runner_task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception) as e:
                logger.debug(f"PipecatVoiceProvider: runner task ended with: {e}")
            self._runner_task = None

        self._task = None
        self._runner = None
        self._transport = None
        self._context = None
        self._context_aggregator = None
        self._llm_service = None
        self._pending_function_calls.clear()
        self._state = ProviderState.DISCONNECTED
        logger.info("PipecatVoiceProvider disconnected")

    # ------------------------------------------------------------------
    # Service builders (lazy imports)
    # ------------------------------------------------------------------

    def _build_stt(self) -> Any:
        cfg = self.config.stt
        if isinstance(cfg, DeepgramSTTConfig):
            from pipecat.services.deepgram.stt import DeepgramSTTService

            api_key = os.environ.get("DEEPGRAM_API_KEY")
            if not api_key:
                raise RuntimeError("DEEPGRAM_API_KEY is required for DeepgramSTTConfig")
            return DeepgramSTTService(
                api_key=api_key,
                settings=DeepgramSTTService.Settings(model=cfg.model),
            )
        if isinstance(cfg, OpenAISTTConfig):
            from pipecat.services.openai.stt import OpenAISTTService

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is required for OpenAISTTConfig")
            kwargs: Dict[str, Any] = {
                "api_key": api_key,
                "settings": OpenAISTTService.Settings(model=cfg.model),
            }
            if cfg.language:
                kwargs["language"] = cfg.language
            return OpenAISTTService(**kwargs)
        raise ValueError(f"Unknown STT config type: {type(cfg).__name__}")

    def _build_llm(self) -> Any:
        cfg = self.config.llm
        if isinstance(cfg, OpenAILLMConfig):
            from pipecat.services.openai.llm import OpenAILLMService

            api_key = os.environ.get(cfg.api_key_env)
            if not api_key:
                raise RuntimeError(f"{cfg.api_key_env} is required for OpenAILLMConfig")
            settings_kwargs: Dict[str, Any] = {"model": cfg.model}
            if cfg.temperature is not None:
                settings_kwargs["temperature"] = cfg.temperature
            if cfg.top_p is not None:
                settings_kwargs["top_p"] = cfg.top_p
            if cfg.max_completion_tokens is not None:
                settings_kwargs["max_completion_tokens"] = cfg.max_completion_tokens
            # Pipecat's Settings schema doesn't model OpenAI Reasoning fields
            # explicitly; route them through ``extra`` which is forwarded
            # verbatim into the chat.completions request.
            extra: Dict[str, Any] = {}
            if cfg.reasoning_effort is not None:
                extra["reasoning_effort"] = cfg.reasoning_effort
            if extra:
                settings_kwargs["extra"] = extra
            return OpenAILLMService(
                api_key=api_key,
                settings=OpenAILLMService.Settings(**settings_kwargs),
            )

        if isinstance(cfg, AnthropicLLMConfig):
            from pipecat.services.anthropic.llm import AnthropicLLMService

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is required for AnthropicLLMConfig"
                )
            settings_kwargs: Dict[str, Any] = {
                "model": cfg.model,
                "max_tokens": cfg.max_tokens,
            }
            if cfg.temperature is not None:
                settings_kwargs["temperature"] = cfg.temperature
            return AnthropicLLMService(
                api_key=api_key,
                settings=AnthropicLLMService.Settings(**settings_kwargs),
            )

        raise ValueError(f"Unknown LLM config type: {type(cfg).__name__}")

    def _build_tts(self) -> Any:
        cfg = self.config.tts
        if isinstance(cfg, CartesiaTTSConfig):
            from pipecat.services.cartesia.tts import CartesiaTTSService

            api_key = os.environ.get("CARTESIA_API_KEY")
            if not api_key:
                raise RuntimeError("CARTESIA_API_KEY is required for CartesiaTTSConfig")
            return CartesiaTTSService(
                api_key=api_key,
                sample_rate=cfg.sample_rate,
                settings=CartesiaTTSService.Settings(
                    model=cfg.model,
                    voice=cfg.voice_id,
                ),
            )
        if isinstance(cfg, DeepgramTTSConfig):
            from pipecat.services.deepgram.tts import DeepgramTTSService

            api_key = os.environ.get("DEEPGRAM_API_KEY")
            if not api_key:
                raise RuntimeError("DEEPGRAM_API_KEY is required for DeepgramTTSConfig")
            return DeepgramTTSService(
                api_key=api_key,
                sample_rate=cfg.sample_rate,
                settings=DeepgramTTSService.Settings(voice=cfg.voice),
            )
        if isinstance(cfg, ElevenLabsTTSConfig):
            from pipecat.services.elevenlabs.tts import ElevenLabsTTSService

            api_key = os.environ.get("ELEVENLABS_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "ELEVENLABS_API_KEY is required for ElevenLabsTTSConfig"
                )
            return ElevenLabsTTSService(
                api_key=api_key,
                sample_rate=cfg.sample_rate,
                settings=ElevenLabsTTSService.Settings(
                    model=cfg.model,
                    voice=cfg.voice_id,
                ),
            )
        if isinstance(cfg, OpenAITTSConfig):
            from pipecat.services.openai.tts import OpenAITTSService

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is required for OpenAITTSConfig")
            return OpenAITTSService(
                api_key=api_key,
                sample_rate=cfg.sample_rate,
                settings=OpenAITTSService.Settings(
                    model=cfg.model,
                    voice=cfg.voice,
                ),
            )
        raise ValueError(f"Unknown TTS config type: {type(cfg).__name__}")

    def _build_vad_analyzer(self) -> Any:
        try:
            from pipecat.audio.vad.silero import SileroVADAnalyzer

            return SileroVADAnalyzer()
        except Exception as e:
            logger.warning(
                f"Failed to instantiate SileroVADAnalyzer ({e}); "
                "running without VAD. Install pipecat-ai[silero] to enable."
            )
            return None

    # ------------------------------------------------------------------
    # Tool wiring
    # ------------------------------------------------------------------

    def _wire_tool_handlers(self, llm_service: Any) -> None:
        """Register a generic placeholder for every tool name.

        Pipecat's LLM service requires registered handlers for each function
        name it might call. We surface the call as a ``TOOL_CALL`` event
        and let the simulation framework execute it externally, then feed
        the result back via ``send_tool_result()``.
        """
        for tool in self._tools:
            try:
                llm_service.register_function(
                    tool.name,
                    self._make_tool_handler(tool.name),
                )
            except Exception as e:
                logger.warning(f"Failed to register Pipecat tool {tool.name}: {e}")

    def _make_tool_handler(self, name: str):
        provider = self

        async def _handler(params):  # signature: FunctionCallParams
            call_id = (
                getattr(params, "tool_call_id", None)
                or getattr(params, "call_id", None)
                or str(uuid.uuid4())
            )
            arguments = getattr(params, "arguments", None) or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}

            tc = ToolCall(id=call_id, name=name, arguments=arguments)
            await provider._event_queue.put(
                PipecatEvent(
                    type=PipecatEventType.TOOL_CALL,
                    data={"tool_call": tc},
                )
            )

            # Hold onto the params so we can resolve later via result_callback
            provider._pending_function_calls[call_id] = params

        return _handler

    # ------------------------------------------------------------------
    # Frame -> event conversion
    # ------------------------------------------------------------------

    def _on_pipecat_event(self, frame: Any) -> None:
        """Translate a Pipecat frame into a PipecatEvent on our queue.

        Called from within the pipeline's async loop (same loop as the
        rest of the provider), so it is safe to use ``put_nowait``.
        """
        try:
            from pipecat.frames.frames import (
                BotStartedSpeakingFrame,
                BotStoppedSpeakingFrame,
                ErrorFrame,
                InterruptionFrame,
                LLMFullResponseEndFrame,
                LLMFullResponseStartFrame,
                LLMTextFrame,
                TranscriptionFrame,
                TTSStartedFrame,
                TTSStoppedFrame,
                TTSTextFrame,
                UserStartedSpeakingFrame,
                UserStoppedSpeakingFrame,
            )
        except Exception:
            return

        evt: Optional[PipecatEvent] = None

        if isinstance(frame, UserStartedSpeakingFrame):
            evt = PipecatEvent(type=PipecatEventType.SPEECH_STARTED)
        elif isinstance(frame, UserStoppedSpeakingFrame):
            evt = PipecatEvent(type=PipecatEventType.SPEECH_ENDED)
        elif isinstance(frame, InterruptionFrame):
            evt = PipecatEvent(type=PipecatEventType.INTERRUPTED)
        elif isinstance(frame, BotStartedSpeakingFrame):
            evt = PipecatEvent(type=PipecatEventType.TTS_STARTED)
        elif isinstance(frame, BotStoppedSpeakingFrame):
            evt = PipecatEvent(type=PipecatEventType.TTS_COMPLETED)
        elif isinstance(frame, TranscriptionFrame):
            text = getattr(frame, "text", "") or ""
            if text:
                evt = PipecatEvent(
                    type=PipecatEventType.TRANSCRIPT_FINAL,
                    data={"transcript": text},
                )
        elif isinstance(frame, LLMFullResponseStartFrame):
            evt = PipecatEvent(type=PipecatEventType.LLM_STARTED)
        elif isinstance(frame, LLMTextFrame):
            # Note: this branch will not fire when an upstream TTSService is
            # in the pipeline because the TTS service consumes ``LLMTextFrame``
            # for sentence aggregation and does not forward it downstream
            # (see ``pipecat.services.tts_service.TTSService.process_frame``).
            # We keep it for parity if someone wires a pipeline without TTS.
            text = getattr(frame, "text", "") or ""
            if text:
                evt = PipecatEvent(
                    type=PipecatEventType.LLM_TOKEN,
                    data={"token": text},
                )
        elif isinstance(frame, TTSTextFrame):
            # ``TTSTextFrame`` IS forwarded downstream by the TTS service, one
            # frame per synthesized sentence (or per word for TTS services
            # with word timestamps). It is the only reliable way to recover
            # the agent's spoken text after the LLM → TTS handoff in a
            # cascaded pipeline. We surface it as an ``LLM_TOKEN`` event so
            # the adapter's existing transcript accumulator (which keys off
            # LLM_TOKEN events) populates ``proportional_transcript``.
            text = getattr(frame, "text", "") or ""
            if text:
                evt = PipecatEvent(
                    type=PipecatEventType.LLM_TOKEN,
                    data={"token": text},
                )
        elif isinstance(frame, LLMFullResponseEndFrame):
            evt = PipecatEvent(type=PipecatEventType.LLM_COMPLETED, data={"text": ""})
        elif isinstance(frame, TTSStartedFrame):
            evt = PipecatEvent(type=PipecatEventType.TTS_STARTED)
        elif isinstance(frame, TTSStoppedFrame):
            evt = PipecatEvent(type=PipecatEventType.TTS_COMPLETED)
        elif isinstance(frame, ErrorFrame):
            evt = PipecatEvent(
                type=PipecatEventType.ERROR,
                data={"error": str(frame)},
            )

        if evt is not None:
            try:
                self._event_queue.put_nowait(evt)
            except asyncio.QueueFull:
                pass

    def _on_output_audio(self, audio: bytes, sample_rate: int) -> None:
        """Capture output audio bytes from the QueueTransport output."""
        if not audio:
            return
        try:
            self._event_queue.put_nowait(
                PipecatEvent(
                    type=PipecatEventType.TTS_AUDIO,
                    data={"audio": audio, "sample_rate": sample_rate},
                )
            )
        except asyncio.QueueFull:
            pass

    # ------------------------------------------------------------------
    # Public async API consumed by the adapter
    # ------------------------------------------------------------------

    async def process_audio(self, audio: bytes) -> AsyncGenerator[PipecatEvent, None]:
        """Push user audio in and yield any events that are ready.

        This is non-blocking: it drains whatever events are currently in
        the queue and returns. The discrete-time adapter calls this on
        every tick.
        """
        if not self.is_connected or self._transport is None:
            raise RuntimeError("Pipecat provider not connected")

        await self._transport.push_user_audio(audio)

        # Yield a small handoff so the pipeline gets a chance to process.
        await asyncio.sleep(0)

        while True:
            try:
                evt = self._event_queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            yield evt

    async def drain_events(self) -> AsyncGenerator[PipecatEvent, None]:
        """Yield queued events without pushing audio. Used by the adapter
        when it wants to wait for late-arriving audio/transcript.
        """
        await asyncio.sleep(0)
        while True:
            try:
                evt = self._event_queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            yield evt

    async def send_tool_result(
        self,
        call_id: str,
        result: str,
        request_response: bool = True,
    ) -> AsyncGenerator[PipecatEvent, None]:
        """Resolve a pending tool call with its result.

        Pipecat's LLM service expects the registered function handler to
        invoke ``params.result_callback(result)``. If the function call
        is still pending we resolve it that way; otherwise we fall back
        to inserting a tool message into the context manually.
        """
        # async generator yielding events that arrive while the LLM
        # continuation runs.
        params = self._pending_function_calls.pop(call_id, None)
        if params is not None and hasattr(params, "result_callback"):
            try:
                cb = params.result_callback
                # result_callback expects a dict-like result, but accepts strings
                payload: Any
                try:
                    payload = json.loads(result)
                except (json.JSONDecodeError, TypeError):
                    payload = result
                await cb(payload)
            except Exception as e:
                logger.warning(f"result_callback for {call_id} failed: {e}")
        else:
            logger.debug(
                f"send_tool_result: no pending function call for {call_id}; "
                "inserting message manually"
            )
            try:
                if self._context is not None:
                    self._context.add_message(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": result,
                        }
                    )
            except Exception as e:
                logger.warning(f"Could not append tool result to context: {e}")

        # Drain any events generated by the result callback.
        await asyncio.sleep(0)
        while True:
            try:
                evt = self._event_queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            yield evt

    async def interrupt(self) -> PipecatEvent:
        """Manually trigger an interruption of any in-flight TTS / LLM."""
        if self._task is None:
            return PipecatEvent(type=PipecatEventType.INTERRUPTED)
        try:
            from pipecat.frames.frames import InterruptionFrame

            await self._task.queue_frame(InterruptionFrame())
        except Exception as e:
            logger.debug(f"interrupt() failed to queue InterruptionFrame: {e}")
        return PipecatEvent(type=PipecatEventType.INTERRUPTED)
