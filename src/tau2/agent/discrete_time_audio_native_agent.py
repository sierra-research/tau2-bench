"""DiscreteTimeAudioNativeAgent - Audio native agent with discrete-time tick semantics.

This agent uses a DiscreteTimeAdapter for tick-based interaction with audio native
APIs (e.g., OpenAI Realtime, Gemini Live). It integrates with the FullDuplexOrchestrator
and handles tool calls through the standard orchestrator flow.

The agent is provider-agnostic - it works with any adapter implementing DiscreteTimeAdapter.

Key features:
- Tick-based audio exchange: one tick = one call to get_next_chunk()
- Proper audio timing: agent audio capped per tick, excess buffered
- Proportional transcript: text distributed based on audio played
- Tool support: tool calls returned to orchestrator for execution
- Interruption handling: client-side truncation on user speech
- Multi-provider support: OpenAI Realtime and Gemini Live

Usage:
    # OpenAI Realtime (default)
    agent = DiscreteTimeAudioNativeAgent(
        tools=env.get_tools(),
        domain_policy=env.get_policy(),
        tick_duration_ms=1000,
    )

    # Gemini Live
    agent = DiscreteTimeAudioNativeAgent(
        tools=env.get_tools(),
        domain_policy=env.get_policy(),
        tick_duration_ms=1000,
        provider="gemini",
    )

    state = agent.get_init_state()

    # In orchestrator loop:
    agent_chunk, state = agent.get_next_chunk(state, user_chunk)

See docs/architecture/discrete_time_audio_native_agent.md for design details.
"""

import base64
from pathlib import Path
from typing import TYPE_CHECKING, List, Literal, Optional, Tuple, Union

from loguru import logger

if TYPE_CHECKING:
    from tau2.voice.audio_native.gemini.provider import GeminiVADConfig
    from tau2.voice.audio_native.livekit.config import CascadedConfig
    from tau2.voice.audio_native.nova.provider import NovaVADConfig
    from tau2.voice.audio_native.openai.provider import OpenAIVADConfig
    from tau2.voice.audio_native.qwen.provider import QwenVADConfig
    from tau2.voice.audio_native.xai.provider import XAIVADConfig

from pydantic import ConfigDict, Field

from tau2.agent.base.streaming import StreamingState, _has_meaningful_content
from tau2.agent.base_agent import FullDuplexAgent, ValidAgentInputMessage
from tau2.config import (
    AUDIO_NATIVE_PROVIDER_TYPES,
    DEFAULT_AUDIO_NATIVE_MAX_INACTIVE_SECONDS,
    DEFAULT_AUDIO_NATIVE_PROVIDER,
    DEFAULT_OPENAI_VAD_THRESHOLD,
    DEFAULT_SEND_AUDIO_INSTANT,
)
from tau2.data_model.audio import TELEPHONY_AUDIO_FORMAT, AudioEncoding, AudioFormat
from tau2.data_model.message import (
    AssistantMessage,
    EnvironmentMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool
from tau2.utils.utils import get_now
from tau2.voice.audio_native.adapter import DiscreteTimeAdapter, create_adapter
from tau2.voice.audio_native.tick_result import TickResult

# Provider type alias
AudioNativeProvider = Literal["openai", "gemini", "xai", "nova", "qwen", "livekit"]

# VAD config union type (string annotations for lazy resolution)
VADConfig = Union[
    "OpenAIVADConfig",
    "GeminiVADConfig",
    "XAIVADConfig",
    "NovaVADConfig",
    "QwenVADConfig",
]

AUDIO_NATIVE_VOICE_INSTRUCTION = """
You are a customer service agent handling a VOICE CALL with a customer.

# Important Voice Call Considerations

1. Respond naturally and conversationally as you would in a real phone call

2. Try to be helpful and always follow the policy.

# User authentication and user information collection

1. When collecting customer information (e.g. names, emails, IDs), ask the customer to spell it out letter by letter (e.g. "J, O, H, N") to ensure you have the correct information and accomodate for customer audio being unclear or background noise.

2. If authenticating the user fails based on user provided information, ALWAYS explicitly ask the customer to SPELL THINGS OUT or provide information LETTER BY LETTER (e.g. "first name J, O, H, N last name S, M, I, T, H").
""".strip()

CASCADED_MODEL_INSTRUCTION = """
You are a customer service agent handling a VOICE CALL with a customer.

# Important Voice Call Considerations

1. For the conversation, you will see transcribed speech, not written text. Expect:
- Misspellings of names, emails, or technical terms
- Missing or incorrect punctuation (periods, commas)
- Run-on sentences or incomplete thoughts

2. Respond naturally and conversationally as you would in a real phone call. Do not use bullets (numbered or unnumbered) or markdown formatting.

3. Try to be helpful and always follow the policy.

# User authentication and user information collection

1. When collecting customer information (e.g. names, emails, IDs), ask the customer to spell it out letter by letter (e.g. "J, O, H, N") to ensure you have the correct information and accomodate for customer audio being unclear or background noise.

2. If authenticating the user fails based on user provided information, ALWAYS explicitly ask the customer to SPELL THINGS OUT or provide information LETTER BY LETTER (e.g. "first name J, O, H, N last name S, M, I, T, H").
""".strip()

# System prompt without XML tags (for xAI and other providers that prefer plain text)
AUDIO_NATIVE_SYSTEM_PROMPT_PLAIN = """
{agent_instruction}

{domain_policy}
""".strip()

# System prompt with XML tags (for providers that work better with structured prompts)
AUDIO_NATIVE_SYSTEM_PROMPT_XML = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
""".strip()


class DiscreteTimeAgentState(StreamingState[ValidAgentInputMessage, AssistantMessage]):
    """State for DiscreteTimeAudioNativeAgent.

    Tracks conversation history, tick information, and utterance transcripts.
    """

    # System messages (domain policy, etc.)
    system_messages: List[SystemMessage] = Field(default_factory=list)

    # Conversation history
    messages: List[Message] = Field(default_factory=list)

    # Tick tracking
    tick_count: int = 0
    total_user_audio_bytes: int = 0
    total_agent_audio_bytes: int = 0

    # Current tick result (for debugging/analysis)
    last_tick_result: Optional[TickResult] = None

    # Pending tool calls (detected but not yet returned)
    pending_tool_calls: List[ToolCall] = Field(default_factory=list)

    # Provider stall detection
    consecutive_inactive_ticks: int = 0

    model_config = ConfigDict(arbitrary_types_allowed=True)


class DiscreteTimeAudioNativeAgent(FullDuplexAgent[DiscreteTimeAgentState]):
    """Audio native agent with discrete-time tick semantics.

    Uses DiscreteTimeAdapter for tick-based API interaction. Supports multiple
    providers (OpenAI Realtime, Gemini Live).

    Integrates with FullDuplexOrchestrator for tool handling and coordination.

    Each call to get_next_chunk() represents one tick of the simulation:
    1. Tool results (if any) are queued on the adapter
    2. User audio is extracted from participant_chunk
    3. Audio is sent to API via adapter.run_tick()
    4. Agent audio (capped) and text are returned in AssistantMessage
    5. Tool calls are detected and returned for orchestrator handling
    """

    STOP_TOKEN = "###STOP###"
    STOP_FUNCTION_NAME = "transfer_to_human_agents"

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        tick_duration_ms: int = 1000,
        modality: str = "audio",
        adapter: Optional[DiscreteTimeAdapter] = None,
        vad_config: Optional[VADConfig] = None,
        send_audio_instant: bool = DEFAULT_SEND_AUDIO_INSTANT,
        audio_format: Optional[AudioFormat] = None,
        provider: AudioNativeProvider = DEFAULT_AUDIO_NATIVE_PROVIDER,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        max_inactive_seconds: float = DEFAULT_AUDIO_NATIVE_MAX_INACTIVE_SECONDS,
        use_xml_prompt: bool = False,
        cascaded_config: Optional["CascadedConfig"] = None,
        audio_taps_dir: Optional[Path] = None,
    ):
        """Initialize the discrete-time audio native agent.

        Args:
            tools: List of tools the agent can use.
            domain_policy: Domain policy (system prompt content).
            tick_duration_ms: Duration of each tick in milliseconds.
            modality: "audio" or "audio_in_text_out".
            adapter: Optional adapter instance. Created if not provided.
            vad_config: VAD configuration. Type depends on provider:
                - OpenAI: OpenAIVADConfig (defaults to SERVER_VAD)
                - Gemini: GeminiVADConfig (defaults to automatic VAD)
            send_audio_instant: If True, send audio instantly (discrete-time mode).
            audio_format: Audio format for external communication. Defaults to
                telephony (8kHz μ-law). Note: Gemini uses different internal
                formats (16kHz/24kHz PCM16) with automatic conversion.
            provider: Audio native provider to use. Options:
                - "openai": OpenAI Realtime API (DEFAULT_AUDIO_NATIVE_PROVIDER)
                - "gemini": Google Gemini Live API
            model: Model to use. Defaults to None. If not provided, the default
                model for the provider will be used.
            max_inactive_seconds: Maximum seconds without provider activity before
                raising a stall error. Set to 0 to disable stall detection.
                Default is 30 seconds.
            use_xml_prompt: Whether to use XML tags in system prompt. Defaults to False (plain text).
                - True: Use XML tags
                - False: Use plain text (no XML tags)
            cascaded_config: Configuration for cascaded (STT→LLM→TTS) providers.
                Only used when provider="livekit". Ignored for other providers.
                Can be a CascadedConfig instance or None to use defaults.
            audio_taps_dir: Directory to save audio taps. Only used when audio_taps_dir is not None.
        """
        self.tools = tools
        self.domain_policy = domain_policy
        self.tick_duration_ms = tick_duration_ms
        self.modality = modality
        self.send_audio_instant = send_audio_instant
        self.provider = provider
        self.use_xml_prompt = use_xml_prompt
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_inactive_seconds = max_inactive_seconds
        self.cascaded_config = cascaded_config

        # Audio format (defaults to telephony)
        self.audio_format = audio_format or TELEPHONY_AUDIO_FORMAT
        # Calculate bytes_per_tick from audio format
        self.bytes_per_tick = int(
            self.audio_format.bytes_per_second * tick_duration_ms / 1000
        )

        # VAD config - defaults depend on provider (lazy imports to avoid
        # pulling in websockets/aiohttp when voice extras aren't installed)
        if vad_config is not None:
            self.vad_config = vad_config
        elif provider == "openai":
            from tau2.voice.audio_native.openai.provider import (
                OpenAIVADConfig,
                OpenAIVADMode,
            )

            self.vad_config = OpenAIVADConfig(
                mode=OpenAIVADMode.SERVER_VAD,
                threshold=DEFAULT_OPENAI_VAD_THRESHOLD,
            )
        elif provider == "gemini":
            from tau2.voice.audio_native.gemini.provider import GeminiVADConfig

            self.vad_config = GeminiVADConfig()
        elif provider == "xai":
            from tau2.voice.audio_native.xai.provider import XAIVADConfig

            self.vad_config = XAIVADConfig()
        elif provider == "qwen":
            from tau2.voice.audio_native.qwen.provider import QwenVADConfig

            self.vad_config = QwenVADConfig()
        elif provider == "livekit":
            from tau2.voice.audio_native.livekit.discrete_time_adapter import (
                LiveKitVADConfig,
            )

            self.vad_config = LiveKitVADConfig()
        else:  # nova
            from tau2.voice.audio_native.nova.provider import NovaVADConfig

            self.vad_config = NovaVADConfig()

        # Adapter - created lazily if not provided
        self._adapter = adapter
        self._owns_adapter = adapter is None

        # Build system prompt
        self.system_prompt = self._build_system_prompt()

        self.done = False

        # Audio taps for recording audio at pipeline stages
        self._agent_input_tap: Optional["AudioTap"] = None
        self._agent_output_tap: Optional["AudioTap"] = None
        if audio_taps_dir is not None:
            from tau2.voice.utils.audio_tap import AudioTap

            audio_taps_dir.mkdir(parents=True, exist_ok=True)
            self._agent_input_tap = AudioTap(
                name="agent_user-input",
                output_dir=audio_taps_dir,
                sample_rate=self.audio_format.sample_rate,
                encoding=self.audio_format.encoding,
            )
            self._agent_output_tap = AudioTap(
                name="agent_output",
                output_dir=audio_taps_dir,
                sample_rate=self.audio_format.sample_rate,
                encoding=self.audio_format.encoding,
            )
            logger.info(f"Agent audio taps enabled, output dir: {audio_taps_dir}")

    def _build_system_prompt(self) -> str:
        """Build the system prompt from domain policy with voice instructions.

        Uses plain text format by default for all providers.
        Can be overridden via use_xml_prompt parameter.
        """
        # Use plain text by default (use_xml_prompt=False), or XML if explicitly set to True
        template = (
            AUDIO_NATIVE_SYSTEM_PROMPT_XML
            if self.use_xml_prompt
            else AUDIO_NATIVE_SYSTEM_PROMPT_PLAIN
        )

        # Use CASCADED_MODEL_INSTRUCTION for cascaded models (e.g., livekit)
        # which work with transcribed speech rather than native audio
        provider_type = AUDIO_NATIVE_PROVIDER_TYPES.get(self.provider, "audio_native")
        if provider_type == "cascaded":
            agent_instruction = CASCADED_MODEL_INSTRUCTION
        else:
            agent_instruction = AUDIO_NATIVE_VOICE_INSTRUCTION

        return template.format(
            agent_instruction=agent_instruction,
            domain_policy=self.domain_policy,
        )

    @property
    def adapter(self) -> DiscreteTimeAdapter:
        """Get the adapter, creating it if needed based on provider."""
        if self._adapter is None:
            self._adapter, self.model = create_adapter(
                provider=self.provider,
                tick_duration_ms=self.tick_duration_ms,
                send_audio_instant=self.send_audio_instant,
                model=self.model,
                reasoning_effort=self.reasoning_effort,
                audio_format=self.audio_format,
                cascaded_config=self.cascaded_config,
            )
        return self._adapter

    def get_init_state(
        self,
        message_history: Optional[List[Message]] = None,
    ) -> DiscreteTimeAgentState:
        """Initialize agent state and connect to API.

        Args:
            message_history: Optional initial message history.

        Returns:
            Initialized agent state.
        """
        if message_history is None:
            message_history = []

        # Connect adapter if not connected
        if not self.adapter.is_connected:
            self.adapter.connect(
                system_prompt=self.system_prompt,
                tools=self.tools,
                vad_config=self.vad_config,
                modality=self.modality,
            )

        return DiscreteTimeAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=list(message_history),
        )

    def get_next_chunk(
        self,
        state: DiscreteTimeAgentState,
        participant_chunk: Optional[UserMessage] = None,
        tool_results: Optional[EnvironmentMessage] = None,
    ) -> Tuple[AssistantMessage, DiscreteTimeAgentState]:
        """Process one tick of the simulation.

        Each tick, the agent receives two independent channels:
        - participant_chunk: user audio/speech from the user simulator
        - tool_results: results from previously executed tool calls (if any)

        Tool results are queued on the adapter before running the tick, so
        the provider sees them alongside the new user audio in a single pass.

        Args:
            state: Current agent state.
            participant_chunk: User message with audio from the user simulator.
            tool_results: Tool results from the environment (ToolMessage or
                MultiToolMessage). None if no tool results are pending.

        Returns:
            Tuple of (AssistantMessage, updated state).
            The AssistantMessage may contain tool_calls if the agent
            requested tool execution.
        """
        # Queue tool results on the adapter so the provider sees them
        # at the start of run_tick, alongside the new user audio.
        if tool_results is not None:
            if isinstance(tool_results, ToolMessage):
                self._handle_tool_result(tool_results)
            elif isinstance(tool_results, MultiToolMessage):
                for i, tool_msg in enumerate(tool_results.tool_messages):
                    is_last = i == len(tool_results.tool_messages) - 1
                    self._handle_tool_result(tool_msg, request_response=is_last)

        state.tick_count += 1

        # Extract user audio from participant chunk
        user_audio = self._extract_user_audio(participant_chunk)
        state.total_user_audio_bytes += len(user_audio)

        if self._agent_input_tap:
            self._agent_input_tap.record(user_audio)

        # Run tick through adapter
        try:
            tick_result = self.adapter.run_tick(
                user_audio=user_audio,
                tick_number=state.tick_count,
            )
        except RuntimeError as e:
            logger.error(f"Tick failed: {e}")
            # Re-raise to trigger proper error handling (retry, error.json, etc.)
            raise

        state.last_tick_result = tick_result

        # Check for provider inactivity (warning only, does not raise)
        if tick_result.has_provider_activity:
            state.consecutive_inactive_ticks = 0
        else:
            state.consecutive_inactive_ticks += 1
            if self.max_inactive_seconds > 0:
                inactive_ms = state.consecutive_inactive_ticks * self.tick_duration_ms
                if inactive_ms >= self.max_inactive_seconds * 1000:
                    # Calculate conversation context for debugging
                    elapsed_ms = state.tick_count * self.tick_duration_ms
                    elapsed_sec = elapsed_ms / 1000
                    stall_start_tick = (
                        state.tick_count - state.consecutive_inactive_ticks
                    )
                    stall_start_sec = (stall_start_tick * self.tick_duration_ms) / 1000
                    inactive_sec = inactive_ms / 1000
                    logger.warning(
                        f"Provider inactivity detected: no activity for {inactive_sec:.1f}s "
                        f"({state.consecutive_inactive_ticks} consecutive ticks). "
                        f"Inactivity started at tick {stall_start_tick} ({stall_start_sec:.1f}s), "
                        f"current tick {state.tick_count} ({elapsed_sec:.1f}s). "
                        f"Audio so far: agent={state.total_agent_audio_bytes} bytes, "
                        f"user={state.total_user_audio_bytes} bytes"
                    )
                    # Reset counter to avoid spamming warnings every tick
                    state.consecutive_inactive_ticks = 0

        # Update audio tracking
        agent_audio = tick_result.get_played_agent_audio()
        state.total_agent_audio_bytes += len(agent_audio)

        if self._agent_output_tap:
            self._agent_output_tap.record(agent_audio)

        # Extract tool calls from events
        tool_calls = self._extract_tool_calls(tick_result)

        # Create response message
        response = self._create_response(tick_result, tool_calls)

        # Update speech timing
        if response.contains_speech or response.is_tool_call():
            state.time_since_last_talk = 0
        else:
            state.time_since_last_talk += 1

        # Update input speech tracking
        if self.speech_detection(participant_chunk):
            state.time_since_last_other_talk = 0
        else:
            state.time_since_last_other_talk += 1

        # Record tick in history. Both channels recorded separately:
        # other_chunk for participant speech, env_chunk for tool results.
        state.record_tick(
            tick_id=state.tick_count,
            timestamp=get_now(),
            self_chunk=response if _has_meaningful_content(response) else None,
            other_chunk=(
                participant_chunk if self.speech_detection(participant_chunk) else None
            ),
            env_chunk=tool_results,
        )

        # Append to message history
        if response.is_tool_call() or _has_meaningful_content(response):
            state.messages.append(response)

        return response, state

    def _get_silence(self) -> bytes:
        """Get one tick of silence in the configured audio format."""
        if self.audio_format.encoding == AudioEncoding.ULAW:
            silence_byte = b"\x7f"  # μ-law silence
        elif self.audio_format.encoding == AudioEncoding.ALAW:
            silence_byte = b"\xd5"  # A-law silence
        else:
            # PCM16: 2 bytes of zeros per sample
            silence_byte = b"\x00\x00"
        return silence_byte * (self.bytes_per_tick // len(silence_byte))

    def _extract_user_audio(self, chunk: ValidAgentInputMessage) -> bytes:
        """Extract audio bytes from user message.

        Args:
            chunk: Incoming message (UserMessage, ToolMessage, etc.).

        Returns:
            Audio bytes in the configured audio format.
            Returns silence if no audio present.
        """
        if not isinstance(chunk, UserMessage):
            # Tool messages don't have audio - return silence
            return self._get_silence()

        if not chunk.is_audio or not chunk.audio_content:
            return self._get_silence()

        # Decode audio content
        audio_bytes = chunk.get_audio_bytes()
        if audio_bytes is None:
            return self._get_silence()

        # Verify format matches configured audio format
        # The user simulator should already output the correct format
        if (
            chunk.audio_format
            and chunk.audio_format.encoding == self.audio_format.encoding
        ):
            return audio_bytes

        # If format doesn't match, we'd need conversion here
        # For now, assume it's already in the right format
        logger.warning(
            f"Expected {self.audio_format.encoding} format, got {chunk.audio_format}"
        )
        return audio_bytes

    def _handle_tool_result(
        self,
        tool_msg: ToolMessage,
        request_response: bool = True,
    ) -> None:
        """Send tool result to the API.

        Args:
            tool_msg: Tool result message from orchestrator.
            request_response: If True, request continuation after tool result.
        """
        self.adapter.send_tool_result(
            call_id=tool_msg.id,
            result=tool_msg.content or "",
            request_response=request_response,
            is_error=tool_msg.error,
        )

    def _extract_tool_calls(self, tick_result: TickResult) -> List[ToolCall]:
        """Extract tool calls from tick result.

        Tool calls are now populated by the adapter in a provider-agnostic way,
        so we simply return them from the TickResult.

        Args:
            tick_result: Result from adapter.run_tick().

        Returns:
            List of ToolCall objects (may be empty).
        """
        return tick_result.tool_calls

    def _create_response(
        self,
        tick_result: TickResult,
        tool_calls: List[ToolCall],
    ) -> AssistantMessage:
        """Create AssistantMessage from tick result.

        Args:
            tick_result: Result from adapter.run_tick().
            tool_calls: Extracted tool calls (may be empty).

        Returns:
            AssistantMessage with text content and audio for playback.
            Audio is always exactly bytes_per_tick (padded by TickResult).
        """
        # get_played_agent_audio() returns exactly bytes_per_tick, already padded
        agent_audio = tick_result.get_played_agent_audio()
        transcript = tick_result.proportional_transcript or ""

        # Use the configured audio format
        audio_format = self.audio_format

        # Check if there was actual speech (not just silence padding)
        has_speech = len(tick_result.agent_audio_data) > 0

        audio_content = base64.b64encode(agent_audio).decode("utf-8")
        content = transcript if transcript else None

        message = AssistantMessage(
            role="assistant",
            content=content,
            tool_calls=tool_calls if tool_calls else None,
            is_audio=False,  # Treat as text for orchestrator
            audio_content=audio_content,  # Always included for time alignment
            audio_format=audio_format,
            timestamp=get_now(),
            contains_speech=has_speech,
            raw_data=tick_result.model_dump(serialize_as_any=True),
            utterance_ids=tick_result.item_ids if tick_result.item_ids else None,
        )
        # Check if this is a stop message (done or transfer tool call)
        message = self._check_if_stop_toolcall(message)
        return message

    def _create_empty_response(self) -> AssistantMessage:
        """Create empty response for error cases."""
        # Text-mode message with silence audio (for time alignment)
        # Use appropriate silence value based on encoding
        if self.audio_format.encoding == AudioEncoding.ULAW:
            silence_byte = b"\x7f"  # μ-law silence
        elif self.audio_format.encoding == AudioEncoding.ALAW:
            silence_byte = b"\xd5"  # A-law silence
        else:
            # PCM16: 2 bytes of zeros per sample
            silence_byte = b"\x00\x00"
        silence = silence_byte * (self.bytes_per_tick // len(silence_byte))
        message = AssistantMessage(
            role="assistant",
            content=None,
            is_audio=False,
            audio_content=base64.b64encode(silence).decode("utf-8"),
            audio_format=self.audio_format,
            timestamp=get_now(),
            chunk_id=0,
            is_final_chunk=True,
            contains_speech=False,
        )
        # check if the message should be considered a stop message.
        message = self._check_if_stop_toolcall(message)
        return message

    def create_initial_message(
        self, content: str = "Hi! How can I help you today?"
    ) -> AssistantMessage:
        """Create the initial greeting message with silence audio.

        The initial message is text-only (no TTS) but includes silence audio
        to maintain temporal alignment in the tick-based trajectory.

        Args:
            content: The greeting text content.

        Returns:
            AssistantMessage with text content and silence audio.
        """
        # Generate silence audio for one tick
        silence = self._get_silence()
        return AssistantMessage(
            role="assistant",
            content=content,
            is_audio=False,
            audio_content=base64.b64encode(silence).decode("utf-8"),
            audio_format=self.audio_format,
            timestamp=get_now(),
            chunk_id=0,
            is_final_chunk=True,
            contains_speech=False,  # It's a text greeting, not synthesized speech
            cost=0.0,
        )

    def speech_detection(self, chunk: Optional[ValidAgentInputMessage]) -> bool:
        """Check if incoming chunk contains speech.

        Args:
            chunk: Incoming message.

        Returns:
            True if chunk contains speech, False otherwise.
        """
        if not isinstance(chunk, UserMessage):
            return False
        return chunk.contains_speech or False

    def set_seed(self, seed: int) -> None:
        """Set random seed (for reproducibility).

        Note: Audio native APIs may not fully support seeded generation.
        """
        pass  # Audio native APIs don't support seeding

    def stop(
        self,
        participant_chunk: Optional[Message] = None,
        state: Optional[DiscreteTimeAgentState] = None,
        tool_results: Optional[EnvironmentMessage] = None,
    ) -> None:
        """Stop the agent and clean up resources.

        Called by the orchestrator at the end of the simulation.
        Disconnects from the audio native API (OpenAI Realtime or Gemini Live).

        Args:
            participant_chunk: The last chunk from the user (unused).
            state: The final agent state (unused).
            tool_results: Any pending tool results not yet delivered (unused).
        """
        if self._agent_input_tap:
            self._agent_input_tap.save()
        if self._agent_output_tap:
            self._agent_output_tap.save()
            logger.info("Saved agent_input audio tap")
        self.cleanup()

    def cleanup(self) -> None:
        """Clean up resources (disconnect from API).

        Note: Prefer using stop() which is called automatically by the orchestrator.
        This method is kept for backwards compatibility and explicit cleanup.
        """
        if self._owns_adapter and self._adapter is not None:
            self._adapter.disconnect()

    def _check_if_stop_toolcall(self, message: AssistantMessage) -> AssistantMessage:
        """Check if the message is a stop message.
        If the message contains a tool call with the name STOP_FUNCTION_NAME,
        then the message is a stop message.
        """
        is_stop = False
        if message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call.name == self.STOP_FUNCTION_NAME:
                    is_stop = True
                    break
        if is_stop:
            if message.content is None:
                message.content = self.STOP_TOKEN
            else:
                message.content += " " + self.STOP_TOKEN
        return message

    @classmethod
    def is_stop(cls, message: AssistantMessage) -> bool:
        """Check if the message is a stop message."""
        if message.content is None:
            return False
        return cls.STOP_TOKEN in message.content


# =============================================================================
# AGENT FACTORY FUNCTION
# =============================================================================


def create_discrete_time_audio_native_agent(tools, domain_policy, **kwargs):
    """Factory function for DiscreteTimeAudioNativeAgent.

    Args:
        tools: Environment tools the agent can call.
        domain_policy: Policy text the agent must follow.
        **kwargs: Additional arguments. Supports:
            - audio_native_config: AudioNativeConfig with provider settings.
              If provided, the following fields are extracted from it:
              tick_duration_ms, send_audio_instant, provider, model, use_xml_prompt.
            - Individual overrides for any of the above fields.
    """
    audio_native_config = kwargs.get("audio_native_config")
    audio_taps_dir = kwargs.get("audio_taps_dir")
    if audio_native_config is not None:
        return DiscreteTimeAudioNativeAgent(
            tools=tools,
            domain_policy=domain_policy,
            tick_duration_ms=audio_native_config.tick_duration_ms,
            modality="audio",
            send_audio_instant=audio_native_config.send_audio_instant,
            provider=audio_native_config.provider,
            model=audio_native_config.model,
            reasoning_effort=audio_native_config.reasoning_effort,
            use_xml_prompt=audio_native_config.use_xml_prompt,
            cascaded_config=getattr(audio_native_config, "cascaded_config", None),
            audio_taps_dir=audio_taps_dir,
        )
    else:
        # Fallback: use individual kwargs or defaults
        return DiscreteTimeAudioNativeAgent(
            tools=tools,
            domain_policy=domain_policy,
            tick_duration_ms=kwargs.get("tick_duration_ms", 1000),
            modality=kwargs.get("modality", "audio"),
            provider=kwargs.get("provider", DEFAULT_AUDIO_NATIVE_PROVIDER),
            model=kwargs.get("model"),
            audio_taps_dir=audio_taps_dir,
        )
