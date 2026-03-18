"""
Gemini Live API provider for end-to-end voice/text processing.

Uses the google-genai library for real-time bidirectional audio communication.
"""

import asyncio
import os
from enum import Enum
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel

from tau2.config import (
    DEFAULT_GEMINI_INPUT_SAMPLE_RATE,
    DEFAULT_GEMINI_LOCATION,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_GEMINI_MODEL_VERTEX,
    DEFAULT_GEMINI_OUTPUT_SAMPLE_RATE,
    DEFAULT_GEMINI_PROACTIVE_AUDIO,
    DEFAULT_GEMINI_VOICE,
)
from tau2.environment.tool import Tool
from tau2.utils.retry import websocket_retry
from tau2.voice.audio_native.gemini.events import (
    BaseGeminiEvent,
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

load_dotenv()

# Audio format constants for Gemini Live (from config)
GEMINI_INPUT_SAMPLE_RATE = DEFAULT_GEMINI_INPUT_SAMPLE_RATE
GEMINI_OUTPUT_SAMPLE_RATE = DEFAULT_GEMINI_OUTPUT_SAMPLE_RATE
GEMINI_INPUT_BYTES_PER_SECOND = GEMINI_INPUT_SAMPLE_RATE * 2  # 16-bit = 2 bytes
GEMINI_OUTPUT_BYTES_PER_SECOND = GEMINI_OUTPUT_SAMPLE_RATE * 2


class GeminiVADMode(str, Enum):
    """Voice Activity Detection modes for Gemini Live.

    Gemini Live has built-in VAD that automatically detects speech.
    """

    AUTOMATIC = "automatic"  # Server handles VAD automatically
    MANUAL = "manual"  # Client controls turns explicitly


class GeminiVADConfig(BaseModel):
    """Configuration for Gemini's Voice Activity Detection.

    Gemini Live handles VAD automatically with built-in interruption support.
    This config allows customization of behavior.

    Attributes:
        mode: VAD mode. Defaults to AUTOMATIC.
        enable_input_transcription: Whether to request input transcription.
    """

    mode: GeminiVADMode = GeminiVADMode.AUTOMATIC
    enable_input_transcription: bool = True


class GeminiLiveProvider:
    """Gemini Live API provider with session-based communication.

    This provider manages a persistent session with Gemini's Live API,
    enabling real-time bidirectional communication for voice and text processing.

    Auto-detects authentication method based on environment variables:
    1. GEMINI_API_KEY → API key auth (AI Studio)
    2. GOOGLE_APPLICATION_CREDENTIALS + GOOGLE_CLOUD_PROJECT → Service account (Vertex AI)

    Attributes:
        DEFAULT_MODEL: The default model for AI Studio.
        DEFAULT_MODEL_VERTEX: The default model for Vertex AI.
        api_key: The Gemini API key (if using API key auth).
        use_vertex_ai: Whether using Vertex AI with service account.
        model: The model identifier for the session.
        client: The google-genai client.
        session: The active live session, or None if disconnected.

    Example:
        ```python
        # Auto-detects auth from env vars
        provider = GeminiLiveProvider()

        await provider.connect(system_prompt="You are helpful.", tools=[])

        # Send audio
        await provider.send_audio(audio_bytes)

        # Receive events for a tick duration
        events = await provider.receive_events_for_duration(0.2)
        for event in events:
            if isinstance(event, GeminiAudioDeltaEvent):
                play_audio(event.data)
            elif isinstance(event, GeminiTextDeltaEvent):
                print(event.text)

        await provider.disconnect()
        ```
    """

    DEFAULT_MODEL = DEFAULT_GEMINI_MODEL
    DEFAULT_MODEL_VERTEX = DEFAULT_GEMINI_MODEL_VERTEX
    DEFAULT_VOICE = DEFAULT_GEMINI_VOICE
    DEFAULT_LOCATION = DEFAULT_GEMINI_LOCATION

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        use_raw_json_schema: bool = True,
        max_resumptions: int = 3,
        resume_only_on_timeout: bool = True,
    ):
        """Initialize the Gemini Live provider.

        Auto-detects authentication method based on environment variables:
        1. If GEMINI_API_KEY is set → use API key auth (AI Studio)
        2. Else if GOOGLE_SERVICE_ACCOUNT_KEY is set → use Vertex AI (JSON content)
        3. Else if GOOGLE_APPLICATION_CREDENTIALS is set → use Vertex AI (file path)

        Args:
            api_key: Gemini API key. If not provided, reads from GEMINI_API_KEY.
            model: Model identifier to use. Auto-selected based on auth method.
            project_id: Google Cloud project ID for Vertex AI. Reads from
                GOOGLE_CLOUD_PROJECT env var if not provided.
            location: Google Cloud region for Vertex AI. Defaults to us-central1.
            use_raw_json_schema: If True (default), pass tool schemas directly
                using parametersJsonSchema (lets SDK handle $ref/$defs).
                If False, manually resolve $ref/$defs before passing.
            max_resumptions: Maximum number of session resumptions to attempt
                when the WebSocket connection is closed. Set to 0 to disable
                session resumption. Defaults to 3.
            resume_only_on_timeout: If True (default), only attempt resumption
                when the connection closes due to the planned ~10 minute timeout
                (indicated by a GoAway message). If False, attempt resumption
                on any connection close.

        Raises:
            ValueError: If no credentials are available.
        """
        self._use_raw_json_schema = use_raw_json_schema
        import json
        import tempfile

        # Try API key first
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")

        # Check for service account credentials (JSON content or file path)
        service_account_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_KEY")
        creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        has_service_account_json = bool(service_account_json)
        has_service_account_file = creds_path and os.path.exists(creds_path)

        # Track temp file for cleanup
        self._temp_creds_file = None

        if self.api_key:
            # API key mode (AI Studio)
            self.use_vertex_ai = False
            self.project_id = None
            self.location = None
            self.model = model or self.DEFAULT_MODEL
            logger.info("Using Gemini AI Studio with API key")

        elif has_service_account_json:
            # Vertex AI mode with service account JSON content
            self.use_vertex_ai = True
            self.api_key = None

            # Parse JSON to extract project_id if not provided
            try:
                sa_data = json.loads(service_account_json)
                self.project_id = (
                    project_id
                    or os.environ.get("GOOGLE_CLOUD_PROJECT")
                    or sa_data.get("project_id")
                )
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"GOOGLE_SERVICE_ACCOUNT_KEY contains invalid JSON: {e}"
                )

            self.location = location or os.environ.get(
                "GOOGLE_CLOUD_LOCATION", self.DEFAULT_LOCATION
            )

            if not self.project_id:
                raise ValueError(
                    "Could not determine project ID. Set GOOGLE_CLOUD_PROJECT env var."
                )

            # Write to temp file for google-auth to use
            self._temp_creds_file = tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            )
            self._temp_creds_file.write(service_account_json)
            self._temp_creds_file.close()
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self._temp_creds_file.name

            self.model = model or self.DEFAULT_MODEL_VERTEX
            logger.info(
                f"Using Vertex AI with service account JSON "
                f"(project={self.project_id}, location={self.location})"
            )

        elif has_service_account_file:
            # Vertex AI mode with service account file
            self.use_vertex_ai = True
            self.api_key = None
            self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
            self.location = location or os.environ.get(
                "GOOGLE_CLOUD_LOCATION", self.DEFAULT_LOCATION
            )

            if not self.project_id:
                # Try to extract from the credentials file
                try:
                    with open(creds_path) as f:
                        sa_data = json.load(f)
                        self.project_id = sa_data.get("project_id")
                except Exception:
                    pass

            if not self.project_id:
                raise ValueError(
                    "Service account found but GOOGLE_CLOUD_PROJECT not set. "
                    "Set GOOGLE_CLOUD_PROJECT env var to your GCP project ID."
                )

            self.model = model or self.DEFAULT_MODEL_VERTEX
            logger.info(
                f"Using Vertex AI with service account file "
                f"(project={self.project_id}, location={self.location})"
            )

        else:
            raise ValueError(
                "No Gemini credentials found. Set one of:\n"
                "  - GEMINI_API_KEY (for AI Studio)\n"
                "  - GOOGLE_SERVICE_ACCOUNT_KEY (JSON content for Vertex AI)\n"
                "  - GOOGLE_APPLICATION_CREDENTIALS (file path for Vertex AI)"
            )

        self._client = None
        self._session = None
        self._session_context_manager = None  # Keep reference to close properly
        self._turn_iterator = None
        self._current_item_id: Optional[str] = None
        self._item_counter = 0

        # Background receive loop state
        self._event_queue: Optional[asyncio.Queue] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._stop_receive: bool = False
        self._preserved_events: List[BaseGeminiEvent] = []

        # Session resumption state
        self._max_resumptions = max_resumptions
        self._resume_only_on_timeout = resume_only_on_timeout
        self._resumption_count = 0
        self._resumption_handle: Optional[str] = None
        self._connect_config: Optional[Dict[str, Any]] = None  # For reconnection

        # Gemini doesn't expose a session ID; use timestamps + API key for debugging
        self.session_id: Optional[str] = None
        self._go_away_received: bool = False
        self._reconnect_at_turn_boundary: bool = False
        self._reconnect_deadline: Optional[float] = None
        self._pending_reconnection: bool = False

    @property
    def client(self):
        """Get the genai client, creating it lazily."""
        if self._client is None:
            try:
                from google import genai

                if self.use_vertex_ai:
                    # Vertex AI mode - uses ADC (Application Default Credentials)
                    self._client = genai.Client(
                        vertexai=True,
                        project=self.project_id,
                        location=self.location,
                    )
                    logger.debug(
                        f"Created Vertex AI client for project {self.project_id}"
                    )
                else:
                    # API key mode
                    self._client = genai.Client(
                        http_options={"api_version": "v1beta"},
                        api_key=self.api_key,
                    )
                    logger.debug("Created Gemini AI Studio client with API key")
            except ImportError:
                raise ImportError(
                    "google-genai package not installed. "
                    "Install with: pip install google-genai"
                )
        return self._client

    @property
    def is_connected(self) -> bool:
        """Check if the session is active.

        Returns:
            True if connected and session is active, False otherwise.
        """
        return self._session is not None

    @websocket_retry
    async def connect(
        self,
        system_prompt: str,
        tools: List[Tool],
        vad_config: Optional[GeminiVADConfig] = None,
        modality: str = "audio",
        voice: Optional[str] = None,
        _resumption_handle: Optional[str] = None,
    ) -> None:
        """Connect to the Gemini Live API and configure the session.

        Args:
            system_prompt: The system instructions for the assistant.
            tools: List of tools available for the assistant to use.
            vad_config: Voice Activity Detection configuration.
            modality: The input/output modality. One of:
                - "audio": Audio input and audio output.
                - "text": Text-only output (audio input still supported).
            voice: Voice name for audio output. Defaults to DEFAULT_VOICE.
            _resumption_handle: Internal parameter for session resumption.
                Pass the handle from a previous session to resume it.
            proactive_audio: If True, allow model to ignore irrelevant audio input.

        Raises:
            RuntimeError: If connection fails.
        """
        if self.is_connected:
            logger.warning("Already connected, disconnecting first")
            await self.disconnect()

        if vad_config is None:
            vad_config = GeminiVADConfig()

        voice = voice or self.DEFAULT_VOICE

        # Store connection config for potential reconnection
        self._connect_config = {
            "system_prompt": system_prompt,
            "tools": tools,
            "vad_config": vad_config,
            "modality": modality,
            "voice": voice,
        }

        # Reset resumption count on fresh connect (not a resumption)
        if _resumption_handle is None:
            self._resumption_count = 0
            self._resumption_handle = None
            # Discard stale preserved events from a prior failed reconnection;
            # on resumption connects the caller just populated them intentionally.
            self._preserved_events.clear()

        # Reset all GoAway-related state on any connect
        self._go_away_received = False
        self._reconnect_at_turn_boundary = False
        self._reconnect_deadline = None
        self._pending_reconnection = False

        try:
            from google.genai import types

            # Build response modalities based on modality setting
            if modality == "audio":
                response_modalities = ["AUDIO"]
            else:
                response_modalities = ["TEXT"]

            # Build tools configuration
            tool_declarations = self._format_tools_for_api(tools)

            # Build the live config
            config_kwargs: Dict[str, Any] = {
                "response_modalities": response_modalities,
                "system_instruction": system_prompt,
            }

            # Add speech config for audio modality
            if modality == "audio":
                config_kwargs["speech_config"] = types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice
                        )
                    )
                )

            # Add tools if any
            if tool_declarations:
                config_kwargs["tools"] = [
                    types.Tool(function_declarations=tool_declarations)
                ]
                logger.info(
                    f"Gemini session configured with {len(tool_declarations)} tools: "
                    f"{[t.name for t in tool_declarations]}"
                )

            # Enable context window compression for long sessions.
            # Leave parameters unset so the API uses model-dependent defaults.
            config_kwargs["context_window_compression"] = (
                types.ContextWindowCompressionConfig()
            )

            # Add session resumption config (enables receiving resumption handles)
            if self._max_resumptions > 0:
                config_kwargs["session_resumption"] = types.SessionResumptionConfig(
                    handle=_resumption_handle,  # None for fresh session, handle for resume
                )
                if _resumption_handle:
                    logger.info(
                        f"Attempting session resumption "
                        f"({self._resumption_count}/{self._max_resumptions})"
                    )

            # Add input transcription config if enabled
            if vad_config.enable_input_transcription:
                config_kwargs["input_audio_transcription"] = (
                    types.AudioTranscriptionConfig()
                )

            # Always enable output audio transcription to get text of what Gemini says
            config_kwargs["output_audio_transcription"] = (
                types.AudioTranscriptionConfig()
            )

            # Enable proactive audio to allow model to ignore irrelevant input
            if DEFAULT_GEMINI_PROACTIVE_AUDIO:
                config_kwargs["proactivity"] = types.ProactivityConfig(
                    proactive_audio=True,
                )

            config = types.LiveConnectConfig(**config_kwargs)

            # Connect to the API - IMPORTANT: keep reference to context manager
            # The context manager holds the lifecycle, session is just the inner object
            self._session_context_manager = self.client.aio.live.connect(
                model=self.model, config=config
            )
            self._session = await self._session_context_manager.__aenter__()

            # NOTE: We don't start the receive loop here. It starts lazily
            # when the first audio is sent, to avoid the session closing
            # due to inactivity before the first tick.

            if _resumption_handle:
                logger.info(
                    f"GeminiLiveProvider resumed session on {self.model} "
                    f"(modality={modality}, voice={voice})"
                )
            else:
                logger.info(
                    f"GeminiLiveProvider connected to {self.model} "
                    f"(modality={modality}, voice={voice})"
                )

        except Exception as e:
            logger.error(f"Failed to connect to Gemini Live API: {e}")
            raise RuntimeError(f"Failed to connect to Gemini Live API: {e}") from e

    async def disconnect(self) -> None:
        """Close the session.

        Gracefully closes the session if one exists.
        Safe to call even if not connected.
        """
        # Stop background receive loop first
        await self._stop_receive_loop()

        # Reset GoAway-related state so needs_reconnection doesn't
        # return True after an intentional disconnect.
        self._go_away_received = False
        self._reconnect_at_turn_boundary = False
        self._reconnect_deadline = None
        self._pending_reconnection = False

        if self._session_context_manager:
            logger.info("GeminiLiveProvider: disconnecting session")
            try:
                await self._session_context_manager.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")
            self._session = None
            self._session_context_manager = None
            self._turn_iterator = None
            logger.info("GeminiLiveProvider: session closed")

        # Clean up temp credentials file if we created one
        if self._temp_creds_file is not None:
            try:
                os.unlink(self._temp_creds_file.name)
                logger.debug("Cleaned up temp credentials file")
            except Exception as e:
                logger.warning(f"Failed to clean up temp credentials file: {e}")
            self._temp_creds_file = None

    async def _reconnect_with_resumption(self) -> bool:
        """Attempt to reconnect using session resumption.

        Returns:
            True if reconnection was successful, False otherwise.

        Raises:
            RuntimeError: If max resumptions exceeded or no handle available.
        """
        if self._resumption_handle is None:
            logger.error("Cannot resume: no resumption handle available")
            return False

        if self._resumption_count >= self._max_resumptions:
            logger.error(
                f"Cannot resume: max resumptions ({self._max_resumptions}) exceeded"
            )
            return False

        if self._connect_config is None:
            logger.error("Cannot resume: no connection config stored")
            return False

        self._resumption_count += 1
        logger.warning(
            f"Attempting session resumption "
            f"({self._resumption_count}/{self._max_resumptions}) "
            f"with handle: {self._resumption_handle[:20]}..."
        )

        # Close current session if any
        if self._session_context_manager:
            try:
                await self._session_context_manager.__aexit__(None, None, None)
            except Exception as e:
                logger.debug(f"Error closing old session during resumption: {e}")
            self._session = None
            self._session_context_manager = None

        # Reconnect with the stored handle
        # connect() resets all GoAway-related state internally
        try:
            await self.connect(
                **self._connect_config,
                _resumption_handle=self._resumption_handle,
            )
            logger.info("Session resumption successful")
            return True
        except Exception as e:
            logger.error(f"Session resumption failed: {e}")
            return False

    @property
    def needs_reconnection(self) -> bool:
        """Whether a GoAway reconnection is pending.

        The adapter should check this at the start of each tick and call
        perform_pending_reconnection() before any session I/O.
        """
        return self._pending_reconnection

    async def perform_pending_reconnection(self) -> bool:
        """Perform a pending GoAway reconnection.

        Called by the adapter at the start of a tick, BEFORE any session I/O
        (send_tool_response, send_audio). This ensures no concurrent operations
        race with the session teardown.

        Returns:
            True if reconnection succeeded, False otherwise.
        """
        if not self._pending_reconnection:
            return True

        self._pending_reconnection = False

        # Clean up the completed receive task (it exited cleanly after
        # setting _pending_reconnection)
        if self._receive_task is not None:
            if self._receive_task.done():
                try:
                    self._receive_task.result()
                except Exception as ex:
                    logger.debug(f"Receive task exited with {type(ex).__name__}: {ex}")
            else:
                self._receive_task.cancel()
                try:
                    await self._receive_task
                except (asyncio.CancelledError, Exception):
                    pass
            self._receive_task = None

        # Drain remaining events before discarding the queue so that
        # turn_complete / audio.done events queued between the last tick's
        # receive window and the reconnection are not silently lost.
        if self._event_queue is not None:
            while not self._event_queue.empty():
                try:
                    self._preserved_events.append(self._event_queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            if self._preserved_events:
                logger.debug(
                    f"Preserved {len(self._preserved_events)} events from "
                    "pre-reconnection queue"
                )
        self._event_queue = None

        try:
            success = await self._reconnect_with_resumption()
            if success:
                logger.info("Adapter-driven GoAway reconnection successful")
                return True
            logger.error("Adapter-driven GoAway reconnection failed")
            return False
        except Exception as e:
            logger.error(f"Adapter-driven GoAway reconnection failed: {e}")
            return False

    # Fields not supported by Gemini FunctionDeclaration when using `parameters`
    _UNSUPPORTED_SCHEMA_FIELDS = {
        "$defs",
        "$ref",
        "additionalProperties",
        "additional_properties",
    }

    def _resolve_refs(
        self, schema: Dict[str, Any], defs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recursively resolve $ref references and remove unsupported fields.

        Used when use_raw_json_schema=False. Gemini's FunctionDeclaration
        `parameters` field doesn't support $ref, $defs, or additionalProperties,
        so we inline references and remove unsupported fields.

        Args:
            schema: The schema or sub-schema to process.
            defs: The $defs dictionary containing type definitions.

        Returns:
            Schema with all $ref references inlined and unsupported fields removed.
        """
        if not isinstance(schema, dict):
            return schema

        # If this is a $ref, resolve it
        if "$ref" in schema:
            ref_path = schema["$ref"]
            # Parse reference like "#/$defs/FlightInfo"
            if ref_path.startswith("#/$defs/"):
                def_name = ref_path.split("/")[-1]
                if def_name in defs:
                    # Return a copy of the referenced definition, resolved
                    return self._resolve_refs(defs[def_name].copy(), defs)
            # If we can't resolve, return schema without $ref
            return {
                k: v
                for k, v in schema.items()
                if k not in self._UNSUPPORTED_SCHEMA_FIELDS
            }

        # Process all values recursively, skipping unsupported fields
        result = {}
        for key, value in schema.items():
            if key in self._UNSUPPORTED_SCHEMA_FIELDS:
                continue
            elif isinstance(value, dict):
                result[key] = self._resolve_refs(value, defs)
            elif isinstance(value, list):
                result[key] = [
                    self._resolve_refs(item, defs) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                result[key] = value

        return result

    def _format_tools_for_api(self, tools: List[Tool]) -> List[Any]:
        """Format tools for the Gemini Live API.

        Converts internal Tool objects to the format expected by the API.

        If use_raw_json_schema is True (default), passes schemas directly via
        parametersJsonSchema (letting the SDK handle $ref/$defs).
        If False, manually resolves $ref/$defs before passing via parameters.

        Args:
            tools: List of Tool objects to format.

        Returns:
            List of FunctionDeclaration objects.
        """
        from google.genai import types

        formatted_tools = []
        for tool in tools:
            schema = tool.openai_schema
            parameters = schema["function"]["parameters"]

            if self._use_raw_json_schema:
                # Let the SDK handle the raw JSON schema (including $ref/$defs)
                formatted_tools.append(
                    types.FunctionDeclaration(
                        name=schema["function"]["name"],
                        description=schema["function"]["description"],
                        parametersJsonSchema=parameters,
                    )
                )
            else:
                # Manually resolve $ref references if $defs exists
                if "$defs" in parameters:
                    defs = parameters.get("$defs", {})
                    parameters = self._resolve_refs(parameters, defs)

                formatted_tools.append(
                    types.FunctionDeclaration(
                        name=schema["function"]["name"],
                        description=schema["function"]["description"],
                        parameters=parameters,
                    )
                )
        return formatted_tools

    def _generate_item_id(self) -> str:
        """Generate a unique item ID for tracking utterances."""
        self._item_counter += 1
        return f"gemini_item_{self._item_counter}"

    async def _start_receive_loop(self) -> None:
        """Start the background receive loop.

        Creates an event queue and spawns a background task that continuously
        receives from the WebSocket and pushes events to the queue.
        """
        if self._receive_task is not None:
            logger.warning("Receive loop already running")
            return

        self._event_queue = asyncio.Queue()
        self._stop_receive = False
        self._current_item_id = self._generate_item_id()

        self._receive_task = asyncio.create_task(self._receive_loop_coro())
        logger.debug("Background receive loop started")

    async def _receive_loop_coro(self) -> None:
        """Background coroutine that continuously receives from the session.

        Runs until _stop_receive is set or an unrecoverable error occurs.
        Pushes all received events to _event_queue.
        """
        logger.debug("Receive loop coroutine starting")
        try:
            while not self._stop_receive and self._session is not None:
                try:
                    # Get a turn iterator
                    logger.debug("Getting turn iterator from session.receive()")
                    turn = self._session.receive()

                    # Consume all responses from this turn
                    response_count = 0
                    deadline_forced_break = False
                    async for response in turn:
                        if self._stop_receive:
                            break

                        response_count += 1
                        # Parse and queue events
                        events = self._parse_response(response)
                        for event in events:
                            await self._event_queue.put(event)

                        # If GoAway was received and the deadline is approaching,
                        # force-break to reconnect before the server disconnects us.
                        if (
                            self._reconnect_at_turn_boundary
                            and self._reconnect_deadline is not None
                            and asyncio.get_running_loop().time()
                            >= self._reconnect_deadline
                        ):
                            logger.warning(
                                "GoAway deadline approaching, "
                                "force-breaking out of turn"
                            )
                            deadline_forced_break = True
                            break

                    logger.debug(f"Turn completed with {response_count} responses")

                    # Emit turn complete unless we force-broke mid-turn
                    if not self._stop_receive and not deadline_forced_break:
                        await self._event_queue.put(
                            GeminiTurnCompleteEvent(type="turn.complete")
                        )
                        await self._event_queue.put(
                            GeminiAudioDoneEvent(
                                type="audio.done",
                                item_id=self._current_item_id,
                            )
                        )
                        # Generate new item ID for next turn
                        self._current_item_id = self._generate_item_id()

                    # GoAway reconnection: signal the adapter and exit.
                    # The adapter will call perform_pending_reconnection() at
                    # the start of the next tick, before any session I/O. This
                    # avoids a race condition where closing the old session here
                    # would conflict with a concurrent send_audio() in the
                    # adapter's asyncio.gather.
                    if self._reconnect_at_turn_boundary and not self._stop_receive:
                        self._reconnect_at_turn_boundary = False
                        self._reconnect_deadline = None
                        self._pending_reconnection = True
                        logger.info(
                            "GoAway: signaling adapter to reconnect at next tick start"
                        )
                        break  # exit while loop cleanly

                except asyncio.CancelledError:
                    logger.debug("Receive loop cancelled")
                    break

                except Exception as e:
                    if self._stop_receive:
                        break

                    code = getattr(e, "code", None)
                    code_matches = code in (1000, 1001) or code in (
                        "1000",
                        "1001",
                    )
                    is_connection_closed = (
                        "ConnectionClosed" in type(e).__name__ or code_matches
                    )

                    logger.error(f"Error in receive loop: {type(e).__name__}: {e}")

                    if is_connection_closed:
                        logger.warning(
                            "Gemini Live API: WebSocket connection closed unexpectedly"
                        )

                        can_resume = (
                            self._max_resumptions > 0
                            and self._resumption_handle is not None
                            and self._resumption_count < self._max_resumptions
                        )

                        # If GoAway was already received, defer reconnection to
                        # the adapter via _pending_reconnection instead of
                        # reconnecting directly here. Calling
                        # _reconnect_with_resumption() from the receive loop
                        # would race with concurrent send_audio() in the
                        # adapter's asyncio.gather.
                        if can_resume and self._go_away_received:
                            self._reconnect_at_turn_boundary = False
                            self._reconnect_deadline = None
                            self._pending_reconnection = True
                            logger.info(
                                "GoAway post-disconnect: signaling adapter "
                                "to reconnect at next tick start"
                            )
                            break

                        # For truly unexpected disconnects (no GoAway), only
                        # attempt direct resumption if resume_only_on_timeout
                        # is disabled.
                        if can_resume and self._resume_only_on_timeout:
                            logger.warning(
                                "Not attempting resumption for unexpected "
                                "disconnect (resume_only_on_timeout=True)"
                            )
                            can_resume = False

                        if can_resume:
                            try:
                                success = await self._reconnect_with_resumption()
                                if success:
                                    logger.info(
                                        "Resumption successful after unexpected "
                                        "disconnect, restarting receive loop"
                                    )
                                    continue
                            except Exception as resume_error:
                                logger.error(
                                    f"Resumption attempt failed: {resume_error}"
                                )

                        raise RuntimeError(
                            f"WebSocket connection closed and resumption failed: {e}"
                        ) from e

                    raise RuntimeError(
                        f"Error receiving from Gemini Live API: {e}"
                    ) from e

        except asyncio.CancelledError:
            logger.debug("Receive loop task cancelled")

        logger.debug("Background receive loop ended")

    async def _stop_receive_loop(self) -> None:
        """Stop the background receive loop.

        Signals the loop to stop and cancels the task.
        """
        self._stop_receive = True

        if self._receive_task is not None:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except (asyncio.CancelledError, Exception):
                pass
            self._receive_task = None

        self._event_queue = None
        logger.debug("Background receive loop stopped")

    async def send_audio(self, audio_data: bytes) -> None:
        """Send audio data to the session.

        Audio should be in 16kHz PCM16 mono format.

        Args:
            audio_data: Raw audio bytes in 16kHz PCM16 format.

        Raises:
            RuntimeError: If not connected to the API.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to API. Call connect() first.")

        # Start receive loop lazily on first audio send
        if self._receive_task is None:
            logger.debug("Starting receive loop on first audio send")
            await self._start_receive_loop()

        from google.genai import types

        audio_blob = types.Blob(
            data=audio_data,
            mime_type=f"audio/pcm;rate={GEMINI_INPUT_SAMPLE_RATE}",
        )
        await self._session.send_realtime_input(audio=audio_blob)
        logger.debug(f"Sent {len(audio_data)} bytes of audio")

    async def send_text(self, text: str, end_of_turn: bool = True) -> None:
        """Send a text message to the session.

        Args:
            text: The text content to send.
            end_of_turn: If True, signals end of user's turn.

        Raises:
            RuntimeError: If not connected to the API.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to API. Call connect() first.")

        await self._session.send(input=text, end_of_turn=end_of_turn)

    async def send_tool_response(
        self,
        call_id: str,
        name: str,
        result: str,
        is_error: bool = False,
    ) -> None:
        """Send the result of a tool/function call back to the API.

        Args:
            call_id: The unique identifier of the function call.
            name: The name of the function that was called.
            result: The string result of the function execution.
            is_error: If True, send the result as an error using the "error"
                key instead of "output". This helps the model understand the
                tool call failed and adjust its behavior accordingly.

        Raises:
            RuntimeError: If not connected to the API.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to API. Call connect() first.")

        from google.genai import types

        if is_error:
            response_payload = {"error": result}
        else:
            response_payload = {"output": result}

        function_response = types.FunctionResponse(
            id=call_id,
            name=name,
            response=response_payload,
        )

        await self._session.send_tool_response(function_responses=[function_response])
        logger.debug(f"Sent tool response for {name}({call_id}), is_error={is_error}")

    async def receive_events_for_duration(
        self, duration_seconds: float
    ) -> List[BaseGeminiEvent]:
        """Receive events for a specified duration.

        Pulls events from the background receive loop's queue for the
        specified duration. The background loop continuously receives from
        the WebSocket, so there are no gaps in reception.

        Args:
            duration_seconds: How long to collect events for.

        Returns:
            List of events received during the duration.

        Raises:
            RuntimeError: If not connected to the API.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to API. Call connect() first.")

        # If receive loop hasn't started yet (no audio sent), start it now
        if self._receive_task is None:
            logger.debug("Starting receive loop in receive_events_for_duration")
            await self._start_receive_loop()

        if self._event_queue is None:
            raise RuntimeError("Background receive loop not running.")

        events: List[BaseGeminiEvent] = []

        # Prepend any events preserved from a pre-reconnection drain
        if self._preserved_events:
            events.extend(self._preserved_events)
            logger.debug(f"Prepended {len(self._preserved_events)} preserved events")
            self._preserved_events.clear()

        start_time = asyncio.get_running_loop().time()
        end_time = start_time + duration_seconds

        while True:
            now = asyncio.get_running_loop().time()
            remaining = end_time - now
            if remaining <= 0:
                break

            try:
                # Use short timeout to check time frequently
                timeout = min(0.05, remaining)
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=timeout,
                )
                events.append(event)

            except asyncio.TimeoutError:
                # No event in queue within timeout, continue checking time
                continue

        return events

    def _serialize_response_for_logging(self, obj: Any, depth: int = 0) -> Any:
        """Serialize a response object for logging, replacing audio data with size info.

        Args:
            obj: The object to serialize.
            depth: Current recursion depth (to prevent infinite recursion).

        Returns:
            A JSON-serializable representation of the object.
        """
        if depth > 10:
            return f"<max depth reached: {type(obj).__name__}>"

        # Handle None
        if obj is None:
            return None

        # Handle bytes - replace with size info
        if isinstance(obj, bytes):
            return f"<bytes: {len(obj)} bytes>"

        # Handle primitive types
        if isinstance(obj, (str, int, float, bool)):
            return obj

        # Handle lists
        if isinstance(obj, (list, tuple)):
            return [
                self._serialize_response_for_logging(item, depth + 1) for item in obj
            ]

        # Handle dicts
        if isinstance(obj, dict):
            return {
                k: self._serialize_response_for_logging(v, depth + 1)
                for k, v in obj.items()
            }

        # Handle objects with attributes
        result = {"_type": type(obj).__name__}
        for attr in dir(obj):
            if attr.startswith("_"):
                continue
            try:
                val = getattr(obj, attr)
                if callable(val):
                    continue
                # Skip some pydantic internals
                if attr in (
                    "model_computed_fields",
                    "model_config",
                    "model_fields",
                    "model_fields_set",
                    "model_extra",
                ):
                    continue
                result[attr] = self._serialize_response_for_logging(val, depth + 1)
            except Exception as e:
                result[attr] = f"<error: {e}>"

        return result

    def _parse_response(self, response: Any) -> List[BaseGeminiEvent]:
        """Parse a Gemini LiveServerMessage into typed events.

        Args:
            response: A LiveServerMessage from session.receive().

        Returns:
            List of typed event objects.
        """
        # Log the full response structure (with audio data replaced by size)
        try:
            serialized = self._serialize_response_for_logging(response)
            import json

            logger.debug(
                f"Gemini response: {json.dumps(serialized, indent=2, default=str)}"
            )
        except Exception as e:
            logger.debug(f"Could not serialize response for logging: {e}")

        events: List[BaseGeminiEvent] = []

        audio_data = None
        if hasattr(response, "data") and response.data:
            audio_data = response.data

        if audio_data:
            events.append(
                GeminiAudioDeltaEvent(
                    type="audio.delta",
                    data=(
                        audio_data
                        if isinstance(audio_data, bytes)
                        else bytes(audio_data)
                    ),
                    item_id=self._current_item_id,
                )
            )

        # Check for tool/function call - LiveServerToolCall contains function_calls array
        if hasattr(response, "tool_call") and response.tool_call:
            tool_call_container = response.tool_call
            # Get the function_calls array from LiveServerToolCall
            function_calls = getattr(tool_call_container, "function_calls", []) or []
            for func_call in function_calls:
                call_id = getattr(func_call, "id", "") or ""
                name = getattr(func_call, "name", "") or ""
                # args can be a dict or a protobuf Struct
                args = getattr(func_call, "args", {})
                if hasattr(args, "items"):
                    args_dict = dict(args)
                else:
                    args_dict = args or {}
                logger.info(f"Gemini function call received: {name}({args_dict})")
                events.append(
                    GeminiFunctionCallDoneEvent(
                        type="function_call.done",
                        call_id=call_id,
                        name=name,
                        arguments=args_dict,
                    )
                )

        # Check for go_away message (server is about to close connection)
        if hasattr(response, "go_away") and response.go_away:
            go_away = response.go_away
            time_left = None
            if hasattr(go_away, "time_left") and go_away.time_left:
                # Parse duration (e.g., "30s" or protobuf Duration)
                time_left_val = go_away.time_left
                if hasattr(time_left_val, "seconds"):
                    time_left = float(time_left_val.seconds)
                elif isinstance(time_left_val, str):
                    # Parse "30s" format
                    try:
                        time_left = float(time_left_val.rstrip("s"))
                    except ValueError:
                        pass
            logger.warning(f"GoAway received from server, time left: {time_left}s")
            self._go_away_received = True
            self._reconnect_at_turn_boundary = True
            safety_margin = 5.0
            effective_time_left = time_left if time_left is not None else 30.0
            self._reconnect_deadline = asyncio.get_running_loop().time() + max(
                effective_time_left - safety_margin, 1.0
            )
            logger.info(
                f"Will reconnect at next turn boundary "
                f"(deadline in {max(effective_time_left - safety_margin, 1.0):.1f}s)"
            )
            events.append(
                GeminiGoAwayEvent(
                    type="go_away",
                    time_left_seconds=time_left,
                )
            )

        # Check for session_resumption_update (handle for resuming session)
        if (
            hasattr(response, "session_resumption_update")
            and response.session_resumption_update
        ):
            update = response.session_resumption_update
            new_handle = getattr(update, "new_handle", None)
            resumable = getattr(update, "resumable", False)

            if new_handle and resumable:
                # Store the handle for potential reconnection
                self._resumption_handle = new_handle
                logger.debug(f"Session resumption handle updated: {new_handle[:20]}...")

            events.append(
                GeminiSessionResumptionEvent(
                    type="session_resumption",
                    new_handle=new_handle,
                    resumable=resumable,
                )
            )

        # Check for server_content (interruption, turn_complete, transcriptions, function calls)
        if hasattr(response, "server_content") and response.server_content:
            server_content = response.server_content
            if hasattr(server_content, "interrupted") and server_content.interrupted:
                events.append(GeminiInterruptionEvent(type="interruption"))
            if (
                hasattr(server_content, "turn_complete")
                and server_content.turn_complete
            ):
                events.append(GeminiTurnCompleteEvent(type="turn.complete"))
            # Check for output transcription (what Gemini said)
            if (
                hasattr(server_content, "output_transcription")
                and server_content.output_transcription
            ):
                transcription = server_content.output_transcription
                if hasattr(transcription, "text") and transcription.text:
                    events.append(
                        GeminiTextDeltaEvent(
                            type="text.delta",
                            text=transcription.text,
                            item_id=self._current_item_id,
                        )
                    )
            # Check for input transcription (what the user said) - inside server_content
            if (
                hasattr(server_content, "input_transcription")
                and server_content.input_transcription
            ):
                transcription = server_content.input_transcription
                if hasattr(transcription, "text") and transcription.text:
                    events.append(
                        GeminiInputTranscriptionEvent(
                            type="input_audio_transcription",
                            transcript=transcription.text,
                        )
                    )
        # If no events were extracted, return unknown event with debug info
        if not events:
            # Collect all attributes for debugging
            attrs = {}
            for attr in dir(response):
                if not attr.startswith("_"):
                    try:
                        val = getattr(response, attr)
                        if not callable(val):
                            attrs[attr] = str(type(val).__name__)
                    except Exception:
                        pass
            logger.debug(f"Unknown response attributes: {attrs}")
            events.append(
                GeminiUnknownEvent(
                    type="unknown",
                    raw={"response_type": str(type(response)), "attrs": attrs},
                )
            )

        return events
