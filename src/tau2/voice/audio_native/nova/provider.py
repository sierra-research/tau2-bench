"""
Amazon Nova Sonic API provider for real-time voice processing.

Uses AWS Bedrock's bidirectional streaming API for speech-to-speech.
Authentication is handled via AWS credentials (boto3/botocore).

Key features:
- Input: LPCM 16kHz audio (requires conversion from G.711 μ-law for telephony)
- Output: LPCM 24kHz audio (requires conversion to G.711 μ-law for telephony)
- Tool/function calling support
- Barge-in (interruption) support with server-side VAD
- VAD requires silence after speech to detect turn end

Reference: AWS Bedrock Nova Sonic documentation
https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-getting-started.html
"""

import asyncio
import base64
import json
import os
import uuid
from enum import Enum
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Optional

from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel

from tau2.config import (
    DEFAULT_NOVA_INPUT_SAMPLE_RATE,
    DEFAULT_NOVA_MODEL,
    DEFAULT_NOVA_OUTPUT_SAMPLE_RATE,
    DEFAULT_NOVA_REGION,
    DEFAULT_NOVA_VOICE,
)
from tau2.environment.tool import Tool
from tau2.voice.audio_native.nova.events import (
    BaseNovaEvent,
    NovaTimeoutEvent,
    NovaUnknownEvent,
    parse_nova_event,
)

# Type hints only - these don't require the packages to be installed
if TYPE_CHECKING:
    from aws_sdk_bedrock_runtime.client import BedrockRuntimeClient


def _check_nova_dependencies() -> None:
    """Check if Nova Sonic dependencies are installed and raise helpful error if not."""
    missing = []
    try:
        import boto3  # noqa: F401
    except ImportError:
        missing.append("boto3")
    try:
        import aws_sdk_bedrock_runtime  # noqa: F401
    except ImportError:
        missing.append("aws-sdk-bedrock-runtime")
    try:
        import smithy_aws_core  # noqa: F401
    except ImportError:
        missing.append("smithy-aws-core")
    try:
        import smithy_core  # noqa: F401
    except ImportError:
        missing.append("smithy-core")

    if missing:
        raise ImportError(
            f"Nova Sonic provider requires additional dependencies: {', '.join(missing)}. "
            f"Install them with: pip install {' '.join(missing)}"
        )


load_dotenv()

# Audio format constants (from config)
NOVA_INPUT_SAMPLE_RATE = DEFAULT_NOVA_INPUT_SAMPLE_RATE
NOVA_SAMPLE_RATE = NOVA_INPUT_SAMPLE_RATE  # Alias for compatibility
NOVA_BYTES_PER_SECOND = NOVA_SAMPLE_RATE * 2  # 2 bytes per sample

NOVA_OUTPUT_SAMPLE_RATE = DEFAULT_NOVA_OUTPUT_SAMPLE_RATE
NOVA_OUTPUT_BYTES_PER_SECOND = NOVA_OUTPUT_SAMPLE_RATE * 2


class NovaVADMode(str, Enum):
    """Voice Activity Detection modes for Nova Sonic.

    Note: Nova Sonic only supports SERVER_VAD. MANUAL mode is defined for
    API compatibility but will raise ValueError if used.
    """

    SERVER_VAD = (
        "server_vad"  # Server handles VAD automatically (default, only supported mode)
    )
    MANUAL = "manual"  # Not supported - will raise ValueError


class NovaVADConfig(BaseModel):
    """Configuration for Nova Sonic's Voice Activity Detection.

    Note: Nova Sonic only supports server-side VAD. The mode must be SERVER_VAD.
    MANUAL mode is not supported and will raise ValueError.

    Attributes:
        mode: VAD mode. Must be SERVER_VAD (Nova handles barge-in automatically).
    """

    mode: NovaVADMode = NovaVADMode.SERVER_VAD


def _create_credentials_resolver(profile_name: Optional[str] = None) -> Any:
    """Create a credentials resolver for AWS authentication.

    This factory function lazily imports the required AWS dependencies.

    Args:
        profile_name: AWS profile name to use. If None, uses default chain.

    Returns:
        An IdentityResolver instance for AWS credentials.
    """
    import boto3
    from smithy_aws_core.identity import AWSCredentialsIdentity
    from smithy_core.aio.interfaces.identity import IdentityResolver

    class Boto3CredentialsResolver(IdentityResolver):
        """IdentityResolver that sources AWS credentials from boto3.

        Delegates to boto3.Session() credential chain which checks environment
        variables, shared credentials files, EC2 instance profiles, etc.
        """

        def __init__(self, profile_name: Optional[str] = None):
            """Initialize with optional profile name.

            Args:
                profile_name: AWS profile name to use. If None, uses default chain.
            """
            self.session = boto3.Session(profile_name=profile_name)
            self._cached_identity: Optional[AWSCredentialsIdentity] = None

        async def get_identity(self, **kwargs) -> AWSCredentialsIdentity:
            """Get AWS credentials as an identity object.

            Returns:
                AWSCredentialsIdentity with access key, secret, and optional token.
            """
            creds = self.session.get_credentials()
            if creds is None:
                raise ValueError("No AWS credentials found")

            frozen = creds.get_frozen_credentials()
            return AWSCredentialsIdentity(
                access_key_id=frozen.access_key,
                secret_access_key=frozen.secret_key,
                session_token=frozen.token,
            )

    return Boto3CredentialsResolver(profile_name)


class NovaSonicProvider:
    """Amazon Nova Sonic API provider with bidirectional streaming.

    This provider manages a bidirectional streaming connection to AWS Bedrock's
    Nova Sonic API, enabling real-time speech-to-speech processing.

    Uses the aws_sdk_bedrock_runtime package which properly handles
    bidirectional event streams via HTTP/2.

    Key differences from OpenAI/xAI:
    - Uses AWS authentication (SigV4)
    - LPCM 16kHz audio format (needs conversion for telephony)
    - Different event protocol (sessionStart, contentStart, audioOutput, etc.)

    Attributes:
        model_id: The Nova Sonic model identifier.
        region: AWS region for Bedrock.
        voice: The voice to use (matthew, tiffany, amy).

    Example:
        ```python
        provider = NovaSonicProvider()
        await provider.connect()
        await provider.configure_session(
            system_prompt="You are a helpful assistant.",
            tools=[],
            vad_config=NovaVADConfig(),
        )
        await provider.send_audio(audio_bytes)
        async for event in provider.receive_events():
            print(event)
        await provider.disconnect()
        ```
    """

    DEFAULT_MODEL = DEFAULT_NOVA_MODEL
    DEFAULT_VOICE = DEFAULT_NOVA_VOICE
    DEFAULT_REGION = DEFAULT_NOVA_REGION

    # Nested class for VAD config (for compatibility with agent factory pattern)
    NovaVADConfig = NovaVADConfig

    def __init__(
        self,
        model_id: Optional[str] = None,
        region: Optional[str] = None,
        voice: Optional[str] = None,
        profile_name: Optional[str] = None,
    ):
        """Initialize the Nova Sonic provider.

        Args:
            model_id: Nova Sonic model ID. Defaults to amazon.nova-2-sonic-v1:0.
            region: AWS region. If not provided, reads from AWS_DEFAULT_REGION
                or AWS_REGION environment variable.
            voice: Voice to use. One of: matthew, tiffany, amy. Defaults to tiffany.
            profile_name: AWS profile name. If not provided, reads from AWS_PROFILE
                environment variable or uses default credential chain.

        Raises:
            ValueError: If AWS credentials cannot be found.
        """
        self.model_id = model_id or self.DEFAULT_MODEL
        self.region = region or os.environ.get(
            "AWS_DEFAULT_REGION", os.environ.get("AWS_REGION", self.DEFAULT_REGION)
        )
        self.voice = voice or self.DEFAULT_VOICE
        self.profile_name = profile_name or os.environ.get("AWS_PROFILE")

        # Session state
        self._session_id: Optional[str] = None
        self._prompt_name: Optional[str] = None
        self._content_id: int = 0
        self._stream: Any = None
        self._output_stream: Any = None  # Separate handle for output stream
        self._client: Optional["BedrockRuntimeClient"] = None
        self._is_connected = False
        self._current_vad_config: Optional[NovaVADConfig] = None
        self._credentials_resolver: Any = None

        # Initialize the client (this will check dependencies)
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the Bedrock runtime client.

        This lazily imports the AWS SDK dependencies, so they're only required
        when actually using the Nova Sonic provider.
        """
        # Check dependencies first - gives helpful error message
        _check_nova_dependencies()

        # Now import the AWS SDK (we know it's available)
        from aws_sdk_bedrock_runtime.client import BedrockRuntimeClient
        from aws_sdk_bedrock_runtime.config import (
            Config,
            HTTPAuthSchemeResolver,
            SigV4AuthScheme,
        )

        try:
            # Create credentials resolver
            self._credentials_resolver = _create_credentials_resolver(self.profile_name)

            if self.profile_name:
                logger.info(f"Using AWS profile: {self.profile_name}")

            # Create config for Smithy-based client
            config = Config(
                endpoint_uri=f"https://bedrock-runtime.{self.region}.amazonaws.com",
                region=self.region,
                aws_credentials_identity_resolver=self._credentials_resolver,
                auth_scheme_resolver=HTTPAuthSchemeResolver(),
                auth_schemes={"aws.auth#sigv4": SigV4AuthScheme(service="bedrock")},
            )

            self._client = BedrockRuntimeClient(config=config)
            logger.info(
                f"Nova Sonic: Initialized Bedrock client for region {self.region}"
            )

        except Exception as e:
            raise ValueError(f"Failed to initialize AWS Bedrock client: {e}")

    @property
    def is_connected(self) -> bool:
        """Check if the streaming connection is active."""
        return self._is_connected and self._stream is not None

    async def connect(self) -> None:
        """Establish a bidirectional streaming connection to Nova Sonic.

        Opens a new streaming session using the aws_sdk_bedrock_runtime SDK.

        Raises:
            RuntimeError: If the connection fails.
        """
        if self.is_connected:
            return

        from aws_sdk_bedrock_runtime.client import (
            InvokeModelWithBidirectionalStreamOperationInput,
        )

        try:
            # Generate session ID
            self._session_id = str(uuid.uuid4())
            self._prompt_name = f"prompt-{uuid.uuid4()}"
            self._content_id = 0

            logger.info(
                f"Nova Sonic: Connecting with session {self._session_id}, "
                f"model {self.model_id}"
            )

            # Start bidirectional stream using the Smithy SDK
            self._stream = await self._client.invoke_model_with_bidirectional_stream(
                InvokeModelWithBidirectionalStreamOperationInput(model_id=self.model_id)
            )

            # Note: We don't call await_output() here because it blocks until
            # the server sends something. We'll get the output stream lazily
            # when we first try to receive events.
            self._output_stream = None  # Will be set on first receive

            self._is_connected = True

            # Send session start event
            await self._send_event(
                {
                    "event": {
                        "sessionStart": {
                            "inferenceConfiguration": {
                                "maxTokens": 4096,  # ~2 min audio at 32 tokens/sec
                                "topP": 0.9,
                                "temperature": 1.0,
                            }
                        }
                    }
                }
            )

            logger.info("Nova Sonic: Connected successfully")

        except Exception as e:
            self._is_connected = False
            self._stream = None
            logger.error(f"Nova Sonic: Connection failed: {e}")
            raise RuntimeError(f"Failed to connect to Nova Sonic: {e}")

    async def disconnect(self) -> None:
        """Close the streaming connection."""
        if not self._is_connected:
            return

        try:
            logger.info("Nova Sonic: Disconnecting")

            # Send session end event
            try:
                await self._send_event({"event": {"sessionEnd": {}}})
            except Exception as e:
                logger.debug(f"Error sending sessionEnd: {e}")

            # Close the output stream
            if self._output_stream:
                try:
                    await self._output_stream.close()
                except Exception as e:
                    logger.debug(f"Error closing output stream: {e}")

            # Close the input stream
            if self._stream:
                try:
                    await self._stream.input_stream.close()
                except Exception as e:
                    logger.debug(f"Error closing input stream: {e}")

            self._output_stream = None
            self._stream = None
            self._is_connected = False
            self._session_id = None

            logger.info("Nova Sonic: Disconnected")

        except Exception as e:
            logger.error(f"Nova Sonic: Error during disconnect: {e}")
            self._is_connected = False
            self._stream = None

    async def _send_event(self, event: Dict) -> None:
        """Send a JSON event to the bidirectional stream.

        Args:
            event: Event dictionary to send.
        """
        if not self._stream:
            raise RuntimeError("Not connected to Nova Sonic")

        from aws_sdk_bedrock_runtime.models import (
            BidirectionalInputPayloadPart,
            InvokeModelWithBidirectionalStreamInputChunk,
        )

        try:
            event_json = json.dumps(event)
            event_bytes = event_json.encode("utf-8")

            # Wrap in SDK types for proper serialization
            chunk = InvokeModelWithBidirectionalStreamInputChunk(
                value=BidirectionalInputPayloadPart(bytes_=event_bytes)
            )

            # Send through the input stream
            await self._stream.input_stream.send(chunk)

            # Log event (excluding audio content)
            log_event = event.copy()
            if "event" in log_event and "audioInput" in log_event.get("event", {}):
                audio_input = log_event["event"]["audioInput"]
                if "content" in audio_input:
                    audio_input["content"] = (
                        f"<{len(audio_input['content'])} b64 chars>"
                    )
            logger.debug(f"Nova Sonic sent: {log_event}")

        except Exception as e:
            logger.error(f"Nova Sonic: Error sending event: {e}")
            raise

    def _format_tools_for_api(self, tools: List[Tool]) -> List[Dict]:
        """Format tools for the Nova Sonic API.

        Nova uses a similar but slightly different tool format.
        IMPORTANT: inputSchema.json must be a JSON STRING, not a dict!
        """
        formatted_tools = []
        for tool in tools:
            schema = tool.openai_schema
            # Nova Sonic expects inputSchema.json to be a JSON STRING
            parameters_json = json.dumps(schema["function"]["parameters"])
            formatted_tools.append(
                {
                    "toolSpec": {
                        "name": schema["function"]["name"],
                        "description": schema["function"]["description"],
                        "inputSchema": {"json": parameters_json},
                    }
                }
            )
        return formatted_tools

    async def configure_session(
        self,
        system_prompt: str,
        tools: List[Tool],
        vad_config: NovaVADConfig,
    ) -> None:
        """Configure the session with instructions, tools, and settings.

        Sends a promptStart event with the system prompt and tool configuration.

        Args:
            system_prompt: The system instructions for the assistant.
            tools: List of tools available for the assistant to use.
            vad_config: Voice Activity Detection configuration.

        Raises:
            RuntimeError: If not connected or if configuration fails.
            ValueError: If vad_config.mode is not SERVER_VAD (only server-side VAD is supported).
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to API. Call connect() first.")

        # Nova Sonic only supports server-side VAD
        if vad_config.mode != NovaVADMode.SERVER_VAD:
            raise ValueError(
                f"Nova Sonic only supports SERVER_VAD mode. Got: {vad_config.mode}. "
                "MANUAL mode is not supported."
            )

        self._current_vad_config = vad_config

        # Build prompt start event
        prompt_start = {
            "event": {
                "promptStart": {
                    "promptName": self._prompt_name,
                    "textOutputConfiguration": {"mediaType": "text/plain"},
                    "audioOutputConfiguration": {
                        "mediaType": "audio/lpcm",
                        "sampleRateHertz": NOVA_OUTPUT_SAMPLE_RATE,
                        "sampleSizeBits": 16,
                        "channelCount": 1,
                        "voiceId": self.voice,
                    },
                }
            }
        }

        # Add tool configuration if tools are provided
        if tools:
            prompt_start["event"]["promptStart"]["toolUseOutputConfiguration"] = {
                "mediaType": "application/json"
            }
            prompt_start["event"]["promptStart"]["toolConfiguration"] = {
                "tools": self._format_tools_for_api(tools)
            }

        await self._send_event(prompt_start)

        # Send system prompt as text content
        if system_prompt:
            await self._send_system_prompt(system_prompt)

        logger.info("Nova Sonic: Session configured successfully")

    async def _send_system_prompt(self, system_prompt: str) -> None:
        """Send the system prompt as a text content block."""
        self._content_id += 1
        content_id = str(self._content_id)

        # Content start - SYSTEM prompt also needs interactive: true per AWS docs
        await self._send_event(
            {
                "event": {
                    "contentStart": {
                        "promptName": self._prompt_name,
                        "contentName": content_id,
                        "type": "TEXT",
                        "interactive": True,
                        "role": "SYSTEM",
                        "textInputConfiguration": {"mediaType": "text/plain"},
                    }
                }
            }
        )

        # Text input
        await self._send_event(
            {
                "event": {
                    "textInput": {
                        "promptName": self._prompt_name,
                        "contentName": content_id,
                        "content": system_prompt,
                    }
                }
            }
        )

        # Content end
        await self._send_event(
            {
                "event": {
                    "contentEnd": {
                        "promptName": self._prompt_name,
                        "contentName": content_id,
                    }
                }
            }
        )

    async def send_text(self, text: str) -> None:
        """Send a text message from the user.

        Args:
            text: The text content of the user's message.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to API")

        self._content_id += 1
        content_id = str(self._content_id)

        # Content start for user text (interactive=true triggers model response)
        await self._send_event(
            {
                "event": {
                    "contentStart": {
                        "promptName": self._prompt_name,
                        "contentName": content_id,
                        "type": "TEXT",
                        "role": "USER",
                        "interactive": True,
                        "textInputConfiguration": {"mediaType": "text/plain"},
                    }
                }
            }
        )

        # Text input
        await self._send_event(
            {
                "event": {
                    "textInput": {
                        "promptName": self._prompt_name,
                        "contentName": content_id,
                        "content": text,
                    }
                }
            }
        )

        # Content end
        await self._send_event(
            {
                "event": {
                    "contentEnd": {
                        "promptName": self._prompt_name,
                        "contentName": content_id,
                    }
                }
            }
        )

    async def start_audio_content(self) -> str:
        """Start an audio content block for streaming audio input.

        Returns:
            The content ID for this audio block.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to API")

        self._content_id += 1
        content_id = str(self._content_id)

        await self._send_event(
            {
                "event": {
                    "contentStart": {
                        "promptName": self._prompt_name,
                        "contentName": content_id,
                        "type": "AUDIO",
                        "role": "USER",
                        "interactive": True,
                        "audioInputConfiguration": {
                            "mediaType": "audio/lpcm",
                            "sampleRateHertz": NOVA_SAMPLE_RATE,
                            "sampleSizeBits": 16,
                            "channelCount": 1,
                        },
                    }
                }
            }
        )

        return content_id

    async def start_audio_stream(self) -> str:
        """Start the audio input stream for continuous audio streaming.

        This establishes the audio content block that Nova Sonic requires
        for real-time interaction. Call this after configure_session() and
        before sending any audio data.

        Returns:
            The content ID for the audio stream.
        """
        return await self.start_audio_content()

    async def send_audio(
        self, audio_data: bytes, content_id: Optional[str] = None
    ) -> str:
        """Send audio data to the input stream.

        Audio should be in LPCM 16kHz format. If you have μ-law audio,
        convert it first using audio_utils.telephony_to_nova_input().

        Args:
            audio_data: Raw audio bytes in LPCM 16kHz format.
            content_id: Optional content ID from start_audio_content().
                If not provided, a new audio block is started automatically.

        Returns:
            The content ID for this audio chunk.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to API")

        # Start a new audio content block if needed
        if content_id is None:
            content_id = await self.start_audio_content()

        # Encode and send audio
        audio_b64 = base64.b64encode(audio_data).decode("utf-8")
        await self._send_event(
            {
                "event": {
                    "audioInput": {
                        "promptName": self._prompt_name,
                        "contentName": content_id,
                        "content": audio_b64,
                    }
                }
            }
        )

        return content_id

    async def end_audio_content(self, content_id: str) -> None:
        """End an audio content block.

        Args:
            content_id: The content ID from start_audio_content() or send_audio().
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to API")

        await self._send_event(
            {
                "event": {
                    "contentEnd": {
                        "promptName": self._prompt_name,
                        "contentName": content_id,
                    }
                }
            }
        )

    async def send_tool_result(
        self, tool_use_id: str, result: str, request_response: bool = True
    ) -> None:
        """Send the result of a tool/function call back to the API.

        Per LiveKit implementation and AWS docs, tool results require:
        1. contentStart with type=TOOL, role=TOOL, toolResultInputConfiguration
        2. toolResult with the actual content
        3. contentEnd

        Args:
            tool_use_id: The unique identifier of the tool use request.
            result: The string result of the function execution.
            request_response: Ignored for Nova (async tool calling continues automatically).
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to API")

        self._content_id += 1
        content_id = str(self._content_id)

        # 1. Content start for tool result
        await self._send_event(
            {
                "event": {
                    "contentStart": {
                        "promptName": self._prompt_name,
                        "contentName": content_id,
                        "type": "TOOL",
                        "role": "TOOL",
                        "interactive": False,
                        "toolResultInputConfiguration": {
                            "toolUseId": tool_use_id,
                            "type": "TEXT",
                            "textInputConfiguration": {"mediaType": "text/plain"},
                        },
                    }
                }
            }
        )

        # 2. Tool result content - must be JSON per toolUseOutputConfiguration
        # If the result is not already JSON, wrap it
        try:
            json.loads(result)
            json_result = result
        except (json.JSONDecodeError, ValueError):
            json_result = json.dumps({"result": result})

        await self._send_event(
            {
                "event": {
                    "toolResult": {
                        "promptName": self._prompt_name,
                        "contentName": content_id,
                        "content": json_result,
                    }
                }
            }
        )

        # 3. Content end
        await self._send_event(
            {
                "event": {
                    "contentEnd": {
                        "promptName": self._prompt_name,
                        "contentName": content_id,
                    }
                }
            }
        )

    async def receive_events(self) -> AsyncGenerator[BaseNovaEvent, None]:
        """Receive and yield events from the response stream.

        Yields:
            BaseNovaEvent: Parsed event objects.

        Raises:
            RuntimeError: If connection closes unexpectedly.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to API")

        while self.is_connected:
            try:
                # Read from response stream with timeout
                try:
                    event_data = await asyncio.wait_for(
                        self._read_next_event(),
                        timeout=0.1,
                    )

                    if event_data is not None:
                        event = parse_nova_event(event_data)
                        yield event
                    else:
                        # Stream ended
                        logger.info("Nova Sonic: Response stream ended")
                        break

                except asyncio.TimeoutError:
                    yield NovaTimeoutEvent(event_type="timeout")

            except StopAsyncIteration:
                logger.info("Nova Sonic: Stream iteration complete")
                break
            except Exception as e:
                logger.error(f"Nova Sonic: Error receiving event: {e}")
                yield NovaUnknownEvent(event_type="error", raw={"error": str(e)})

    async def _ensure_output_stream(self) -> bool:
        """Lazily initialize the output stream.

        Returns:
            True if output stream is available, False otherwise.
        """
        if self._output_stream is not None:
            return True

        if self._stream is None:
            return False

        try:
            # This blocks until server sends first response
            _, self._output_stream = await self._stream.await_output()
            return True
        except Exception as e:
            logger.error(f"Nova Sonic: Failed to get output stream: {e}")
            return False

    async def _read_next_event(self) -> Optional[Dict]:
        """Read the next event from the response stream.

        Returns:
            Event data dictionary, or None if stream ended.
        """
        try:
            # Lazily get output stream
            if not await self._ensure_output_stream():
                return None

            # Get next event from the output stream using receive()
            result = await self._output_stream.receive()
            if result is None:
                return None

            # Extract the bytes from the result
            # The result is an InvokeModelWithBidirectionalStreamOutputChunk
            if hasattr(result, "value") and result.value is not None:
                value = result.value
                if hasattr(value, "bytes_") and value.bytes_ is not None:
                    data = json.loads(value.bytes_.decode("utf-8"))
                    return data

            return None

        except StopAsyncIteration:
            return None
        except Exception as e:
            logger.debug(f"Nova Sonic: Error reading event: {e}")
            return None

    async def receive_events_for_duration(
        self, duration_seconds: float
    ) -> List[BaseNovaEvent]:
        """Receive events for a specified duration.

        Collects all events that arrive within the specified time window.
        Useful for tick-based processing.

        Args:
            duration_seconds: How long to collect events.

        Returns:
            List of events received during the duration.
        """
        events = []
        end_time = asyncio.get_event_loop().time() + duration_seconds

        async for event in self.receive_events():
            if not isinstance(event, NovaTimeoutEvent):
                events.append(event)

            if asyncio.get_event_loop().time() >= end_time:
                break

        return events
