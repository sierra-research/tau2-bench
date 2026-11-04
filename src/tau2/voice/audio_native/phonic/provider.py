"""Phonic speech-to-speech provider for tau2 voice simulation.

Wraps the Phonic speech-to-speech WebSocket. Phonic accepts and returns G.711
mu-law at 8 kHz (``mulaw_8000``), which is tau2's telephony format, so no audio
conversion is needed.

Each session sends the benchmark's system prompt and tools inline. Tools are
``custom_websocket`` tools with their exact tau names, so tool calls arrive as
``tool_call`` events on the same socket, the orchestrator executes them
against the tau environment, and the results go back unchanged via
``send_tool_call_output``.

Every session sets ``phonic_model`` and ``intelligence_level`` from tau2's
model and reasoning-effort settings, ``default_language`` from the call's
language, and uses the default voice. Set
``PHONIC_PROJECT`` to file the conversations under a project other than
``main``.

The ``phonic`` SDK is imported lazily so this module stays importable without
the voice extra.
"""

import asyncio
import base64
import os
from typing import Any, Dict, List, Optional

from loguru import logger

from tau2.config import (
    DEFAULT_PHONIC_API_BASE,
    DEFAULT_PHONIC_PROJECT,
    DEFAULT_PHONIC_VOICE,
    DEFAULT_PHONIC_WS_BASE,
)
from tau2.environment.tool import Tool


class PhonicProvider:
    """Phonic speech-to-speech WebSocket client.

    Lifecycle: ``connect`` (open socket, send config) -> ``send_audio`` /
    ``recv_message`` / ``send_tool_result`` -> ``disconnect``.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        phonic_model: Optional[str] = None,
        intelligence_level: Optional[str] = None,
        language: Optional[str] = None,
    ):
        """Initialize the Phonic provider.

        Args:
            api_key: Phonic API key. If not provided, reads from PHONIC_API_KEY.
            phonic_model: Phonic model sent with each session (e.g. phonic-1).
            intelligence_level: "standard" or "high", sent with each session.
            language: The call's language, sent as default_language so speech
                recognition and voice match it (Phonic otherwise assumes English).
        """
        self.api_key = api_key or os.environ.get("PHONIC_API_KEY")
        if not self.api_key:
            raise ValueError("Phonic API key not provided. Set PHONIC_API_KEY.")
        self.project = os.environ.get("PHONIC_PROJECT", DEFAULT_PHONIC_PROJECT)
        self.phonic_model = phonic_model
        self.intelligence_level = intelligence_level
        self.language = language

        self._cm = None
        self._socket = None
        self._connected = False
        # Phonic conversation ID of the current session, set from
        # conversation_created; recorded as the simulation's provider session.
        self.session_id: Optional[str] = None
        # Serializes websocket writes: audio streams continuously while tool
        # outputs are sent from the tick loop.
        self._send_lock: Optional[asyncio.Lock] = None

    @property
    def is_connected(self) -> bool:
        return self._connected and self._socket is not None

    def _write_lock(self) -> asyncio.Lock:
        if self._send_lock is None:
            self._send_lock = asyncio.Lock()
        return self._send_lock

    async def connect(
        self,
        system_prompt: str,
        tools: List[Tool],
    ) -> None:
        """Open the WebSocket and send the session config."""
        from phonic.client import AsyncPhonic
        from phonic.environment import PhonicEnvironment
        from phonic.types.config_payload import ConfigPayload

        if self.is_connected:
            await self.disconnect()

        environment = PhonicEnvironment(
            base=DEFAULT_PHONIC_API_BASE, production=DEFAULT_PHONIC_WS_BASE
        )
        client = AsyncPhonic(environment=environment, api_key=self.api_key)
        self._send_lock = None

        config: Dict[str, Any] = dict(
            project=self.project,
            input_format="mulaw_8000",
            output_format="mulaw_8000",
            system_prompt=system_prompt,
            tools=[
                {
                    "type": "custom_websocket",
                    "tool_schema": tool.openai_schema,
                    "execution_mode": "sync",
                }
                for tool in tools
            ],
            generate_welcome_message=False,
            welcome_message="",
            voice_id=DEFAULT_PHONIC_VOICE,
        )
        if self.phonic_model:
            config["phonic_model"] = self.phonic_model
        if self.intelligence_level:
            config["intelligence_level"] = self.intelligence_level
        if self.language:
            config["default_language"] = self.language

        self._cm = client.conversations.connect()
        self.session_id = None
        self._socket = await self._cm.__aenter__()
        await self._socket.send_config(ConfigPayload(**config))
        self._connected = True
        logger.info(
            f"PhonicProvider connected (project={self.project}, "
            f"model={self.phonic_model}, intelligence={self.intelligence_level}, "
            f"tools={len(tools)})"
        )

    async def disconnect(self) -> None:
        if self._cm is not None:
            # Closing the socket can take several seconds after the
            # conversation has ended; don't block disconnect on it.
            try:
                await asyncio.wait_for(self._cm.__aexit__(None, None, None), 2.0)
            except asyncio.TimeoutError:
                logger.debug("Phonic socket close handshake timed out")
            except Exception as e:
                logger.warning(f"Error closing Phonic socket: {e}")
        self._cm = None
        self._socket = None
        self._connected = False

    async def send_audio(self, audio: bytes) -> None:
        """Send one chunk of mu-law 8 kHz audio."""
        if not self.is_connected:
            raise RuntimeError("Not connected to Phonic. Call connect() first.")
        from phonic.types.audio_chunk_payload import AudioChunkPayload

        async with self._write_lock():
            await self._socket.send_audio_chunk(
                AudioChunkPayload(audio=base64.b64encode(audio).decode("utf-8"))
            )

    async def recv_message(self, timeout: float) -> Optional[Any]:
        """Receive one payload, or None if nothing arrives within timeout."""
        if not self.is_connected:
            raise RuntimeError("Not connected to Phonic. Call connect() first.")
        try:
            return await asyncio.wait_for(self._socket.recv(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def send_tool_result(self, call_id: str, result: str) -> None:
        """Answer a Phonic tool_call with the tau tool's output."""
        if not self.is_connected:
            raise RuntimeError("Not connected to Phonic. Call connect() first.")
        from phonic.types.tool_call_output_payload import ToolCallOutputPayload

        async with self._write_lock():
            await self._socket.send_tool_call_output(
                ToolCallOutputPayload(tool_call_id=call_id, output=result)
            )
