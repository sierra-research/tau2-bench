"""BaiRong OpenAI-Realtime-compatible WebSocket provider.

The protocol implementation is inherited from the OpenAI provider. Only the
endpoint and default model are provider-specific, so the same
event parser and interruption/tool-call behavior are exercised by tau-voice.
"""

from __future__ import annotations

from typing import Optional

import websockets

from tau2.config import (
    DEFAULT_BAIRONG_REALTIME_API_KEY,
    DEFAULT_BAIRONG_REALTIME_BASE_URL,
    DEFAULT_BAIRONG_REALTIME_MODEL,
)
from tau2.data_model.audio import TELEPHONY_AUDIO_FORMAT, AudioFormat
from tau2.voice.audio_native.openai.provider import (
    OpenAIRealtimeProvider,
    OpenAIVADConfig,
)


class BaiRongRealtimeProvider(OpenAIRealtimeProvider):
    """OpenAI-Realtime-compatible client for BaiRong's voice service."""

    BASE_URL = DEFAULT_BAIRONG_REALTIME_BASE_URL
    DEFAULT_MODEL = DEFAULT_BAIRONG_REALTIME_MODEL

    def __init__(
        self,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ):
        """Initialize the client; the hosted endpoint needs no user credential."""
        self.model = model or self.DEFAULT_MODEL
        self.base_url = self.BASE_URL
        self.api_key = DEFAULT_BAIRONG_REALTIME_API_KEY
        self.reasoning_effort = reasoning_effort
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self._current_vad_config: Optional[OpenAIVADConfig] = None
        self._audio_format: AudioFormat = TELEPHONY_AUDIO_FORMAT
        self.session_id: Optional[str] = None


__all__ = [
    "BaiRongRealtimeProvider",
    "DEFAULT_BAIRONG_REALTIME_API_KEY",
    "DEFAULT_BAIRONG_REALTIME_BASE_URL",
    "DEFAULT_BAIRONG_REALTIME_MODEL",
]
