"""Pine provider for tau2-bench audio-native evaluation.

Pine exposes an OpenAI-Realtime-API-compatible endpoint. The wire format
(event names, JSON shapes, base64 audio framing) matches the OpenAI provider;
the differences are:
- Different WebSocket endpoint (configurable via PINE_BASE_URL).
- Different bearer token (PINE_API_KEY).
- Single-modality audio sessions only.
- Server-VAD turn detection only.

See https://tau-bench.pinevoice.ai/ for how to obtain an API key.
"""

from tau2.voice.audio_native.pine.discrete_time_adapter import (
    DiscreteTimePineAdapter,
)
from tau2.voice.audio_native.pine.events import (
    BasePineEvent,
    parse_pine_event,
)
from tau2.voice.audio_native.pine.provider import (
    PineProvider,
    PineVADConfig,
    PineVADMode,
)

__all__ = [
    "DiscreteTimePineAdapter",
    "PineProvider",
    "PineVADConfig",
    "PineVADMode",
    "BasePineEvent",
    "parse_pine_event",
]
