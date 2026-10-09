"""Provider for BaiRong's OpenAI-Realtime-compatible voice service."""

from tau2.voice.audio_native.bairong.discrete_time_adapter import (
    DiscreteTimeBaiRongRealtimeAdapter,
)
from tau2.voice.audio_native.bairong.provider import (
    DEFAULT_BAIRONG_REALTIME_API_KEY,
    DEFAULT_BAIRONG_REALTIME_BASE_URL,
    DEFAULT_BAIRONG_REALTIME_MODEL,
    BaiRongRealtimeProvider,
)
from tau2.voice.audio_native.openai.provider import OpenAIVADConfig, OpenAIVADMode

BaiRongRealtimeVADConfig = OpenAIVADConfig
BaiRongRealtimeVADMode = OpenAIVADMode

__all__ = [
    "DiscreteTimeBaiRongRealtimeAdapter",
    "BaiRongRealtimeProvider",
    "BaiRongRealtimeVADConfig",
    "BaiRongRealtimeVADMode",
    "DEFAULT_BAIRONG_REALTIME_API_KEY",
    "DEFAULT_BAIRONG_REALTIME_BASE_URL",
    "DEFAULT_BAIRONG_REALTIME_MODEL",
]
