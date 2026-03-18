"""Qwen Omni Flash Realtime provider for audio native agent.

This module provides integration with Alibaba Cloud's Qwen Omni realtime API
via DashScope for speech-to-speech processing.

Environment Variables:
    DASHSCOPE_API_KEY: API key from Alibaba Cloud Model Studio

Audio Formats:
    - Input: PCM16 at 16kHz
    - Output: PCM16 at 24kHz ("pcm24" refers to sample rate, not bit depth)

Reference: https://www.alibabacloud.com/help/en/model-studio/realtime
"""

from tau2.voice.audio_native.qwen.audio_utils import (
    StreamingQwenConverter,
    calculate_qwen_bytes_per_tick,
    calculate_telephony_bytes_per_tick,
    qwen_output_to_telephony,
    telephony_to_qwen_input,
)
from tau2.voice.audio_native.qwen.discrete_time_adapter import DiscreteTimeQwenAdapter
from tau2.voice.audio_native.qwen.events import (
    BaseQwenEvent,
    QwenAudioDeltaEvent,
    QwenAudioDoneEvent,
    QwenAudioTranscriptDeltaEvent,
    QwenAudioTranscriptDoneEvent,
    QwenErrorEvent,
    QwenFunctionCallArgumentsDoneEvent,
    QwenInputAudioTranscriptionCompletedEvent,
    QwenResponseDoneEvent,
    QwenSpeechStartedEvent,
    QwenSpeechStoppedEvent,
    QwenTimeoutEvent,
    QwenUnknownEvent,
    parse_qwen_event,
)
from tau2.voice.audio_native.qwen.provider import (
    DEFAULT_QWEN_MODEL,
    DEFAULT_QWEN_REALTIME_URL,
    DEFAULT_QWEN_VOICE,
    QWEN_INPUT_BYTES_PER_SECOND,
    QWEN_INPUT_SAMPLE_RATE,
    QWEN_OUTPUT_BYTES_PER_SECOND,
    QWEN_OUTPUT_SAMPLE_RATE,
    QwenRealtimeProvider,
    QwenVADConfig,
    QwenVADMode,
)

__all__ = [
    # Provider
    "QwenRealtimeProvider",
    "QwenVADConfig",
    "QwenVADMode",
    # Adapter
    "DiscreteTimeQwenAdapter",
    # Audio utilities
    "StreamingQwenConverter",
    "telephony_to_qwen_input",
    "qwen_output_to_telephony",
    "calculate_qwen_bytes_per_tick",
    "calculate_telephony_bytes_per_tick",
    # Constants
    "DEFAULT_QWEN_MODEL",
    "DEFAULT_QWEN_REALTIME_URL",
    "DEFAULT_QWEN_VOICE",
    "QWEN_INPUT_SAMPLE_RATE",
    "QWEN_INPUT_BYTES_PER_SECOND",
    "QWEN_OUTPUT_SAMPLE_RATE",
    "QWEN_OUTPUT_BYTES_PER_SECOND",
    # Events
    "BaseQwenEvent",
    "QwenAudioDeltaEvent",
    "QwenAudioDoneEvent",
    "QwenAudioTranscriptDeltaEvent",
    "QwenAudioTranscriptDoneEvent",
    "QwenErrorEvent",
    "QwenFunctionCallArgumentsDoneEvent",
    "QwenInputAudioTranscriptionCompletedEvent",
    "QwenResponseDoneEvent",
    "QwenSpeechStartedEvent",
    "QwenSpeechStoppedEvent",
    "QwenTimeoutEvent",
    "QwenUnknownEvent",
    "parse_qwen_event",
]
