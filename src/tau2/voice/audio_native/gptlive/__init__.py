"""GPT-Live API (alpha) audio native provider.

Architecture type: Native Multimodal (full-duplex). A single model
continuously processes input audio while generating output audio — it can
listen and speak at the same time. Tool calling is handled via Responses
delegation: the live model delegates work to a configured Responses API
backend model, and client-actionable function calls are returned to the
application for completion.

CONFIDENTIAL: gpt-live is a limited-access alpha (internal testing only).
Expect breaking API changes throughout the alpha.
"""

from tau2.voice.audio_native.gptlive.discrete_time_adapter import (
    DiscreteTimeGPTLiveAdapter,
)
from tau2.voice.audio_native.gptlive.provider import GPTLiveProvider

__all__ = ["DiscreteTimeGPTLiveAdapter", "GPTLiveProvider"]
