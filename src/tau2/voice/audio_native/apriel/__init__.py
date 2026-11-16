"""Apriel realtime provider using the OpenAI-compatible wire protocol.

The provider and adapter are deliberately thin and share the same event/model
surface as tau's OpenAI implementation, but Apriel uses its own tick runner
implementation (mirroring the OpenAI tick runner behavior).
"""

from tau2.voice.audio_native.apriel.discrete_time_adapter import DiscreteTimeAprielAdapter
from tau2.voice.audio_native.apriel.provider import AprielRealtimeProvider

__all__ = ["AprielRealtimeProvider", "DiscreteTimeAprielAdapter"]
