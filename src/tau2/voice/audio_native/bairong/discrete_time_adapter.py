"""Discrete-time adapter for BaiRong Voice Realtime."""

from __future__ import annotations

from tau2.voice.audio_native.bairong.provider import BaiRongRealtimeProvider
from tau2.voice.audio_native.openai.discrete_time_adapter import (
    DiscreteTimeOpenAIAdapter,
)


class DiscreteTimeBaiRongRealtimeAdapter(DiscreteTimeOpenAIAdapter):
    """Reuse OpenAI Realtime tick handling against the BaiRong endpoint."""

    USAGE_PROVIDER = "bairong"

    @property
    def provider(self) -> BaiRongRealtimeProvider:
        if self._provider is None:
            self._provider = BaiRongRealtimeProvider(
                model=self.model,
                reasoning_effort=self.reasoning_effort,
            )
        return self._provider


__all__ = ["DiscreteTimeBaiRongRealtimeAdapter"]
