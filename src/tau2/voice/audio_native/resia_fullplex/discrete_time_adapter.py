"""Discrete-time adapter for Resia Fullplex 1.

Resia Fullplex 1 speaks the GPT-Live protocol, so this adapter is the OpenAI
Live adapter with a Resia Fullplex provider: the same tick lifecycle, tool
batches, played-audio speech classification, and timed Live transcript
evidence. Only the provider differs (provider.py).
"""

from pathlib import Path
from typing import Any

from tau2.data_model.audio import AudioFormat
from tau2.data_model.usage import UsageRecord
from tau2.voice.audio_native.openai.live_adapter import DiscreteTimeOpenAILiveAdapter
from tau2.voice.audio_native.tick_result import TickResult

from .provider import ResiaFullplexProvider


class DiscreteTimeResiaFullplexAdapter(DiscreteTimeOpenAILiveAdapter):
    def __init__(
        self,
        *,
        tick_duration_ms: int,
        model: str,
        backend_model: str | None = None,
        reasoning_effort: str | None = None,
        send_audio_instant: bool = False,
        audio_format: AudioFormat | None = None,
        trace_path: Path | None = None,
    ) -> None:
        provider = ResiaFullplexProvider(
            model=model,
            backend_model=backend_model,
            reasoning_effort=reasoning_effort,
            trace_path=trace_path,
        )
        # The Live adapter also builds an OpenAI Live (WebRTC) provider, which
        # reads OPENAI_API_KEY and is never connected; it is replaced here.
        super().__init__(
            tick_duration_ms=tick_duration_ms,
            model=model,
            config=provider.config,
            reasoning_effort=reasoning_effort,
            send_audio_instant=send_audio_instant,
            audio_format=audio_format,
            trace_path=trace_path,
        )
        self._provider = provider

    def connect(self, system_prompt, tools, vad_config=None, modality="audio") -> None:
        if self.is_connected:
            self.disconnect()
        if self.provider._closing:  # one provider per session, as the Live adapter
            self._provider = ResiaFullplexProvider(
                model=self.model,
                backend_model=self._live_config.backend_model,
                reasoning_effort=self._live_reasoning_effort,
                trace_path=self._trace_path,
            )
        super().connect(system_prompt, tools, vad_config, modality)

    async def _async_disconnect(self) -> None:
        try:
            await super()._async_disconnect()
        finally:
            self._record_usage()  # usage that came with the close

    async def _process_event(self, result: TickResult, event: Any) -> None:
        self._record_usage()
        await super()._process_event(result, event)

    def _record_usage(self) -> None:
        """One llm record per final backend response (the model it reports),
        and the server's session seconds as one cumulative realtime record."""
        provider = self.provider
        while provider.helper_usage:
            self.record_usage(provider.helper_usage.pop(0))
        if provider.session_seconds is not None and provider.session_id:
            self.record_usage(
                UsageRecord(
                    provider="resia_fullplex",
                    model=self.model,
                    component="realtime",
                    semantics="cumulative",
                    scope_id=provider.session_id,
                    audio_input_seconds=provider.session_seconds,
                )
            )
            provider.session_seconds = None
