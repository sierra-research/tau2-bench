"""Apriel discrete-time adapter using the Apriel tick runner."""

from typing import Any, List, Optional

from loguru import logger

from tau2.config import (
    DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT,
    DEFAULT_AUDIO_NATIVE_DISCONNECT_TIMEOUT,
    DEFAULT_AUDIO_NATIVE_TICK_TIMEOUT_BUFFER,
)
from tau2.data_model.audio import AudioFormat
from tau2.environment.tool import Tool
from tau2.voice.audio_native.adapter import DiscreteTimeAdapter
from tau2.voice.audio_native.apriel.provider import AprielRealtimeProvider
from tau2.voice.audio_native.apriel.tick_runner import TickRunner
from tau2.voice.audio_native.async_loop import BackgroundAsyncLoop
from tau2.voice.audio_native.tick_result import TickResult


class DiscreteTimeAprielAdapter(DiscreteTimeAdapter):
    """Adapter for discrete-time simulation with Apriel's realtime gateway."""

    def __init__(
        self,
        tick_duration_ms: int,
        send_audio_instant: bool,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        audio_format: Optional[AudioFormat] = None,
        provider_trace: bool = True,
        buffer_until_complete: bool = False,
        fast_forward_mode: bool = False,
        manual_commit: bool = False,
    ):
        super().__init__(
            tick_duration_ms=tick_duration_ms,
            audio_format=audio_format,
            send_audio_instant=send_audio_instant,
        )

        self.model = model
        self.reasoning_effort = reasoning_effort

        self._provider = AprielRealtimeProvider(
            model=model,
            trace_enabled=provider_trace,
            reasoning_effort=reasoning_effort,
        )
        self._owns_provider = True
        self._bg_loop = BackgroundAsyncLoop()
        self._connected = False

        chunk_size = int(
            self.audio_format.bytes_per_second * self._voip_interval_ms / 1000
        )
        self._tick_runner = TickRunner(
            provider=self._provider,
            tick_duration_ms=self.tick_duration_ms,
            bytes_per_tick=self.bytes_per_tick,
            send_audio_instant=self.send_audio_instant,
            buffer_until_complete=buffer_until_complete,
            chunk_size=chunk_size,
            voip_packet_interval_ms=self._voip_interval_ms,
            audio_format=self.audio_format,
            fast_forward_mode=fast_forward_mode,
            manual_commit=manual_commit,
        )

    @property
    def provider(self) -> AprielRealtimeProvider:
        return self._provider

    @property
    def is_connected(self) -> bool:
        return self._connected and self._bg_loop.is_running

    def connect(
        self,
        system_prompt: str,
        tools: List[Tool],
        vad_config: Any = None,
        modality: str = "audio",
    ) -> None:
        if self._connected:
            logger.warning("Already connected, disconnecting first")
            self.disconnect()

        self._bg_loop.start()
        try:
            self._bg_loop.run_coroutine(
                self._async_connect(system_prompt, tools, vad_config, modality),
                timeout=DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT,
            )
            self._connected = True
            logger.info(
                "DiscreteTimeAprielAdapter connected "
                f"(tick={self.tick_duration_ms}ms, bytes_per_tick={self.bytes_per_tick})"
            )
        except Exception as exc:
            self._bg_loop.stop()
            raise RuntimeError(
                f"Failed to connect to Apriel Realtime API: {exc}"
            ) from exc

    async def _async_connect(self, system_prompt, tools, vad_config, modality) -> None:
        await self.provider.connect()
        await self.provider.configure_session(
            system_prompt=system_prompt,
            tools=tools,
            vad_config=vad_config,
            modality=modality,
            audio_format=self.audio_format,
        )

    def disconnect(self) -> None:
        if not self._connected:
            return

        if self._bg_loop.is_running:
            try:
                self._bg_loop.run_coroutine(
                    self._async_disconnect(),
                    timeout=DEFAULT_AUDIO_NATIVE_DISCONNECT_TIMEOUT,
                )
            except Exception as exc:
                logger.warning(f"Error during disconnect: {exc}")

        self._bg_loop.stop()
        self._connected = False
        self._tick_count = 0
        self._cumulative_user_audio_ms = 0
        self.clear_buffers()
        logger.info("DiscreteTimeAprielAdapter disconnected")

    async def _async_disconnect(self) -> None:
        if self._owns_provider and self._provider is not None:
            await self.provider.disconnect()

    def run_tick(
        self, user_audio: bytes, tick_number: Optional[int] = None
    ) -> TickResult:
        if not self.is_connected:
            raise RuntimeError(
                "Not connected to Apriel Realtime API. Call connect() first."
            )

        if tick_number is None:
            tick_number = self._tick_count
        self._tick_count = tick_number + 1

        try:
            return self._bg_loop.run_coroutine(
                self._async_run_tick(user_audio, tick_number),
                timeout=self.tick_duration_ms / 1000
                + DEFAULT_AUDIO_NATIVE_TICK_TIMEOUT_BUFFER,
            )
        except Exception as exc:
            logger.error(f"Error in run_tick (tick={tick_number}): {exc}")
            raise

    async def _async_run_tick(self, user_audio: bytes, tick_number: int) -> TickResult:
        await self._flush_pending_tool_results()
        result = await self._tick_runner.run_tick(
            user_audio=user_audio, tick_number=tick_number
        )
        if self._tick_usage_buffer:
            result.usage_records.extend(self._tick_usage_buffer)
            self._tick_usage_buffer.clear()
        return result

    async def _execute_tick(
        self,
        user_audio: bytes,
        tick_number: int,
        result: TickResult,
        tick_start: float,
    ) -> None:
        raise RuntimeError(
            "DiscreteTimeAprielAdapter uses tau2.voice.audio_native.apriel.tick_runner.TickRunner; "
            "_execute_tick is not used."
        )

    async def _flush_pending_tool_results(self) -> None:
        for (
            call_id,
            result_str,
            request_response,
            _is_error,
        ) in self._pending_tool_results:
            await self.provider.send_tool_result(call_id, result_str, request_response)
        self._pending_tool_results.clear()
