import asyncio

import pytest

pytest.importorskip("aiortc")

import tau2.voice.audio_native.avartha_pipecat.webrtc as module  # noqa: E402
from tau2.voice.audio_native.avartha_pipecat.config import (
    AvarthaPipecatConfig,  # noqa: E402
)
from tau2.voice.audio_native.avartha_pipecat.provider import (
    ProviderErrorEvent,  # noqa: E402
)


@pytest.mark.parametrize("failure", ["recv", "resample"])
def test_media_task_failure_reaches_provider_error(monkeypatch, failure):
    class Track:
        async def recv(self):
            if failure == "recv":
                raise RuntimeError("injected decoder failure")
            return object()

    class Resampler:
        def __init__(self, **kw):
            pass

        def resample(self, frame):
            raise ValueError("injected resampler failure")

    monkeypatch.setattr(module, "AudioResampler", Resampler)

    async def run():
        provider = module.AvarthaWebRTCProvider(
            AvarthaPipecatConfig(base_url="https://platform.example/agent")
        )
        try:
            await provider._receive_audio(Track())
        except Exception:
            pass
        assert provider._session_error is not None, "Media task failure was lost"
        event = provider._queue.get_nowait()
        assert isinstance(event, ProviderErrorEvent)
        assert "injected" in event.message

    asyncio.run(run())


def test_media_cancellation_does_not_fail_session():
    class Track:
        async def recv(self):
            raise asyncio.CancelledError

    async def run():
        provider = module.AvarthaWebRTCProvider(
            AvarthaPipecatConfig(base_url="https://platform.example/agent")
        )
        with pytest.raises(asyncio.CancelledError):
            await provider._receive_audio(Track())
        assert provider._session_error is None
        assert provider._queue.empty()

    asyncio.run(run())
