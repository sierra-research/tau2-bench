"""TTS concurrency ceiling (see tau2.voice.synthesis.synthesize).

Providers cap concurrent requests per subscription and answer 429
`concurrent_limit_exceeded` over it. A synthesis that burns its retries is a
caller utterance that arrives late or not at all, so the ceiling is a
correctness property of a run, not a politeness setting.
"""

import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tau2.voice.synthesis import synthesize as synth_module
from tau2.voice.synthesis.synthesize import synthesize_voice


def _pcm16_audio():
    """Minimal stand-in accepted by synthesize_voice's format check."""
    return SimpleNamespace(format=SimpleNamespace(is_pcm16=True, encoding="pcm_s16le"))


def test_concurrent_synthesis_is_bounded_by_the_semaphore():
    """The ceiling must hold across threads, not just per call site.

    Driven with more threads than the limit: OutOfTurnSpeechGenerator fans out
    into a thread pool, so a single simulation can exceed the cap on its own.
    """
    limit = synth_module._TTS_SEMAPHORE._initial_value
    live = 0
    peak = 0
    lock = threading.Lock()

    def fake_tts(text, config):
        nonlocal live, peak
        with lock:
            live += 1
            peak = max(peak, live)
        time.sleep(0.02)
        with lock:
            live -= 1
        return _pcm16_audio()

    with patch.object(synth_module, "tts_elevenlabs", fake_tts):
        threads = [
            threading.Thread(
                target=synthesize_voice,
                kwargs={
                    "text": "hello",
                    "provider": "elevenlabs",
                    "provider_config": None,
                },
            )
            for _ in range(limit * 4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert peak <= limit, f"{peak} concurrent TTS calls exceeded the limit of {limit}"
    # Guards against a semaphore so tight it serializes everything, which would
    # make a run needlessly slow rather than merely correct.
    assert peak > 1, "expected real parallelism up to the limit"


def test_unsupported_provider_raises_without_consuming_a_slot():
    """A config error must not queue behind in-flight synthesis."""
    before = synth_module._TTS_SEMAPHORE._value
    with pytest.raises(ValueError, match="Unsupported synthesis provider"):
        synthesize_voice(text="hello", provider="nope", provider_config=None)
    assert synth_module._TTS_SEMAPHORE._value == before


def test_slot_is_released_when_the_provider_raises():
    """A failing call must not leak its slot, or the allowance bleeds away.

    Without release-on-exception the limit degrades run-long: each provider
    error permanently costs one slot until synthesis stops entirely.
    """
    before = synth_module._TTS_SEMAPHORE._value

    def boom(text, config):
        raise ConnectionError("provider down")

    # tts_retry re-raises after exhausting attempts; every attempt must release.
    with patch.object(synth_module, "tts_elevenlabs", boom):
        with pytest.raises(ConnectionError):
            synthesize_voice(text="hello", provider="elevenlabs", provider_config=None)

    assert synth_module._TTS_SEMAPHORE._value == before


@pytest.mark.parametrize("bad", ["abc", "0", "-1"])
def test_invalid_env_override_is_rejected(bad, monkeypatch):
    """A typo'd override must fail loudly, not silently pick a default.

    Falling back would reintroduce the 429s the ceiling exists to prevent, and
    the run would look merely flaky.
    """
    monkeypatch.setenv("TAU2_TTS_MAX_CONCURRENCY", bad)
    with pytest.raises(ValueError, match="TAU2_TTS_MAX_CONCURRENCY"):
        synth_module._tts_concurrency_limit()


def test_env_override_is_honored(monkeypatch):
    monkeypatch.setenv("TAU2_TTS_MAX_CONCURRENCY", "2")
    assert synth_module._tts_concurrency_limit() == 2
    monkeypatch.delenv("TAU2_TTS_MAX_CONCURRENCY")
    from tau2.config import DEFAULT_TTS_MAX_CONCURRENCY

    assert synth_module._tts_concurrency_limit() == DEFAULT_TTS_MAX_CONCURRENCY
