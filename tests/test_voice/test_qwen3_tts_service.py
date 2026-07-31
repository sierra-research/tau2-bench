import numpy as np

from services.qwen3_tts.app import _normalize_text, _to_pcm16_mono


def test_to_pcm16_mono_resamples_to_16khz():
    one_second_at_24khz = np.zeros(24_000, dtype=np.float32)

    pcm = _to_pcm16_mono(one_second_at_24khz, source_rate=24_000)

    assert len(pcm) == 16_000 * 2
    assert np.frombuffer(pcm, dtype="<i2").shape == (16_000,)


def test_normalize_text_removes_unsupported_elevenlabs_tags():
    assert _normalize_text("Hello [pause] [cough]") == "Hello ..."
