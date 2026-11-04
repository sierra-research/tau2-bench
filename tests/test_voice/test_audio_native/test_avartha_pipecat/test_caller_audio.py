import audioop

import numpy as np
import pytest

from tau2.voice.audio_native.audio_converter import StreamingTelephonyConverter
from tau2.voice.audio_native.avartha_pipecat.caller_audio import CallerAudioUpsampler


def _tone_ulaw(hz, seconds=1.0, amplitude=8000):
    t = np.arange(int(8000 * seconds)) / 8000
    pcm = (amplitude * np.sin(2 * np.pi * hz * t)).astype("<i2")
    return audioop.lin2ulaw(pcm.tobytes(), 2)


def _level_db(pcm16, hz, rate=16000):
    x = np.frombuffer(pcm16, dtype="<i2").astype(np.float64)
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1 / rate)
    return 20 * np.log10(spectrum[np.abs(freqs - hz) < 30].max())


def test_each_chunk_returns_exactly_twice_its_samples():
    upsampler = CallerAudioUpsampler(16000)
    audio = _tone_ulaw(1000)
    start = 0
    for size in (1, 160, 1600, 7, 333, 1600):
        chunk = audio[start : start + size]
        assert len(upsampler.convert_input(chunk)) == 2 * 2 * len(chunk)
        start += size


def test_chunked_conversion_matches_one_pass():
    audio = _tone_ulaw(1200)
    whole = CallerAudioUpsampler(16000).convert_input(audio)
    upsampler = CallerAudioUpsampler(16000)
    pieces = b"".join(
        upsampler.convert_input(audio[i : i + 160]) for i in range(0, len(audio), 160)
    )
    assert pieces == whole


def test_no_mirror_image_above_4khz():
    # A 3 kHz tone must not gain a 5 kHz twin (its reflection around 4 kHz).
    audio = _tone_ulaw(3000)
    fixed = CallerAudioUpsampler(16000).convert_input(audio)
    assert _level_db(fixed, 3000) - _level_db(fixed, 5000) > 35
    # The shared linear-interpolation converter keeps the twin within 15 dB.
    linear = StreamingTelephonyConverter(input_sample_rate=16000).convert_input(audio)
    assert _level_db(linear, 3000) - _level_db(linear, 5000) < 15


def test_the_voice_band_keeps_its_level():
    audio = _tone_ulaw(1000)
    fixed = CallerAudioUpsampler(16000).convert_input(audio)
    pcm8 = audioop.ulaw2lin(audio, 2)
    assert (
        abs(
            _level_db(fixed, 1000)
            - (_level_db(pcm8, 1000, rate=8000) + 20 * np.log10(2))
        )
        < 1
    )


def test_reset_starts_a_fresh_stream():
    audio = _tone_ulaw(700, seconds=0.2)
    upsampler = CallerAudioUpsampler(16000)
    first = upsampler.convert_input(audio)
    upsampler.convert_input(_tone_ulaw(2500, seconds=0.2))
    upsampler.reset()
    assert upsampler.convert_input(audio) == first


def test_rate_must_be_a_multiple_of_telephony():
    with pytest.raises(ValueError):
        CallerAudioUpsampler(22050)
