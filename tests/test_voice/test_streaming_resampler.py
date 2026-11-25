"""Tests for the band-limited streaming resampler and the mu-law codec."""

import math
import warnings

import numpy as np
import pytest
from scipy import signal

from tau2.voice.audio_native.audio_converter import StreamingTelephonyConverter
from tau2.voice.utils.streaming_resampler import (
    StreamingResampler,
    pcm16_to_ulaw,
    ulaw_to_pcm16,
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    try:
        import audioop
    except ImportError:  # Python 3.13+
        audioop = None

PROVIDER_RATES = [16000, 22050, 24000, 44100, 48000]


def _tone(hz, rate, seconds=1.0, amplitude=10000.0):
    t = np.arange(int(rate * seconds)) / rate
    return (amplitude * np.sin(2 * np.pi * hz * t)).astype("<i2")


def _resample(samples, input_rate, output_rate):
    out = StreamingResampler(input_rate, output_rate).process(samples.tobytes())
    return np.frombuffer(out, dtype="<i2")


def _peak_db(samples):
    """Level of the strongest component, ignoring the start-up transient."""
    x = samples[len(samples) // 4 :].astype(np.float64)
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return 20 * np.log10(spectrum.max() + 1e-12)


def _rms(samples):
    x = samples[len(samples) // 4 :].astype(np.float64)
    return math.sqrt(np.mean(x**2))


@pytest.mark.parametrize("rate", PROVIDER_RATES)
def test_downsampling_rejects_aliases_by_60_db(rate):
    reference = _peak_db(_resample(_tone(1000, rate), rate, 8000))
    for hz in np.arange(4000, rate / 2, 100):
        out = _resample(_tone(hz, rate, seconds=0.5), rate, 8000)
        assert _peak_db(out) - reference <= -60, f"{hz} Hz"


@pytest.mark.parametrize("rate", [16000, 24000, 48000])
def test_upsampling_rejects_images_by_60_db(rate):
    for hz in np.arange(100, 3401, 100):
        out = _resample(_tone(hz, 8000), 8000, rate)
        x = out[len(out) // 4 :].astype(np.float64)
        spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
        freqs = np.fft.rfftfreq(len(x), 1 / rate)
        tone = spectrum[np.abs(freqs - hz) < 20].max()
        images = spectrum[freqs >= 4000].max()
        assert 20 * np.log10(images / tone) <= -60, f"{hz} Hz"


@pytest.mark.parametrize(
    "input_rate,output_rate",
    [(r, 8000) for r in PROVIDER_RATES] + [(8000, 16000), (8000, 24000)],
)
def test_passband_is_flat_to_3400_hz(input_rate, output_rate):
    amplitude = 10000.0
    for hz in np.arange(100, 3401, 100):
        out = _resample(
            _tone(hz, input_rate, amplitude=amplitude), input_rate, output_rate
        )
        gain_db = 20 * np.log10(_rms(out) / (amplitude / math.sqrt(2)))
        assert abs(gain_db) < 0.1, f"{hz} Hz: {gain_db:.3f} dB"


@pytest.mark.parametrize(
    "input_rate,output_rate",
    [(24000, 8000), (16000, 8000), (22050, 8000), (8000, 16000), (8000, 24000)],
)
def test_streaming_matches_one_shot_bit_exactly(input_rate, output_rate):
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(input_rate * 2) * 6000).astype("<i2")
    whole = StreamingResampler(input_rate, output_rate).process(audio.tobytes())
    resampler = StreamingResampler(input_rate, output_rate)
    pieces, start = [], 0
    while start < len(audio):
        size = int(rng.choice([0, 1, 2, 7, 160, 333, 1024, 4800]))
        pieces.append(resampler.process(audio[start : start + size].tobytes()))
        start += size
    assert b"".join(pieces) == whole


@pytest.mark.parametrize("input_rate", PROVIDER_RATES)
def test_sample_counts_match_the_rate_ratio(input_rate):
    # Byte-based tick accounting relies on exact counts, as with audioop.ratecv.
    resampler = StreamingResampler(input_rate, 8000)
    seen = made = 0
    for size in (1, 512, 4800, 3, 4410, 960):
        made += len(resampler.process(bytes(2 * size))) // 2
        seen += size
        assert made == -(-seen * 8000 // input_rate)
    one_tick = StreamingResampler(input_rate, 8000).process(bytes(input_rate // 5 * 2))
    assert len(one_tick) == 1600 * 2


@pytest.mark.parametrize(
    "input_rate,output_rate",
    [(r, 8000) for r in PROVIDER_RATES] + [(8000, 16000), (8000, 24000)],
)
def test_group_delay_is_fixed_and_small(input_rate, output_rate):
    resampler = StreamingResampler(input_rate, output_rate)
    assert resampler.delay_seconds == pytest.approx(0.003625)  # 29 samples at 8 kHz
    # A click comes out delay_seconds later.
    click = np.zeros(input_rate // 10, dtype="<i2")
    click[input_rate // 50] = 20000
    out = np.frombuffer(resampler.process(click.tobytes()), dtype="<i2")
    expected = (1 / 50 + resampler.delay_seconds) * output_rate
    assert int(np.argmax(np.abs(out))) == round(expected)


@pytest.mark.parametrize(
    "input_rate,output_rate",
    [(r, 8000) for r in PROVIDER_RATES] + [(8000, r) for r in PROVIDER_RATES],
)
def test_matches_a_one_shot_polyphase_reference(input_rate, output_rate):
    resampler = StreamingResampler(input_rate, output_rate)
    up, down = resampler._up, resampler._down
    # _phases[p, j] holds taps[p + (L - 1 - j) * up]; rebuild the prototype.
    taps = resampler._phases[:, ::-1].T.reshape(-1)
    rng = np.random.default_rng(1)
    audio = (rng.standard_normal(input_rate // 2) * 6000).astype("<i2")
    out = np.frombuffer(resampler.process(audio.tobytes()), dtype="<i2")
    expected = signal.upfirdn(taps, audio.astype(np.float64), up, down)[: len(out)]
    expected = np.clip(np.rint(expected), -32768, 32767)
    assert np.max(np.abs(out - expected)) <= 1


def test_reset_starts_a_fresh_stream():
    first = _tone(700, 24000, seconds=0.2).tobytes()
    resampler = StreamingResampler(24000, 8000)
    expected = resampler.process(first)
    resampler.process(_tone(3000, 24000, seconds=0.13).tobytes())
    resampler.reset()
    assert resampler.process(first) == expected


def test_equal_rates_pass_through():
    audio = _tone(440, 8000).tobytes()
    resampler = StreamingResampler(8000, 8000)
    assert resampler.process(audio) == audio
    assert resampler.delay_seconds == 0


def test_odd_byte_count_is_rejected():
    with pytest.raises(ValueError):
        StreamingResampler(24000, 8000).process(b"\x00\x00\x00")
    with pytest.raises(ValueError):
        pcm16_to_ulaw(b"\x00")


def test_ulaw_round_trip_is_within_one_quantization_step():
    pcm = np.arange(-32768, 32768, dtype="<i2")
    decoded = np.frombuffer(ulaw_to_pcm16(pcm16_to_ulaw(pcm.tobytes())), "<i2")
    error = np.abs(decoded.astype(np.int32) - pcm)
    # mu-law steps double per segment: 8 near zero, 1024 in the top segment.
    assert np.all(error <= np.maximum(16, np.abs(pcm.astype(np.int32)) // 16))
    # Every code except negative zero (0x7F) survives decode then encode.
    codes = bytes(code for code in range(256) if code != 0x7F)
    assert pcm16_to_ulaw(ulaw_to_pcm16(codes)) == codes


@pytest.mark.skipif(audioop is None, reason="audioop removed in Python 3.13")
def test_ulaw_codec_is_bit_exact_with_audioop():
    pcm = np.arange(-32768, 32768, dtype="<i2").tobytes()
    assert pcm16_to_ulaw(pcm) == audioop.lin2ulaw(pcm, 2)
    codes = bytes(range(256))
    assert ulaw_to_pcm16(codes) == audioop.ulaw2lin(codes, 2)


@pytest.mark.skipif(audioop is None, reason="audioop removed in Python 3.13")
@pytest.mark.parametrize("rate", [16000, 24000])
def test_converter_output_drops_the_alias_that_ratecv_kept(rate):
    # 5 kHz at a 24 kHz provider aliases to 3 kHz at 8 kHz with linear
    # interpolation; band-limited conversion removes it (mu-law noise aside).
    tone = _tone(5000, rate).tobytes()
    reference = _tone(1000, rate).tobytes()

    def level(converter_output):
        return _peak_db(np.frombuffer(ulaw_to_pcm16(converter_output), "<i2"))

    new = StreamingTelephonyConverter(output_sample_rate=rate)
    new_alias = level(new.convert_output(tone))
    new.reset()
    new_ref = level(new.convert_output(reference))
    old_alias = level(
        audioop.lin2ulaw(audioop.ratecv(tone, 2, 1, rate, 8000, None)[0], 2)
    )
    assert old_alias - new_ref > -10  # the old path folds it in almost intact
    assert new_alias - new_ref < -45
