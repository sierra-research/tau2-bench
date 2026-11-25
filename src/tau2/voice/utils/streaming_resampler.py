"""Streaming band-limited resampling and G.711 mu-law coding for PCM16 audio.

``audioop.ratecv`` resamples by linear interpolation without a low-pass
filter. Taking 16 or 24 kHz speech down to 8 kHz with it folds everything
above 4 kHz back into the 0-4 kHz telephone band, and taking 8 kHz up leaves
mirror images of the voice above 4 kHz. This module needs no ``audioop``,
which Python 3.13 removed from the standard library.

``StreamingResampler`` is a polyphase FIR resampler for any pair of integer
sample rates. Its Kaiser-windowed sinc low-pass keeps 0 to 85% of the lower
rate's Nyquist frequency flat (0 to 3.4 kHz when one side is 8 kHz) and
attenuates everything at or above that Nyquist frequency (4 kHz) by at least
60 dB. State carries across chunks: any chunking of a stream produces the same
samples as converting it in one call.

Timing. The filter is causal and linear phase. Its group delay,
``delay_seconds``, is set by the width of the transition band and is a whole
number of samples at the lower rate: 29 samples, 3.625 ms, whenever one side
is 8 kHz. Over a stream, ``n`` input samples yield exactly
``ceil(n * output_rate / input_rate)`` output samples. For integer
downsampling (16, 24 or 48 kHz to 8 kHz) every chunk's count equals
``audioop.ratecv``'s, so byte-based tick accounting is unchanged; when
upsampling, ``ratecv`` returns a few samples fewer at the start of a stream.
The content comes out ``delay_seconds`` later within those samples: a stream
starts with that much near-silence, and the newest ``delay_seconds`` of input
stay in the filter until the next chunk pushes them out. ``reset()`` discards
them.

Unlike linear interpolation, the filter can overshoot: content near full
scale (a square wave, clipped speech) may clip slightly on output.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# Filter specification, relative to the Nyquist frequency of the lower rate.
_PASSBAND_FRACTION = 0.85  # flat to 3.4 kHz when the lower rate is 8 kHz
_STOPBAND_ATTENUATION_DB = 70.0  # design target; >= 60 dB is guaranteed by tests


def _kaiser_lowpass(
    num_taps: int, cutoff_hz: float, rate_hz: float, beta: float
) -> np.ndarray:
    """Kaiser-windowed sinc low-pass with unit DC gain at ``rate_hz``."""
    center = (num_taps - 1) / 2
    t = np.arange(num_taps) - center
    taps = np.sinc(2 * cutoff_hz / rate_hz * t) * np.kaiser(num_taps, beta)
    return taps / taps.sum()


class StreamingResampler:
    """Stateful band-limited PCM16 resampler between two integer rates.

    Args:
        input_rate: Sample rate of the audio passed to ``process``.
        output_rate: Sample rate of the audio ``process`` returns.
    """

    def __init__(self, input_rate: int, output_rate: int):
        if input_rate <= 0 or output_rate <= 0:
            raise ValueError(
                f"Sample rates must be positive, got {input_rate} -> {output_rate}"
            )
        self.input_rate = int(input_rate)
        self.output_rate = int(output_rate)
        gcd = math.gcd(self.input_rate, self.output_rate)
        # Polyphase view: upsample by `up`, filter at input_rate * up, keep
        # every `down`-th sample.
        self._up = self.output_rate // gcd
        self._down = self.input_rate // gcd

        if self._up == self._down:
            self._taps_per_phase = 0
            self._delay = 0
            self._phases = np.zeros((1, 0))
        else:
            nyquist = min(self.input_rate, self.output_rate) / 2
            passband = _PASSBAND_FRACTION * nyquist
            filter_rate = self.input_rate * self._up
            width = 2 * math.pi * (nyquist - passband) / filter_rate
            attenuation = _STOPBAND_ATTENUATION_DB
            # Kaiser's design formulas for length and window shape.
            num_taps = math.ceil((attenuation - 7.95) / (2.285 * width)) + 1
            # Odd length, with the group delay a whole number of samples at
            # the lower rate (a multiple of `down`, or of `up` when upsampling).
            quantum = self._down if self._up < self._down else self._up
            half = -(-(num_taps - 1) // (2 * quantum)) * quantum
            num_taps = 2 * half + 1
            beta = 0.1102 * (attenuation - 8.7)
            taps = _kaiser_lowpass(
                num_taps, (passband + nyquist) / 2, filter_rate, beta
            )
            taps *= self._up  # zero stuffing divides the level by `up`
            self._taps_per_phase = -(-num_taps // self._up)
            padded = np.zeros(self._taps_per_phase * self._up)
            padded[:num_taps] = taps
            # _phases[p, j] = taps[p + j * up]; reversed so a row multiplies
            # input samples in ascending time order.
            self._phases = np.ascontiguousarray(
                padded.reshape(self._taps_per_phase, self._up).T[:, ::-1]
            )
            self._delay = (num_taps - 1) // 2  # samples at filter_rate
        self.reset()

    @property
    def delay_seconds(self) -> float:
        """Fixed latency the filter adds to the content of the stream."""
        return self._delay / (self.input_rate * self._up)

    def reset(self) -> None:
        """Start a new stream: forget buffered input and the stream position."""
        # _history holds input samples [_history_start, _inputs_seen), preceded
        # by zeros for the time before the stream began.
        self._history = np.zeros(self._taps_per_phase)
        self._history_start = -self._taps_per_phase
        self._inputs_seen = 0
        self._outputs_made = 0

    def process(self, pcm16: bytes) -> bytes:
        """Resample a chunk of little-endian PCM16 mono audio."""
        if len(pcm16) % 2:
            raise ValueError("PCM16 audio must contain whole 2-byte samples")
        samples = np.frombuffer(pcm16, dtype="<i2")
        if self._up == self._down:
            return samples.astype("<i2").tobytes()

        self._history = np.concatenate([self._history, samples])
        self._inputs_seen += len(samples)

        # Output n is the filtered signal at filter-rate position n * down.
        # It needs input up to index position // up, so it is ready once
        # n * down < inputs_seen * up.
        ready = -(-(self._inputs_seen * self._up) // self._down)
        n = np.arange(self._outputs_made, max(ready, self._outputs_made))
        position = n * self._down
        newest_input = position // self._up
        phase = position % self._up
        # Output n weighs the taps_per_phase inputs ending at newest_input,
        # with filter phase `phase`. Phases repeat every `up` outputs, and
        # outputs `up` apart are `down` inputs apart, so each phase's windows
        # are a strided view of the history: no copies.
        out = np.empty(len(n))
        if len(n):
            windows = sliding_window_view(self._history, self._taps_per_phase)
            first = newest_input - self._history_start - (self._taps_per_phase - 1)
        for k in range(min(self._up, len(n))):
            rows = windows[first[k] :: self._down][: len(out[k :: self._up])]
            out[k :: self._up] = np.einsum("ij,j->i", rows, self._phases[phase[k]])
        self._outputs_made += len(n)

        # Keep only the input the next output still needs.
        next_newest = (self._outputs_made * self._down) // self._up
        keep_from = next_newest - self._taps_per_phase + 1
        drop = keep_from - self._history_start
        if drop > 0:
            self._history = self._history[drop:]
            self._history_start = keep_from

        return np.clip(np.rint(out), -32768, 32767).astype("<i2").tobytes()


def _ulaw_encode_table() -> np.ndarray:
    """mu-law byte for every PCM16 value, bit-exact with audioop.lin2ulaw."""
    pcm = np.arange(-32768, 32768, dtype=np.int32) >> 2  # 14-bit, as audioop
    mask = np.where(pcm < 0, 0x7F, 0xFF)
    magnitude = np.minimum(np.abs(pcm), 8159) + 33
    segment = np.searchsorted(
        np.array([0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF]),
        magnitude,
    )
    mantissa = (magnitude >> (segment + 1)) & 0xF
    code = np.where(segment < 8, (segment << 4) | mantissa, 0x7F)  # 8: clipped
    return (code ^ mask).astype(np.uint8)


def _ulaw_decode_table() -> np.ndarray:
    """PCM16 value for every mu-law byte, bit-exact with audioop.ulaw2lin."""
    code = ~np.arange(256, dtype=np.int32) & 0xFF
    exponent = (code >> 4) & 0x07
    magnitude = ((((code & 0x0F) << 3) + 0x84) << exponent) - 0x84
    return np.where(code & 0x80, -magnitude, magnitude).astype("<i2")


_ULAW_ENCODE = _ulaw_encode_table()
_ULAW_DECODE = _ulaw_decode_table()


def pcm16_to_ulaw(pcm16: bytes) -> bytes:
    """Encode little-endian PCM16 mono audio as G.711 mu-law."""
    if len(pcm16) % 2:
        raise ValueError("PCM16 audio must contain whole 2-byte samples")
    samples = np.frombuffer(pcm16, dtype="<i2").astype(np.int32)
    return _ULAW_ENCODE[samples + 32768].tobytes()


def ulaw_to_pcm16(ulaw: bytes) -> bytes:
    """Decode G.711 mu-law audio to little-endian PCM16 mono."""
    return _ULAW_DECODE[np.frombuffer(ulaw, dtype=np.uint8)].tobytes()
