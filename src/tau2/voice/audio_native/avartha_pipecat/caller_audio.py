"""Caller audio for Avartha: 8 kHz telephony mu-law to the server's PCM16 rate.

The shared ``StreamingTelephonyConverter`` upsamples with ``audioop.ratecv``,
which is linear interpolation. At 2x it leaves a mirror image of the 0-4 kHz
voice at 4-8 kHz only 5-13 dB below it, which a phone call never carries.
This converter uses the windowed-sinc filter of ``scipy.signal.resample_poly``,
streamed with its filter state so consecutive ticks join without clicks, and
returns exactly ``factor`` samples per input sample.
"""

from __future__ import annotations

import audioop

import numpy as np
from scipy import signal

from tau2.config import DEFAULT_TELEPHONY_RATE


class CallerAudioUpsampler:
    """Streaming telephony-to-PCM16 converter for an integer rate ratio."""

    def __init__(self, output_sample_rate: int):
        if output_sample_rate % DEFAULT_TELEPHONY_RATE:
            raise ValueError(
                f"{output_sample_rate} Hz is not a multiple of {DEFAULT_TELEPHONY_RATE} Hz"
            )
        self.factor = output_sample_rate // DEFAULT_TELEPHONY_RATE
        # resample_poly's default design for up=factor, down=1.
        half_len = 10 * self.factor
        self._taps = (
            signal.firwin(2 * half_len + 1, 1 / self.factor, window=("kaiser", 5.0))
            * self.factor
        )
        self.reset()

    def reset(self) -> None:
        """Forget the previous chunk. Call on interruption or new session."""
        self._state = np.zeros(len(self._taps) - 1)

    def convert_input(self, telephony_audio: bytes) -> bytes:
        """Convert telephony (8 kHz mu-law) to PCM16 at the output rate."""
        if not telephony_audio:
            return b""
        pcm = np.frombuffer(audioop.ulaw2lin(telephony_audio, 2), dtype="<i2")
        stuffed = np.zeros(len(pcm) * self.factor)
        stuffed[:: self.factor] = pcm
        out, self._state = signal.lfilter(self._taps, 1.0, stuffed, zi=self._state)
        return np.clip(np.round(out), -32768, 32767).astype("<i2").tobytes()
