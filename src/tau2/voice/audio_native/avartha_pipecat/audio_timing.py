"""Avartha audio timing over WebRTC (``avartha-audio-timing/1``).

Avartha places each spoken word's end on its source-media clock and binds that
clock to the Opus RTP clock of the audio track (Avartha Platform
``docs/pipecat-agent-server.md``, "Audio timing"):

- ``server-message`` ``audio_timeline``: ``stream_id`` (a UUID naming one clock
  epoch), ``rtp_timestamp`` of an encoded packet and the source-media sample
  ``media_sample_offset`` its first decoded sample carries. Both clocks run at
  48 kHz.
- ``bot-tts-text`` ``data.audio_time``: the ``stream_id`` and the source-media
  ``sample_offset`` where the word ends.

A word ends at RTP timestamp
``(rtp_timestamp + sample_offset - media_sample_offset) mod 2**32``. Platform
keeps RTP timestamps across its media bridge, so the receiver can compare that
with the timestamps of the frames it decoded. This module parses the two
messages and does the clock arithmetic; the receiver keeps its frames'
timestamps (``webrtc.py``) and the adapter compares them with what it played.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Optional

AUDIO_TIMING_PROTOCOL = "avartha-audio-timing/1"
# Both the source-media clock and the Opus RTP clock run at 48 kHz.
RTP_CLOCK_HZ = 48_000
_RTP_MODULUS = 1 << 32
_HALF_RTP_MODULUS = 1 << 31
# JSON clients keep integers exactly only up to 2**53 - 1.
_MAX_JSON_INTEGER = (1 << 53) - 1


def _json_integer(value: Any) -> Optional[int]:
    """``value`` if it is an integer a JSON client keeps exactly, else None."""

    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if abs(value) <= _MAX_JSON_INTEGER else None


def _is_stream_id(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        uuid.UUID(value)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class AudioClockBinding:
    """One clock epoch: source-media ``media_sample_offset`` is RTP ``rtp_timestamp``."""

    stream_id: str
    rtp_timestamp: int
    media_sample_offset: int

    def rtp_timestamp_of(self, sample_offset: int) -> int:
        """The RTP timestamp of source-media sample ``sample_offset``."""

        return (
            self.rtp_timestamp + sample_offset - self.media_sample_offset
        ) % _RTP_MODULUS


@dataclass(frozen=True)
class WordEndTime:
    """Where a word's sound ends, on the source-media clock of ``stream_id``."""

    stream_id: str
    sample_offset: int


def parse_audio_timeline(data: dict[str, Any]) -> Optional[AudioClockBinding]:
    """The binding in an ``audio_timeline`` server-message, or None if invalid."""

    rtp_timestamp = _json_integer(data.get("rtp_timestamp"))
    media_sample_offset = _json_integer(data.get("media_sample_offset"))
    if (
        data.get("protocol") != AUDIO_TIMING_PROTOCOL
        or data.get("sample_rate_hz") != RTP_CLOCK_HZ
        or not _is_stream_id(data.get("stream_id"))
        or rtp_timestamp is None
        or not 0 <= rtp_timestamp < _RTP_MODULUS
        or media_sample_offset is None
    ):
        return None
    return AudioClockBinding(data["stream_id"], rtp_timestamp, media_sample_offset)


def parse_word_end(data: dict[str, Any]) -> Optional[WordEndTime]:
    """The word end in a ``bot-tts-text`` payload, or None if absent or invalid."""

    audio_time = data.get("audio_time")
    if not isinstance(audio_time, dict):
        return None
    sample_offset = _json_integer(audio_time.get("sample_offset"))
    if (
        audio_time.get("boundary") != "word_end"
        or audio_time.get("sample_rate_hz") != RTP_CLOCK_HZ
        or not _is_stream_id(audio_time.get("stream_id"))
        or sample_offset is None
    ):
        return None
    return WordEndTime(audio_time["stream_id"], sample_offset)


def unwrap_rtp(timestamp: int, near: int) -> int:
    """The position of 32-bit RTP ``timestamp`` closest to unwrapped ``near``."""

    delta = (timestamp - near + _HALF_RTP_MODULUS) % _RTP_MODULUS - _HALF_RTP_MODULUS
    return near + delta
