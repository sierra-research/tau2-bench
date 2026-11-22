"""Transcript pacing: the caller never sees agent words before they are spoken.

Hydra sends the caption of each clause just before that clause's audio. The
default reveal (``UtteranceTranscript``) shows text in proportion to the
utterance's audio played / audio received, so a caption that arrives with only
its first 200 ms of audio is shown whole on the first tick, and the simulated
caller reads, and can interrupt on, words the agent has not said yet.

``PacedUtteranceTranscript`` adds a speaking-rate cap: once ``s`` seconds of an
utterance's audio have played, at most ``floor(rate * s)`` of its characters
are shown (default 15 characters per second, about 2.7 words per second). The
proportional rule stays as a second limit. When Hydra reports the reply
``completed`` and every received byte of it has played, the rest is shown, so
a voice faster than the cap does not strand the end of each sentence. A reply
cut short (barge-in, or ``cancelled`` / ``incomplete`` / ``failed``) never
shows its unspoken rest.

Example, 200 ms ticks of 8 kHz u-law (8000 bytes per second): the caption
"I can change that flight for you." (33 characters) arrives with the first
1600 bytes of its 2.2 s of audio, and the rest of the audio follows in real
time. Proportional reveal alone shows all 33 characters on tick 1 (all 1600
bytes received have played). With the cap, tick 1 shows floor(15 * 0.2) = 3
characters ("I c") and tick 2 three more ("an "). If the caller barges in after
1 s, the caller has seen 15 characters ("I can change th") and never sees the
rest.

A rate of 0 disables the cap and restores the default reveal byte for byte.
"""

import math
import os
from typing import Mapping, Optional

from loguru import logger
from pydantic import Field

from tau2.config import DEFAULT_SMALLEST_TRANSCRIPT_REVEAL_CPS, DEFAULT_TELEPHONY_RATE
from tau2.voice.audio_native.tick_result import UtteranceTranscript
from tau2.voice.utils.transcript_utils import get_proportional_text

TRANSCRIPT_REVEAL_CPS_ENV = "SMALLEST_TRANSCRIPT_REVEAL_CPS"

_logged_rates: set[float] = set()


def resolve_reveal_cps(
    value: Optional[float] = None, env: Optional[Mapping[str, str]] = None
) -> float:
    """The reveal rate in characters per second of audio played.

    An explicit ``value`` wins; otherwise ``SMALLEST_TRANSCRIPT_REVEAL_CPS``;
    unset or empty means the default (15). Anything that is not a finite number
    >= 0 raises: a typo must not silently change how a run is measured.
    """
    if value is not None:
        source = "transcript_reveal_cps"
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{source} must be a number, got {value!r}")
        parsed = float(value)
    else:
        source = TRANSCRIPT_REVEAL_CPS_ENV
        raw = (os.environ if env is None else env).get(TRANSCRIPT_REVEAL_CPS_ENV)
        if raw is None or not raw.strip():
            return DEFAULT_SMALLEST_TRANSCRIPT_REVEAL_CPS
        try:
            parsed = float(raw)
        except ValueError as e:
            raise ValueError(
                f"{source}={raw!r} is not a number "
                "(characters per second; 0 disables the cap)"
            ) from e
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{source} must be a finite number >= 0, got {parsed!r}")
    return parsed


def run_settings() -> dict:
    """The adapter settings a saved run must record (its ``agent_info.llm_args``):
    the effective reveal cap, which changes what the caller sees but is set by an
    environment variable, so two runs would otherwise save identical configs."""
    return {"transcript_reveal_cps": resolve_reveal_cps()}


def log_reveal_mode(cps: float) -> None:
    """Log the reveal mode once per process and rate; non-default warns."""
    if cps in _logged_rates:
        return
    _logged_rates.add(cps)
    if cps == DEFAULT_SMALLEST_TRANSCRIPT_REVEAL_CPS:
        logger.info(
            f"Smallest transcript reveal: at most {cps:g} characters per second "
            f"of agent audio played ({TRANSCRIPT_REVEAL_CPS_ENV})"
        )
    elif cps > 0:
        logger.warning(
            f"Smallest transcript reveal: NON-DEFAULT cap of {cps:g} characters "
            f"per second ({TRANSCRIPT_REVEAL_CPS_ENV}; default "
            f"{DEFAULT_SMALLEST_TRANSCRIPT_REVEAL_CPS:g})"
        )
    else:
        logger.warning(
            f"Smallest transcript reveal: cap OFF ({TRANSCRIPT_REVEAL_CPS_ENV}=0); "
            "the caller may see agent text before it is spoken"
        )


def rate_cap_chars(
    reveal_chars_per_s: float, audio_bytes_played: int, bytes_per_second: int
) -> Optional[int]:
    """Most characters speakable in ``audio_bytes_played`` (None: cap off)."""
    if reveal_chars_per_s <= 0:
        return None
    if bytes_per_second <= 0:
        raise ValueError(f"bytes_per_second must be > 0, got {bytes_per_second}")
    # Multiply before dividing so tick-aligned values stay exact
    # (15 * 1600 / 8000 == 3.0); the epsilon guards floor() against float noise.
    return int(
        math.floor(reveal_chars_per_s * audio_bytes_played / bytes_per_second + 1e-9)
    )


class PacedUtteranceTranscript(UtteranceTranscript):
    """``UtteranceTranscript`` whose text is shown no faster than it is spoken.

    Audio bytes are counted in the adapter's external format, so
    ``bytes_per_second`` must be that format's rate. The adapter sets
    ``completed`` or ``truncated`` from how Hydra ended the reply.
    """

    reveal_chars_per_s: float = DEFAULT_SMALLEST_TRANSCRIPT_REVEAL_CPS
    bytes_per_second: int = DEFAULT_TELEPHONY_RATE
    completed: bool = Field(
        default=False, description="Hydra reported the reply completed."
    )
    truncated: bool = Field(
        default=False,
        description="The reply was cut short; its unspoken text is never shown.",
    )

    @property
    def audio_fully_played(self) -> bool:
        return 0 < self.audio_bytes_received <= self.audio_bytes_played

    @property
    def has_spoken_tail(self) -> bool:
        """Text the cap still holds back although all of it has been spoken."""
        return (
            self.reveal_chars_per_s > 0
            and self.completed
            and not self.truncated
            and self.audio_fully_played
            and self.transcript_chars_shown < len(self.transcript_received)
        )

    def get_transcript_for_audio(self, audio_bytes_to_play: int) -> str:
        """Text revealed by playing ``audio_bytes_to_play`` more bytes (may be 0)."""
        self.audio_bytes_played += audio_bytes_to_play
        if self.audio_bytes_received == 0 or not self.transcript_received:
            return ""
        start = self.transcript_chars_shown
        _, end = get_proportional_text(
            transcript=self.transcript_received,
            total_duration=self.audio_bytes_received,
            audio_played=self.audio_bytes_played,
            start_char=start,
        )
        cap = rate_cap_chars(
            self.reveal_chars_per_s, self.audio_bytes_played, self.bytes_per_second
        )
        if cap is not None:
            end = max(start, min(end, cap))
        if self.has_spoken_tail:
            end = len(self.transcript_received)
        self.transcript_chars_shown = end
        return self.transcript_received[start:end]
