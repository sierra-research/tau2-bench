"""Bookkeeping for Hydra replies: one paced utterance transcript per reply.

Hydra tags every audio and caption delta with the ``response_id`` of its
reply, and one reply is one spoken utterance (a reply that calls a tool keeps
its id across the tool round trip). The ledger maps replies to
``PacedUtteranceTranscript`` entries in the adapter's shared transcript dict,
records how each reply ended, and lists the utterances whose spoken tail must
be released on a tick that plays none of their audio.
"""

from typing import Optional

from tau2.voice.audio_native.smallest.pacing import PacedUtteranceTranscript
from tau2.voice.audio_native.tick_result import UtteranceTranscript

COMPLETED = "completed"


class HydraResponses:
    """Which reply each event belongs to, and how each reply ended."""

    def __init__(
        self,
        transcripts: dict[str, UtteranceTranscript],
        reveal_chars_per_s: float,
        bytes_per_second: int,
    ):
        self._transcripts = transcripts
        self._cps = reveal_chars_per_s
        self._bps = bytes_per_second
        self.reset()

    def reset(self) -> None:
        # Replies created and not yet done, oldest first.
        self._in_flight: dict[str, None] = {}
        # The reply whose audio is still arriving: response.created until
        # response.output_audio.done. A barge-in skips its later audio.
        self.streaming: Optional[str] = None
        self._unnamed = 0

    def key(self, response_id: Optional[str]) -> str:
        """Utterance key of an event: its reply id, else the newest open reply."""
        if response_id:
            return response_id
        if self._in_flight:
            return next(reversed(self._in_flight))
        return self.open(None)

    def open(self, response_id: Optional[str]) -> str:
        """A reply started (``response.created``)."""
        if not response_id:
            self._unnamed += 1
            response_id = f"smallest_response_{self._unnamed}"
        self._in_flight[response_id] = None
        self.streaming = response_id
        self.transcript(response_id)
        return response_id

    def transcript(self, key: str) -> PacedUtteranceTranscript:
        tracker = self._transcripts.get(key)
        if not isinstance(tracker, PacedUtteranceTranscript):
            tracker = PacedUtteranceTranscript(
                item_id=key,
                reveal_chars_per_s=self._cps,
                bytes_per_second=self._bps,
            )
            self._transcripts[key] = tracker
        return tracker

    def audio_done(self, response_id: Optional[str]) -> None:
        if response_id is None or response_id == self.streaming:
            self.streaming = None

    def close(self, response_id: Optional[str], status: Optional[str]) -> Optional[str]:
        """``response.done``: completed releases the spoken tail, else never.

        A reply already cut by a barge-in stays cut. Without a status the
        utterance is left open (proportional and capped reveal still apply).
        Returns the reply's key, or None when there is no reply to close.
        """
        key = response_id or next(reversed(self._in_flight), None)
        if key is None:
            return None
        self._in_flight.pop(key, None)
        tracker = self._transcripts.get(key)
        if isinstance(tracker, PacedUtteranceTranscript) and status is not None:
            if tracker.truncated:
                pass
            elif status == COMPLETED:
                tracker.completed = True
            else:
                tracker.truncated = True
        return key

    def cut(self) -> Optional[str]:
        """Barge-in: replies still in flight never show their unspoken text.

        Returns the reply whose later audio must be skipped (None when every
        reply's audio has already arrived).
        """
        for key in self._in_flight:
            self.transcript(key).truncated = True
        return self.streaming

    def flush_chunks(self, keys_with_audio: set[str]) -> list[tuple[bytes, str]]:
        """Zero-length audio chunks for utterances with a spoken tail to show.

        ``get_proportional_transcript`` only asks utterances that have audio in
        the tick for text. A reply whose last audio played before Hydra said
        ``completed`` would keep its held-back tail forever; a zero-length
        chunk carrying its id makes the tick release that tail (and labels
        the tick with the utterance it belongs to).
        """
        return [
            (b"", key)
            for key, tracker in self._transcripts.items()
            if key not in keys_with_audio
            and isinstance(tracker, PacedUtteranceTranscript)
            and tracker.has_spoken_tail
        ]
