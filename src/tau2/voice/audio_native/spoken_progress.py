"""Release an agent's spoken text by the audio played, and keep its spoken reference.

A voice agent sends its text ahead of the audio that speaks it, then reports
progress through it (spoken words, played captions). ``SpokenProgress`` holds
that text and releases it to the caller only once the audio carrying the
evidence has been played. Adapters subclass it with their provider's matching
of progress events against the text; this class owns the positions.

A progress mark is positioned where the evidence says its text ends in the
received audio, in bytes of the adapter's output format. Without more, that is
the audio received when the event arrived. A provider may instead place a word
end in its own output-audio stream (``source_offset``: sample frames from the
first sample of the connection's audio, which the adapter also counts as it
receives audio, see ``add_audio``). Such a mark lands on the audio that really
carries the word, however late or early its event arrives.

At a cut the caller has heard only the audio played so far. Text the caller has
not been shown is dropped from the conversation, but its words may still have
played: their events can arrive after the cut. When the provider places word
ends, the cut keeps the interrupted text and the stream position of the played
cutoff, and a word that ended before the cutoff, reported then or later, extends
a *reference amendment*: text the cut utterance's audio spoke beyond what the
caller was shown. Amendments never reach the caller; judges that compare an
utterance's audio with its text append them (see
``tau2.utils.transcript.views.agent_reference_amendments``).
"""

from __future__ import annotations

import copy
import math
from bisect import bisect_right
from typing import Optional


def placed_word_end(data: dict) -> Optional[tuple[int, int]]:
    """A progress message's ``(source offset, sample rate)`` word end, if it has one.

    The optional fields a provider may add to a spoken-word message:
    ``pcm_stream_offset`` (sample frames from the first sample of the
    connection's output audio to the end of the word), ``pcm_sample_rate_hz``
    and ``pcm_boundary``. Only ``word_end`` places a word; anything else, or a
    malformed value, leaves the word unplaced.
    """

    offset = data.get("pcm_stream_offset")
    rate = data.get("pcm_sample_rate_hz")
    if (
        data.get("pcm_boundary") != "word_end"
        or type(offset) is not int
        or type(rate) is not int
        or offset < 0
        or rate <= 0
    ):
        return None
    return offset, rate


class SourceStream:
    """Counts a connection's received output audio in the provider's own samples.

    A provider that places word ends counts the output-audio samples it sent on
    the connection, from the first; so does this, counting every sample
    received, including audio the adapter then discards. A change of sample
    rate ends the count: positions in one rate cannot be compared with audio in
    another. Received audio is converted to the adapter's format in streaming
    chunks, so a position is exact to within that converter's buffering (a few
    milliseconds).
    """

    def __init__(self) -> None:
        self._samples: Optional[int] = 0
        self.sample_rate: Optional[int] = None

    def take(self, num_samples: int, sample_rate: int) -> Optional[tuple[int, int]]:
        """The source span of the next ``num_samples`` received, if still counted."""

        if self._samples is None:
            return None
        if self.sample_rate is None:
            self.sample_rate = sample_rate
        if sample_rate != self.sample_rate:
            self._samples = None
            return None
        start = self._samples
        self._samples += num_samples
        return start, self._samples


class SpokenProgress:
    """Text shown by audio played, with optional source-stream word-end positions.

    Implements the ``get_transcript_for_audio`` protocol that
    ``get_proportional_transcript`` calls on the audio played each tick.
    Subclasses implement ``_match_word`` (advance past one spoken word, calling
    ``_advance``) and extend ``_truncate`` and ``_restart_matching`` with their
    own matching state.
    """

    def __init__(self, due_slack_bytes: int = 0) -> None:
        # Audio the adapter may be holding back at an utterance's end; a mark
        # this close to playback counts as played.
        self._due_slack = due_slack_bytes
        self.text = ""
        self._audio_received = 0
        self._audio_played = 0
        self._match_cursor = 0
        self._spoken_chars = 0
        self._shown_chars = 0
        # (received-audio position, text length aired once playback reaches it,
        # placed by the provider), ordered by position with growing lengths.
        self._marks: list[tuple[int, int, bool]] = []
        # Received audio with its source-stream span:
        # (source start, source end, received start, received end).
        self._spans: list[tuple[int, int, int, int]] = []
        # Placed words whose audio has not arrived yet: (source end, text length).
        self._unplaced: list[tuple[int, int]] = []
        # Placed words reported after a cut but before any new audio: whether
        # they belong to the next reply or to discarded audio is known only
        # once that reply's first audio shows where it starts.
        self._early_words: list[tuple[str, int]] = []
        # Placed word ends are in use once any audio carried a source span.
        self._placing = False
        # Source position from which audio is still playable: everything before
        # it was played or was cut unplayed. None after a cut until new audio.
        self._playable_from: Optional[int] = 0
        self._cut: Optional[_CutText] = None
        self._amendments: list[dict[str, str]] = []
        # Set while matching a placed word (see add_word).
        self._placed_at: Optional[int] = None

    # --- audio -----------------------------------------------------------

    def add_audio(
        self, num_bytes: int, source_span: Optional[tuple[int, int]] = None
    ) -> None:
        """Count received audio, with its span in the provider's stream if known."""

        start = self._audio_received
        self._audio_received += num_bytes
        if source_span is None:
            return
        self._placing = True
        source_start, source_end = source_span
        self._spans.append((source_start, source_end, start, self._audio_received))
        if self._playable_from is None:
            self._playable_from = source_start
            early, self._early_words = self._early_words, []
            for word, offset in early:
                self.add_word(word, offset)
        unplaced, self._unplaced = self._unplaced, []
        for offset, length in unplaced:
            position = self._received_position(offset)
            if position is None:
                self._unplaced.append((offset, length))
            else:
                self._insert_mark(position, length, placed=True)

    def _received_position(self, offset: int) -> Optional[int]:
        """Received-audio position of a source word end; None until it arrives."""

        index = bisect_right(self._spans, (offset, math.inf)) - 1
        for source_start, source_end, start, end in self._spans[max(index, 0) :]:
            if offset <= source_end:
                if offset <= source_start:
                    return start
                return start + math.ceil(
                    (offset - source_start)
                    * (end - start)
                    / (source_end - source_start)
                )
        return None

    def _source_position(self, position: int) -> Optional[int]:
        """Source position of received audio ``position``, rounded down.

        At a span boundary, the earlier span's end: a gap between spans is
        audio never received, so it never played.
        """

        source = None
        for source_start, source_end, start, end in self._spans:
            if position <= start:
                return source_start if source is None else source
            if position >= end:
                source = source_end
                continue
            return source_start + (position - start) * (source_end - source_start) // (
                end - start
            )
        return source

    # --- progress --------------------------------------------------------

    def add_word(self, word: str, source_offset: Optional[int] = None) -> None:
        """A spoken word, at its source-stream end when the provider placed it.

        A placed word that ended in the last cut's played audio belongs to the
        cut text; one inside audio a cut discarded never aired; one before the
        last cut's audio began belongs to an earlier cut and is not kept.
        """

        if source_offset is None or not self._placing:
            self._match_word(word)
            return
        if self._cut is not None and source_offset <= self._cut.source_cutoff:
            if source_offset >= self._cut.source_floor and (
                amendment := self._cut.add_word(word)
            ):
                self._amendments.append(amendment)
            return
        if self._playable_from is None:
            self._early_words.append((word, source_offset))
            return
        if source_offset < self._playable_from:
            return
        self._placed_at = source_offset
        try:
            self._match_word(word)
        finally:
            self._placed_at = None

    def _match_word(self, word: str) -> None:
        raise NotImplementedError

    def _advance(self, cursor: int) -> None:
        if cursor <= self._match_cursor:
            return
        self._match_cursor = cursor
        if self._placed_at is None:
            self._insert_mark(self._audio_received, cursor, placed=False)
            return
        position = self._received_position(self._placed_at)
        if position is None:
            self._unplaced.append((self._placed_at, cursor))
        else:
            self._insert_mark(position, cursor, placed=True)

    def _insert_mark(self, position: int, length: int, *, placed: bool) -> None:
        """Keep marks ordered by position; a later mark showing less is moot."""

        index = bisect_right([mark[0] for mark in self._marks], position)
        if index and self._marks[index - 1][1] >= length:
            return
        end = index
        while end < len(self._marks) and self._marks[end][1] <= length:
            end += 1
        self._marks[index:end] = [(position, length, placed)]

    def _snapshot(self) -> SpokenProgress:
        """An independent copy of this text and its matching state."""

        cut, self._cut = self._cut, None
        try:
            return copy.deepcopy(self)
        finally:
            self._cut = cut

    def take_reference_amendments(self) -> list[dict[str, str]]:
        """Amendments made since the last call, oldest first."""

        amendments, self._amendments = self._amendments, []
        return amendments

    # --- cut and playback ------------------------------------------------

    def interrupt(self, utterance_id: Optional[str] = None) -> None:
        """Forget everything the caller has not heard yet.

        With placed word ends, the cut text is kept for ``utterance_id`` (the
        utterance whose audio played last) up to the played cutoff.
        """

        source_cutoff = (
            self._source_position(self._audio_played) if self._placing else None
        )
        if source_cutoff is not None and utterance_id is not None and self._spans:
            heard = max(
                [self._shown_chars]
                + [
                    length
                    for position, length, placed in self._marks
                    if placed and position <= self._audio_played
                ]
            )
            cut = _CutText(self, utterance_id, self._spans[0][0], source_cutoff, heard)
            self._cut = cut
            if (amendment := cut.amendment()) is not None:
                self._amendments.append(amendment)
        self._truncate(self._shown_chars)
        self._audio_received = self._audio_played
        self._marks.clear()
        self._spans.clear()
        self._unplaced.clear()
        self._early_words.clear()
        self._playable_from = None
        self._restart_matching(self._shown_chars)
        self._spoken_chars = self._shown_chars

    def _truncate(self, length: int) -> None:
        """Drop the text past ``length``."""

        self.text = self.text[:length]

    def _restart_matching(self, cursor: int) -> None:
        """Continue matching from ``cursor`` of the text."""

        self._match_cursor = cursor

    def get_transcript_for_audio(self, audio_bytes_to_play: int) -> str:
        self._audio_played += audio_bytes_to_play
        due = self._audio_played + self._due_slack
        while self._marks and self._marks[0][0] <= due:
            self._spoken_chars = max(self._spoken_chars, self._marks.pop(0)[1])
        end = min(self._spoken_chars, len(self.text))
        text = self.text[self._shown_chars : end]
        self._shown_chars = max(self._shown_chars, end)
        return text


class _CutText:
    """The text of a cut, matched against placed words that ended before its cutoff."""

    def __init__(
        self,
        progress: SpokenProgress,
        utterance_id: str,
        source_floor: int,
        source_cutoff: int,
        heard: int,
    ) -> None:
        self.utterance_id = utterance_id
        # The cut's audio: from the first sample received since the previous
        # cut to the played cutoff.
        self.source_floor = source_floor
        self.source_cutoff = source_cutoff
        self._shown = progress._shown_chars
        self._heard = heard
        self._reported = progress._shown_chars
        # A frozen copy of the interrupted text and its matching state. Only its
        # matching runs again; nothing it does reaches the caller.
        self._matcher = progress._snapshot()
        self._matcher._placing = False
        self._matcher._cut = None
        # Text past what is proven heard is unproven, even if some unplaced
        # event already matched it.
        self._matcher._restart_matching(heard)

    def add_word(self, word: str) -> Optional[dict[str, str]]:
        self._matcher._match_word(word)
        self._heard = max(self._heard, self._matcher._match_cursor)
        return self.amendment()

    def amendment(self) -> Optional[dict[str, str]]:
        """The whole amendment, when it grew since last reported."""

        if self._heard <= self._reported:
            return None
        self._reported = self._heard
        return {
            "utterance_id": self.utterance_id,
            "text": self._matcher.text[self._shown : self._heard],
        }
