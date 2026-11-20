"""Discrete-time adapter for a hosted Avartha voice agent over Pipecat RTVI."""

from __future__ import annotations

import asyncio
import audioop
import html
import math
import re
import statistics
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass
from typing import Any, List, Optional

from loguru import logger
from pydantic import Field, model_serializer

from tau2.config import (
    DEFAULT_AUDIO_NATIVE_DISCONNECT_TIMEOUT,
    DEFAULT_AUDIO_NATIVE_TICK_TIMEOUT_BUFFER,
    DEFAULT_AVARTHA_INPUT_SAMPLE_RATE,
    DEFAULT_AVARTHA_MODEL,
)
from tau2.data_model.audio import TELEPHONY_AUDIO_FORMAT, AudioFormat
from tau2.data_model.message import ToolCall
from tau2.environment.tool import Tool
from tau2.voice.audio_native.adapter import DiscreteTimeAdapter
from tau2.voice.audio_native.async_loop import BackgroundAsyncLoop
from tau2.voice.audio_native.audio_converter import StreamingTelephonyConverter
from tau2.voice.audio_native.avartha_pipecat.audio_timing import (
    RTP_CLOCK_HZ,
    SPEECH_FRAME_MS,
    AudioClockBinding,
    speech_spans,
    unwrap_rtp,
)
from tau2.voice.audio_native.avartha_pipecat.caller_audio import CallerAudioUpsampler
from tau2.voice.audio_native.avartha_pipecat.config import AvarthaPipecatConfig
from tau2.voice.audio_native.avartha_pipecat.cut_token import (
    DEFAULT_UNIT_MS,
    UNIT_MS_FLOOR,
    split_token,
    units_heard,
)
from tau2.voice.audio_native.avartha_pipecat.provider import (
    AgentAudioEvent,
    AgentAudioTimelineEvent,
    AgentLLMStartedEvent,
    AgentLLMTextEvent,
    AgentPlaybackEvent,
    AgentStartedSpeakingEvent,
    AgentWordEvent,
    AvarthaBootError,
    AvarthaPipecatProvider,
    AvarthaProtocolError,
    ProviderErrorEvent,
    SpeechEvent,
    ToolCallEvent,
    TranscriptEvent,
)
from tau2.voice.audio_native.tick_result import TickResult

# Key of the call's one _SpokenTranscript. Audio carries it while a tick's text
# is taken, then its utterance's own ID (see _label_utterances).
_ITEM_ID = "avartha"
# How far past the last matched word a spoken word may land. It skips a word
# or two the TTS alignment dropped. A word never matches past the end of the
# sentence being spoken: that would mark unaired text as heard.
_WORD_MATCH_SLACK_CHARS = 16
# Shorter words ("I", "a", "to") match only the very next word, never skip.
_WORD_SKIP_MIN_CHARS = 3
# Hangul, kana, CJK ideographs and Devanagari. TTS alignment times these a
# syllable at a time ("확", "인", "해" for "확인해"; "छो", "टी" for "छोटी"), so a
# spoken piece is rarely a whole word.
_SYLLABIC_CHARS = re.compile(
    r"[ᄀ-ᇿ぀-ヿ㄰-㆏㐀-䶿"
    r"一-鿿가-힣豈-﫿ऀ-ॿ꣠-ꣿ]"
)
# How far past the last matched syllable a syllable may land: a dropped
# syllable or two and the space or punctuation between words.
_SYLLABLE_MATCH_SLACK_CHARS = 3
# CJK and full-width punctuation ("，", "。", "："), which no space follows.
_CJK_PUNCTUATION = re.compile(r"[　-〿！-／：-＠［-｀｛-･]")


def _is_syllabic(text: str) -> bool:
    """Whether every letter of ``text`` is a Hangul, kana, CJK or Devanagari syllable.

    Devanagari vowel signs and the virama are combining marks, not letters, so
    a piece such as "छो" or "र्ड" counts by its consonants.
    """

    letters = [char for char in text if char.isalnum()]
    return bool(letters) and all(_SYLLABIC_CHARS.match(char) for char in letters)


def _script_change(before: str, after: str) -> bool:
    """Whether two adjacent letters switch into or out of a syllabic script.

    Korean and Chinese attach particles and units to numbers and Latin words
    ("2,530,720원이고", "PayPal로", "17英寸"), so the switch ends a word.
    """

    return (
        before.isalnum()
        and after.isalnum()
        and bool(_SYLLABIC_CHARS.match(before)) != bool(_SYLLABIC_CHARS.match(after))
    )


def _word_edge(before: str, after: str) -> bool:
    """Whether a word may end between two adjacent characters.

    Besides a non-letter after it: a script change, and CJK punctuation
    before it, which no space follows ("1，1，6，0"). ASCII punctuation
    inside a word ("$13.46") does not end it.
    """

    return (
        not after.isalnum()
        or _script_change(before, after)
        or bool(_CJK_PUNCTUATION.match(before))
    )


# A caption owed by a line a caller cut is recognised by the first characters
# of the line's text (see _owed_caption). Captions come in playback order, so
# only lines cut since the last caption owe one; a bound of a few keeps a run
# of cuts with no caption between them from growing the list.
_CAPTION_KEY_CHARS = 24
_OWED_CAPTIONS_KEPT = 4


def _same_opening(caption_key: str, line_key: str) -> bool:
    length = min(len(caption_key), len(line_key), _CAPTION_KEY_CHARS)
    return bool(length) and caption_key[:length] == line_key[:length]


def _caption_key(text: str) -> str:
    """``text`` compared as a caption: unescaped, single-spaced, lower case."""

    return " ".join(_unescape(text).split()).lower()


# Ideographs and kana: scripts written without spaces, where the TTS times
# each character as a word.
_UNSPACED_CHARS = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def _next_word(text: str, start: int) -> str:
    """The word after ``start`` in ``text``, with any space before it.

    A word runs to the next space or script change; in a script written
    without spaces it is one character, with any punctuation attached
    ("好！"), as the TTS times it. Punctuation before it comes with it
    (", 6941833."); punctuation alone is no word.
    """

    first = start
    while first < len(text) and not text[first].isalnum():
        first += 1
    if first == len(text):
        return ""
    end = first + 1
    if _UNSPACED_CHARS.match(text[first]):
        while end < len(text) and _CJK_PUNCTUATION.match(text[end]):
            end += 1
    else:
        while (
            end < len(text)
            and not text[end].isspace()
            and not _script_change(text[end - 1], text[end])
            and not _UNSPACED_CHARS.match(text[end])
            and not _CJK_PUNCTUATION.match(text[end])
        ):
            end += 1
        while end < len(text) and _CJK_PUNCTUATION.match(text[end]):
            end += 1
    return text[start:end]


# Agent speech played since the last shown word that shows a cut caught the
# next word's sound: two speech frames, more than a click.
_STARTED_WORD_MIN_MS = 2 * SPEECH_FRAME_MS


@dataclass(frozen=True)
class _StartedWord:
    """What a cut showed of the word it caught after its sound started.

    ``text`` is shown; for a token read unit by unit (cut_token.py) it holds
    the ``units_heard`` of its ``units`` whose sound played, else both are
    None (an ordinary word, shown whole)."""

    text: str
    units_heard: Optional[int] = None
    units: Optional[int] = None


class AvarthaTickResult(TickResult):
    """A cut tick that shows a word the cut caught after its sound started.

    ``started_word`` is the text shown of it, part of
    ``proportional_transcript`` (see _HeardTranscript.interrupt): text the
    adapter inferred from the played audio rather than from a reported word
    end. For a code, identifier or digit string read unit by unit it is the
    units heard, ``started_word_units_heard`` of ``started_word_units``.
    Serialized only when set, so every other tick records what it did before.
    The agent records ticks with ``model_dump(serialize_as_any=True)``, which
    keeps these fields; a dump through a ``TickResult``-typed field drops them.
    """

    started_word: Optional[str] = Field(
        default=None,
        description="The text shown of the one word a cut caught mid-sound, inferred (no end reported)",
    )
    started_word_units_heard: Optional[int] = Field(
        default=None,
        description="Units of a token read unit by unit whose sound played: those shown",
    )
    started_word_units: Optional[int] = Field(
        default=None,
        description="Units of the started token read unit by unit",
    )

    @classmethod
    def with_started_word(
        cls, result: TickResult, started: _StartedWord
    ) -> AvarthaTickResult:
        """``result``, a TickResult the base adapter built, with ``started`` recorded.

        A field-for-field copy (no re-validation of its audio and events)."""

        fields = {
            "started_word": started.text,
            "started_word_units_heard": started.units_heard,
            "started_word_units": started.units,
        }
        return cls.model_construct(
            _fields_set=result.model_fields_set
            | {name for name, value in fields.items() if value is not None},
            **dict(result),
            **fields,
        )

    @model_serializer(mode="wrap")
    def _serialize(self, handler: Any) -> dict:
        data = handler(self)
        for name in ("started_word", "started_word_units_heard", "started_word_units"):
            if getattr(self, name) is None:
                data.pop(name, None)
        return data


def _unescape(text: str) -> str:
    """Avartha HTML-escapes some text, sometimes twice ("I&amp;#x27;m")."""

    for _ in range(3):
        unescaped = html.unescape(text)
        if unescaped == text:
            break
        text = unescaped
    return text


class _SpokenTranscript:
    """Reveal the agent's sentence text only as far as it has been aired.

    Avartha sends each sentence (``TranscriptEvent``) as soon as the LLM
    writes it, ahead of the audio, and streams the audio at about real time.
    Releasing text in proportion to audio played / audio received over the
    whole call therefore showed every sentence on its first audio chunk, and
    the simulated caller answered questions it had not heard yet.

    Text is released only on evidence that it aired, once the audio that
    carried the evidence has been played:

    - a spoken word (``AgentWordEvent``, sent by the TTS clock as it plays),
      matched in order within the sentence being spoken;
    - the played caption (``AgentPlaybackEvent``) Avartha reports when an
      utterance ends, which also covers words the alignment dropped.

    Without evidence text waits; it is never guessed. The text shown is always
    the sentence text, the words only mark progress through it.

    Implements the ``get_transcript_for_audio`` protocol that
    ``get_proportional_transcript`` calls on the audio played each tick.
    """

    def __init__(self, due_slack_bytes: int = 0) -> None:
        # Audio the adapter may be holding back at an utterance's end; a mark
        # this close to playback counts as played (see _hold_utterance_end).
        self._due_slack = due_slack_bytes
        self.item_id = _ITEM_ID
        self.text = ""
        self._sentence_ends: list[int] = []
        self._audio_received = 0
        self._audio_played = 0
        self._match_cursor = 0
        # A word that matched nothing yet: possibly the first piece of a word
        # the alignment split ("T-shir" + "t", "a" + "t"). See add_word.
        self._pending_piece = ""
        # Where the oldest line without its caption starts in ``text``: the
        # cursor, or (after a caption that came late) a reply start between
        # the last caption's end and the cursor. Always <= _match_cursor.
        self._utterance_start = 0
        self._spoken_chars = 0
        self._shown_chars = 0
        # (audio position, text length aired once playback reaches it)
        self._marks: deque[tuple[int, int]] = deque()
        # Audio position of the last mark that aired: where the next word's
        # sound can start at the earliest (see _HeardTranscript.interrupt).
        self._aired_position = 0
        # Spoken words that came before their sentence's text, in order, with
        # their timing (see add_word).
        self._early_words: list[tuple[str, Optional[int]]] = []
        # Where each reply's text starts, and whether the next sentence starts
        # a reply (see drop_unplayed_reply).
        self._reply_starts: list[int] = []
        self._reply_pending = False

    def begin_reply(self) -> None:
        """The next sentence starts a new reply (the LLM started one).

        Words still waiting for text belonged to the reply before; they would
        match the new reply's first words with the old words' timing. (An LLM
        run after a tool call is a new reply too: its words come after it.)"""

        self._reply_pending = True
        self._early_words.clear()

    def _splice(self, start: int, end: int, new: str) -> None:
        """Replace ``text[start:end]`` with ``new``, keeping every position
        into the text (sentence ends, reply starts) on the text it marks.

        A sentence end at ``start`` stays before the edit; a reply start at
        ``end`` stays at the start of its reply, after the edit. Positions
        inside a removed range go: a removed sentence's end and a removed
        reply's start mark nothing left.

        ``_utterance_start`` needs no shift: every edit starts at or after it
        (captions insert at or after it, drops start at or after the cursor).
        """

        delta = len(new) - (end - start)
        self.text = self.text[:start] + new + self.text[end:]
        self._sentence_ends = sorted(
            {
                e if e <= start else e + delta
                for e in self._sentence_ends
                if not start < e <= end or start == end
            }
        )
        self._reply_starts = [
            at if at < start else at + delta
            for at in self._reply_starts
            if not start <= at < end
        ]

    def drop_unplayed_reply(self, caption_end: Optional[int]) -> None:
        """The agent cut its own line (a caption with ``cut``, no caller
        speech): the rest of the caption's reply after the text it covers
        will never play.

        Left in place, it stood between the cursor and the next reply's text,
        so every later word matched nothing in the cut sentence and the next
        line showed nothing (seen live: 9 s of speech, no
        text). Only the caption's own reply is cut: a later reply, whose
        text may have come already or be speaking (an earlier line's caption
        can come late), stays. A caption whose text was not found
        (``caption_end`` None) drops nothing.
        """

        if caption_end is None:
            return
        start = max(caption_end, self._match_cursor)
        # The caption's reply ends where the next one starts; a reply starting
        # at the caption's end or later is a newer one.
        end = next(
            (at for at in self._reply_starts if at >= caption_end), len(self.text)
        )
        if end <= start:
            return
        self._splice(start, end, "")
        if start not in self._sentence_ends:
            self._sentence_ends = sorted(self._sentence_ends + [start])

    def add_text(self, text: str) -> None:
        """Append one bot-transcription sentence, keeping its own spacing.

        Avartha's splitter also breaks at decimal points ("$13." then
        "46"), and marks real sentence starts with a leading space. Only the
        first sentence of a response lacks one.
        """

        text = _unescape(text)
        if self._reply_pending:
            self._reply_pending = False
            self._reply_starts.append(len(self.text))
        if not self.text.strip():
            text = text.lstrip()
        elif not self.text[-1].isspace() and not text[0].isspace():
            if not (text[0].isdigit() and self.text[-1] in ".,"):
                text = " " + text
        self.text += text
        self._sentence_ends.append(len(self.text))
        early, self._early_words = self._early_words, []
        for word, timing in early:
            self._add_timed_word(word, timing)

    def _word_timing(self) -> Optional[int]:
        """The timing of the word being added, kept if it must wait for its text."""

        return None

    def _add_timed_word(self, word: str, timing: Optional[int]) -> None:
        self.add_word(word)

    def add_audio(self, num_bytes: int) -> None:
        self._audio_received += num_bytes

    def add_word(self, word: str) -> None:
        """Advance past a spoken word, joining it to a split word's first piece.

        TTS alignment splits words into pieces ("T-shir" + "t", "a" + "t",
        "8408879784" + "35."). A piece matches no whole word, so on its own it
        would leave text waiting for the caption at the utterance's end, and a
        barge-in before then discards the rest of the line although its audio
        played. A word that matches nothing is held; when the next word joins
        it into one that matches, both count as spoken. Otherwise the held
        piece is dropped and the new word is matched alone, so "I" followed by
        "If" still never matches inside "If".

        Korean, Japanese and Chinese are timed a syllable at a time ("확",
        "인", "해" for "확인해"). Holding one piece joined only two-syllable
        words, so a Korean line's text stalled at its first longer word. A
        syllable therefore matches in place (see _match). Hindi is timed the
        same way ("ए", "क" for "एक"; "छो", "टी" for "छोटी"), sometimes with a
        word's last syllable after the next word's first ("छो", "सी", "बा",
        "टी", "त" for "छोटी सी बात"); a syllable behind the text matches
        nothing and the next one continues.
        """

        needle = _unescape(word).strip().lower()
        if not needle:
            return
        if bisect_right(self._sentence_ends, self._match_cursor) == len(
            self._sentence_ends
        ):
            # Its sentence has not arrived: the TTS can speak a sentence before
            # Avartha sends its text (often in Chinese). Dropped, its words
            # left the cursor behind, and the words after them, matched only a
            # syllable or two ahead, never found their place: the line showed
            # nothing until its caption, and nothing at all if cut first. It
            # waits for the text (add_text), keeping its timing.
            self._early_words.append((word, self._word_timing()))
            return
        piece, self._pending_piece = self._pending_piece, ""
        if piece and self._match(piece + needle):
            return
        if not self._match(needle):
            self._pending_piece = needle

    def _match(self, needle: str) -> bool:
        """Advance past ``needle`` if it is the next spoken word; say whether it was."""

        sentence = bisect_right(self._sentence_ends, self._match_cursor)
        if sentence == len(self._sentence_ends):
            return False  # its sentence has not arrived yet
        haystack = self.text.lower()
        limit = min(
            self._match_cursor + len(needle) + _WORD_MATCH_SLACK_CHARS,
            self._sentence_ends[sentence],
        )
        cursor = self._match_cursor
        if _is_syllabic(needle):
            # Each syllable is a sound of the next word or of the rest of this
            # one, so it matches where it stands, mid-word or not, skipping
            # only a dropped syllable or two.
            index = haystack.find(
                needle,
                cursor,
                min(
                    cursor + len(needle) + _SYLLABLE_MATCH_SLACK_CHARS,
                    self._sentence_ends[sentence],
                ),
            )
            if index >= 0:
                self._advance(index + len(needle))
                return True
            return False
        if 0 < cursor < len(haystack) and (
            haystack[cursor - 1].isalnum()
            and haystack[cursor].isalnum()
            and not _script_change(haystack[cursor - 1], haystack[cursor])
        ):
            # Mid-word: only the rest of that word may follow ("Wha" + "t").
            if haystack.startswith(needle, cursor):
                self._advance(cursor + len(needle))
                return True
            return False
        # Otherwise a whole word. A fragment ("Wha") is indistinguishable from
        # a short word inside a longer one ("I" in "If"), so it matches nothing
        # here; add_word joins it with the next piece instead.
        if len(needle.strip(".,!?;:—-")) < _WORD_SKIP_MIN_CHARS:
            # Too common to skip ahead for: only the very next word.
            nxt = cursor
            while nxt < limit and not haystack[nxt].isalnum():
                nxt += 1
            limit = min(limit, nxt + len(needle))
        index = haystack.find(needle, cursor, limit)
        while index >= 0 and not self._is_whole_word(haystack, index, len(needle)):
            index = haystack.find(needle, index + 1, limit)
        if index >= 0:
            self._advance(index + len(needle))
            return True
        return False

    @staticmethod
    def _is_whole_word(text: str, index: int, length: int) -> bool:
        end = index + length
        starts = index == 0 or not text[index - 1].isalnum()
        if not starts:
            starts = _script_change(text[index - 1], text[index]) or bool(
                _CJK_PUNCTUATION.match(text[index])
            )
        return starts and (end == len(text) or _word_edge(text[end - 1], text[end]))

    def add_playback(self, caption: str) -> Optional[int]:
        """Avartha's caption of what this utterance actually played.

        Text in the caption that the sentences lack (speech no LLM sentence
        produced, or a reply's unpunctuated tail) is inserted where it aired.

        Returns where the caption's text ends in ``text``, never past the end
        of the line it was placed in, or None if it was placed nowhere (see
        _place_caption for how a caption finds its line). A caption that
        comes late, after a later reply has started speaking, moves the line
        start only to the next reply, so the next late caption still finds
        its line; so does a caption placed nowhere.
        """

        self._pending_piece = ""
        # Words of an ended utterance that never found text are not the next's.
        self._early_words.clear()
        caption = " ".join(_unescape(caption).split())
        if not caption:
            return None
        caption_end = self._place_caption(caption)
        if caption_end is None:
            # Placed nowhere, it is still the oldest line's caption: the next
            # caption is due at the next reply, if one has started.
            newer = next(
                (
                    at
                    for at in self._reply_starts
                    if self._utterance_start < at <= self._match_cursor
                ),
                None,
            )
        else:
            newer = next(
                (
                    at
                    for at in self._reply_starts
                    if caption_end <= at < self._match_cursor
                ),
                None,
            )
        self._utterance_start = self._match_cursor if newer is None else newer
        return caption_end

    def _line_starts(self) -> list[int]:
        """Where the caption now due can start, oldest first: the oldest
        uncaptioned line, then each later reply whose text has come, up to the
        cursor (its line's predecessor may have sent no caption)."""

        return [self._utterance_start] + [
            at
            for at in self._reply_starts
            if self._utterance_start < at <= self._match_cursor
        ]

    def _reply_stop(self, start: int) -> int:
        # The caption starts in its own reply, never in a later one whose text
        # came already. It may run on into a later LLM run (a reply resumed
        # after a tool call is spoken as one utterance).
        return next((at for at in self._reply_starts if at > start), len(self.text))

    def _place_caption(self, caption: str) -> Optional[int]:
        """Place ``caption`` in the line it reports; return its end.

        The one statement of the rules (the README refers here):

        - Captions come one per line, in playback order, and name no
          utterance, so a caption is the oldest uncaptioned line's, starting
          at ``utterance_start``. A line can send no caption, so each later
          reply whose text has come, up to the cursor, is tried after it.
        - Every start is tried whole first, oldest first, then aligned,
          oldest first: a short shared opening is weaker evidence than a
          later reply holding the caption whole.
        - The oldest line takes the caption whole anywhere in its reply, or
          aligned sharing any opening; a later reply, or the oldest once the
          cursor is in a later reply, only whole at its start or sharing at
          least the caption's first word.
        - A caption runs on into the next LLM run (one line over a reply
          resumed after a tool call) only ending where a word of that run
          ends; aligned, only into a run already heard that shares its first
          word.
        - In unspaced scripts a "word" is the whole caption, so captions
          there are placed at a later reply, or run on, only whole.
        """

        starts = self._line_starts()
        for index, start in enumerate(starts):
            end = self._place_whole(caption, start, later=index > 0)
            if end is not None:
                return end
        for index, start in enumerate(starts):
            end = self._align_caption(caption, start, later=index > 0)
            if end is not None:
                return end
        return None

    def _shared_opening(self, caption: str, start: int, stop: int) -> int:
        """How many characters ``caption`` shares with ``text[start:stop]``."""

        aired = self.text[start:stop].lower()
        common = 0
        while (
            common < len(caption)
            and common < len(aired)
            and aired[common] == caption[common].lower()
        ):
            common += 1
        return common

    @staticmethod
    def _skip_space(text: str, start: int) -> int:
        while start < len(text) and text[start].isspace():
            start += 1
        return start

    def _place_whole(self, caption: str, start: int, *, later: bool) -> Optional[int]:
        """The end of ``caption`` found whole in the line starting at
        ``start``: anywhere in the oldest line's reply, at a ``later``
        reply's start only."""

        start = self._skip_space(self.text, start)
        stop = self._reply_stop(start)
        index = self.text.lower().find(caption.lower(), start)
        end = index + len(caption)
        if not 0 <= index < stop or (later and index != start):
            return None
        if end > stop and end < len(self.text) and self.text[end].isalnum():
            # Run on into the next LLM run (one line over a reply resumed
            # after a tool call) only ending where a word of it ends: "The"
            # inside "There" is an earlier line's caption, not this run's.
            return None
        self._advance(end)
        return end

    def _align_caption(self, caption: str, start: int, *, later: bool) -> Optional[int]:
        """Align ``caption`` with the text of the line starting at ``start``,
        inserting the text it adds; return its end, bounded by that line's
        reply, or None if it does not belong there.

        The oldest line takes a caption sharing any opening with its text, or
        (with no text past the cursor to share) as text it adds. A ``later``
        reply, or the oldest once the cursor is in a later reply, must share
        at least the caption's first word. (In unspaced scripts that is the whole
        caption, so such captions are placed there only whole.)"""

        start = self._skip_space(self.text, start)
        stop = self._reply_stop(start)
        common = self._shared_opening(caption, start, stop)
        while common == stop - start < len(caption) and stop < self._match_cursor:
            # The caption runs on past this LLM run into the next, already
            # heard: one line over a reply resumed after a tool call, if it
            # shares at least that run's first word, ending where a caption
            # word ends too (not only a leading space, a letter, or "The" of
            # "There"). A run starting with a word the caption spells
            # otherwise is taken for a new line: the caption may be an
            # earlier line's, and its reply is not cut.
            further = self._reply_stop(stop)
            shared = self._shared_opening(caption, start, further)
            words = self.text[stop:further].split()
            if not words:
                break
            first_end = self._skip_space(self.text, stop) + len(words[0])
            word_ends = first_end - start
            if start + shared < first_end or (
                word_ends < len(caption) and caption[word_ends].isalnum()
            ):
                break
            stop, common = further, shared
        if common == len(caption):
            # Shared whole: the whole-caption pass declined it (it ends inside
            # a word of the next run), so it is not this line's either.
            return None
        # The cursor in a later reply: this line's caption may have come late,
        # or not at all and the caption is a later line's.
        left = stop < len(self.text) and self._match_cursor >= stop
        if (later or left) and common < len(caption.split()[0]):
            return None
        at = start + common
        if at < self._match_cursor:
            # Parted behind the cursor (a character the caption spells
            # differently). With no opening shared, it may be an earlier
            # line's. If the cursor is still in this reply, the caption's rest
            # is aligned with the text after it; past it, the caption ends
            # with its reply.
            if not common:
                return None
            if self._match_cursor >= stop:
                return stop
            rest = caption[self._match_cursor - start :].lower()
            if rest and self.text.lower().startswith(rest, self._match_cursor):
                self._advance(self._match_cursor + len(rest))
            return min(self._match_cursor, stop)
        missing = caption[common:]
        if at > 0 and not self.text[at - 1].isspace() and not missing[0].isspace():
            missing = " " + missing if common == 0 else missing
        if self.text[at:].strip() and not missing[-1].isspace():
            missing += " "
        logger.debug(f"Avartha playback adds unsent text: {missing!r}")
        self._splice(at, at, missing)
        self._sentence_ends = sorted(
            set(self._sentence_ends) | {at + len(missing.rstrip())}
        )
        self._advance(at + len(missing.rstrip()))
        return at + len(missing.rstrip())

    def _advance(self, cursor: int) -> None:
        if cursor <= self._match_cursor:
            return
        self._match_cursor = cursor
        self._marks.append((self._audio_received, cursor))

    def interrupt(self) -> None:
        """Forget everything the caller has not heard yet."""

        self.text = self.text[: self._shown_chars]
        self._sentence_ends = [e for e in self._sentence_ends if e < len(self.text)]
        self._sentence_ends.append(len(self.text))
        self._audio_received = self._audio_played
        self._match_cursor = self._spoken_chars = self._shown_chars
        self._utterance_start = self._match_cursor
        self._pending_piece = ""
        self._early_words.clear()
        self._reply_starts = [at for at in self._reply_starts if at < len(self.text)]
        self._marks.clear()
        self._aired_position = self._audio_played

    @property
    def shown_chars(self) -> int:
        """How much of ``text`` has been shown."""

        return self._shown_chars

    @property
    def utterance_start(self) -> int:
        """Where the oldest line still owing its caption starts in ``text``:
        the line now playing, unless a caption came late (see add_playback)."""

        return self._utterance_start

    @property
    def matched_chars(self) -> int:
        """How far into ``text`` the spoken words and captions have reached."""

        return self._match_cursor

    def _air(self, position: int, cursor: int) -> None:
        """The mark at audio ``position``, text length ``cursor``, has aired."""

        self._spoken_chars = max(self._spoken_chars, cursor)
        self._aired_position = max(self._aired_position, position)

    def get_transcript_for_audio(self, audio_bytes_to_play: int) -> str:
        self._audio_played += audio_bytes_to_play
        while self._marks and self._marks[0][0] <= self._audio_played + self._due_slack:
            self._air(*self._marks.popleft())
        end = min(self._spoken_chars, len(self.text))
        text = self.text[self._shown_chars : end]
        self._shown_chars = max(self._shown_chars, end)
        return text


def _mcp_tool_schema(tool: Tool) -> dict[str, Any]:
    """Translate Tau2's OpenAI-shaped schema into the harness MCP profile."""

    function = tool.openai_schema["function"]
    return {
        "name": function["name"],
        "description": function.get("description") or function["name"],
        "inputSchema": function["parameters"],
    }


@dataclass(frozen=True)
class _CaptionTail:
    """A captioned line whose audio may still arrive (see
    _continues_captioned_line and _discarded_utterance)."""

    line_id: str
    # Contiguous audio starting before this RTP position is still the line's.
    end_rtp: int
    # Where the line's last word's sound ends, if its last timed word is the
    # line's last word: audio from here on is the line's quiet tail.
    quiet_from_rtp: Optional[int]


@dataclass(frozen=True, slots=True)
class _AudioSpan:
    """Received audio on the RTP clock: ``rtp_samples`` ticks from ``rtp_start``
    became ``byte_count`` telephony bytes from ``byte_start``."""

    rtp_start: int
    rtp_samples: int
    byte_start: int
    byte_count: int
    # Media decoded without a gap from here up to rtp_start (see AgentAudioEvent).
    decoded_from: Optional[int]

    @property
    def rtp_end(self) -> int:
        return self.rtp_start + self.rtp_samples


# A word end this far past the audio received is not on this clock (a wrong
# binding, or a bridge that rewrote timestamps): place the word as untimed.
_MAX_RTP_LEAD = 60 * RTP_CLOCK_HZ
# Mark position of a timed word whose audio has not arrived: beyond any playback.
_UNPLACED = 1 << 62
# How far past its last timed word's end a line's audio may still arrive after
# its played caption, and belong to it: the word's remaining sound and the
# line's quiet tail (~130 ms measured on recorded calls). See
# _continues_captioned_line.
_CAPTION_TAIL_RTP = RTP_CLOCK_HZ // 4
# A caption overtakes its line's audio by the data channel's lead over the
# media track; a last word end further ahead of the audio is not trusted.
_CAPTION_MAX_LEAD_RTP = RTP_CLOCK_HZ
# A gap of lost media no longer than this may fall inside a word (a burst of
# lost packets); a longer one is taken as the silence between sounds.
_MAX_LOST_IN_WORD_RTP = RTP_CLOCK_HZ // 5


def _sound_restarts(
    previous_end: int, rtp_start: int, decoded_from: Optional[int]
) -> bool:
    """Whether sound restarts at audio received after a gap in the RTP clock.

    The receiver decodes from ``decoded_from`` without a break, dropping the
    silence it decodes. Silence decoded just before this audio (``decoded_from
    < rtp_start``), even after lost packets, means its sound starts here. A gap
    that is all lost media (``decoded_from >= rtp_start``: equal, or the
    resampler's delay later, as the receiver stamps ``rtp_start`` after
    resampling) may have cut a word; a short one is taken as such, so the word
    keeps the sound it had before the loss. With no ``decoded_from`` the gap
    is unknown and taken as silence.
    Known limit: a pause of at most that length between two lines whose
    packets were all lost reads as such a burst, so the next line's first
    word is credited the previous line's played tail. This assumes the sender
    transmits silence as packets (no Opus DTX).
    """

    if rtp_start <= previous_end:
        return False
    if decoded_from is None or decoded_from < rtp_start:
        return True
    return rtp_start - previous_end > _MAX_LOST_IN_WORD_RTP


class _HeardTranscript(_SpokenTranscript):
    """``_SpokenTranscript`` with each timed word placed where its sound ends.

    Over WebRTC a word event reaches the adapter on the data channel, on its
    own clock, so its arrival says nothing about its audio: a word heard
    before a cut but reported in the cut tick looked the same as one whose
    audio the cut discarded. Avartha now times words on the audio's RTP clock
    (``audio_timing.py``) and the receiver keeps each audio event's RTP
    position, so a timed word airs once playback has passed the end of its
    sound, and a cut keeps the words whose sound ended in the audio played.
    "Played" counts the utterance's held last millisecond, as for every mark
    (``_hold_utterance_end``).

    A word whose end is unknown (no binding, another clock epoch, media lost
    or the receiver keeps no timestamps) is placed as ``_SpokenTranscript``
    places every word: where the audio received stood on its arrival. With no
    timed words the two classes behave identically.
    """

    def __init__(
        self,
        due_slack_bytes: int = 0,
        *,
        sample_rate: int = TELEPHONY_AUDIO_FORMAT.sample_rate,
    ) -> None:
        super().__init__(due_slack_bytes)
        # Audio positions count bytes of played u-law: one per sample.
        self._sample_rate = sample_rate
        self._spans: list[_AudioSpan] = []
        self._span_starts: list[int] = []
        # Whether the receiver has placed any of this call's audio on the RTP
        # clock; until it does, timed words cannot be placed.
        self._clocked = False
        # A rewind invalidates this call's timing epoch; do not reuse it.
        self._clock_invalid = False
        # RTP timestamp of the end of the word being matched (add_word).
        self._word_end: Optional[int] = None
        # Text lengths whose mark is timed, and timed marks whose audio has
        # not arrived yet, in text order: length -> (RTP timestamp, arrival).
        self._timed: set[int] = set()
        self._waiting: dict[int, tuple[int, int]] = {}
        # RTP timestamp through which the receiver has decoded only silence
        # after the last span, if it has reported that (decoded_silence).
        self._silent_through: Optional[int] = None
        # Heard text a cut kept that no tick has shown yet (see interrupt).
        self._kept = ""
        # Whether the last cut kept the word it caught mid-sound: the cut
        # stopped a word, so it cut off the line that word belongs to.
        self.kept_caught_word = False
        # The word a cut showed after its sound started (see interrupt), until
        # the cut tick records it.
        self._started: Optional[_StartedWord] = None
        # The last aired mark, (position, text length), if it was timed: the
        # start of the next timed word's sound (see _air).
        self._last_timed_aired: Optional[tuple[int, int]] = None
        # Completed codes' sound per unit, in audio bytes (see _air).
        self._unit_bytes: list[float] = []
        # Speech in the audio played after the last aired mark: spans of
        # 20 ms speech frames, in audio positions (note_played).
        self._speech: deque[tuple[int, int]] = deque()
        # Audio position through which played audio has been noted.
        self._noted_through = 0
        # Received-byte positions where sound began, in order: a line's first
        # audio, or audio after silence the receiver dropped. A word's sound
        # starts no earlier than the last of them before its end (interrupt).
        self._sound_starts: list[int] = [0]
        self.timed_words = 0
        self.untimed_words = 0
        # The last word placed on the clock: its sound's end (RTP) and its
        # end in ``text``.
        self.last_timed_word: Optional[tuple[int, int]] = None

    @property
    def clocked(self) -> bool:
        return self._clocked

    def add_audio(
        self,
        num_bytes: int,
        rtp_start: Optional[int] = None,
        rtp_samples: int = 0,
        decoded_from: Optional[int] = None,
    ) -> None:
        if not self._clock_invalid and rtp_start is not None and rtp_samples > 0:
            self._clocked = True
            if self._spans and rtp_start < self._spans[-1].rtp_end:
                # Not after the audio before it: the clock is not one we can
                # follow, so stop timing from it.
                self._spans.clear()
                self._span_starts.clear()
                self._clocked = False
                self._clock_invalid = True
                self._silent_through = None
                self.untime_waiting()
            else:
                if self._spans and _sound_restarts(
                    self._spans[-1].rtp_end, rtp_start, decoded_from
                ):
                    self._add_sound_start()
                self._spans.append(
                    _AudioSpan(
                        rtp_start,
                        rtp_samples,
                        self._audio_received,
                        num_bytes,
                        decoded_from,
                    )
                )
                self._span_starts.append(rtp_start)
                self._silent_through = None
        super().add_audio(num_bytes)
        if self._waiting:
            self._place_waiting()

    def add_word(self, word: str, end_rtp_timestamp: Optional[int] = None) -> None:
        self._word_end = end_rtp_timestamp
        try:
            super().add_word(word)
        finally:
            self._word_end = None

    def _word_timing(self) -> Optional[int]:
        return self._word_end

    def _add_timed_word(self, word: str, timing: Optional[int]) -> None:
        self.add_word(word, timing)

    def _advance(self, cursor: int) -> None:
        if self._word_end is None or cursor <= self._match_cursor:
            super()._advance(cursor)
            return
        self._match_cursor = cursor
        position = self._position(self._word_end)
        if position is None:
            self.untimed_words += 1
            self._marks.append((self._audio_received, cursor))
            return
        self.timed_words += 1
        self.last_timed_word = (self._word_end, cursor)
        self._timed.add(cursor)
        if position is _NOT_YET_RECEIVED:
            # Its sound has not arrived yet: nothing after it airs before it.
            self._waiting[cursor] = (self._word_end, self._audio_received)
            self._marks.append((_UNPLACED, cursor))
        else:
            self._marks.append((position, cursor))

    def _position(self, rtp_timestamp: int) -> Any:
        """Received-byte position where the sound at ``rtp_timestamp`` has played.

        ``_NOT_YET_RECEIVED`` if that audio has not arrived yet, None if it is
        unknown (no timed audio, lost media, before the first audio or too
        far ahead to be on this clock).
        """

        if not self._spans:
            # Audio comes after this word (a new reply, or after a cut).
            return _NOT_YET_RECEIVED if self._clocked else None
        last = self._spans[-1]
        end = unwrap_rtp(rtp_timestamp, last.rtp_end)
        decoded_through = max(last.rtp_end, self._silent_through or last.rtp_end)
        if end > decoded_through + _MAX_RTP_LEAD:
            return None
        if end > last.rtp_end:
            return _NOT_YET_RECEIVED
        index = bisect_right(self._span_starts, end) - 1
        if index < 0:
            return None
        span = self._spans[index]
        if end <= span.rtp_end:
            return span.byte_start + math.ceil(
                (end - span.rtp_start) * span.byte_count / span.rtp_samples
            )
        following = self._spans[index + 1]
        if following.decoded_from is not None and following.decoded_from <= end:
            # The sound ended in silence the receiver dropped.
            return following.byte_start
        return None  # in media that was lost

    def _place_waiting(self) -> None:
        placed: dict[int, int] = {}
        for cursor, (rtp_timestamp, arrival) in list(self._waiting.items()):
            position = self._position(rtp_timestamp)
            if position is _NOT_YET_RECEIVED:
                break  # later words wait behind it anyway (marks air in order)
            del self._waiting[cursor]
            if position is None:
                self._timed.discard(cursor)
                placed[cursor] = arrival
            else:
                placed[cursor] = position
        self._replace_marks(placed)

    def _replace_marks(self, placed: dict[int, int]) -> None:
        if placed:
            self._marks = deque(
                (placed.get(cursor, position), cursor)
                for position, cursor in self._marks
            )

    def is_timed(self, cursor: int) -> bool:
        """Whether the word ending at text ``cursor`` is placed by its sound."""

        return cursor in self._timed

    def untime_waiting(self) -> None:
        """Place the words still waiting for their audio as untimed: at their arrival."""

        placed = {cursor: arrival for cursor, (_, arrival) in self._waiting.items()}
        for cursor in placed:
            self._timed.discard(cursor)
        self._waiting.clear()
        self._replace_marks(placed)

    def decoded_silence(self, after: int, through: int) -> None:
        """The receiver decoded silence from RTP ``after``, the end of the audio
        received, through ``through``: words that ended in it have played once
        the audio received has."""

        if not self._spans or self._spans[-1].rtp_end != after:
            return  # audio the adapter has not received yet came first
        self._silent_through = through
        end_of_audio = self._audio_received
        placed: dict[int, int] = {}
        for cursor, (rtp_timestamp, _) in list(self._waiting.items()):
            if unwrap_rtp(rtp_timestamp, after) > through:
                break
            del self._waiting[cursor]
            placed[cursor] = end_of_audio
        self._replace_marks(placed)

    def add_playback(self, caption: str) -> Optional[int]:
        # The utterance has ended: a timed word whose audio never arrived is
        # placed as it would be untimed, and later words need none of its spans.
        self.untime_waiting()
        del self._spans[:-1], self._span_starts[:-1]
        return super().add_playback(caption)

    def begin_line(self) -> None:
        """The audio received next starts a new line (utterance)."""

        self._add_sound_start()

    def _add_sound_start(self) -> None:
        # A start at or before the last aired mark never bounds a word's sound
        # (interrupt starts from that mark), so only later ones are kept.
        self._sound_starts = [
            at for at in self._sound_starts if at > self._aired_position
        ]
        self._sound_starts.append(self._audio_received)

    def _rtp_at(self, position: int, *, end: bool = False) -> Optional[int]:
        """RTP timestamp of received-byte ``position``, if a span holds it.

        A position between two spans is the later span's start, or with
        ``end`` the earlier span's end: the silence between them is neither.
        """

        for span in reversed(self._spans):
            if span.byte_start < position or (span.byte_start == position and not end):
                if position > span.byte_start + span.byte_count:
                    return None  # past the audio received
                return span.rtp_start + (
                    (position - span.byte_start) * span.rtp_samples // span.byte_count
                )
        return None

    def _mostly_played(self, start: int, position: int, cursor: int) -> bool:
        """Whether more than half of the word whose sound runs from ``start``
        (received bytes) to its end mark (``position``, ``cursor``) played.

        Measured against the audio actually played, without the held-tail
        slack that decides whether a mark has aired."""

        played = self._audio_played
        if played <= start:
            return False
        if position != _UNPLACED:
            return played - start > (position - start) / 2
        # The audio carrying its end has not arrived: compare on the RTP clock.
        waiting = self._waiting.get(cursor)
        start_rtp = self._rtp_at(start)
        played_rtp = self._rtp_at(played, end=True)
        if waiting is None or start_rtp is None or played_rtp is None:
            return False
        end_rtp = unwrap_rtp(waiting[0], played_rtp)
        return played_rtp - start_rtp > (end_rtp - start_rtp) / 2

    def interrupt(self, *, speech_before_cut: bool = False) -> None:
        """Keep the timed words the caller heard: those whose sound ended in
        the audio played, and the word the cut caught if more than half of its
        sound had played.

        A word's sound runs from the previous word's end, or from where its
        sound began (the last of ``_sound_starts`` before its end) if later,
        to its own end; the previous line's tail and the silence between
        lines are not its. Keeping only words whose sound had ended dropped
        the one the cut caught mid-sound, although its clip carries most of
        it, so a judge comparing the clip with its text heard a word the text
        lacked. A word the cut caught at its start is still dropped.

        The kept text is held for the cut tick (``take_kept``); it counts as
        shown, so a second cut in the same tick keeps it too.

        ``speech_before_cut``: the line still streams at the cut and the tick
        before it played speech. If at least 40 ms of speech played after the
        last shown word (``note_played``), the cut keeps no word, no later
        word's end was reported, and the shown text stops inside a sentence,
        that speech is the sentence's next word, caught mid-sound before its
        end was reported (a code read digit by digit is one word of several
        seconds). That one inferred word is kept too, and recorded as the started word
        (``take_started``): the cut tick's clip carries its sound, and without
        its text a judge comparing the clip with the line heard words the text
        lacked. A short ordinary word is kept whole: the caller may have heard
        only its start. Of a code read unit by unit only the units whose sound
        surely played are kept, and of a long word or a number read as number
        words nothing (``_spoken_part``), so the caller is not shown digits it
        never heard. A second cut in the same tick infers no word: nothing has
        played after the kept text.
        """

        played = self._audio_played + self._due_slack
        heard = max(
            (
                cursor
                for position, cursor in self._marks
                if cursor in self._timed and position <= played
            ),
            default=0,
        )
        # The word the cut caught: the first timed mark after ``heard``. Its
        # sound starts where the last mark at or before it that played stands
        # (an untimed mark stands at its arrival, an approximation). An
        # unplayed untimed mark before it means the caller heard neither.
        start = self._aired_position
        caught: Optional[tuple[int, int]] = None
        for position, cursor in self._marks:
            if cursor <= heard:
                if position <= played:
                    start = max(start, position)
                continue
            if cursor in self._timed:
                caught = (position, cursor)
                break
            if position > played:
                break
            start = max(start, position)
        if caught is not None:
            # Starts of sound received after the word's end are a later
            # phrase's or line's; a word whose end has not arrived ends after
            # every start received.
            start = max([start] + [at for at in self._sound_starts if at < caught[0]])
        self.kept_caught_word = caught is not None and self._mostly_played(
            start, *caught
        )
        if caught is not None and self.kept_caught_word:
            heard = caught[1]
        if heard > self._shown_chars:
            self._kept += self.text[self._shown_chars : heard]
            self._shown_chars = heard
        elif (
            speech_before_cut
            and self._speech_after_shown_ms() >= _STARTED_WORD_MIN_MS
            # Its end was never reported: a word whose end mark came keeps the
            # rule above (dropped unless mostly played).
            # (_marks holds every reported end, placed or still waiting.)
            and not any(cursor > self._shown_chars for _, cursor in self._marks)
            and len(self.text[: self._shown_chars].rstrip()) not in self._sentence_ends
        ):
            word = _next_word(self.text, self._shown_chars)
            started = self._spoken_part(word) if word else None
            if started is not None and started.text:
                self._kept += started.text
                self._shown_chars += len(started.text)
                self._started = started
        super().interrupt()
        # Nothing played after the kept text yet: a second cut in this tick
        # infers no word.
        self._speech.clear()
        self._last_timed_aired = None
        self._sound_starts = [self._audio_played]
        self._spans.clear()
        self._span_starts.clear()
        self._timed.clear()
        self._waiting.clear()
        self._silent_through = None

    def take_kept(self) -> str:
        """Heard text the last cut kept, once."""

        kept, self._kept = self._kept, ""
        return kept

    @property
    def caught_started_word(self) -> bool:
        """Whether the kept text holds a word caught after its sound started."""

        return self._started is not None

    def take_started(self) -> Optional[_StartedWord]:
        """What the kept text shows of the word caught after its sound
        started, once (see interrupt)."""

        started, self._started = self._started, None
        return started

    def _air(self, position: int, cursor: int) -> None:
        """Also measures a unit of a completed code (see the base method; any
        other place that airs a mark must call it too)."""
        # A timed word right after a timed word, in the same stretch of sound,
        # whose sound runs from the earlier's end to its own: a token that
        # SplitToken.measures_unit accepts measures a unit's duration for the
        # rate (_units_heard).
        timed = cursor in self._timed
        before = self._last_timed_aired
        if (
            timed
            and before is not None
            and position > before[0]
            and cursor > before[1]
            and not any(before[0] < at < position for at in self._sound_starts)
        ):
            token = split_token(self.text[before[1] : cursor])
            if token.measures_unit:
                self._unit_bytes.append((position - before[0]) / len(token.units))
        self._last_timed_aired = (position, cursor) if timed else None
        super()._air(position, cursor)

    def _unit_ms(self) -> float:
        """A unit's duration: the median on this call's completed codes, but
        no less than cut_token.UNIT_MS_FLOOR; else cut_token.DEFAULT_UNIT_MS."""

        if self._unit_bytes:
            median = statistics.median(self._unit_bytes) * 1000 / self._sample_rate
            return max(median, UNIT_MS_FLOOR)
        return DEFAULT_UNIT_MS

    def note_played(self, audio: bytes) -> bool:
        """``audio``, the u-law bytes ``get_transcript_for_audio`` has just
        counted as played, in order: notes its speech frames (20 ms, PCM16 peak
        over 32; audio_timing.speech_spans) after the last shown word. Returns
        whether any of it is speech.

        This is the one clock of agent speech after the last shown word: the
        gate for a started word and a started code's sound (interrupt). Frames
        are counted from the shown word's end, so a frame never carries the
        end of the shown word's sound. (Marks air before this is called, so
        the start is known for all of ``audio``.)"""

        start = self._audio_played - len(audio)
        if start != self._noted_through:
            # Internal invariant: every played byte is noted once, in order.
            raise RuntimeError(
                f"Avartha played audio noted out of order: {len(audio)} bytes "
                f"ending at {self._audio_played}, noted through {self._noted_through}"
            )
        self._noted_through = self._audio_played
        pcm = audioop.ulaw2lin(audio, 2)
        since = self._token_start()
        while self._speech and self._speech[0][1] <= since:
            self._speech.popleft()
        skip = min(len(audio), max(0, since - start))
        after = speech_spans(pcm[2 * skip :], self._sample_rate)
        for first, last in after:
            first, last = start + skip + first, start + skip + last
            if self._speech and self._speech[-1][1] == first:
                self._speech[-1] = (self._speech[-1][0], last)
            else:
                self._speech.append((first, last))
        return bool(after) or bool(
            skip and speech_spans(pcm[: 2 * skip], self._sample_rate)
        )

    def _token_start(self) -> int:
        """Where the sound of the word after the shown text starts: the last
        aired mark (the last shown word's end), or where sound began after it."""

        played = self._audio_played
        return max(
            [self._aired_position] + [at for at in self._sound_starts if at <= played]
        )

    def _speech_after(self, start: int) -> list[tuple[int, int]]:
        return [(max(a, start), b) for a, b in self._speech if b > start]

    def _speech_after_shown_ms(self) -> float:
        """Speech played after the last shown word, in ms."""

        spans = self._speech_after(self._token_start())
        return sum(b - a for a, b in spans) * 1000 / self._sample_rate

    def _units_heard(self) -> int:
        """Units of a started code whose sound surely played up to the cut
        (cut_token.units_heard).

        Its sound runs from the first speech frame after the last shown word
        to the end of the last speech frame before the cut: quiet before it
        is no code sound, quiet inside it is."""

        spans = self._speech_after(self._token_start())
        sound = spans[-1][1] - spans[0][0] if spans else 0
        return units_heard(sound * 1000 / self._sample_rate, self._unit_ms())

    def _spoken_part(self, word: str) -> _StartedWord:
        """What to show of ``word``, caught after its sound started: a short
        ordinary word whole; of a code read unit by unit the units surely heard,
        possibly none; of anything else nothing (cut_token.py)."""

        token = split_token(word)
        if not token.read_by_units:
            return _StartedWord(token.shown(0))
        heard = min(self._units_heard(), len(token.units))
        return _StartedWord(token.shown(heard), heard, len(token.units))


# _HeardTranscript._position: the audio carrying the word's end has not arrived.
_NOT_YET_RECEIVED = object()


class DiscreteTimeAvarthaPipecatAdapter(DiscreteTimeAdapter):
    """Bridge tau2 telephony ticks to a hosted Avartha agent's call."""

    def __init__(
        self,
        tick_duration_ms: int,
        send_audio_instant: bool = True,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        audio_format: Optional[AudioFormat] = None,
        avartha_config: Optional[AvarthaPipecatConfig] = None,
        provider: Optional[AvarthaPipecatProvider] = None,
        language: Optional[str] = None,
    ):
        if reasoning_effort is not None:
            raise ValueError(
                "Avartha reasoning is configured on the hosted agent; "
                "--reasoning-effort is not supported"
            )

        # An injected provider already carries its call settings.
        config = avartha_config or (provider.config if provider is not None else None)
        if config is None:
            raise ValueError(
                "Avartha needs the hosted agent's URL (avartha_config.base_url)"
            )

        if audio_format is not None and audio_format != TELEPHONY_AUDIO_FORMAT:
            # Output is always converted to 8 kHz u-law and input is read as
            # it; any other format would be mislabelled and mistimed.
            raise ValueError(
                "Avartha supports only telephony audio (8 kHz u-law), "
                f"got {audio_format}"
            )

        super().__init__(
            tick_duration_ms,
            audio_format=audio_format,
            send_audio_instant=send_audio_instant,
        )
        self.model = model or DEFAULT_AVARTHA_MODEL
        self.reasoning_effort = reasoning_effort
        self.avartha_config = config
        # The call's language: the agent listens in it and leaves the greeting
        # to tau2 (see webrtc_start_body).
        self.language = language
        # The hosted call is live WebRTC: the agent speaks and listens on the
        # wall clock, so ticks must keep to it. Unpaced, every stall of the
        # user simulator left the caller track sending silence mid-word and
        # queued agent audio that a barge-in then discarded as unheard,
        # although the agent had already counted it as played.
        self.realtime_pacing = True

        self._input_converter = CallerAudioUpsampler(DEFAULT_AVARTHA_INPUT_SAMPLE_RATE)
        self._output_converters: dict[int, StreamingTelephonyConverter] = {}
        self._chunk_size = int(
            DEFAULT_AVARTHA_INPUT_SAMPLE_RATE * 2 * self._voip_interval_ms / 1000
        )

        self._utterance_count = 0
        self._provider = provider
        self._owns_provider = provider is None
        self._bg_loop = BackgroundAsyncLoop()
        self._connected = False
        self._end_hold_bytes = self.audio_format.bytes_per_second // 1000
        self._reset_interruption_state()

    def _reset_interruption_state(self) -> None:
        # After a cut, text and audio of the cut-off response are still in
        # flight. Text is stale until the next response starts generating,
        # audio until its first utterance starts playing.
        self._stale_text = False
        self._stale_audio = False
        # Replacement audio that arrived in the same tick as the cut. The
        # base class discards a cut tick's overflow, so it is replayed from
        # the next tick instead.
        self._audio_after_cut: list[tuple[bytes, Optional[str]]] = []
        # True from an utterance's first audio until its played caption.
        self._awaiting_playback = False
        # Lines a caller cut while they still owed their caption, by the start
        # of their text (see _owed_caption); the newest few.
        self._owed_captions: deque[str] = deque(maxlen=_OWED_CAPTIONS_KEPT)
        # See _hold_utterance_end.
        self._held_end: list[tuple[bytes, Optional[str]]] = []
        # The call's audio clock binding (AgentAudioTimelineEvent); a second,
        # different one makes word times unusable for the rest of the call.
        self._audio_clock: Optional[AudioClockBinding] = None
        self._audio_clock_conflict = False
        self._warned_unclocked = False
        # Whether the receiver has sent audio it could not place on the clock.
        self._unclocked_audio = False
        # Pipecat's RTVI observer builds bot-transcription from bot-llm-text
        # and only empties its buffer when it sends a sentence, so a reply's
        # unsent tail (cut off, or unpunctuated) opens the next reply's first
        # sentence. This mirrors that buffer to strip it.
        self._llm_unsent = ""
        self._llm_carryover = ""
        # The utterance new agent audio belongs to, open from its first audio
        # until its played caption or a cut, and the bytes each utterance still
        # has queued, in arrival order. See _label_utterances.
        self._utterance_id: Optional[str] = None
        self._utterance_audio: deque[list] = deque()
        # The utterance whose audio a cut in this tick discarded. The cut tick
        # plays none of it, but it carries the truncation, so it keeps that ID.
        self._cut_utterance_id: Optional[str] = None
        # The utterance still streaming when this tick's cut came. Its stale
        # audio arriving later in the same tick shows the cut stopped it.
        self._open_at_cut: Optional[str] = None
        # Whether agent audio arrived in this tick and in the one before. See
        # _streaming_utterance.
        self._audio_in_tick = False
        self._audio_in_last_tick = False
        # RTP position where the audio received so far ends, when the receiver
        # placed it on the clock; and the open utterance's last timed word:
        # its sound's end and its end in the transcript's text. See
        # _continues_captioned_line.
        self._audio_rtp_end: Optional[int] = None
        self._last_timed_word: Optional[tuple[int, int]] = None
        self._caption_tail: Optional[_CaptionTail] = None
        # Captioned utterances whose text has not all been shown, oldest
        # first: ID and where the text starts and ends; and how much text had
        # shown when the last tick was labelled. See _label_utterances.
        self._captioned_lines: deque[tuple[str, int, int]] = deque()
        self._shown_at_label = 0
        # Utterances with played audio: a cut records a truncation only if it
        # stopped one of them (see _async_run_tick).
        self._played_lines: set[str] = set()
        # Whether the last tick played speech: whether a cut can have caught
        # a word mid-sound (_HeardTranscript.interrupt).
        self._speech_in_last_tick = False

    @property
    def provider(self) -> AvarthaPipecatProvider:
        """Get the provider, creating it lazily."""

        if self._provider is None:
            # aiortc ships in the live extra; only a real call needs it.
            from tau2.voice.audio_native.avartha_pipecat.webrtc import (
                AvarthaWebRTCProvider,
            )

            self._provider = AvarthaWebRTCProvider(
                self.avartha_config, language=self.language
            )
        return self._provider

    @property
    def is_connected(self) -> bool:
        """Whether both the adapter and background loop are active."""

        return self._connected and self._bg_loop.is_running

    def connect(
        self,
        system_prompt: str,
        tools: List[Tool],
        vad_config: Any = None,
        modality: str = "audio",
    ) -> None:
        """Connect to Avartha and establish a ready, tool-aware session."""

        if modality != "audio":
            raise ValueError("Avartha currently supports audio modality only")
        if self._connected:
            self.disconnect()

        try:
            self._bg_loop.start()
            timeout = (
                self.avartha_config.boot_attempts
                * (
                    self.avartha_config.connect_timeout_seconds
                    + self.avartha_config.boot_timeout_seconds
                )
                + self.avartha_config.tool_registration_timeout_seconds
                + 5
            )
            self._bg_loop.run_coroutine(
                self._async_connect(system_prompt, tools),
                timeout=timeout,
            )
            self._connected = True
            logger.info(
                "DiscreteTimeAvarthaPipecatAdapter connected "
                f"({self.avartha_config.base_url}, tick={self.tick_duration_ms}ms)"
            )
        except Exception as exc:
            if (
                self._bg_loop.is_running
                and self._provider is not None
                and self._provider.is_connected
            ):
                try:
                    self._bg_loop.run_coroutine(
                        self._provider.disconnect(),
                        timeout=DEFAULT_AUDIO_NATIVE_DISCONNECT_TIMEOUT,
                    )
                except Exception as cleanup_exc:
                    logger.debug(f"Avartha connect cleanup failed: {cleanup_exc}")
            self._bg_loop.stop()
            if isinstance(exc, AvarthaProtocolError):
                raise
            raise RuntimeError(f"Failed to connect to Avartha: {exc}") from exc

    async def _async_connect(
        self,
        system_prompt: str,
        tools: List[Tool],
    ) -> None:
        for attempt in range(1, self.avartha_config.boot_attempts + 1):
            try:
                await self.provider.connect(system_prompt=system_prompt)
                break
            except AvarthaBootError as exc:
                await self.provider.disconnect()
                if attempt == self.avartha_config.boot_attempts:
                    logger.error(
                        f"Avartha boot failed {attempt} time(s), giving up: {exc}"
                    )
                    raise
                logger.warning(
                    "Avartha boot failed "
                    f"(attempt {attempt}/{self.avartha_config.boot_attempts}): "
                    f"{exc}; retrying"
                )
                self.provider.reset_for_retry()

        schemas = [_mcp_tool_schema(tool) for tool in tools]
        await self.provider.register_tools(schemas)

    def disconnect(self) -> None:
        """Close the call and reset tick/audio state."""

        if self._connected and self._bg_loop.is_running:
            try:
                self._bg_loop.run_coroutine(
                    self._async_disconnect(),
                    timeout=DEFAULT_AUDIO_NATIVE_DISCONNECT_TIMEOUT,
                )
            except Exception as exc:
                logger.warning(f"Error disconnecting Avartha: {exc}")

        self._bg_loop.stop()
        self._connected = False
        self._tick_count = 0
        self._cumulative_user_audio_ms = 0
        transcript = self._utterance_transcripts.get(_ITEM_ID)
        # These counters classify timed marks initially, not their final placement.
        timed = (
            transcript.timed_words if isinstance(transcript, _HeardTranscript) else 0
        )
        untimed = (
            transcript.untimed_words if isinstance(transcript, _HeardTranscript) else 0
        )
        if self._audio_clock is not None or timed or untimed:
            logger.info(
                "Avartha word timing metadata: {} marks initially accepted for timing, "
                "{} timed marks without a usable audio position",
                timed,
                untimed,
            )
        self.clear_buffers()
        self._input_converter.reset()
        for converter in self._output_converters.values():
            converter.reset()
        self._output_converters.clear()
        self._reset_interruption_state()
        logger.info("DiscreteTimeAvarthaPipecatAdapter disconnected")

    async def _async_disconnect(self) -> None:
        if self._provider is not None and self._provider.is_connected:
            # Results of tools tau2 already executed are still queued for the
            # next tick, which will not come. Deliver them before hanging up.
            try:
                await self._flush_pending_tool_results()
            except Exception as exc:
                logger.warning(f"Avartha tool results lost at shutdown: {exc}")
        if self._owns_provider and self._provider is not None:
            await self.provider.disconnect()

    def run_tick(
        self,
        user_audio: bytes,
        tick_number: Optional[int] = None,
    ) -> TickResult:
        """Exchange one simulation tick of audio and events."""

        if not self.is_connected:
            raise RuntimeError("Not connected to Avartha. Call connect() first.")
        if tick_number is None:
            tick_number = self._tick_count
        self._tick_count = tick_number + 1
        return self._bg_loop.run_coroutine(
            self._async_run_tick(user_audio, tick_number),
            timeout=(
                self.tick_duration_ms / 1000 + DEFAULT_AUDIO_NATIVE_TICK_TIMEOUT_BUFFER
            ),
        )

    async def _flush_pending_tool_results(self) -> None:
        """Return all queued tau2 tool results before sending tick audio."""

        for (
            call_id,
            result,
            request_response,
            is_error,
        ) in self._pending_tool_results:
            await self.provider.send_tool_result(
                call_id,
                result,
                request_response=request_response,
                is_error=is_error,
            )
        self._pending_tool_results.clear()

    async def _async_run_tick(self, user_audio: bytes, tick_number: int) -> TickResult:
        """Retain post-interruption audio and utterance tails after tick buffering."""

        result = await super()._async_run_tick(user_audio, tick_number)
        # The audio the base class has just counted as played (its text
        # taken); it still carries the transcript's ID (see _label_utterances).
        played = b"".join(
            data for data, item_id in result.agent_audio_chunks if item_id == _ITEM_ID
        )
        self._speech_in_last_tick = self._transcript().note_played(played)
        started = self._transcript().take_started()
        if started is not None:
            result = AvarthaTickResult.with_started_word(result, started)
        kept = self._transcript().take_kept()
        if kept:
            # Timed words whose sound played before this tick's cut (see
            # _HeardTranscript.interrupt). The cut tick plays no audio, so the
            # base class took no text. It is labelled as any cut tick is: with
            # the utterance the cut discarded audio of, if any.
            result.proportional_transcript += kept
        if self._audio_after_cut:
            self._buffered_agent_audio[:0] = self._audio_after_cut
            self._audio_after_cut = []
        self._buffered_agent_audio.extend(self._held_end)
        self._held_end = []
        cut_line = self._cut_utterance_id
        self._label_utterances(result)
        if (
            result.was_truncated
            and cut_line not in self._played_lines
            and not result.proportional_transcript.strip()
        ):
            # Caller speech stopped no line the caller had heard: nothing was
            # playing, or the cut discarded a reply before any of its audio
            # played (see _label_utterances). The tick still discarded what it
            # had to, and counts it in truncated_audio_bytes, but records no
            # truncation. Readers attribute a truncated tick to the agent speech
            # before it (utils/transcript/live.py marks the latest speech span
            # cut off), and these ticks follow every caller turn start. A tick
            # that shows text the cut kept stays truncated: it is its own
            # utterance on the tick path.
            result.was_truncated = False
        self._audio_in_last_tick, self._audio_in_tick = self._audio_in_tick, False
        return result

    def _label_utterances(self, result: TickResult) -> None:
        """Tag this tick's agent audio with the Avartha utterance it plays.

        tau2 groups agent ticks into utterances by these IDs (judge clips and
        turns, transcript views, communicate scoring), and a tick without one
        stands alone. One ID for the whole call left only silent ticks to split
        on, so a pause inside a sentence cut it in two and a judge clip ran a
        word or two past its text. Each utterance gets its own ID from its
        first audio until its played caption or a cut; a tick inside it that
        plays no audio still carries it, and so does the silent tick of a cut
        that discarded its audio. Without that ID the cut tick stood alone, so
        its truncation never reached the utterance it cut: judges saw a reply cut
        mid-word with no cut-off marker. A tick that plays two utterances goes
        to the one it plays most of. Labels are set after the tick's text is
        taken, so text and audio are unchanged.

        A tick that shows text of a captioned line and nothing past it is that
        line's, whatever audio it plays. Its last word can show on a tick that
        plays mostly audio queued after the caption (tail audio past the tail
        window, or the next line's head), or on a tick that plays nothing; by
        audio alone that tick opened a group of its own, so the line was judged
        without its last word and the word stood alone as a reply.
        """

        cut, self._cut_utterance_id = self._cut_utterance_id, None
        self._open_at_cut = None
        line = self._captioned_line_shown(result)
        played = sum(len(chunk) for chunk, _ in result.agent_audio_chunks)
        if played == 0:
            if result.was_truncated:
                if cut is not None:
                    result.agent_audio_chunks.append((b"", cut))
            elif line is not None or self._utterance_id is not None:
                result.agent_audio_chunks.append((b"", line or self._utterance_id))
            return
        shares: dict[str, int] = {}
        remaining = played
        while remaining and self._utterance_audio:
            entry = self._utterance_audio[0]
            taken = min(remaining, entry[1])
            shares[entry[0]] = shares.get(entry[0], 0) + taken
            entry[1] -= taken
            remaining -= taken
            if entry[1] == 0:
                self._utterance_audio.popleft()
        label = line or (
            max(shares, key=shares.__getitem__)
            if shares
            else self._utterance_id or _ITEM_ID
        )
        result.agent_audio_chunks = [
            (chunk, label) for chunk, _ in result.agent_audio_chunks
        ]
        self._played_lines.update(shares)
        self._played_lines.add(label)

    def _captioned_line_shown(self, result: TickResult) -> Optional[str]:
        """The captioned line whose text holds all the text this tick shows.

        A cut tick is labelled by what it cut instead (and a cut forgets the
        captioned lines). A line is forgotten once all its text has shown.
        Known limit: a tick that shows a line's last word and the next line's
        first is labelled by audio, so the last word can still go to the next
        line.
        """

        shown = self._transcript().shown_chars
        before, self._shown_at_label = self._shown_at_label, shown
        lines = self._captioned_lines
        line_id = None
        if not result.was_truncated and result.proportional_transcript.strip():
            for candidate, start, end in lines:
                if start <= before and shown <= end:
                    line_id = candidate
                    break
        while lines and lines[0][2] <= shown:
            lines.popleft()
        return line_id

    def _discarded_utterance(self) -> Optional[str]:
        """The utterance whose audio a cut now discards, if any.

        Queued audio is unplayed, oldest first, so the first utterance with
        more than the held 1 ms tail left is the one the caller cut. A reply
        that has played all its audio only waits for its caption: caller
        speech after it cuts nothing, and the reply is not marked cut off.
        "Played all" is judged by audio arrival (see _streaming_utterance): a
        reply whose last audio arrived this tick or the last still counts as
        streaming, so a cut that keeps the word it caught (or meets the
        reply's stale audio) marks it cut off, even if that word ended in
        trailing silence the receiver had not yet reported.
        Audio of a reply streamed at playback pace can reach the adapter after
        the cut in the same tick; that stale audio names the reply instead
        (see _streaming_utterance). A cut in a TTS pause, with no audio
        queued or arriving, is deliberately unmarked even though unspoken
        text is dropped: the caller cut no audio there.
        Nor does a cut in a captioned line's quiet tail: audio queued to the
        line after its caption that starts at or after the end of its last
        word's sound (see _CaptionTail) carries none of its words. Known
        limits: quiet audio that arrived before the caption still counts as
        the line's (judged by arrival, as above); and after an overtaking
        caption a gapless next line's head inside the tail window is counted
        quiet to the captioned line (see _continues_captioned_line), so a cut
        discarding only that head marks no line.
        """

        for utterance_id, remaining, quiet in self._utterance_audio:
            if remaining > self._end_hold_bytes and not quiet:
                return utterance_id
        return None

    def _streaming_utterance(self) -> Optional[str]:
        """The utterance whose audio was still arriving when a cut came, if any.

        A reply streamed at playback pace has only its held tail queued at a
        cut, so _discarded_utterance finds nothing; if its stale audio then
        arrives in the cut tick, the cut stopped it (see the AgentAudioEvent
        branch). Such a reply receives audio every tick. One that received
        none in this tick or the last has finished and only waits for its
        played caption, or is in a TTS pause; audio arriving after the cut
        does not show the caller cut it, so it is not named.
        """

        if self._audio_in_tick or self._audio_in_last_tick:
            return self._utterance_id
        return None

    def _queue_utterance_audio(
        self, num_bytes: int, utterance_id: Optional[str] = None, quiet: bool = False
    ) -> None:
        """Count arriving audio toward its utterance: ``utterance_id``, else the
        open one, opening one if none is. ``quiet`` audio is a captioned line's
        tail past its last word (see _discarded_utterance)."""

        if utterance_id is None:
            if self._utterance_id is None:
                self._utterance_count += 1
                self._utterance_id = f"{_ITEM_ID}-{self._utterance_count}"
                self._transcript().begin_line()
            utterance_id = self._utterance_id
        last = self._utterance_audio[-1] if self._utterance_audio else None
        if last is not None and last[0] == utterance_id and last[2] == quiet:
            last[1] += num_bytes
        else:
            self._utterance_audio.append([utterance_id, num_bytes, quiet])

    def _continues_captioned_line(self, event: AgentAudioEvent) -> bool:
        """Whether ``event`` is the rest of the line whose caption came already.

        Avartha's played caption travels on the data channel and can overtake
        the line's last audio, or come with it while the line's quiet tail is
        still on its way. That audio used to open a new utterance; the tick
        playing the line's last word went to it, so the line was judged
        against text missing that word and the word stood alone as a reply.
        So audio that follows the line's without a gap on the RTP clock and
        starts within ``_CAPTION_TAIL_RTP`` of the line's last timed word end
        is the line's, until the next line's text arrives. If that text came
        before the caption, only a caption that overtook the line's audio
        opens the window. Without timing every new audio opens an utterance,
        as before. Known limit: after an overtaking caption, a next line that
        follows without a gap and starts within the window has its head
        counted to the captioned line.
        """

        tail = self._caption_tail
        return (
            tail is not None
            and self._utterance_id is None
            and event.rtp_start is not None
            and event.rtp_start == self._audio_rtp_end
            and event.rtp_start < tail.end_rtp
        )

    async def _execute_tick(
        self,
        user_audio: bytes,
        tick_number: int,
        result: TickResult,
        tick_start: float,
    ) -> None:
        """Convert audio, exchange provider events, and populate the tick."""

        provider_audio = self._input_converter.convert_input(user_audio)

        async def send_audio(chunk: bytes) -> None:
            await self.provider.send_audio(
                chunk,
                sample_rate=DEFAULT_AVARTHA_INPUT_SAMPLE_RATE,
            )

        async def receive_events() -> list[Any]:
            elapsed = asyncio.get_running_loop().time() - tick_start
            remaining = max(0.01, self.tick_duration_ms / 1000 - elapsed)
            return await self.provider.receive_events_for_duration(remaining)

        tasks = [
            asyncio.create_task(
                self._send_audio_chunked(
                    provider_audio,
                    send_audio,
                    self._chunk_size,
                )
            ),
            asyncio.create_task(receive_events()),
        ]
        try:
            _, events = await asyncio.gather(*tasks)
        finally:
            # gather does not cancel its sibling when audio sending fails.
            # Drain both before disconnect can stop their background loop.
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        # Sending can outlast the receive window, especially when a paced
        # tick is behind its wall-clock deadline. Include events that arrived
        # before send completion instead of delaying them until the next tick.
        events.extend(self.provider.drain_pending_events())
        for event in events:
            self._process_event(result, event)
        self._apply_decoded_silence()
        self._hold_utterance_end(result)

    def _apply_decoded_silence(self) -> None:
        """Place timed words that ended in the silence the receiver dropped.

        The receiver drops digital silence, so a word ending in the silence
        after a reply's audio would otherwise wait for audio that never comes;
        untimed, it airs on arrival.
        """

        silence = self.provider.rtp_silence
        if silence is not None:
            self._transcript().decoded_silence(*silence)

    def _hold_utterance_end(self, result: TickResult) -> None:
        """Keep the last millisecond of an unfinished utterance for later.

        The end of a line's text is released by its played caption. If that
        arrives after the line's audio has all been played, its text would
        land on a tick with no audio, and the user simulator drops text from
        silent ticks. Holding back 1 ms until the caption arrives makes the
        last text always ride on audio.
        """

        if not self._awaiting_playback or result.was_truncated:
            return
        available = sum(len(chunk) for chunk, _ in result.agent_audio_chunks)
        if available == 0 or available > self.bytes_per_tick:
            return  # nothing to play, or the tail is buffered anyway
        hold = min(available, self._end_hold_bytes)
        data, item_id = result.agent_audio_chunks.pop()
        while len(data) < hold:
            previous, item_id = result.agent_audio_chunks.pop()
            data = previous + data
        if len(data) > hold:
            result.agent_audio_chunks.append((data[:-hold], item_id))
        self._held_end = [(data[-hold:], item_id)]

    def _transcript(self) -> _HeardTranscript:
        if _ITEM_ID not in self._utterance_transcripts:
            # Duck-typed: the base class only calls get_transcript_for_audio.
            self._utterance_transcripts[_ITEM_ID] = _HeardTranscript(  # type: ignore[assignment]
                due_slack_bytes=self._end_hold_bytes,
                sample_rate=self.audio_format.sample_rate,
            )
        return self._utterance_transcripts[_ITEM_ID]  # type: ignore[return-value]

    def _owed_caption(self, caption: str) -> bool:
        """Whether ``caption`` is owed by a line a caller cut; consumes it.
        Checked for every caption, so the owed list follows playback order.

        Avartha's captions name no utterance, so a cut line's caption is
        known by its text: it starts as the line's text did. An agent cutting
        its own line sends a cut caption no caller cut owes, with its own
        text, so it is not mistaken for one, even if a cut line's caption
        never came."""

        key = _caption_key(caption)
        for index, line in enumerate(self._owed_captions):
            if _same_opening(key, line):
                del self._owed_captions[index]
                return True
        if self._owed_captions:
            transcript = self._transcript()
            if not _same_opening(
                key, _caption_key(transcript.text[transcript.utterance_start :])
            ):
                # Not the open line's either: a cut line's caption that starts
                # otherwise than its text did (a code spelled out).
                self._owed_captions.popleft()
                return True
        # Captions come in playback order: one no cut line owes means those
        # still owed will not come (the cut line sent none), and a later line
        # opening the same way must not be taken for one of them.
        self._owed_captions.clear()
        return False

    def _note_timed_word(self, transcript: _HeardTranscript, timed_before: int) -> None:
        """Track the open line's last timed word, if the transcript just
        placed one on the clock (see _continues_captioned_line)."""

        if transcript.timed_words > timed_before:
            self._last_timed_word = transcript.last_timed_word

    def _word_end_rtp(self, event: AgentWordEvent) -> Optional[int]:
        """The RTP timestamp where ``event``'s word ends, if Avartha timed it
        on the call's audio clock."""

        clock = self._audio_clock
        if (
            event.end is None
            or clock is None
            or self._audio_clock_conflict
            or event.end.stream_id != clock.stream_id
        ):
            return None
        if (
            self._unclocked_audio
            and not self._transcript().clocked
            and not self._warned_unclocked
        ):
            self._warned_unclocked = True
            logger.warning(
                "Avartha times its words but this receiver cannot place its "
                "audio on the RTP clock (aiortc's RTP origin unavailable, or "
                "the first frame not at it); words are placed on arrival"
            )
        return clock.rtp_timestamp_of(event.end.sample_offset)

    def _process_event(self, result: TickResult, event: Any) -> None:
        result.events.append(event)
        if not isinstance(event, AgentAudioEvent):
            logger.debug(
                f"Avartha event: {event!r} stale_text={self._stale_text} "
                f"stale_audio={self._stale_audio}"
            )

        if isinstance(event, AgentAudioEvent):
            converter = self._output_converters.setdefault(
                event.sample_rate,
                StreamingTelephonyConverter(
                    input_sample_rate=DEFAULT_AVARTHA_INPUT_SAMPLE_RATE,
                    output_sample_rate=event.sample_rate,
                ),
            )
            telephony_audio = converter.convert_output(event.audio)
            if telephony_audio:
                if self._stale_audio:
                    result.truncated_audio_bytes += len(telephony_audio)
                    if self._cut_utterance_id is None:
                        self._cut_utterance_id = self._open_at_cut
                    return
                chunk = (telephony_audio, _ITEM_ID)
                if self._continues_captioned_line(event):
                    # Its caption came already: nothing to hold back for.
                    tail = self._caption_tail
                    self._queue_utterance_audio(
                        len(telephony_audio),
                        tail.line_id,
                        quiet=tail.quiet_from_rtp is not None
                        and event.rtp_start >= tail.quiet_from_rtp,
                    )
                else:
                    self._caption_tail = None
                    self._queue_utterance_audio(len(telephony_audio))
                    self._awaiting_playback = True
                self._audio_in_tick = True
                if result.was_truncated:
                    self._audio_after_cut.append(chunk)
                else:
                    result.agent_audio_chunks.append(chunk)
                if event.rtp_start is None:
                    self._unclocked_audio = True
                rtp_ticks = len(event.audio) // 2 * RTP_CLOCK_HZ
                self._audio_rtp_end = (
                    event.rtp_start + rtp_ticks // event.sample_rate
                    if event.rtp_start is not None
                    and rtp_ticks % event.sample_rate == 0
                    else None
                )
                self._transcript().add_audio(
                    len(telephony_audio),
                    rtp_start=(
                        event.rtp_start if rtp_ticks % event.sample_rate == 0 else None
                    ),
                    rtp_samples=rtp_ticks // event.sample_rate,
                    decoded_from=event.rtp_decoded_from,
                )
        elif isinstance(event, AgentAudioTimelineEvent):
            if self._audio_clock is None:
                self._audio_clock = event.binding
            elif event.binding != self._audio_clock and not self._audio_clock_conflict:
                # Another epoch on the same call: which RTP clock the received
                # audio follows is no longer known.
                logger.warning(
                    "Avartha changed its audio clock binding mid-call; word "
                    "times are ignored from now on"
                )
                self._audio_clock_conflict = True
                self._transcript().untime_waiting()
        elif isinstance(event, ToolCallEvent):
            result.tool_calls.append(
                ToolCall(
                    id=event.id,
                    name=event.name,
                    arguments=event.arguments,
                    requestor="assistant",
                )
            )
            logger.info(f"Avartha tool call: {event.name}({event.arguments})")
        elif isinstance(event, AgentLLMTextEvent):
            self._llm_unsent += event.text
        elif isinstance(event, TranscriptEvent):
            text = event.text
            carryover, self._llm_carryover, self._llm_unsent = (
                self._llm_carryover,
                "",
                "",
            )
            if carryover and text.startswith(carryover):
                text = text[len(carryover) :]
            if self._stale_text:
                return
            if text.strip():
                # The next line's text: audio from here on is not the
                # captioned line's tail.
                self._caption_tail = None
                transcript = self._transcript()
                timed_before = transcript.timed_words
                # Places words spoken before this text came (add_word).
                transcript.add_text(text)
                self._note_timed_word(transcript, timed_before)
                logger.debug(f"Avartha transcript: {text!r}")
        elif isinstance(event, AgentWordEvent):
            if not self._stale_audio:
                end_rtp = self._word_end_rtp(event)
                transcript = self._transcript()
                timed_before = transcript.timed_words
                transcript.add_word(event.text, end_rtp)
                self._note_timed_word(transcript, timed_before)
        elif isinstance(event, AgentPlaybackEvent):
            owed = self._owed_caption(event.text)
            if owed and event.cut:
                # The caption of a line a caller cut. It usually comes in the
                # cut tick, but can come after the next line started (after
                # its AgentStartedSpeakingEvent cleared _stale_audio); read
                # then, it put the cut line's text back into the next line's
                # (seen live: a whole cut reply was shown again).
                # interrupt() already kept what the caller heard.
                return
            if not self._stale_audio:
                self._awaiting_playback = False
                self._caption_tail = None
                captioned, self._utterance_id = self._utterance_id, None
                last_word, self._last_timed_word = self._last_timed_word, None
                transcript = self._transcript()
                start = transcript.utterance_start
                # Still timed: the receiver may have lost its sound since.
                last_timed = last_word is not None and transcript.is_timed(last_word[1])
                caption_end = transcript.add_playback(event.text)
                if event.cut and event.text.strip():
                    # No caller cut owed this caption: the agent cut its own line.
                    transcript.drop_unplayed_reply(caption_end)
                if captioned is None:
                    return
                # The utterance's audio ran up to this caption, so the cursor
                # (not caption_end, which a late caption keeps behind it) ends
                # the text it labels.
                end = transcript.matched_chars
                self._captioned_lines.append((captioned, start, end))
                if last_word is not None and self._audio_rtp_end is not None:
                    last_end = unwrap_rtp(last_word[0], self._audio_rtp_end)
                    lead = last_end - self._audio_rtp_end
                    # With the next line's text already here, its audio may
                    # follow without a gap: only audio the caption overtook
                    # is surely the line's.
                    next_text = transcript.text[end:].strip()
                    if lead <= _CAPTION_MAX_LEAD_RTP and (lead > 0 or not next_text):
                        # Past the last timed word is the quiet tail only if no
                        # word of the line follows it (its mark may be dropped).
                        ends_line = last_timed and not re.sub(
                            r"[\W_]", "", transcript.text[last_word[1] : end]
                        )
                        self._caption_tail = _CaptionTail(
                            captioned,
                            last_end + _CAPTION_TAIL_RTP,
                            last_end if ends_line else None,
                        )
        elif isinstance(event, AgentLLMStartedEvent):
            self._transcript().begin_reply()
            self._stale_text = False
            self._llm_carryover = self._llm_unsent
        elif isinstance(event, AgentStartedSpeakingEvent):
            # Subsequent stale audio belongs to this replacement, not to an
            # earlier reply that was merely waiting for its played caption.
            self._open_at_cut = None
            self._caption_tail = None
            if self._stale_audio:
                self._stale_audio = False
                for converter in self._output_converters.values():
                    converter.reset()
        elif isinstance(event, SpeechEvent):
            result.vad_events.append(event.kind)
            if event.kind in {"speech_started", "interrupted"}:
                # The line still streaming at the cut (audio this tick or the
                # last, as _streaming_utterance): open, or captioned with its
                # audio still arriving. A line that has played all its audio is
                # not cut off.
                speaking = (
                    (
                        self._utterance_id
                        or (self._caption_tail.line_id if self._caption_tail else None)
                    )
                    if self._audio_in_tick or self._audio_in_last_tick
                    else None
                )
                self._caption_tail = None
                self._captioned_lines.clear()
                self._last_timed_word = None
                result.was_truncated = True
                result.truncated_audio_bytes += sum(
                    len(audio) for audio, _item_id in result.agent_audio_chunks
                )
                result.agent_audio_chunks.clear()
                self._buffered_agent_audio.clear()
                self._audio_after_cut.clear()
                self._held_end = []
                if self._awaiting_playback or self._utterance_id is not None:
                    transcript = self._transcript()
                    line = _caption_key(transcript.text[transcript.utterance_start :])
                    if line:
                        self._owed_captions.append(line)
                self._awaiting_playback = False
                if self._cut_utterance_id is None:
                    self._cut_utterance_id = self._discarded_utterance()
                    self._open_at_cut = self._open_at_cut or self._streaming_utterance()
                self._utterance_id = None
                self._utterance_audio.clear()
                self._apply_decoded_silence()
                # Only a line still streaming, with speech up to the cut, can
                # have a word caught mid-sound.
                self._transcript().interrupt(
                    speech_before_cut=speaking is not None and self._speech_in_last_tick
                )
                if (
                    self._transcript().kept_caught_word
                    or self._transcript().caught_started_word
                ) and self._cut_utterance_id is None:
                    # The cut stopped a streaming line mid-word, so it cut that
                    # line off even with none of its audio queued; the cut
                    # tick, which carries the kept word, belongs to it. Known
                    # limit: this is the line streaming at the cut, not
                    # necessarily the kept word's line, and with none
                    # streaming the kept word's tick stays unlabelled. A
                    # started word is only caught on the line streaming.
                    self._cut_utterance_id = speaking
                self._stale_text = self._stale_audio = True
                for converter in self._output_converters.values():
                    converter.reset()
        elif isinstance(event, ProviderErrorEvent):
            raise RuntimeError(f"Avartha reader failed: {event.message}")
