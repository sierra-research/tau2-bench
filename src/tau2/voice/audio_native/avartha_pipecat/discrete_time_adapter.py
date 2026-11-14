"""Discrete-time adapter for a hosted Avartha voice agent over Pipecat RTVI."""

from __future__ import annotations

import asyncio
import html
import math
import re
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass
from typing import Any, List, Optional

from loguru import logger

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
    AudioClockBinding,
    unwrap_rtp,
)
from tau2.voice.audio_native.avartha_pipecat.caller_audio import CallerAudioUpsampler
from tau2.voice.audio_native.avartha_pipecat.config import AvarthaPipecatConfig
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
        # Where the utterance now playing starts in ``text``.
        self._utterance_start = 0
        self._spoken_chars = 0
        self._shown_chars = 0
        # (audio position, text length aired once playback reaches it)
        self._marks: deque[tuple[int, int]] = deque()

    def add_text(self, text: str) -> None:
        """Append one bot-transcription sentence, keeping its own spacing.

        Avartha's splitter also breaks at decimal points ("$13." then
        "46"), and marks real sentence starts with a leading space. Only the
        first sentence of a response lacks one.
        """

        text = _unescape(text)
        if not self.text.strip():
            text = text.lstrip()
        elif not self.text[-1].isspace() and not text[0].isspace():
            if not (text[0].isdigit() and self.text[-1] in ".,"):
                text = " " + text
        self.text += text
        self._sentence_ends.append(len(self.text))

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

    def add_playback(self, caption: str) -> None:
        """Avartha's caption of what this utterance actually played.

        Text in the caption that the sentences lack (speech no LLM sentence
        produced, or a reply's unpunctuated tail) is inserted where it aired.
        """

        self._pending_piece = ""
        caption = " ".join(_unescape(caption).split())
        if not caption:
            return
        start = self._utterance_start
        while start < len(self.text) and self.text[start].isspace():
            start += 1
        index = self.text.lower().find(caption.lower(), start)
        if index >= 0:
            self._advance(index + len(caption))
        else:
            aired = self.text[start:].lower()
            common = 0
            while (
                common < len(caption)
                and common < len(aired)
                and aired[common] == caption[common].lower()
            ):
                common += 1
            at = start + common
            missing = caption[common:]
            if at > 0 and not self.text[at - 1].isspace() and not missing[0].isspace():
                missing = " " + missing if common == 0 else missing
            if self.text[at:].strip() and not missing[-1].isspace():
                missing += " "
            logger.debug(f"Avartha playback adds unsent text: {missing!r}")
            self.text = self.text[:at] + missing + self.text[at:]
            self._sentence_ends = sorted(
                [e if e <= at else e + len(missing) for e in self._sentence_ends]
                + [at + len(missing.rstrip())]
            )
            self._advance(at + len(missing.rstrip()))
        self._utterance_start = self._match_cursor

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
        self._marks.clear()

    def get_transcript_for_audio(self, audio_bytes_to_play: int) -> str:
        self._audio_played += audio_bytes_to_play
        while self._marks and self._marks[0][0] <= self._audio_played + self._due_slack:
            self._spoken_chars = max(self._spoken_chars, self._marks.popleft()[1])
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


class _HeardTranscript(_SpokenTranscript):
    """``_SpokenTranscript`` with each timed word placed where its sound ends.

    Over WebRTC a word event reaches the adapter on the data channel, on its
    own clock, so its arrival says nothing about its audio: a word heard
    before a cut but reported in the cut tick looked the same as one whose
    audio the cut discarded. Avartha now times words on the audio's RTP clock
    (``audio_timing.py``) and the receiver keeps each audio event's RTP
    position, so a timed word airs once playback has passed the end of its
    sound, and a cut keeps exactly the words whose sound ended in the audio
    played.

    A word whose end is unknown (no binding, another clock epoch, media lost
    or the receiver keeps no timestamps) is placed as ``_SpokenTranscript``
    places every word: where the audio received stood on its arrival. With no
    timed words the two classes behave identically.
    """

    def __init__(self, due_slack_bytes: int = 0) -> None:
        super().__init__(due_slack_bytes)
        self._spans: list[_AudioSpan] = []
        self._span_starts: list[int] = []
        # Whether the receiver has placed any of this call's audio on the RTP
        # clock; until it does, timed words cannot be placed.
        self._clocked = False
        # RTP timestamp of the end of the word being matched (add_word).
        self._word_end: Optional[int] = None
        # Text lengths whose mark is timed, and timed marks whose audio has
        # not arrived yet: length -> (RTP timestamp, arrival position).
        self._timed: set[int] = set()
        self._waiting: dict[int, tuple[int, int]] = {}

    def add_audio(
        self,
        num_bytes: int,
        rtp_start: Optional[int] = None,
        rtp_samples: int = 0,
        decoded_from: Optional[int] = None,
    ) -> None:
        if rtp_start is not None and rtp_samples > 0:
            self._clocked = True
            if self._spans and rtp_start < self._spans[-1].rtp_end:
                # Not after the audio before it: the clock is not one we can
                # follow, so stop timing from it.
                self._spans.clear()
                self._span_starts.clear()
            else:
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
        super().add_audio(num_bytes)
        if self._waiting:
            self._place_waiting()

    def add_word(self, word: str, end_rtp_timestamp: Optional[int] = None) -> None:
        self._word_end = end_rtp_timestamp
        try:
            super().add_word(word)
        finally:
            self._word_end = None

    def _advance(self, cursor: int) -> None:
        if self._word_end is None or cursor <= self._match_cursor:
            super()._advance(cursor)
            return
        self._match_cursor = cursor
        position = self._position(self._word_end)
        if position is None:
            self._marks.append((self._audio_received, cursor))
            return
        self._timed.add(cursor)
        if position is _NOT_YET_RECEIVED:
            # Its sound has not arrived yet: nothing after it airs before it.
            self._waiting[cursor] = (self._word_end, self._audio_received)
            self._marks.append((math.inf, cursor))
        else:
            # Exact: the end-of-utterance slack is for untimed marks only.
            self._marks.append((position + self._due_slack, cursor))

    def _position(self, rtp_timestamp: int) -> Any:
        """Received-byte position where the sound at ``rtp_timestamp`` has played.

        ``_NOT_YET_RECEIVED`` if that audio has not arrived yet, None if it is
        unknown (no timed audio, lost media or before the first audio).
        """

        if not self._spans:
            # Audio comes after this word (a new reply, or after a cut).
            return _NOT_YET_RECEIVED if self._clocked else None
        last = self._spans[-1]
        end = unwrap_rtp(rtp_timestamp, last.rtp_end)
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
        placed: dict[int, float] = {}
        for cursor, (rtp_timestamp, arrival) in list(self._waiting.items()):
            position = self._position(rtp_timestamp)
            if position is _NOT_YET_RECEIVED:
                continue
            del self._waiting[cursor]
            if position is None:
                self._timed.discard(cursor)
                placed[cursor] = arrival
            else:
                placed[cursor] = position + self._due_slack
        if placed:
            self._marks = deque(
                (placed.get(cursor, position), cursor)
                for position, cursor in self._marks
            )

    def add_playback(self, caption: str) -> None:
        # The utterance has ended; a timed word whose audio never arrived is
        # placed as it would be untimed.
        if self._waiting:
            for cursor in self._waiting:
                self._timed.discard(cursor)
            arrivals = {
                cursor: arrival for cursor, (_, arrival) in self._waiting.items()
            }
            self._waiting.clear()
            self._marks = deque(
                (arrivals.get(cursor, position), cursor)
                for position, cursor in self._marks
            )
        super().add_playback(caption)

    def interrupt(self) -> None:
        """Keep the timed words whose sound ended in the audio played."""

        heard = max(
            (
                cursor
                for position, cursor in self._marks
                if cursor in self._timed
                and position - self._due_slack <= self._audio_played
            ),
            default=0,
        )
        shown = self._shown_chars
        if heard > shown:
            # interrupt() keeps the text shown; keep the heard text too, and
            # let the next get_transcript_for_audio show it.
            self._shown_chars = heard
            super().interrupt()
            self._shown_chars = shown
        else:
            super().interrupt()
        self._spans.clear()
        self._span_starts.clear()
        self._timed.clear()
        self._waiting.clear()


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
        # See _hold_utterance_end.
        self._held_end: list[tuple[bytes, Optional[str]]] = []
        # The call's audio clock binding (AgentAudioTimelineEvent); a second,
        # different one makes word times unusable for the rest of the call.
        self._audio_clock: Optional[AudioClockBinding] = None
        self._audio_clock_conflict = False
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
        if result.was_truncated:
            # Timed words whose sound played before the cut (see
            # _HeardTranscript.interrupt). The cut tick plays no audio, so the
            # base class asked for no text; the label below joins it to the
            # utterance the cut ended.
            heard = self._transcript().get_transcript_for_audio(0)
            if heard:
                result.proportional_transcript += heard
        if self._audio_after_cut:
            self._buffered_agent_audio[:0] = self._audio_after_cut
            self._audio_after_cut = []
        self._buffered_agent_audio.extend(self._held_end)
        self._held_end = []
        self._label_utterances(result)
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
        """

        cut, self._cut_utterance_id = self._cut_utterance_id, None
        self._open_at_cut = None
        played = sum(len(chunk) for chunk, _ in result.agent_audio_chunks)
        if played == 0:
            if result.was_truncated:
                if cut is not None:
                    result.agent_audio_chunks.append((b"", cut))
            elif self._utterance_id is not None:
                result.agent_audio_chunks.append((b"", self._utterance_id))
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
        label = (
            max(shares, key=shares.__getitem__)
            if shares
            else self._utterance_id or _ITEM_ID
        )
        result.agent_audio_chunks = [
            (chunk, label) for chunk, _ in result.agent_audio_chunks
        ]

    def _discarded_utterance(self) -> Optional[str]:
        """The utterance whose audio a cut now discards, if any.

        Queued audio is unplayed, oldest first, so the first utterance with
        more than the held 1 ms tail left is the one the caller cut. A reply
        that has played all its audio only waits for its caption: caller
        speech after it cuts nothing, and the reply is not marked cut off.
        Audio of a reply streamed at playback pace can reach the adapter after
        the cut in the same tick; that stale audio names the reply instead
        (see _streaming_utterance). A cut in a TTS pause, with no audio
        queued or arriving, is deliberately unmarked even though unspoken
        text is dropped: the caller cut no audio there.
        """

        for utterance_id, remaining in self._utterance_audio:
            if remaining > self._end_hold_bytes:
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

    def _queue_utterance_audio(self, num_bytes: int) -> None:
        """Count arriving audio toward its utterance, opening one if none is."""

        if self._utterance_id is None:
            self._utterance_count += 1
            self._utterance_id = f"{_ITEM_ID}-{self._utterance_count}"
        if self._utterance_audio and self._utterance_audio[-1][0] == self._utterance_id:
            self._utterance_audio[-1][1] += num_bytes
        else:
            self._utterance_audio.append([self._utterance_id, num_bytes])

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
        self._hold_utterance_end(result)

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
                due_slack_bytes=self._end_hold_bytes
            )
        return self._utterance_transcripts[_ITEM_ID]  # type: ignore[return-value]

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
                self._queue_utterance_audio(len(telephony_audio))
                self._audio_in_tick = True
                self._awaiting_playback = True
                if result.was_truncated:
                    self._audio_after_cut.append(chunk)
                else:
                    result.agent_audio_chunks.append(chunk)
                rtp_ticks = len(event.audio) // 2 * RTP_CLOCK_HZ
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
                self._transcript().add_text(text)
                logger.debug(f"Avartha transcript: {text!r}")
        elif isinstance(event, AgentWordEvent):
            if not self._stale_audio:
                self._transcript().add_word(event.text, self._word_end_rtp(event))
        elif isinstance(event, AgentPlaybackEvent):
            if not self._stale_audio:
                self._awaiting_playback = False
                self._utterance_id = None
                self._transcript().add_playback(event.text)
        elif isinstance(event, AgentLLMStartedEvent):
            self._stale_text = False
            self._llm_carryover = self._llm_unsent
        elif isinstance(event, AgentStartedSpeakingEvent):
            # Subsequent stale audio belongs to this replacement, not to an
            # earlier reply that was merely waiting for its played caption.
            self._open_at_cut = None
            if self._stale_audio:
                self._stale_audio = False
                for converter in self._output_converters.values():
                    converter.reset()
        elif isinstance(event, SpeechEvent):
            result.vad_events.append(event.kind)
            if event.kind in {"speech_started", "interrupted"}:
                result.was_truncated = True
                result.truncated_audio_bytes += sum(
                    len(audio) for audio, _item_id in result.agent_audio_chunks
                )
                result.agent_audio_chunks.clear()
                self._buffered_agent_audio.clear()
                self._audio_after_cut.clear()
                self._held_end = []
                self._awaiting_playback = False
                if self._cut_utterance_id is None:
                    self._cut_utterance_id = self._discarded_utterance()
                    self._open_at_cut = self._open_at_cut or self._streaming_utterance()
                self._utterance_id = None
                self._utterance_audio.clear()
                self._transcript().interrupt()
                self._stale_text = self._stale_audio = True
                for converter in self._output_converters.values():
                    converter.reset()
        elif isinstance(event, ProviderErrorEvent):
            raise RuntimeError(f"Avartha reader failed: {event.message}")
