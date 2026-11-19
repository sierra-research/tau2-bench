# Avartha provider (Pipecat RTVI)

This adapter connects τ-bench's discrete-time audio agent to a hosted Avartha
voice agent. It calls the agent through Pipecat's agent-server (SmallWebRTC)
protocol. τ-bench does not start or configure the agent. Model, speech, and
turn-taking settings belong to the agent.

## Install

The WebRTC client (aiortc) ships in the live extra:

```bash
uv sync --extra voice --extra live
```

## Run

```bash
export AVARTHA_API_KEY=<your-api-key>
tau2 run --audio-native --benchmark none --domain retail \
  --audio-native-provider avartha_pipecat \
  --avartha-base-url https://<host>/inference/serverless/pipecat/v1/public/<agent_id> \
  --num-tasks 1
```

| Option or variable | Description |
|---|---|
| `--avartha-base-url` | The hosted agent's URL (required) |
| `AVARTHA_API_KEY` | The agent's API key, sent as a bearer token when starting each call (required) |
| `--audio-native-model` | Label recorded with the results, `avartha` by default; the URL selects the agent |

`--reasoning-effort` is rejected: reasoning is configured on the agent.

The agent speaks and listens on the wall clock, so ticks are paced in real
time and the user simulator's LLM and TTS run without blocking them, as for
`openai_live`. `--no-realtime-generation` turns the latter off.

## What happens on each call

1. `POST <agent-url>/start` with `caller_tools: true` and the unmodified domain
   policy as the agent prompt override. The reply carries the session ID and
   ICE servers.
2. `POST <agent-url>/sessions/<session-id>/api/offer` exchanges the SDP offer
   and answer. The call carries one Opus audio track each way and an ordered
   `chat` data channel for RTVI messages.
3. The adapter announces RTVI `client-ready` and waits for the agent's
   `session_boot` readiness verdict. A call that does not boot is retried, for
   up to three attempts.
4. It registers the domain tools as self-contained MCP tool definitions
   (`inputSchema`) under the versioned `avartha-harness-tools/1` protocol and
   waits for the agent to acknowledge them.
5. Each tick converts τ-bench's 8 kHz μ-law audio to 16 kHz PCM16 with a
   windowed-sinc filter (`caller_audio.py`; the shared linear-interpolation
   converter leaves a mirror image of the voice above 4 kHz) and paces it onto
   the caller track in 20 ms frames. The agent's track is resampled to 24 kHz
   PCM16, frames of digital silence between utterances are dropped, and the
   rest is converted back to telephony audio. `tool_call` messages become
   τ-bench tool calls and their results are returned as MCP `CallToolResult`
   payloads.
6. On disconnect it sends RTVI `disconnect-bot` and closes the peer connection.

The agent's transcript is revealed only as far as its audio has actually
played. The agent sends each sentence of text before the audio that speaks it;
the adapter holds the text back until the agent's word-timing events or its
end-of-utterance playback caption show that the words aired. A word the
alignment splits in two ("T-shir" + "t") counts once both pieces have aired.
A word spoken after all the text received so far (its sentence's text not
yet sent, common in Chinese) waits for the next text of its reply, keeping its
timing. On barge-in, text the caller has not heard is discarded, and the cut
line's caption, which can arrive after the next line started, is ignored; it is
recognised by its text (it starts as the cut line did), since captions name no
utterance. A caption with `cut` that no barge-in owes is the agent cutting its
own line: the rest of that reply's text is discarded too. Known limits: a cut
line's late caption without `cut` is still read as the next line's; and when a
cut line never sends its caption and the next caption is a self-cut opening
the same way, it is taken for the owed one (the next reply then waits for its
caption, as before).

When the agent times its words on the audio clock
(`avartha-audio-timing/1`: an `audio_timeline` server-message binding its
media clock to the track's RTP clock, and `audio_time` word ends on
`bot-tts-text`), each word airs once playback passes the end of its sound. A
word that ends in the silence the receiver drops airs with the audio before
that silence. A barge-in keeps the timed words whose sound ended in audio
already played, even if their word event arrives in the barge-in tick, and the
word the cut caught if more than half of that word's sound had played. A
word's sound starts at the previous word's end, but never before its line's
first audio or audio after silence the receiver dropped (a short gap of lost
media, up to 200 ms, is not a restart, so a line after a pause that was lost
completely, if no longer than that, is measured from the previous line's
tail). When a cut keeps no word, no word end after the shown text was
reported, agent speech played after the last shown word's end (at least 40 ms
of 20 ms frames whose PCM16 peak exceeds 32, counted from that end, with speech
in the tick before the cut), and the shown text stops inside a sentence, the
cut caught the sentence's next word mid-sound (a code read digit by digit is
one long word). The cut tick shows that started word, or the part of it the
caller surely heard, so the line's clip has text for the audio it carries, and
records the text shown in the tick's `started_word` field (`cut_token.py`):

- A short ordinary word (at most 10 letters and digits) is shown whole; the
  caller may have heard only its start. A longer word shows nothing.
- A token the TTS reads unit by unit is never shown whole: one with `_` or
  `@`, a dot between letters or digits (except a dotted abbreviation of two
  runs of at most 2 letters closed by a dot, `p.m.`, `e.g.`, `U.S.`, which is
  a word), a digit among at least 5 letters and digits, or at least 5
  capitals (ASCII or full-width) (`ivan_rossi_8555`, `hao.h18836`, `HAT110`,
  `6941833`, `IFOYYZ`). It shows its first k units (each digit, each
  capital, each separator, each other letter run), with k = floor(sound /
  (unit x 2.3)). Its sound runs from the first speech frame after the shown
  word's end to the end of the last one before the cut, so a pause before it
  is not counted. The unit is the median over the call's completed codes (from
  the previous word's end to the code's end, over its units), at least 150 ms,
  else 400 ms. With k = 0, or a token the unit model cannot parse
  (`HAT138's`), nothing is shown. The tick records `started_word_units_heard`
  of `started_word_units`.
- An amount or a number read as number words (a currency sign or a digit
  grouping, `R$137,30`, `1,387`, `2.420`, or 3-4 digits, `1177`) shows nothing.
- Only some completed tokens measure the call's unit (`measures_unit`, at
  least 5 units): ASCII capitals and digits with a digit like `6941833` or
  `HAT110`, but not a round number (`10000`) or a number and a short unit
  (`128GB`); an identifier with `_` or `@`; or a lower-case name and at least
  4 digits like `garcia5901`. Words shown by units for safety but read as
  words (`24-hour`, `750ml`, `NEVER`) do not.

Limits: quiet inside a started code counts as code sound, so a long
near-silent stall in the middle of a code (say 3 s, with no word end
reported) can show units that did not play. The 40 ms of speech count from the
last shown word's mark, not from the tick after the one that showed it: an
untimed word's mark stands where the audio received stood at its arrival, so
speech played before that does not count (with 20 ms ticks and the 1 ms held
tail, two ticks of speech after it make 39 ms, three are needed). With no
word events at all, a cut far into a line infers only the line's first word,
showing less than was heard.

A cut that keeps a word caught mid-sound, timed or started, cuts off its line,
so the cut tick carries that line's utterance. The played caption can overtake
a line's last audio or come before its quiet tail; audio that follows the
line's without an RTP gap, within 250 ms of its last timed word's end and
before the next line's text, stays in that line's utterance instead of opening
a new one (if the next line's text came first, only when the caption overtook
the line's audio). A tick that shows only a captioned line's text is that
line's, whatever audio it plays, so the line's last word stays in its
utterance; a tick that also shows the next line's first word is still labelled
by the audio it plays. This label also applies without timing (the caption
alone ends a line's text); the tail window needs timed words. A cut in a
captioned line's quiet tail (audio past its last timed word's end) does not
mark that line cut off. "Played" includes the utterance's last millisecond,
which the adapter holds back for its caption, as it does for every word. As
upstream, a cut tick that discarded no audio is not labelled with the reply. A
cut records `was_truncated` only when it stopped a line whose audio had played,
or when its tick shows text the cut kept; caller speech while nothing plays, or
a cut that discards a reply before any of it played, discards what it must
(counted in `truncated_audio_bytes`) but records no truncation. Readers
attribute a truncated tick to the agent speech before it. The receiver keeps
each decoded frame's RTP timestamp for this; aiortc exposes its RTP origin only
privately. Without that origin, a binding, or a word's `audio_time`, the word
is placed on arrival as before. A word reported after the barge-in is ignored,
timed or not. If the received audio clock rewinds, timing is disabled for the
rest of that call and words fall back to arrival placement; reconnecting starts
a new clock.

## Protocol compatibility

The agent's `tools_registered` or `tools_registration_error` reply must carry
`"protocol": "avartha-harness-tools/1"`. A missing or different protocol fails
the run immediately with `Incompatible Avartha tool protocol`: this is a
configuration error, so local and worker runs stop scheduling tasks and skip
simulation retries instead of recording the mismatch as a benchmark score.

## Result metadata

The call's session ID is saved as `provider_session_id` on `SimulationRun`.
The hosted agent reports no cost.

The audio-clock receiver does not change the saved-input version, judge
fingerprint computation, or the saved simulation schema. Existing saved judge verdicts retain
their validity; newly recorded calls can have different delivered transcript
text because words are placed by the audio played.

## Live provider suite

```bash
AVARTHA_PIPECAT_TEST_ENABLED=1 \
TAU2_AVARTHA_BASE_URL=https://<host>/inference/serverless/pipecat/v1/public/<agent_id> \
AVARTHA_API_KEY=<your-api-key> \
uv run pytest tests/test_voice/test_audio_native/test_provider_suite.py \
  -v -s -k avartha_pipecat
```
