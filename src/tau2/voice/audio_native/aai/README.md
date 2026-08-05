# AAI audio-native provider

Runs tau2-bench against an [AAI](https://github.com/alexkroman/agent) voice
agent over its WebSocket session protocol.

**"AAI" here is the `@alexkroman1/aai` agent framework, not AssemblyAI the
transcription service.** The framework happens to default to AssemblyAI
providers for STT/LLM/TTS, but the thing under test is an agent you run
yourself — locally via `aai dev`, or deployed to the AAI platform.

The provider always connects in **host mode** (`?host=1`): tau2 supplies the
system prompt and the domain's tool schemas, and the agent contributes only its
provider triple (STT + LLM + TTS). That is what makes this harness useful for
comparing *speech* pipelines — the reasoning half of the agent is tau2's, held
constant, so a score difference is attributable to the voice stack.

## Quick start (local agent)

```sh
# terminal 1 — the agent
cd ~/Code/agent/<your-project> && AAI_ALLOW_HOST=1 npx aai dev

# terminal 2 — the benchmark
AAI_WS_URL=ws://localhost:3000/websocket uv run tau2 run \
  --domain retail --audio-native --audio-native-provider aai \
  --num-tasks 5 --max-concurrency 1 --save-to my-run --verbose-logs
```

`AAI_ALLOW_HOST=1` is required and easy to miss. Host mode lets the *client*
supply the agent definition while the session spends the operator's provider
credentials, so the AAI dev server refuses it unless explicitly enabled — the
symptom otherwise is a rejected upgrade, not a helpful error.

`AAI_WS_URL` defaults to `ws://localhost:3000/websocket`, which matches
`aai dev`'s default port, so the local case needs no override. A **deployed**
agent uses `wss://<host>/<slug>/websocket` and also needs `ASSEMBLYAI_API_KEY`.

---

# Reproducing the STT endpoint A/B

This is the comparison the provider was built for: the same agent, run twice,
differing **only** in which STT streaming endpoint it dials. Everything below
assumes a local agent.

## 1. Prerequisites

- The [agent repo](https://github.com/alexkroman/agent) cloned, with
  `pnpm install` run and `npx aai login` completed. The login key is what the
  dev server falls back to for `ASSEMBLYAI_API_KEY`, so without it the agent
  boots and then fails at STT connect.
- An SDK build containing `assemblyAIStt({ streamingUrl })`
  ([agent#976](https://github.com/alexkroman/agent/pull/976)). Without it the
  descriptor can only select a host via `region: "us" | "eu"`, and arm B below
  silently runs on the default endpoint — i.e. you measure nothing and it looks
  like a null result.

## 2. The agent project

Both arms must run **byte-identical agent code**, or a score difference could
be a source drift between two copies rather than the endpoint. So: one project,
one `agent.ts`, and the endpoint comes from the environment.

Create a project directory (this lives outside both repos — it is your local
harness, not a checked-in fixture) whose `node_modules/@alexkroman1/*` link to
the agent repo's `packages/*`, then write:

```ts
// agent.ts
import { agent } from "@alexkroman1/aai";
import { assemblyAILlm } from "@alexkroman1/aai/llm";
import { assemblyAIStt } from "@alexkroman1/aai/stt";
import { assemblyAITts } from "@alexkroman1/aai/tts";

// tau2 connects with ?host=1 and injects its own system prompt + tool schemas,
// so only this provider triple is inherited — which is what makes this file the
// right place to A/B an STT endpoint.
//
// AAI_STT_URL unset (arm A) leaves the SDK's own default endpoint in place;
// set (arm B) points the socket at that URL instead.
const streamingUrl = process.env.AAI_STT_URL;

export default agent({
  name: "tau2-pipeline",
  greeting: "",
  // No sttPrompt on purpose: the harness sends its own via the host config
  // block, keeping the benchmark's STT biasing with the benchmark.
  stt: assemblyAIStt(streamingUrl ? { streamingUrl } : {}),
  llm: assemblyAILlm({ model: "gpt-5.5" }),
  tts: assemblyAITts(),
});
```

`process.env` works here because `aai dev` evaluates the bundle **in the CLI's
own process** and the bundler sets no Vite `define`. This is a dev-only
affordance: after `aai deploy` the same code runs in a guest sandbox whose env
comes from `.env` / `aai secret`, not your shell.

## 3. Start both dev servers

```sh
# terminal 1 — arm A, default STT host
cd <project> && AAI_ALLOW_HOST=1 npx aai dev --port 3000

# terminal 2 — arm B, sandbox STT host
cd <project> && AAI_ALLOW_HOST=1 \
  AAI_STT_URL=wss://streaming.sandbox000.assemblyai-labs.com/v3/ws \
  npx aai dev --port 3001
```

Two servers can share one project directory: with no `client.tsx`, `aai dev`
starts no Vite server and binds the requested port directly, and it keeps no
on-disk state.

Wait for each to log `Session mode resolved … mode: 'pipeline'`, then:

```sh
curl -s -o /dev/null -w "3000:%{http_code} " http://localhost:3000/health
curl -s -o /dev/null -w "3001:%{http_code}\n" http://localhost:3001/health
```

The `streamingUrl` must include the versioned path (`/v3/ws`) — the SDK
supplies that only for its own default host, so a bare origin connects to the
wrong route.

## 4. Verify arm B before spending a full run on it

Do not skip this. If the cluster rejects the key or the path, every arm-B
session dies at STT connect and the arm scores 0.0 — which is indistinguishable
from a quality result in `results.json`.

```sh
AAI_WS_URL=ws://localhost:3001/websocket uv run tau2 run \
  --domain retail --audio-native --audio-native-provider aai \
  --num-tasks 1 --max-concurrency 1 --save-to smoke-sandbox --verbose-logs
```

Then check terminal 2 for `stt_connect_failed` / `stt_auth_failed`. If the
cluster needs its own key, restart **only** arm B with it prefixed — a shell
value beats the login-key fallback, and arm A is untouched:

```sh
ASSEMBLYAI_API_KEY=<sandbox-key> AAI_ALLOW_HOST=1 \
  AAI_STT_URL=wss://streaming.sandbox000.assemblyai-labs.com/v3/ws \
  npx aai dev --port 3001
```

`AAI_DEBUG=1` on either server logs each STT turn, which is the fastest way to
confirm words are arriving.

## 5. Run both arms

```sh
# terminal 3
AAI_WS_URL=ws://localhost:3000/websocket uv run tau2 run \
  --domain retail --audio-native --audio-native-provider aai \
  --num-tasks 20 --max-concurrency 1 --save-to retail-stt-default-20 --verbose-logs

# terminal 4
AAI_WS_URL=ws://localhost:3001/websocket uv run tau2 run \
  --domain retail --audio-native --audio-native-provider aai \
  --num-tasks 20 --max-concurrency 1 --save-to retail-stt-sandbox-20 --verbose-logs
```

### Concurrency: the TTS cap binds across BOTH runs

The user simulator's voice is ElevenLabs, and its subscription caps concurrent
requests (5 on lower tiers), answering `429 concurrent_limit_exceeded` over it.
A synthesis that burns its retries is a **caller utterance that arrives late or
not at all**, and on retail the delayed ones are disproportionately the
spelled-out names and ZIPs that authentication depends on — so exceeding the cap
corrupts the variable this A/B measures.

`--max-concurrency` does not bound it on its own: a single simulation fans out
(`OutOfTurnSpeechGenerator` pre-generates its inserts in a thread pool).
`DEFAULT_TTS_MAX_CONCURRENCY` (4) is the real ceiling, enforced by a semaphore
in `tau2.voice.synthesis.synthesize`.

**That ceiling is per PROCESS.** Two arms in parallel means two semaphores, so
set each to half the cap:

```sh
TAU2_TTS_MAX_CONCURRENCY=2 AAI_WS_URL=... uv run tau2 run ...
```

Or run the arms sequentially and leave it at the default. To confirm afterwards
that the cap held:

```sh
grep -c "synthesize_voice failed" data/simulations/<run>/artifacts/task_*/*/task.log
```

Anything above zero means some caller speech was delayed; treat the run as
noisy rather than comparable.

## 6. Read the results

```sh
uv run python scripts/failure_report.py retail-stt-default-20 retail-stt-sandbox-20

# or, to paste into a coding agent
uv run python scripts/failure_report.py retail-stt-default-20 --top 12 --out failures.md
```

The report is ordered the way a fix needs it:

- **Run trust first**, before any score — caller-voice 429s, sessions that
  produced no scored row, reconnects, `error`/`idle_timeout` counts. If these
  are non-zero, everything below is partly measuring the harness.
- **Wire anomalies, separately from reward** — transcripts emitted past a turn's
  terminal `cancelled` frame, user turns that never got a `reply_done`,
  first-word gaps over 10s, zero-duration speech windows. A 0.0 caused by a
  provider that never connected is not evidence about agent quality, and the two
  want different fixes.
- **Failures worst-first**, each with a named failure mode, the expected action
  beside the call the agent really made, and the judge's own justification for
  every missed assertion.

The argument diff is usually the finding. On an STT-bound domain a wrong call is
the mis-hearing made visible — and it is worth reading closely, because the
failure is not always a mangled digit. One measured run returned an item on
order `#W5490111` with `credit_card_3124723` where the task expected `#W7387996`
with `paypal_9497703`: a different order *and* payment method, i.e. an
STT-mangled email that resolved to **a different real user**, scored NL 1.0
because the call itself was conducted well.

The script handles three traps that produced wrong conclusions when this was
done by hand — it collapses retries to the highest trial per task (a retry
*overwrites* the earlier score, so counting every row mixes a superseded score
into the mean), labels an incomplete run instead of averaging it as final, and
keeps wire failures out of the reward numbers.

Two more it cannot handle for you:

- **A low reward with high `NL_ASSERTION` means the agent acted wrong, not that
  it spoke badly** — most often that it never got past authentication, since a
  call that fails `find_user_id_by_name_zip` executes zero expected actions and
  still scores NL 1.0 for handling it gracefully. The report names this
  `NEVER_AUTHENTICATED` / `WRONG_ACTIONS_GOOD_TALK`, but which one you care
  about is a judgement about what you changed.
- **Failing calls run long**, because an agent that cannot authenticate keeps
  retrying. Any wall-clock cap therefore kills already-doomed conversations
  preferentially, so the worse arm loses more sessions to timeouts than to
  scoring and the gap looks larger than it is.

For raw detail the report points at both files per failure: the wire events in
`data/simulations/<run>/artifacts/task_*/sim_*/task.log` (`grep 'AAI event:'`)
and the per-tick tool calls in `data/simulations/<run>/simulations/*.json`. Note
`sim["messages"]` is empty in audio-native runs — the ticks are the only record.

## 7. Find the STT errors behind a failure

For an STT A/B this is the report that matters, because it is the only one that
distinguishes "the agent did the wrong thing" from "the agent was told the wrong
thing":

```sh
uv run python scripts/stt_errors.py retail-stt-default-20 retail-stt-sandbox-20

# every mis-hearing, including ones that never reached a tool call
uv run python scripts/stt_errors.py retail-stt-default-20 --all-errors

# skip tasks that passed
uv run python scripts/stt_errors.py retail-stt-default-20 --failing-only
```

Ground truth exists because tau2 *synthesizes* the caller: every `user_chunk`
carries an `audio_script_gold` naming the exact text spoken. The script pairs
those against the agent's `user_transcript` wire events and prints
said/heard/diff per utterance.

**By default it shows only the mis-hearings that reached a tool call**, across
every task — including tasks that PASSED, since a mis-heard argument the agent
recovered from is a near miss worth seeing before it costs a run. Each row
carries a `caused:` line naming the argument:

```
- SUBSTITUTION — words swapped (homophone or proper noun)
  - said:  `M, E, I—D, A, V, I, S. Zip is eight, zero, two, one, seven.`
  - heard: `M-E-I-D-A-B-I-S. Zip is 80217.`
  - caused: `find_user_id_by_name_zip.last_name`: agent used 'dabis', which
            appears in the transcript but not in what the caller said
```

The rule needs no expected value, which is what lets it cover passing tasks: an
argument whose value is in the transcript and was never *said* is a mis-hearing
that got as far as a tool call. When a failed check does name an expected value,
two things are added — the reason says what the caller actually said, and a value
that was spoken but never reached the tool at all is reported too (an action the
agent could not perform because it was never told the right thing). A value that
*is* present in what the caller said is never evidence, however unlike the
transcript it looks.

This matters because most mis-hearings are harmless: on the measured runs only
**~10%** of them reached a tool call. A report listing all of them invites fixing
the loudest rather than the costly one. The hidden count is still printed, since
that number is also the honest ceiling on what a language or endpoint fix would
buy.

Three false-positive sources were found by reading its own output, and all three
are worth knowing about if you extend it:

- **Spelled-out separators.** A caller spelling an email says `dot` and `at`
  aloud; STT writes punctuation. Both are folded, gated on the utterance looking
  address-ish, so `mia.garcia2723@example.com` stops reading as an error — it is a
  *correct* transcription.
- **Cross-word substring matches.** Comparing against a space-less form made
  `"And when you say that"` contain `usa`, so an agent's `country='usa'` was
  blamed on an utterance that never mentioned a country. Matching is now anchored
  to token boundaries (inside one token, or an exact run of consecutive ones).
- **Asymmetric spelling punctuation.** Gold `"M, E, I—D, A, V, I, S"` against
  heard `"M-E-I-D-A-B-I-S"`: deleting dashes alone left the gold as
  `me | id | avis` versus one token for the transcript, so *every* argument looked
  absent from what the caller said. Spelled runs are collapsed before any
  dash handling.

Errors are classed because the classes need different remedies, and only one of
them is fixable agent-side:

| class | meaning | agent-side retry helps? |
| --- | --- | --- |
| `NON_LATIN_SCRIPT` | English came back in Devanagari/Hebrew/etc. | no — language detection |
| `NEGATION` | polarity flipped; meaning inverted | no |
| `EMAIL` | address structure lost (`@` gone) | no |
| `DIGITS` | ZIP / order id substitution | each retry is an independent coin flip |
| `SUBSTITUTION` | homophone or proper noun swapped | sometimes — one candidate may be right |
| `SPLIT` / `MERGED` | utterance boundaries disagree with the simulator's | no — endpointing |
| `TRUNCATED` / `NOT_HEARD` | most or all of it missing | no |
| `MINOR` | inflection or filler only | n/a, not worth chasing |

Two things it is careful about, both of which produced false findings in the
first draft:

- **The comparison is normalized.** A ZIP spoken `"one, nine, one, two, two"`
  and transcribed `"19122"` is *correct*; so is `"Y-U-S-U-F"` for
  `"Y, U, S, U, F"`. Raw diffing flags nearly every authentication utterance,
  which buries the real errors. Digit words fold to digits, digit runs join,
  hyphens are deleted (not spaced — otherwise `t-shirt` splits into two tokens
  and a perfect transcript scores as a mismatch), and only what remains counts.
- **Alignment is not positional.** STT decides utterance boundaries and the
  simulator decides utterances, and they disagree — a pause or a `[sneeze]`
  splits one into two. Index pairing shifts every later row after the first
  split, so the whole rest of the call reads as errors. The script aligns
  greedily over 1:1, 1:2 and 2:1 and reports the cardinality, because a split is
  itself a finding: the agent answered half a sentence.
