# Voice Interaction Metrics

## The facts-first interaction suite (`tau2 metrics interaction-facts`)

The τ-ML paper's headline interaction report is a **descriptive suite of
deterministic interaction facts** — per call and aggregated per
(language × domain × provider × reasoning-effort) cell — produced by:

```bash
tau2 metrics interaction-facts <run-dirs...> -o facts.json \
  [--langs es pt] [--domain telecom] [--long-gap-threshold 3.0] [--no-turns]
```

Composite scores are deliberately demoted (binding decision, 2026-08-18):
EVA-Bench-style thresholds and composite scores are arbitrary and lossy,
especially across languages. The EVA turn-taking composite is still computed
and version-stamped for comparability, but it lives in a structurally
separate `shadow_scores` section labeled `uncalibrated-shadow` — never ranked
or headlined. The LLM-dependent composites (the full EVA-X conjunction and
the τ quality composite) live in the `tau2 judges legacy conversation` artifact's
shadow section (see [Conversation judges](conversation-judge.md)). The
organizing rule: deterministic computation lives under `tau2 metrics`;
LLM-judged computation lives under `tau2 judges`.

Event sourcing is exclusively the orchestrator's **typed** sim actions and
injected effects, read through the canonical extractor layer in
`voice_interaction_metrics.py` and routed by the canonical scorer in
`tau2.metrics.turn_taking` — one event taxonomy shared with the leaderboard
panel below. Nothing is re-inferred from audio-only heuristics.

### First-class fact columns (per call, and mean/median per cell)

- **Response latency** (ordinary turns): mean/p50/p90, plus the tool-turn vs
  plain-turn latency split (`tool_turn_latency_mean_ms` /
  `plain_turn_latency_mean_ms`).
- **Responsiveness**: `missed_response_rate` (scored turns left unanswered)
  and the promoted τ-voice panel `response_rate`.
- **Interruption behavior**: `agent_interruption_rate`,
  `caller_interruption_rate`, `incursion_overlap_mean_ms` (overlap duration
  on agent incursions), yield latency p50/p90, post-interrupt recovery
  latency p50/p90.
- **Turn route mix**: the fraction of scored turns per route class
  (ordinary / agent-interruption / caller-interruption / dual /
  missed-response).
- **Selectivity**: `selectivity_backchannel`, `selectivity_vocal_tic`,
  `selectivity_non_directed` — promoted from the τ-voice extractor below,
  not reimplemented.
- **Dead air** (the one new instrument): maximal spans of the (end-filtered)
  tick timeline where neither party is speaking. Reported as
  `dead_air_total_s`, `dead_air_fraction` (of call duration),
  `dead_air_max_gap_s`, and `dead_air_long_gap_count` (gaps strictly longer
  than the long-silence threshold; default the shared 3.0s constant,
  parameterized via `--long-gap-threshold`). A gap that IS a measured
  ordinary response latency is still dead air acoustically and counts once
  in the totals; `dead_air_excl_response_s` additionally excludes those
  ordinary response-latency gaps, isolating silence no pending response
  accounts for.

### Typed N/A, never silent gaps

Every fact that can be unscoreable is a typed `Fact` carrying an
N/A-with-reason (`no_tick_timeline` for text runs, `no_scored_turns`,
`no_events` for empty denominators) — the same pattern as the turn-taking
scorer's reasoned N/A (`no_scoreable_turns`, `agent_mute` for the
infrastructure-mute population). Cell aggregates report per-reason N/A counts
next to every mean, and shadow-score cell blocks report per-reason shadow N/A
counts. A mute call keeps its descriptive facts (they are true observations)
while its shadow composite is a reasoned N/A that drops out of shadow means.

The artifact also carries per-turn evidence (`turns`: route, latency,
overlap, interrupt count, yield/recovery latencies per real caller turn;
drop with `--no-turns` on very large pools), canonical panel event counts,
and full provenance (instrument version, config, git sha, content-derived
artifact id, per-run source metadata).

---

The τ-voice leaderboard reports **interaction quality** alongside task success
(pass^1). Interaction metrics measure the conversational dynamics of a voice
agent on the open, full-duplex audio channel: how fast it responds, whether it
yields when interrupted, whether it talks over the user, and whether it can
tell real user turns apart from backchannels, vocal tics, and speech not
directed at it.

All interaction metrics are computed **offline from the tick-level
trajectories** that every voice submission already uploads — no extra runs, no
judges, no self-reported numbers. Maintainers recompute them from the
submitted trajectories during review.

```bash
tau2 submit interaction-metrics <experiment-dirs-or-trajectories-dir> [--output metrics.json]
```

## The metric panel

| Group | Metric | Field | Direction | Definition |
|-------|--------|-------|-----------|------------|
| Latency | L_R | `response_latency_mean` | ↓ | Mean seconds from the end of a user turn to the start of the agent's response. |
| Latency | L_Y | `yield_latency_mean` | ↓ | Mean seconds for the agent to stop speaking after the user interrupts. |
| Responsiveness | R_R | `response_rate` | ↑ | Fraction of user turns that received an agent response before the user had to speak again. |
| Responsiveness | R_Y | `yield_rate` | ↑ | Fraction of user interruptions where the agent yielded within the no-yield window. |
| Interrupt | I_A | `agent_interruption_rate` | ↓ | Agent-interrupts-user events per user turn (`agent_interrupts_count / response_total`). |
| Selectivity | S_BC | `selectivity_backchannel` | ↑ | Fraction of user backchannels ("mm-hmm") the agent correctly talked through. |
| Selectivity | S_VT | `selectivity_vocal_tic` | ↑ | Fraction of vocal tics ("um", coughs) the agent correctly ignored. |
| Selectivity | S_ND | `selectivity_non_directed` | ↑ | Fraction of non-agent-directed speech ("hold on", side conversations) the agent correctly ignored. |

Latencies are in seconds. Rates and selectivity are fractions in [0, 1],
except I_A, which counts events per response-eligible user turn and can
exceed 1 when the agent interrupts the same turn more than once; selectivity
values are correct-rates (1 − error rate). Every rate is stored with the
event count backing it (the `counts` block); the leaderboard hides rates
computed from fewer than 10 events (for L_R the supporting count is the
number of responded turns, `response_rate × response_total`).

The leaderboard's Selectivity column is the unweighted mean of S_BC, S_VT,
and S_ND. It is shown only when all three components are present and backed
by at least 10 events — a partial mean would not be comparable across rows —
and an Overall value is additionally hidden if any contributing domain rate
falls below that threshold.

## Universal quality-rubric mapping

The per-simulation universal quality rubric reuses this same event extractor;
it does not maintain a second definition of response gaps or speech overlap. The
eight panel measurements map to six non-overlapping deterministic checks:

| Quality factor | Panel evidence | Failure condition |
|----------------|----------------|-------------------|
| `responsiveness` | L_R + R_R | A caller turn is unanswered, or mean response latency exceeds the configured threshold (3.0 s by default). |
| `yielding` | L_Y + R_Y | The agent fails to yield to any real caller interruption within the 2.0 s event window. |
| `inappropriate_interruption` | I_A | The agent begins speaking over the caller. |
| `backchannel_selectivity` | S_BC | The agent incorrectly yields to a backchannel. |
| `vocal_tic_selectivity` | S_VT | The agent responds or yields to a vocal tic. |
| `non_directed_selectivity` | S_ND | The agent responds or yields to non-directed speech. |

Latency and rate are still recorded separately. Each related pair contributes
only one binary factor, preventing response behavior or yielding behavior from
receiving twice the weight of interruption and selectivity. The rubric's
`monologue` factor remains separate because floor-hold duration is not one of the
original τ-voice panel metrics.

These definitions and all detection windows are identical to the τ-voice
paper's analysis pipeline; the implementation lives in
[`src/tau2/metrics/voice_interaction_metrics.py`](../src/tau2/metrics/voice_interaction_metrics.py)
and is guarded by a golden parity test against the original pipeline's output
(`tests/test_voice/test_interaction_metrics/`).

## How events are detected

Full-duplex simulations advance in discrete **ticks** (200 ms by default).
Each tick records whether the user and the agent are producing speech
(`contains_speech`), the user simulator's turn-taking decision
(`turn_taking_action`), and any injected audio effects
(`speech_effects` / `source_effects`).

### Preprocessing

Conversations end with a `###STOP###` signal that produces a spurious 1-tick
user speech segment at the very end. That tick is removed before analysis
(`filter_end_of_conversation_ticks`), preventing false no-yield events.

### Speech segments

Contiguous speech ticks are grouped into user and agent **segments**. A user
segment records the turn-taking action from its first tick (backchannels are
segments whose action is `backchannel`), whether the agent was speaking when
it started (`is_interruption`), and which audio effects (vocal tic /
non-directed speech) were active during it.

### Response events (L_R, R_R)

For each user turn that ended with the agent silent (backchannels and
unresolved interruptions excluded):

- **response** — the agent started speaking before the user spoke again.
  The gap is the response latency.
- **no_response** — the user had to speak again before any agent response.

### Yield events (L_Y, R_Y)

When the user starts speaking while the agent is talking (a real
interruption, not a backchannel/tic/non-directed event):

- **yield** — the agent stopped within the no-yield window (2.0 s).
  The time to stop is the yield latency.
- **no_yield** — the agent kept talking through the window.

### Agent interruptions (I_A)

Every agent segment that starts while the user is speaking counts as one
agent-interrupts-user event. I_A normalizes this by the number of user turns
(`response_total`).

### Selectivity events (S_BC, S_VT, S_ND)

Signals the agent should *not* treat as a user turn:

| Event | While agent is speaking (error = yields within 1.0 s) | While agent is silent (error = responds within 2.0 s) |
|-------|--------------------------------------------------------|--------------------------------------------------------|
| Backchannel | backchannel_error / backchannel_correct | — (backchannels only occur during agent speech) |
| Vocal tic | vocal_tic_error / vocal_tic_correct | vocal_tic_error / vocal_tic_correct |
| Non-directed speech | non_directed_error / non_directed_correct | non_directed_error / non_directed_correct |

When a user segment starting during agent speech matches several categories,
classification priority is: backchannel > vocal tic > non-directed speech >
real interruption. Vocal tics and non-directed speech occurring during
silence (out-of-turn effects) are also scored.

## Detection windows

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `tick_duration_sec` | 0.2 | Tick length (read from each experiment's recorded config when available) |
| `no_yield_window_sec` | 2.0 | Time for the agent to yield after a real interruption |
| `backchannel_yield_window_sec` | 1.0 | Yield within this window after a backchannel = error |
| `vocal_tic_yield_window_sec` | 1.0 | Yield within this window after a vocal tic = error |
| `non_directed_yield_window_sec` | 1.0 | Yield within this window after non-directed speech = error |
| `vocal_tic_response_window_sec` | 2.0 | Response within this window to a silent-period vocal tic = error |
| `non_directed_response_window_sec` | 2.0 | Response within this window to silent-period non-directed speech = error |

The windows used for a computation are recorded in the
`interaction_metrics.config` block of each submission, and
`interaction_metrics.version` stamps the metric-code version. The recorded
`tick_duration_sec` is the value the experiments actually used (taken from
each experiment's `audio_native_config`), not the tool default.

## Aggregation

Metrics are computed per domain over all simulations of an experiment. The
leaderboard's "Overall" view is the per-metric mean across available domains
(missing values skipped), mirroring how pass^1 is averaged. Event counts are
summed.

## Relationship to Full-Duplex-Bench

τ-voice's interaction panel covers the interaction-quality axis that
[Full-Duplex-Bench](https://arxiv.org/abs/2503.04721) (v1/v1.5) measures,
inside a harder, task-oriented setting:

- FDB **takeover rate** ↔ yield rate R_Y; FDB **stop/response latency** ↔
  L_Y / L_R; FDB **backchannel handling** ↔ S_BC (plus S_VT / S_ND, which FDB
  does not cover).
- FDB's **backchannel-timing JSD** against a human corpus is not portable to
  synthetic task dialogues and is not reported.
- FDB's LLM-judged response quality, MOS/prosody scores, and pause-handling
  metrics are deliberately out of scope for the leaderboard (no judged or
  audio-perceptual metrics).
- Numbers are **not directly comparable** to FDB: τ-voice uses different
  dialogues, different signal injection, and tick-quantized (200 ms) timing.

## Computing metrics yourself

```bash
# One or more experiment directories (results.json + simulations/)
tau2 submit interaction-metrics data/tau2/simulations/my_voice_run

# A downloaded submission's trajectories directory (all domains at once)
aws s3 sync s3://sierra-tau-bench-public/submissions/<dir>/trajectories/ /tmp/trajs --no-sign-request
tau2 submit interaction-metrics /tmp/trajs --output interaction_metrics.json
```

The command fails loudly if the input contains no tick-bearing (full-duplex)
simulations — interaction metrics are only defined for voice runs.

For maintainers: `python -m tau2.scripts.leaderboard.backfill_interaction_metrics`
recomputes the block for existing leaderboard submissions from their public
S3 trajectories, and `review_submission.py` recomputes it whenever a
submission is reviewed (submitted values are always replaced by recomputed
ones).
