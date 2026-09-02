# Voice Interaction Metrics

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
