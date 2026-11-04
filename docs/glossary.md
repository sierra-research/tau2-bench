# Glossary

Terms used across the τ-bench 2.0 docs, in alphabetical order.

## Benchmark manifest

A JSON file under `data/tau2/benchmarks/` that defines a released benchmark:
its version, call mode (text or voice), the frozen task subset for each domain,
and the list of [cells](#cell) with their [binding revisions](#binding-revision). Three manifests ship with 2.0:

| File | Benchmark version | Mode | Cells |
|---|---|---|---|
| `text.json` | `text-v1` | text | 6 languages × 3 domains = 18 |
| `voice.json` | `voice-v2` | voice | 6 languages × 3 domains = 18 |
| `voice-v1.json` | `voice-v1` | voice | English × 3 domains = 3 |

`tau2 benchmarks list` prints them. `tau2 run --benchmark <name>` selects one by
file name (`text`, `voice`, `voice-v1`).

## Binding revision

A SHA-256 fingerprint of everything a cell feeds into a call: the [pack](#pack) revision, the localized tasks, database and policy, the
prompt and guideline versions, and the call mode. A text cell and a voice cell
for the same language and domain have different binding revisions. A run refuses
to start if the current checkout no longer produces the binding revision the
manifest records.

## Cell

One language × domain pair in a benchmark, for example `es-retail`. A
benchmark run saves each cell in its own directory (`<root>/es-retail/`). Scores are
averaged over tasks within a cell, then over cells with equal weight.

## Frame

The full, fixed population a benchmark is scored over: every cell, every
frozen task ID, and the trial indices. A run that covers only part of the
frame (a slice or a `--num-tasks` sample) has no official score; the frame is the
denominator. Calls that ended in an infrastructure error are the exception: they
are harness faults, so they are excluded from the denominator and reported as
`error_trials`. A saved run records its frame in `benchmark-frame.json` at the root.

## Generation quality

The public name for the τ-i18n measure that papers, data and code call
**Experience** (`experience`). An agent utterance has an issue when the
naturalness or the speech-fidelity judge flagged it. The paper reports the
share of aligned utterances without an issue. Voice-v2 packages score it per
call (`voice-v2-experience-call-v2`, in `benchmark-quality.json`): a call
passes when at most 25% of its eligible agent utterances have an issue. It is
*unscored*, never failed, unless both judges returned complete verdicts that
pair every utterance. Live speech recorded without a transcript is a recording
gap: those utterances are left out and the rest of the call is scored. So is
an utterance whose speech-fidelity clip still hit the judge's output limit
after retries, unless naturalness flagged it, in which case it counts as an
issue. The leaderboard shows it as Generation quality and rates
it as passed / (passed + failed). Keys such as `experience` and the paper's
`experience.json` keep the old name.

## Interaction

The call-level turn-taking measure of voice-v2 packages
(`voice-v2-interaction-call-v1`, in `benchmark-quality.json`). Each scored
call gets five flags: **nonresponse** (a missed caller turn, a mean reply
latency over 3 s, or an unyielded interruption), **interruption** (more than 3
agent onsets over caller speech), **selectivity** (more than 3 errors summed
over backchannels, vocal tics and non-directed speech), **monologue** (an agent
floor hold over 45 s) and **tool_misuse** (the saved `tool_use` judge verdict).
The first four come from the call's ticks; a flag is unknown when the call has
no opportunity for it. A call fails with at least two known flags set, passes
when even all unknown flags set could not reach two, and is otherwise
unscored. The rate is passed / (passed + failed).

## Judge

A model-based diagnostic run after the simulation, with `tau2 judges suite`. Judges never change the task reward. Each judge is defined
by a contract (model, prompt version, rubric version, scope) in
`src/tau2/judges/production_judges.json`. For voice-v2 submissions the judges
must be run before `tau2 submit prepare`.

## Measure

One named quantity a judge reports, such as `naturalness`, `speech_fidelity`
or `tool_use`. A measure applies to some languages and not
others; `tau2 judges summarize` lists, for each cell and measure, how many calls
it applies to, how many were judged, and the pass and fail counts. Calls a
measure cannot judge because recorded Live speech has no transcript are
excluded from it and counted as `input_excluded_calls`; they do not block a
submission. Speech-fidelity clips whose judge reply hit its output limit are
counted as `output_limit_units` and do not block either, up to 0.1% of a
package's clips.

## Pack

A language pack: `data/tau2/multilingual/<lang>/pack.yaml` plus its
localization files. It holds the caller personas, prompt clauses, voice settings,
and the localized tasks and database for each domain it covers. Every run
resolves a pack; with no `--language`, the English pack is used. See
[Language packs](language-packs.md).

## Pin

A value recorded in a manifest so that later runs can be checked against it: the subset digest of each task subset and the binding revision of
each cell. `tau2 benchmarks pin <name>` recomputes pins from the checkout and
refuses to change them unless you also pass a new `--revision`.

## Trial

One run of one task. `--num-trials N` runs each task N times with different
seeds. The released benchmarks are scored at one trial (pass@1);
`pass^k` is the share of tasks passed in all k trials.

## τ-i18n / τ-multi

The multilingual voice benchmark (`voice-v2`). Older docs and identifiers call it τ-Multilingual or "multilingual".

## τ-Entity

The entity-capture benchmark on the `intake` domain. Older docs and identifiers call it "elicitation". See [τ-Entity](elicitation.md).
