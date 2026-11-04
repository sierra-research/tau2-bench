# Voice V2 release packages

The frozen 2026-10-02 voice-v2 release calls ran before the current matrix
layout (`<language>-<domain>/` cells, `benchmark-frame.json`) and before some
recorded condition fields existed, so `tau2 submit prepare <matrix-root>` cannot
read them. A maintainer-only adapter reads the saved release exports and writes
**the same package** a new submission produces: `benchmark-submission.json`,
`benchmark-diagnostics.json`, `benchmark-quality.json` and
`benchmark-metadata.json` (see
[What the 2.0 package contains](leaderboard-submission.md#what-the-20-package-contains)).
The website and the review tooling treat both alike.

> Terms such as *pack*, *cell*, *binding revision* and *pin* are defined in the
> [glossary](glossary.md).

## Inputs

- The provider's export: `trajectories/<language>/<domain>/` for all 18 cells,
  each with `results.json`, `simulations/` and audio under `artifacts/`, plus an
  optional `correction.json` retry-correction manifest.
- The saved judge verdicts for that provider, one JSON file per scored call at
  `<language>/<simulation-id>.json`. The release keeps them in two judge suites;
  pass each suite's provider directory with its own `--judges-root`.
- A `--metadata-file` with the submitter fields
  ([Package metadata](leaderboard-submission.md#package-metadata)).

```bash
R=data/simulations/release_runs/2026-10-02
tau2 submit prepare $R/leaderboard-v2-corrected-2026-10-09/<provider-export> \
  --historical-voice-v2 \
  --judges-root $R/judges/en-hi-pt/verdicts/<provider> \
  --judges-root $R/judges/es-ko-zh/verdicts/<provider> \
  --metadata-file publication-metadata.json \
  --output ./packages

tau2 submit validate ./packages/<name> --historical-voice-v2 \
  --results $R/leaderboard-v2-corrected-2026-10-09/<provider-export> \
  --judges-root $R/judges/en-hi-pt/verdicts/<provider> \
  --judges-root $R/judges/es-ko-zh/verdicts/<provider>
```

Nothing is copied or modified: the package cites the export by the SHA-256 of
each cell's `results.json` and of each judged verdict. The package records
`"evidence": "historical-voice-v2-2026-10-02"`, so validation and review know
which reader recomputes it.

## Checks

The adapter reads `results.json` and verdicts as recorded JSON, so historical
values the current run-config parser rejects (provider `openai_live`, xAI
reasoning effort `high`) are carried as recorded strings; new runs keep the
strict parser. For every cell it checks the explicit `language_tasks`
canonical ID map, the binding revision, the selected 50 tasks and one trial
against the published frame, the result index against each simulation file,
and that unscored calls are recorded infrastructure errors. Any
`correction.json` must still match the corrected files. All 18 cells must
record one agent model and one caller model; the agent provider and model go
into `benchmark-metadata.json`. It preserves the recorded condition fingerprint
and never invents condition fields added after those calls ran. It must never
be used to qualify an incomplete new run.

Judge diagnostics come from the saved verdicts alone. Each language's measures
are the measures its verdicts record. Every scored call is planned for each of
them, so a missing verdict is reported as *not judged*, and a verdict with unit
errors as an *error call* (its successful units still count, as in
`tau2 judges summarize`). Coverage gaps stay visible in the summary and on the
leaderboard; they never become failures. All verdicts must share one judge
catalog revision. New submissions are stricter: `prepare` refuses unless every
applicable measure is fully judged.

Call quality (`benchmark-quality.json`) is computed the same way as for a new
submission: Interaction from each scored call's saved ticks plus its saved
`tool_use` verdict, Experience from its saved `naturalness` and
`speech_fidelity` verdicts (see [Interaction](glossary.md#interaction) and
[Generation quality](glossary.md#generation-quality)). A partial verdict
leaves the affected component unknown or the call's Experience unscored.
Utterances blocked by a missing Live transcript are the exception: as in judge
diagnostics, they are left out and the rest of the call is scored
(`voice-v2-experience-call-v2`). So is an utterance whose speech-fidelity clip
hit the judge's output limit after retries, unless naturalness flagged it
(then it counts as an issue); preparation caps these at 0.1% of clips. Infrastructure-error calls carry no quality
record. Outside Live
runs the two judges number utterances differently. Pairing follows the agent
chunk groups of the saved ticks (`utterance_alignment`), never ordinal
position.

For the 2026-10-02 export, `voice-v2-experience-call-v1` reproduced the
release's final pooled counts (`quality-scores.json`) for all three providers
and nine core cells; v2 changes only GPT Live's Experience. No call
is `excluded`. Earlier drafts excluded twelve Korean calls with an
unresponsive period. Those calls were replaced before the final export, and
the final scores keep the remaining unresponsive-period calls scored.

## Banking and entity packages

The same export holds the release's banking and τ-Entity (intake) calls:
`trajectories/<language>/banking/` and
`trajectories/<language>/intake/regular/`. The entity frame scores one trial
of the regular condition; `intake/channel-heavy/` and `intake/speech-heavy/`
are extras and are never read. Select the frame with `--benchmark`:

```bash
tau2 submit prepare $R/leaderboard/<provider> --historical-voice-v2 \
  --benchmark banking --metadata-file publication-metadata.json --output ./packages
tau2 submit validate ./packages/<name> --historical-voice-v2 \
  --results $R/leaderboard/<provider>
```

`--benchmark entity` builds `voice-v2-entity` the same way. The package
directories are `<name>_banking` and `<name>_entity`, so all three packages of
one agent can use the same metadata file. Neither frame is
judged (see `JUDGED_BENCHMARKS`), so neither package has
`benchmark-diagnostics.json` or `benchmark-quality.json`, and `--judges-root`
is rejected. `validate` reads
the frame from the package; a `--benchmark` given there must name the same one.

These calls ran before the `voice-v2-banking` and `voice-v2-entity` manifests
existed, so `info.benchmark` is null and the adapter derives the frame identity
instead of reading it:

- **Binding.** The adapter loads each released cell from this checkout
  (`load_cell_language`). Its binding revision must equal the manifest's.
  `info.language_artifacts` and every `info.language_tasks` entry must match
  it on pack and domain revision, pack extension, call mode, task-loader
  options and every `LanguageArtifactPins` field. The trials carry the
  manifest's binding revision.
- **Runtime rules.** `runtime_rules_sha256` hashes Python source files, and
  some of those files changed after the release calls ran. The source diffs
  between the run commits and v2.0.0 were checked and change no behavior for
  these runs. English cells record no digest and must match exactly.
  Non-English cells may record one of the three digests in
  `ACCEPTED_RUNTIME_RULES`: two for banking_knowledge, one for intake. Any
  other digest is rejected.
- **Tasks.** Localized task IDs map to canonical IDs through
  `language_tasks[*].canonical_task_id`. They must equal the frozen subset
  (`banking_knowledge_97`, `intake_200`), with one trial-0 call per task, under
  the regular speech, channel and speech-effects condition.
- **Condition fingerprint.** No fingerprint was recorded, and today's
  `encode_conditions` cannot parse the historical provider values. The adapter
  takes the recorded `info` without the per-cell fields: provenance
  (`git_commit`), the language binding checked above, the language-resolved
  elicitation contract versions, and the provider language. It hashes that
  with the evidence tag. Every cell must give the same value, so a package
  cannot mix run settings. This value is not a matrix condition fingerprint
  and is not comparable with one.

The core checks are unchanged: the result index must match every simulation
file, an unscored call must be a recorded infrastructure error, and the package
must have one agent model and one caller model.

## Uploading

Upload with the review tool, which validates again first:

```bash
python -m tau2.scripts.leaderboard.review_submission \
  web/leaderboard/public/submissions/<name> <provider-export> \
  --historical-voice-v2 --judges-root ... --judges-root ... [--upload]
```

It copies `trajectories/<language>/<domain>/` (for banking and entity, from
the export's `banking/` and `intake/regular/` directories to
`trajectories/<language>/banking_knowledge/` and `.../intake/`), the export's `correction.json`
(as `source_correction.json`) and the verdicts (as
`judges/<agent_provider>/<language>/`) under
`s3://sierra-tau-bench-public/submissions/<name>/`, the layout of the uploaded
Gemini entry. To re-validate a download of that prefix, rename
`source_correction.json` to `correction.json` and pass `--results <prefix>
--judges-root <prefix>/judges/<agent_provider>`. See the
[maintainer guide](../src/tau2/scripts/leaderboard/MAINTAINER.md).

## The 2026-10-02 Gemini package

The corrected Gemini export at
`data/simulations/release_runs/2026-10-02/leaderboard-v2-corrected-2026-10-09`
replaces two infrastructure attempts (hi-airline, zh-telecom) with their
completed retries; `correction.json` records old and new simulation IDs,
rewards and hashes, and the original draft is preserved. The package built from
it has 900/900 scored calls; its judge diagnostics cover every measure and
record the few partial verdicts as coverage gaps. Its evidence is already at
`s3://sierra-tau-bench-public/submissions/gemini-3-8-live-extended-thinking_google_2026-10-07/`
in this layout; the package files are added there by the review tool when the
entry is listed.
