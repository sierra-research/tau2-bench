# Contributing and validating language judges

A language can be contributed with its user simulator alone or with agent judges.
Use the [submission template](language-submission-template.md) for the README beside
the language pack. Judge status belongs to an exact language, measure, model,
prompt/rubric version and applicability scope; it does not apply to a whole pack.

> Terms such as *pack*, *cell*, *binding revision* and *pin* are defined in the
> [glossary](glossary.md).

## Production selection

[`production_judges.json`](../src/tau2/judges/production_judges.json) selects one
definition per language/measure. Its naturalness v21 selection is a reviewed,
**unvalidated** operational default requested for this release; the other
selected definitions retain their existing validation status.
It is not a history of past judge versions. Version numbers belong to separate
judging methods; a newer naturalness rubric does not replace a counting-units or
gender-agreement diagnostic.

| Definition | Selected coverage |
| --- | --- |
| Naturalness v17 / rubric v21, validation pending | Natural wording in Spanish, Portuguese, Hindi, Korean and Chinese |
| Language diagnostics v15 / rubric v19 | Counting units (es, pt, zh), register (pt), gender agreement (pt, female agent), name/address conventions (hi), honorific agreement (ko), modal particles (zh) |
| Hindi gender v16 / rubric v26 | Gender agreement with a male agent and recorded caller gender |
| Tool use v7 / rubric v4 | Combined tool errors, authentication arguments and tool parameters across all six languages |
| Speech fidelity v7, validation pending | Audio fidelity across all six languages |

Naturalness v17 renamed the v16 factor identifier from `natural_word_choice` to
`naturalness`. Rubric v21 now selects the exact language-specific wording in the
[reviewed candidates](../data/tau2/judge_catalog/README.md). The v17 wrapper,
model/settings and execution method are unchanged. Its new contract and catalog
revisions keep earlier v20 verdicts from being reused. The review and candidate
runs are not held-out accuracy validation; v21 results must be labeled as such.
The catalog's source hash still identifies the original validation archive, not
validation evidence for v21. The catalog records the exception in
`unvalidated_contract_ids`; authoring lists the five reviewed candidates rather
than presenting this runtime selection as a validated source.

Speech fidelity v7 keeps the v6 model, prompt, rubric and decision rule. It
asks the provider for a reply constrained to the findings schema the judge
parses (closed categories, severity 1-3, at most eight findings) and lowers the
output cap from 32,768 to 8,192 tokens. On a small share of clips, often
digit-by-digit IDs and Mandarin homophones, the judge keeps second-guessing a
word until it reaches its output limit. Each such clip leaves its call
unscored until it is judged again. The schema keeps the reply parseable and the
cap bounds what one such clip costs. Like v6, v7 has no held-out human
validation and is listed in `unvalidated_contract_ids`. Its new contract and
catalog revisions keep v6 verdicts from being reused.

### Benchmark-authored speech

Each definition records an `input_exclusions` rule, and every saved verdict
records the rule it was judged under, so a rule change makes earlier verdicts
stale. Catalog revision `validated-measures-v4` introduced the rules
(`tau2.judges.exclusions`):

| Rule | Definitions | Removed before judging |
| --- | --- | --- |
| `injected-opener-and-transfer-line-v1` | Naturalness, language diagnostics, Hindi gender, authored candidates | The injected opener and the policy-mandated English transfer line |
| `injected-opener-v1` | Speech fidelity | The injected opener clip |
| `none` | Tool use | Nothing |

The **injected opener** is the first agent message the orchestrator seeds:
the task's `agent_opener`, else the recorded language context's
`agent_greeting`, else the English default opener. It is identified
structurally. It must be the call's first agent message (tick 0 in full-duplex
runs) and carry exactly the recorded opener text. A model that later repeats the
greeting is still judged.

The **transfer line** is `YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD
ON.`, which stays in English in localized policies. Complete occurrences are
removed anywhere in a turn. When the turn was interrupted, a delivered prefix at
its end is removed too. `###STOP###`/`###TRANSFER###` markers are also removed,
and turns left empty are dropped. Unit IDs keep their original turn ordinals.

Speech fidelity keeps the transfer line. It is audio the agent synthesized, and
removing it from the reference would itself create a reference/audio mismatch.
Tool use keeps the full transcript because the transfer line is evidence of a
policy-conformant transfer. The rules change judge inputs only; prompts, models
and parsers are unchanged, and the underlying validation evidence is not
re-derived.

Provenance fields (`donor_revision`, `source_revision`) hold neutral package
labels. Schema validation rejects commit hashes.

Language-specific text rules live in each definition's
`nativeness_criteria[language]`; speech fidelity uses
`speech_material.rubrics[language]`. The shared judge implementations read these
rules for the call's language. The `pairs` entries select which diagnostics are
enabled for each language and participant scope.

Some calibrated requests contain several criteria together. Retain those peer
criteria even when only one is reported: removing them changes the prompt used
in validation. Only entries in `pairs` enable reported production measures.

Normally, replace a selected definition only with validation evidence for the
same language, measure and participant scope. The user-requested v21 operational
default is the explicit exception above; it must not be called validated until
held-out evidence meets the requirements below. Remove a previous definition
once no selected pair needs it. Candidate files remain in the authoring catalog.

## Separate agent expectations from caller behavior

Judge instructions describe agent speech or actions. Caller language, register
and explicitly recorded identity can provide context, but caller persona rules
are not agent requirements. An impatient caller does not require an impatient
agent. Agent gender comes from the agent configuration, not the caller persona.

Authoring supports a naturalness extension for a new language, adaptation of an
existing catalog diagnostic, or a new diagnostic with its own identifier. Each
needs an observable question, opportunity definition, violation rule, exceptions
and concrete agent examples. A new diagnostic can reuse an existing judging
method when its inputs and result shape fit; a different method needs an
implementation and correctness tests as well.

## Submission states

| State | What contributors may submit | Production behavior |
| --- | --- | --- |
| User-only | Reviewed user pack, tasks, voices and README | No new judge enabled |
| Unvalidated judge | Reviewed runnable candidate and its limitations | Explicit candidate execution; the v21 naturalness release default is a documented exception |
| Validated judge | Exact candidate plus passing evidence/report and reviewed catalog change | Enabled only for the approved scope |

Missing evidence, failed thresholds, insufficient examples and incomplete
execution all leave a judge **unvalidated**. Unit tests, model-generated examples
and human review of a rubric alone do not establish judging accuracy. Changing
judging behavior or extending the validated scope requires fresh evidence.

## Author and run an unvalidated judge

For example, save this independent agent request as `judge-request.json`:

```json
{
  "mode": "new_diagnostic",
  "diagnostic_id": "next_step_clarity",
  "language": "ro",
  "variety": "Romanian as used in Romania",
  "agent_requirements": "When asking the customer to take an action, state the next action plainly. Do not require an action when none is needed. Judge the agent, not the caller's phrasing.",
  "domains": ["retail"],
  "evaluation_level": "utterance",
  "agent_gender": null,
  "author_model": "openai/gpt-5.5",
  "author_model_args": {"reasoning_effort": "low"},
  "judge_model": "openai/gpt-5.5",
  "judge_model_args": {"reasoning_effort": "medium"}
}
```

The settings are explicit author choices, not validated defaults. `agent_gender:
null` proposes both male and female agent coverage; validation must support the
requested scope. Then run:

```sh
tau2 factory draft-judge --request judge-request.json --output judge-draft.json
```

This calls the author model. Review/edit the draft's criterion, examples and
exceptions. Assemble a `CompileJudgeRequest` JSON containing the **complete edited
draft object**, `revision`, and `review` with `reviewed_by` and `content_sha256`.
Compute the digest of that edited `JudgeDraft` with
`tau2.utils.artifacts.content_digest`; the old digest is invalid after edits.
Record the actual reviewer rather than inventing an approval. Compile locally:

```sh
tau2 factory compile-judge --request reviewed-judge.json --output candidate.json
tau2 judges candidate saved-results --candidate candidate.json --output candidate-run --dry-run
tau2 judges candidate saved-results --candidate candidate.json --output candidate-run
```

Compilation makes no model call. The resulting `CandidateJudge` has status
`unvalidated`. The dry run checks saved-call applicability; actual execution calls
the judge and records resumable output in `candidate-run/`, separate from the saved
run and production diagnostics. The output `candidate-results.json` contains the
exact candidate, plan and `PredictionBatch`, including errors. Human evidence
must reference those stable row IDs and source-input fingerprints. New inputs or
candidate content invalidate reuse.

Other authoring choices use the same stages:

- **Naturalness extension:** `mode="naturalness_extension"`,
  `diagnostic_id="naturalness"`, `evaluation_level="utterance"`. This uses the
  existing naturalness judge's structure with reviewed language-specific criteria.
- **Adapt a catalog factor:** `mode="adapt_diagnostic"`,
  `source_contract_id="authoring-catalog"`, and `source_measure`, such as
  `honorific_agreement`. These 31 authoring factors are candidate references, not
  blanket validation claims.
- **Adapt an active text criterion:** specify its exact `source_contract_id`,
  `source_language` and `source_measure`. Source accuracy does not transfer to the
  new language. Speech-fidelity adaptation needs an audio implementation; this text
  authoring path cannot create audio judges.

The user pack and unvalidated judge can be submitted now. Execution, validation and
production promotion are optional additional work, not conditions for submitting
an explicitly unvalidated candidate.

## Register and reuse a reviewed diagnostic

A reviewed candidate is also a portable authoring-catalog entry. Registration
checks its complete draft, review digest, compiled criteria and source identity;
it does **not** validate the judge or enable it in production. No second rubric
format or copy of its prompt is needed.

```sh
tau2 factory register-judge --candidate candidate.json \
  --output data/tau2/judge_catalog/next_step_clarity-ro-v1.json
tau2 factory list-judges
```

The standard contribution directory is `data/tau2/judge_catalog/` (under
`TAU2_DATA_DIR` when that root is relocated). `list-judges` and `draft-judge`
automatically discover its `*.json` entries. The directory includes five [reviewed naturalness candidates](../data/tau2/judge_catalog/README.md)
for Spanish, Portuguese, Hindi, Korean and Mandarin. Their exact approved
language-specific rules are runnable through `judges candidate` and remain
unvalidated. This release selects their exact v21 rubric content as the default
naturalness definition by explicit user request, without changing their validation
status. Authors can contribute
additional reviewed entries in a PR. Registration writes only the explicit
output path; it never changes the packaged validated production catalog. Existing
files are immutable, and conflicting source contract IDs are rejected. Keep
versioned filenames and create a new candidate revision for changed definitions.

The listing includes the 31 packaged factor ideas, exact validated source scopes,
and contributed reviewed candidates. It exposes `source_contract_id`,
`diagnostic_id`, source `language`, revision and status. `validated` describes only
the listed source's original scope; every new adaptation starts unvalidated.

For another language, set the following fields on a complete `DraftJudgeRequest`
(the listed contract ID must match the actual entry):

```json
{
  "mode": "adapt_diagnostic",
  "diagnostic_id": "next_step_clarity",
  "language": "sv",
  "variety": "Swedish as used in Sweden",
  "source_contract_id": "next_step_clarity-ro-v1",
  "source_language": "ro",
  "source_measure": "next_step_clarity"
}
```

Retain explicit independent agent requirements, target domain/participant scope
and model settings from the complete request format above. Then use the ordinary
`draft-judge` and `compile-judge` stages. The new draft retains the selected
criterion, independent source requirements, review attribution and full source
artifact digest; loading and compiling it does not require the original catalog
file. Changing a selected source invalidates reuse of an existing draft.

An external entry can be supplied without copying it into the contribution tree:

```sh
tau2 factory list-judges --catalog-entry reviewed-external-candidate.json
tau2 factory draft-judge --request swedish-request.json \
  --catalog-entry reviewed-external-candidate.json --output swedish-draft.json
```

The optional flag is repeatable. Select an existing diagnostic for adaptation;
`new_diagnostic` rejects IDs already present in the built-in or supplied authoring
catalog. Contribute the source entry alongside the language's candidate and README
so future authors can discover it. The adapted candidate still needs its own
language/scope-specific evidence before any production promotion.

## Evidence for new submissions

Freeze the candidate and a held-out evaluation set before evaluating it. The set
must be separate from examples used to develop the prompt/rubric. If you tune on
evaluation failures, that set becomes development material; use a new held-out
set for the next validation claim.

Include:

1. **Data and sampling.** Stable item IDs, source run/revision, target language and
   locale, domains and agent conditions, sampling procedure, and any exclusions.
   Include both actual violations and nonviolations. Judge-positive samples alone
   cannot establish recall. Disclose curated or enriched sampling; its measured
   precision is not an estimate for a different population.
2. **Human reference labels.** Qualified language reviewers label the exact
   evaluation units without seeing judge predictions. Record reviewer expertise,
   instructions, opportunity/violation labels and how disagreements were resolved.
   Audio judges require review of the same audio windows and reference used by the
   judge; transcript-only review does not establish audio accuracy.
3. **Exact candidate.** Model and arguments, versioned prompt/rubric, implementation,
   language/variety, call-versus-utterance unit, input preparation, and participant
   restrictions. Hash-bind the evidence/report to the submitted candidate.
4. **Aligned predictions.** Preserve row-level human labels, judge predictions,
   missing inputs, parsing failures and execution errors. Do not count failures as
   successful nonviolations or silently remove them. Keep the denominators and
   no-opportunity treatment explicit.
5. **Reproducible report.** Confusion counts (TP/FP/FN/TN), human-positive/negative
   support, precision, recall, F1, Cohen's kappa, coverage, gate results and exact
   commands. Supply accessible evidence, not only a screenshot or aggregate score.

Human review must assess the claimed data independence, reviewer qualifications
and scope. Typed validation can verify consistency and reproduce metrics; it
cannot prove that a contributor's human-review declarations are true.

## Numerical acceptance rule

Use the existing frozen evidence rule, applied separately to each submitted
language/measure/scope:

| Metric | Required result |
| --- | --- |
| Precision: TP / (TP + FP) | **Greater than 0.75** |
| Recall: TP / (TP + FN) | **Greater than 0.75** |
| F1: harmonic mean of precision and recall | **Greater than 0.80** |
| Human violations: TP + FN | **At least 10** |
| Cohen's kappa: agreement adjusted for chance | **Greater than 0.60 when there are at least five human negatives** |

The inequalities are strict: precision of exactly 0.75 does not pass. Undefined
required metrics cannot pass. Report when the conditional kappa requirement does
not apply. These are minimum acceptance thresholds, not a guarantee of accuracy
outside the observed population; report sample sizes and limitations prominently.

The historical rule is pinned in the retained validation authority, an
unpublished research archive. That archive includes fixed curated and other recorded designs. New submissions
must meet the held-out requirements above; this guide does not relabel historical
data as held out or broaden historical validation claims.

## Production promotion

Submit the candidate, evidence, recomputed passing report and exact review
approval together. Promotion must verify they refer to the same contract and
scope, then produce a reviewable catalog change. It does not silently replace
existing production judges. A new diagnostic identifier is allowed; approval of
one diagnostic does not validate other outputs from the same model call.

Candidate outputs remain separate from production diagnostics and task success.
The production runner selects the catalog's explicit combinations; naturalness
v21 is the documented unvalidated release exception.
For example, validation for a female agent cannot authorize a male-agent frame.

Use a `ValidationRequest` with the exact `candidate`, `dataset` (`HeldOutDataset`),
`labels` (`HumanLabel` records) and `predictions` (`PredictionBatch`):

```sh
tau2 judges validate --request validation-request.json --output validation-report.json
```

Then a `PromotionRequest` contains that full validation request as `evidence`, the
computed `report`, and a `ReviewApproval` bound to the exact report. The report
itself binds the underlying evidence by digest. After reviewing those inputs:

```sh
tau2 judges promote --request promotion-request.json --output validated-judge.json
tau2 judges suite saved-results --registry validated-judge.json --dry-run
tau2 judges suite saved-results --registry validated-judge.json
```

Validation and promotion are offline. Promotion recomputes the report rather than
trusting a supplied `passed` flag. It exports a reviewable validated registry bundle;
it never edits the packaged catalog automatically. `--registry` is explicit and
rejects unvalidated candidate files. Include the resulting registry/evidence in the
PR; maintainers decide production registration. Without that option, the runner
uses the existing packaged validated catalog.

The definition, plan and verdict schemas live in `tau2.data_model.judges`.
Authoring and validation schemas live in `tau2.judges.authoring_models` and `validation`;
obtain JSON field definitions with each model's
`model_json_schema()`. This workflow consumes portable evidence files and does
not require importing annotation sheets or tooling. User-only submissions never
need to run a diagnostic judge.

## Runtime implementation

Both `judges suite` and `judges candidate` use the same typed execution path.
`run_evidence.SavedRun` reads metadata once and verifies each call against its
planned fingerprint before execution. `RunEvidence` carries the recorded task,
policy, language, participant genders, injected opener and audio reference. A
saved run with no recorded language is treated as English (`en`).
Implementations select
caller-visible task fields for prompts; evaluator goldens and task rewards are not
judge context.

`CriteriaJudge` executes naturalness and language diagnostics, including calibrated
peer batches and the Hindi precheck. `ToolUseJudge` combines its three criteria.
`FidelityJudge` consumes aligned audio and preserves the clip-relative final-second
filter. Each returns `MeasureVerdict`; the suite checkpoints typed verdicts and
errors, and summary/export reports coverage separately from task success.

Utterance judges keep successful units alongside explicit `unit_errors`. A Live
speech group without a usable transcript is unjudged; it does not block the other
utterances or count as a pass/no-opportunity unit. Malformed and token-limited
responses retain their response text, finish reason and usage when available.
Matching partial verdicts retry only failed units. Summaries report partial calls
and error-unit counts separately, and never mark partial coverage complete.
Call-level criteria still require complete input. Checkpoints are saved after each
call/measure, so interrupting a running call can still lose its unsaved work.

Shared transcript reconstruction supplies delivered speech. Presentation wording
is part of the calibrated input: changing an interruption marker also requires a
new definition/prompt version and held-out validation before replacing a production
contract. Existing approvals do not automatically apply to that change.
