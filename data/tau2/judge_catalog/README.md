# Reviewed naturalness candidates

These five `CandidateJudge` entries incorporate the wording approved in the
`naturalness-language-criteria-review-v2` review.
They are **reviewed and runnable, but unvalidated**. The packaged production
naturalness judge and every specific judge remain unchanged. After a separate
validation round, these candidates are intended to replace naturalness for their
validated language/scope; they do not replace the specific diagnostics.

| Language | Candidate file |
| --- | --- |
| Spanish (Madrid) | `naturalness-es-nativeness-rubric-v21-language-specific-draft2.json` |
| Portuguese (Brazil) | `naturalness-pt-nativeness-rubric-v21-language-specific-draft2.json` |
| Hindi | `naturalness-hi-nativeness-rubric-v21-language-specific-draft2.json` |
| Korean (South Korea) | `naturalness-ko-nativeness-rubric-v21-language-specific-draft2.json` |
| Mandarin (Mainland China) | `naturalness-zh-nativeness-rubric-v21-language-specific-draft2.json` |

All criterion fields and the prompt wrapper match the approved review exactly.
The existing compiler assigns each locale a distinct contract ID using the
approved rubric revision `nativeness-rubric-v21-language-specific-draft2`.
The `draft2` suffix identifies the reviewed wording, not a validation result.
The judge remains `gpt-5.5` with `reasoning_effort=xhigh`, using one delivered
agent utterance, preceding customer text and recorded participant genders.
No English naturalness criterion is added.

The proposed scope is airline, retail, telecom, banking_knowledge and intake,
with either agent gender. That is a candidate scope only: existing Portuguese
female-agent and Hindi male-agent validation does not transfer. Portuguese
call-wide register drift and Mandarin call-wide particle patterns remain covered
by the existing specific judges, since no call history is added to naturalness.

The authoring envelope records the author and the wording review; these were
directly authored and reviewed prompt edits, not a new `draft-judge` model
invocation. Compilation and registration made no model calls. The review
digest binds the approved wording and its source/proposed execution metadata;
it is not human accuracy-label evidence.

The existing CLI discovers these entries automatically:

```sh
tau2 factory list-judges
```

To preview applicability for a saved run without executing any judge:

```sh
tau2 judges candidate saved-results \
  --candidate data/tau2/judge_catalog/naturalness-es-nativeness-rubric-v21-language-specific-draft2.json \
  --output candidate-runs/naturalness-es-v21 --dry-run
```

Remove `--dry-run` only when starting the separately planned validation work.
Candidate output must remain separate from saved runs and production diagnostics.
See [judge validation](../../../docs/judge-validation.md) for the existing
held-out validation and promotion workflow. Supplying these files to the
production `--registry` option is rejected because review alone is not validation.
