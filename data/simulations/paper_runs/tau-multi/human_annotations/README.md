# Tau-multilingual human annotations

This archive co-locates judge validation, first-critical-error review, and
supporting native-speaker review. Validation and FCE cover English (`en`) and
the five localized languages: Spanish (`es`), Brazilian Portuguese (`pt`),
Hindi (`hi`), Korean (`ko`), and Mandarin Chinese (`zh`). The miscellaneous
language-review tables remain scoped to the five localized languages. Audio is
not duplicated here.

## Layout

- `validations/README.md`: guide to the sole judge-validation archive.
- `validations/index.csv`: the paper-facing entry for each retained language and
  measure, including human-positive support and validation-gate status.
- `validations/<call|utterance>_level/<measure>/<language>/validation.csv`: one
  final human label and stored judge verdict per comparison unit.
- `validations/<call|utterance>_level/<measure>/<language>/metrics.csv`: the
  confusion matrix, precision, recall, F1, and Cohen's kappa derived from the
  sibling validation rows.
- `validations/<call|utterance>_level/<measure>/<language>/manifest.json`: the
  leaf package's schema and artifact hashes.
- `validations/manifest.json`: the complete factor/language package inventory.
- `validations/prompts.json`: the sole learned-judge prompt inventory, containing
  only the exact contracts that produced the frozen verdicts.
- `user_sim_review/fce/`: final call reviews for English and the five localized
  languages.
- `misc/`: final supporting language-review and call-experience evidence for
  the five localized languages.
- `manifest.json`: the inventory and hashes for the complete human-annotation
  archive.

There is no sibling validation tree and no second normalized copy. Each index
row resolves to one factor/language leaf package under `validations/`.

Rows are compared at utterance level when the human label and stored judge
verdict identify an utterance unit. Factors whose evidence applies to a whole
call remain call-level. Each comparison unit is represented once. `held_out`,
`fixed_curated`, `recall`, and `precision` describe validation design or
sampling strata; they are not directory levels.

The combined-naturalness measure uses the runtime factor id
`natural_word_choice`; its language-specific rubric covers natural word choice,
translation-shaped phrasing, and applicable grammar or regional clauses. Only
its final `held_out` rows are shipped. A frozen multi-factor contract may mention
criteria outside the retained measure set when those criteria shared the
original batched judge call; unused outputs are not shipped as measures.

The index retains low-support cohorts with an explicit gate status, while the
paper reports only measures that pass. Retired measures are excluded.
Deterministic checks, such as email-symbol verbalization, are verified in code
rather than presented as human-validated learned judges.

Validate the archive without network or model calls:

```bash
uv run tau2 judges tau-multi-validation data/simulations/paper_runs/tau-multi/human_annotations
```
