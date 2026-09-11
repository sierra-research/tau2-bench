# Human failure validation: final 90-call set

This directory contains final structured human failure labels for exactly 90
reward-zero calls: 30 each from OpenAI, Gemini, and xAI systems. It contains no
free-text notes, speech-fidelity fields, transcript text, audio, or intermediate
review state.

## Files

- `calls.csv` contains stable call identifiers and final failure-source and
  failure-subtype labels.
- `metrics.json` records aggregate counts and the SHA-256 digest of `calls.csv`.

Four calls have `error_source=unresolved`. Within the 81 agent-attributed calls,
15 have `error_subtype=unresolved`. No additional source or subtype is inferred
for those rows.

## CSV schema

| Column | Meaning |
| --- | --- |
| `schema_version` | Always `tau-elicit-human-failure-call-v2`. |
| `validation_set_id` | Always `human_failure_validation_90`. |
| `provider` | `openai`, `gemini`, or `xai`. |
| `simulation_id` | Stable simulation UUID. |
| `task_id` | Stable benchmark task identifier. |
| `bank` | Entity-bank family. |
| `tier` | `easy` or `hard`. |
| `reward` | Always `0`. |
| `error_source` | Final `agent`, `user`, `system`, `no_error`, or `unresolved` label. |
| `error_subtype` | Final subtype, `unresolved`, or empty when no subtype applies. |
