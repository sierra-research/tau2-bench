# Speech-fidelity validation: final 60-utterance set

This directory contains final utterance-level human reference labels and speech-
fidelity judge predictions for exactly 60 utterances, split evenly between Gemini
and xAI. It contains no transcript text, audio, free-text notes, individual review
state, or intermediate labeling metadata.

## Files

- `utterances.csv` contains stable utterance identifiers, final binary human
  labels, judge predictions at two operating points, and confusion cells.
- `metrics.json` records aggregate counts, metrics, and the SHA-256 digest of
  `utterances.csv`.

The primary operating point is `judge_severity_ge_2_positive`: the maximum
retained fidelity-finding severity is at least 2. It yields TP=12, FP=3, FN=4,
TN=41, precision=0.800, recall=0.750, and F1=0.7741935483870968.

The secondary operating point, `judge_any_finding_positive`, means that at least
one fidelity finding survives filtering regardless of severity. It yields TP=13,
FP=4, FN=3, TN=40, precision=0.7647058823529411, recall=0.8125, and
F1=0.7878787878787878.

## CSV schema

| Column | Meaning |
| --- | --- |
| `utterance_id` | Stable composite of simulation UUID and utterance index. |
| `provider` | `gemini` or `xai`. |
| `simulation_id` | Stable simulation UUID. |
| `task_id` | Stable benchmark task identifier. |
| `utterance_idx` | Stable utterance index within the simulation. |
| `human_fidelity_positive` | Final binary human reference label. |
| `judge_any_finding_positive` | Secondary any-retained-finding prediction. |
| `judge_max_fidelity_severity` | Maximum retained fidelity severity, or 0. |
| `judge_severity_ge_2_positive` | Primary severity-at-least-2 prediction. |
| `confusion_severity_ge_2` | Primary TP/FP/FN/TN cell. |
| `confusion_any_finding` | Secondary TP/FP/FN/TN cell. |
