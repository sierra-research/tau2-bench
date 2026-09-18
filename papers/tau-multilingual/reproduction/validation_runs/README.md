# Validation-only historical artifacts

`pre-retail-name-role-v1/` preserves the paper artifacts tied to the original
Korean and Mandarin retail calls. They are historical validation inputs, not
active benchmark outputs:

- `audit.json` is the byte-identical pre-replacement cohort audit.
- `experience.json` is the byte-identical bootstrap manifest required by the
  archived naturalness replay.
- `tau_multilingual_experience_without_fluency_2026-09-03.json` is the
  byte-identical pre-replacement analysis formerly stored under
  `data/analysis/`.

The active paper figures read corrected metrics from the 2026-09-18 analysis
and corrected latency values from `../latency.json`. The latency artifact
recomputes all 90 active trial-zero voice cells, records the 80 nonreplaced cells
as shared inputs, and compares archived versus corrected bytes for the ten
replaced Korean and Mandarin retail cells.

`intermediate-trial0-results-binding-v1/` preserves the exact corrected
Experience analysis used as an input when the canonical naturalness sidecar was
rebound. It is a predecessor snapshot, not the active analysis; keeping it at a
versioned path makes the immutable replacement receipt independently
verifiable without introducing a forward-reference cycle.
