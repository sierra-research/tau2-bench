# Retail localization-ablation transcripts

This directory contains a compact transcript record for every call supporting
the Hindi and Mandarin retail localization ablation. The 360 JSONL rows cover
two languages, two voice systems, three unique call conditions, and the fixed
30-task retail frame. The baseline condition supports both the localized-task
and romanized-database columns in the paper table, so it is not duplicated.

Each row includes the recorded outcome, termination reason, delivered
caller/agent turns, interruption markers, and hashes of its source result and
simulation files. Raw tick streams and audio remain in the separately
distributed frozen result bundle.

Rebuild from that bundle with:

```bash
tau2 paper multilingual-ablation-transcripts \
  --evidence-root /path/to/tau-multi \
  --out papers/tau-multilingual/reproduction/retail_ablation_transcripts
```

Verify the checked-in JSONL and manifest offline with:

```bash
tau2 paper multilingual-verify-ablation-transcripts \
  --root papers/tau-multilingual/reproduction/retail_ablation_transcripts
```
