# Failure distribution (F1-F6)

Counts and percentage of failed trials. Uses final_code where set, otherwise the first-pass proposal.
Extra and duplicate records are coded under F5, together with missing, partial, and wrong-stage mutations.

| Model | Failed | F1 | F2 | F3 | F4 | F5 | F6 | Reviewed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DeepSeek chat | 28 | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 28 (100.0%) | 0 (0.0%) | 0/28 |
| GPT-OSS 20B | 28 | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 12 (42.9%) | 12 (42.9%) | 4 (14.3%) | 0/28 |
| Phi-4-mini 3.8B | 80 | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 80 (100.0%) | 0/80 |
| Qwen 3.5 4B | 21 | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 21 (100.0%) | 0 (0.0%) | 0/21 |
| Qwen 3.5 9B | 15 | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 1 (6.7%) | 14 (93.3%) | 0 (0.0%) | 0/15 |
| **All models** | **172** | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 13 (7.6%) | 75 (43.6%) | 84 (48.8%) | 0/172 |

Worksheet: `failure_distribution/failures_to_code.csv`. Set `final_code` to F1-F6 and rerun with `--review-csv`.
