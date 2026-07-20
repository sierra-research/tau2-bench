# BFCL Shift Inventory

## Purpose

Define reproducible candidate source and target groups for the BFCL v4 non-live single-turn context-shift analysis. Groups are defined only from BFCL category metadata and never from `y`.

## Dataset

- Dataset: BFCL v4 non-live single-turn subset
- Model: `gpt-4o-mini-2024-07-18-FC`
- Rows: 1,240
- Label scope: `test_case_level`
- Label origin: `bfcl_evaluator`
- Synthetic rows: `False`
- Outcome: `bfcl_test_case_correctness`
- X representation field: `x_raw`
- S representation field: `s_raw`

## Category Summary

| Category | n | Y=1 | Y=0 | Success rate |
|---|---:|---:|---:|---:|
| irrelevance | 240 | 199 | 41 | 0.8292 |
| multiple | 200 | 176 | 24 | 0.8800 |
| parallel | 200 | 174 | 26 | 0.8700 |
| parallel_multiple | 200 | 160 | 40 | 0.8000 |
| simple_python | 400 | 350 | 50 | 0.8750 |

## Candidate Shifts

Primary complexity shifts:

- `bfcl_simple_python_to_multiple`
- `bfcl_simple_python_to_parallel`
- `bfcl_simple_python_to_parallel_multiple`
- `bfcl_multiple_to_parallel_multiple`
- `bfcl_parallel_to_parallel_multiple`

Separately labeled behavioral/abstention shift:

- `bfcl_simple_python_to_irrelevance`

## Inventory Table

| Shift | Type | Source n | Target n | Source rate | Target rate | Delta_y |
|---|---|---:|---:|---:|---:|---:|
| bfcl_simple_python_to_multiple | primary_complexity | 400 | 200 | 0.8750 | 0.8800 | 0.0050 |
| bfcl_simple_python_to_parallel | primary_complexity | 400 | 200 | 0.8750 | 0.8700 | -0.0050 |
| bfcl_simple_python_to_parallel_multiple | primary_complexity | 400 | 200 | 0.8750 | 0.8000 | -0.0750 |
| bfcl_multiple_to_parallel_multiple | primary_complexity | 200 | 200 | 0.8800 | 0.8000 | -0.0800 |
| bfcl_parallel_to_parallel_multiple | primary_complexity | 200 | 200 | 0.8700 | 0.8000 | -0.0700 |
| bfcl_simple_python_to_irrelevance | behavioral_abstention | 400 | 240 | 0.8750 | 0.8292 | -0.0458 |

## Constraints

- This is exploratory analysis only.
- The inventory makes no causal, deployment-safe, or retraining-required claims.
- BFCL rows are not mixed with tau2 or API-Bank as IID samples.
- `label_scope` and `label_origin` are preserved in the outputs.
- Candidate groups are defined from `category`; `y`, evaluator errors, and model responses are not used for membership.
- The analysis relies on the 1,240 sample-level labels, not partial leaderboard overall scores.
