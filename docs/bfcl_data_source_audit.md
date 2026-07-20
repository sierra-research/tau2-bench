# BFCL Data Source Audit

## Source Artifacts

- Sample labels: `data/processed/bfcl/bfcl_v4_non_live_1240_xy.jsonl`
- Source summary: `data/processed/bfcl/bfcl_v4_non_live_1240_summary.json`

## Validated Facts

- Dataset: BFCL v4 non-live single-turn subset
- Model: `gpt-4o-mini-2024-07-18-FC`
- Total rows: 1,240
- Label scope: `test_case_level`
- Label origin: `bfcl_evaluator`
- Synthetic rows: `False`
- Y distribution: `{'0': 181, '1': 1059}`
- X representation field: `x_raw`
- S representation field: `s_raw`

| Category | n | Y=1 | Y=0 | Success rate |
|---|---:|---:|---:|---:|
| irrelevance | 240 | 199 | 41 | 0.8292 |
| multiple | 200 | 176 | 24 | 0.8800 |
| parallel | 200 | 174 | 26 | 0.8700 |
| parallel_multiple | 200 | 160 | 40 | 0.8000 |
| simple_python | 400 | 350 | 50 | 0.8750 |

## Audit Decisions

- Treat each BFCL row as one evaluator-labeled test case.
- Preserve `label_scope=test_case_level` and `label_origin=bfcl_evaluator` in derived outputs.
- Define candidate groups from category metadata only.
- Exclude `y`, `s_raw`, evaluator error fields, `label_scope`, `label_origin`, and `is_synthetic` from group membership rules.
- Do not pool these BFCL rows with tau2 or API-Bank as IID samples.
- Do not use partial BFCL leaderboard overall scores; all estimates come from the 1,240 sample-level labels.

## Limitations

This audit verifies the local processed artifacts and their internal counts. It does not re-run the upstream BFCL evaluator or establish causal relationships between category membership and correctness.
