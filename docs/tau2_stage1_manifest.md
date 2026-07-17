# Tau2 Stage 1 Manifest

## Selection

This dry-run manifest selects exactly 12 unused retail tasks from `data/processed/tau2_additional_sampling_plan.json`. Selection is deterministic, records seed `20260717`, excludes telecom, and does not use observed `y`, reward, or prior success/failure labels.

| Task ID | Group | Actions | Reads | Writes | DB mutation | Previously attempted | Previously retained |
|---:|---|---:|---:|---:|---|---|---|
| 54 | two_plus_writes | 12 | 9 | 3 | True | False | False |
| 55 | two_plus_writes | 13 | 9 | 4 | True | False | False |
| 64 | two_plus_writes | 8 | 6 | 2 | True | False | False |
| 71 | two_plus_writes | 2 | 0 | 2 | True | False | False |
| 72 | two_plus_writes | 2 | 0 | 2 | True | False | False |
| 74 | two_plus_writes | 2 | 0 | 2 | True | False | False |
| 76 | two_plus_writes | 2 | 0 | 2 | True | False | False |
| 81 | two_plus_writes | 2 | 0 | 2 | True | False | False |
| 57 | no_write | 0 | 0 | 0 | False | False | False |
| 62 | no_write | 5 | 5 | 0 | False | False | False |
| 50 | low_action_one_write | 1 | 0 | 1 | True | False | False |
| 70 | low_action_one_write | 1 | 0 | 1 | True | False | False |

## Composition

- `two_plus_writes`: 8
- `no_write`: 2
- `low_action_one_write`: 2

## Illustrative Batch Tau2 Command

This command documents the selected task IDs and model/runtime settings in one
tau2 invocation. The executed Stage 1 path is the runner below, which launches
one task per subprocess and writes one raw JSON copy per task.

```bash
uv run tau2 run --domain retail --agent llm_agent --user user_simulator --agent-llm gpt-4o-mini --user-llm gpt-4o-mini --num-trials 1 --task-ids 54 55 64 71 72 74 76 81 57 62 50 70 --max-concurrency 1 --seed 20260717 --log-level DEBUG --verbose-logs --llm-log-mode all --auto-resume --save-to tau2_stage1_raw/stage1_retail_12
```

## Runner Commands

```bash
uv run python scripts/run_tau2_stage1.py
uv run python scripts/run_tau2_stage1.py --execute
```

## Runtime And Cost Controls

- Output directory: `data/processed/tau2_stage1_raw`
- Native tau2 output prefix: `data/simulations/tau2_stage1_raw/`
- One trial per task: `True`
- Maximum concurrency: `1`
- Estimated maximum LLM calls: `102`

## Stop Conditions

- Stop before execution unless --execute is explicitly supplied.
- Stop on any missing manifest task or task outside retail.
- Stop if a selected task was previously attempted or retained.
- Stop if cumulative observed cost reaches an operator-supplied --max-total-cost.
- Stop after the first subprocess failure unless --continue-on-error is supplied.
