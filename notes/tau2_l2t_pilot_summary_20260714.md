# tau2 L2T Pilot Summary - 2026-07-14

## 1. Goal

Summarize the current tau2-to-L2T pilot for Minxing. The goal was to check
whether tau2 simulation outputs can be converted into the `.pkl` format used
by the Learning-to-Test code, and to identify preliminary harmful-shift
candidates from the filtered retail/airline pilot.

This is a pilot, not a final benchmark. The current `gpt-4o-mini` baseline is
low, so the results should be treated as a pipeline validation and source of
candidate shift definitions.

## 2. tau2 runs used

Primary filtered dataset:

- Retail 50 run:
  `data/simulations/20260714_131548_retail_llm_agent_gpt-4o-mini_user_simulator_gpt-4o-mini`
- Airline 50 run:
  `data/simulations/20260714_140540_airline_llm_agent_gpt-4o-mini_user_simulator_gpt-4o-mini`
- Agent/user model: `gpt-4o-mini`
- Trials per task: 1
- Raw target size: 50 retail + 50 airline simulations

Earlier 10-task retail/airline runs were useful for initial inspection, but
the current summary uses the filtered 50-task retail/airline pilot.

Telecom was attempted but not continued. The default telecom tasks repeatedly
hit `max_steps`, making the run costly and low-yield for this first L2T pilot.

## 3. Filtering abnormal simulations

The clean dataset keeps only normal-stop simulations with usable evaluation
details.

- Retail retained 46 / 50 simulations.
- Airline retained 47 / 50 simulations.
- Filtered total: `N = 93`.

Excluded simulations included `max_steps` and `too_many_errors` cases where
evaluation details such as `db_check` or `action_checks` were null or
incomplete. These were removed so the pilot summary compares completed
simulations rather than mixing in abnormal terminations.

## 4. Task-level L2T dataset

Generated pickle:

```text
data/processed/tau2_l2t_success_retail_airline_50_filtered_20260714.pkl
```

Dataset format:

- `X`: task/context features, shape `(93, 12)`
- `y`: task-level binary success label, shape `(93,)`
- `traj["s"]`: fixed-length trajectory encoding, shape `(93, 64)`

Important label convention: `y = 1` means task success. It does not mean
"harmful shift." In this pilot, `y = 1` if `reward_info.reward == 1.0`, else
`0`.

## 5. L2T smoke test result

The L2T-compatible pickle loaded and trained successfully in Minxing's
`run_baseline.py`.

Smoke command used the filtered pickle with `--models proposed_only`,
`--epochs 5`, `--batch-train 16`, `--batch-val 16`, `--skip-sweeps`, and
`--num-workers 0`.

Observed result:

- The external tau2-derived dataset loaded successfully.
- `d_x` was overridden from 2 to 12.
- Shapes were recognized as `X = (93, 12)` and `traj["s"] = (93, 64)`.
- The proposed model trained successfully for 5 epochs.

This verifies the bridge:

```text
tau2 results.json -> filtered L2T-compatible pickle -> run_baseline.py training
```

## 6. 30-epoch L2T pilot

A longer pilot run was also completed using the same filtered input pickle:

```text
/Users/xuyida/Research/llm-toolcalling-benchmarks/tau2-bench/data/processed/tau2_l2t_success_retail_airline_50_filtered_20260714.pkl
```

Output directory:

```text
/Users/xuyida/Research/physics_informed_testing/share_code/results/tau2_l2t_success_retail_airline_50_filtered_20260714_epochs30
```

The command used `proposed_only`, `epochs=30`, `batch_train=16`,
`batch_val=16`, `skip_sweeps=True`, `num_workers=0`, and
`time_window=0.1 3.0`.

Observed setup and endpoints:

- The dataset loaded successfully.
- `d_x` was overridden from 2 to 12.
- Shapes were recognized as `X = (93, 12)` and `trajectory = (93, 64)`.
- Epoch 1: `train-loss=36.175673`, `train-recon=2.269067`,
  `train-div=0.724943`, `train-acc=0.6250`, `val-recon=2.332024`,
  `val-div=0.654115`, `val-acc=0.6316`.
- Epoch 30: `train-loss=3.042825`, `train-recon=2.411330`,
  `train-div=0.999976`, `train-acc=0.6406`, `val-recon=2.574448`,
  `val-div=1.000000`, `val-acc=0.6316`.

This remains a pilot, not a final benchmark. The longer run confirms that the
tau2-derived dataset can be used for longer L2T training, but validation
accuracy stayed around `0.6316`, so it does not yet show strong performance
improvement. Because `N = 93` is small and `skip_sweeps=True`, the result
should be treated as a feasibility check rather than a model-performance
claim.

## 7. Shift-level harmfulness definition

Harmfulness is computed at the shift level, not from an individual task label.

For a candidate source group and target group:

```text
drop_pp = 100 * (P(y = 1 | source) - P(y = 1 | target))
```

A positive `drop_pp` means the target group has lower task success than the
source group. In this pilot summary, `harmful_candidate = true` when
`drop_pp > 10`.

## 8. Top harmful-shift candidates

Source table:
`data/processed/tau2_shift_level_summary_20260714.csv`

| Rank | Shift | Source success | Target success | Drop |
| ---: | --- | ---: | ---: | ---: |
| 1 | zero or one expected write -> two or more expected writes | 44.93% | 8.33% | 36.59 pp |
| 2 | short messages -> long messages | 43.75% | 26.67% | 17.08 pp |
| 3 | no expected write actions -> expected write actions > 0 | 46.15% | 31.34% | 14.81 pp |
| 4 | few tool calls -> many tool calls | 41.18% | 28.57% | 12.61 pp |
| 5 | retail -> airline | 41.30% | 29.79% | 11.52 pp |

The top candidate is the shift from zero/one expected write action to two or
more expected write actions, with a 36.59 percentage-point drop.

The domain shift `retail -> airline` has an 11.52 percentage-point drop, but
it is not the strongest shift in this pilot.

## 9. Interpretation

The strongest current signal is not domain change alone. The larger drops are
associated with increased write-action burden and interaction complexity:
more expected writes, longer conversations, and more tool calls.

This suggests a useful near-term harmful-shift framing for tau2: compare
simpler or mostly read-only tasks against tasks requiring multiple
state-changing tool calls. The current evidence is descriptive and based on a
small filtered pilot, so it should not be interpreted causally or as final
benchmark performance.

## 10. Next steps

1. Confirm with Minxing that the task-level label convention is correct:
   `y = 1` means task success.
2. Decide whether the first harmful-shift experiment should use expected-write
   burden rather than domain as the primary source/target split.
3. Rerun with a stronger baseline model or larger sample before claiming a
   benchmark-level harmful shift.
4. Keep abnormal termination filtering explicit and reproducible.
5. If telecom is revisited, adjust task selection or run settings first so it
   does not repeatedly hit `max_steps`.
