# tau2 L2T Formulation - 2026-07-14

## 1. Problem setup

The goal is to apply Minxing's Learning-to-Test (L2T) / distribution-shift
testing setup to tau2-bench LLM tool-calling simulations.

Each tau2 example is a completed customer-service task simulation. The agent
receives a domain policy, interacts with a user simulator, calls tools, and is
scored by tau2 task evaluation. We want to convert these simulations into an
L2T-style dataset where a tester can identify source-to-target shifts under
which task success drops.

In this formulation, the unit of prediction is a task simulation, while the
unit of harmfulness is a group shift.

## 2. Mapping tau2 to X, S, Y

- `X`: task/context features available before or independent of the realized
  trajectory. Current examples include domain, expected action counts, expected
  read/write counts, whether a DB mutation is required, assertion types, and
  reward-basis indicators.
- `S`: the tool-calling trajectory, represented by the sequence of
  user/assistant/tool events. The current pilot uses a fixed-length trajectory
  encoding derived from the simulation messages and tool calls.
- `Y`: task-level success. The binary label is `y = 1` if
  `reward_info.reward == 1.0`, and `y = 0` otherwise.

Thus, L2T sees task features `X`, trajectory signal `S`, and an observed
success label `Y`.

## 3. Task-level dataset definition

The current filtered task-level dataset is:

```text
data/processed/tau2_l2t_success_retail_airline_50_filtered_20260714.pkl
```

It contains normal-stop retail and airline simulations from the current
`gpt-4o-mini` pilot:

- Filtered total: `N = 93`
- Retail retained: `46 / 50`
- Airline retained: `47 / 50`
- `X` shape: `(93, 12)`
- `y` shape: `(93,)`
- `traj["s"]` shape: `(93, 64)`
- Positive count: `33`
- Overall success rate: `35.5%`

Abnormal simulations such as `max_steps` and `too_many_errors` were removed
when evaluation details were null or incomplete. This keeps the task-level
dataset focused on completed simulations with usable labels.

## 4. Shift-level harmfulness definition

A shift is a source group -> target group comparison over task simulations.
Groups can be defined by domain, expected task structure, trajectory
complexity, or other interpretable features.

For a candidate shift, harmfulness is measured as:

```text
drop_pp = 100 * (P(y = 1 | source) - P(y = 1 | target))
```

A positive `drop_pp` means the target group has lower task success than the
source group. In the current pilot summaries, a shift is marked as a harmful
candidate when `drop_pp > 10`, but this threshold is provisional.

## 5. Candidate shift families

The tau2 setting naturally supports several source -> target shift families:

- Domain shifts, e.g. `retail -> airline`.
- Write-burden shifts, e.g. zero/one expected write action -> two or more
  expected write actions.
- Read/write composition shifts, e.g. read-only or mostly read-only tasks ->
  tasks requiring state-changing tool calls.
- Tool-complexity shifts, e.g. few tool calls -> many tool calls.
- Trajectory-complexity shifts, e.g. short message trajectories -> long
  message trajectories.
- Assertion/reward-basis shifts, e.g. tasks checked mainly by DB state ->
  tasks involving communication or natural-language assertions.

These are interpretable shift definitions that can be computed from task
metadata, expected actions, evaluation assertions, and observed trajectories.

## 6. Pilot findings

The shift-level summary/dataset identifies several harmful-shift candidates in
the filtered `N = 93` pilot.

The strongest current shift is:

```text
zero or one expected write -> two or more expected writes
drop_pp = 36.59 pp
source success = 44.93%
target success = 8.33%
```

Other observed drops include:

- Short messages -> long messages: `17.08 pp`
- No expected write actions -> expected write actions > 0: `14.81 pp`
- Few tool calls -> many tool calls: `12.61 pp`
- Retail -> airline: `11.52 pp`

The L2T smoke test also succeeded: the tau2-derived pickle loaded in
Minxing's `run_baseline.py`, the feature dimension was recognized as `d_x =
12`, and the proposed model trained for 5 epochs. This validates the current
bridge from tau2 `results.json` outputs to an L2T-compatible dataset.

## 7. 30-epoch L2T pilot

A longer L2T pilot used the same task-level dataset:

```text
/Users/xuyida/Research/llm-toolcalling-benchmarks/tau2-bench/data/processed/tau2_l2t_success_retail_airline_50_filtered_20260714.pkl
```

and wrote outputs to:

```text
/Users/xuyida/Research/physics_informed_testing/share_code/results/tau2_l2t_success_retail_airline_50_filtered_20260714_epochs30
```

The run used `proposed_only`, `epochs=30`, `batch_train=16`, `batch_val=16`,
`skip_sweeps=True`, `num_workers=0`, and `time_window=0.1 3.0`. The dataset
loaded successfully, `d_x` was overridden from 2 to 12, and the recognized
shapes were `X = (93, 12)` and `trajectory = (93, 64)`.

Endpoint metrics:

- Epoch 1: `train-loss=36.175673`, `train-recon=2.269067`,
  `train-div=0.724943`, `train-acc=0.6250`, `val-recon=2.332024`,
  `val-div=0.654115`, `val-acc=0.6316`.
- Epoch 30: `train-loss=3.042825`, `train-recon=2.411330`,
  `train-div=0.999976`, `train-acc=0.6406`, `val-recon=2.574448`,
  `val-div=1.000000`, `val-acc=0.6316`.

Interpretation: this is still a pilot, not a final benchmark. The longer run
confirms that the tau2-derived dataset can be used for longer L2T training,
but validation accuracy stayed around `0.6316`, so this does not show strong
performance improvement yet. Because `N = 93` is small and `skip_sweeps=True`,
the result should be treated as a feasibility check rather than a
model-performance claim.

## 8. Limitations

This is a pilot, not a benchmark-level result.

- The current model is `gpt-4o-mini`, whose baseline success is low in this
  setup.
- The filtered dataset is small (`N = 93`) and only covers retail and airline.
- Shift definitions are descriptive group comparisons, not causal claims.
- Some groups are defined using observed trajectory quantities such as message
  count or tool-call count; these may be useful for analysis but are not purely
  pre-task covariates.
- Abnormal terminations were filtered out, so separate analysis is needed if
  max-step failures or tool-error loops are considered part of harmfulness.

## 9. Next experimental directions

1. Treat write-burden shift as the first primary tau2 harmful-shift candidate,
   especially zero/one expected write -> two or more expected writes.
2. Replicate the same shift definitions with a stronger baseline model and
   larger samples before making benchmark claims.
3. Separate pre-task shifts based only on `X` from post-hoc trajectory shifts
   involving realized `S`.
4. Add more domains once their runs complete reliably, especially if telecom
   max-step behavior can be controlled.
5. Evaluate whether L2T can learn to prioritize target groups with high
   `drop_pp`, not merely predict individual task failure.
