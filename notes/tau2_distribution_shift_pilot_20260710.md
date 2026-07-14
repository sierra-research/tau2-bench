# tau2-bench Distribution Shift Pilot Notes

Date: 2026-07-10  
Repo: tau2-bench  

## Goal

Apply Minxing Zheng's physics-informed / Learning-to-Test code to
tau2-bench tool-calling results.

The near-term goal is to understand both sides of the pipeline:

- produce clean tau2-bench simulation results;
- verify that Minxing's L2T code runs locally;
- eventually convert tau2-bench `results.json` outputs into the personal
  `.pkl` input format expected by Minxing's code.

## tau2-bench Pilot Setup

OpenRouter/free runs had infrastructure and rate-limit problems, so the clean
pilot used OpenAI `gpt-4o-mini`.

Common setup:

- Agent: `llm_agent`
- User simulator: `user_simulator`
- Model: `gpt-4o-mini`
- Num trials: 1
- Num tasks per domain: 10
- Max concurrency: 1
- Verbose logs enabled
- API key configured using `.env`

Retail command:

```bash
uv run tau2 run \
  --domain retail \
  --agent-llm gpt-4o-mini \
  --user-llm gpt-4o-mini \
  --num-trials 1 \
  --num-tasks 10 \
  --max-concurrency 1 \
  --verbose-logs
```

Airline command:

```bash
uv run tau2 run \
  --domain airline \
  --agent-llm gpt-4o-mini \
  --user-llm gpt-4o-mini \
  --num-trials 1 \
  --num-tasks 10 \
  --max-concurrency 1 \
  --verbose-logs
```

## tau2-bench Results

Retail clean run folder:

```text
data/simulations/20260710_151552_retail_llm_agent_gpt-4o-mini_user_simulator_gpt-4o-mini
```

Retail result:

- Total simulations: 10
- Total tasks: 10
- Infra errors: 0
- Average reward: 0.3000
- Pass^1: 0.300
- Avg cost/conversation: $0.0057
- Read actions: 55/64 = 85.9%
- Write actions: 4/11 = 36.4%
- DB match: 4/10 = 40.0%
- Normal stop: 10

Airline clean run folder:

```text
data/simulations/20260710_153139_airline_llm_agent_gpt-4o-mini_user_simulator_gpt-4o-mini
```

Airline result:

- Total simulations: 10
- Total tasks: 10
- Infra errors: 0
- Average reward: 0.5000
- Pass^1: 0.500
- Avg cost/conversation: $0.0028
- Read actions: 12/21 = 57.1%
- Write actions: 1/4 = 25.0%
- DB match: 5/10 = 50.0%
- Normal stop: 10

Summary table:

| Domain | Tasks | Infra Errors | Avg Reward | Pass^1 | Avg Cost | Read Actions | Write Actions | DB Match | Stop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| retail | 10 | 0 | 0.3000 | 0.300 | $0.0057 | 55/64 = 85.9% | 4/11 = 36.4% | 4/10 = 40.0% | normal: 10 |
| airline | 10 | 0 | 0.5000 | 0.500 | $0.0028 | 12/21 = 57.1% | 1/4 = 25.0% | 5/10 = 50.0% | normal: 10 |

## Initial Interpretation

Retail reward was 0.30 and airline reward was 0.50.

Using retail as the source context and airline as the shifted context:

```text
Delta = R_retail - R_airline = 0.30 - 0.50 = -0.20
```

In this small pilot, changing domain from retail to airline did not create a
harmful performance drop. Airline performed better than retail.

This suggests that domain shift alone may not be sufficient to define a
harmful shift. Task and action difficulty may be more informative, especially
read-only versus write/state-changing tasks.

Retail read actions were relatively strong, but write actions were much
weaker. A useful next shift design may compare easier/read-only tasks against
tasks that require state-changing tool calls.

## Shift-Level Summary Table

Created:

- `scripts/build_shift_level_summary.py`
- `data/processed/tau2_shift_level_summary_20260714.csv`
- `notes/tau2_shift_level_summary_20260714.md`

Input:

```text
data/processed/tau2_l2t_success_retail_airline_50_filtered_20260714.pkl
```

There are two data levels in this pilot.

Task-level dataset:

- One row = one tau2 simulation.
- `X` = task/context features.
- `y` = task success.
- `traj["s"]` = trajectory encoding.
- `N = 93`.

Shift-level summary:

- One row = one source group -> target group comparison.
- Harmfulness is measured by
  `drop_pp = 100 * (P(y=1 | source) - P(y=1 | target))`.
- `harmful_candidate` is true if `drop_pp > 10` in this pilot.

Top harmful candidates:

| Rank | Source group -> target group | Drop |
|---:|---|---:|
| 1 | zero or one expected write -> two or more expected writes | 36.59 pp |
| 2 | short messages -> long messages | 17.08 pp |
| 3 | no expected write actions -> expected write actions > 0 | 14.81 pp |
| 4 | few tool calls -> many tool calls | 12.61 pp |
| 5 | retail -> airline | 11.52 pp |

Interpretation:

- The strongest harmful-shift candidate is increased write-action burden.
- Domain shift retail -> airline exists, but it is weaker than
  action-complexity and trajectory-complexity shifts.
- This suggests that for LLM tool-calling systems, harmful shifts may arise
  more from capability/interaction complexity than from domain change alone.
- The current pilot still has low baseline success with `gpt-4o-mini`, so it
  does not reproduce Minxing's example of 90% -> 50%, but it identifies
  promising shift directions.

## Minxing L2T Code Setup

Repo:

```text
https://github.com/Minxing-Zheng/physics_informed_testing
```

Local path:

```text
~/Research/physics_informed_testing/share_code
```

README meaning:

- The package is for collaborators to run `experiment/run_baseline.py`.
- It can run on built-in synthetic spring data or personal data provided as a
  `.pkl` file.
- Personal `.pkl` format should contain:
  - `X`: numpy array of shape `(N, d_x)`
  - `y`: numpy array of shape `(N,)`, binary 0/1
  - `traj["s"]`: numpy array of shape `(N, T)`

Local environment:

- Miniforge installed
- Conda version: 26.3.2
- Conda env: `l2t`

## L2T Smoke Test Result

Successful smoke test command:

```bash
python experiment/run_baseline.py \
  --models proposed_only \
  --epochs 5 \
  --n-samples 200 \
  --test-n-samples 100 \
  --batch-train 32 \
  --batch-val 32 \
  --skip-sweeps \
  --num-workers 0
```

Result:

- The script successfully parsed configuration.
- It generated synthetic spring data.
- Shapes: `X = (200, 2)`, trajectory shape = `(200, 201)`.
- Device: `mps`.
- Training reached and completed the proposed model for the smoke setting.
- Sweeps were skipped because `--skip-sweeps` was used.
- Outputs were saved to:

```text
/Users/xuyida/Research/physics_informed_testing/share_code/results/diagnostics_results_baselines/seed_1_lr_0p003_lambda_mmd_0p01_loss_type_stability_asymmetric_recon_tau_1_perturb_type_mean_epochs_5_n_200
```

Inspected output directory:

```text
~/Research/physics_informed_testing/share_code/results/diagnostics_results_baselines/seed_1_lr_0p003_lambda_mmd_0p01_loss_type_stability_asymmetric_recon_tau_1_perturb_type_mean_epochs_5_n_200
```

Files observed:

- `run_config.json`
- `training_history_proposed.csv`
- `train_reconstruction_snapshot_proposed.pkl`
- `val_reconstruction_snapshot_proposed.pkl`
- `summary_df.csv`
- `rejection_rate_summary.csv`
- `label_probability_summary.csv`
- `gap_df.csv`
- `gap_table.csv`
- `mean_summary.csv`
- `mean_summary.json`

Because `--skip-sweeps` was used, the summary/sweep-related CSV and JSON
files are mostly empty. This is expected for the smoke run.

Key `run_config.json` fields:

- `seed`: `1`
- `n_samples`: `200`
- `test_n_samples`: `100`
- `T`: `10.0`
- `dt`: `0.05`
- `stride`: `1`
- `loss_type`: `stability_asymmetric_recon`
- `tau`: `1.0`
- `time_window`: `[6.0, 10.0]`
- `batch_train`: `32`
- `batch_val`: `32`
- `epochs`: `5`
- `models`: `["proposed"]`
- `input_data`: `null`
- `skip_sweeps`: `true`
- `d_x`: `2`
- `d_o`: `1`

`training_history_proposed.csv` was generated successfully. It contains
per-epoch diagnostics including:

- `train_loss`
- `val_loss`
- `val_recon_raw`
- `val_divergence_raw`
- `train_recon`
- `train_divergence`
- `train_total_loss`
- `train_safety_acc`
- `val_safety_acc`
- `train_frac_y1` / `train_frac_y0`
- `val_frac_y1` / `val_frac_y0`

Observed training history:

- 5 epochs were completed.
- Epoch 1 `train_safety_acc` was `0.925` and `val_safety_acc` was `0.95`.
- Epoch 5 `train_safety_acc` was `0.9875` and `val_safety_acc` was `0.975`.
- This confirms the smoke run completed and produced usable training
  diagnostics.

Interpretation:

The L2T code is working locally on the built-in synthetic spring dataset. The
next step is not to run larger experiments yet, but to inspect the tau2
`results.json` structure and design a mapping into the required personal-data
`.pkl` format:

- `X`: tau2 task/context features
- `y`: binary task-success label
- `traj["s"]`: trajectory/time-series representation derived from tau2
  agent-user-tool interactions

Example training output:

```text
[proposed][01] train-loss=0.280736 train-recon=0.040267 train-div=0.024266 train-acc=0.9250 val-recon=0.033669 val-div=0.001445 val-acc=0.9500
```

Earlier failed attempt:

- I first tried the README-style setup with `n_samples=200` and `epochs=100`
  without specifying smaller batch sizes.
- It reached the training stage but failed with
  `ZeroDivisionError: float division by zero`.
- Likely cause: small sample size with default batch settings produced zero
  training batches.
- Adding `--batch-train 32` and `--batch-val 32` fixed the issue.

## tau2 results.json Structure and Proposed L2T Mapping

Inspected clean result files:

- `data/simulations/20260710_151552_retail_llm_agent_gpt-4o-mini_user_simulator_gpt-4o-mini/results.json`
- `data/simulations/20260710_153139_airline_llm_agent_gpt-4o-mini_user_simulator_gpt-4o-mini/results.json`

Top-level structure:

- `timestamp`
- `info`
- `tasks`
- `simulations`
- `simulation_index`

Important static task fields:

- task id
- `user_scenario`
- domain
- `reason_for_call`
- `known_info` / `unknown_info`
- `task_instructions`
- `evaluation_criteria`
- expected actions
- `env_assertions`
- `communicate_info`
- `nl_assertions`
- `reward_basis`

Important simulation/evaluation fields:

- `task_id`
- `duration`
- `termination_reason`
- `agent_cost` / `user_cost`
- `reward_info.reward`
- `reward_info.db_check.db_match`
- `reward_info.action_checks`
- `reward_info.communicate_checks`
- `reward_info.nl_assertions`
- `reward_info.reward_breakdown`
- `messages`

Message/tool fields:

- `role`
- `content`
- `tool_calls`
- `turn_idx`
- `timestamp`
- `cost`
- `usage`
- `raw_data`
- `generation_time_seconds`
- `error`
- `requestor`

Summary:

- Retail clean run has 10 simulations, all `user_stop`. Average reward was
  0.30.
- Airline clean run has 10 simulations, all `user_stop`. Average reward was
  0.50.
- Combining both gives `N = 20` simulation rows for a first toy L2T input
  dataset.

Proposed first-version L2T mapping:

`X`: one row per simulation. Candidate features:

- `domain_retail`
- `domain_airline`
- `expected_action_count`
- `expected_read_action_count`
- `expected_write_action_count`
- `requires_db_mutation`
- `has_communication_checks`
- `has_nl_assertions`
- `has_env_assertions`
- `reward_basis_has_DB`
- `reward_basis_has_COMMUNICATE`
- `reward_basis_has_NL_ASSERTION`
- `duration_seconds`
- `message_count`
- `assistant_message_count`
- `user_message_count`
- `tool_call_count`
- `tool_output_count`
- `termination_user_stop`

For a cleaner pre-run context `X`, exclude post-run features such as
`duration_seconds`, `message_count`, `tool_call_count`, and
`tool_output_count`. For a diagnostic/performance-modeling version, keep them.

`y`: first version should use task success:

```text
y = 1 if reward_info.reward == 1.0 else 0
```

Correct label interpretation from the meeting note with Minxing:

- Task-level `y = 1` means task success, reliable behavior, or correct
  completion.
- Task-level `y = 0` means task failure.
- Harmful shift is not the per-task label itself.
- Harmful shift is a distribution-level effect: the proportion of `y = 1`
  decreases under a shifted context.
- Minxing's "reduced 50%" example should be interpreted as an absolute
  target success-rate example, not as a 50% relative reduction from the source
  rate.
- In other words, "reduced to 50%" means the shifted/target context has
  `P(y = 1)` around 50%.

Harmfulness should therefore be computed by comparing success rates across
source and target distributions:

```text
source_success_rate = P(y = 1 | source)
target_success_rate = P(y = 1 | target)
drop = source_success_rate - target_success_rate
```

A harmful shift can be defined when `drop` is large, for example when the
target success rate falls to around 50% or when the drop exceeds a chosen
threshold. The exact threshold should be confirmed with Minxing.

Example: if source retail has `P(y = 1)` around 90%, but target airline
ticketing has `P(y = 1)` around 50%, then the success label proportion has
dropped and the shift may be harmful. This is a 90% -> 50% absolute
success-rate comparison, not a 90% -> 45% relative-halving example.

`traj["s"]`: first version can use a fixed-length integer event sequence,
padded/truncated:

```text
0 = pad
1 = user message
2 = assistant text
3 = assistant read tool call
4 = assistant write tool call
5 = tool output success
6 = tool output error
7 = terminal event
```

Build each row by walking `messages` in `turn_idx` order.

Missing/ambiguous:

- No direct per-turn action correctness annotation; `action_checks` are final
  evaluator results.
- Tool output content may be JSON string or scalar text.
- `ticks` and `effect_timeline` are null in these half-duplex runs.
- Harmful-shift assessment requires additional source/target comparison data
  and should be derived from differences in `P(y = 1)`.
- Some evaluator fields are nullable.

Interpretation:

The first practical converter should create a toy dataset with `N = 20` using
task success `y`. This can validate that tau2 results can be converted into
the required `.pkl` structure before constructing source/target groups and
measuring harmfulness through changes in `P(y = 1)`.

## tau2-derived L2T Input Smoke Test

Generated a toy tau2-derived L2T input pickle using:

```text
scripts/convert_tau2_results_to_l2t_pkl.py
```

Output pickle:

```text
data/processed/tau2_l2t_toy_success_20260710.pkl
```

Converter output summary:

- `X` shape: `(20, 12)`
- `y` shape: `(20,)`
- `traj["s"]` shape: `(20, 64)`
- Positive label count: `8`

Feature names:

- `domain_retail`
- `domain_airline`
- `expected_action_count`
- `expected_read_action_count`
- `expected_write_action_count`
- `requires_db_mutation`
- `has_communication_checks`
- `has_nl_assertions`
- `has_env_assertions`
- `reward_basis_has_DB`
- `reward_basis_has_COMMUNICATE`
- `reward_basis_has_NL_ASSERTION`

Then tested this tau2-derived pickle inside Minxing's L2T code.

Command run in:

```text
~/Research/physics_informed_testing/share_code
```

Command:

```bash
python experiment/run_baseline.py \
  --input-data /Users/xuyida/Research/llm-toolcalling-benchmarks/tau2-bench/data/processed/tau2_l2t_toy_success_20260710.pkl \
  --models proposed_only \
  --time-window 0.1 3.0 \
  --epochs 5 \
  --batch-train 8 \
  --batch-val 8 \
  --skip-sweeps \
  --num-workers 0 \
  --output-dir results/tau2_l2t_toy_success_20260710
```

Result:

- The L2T code successfully loaded the external tau2-derived dataset.
- It overrode `d_x` from `2` to `12`.
- It printed:
  `Shapes -> X: (20, 12) trajectory shape: (20, 64)`.
- It built train/validation loaders.
- It reached and ran `Training proposed`.
- Example output:

```text
[proposed][01] train-loss=38.627996 train-recon=3.201939 train-div=0.469861 train-acc=0.3750 val-recon=2.900763 val-div=0.138010 val-acc=0.5000
```

- It skipped perturbation sweeps because `--skip-sweeps` was used.
- It saved outputs to:

```text
/Users/xuyida/Research/physics_informed_testing/share_code/results/tau2_l2t_toy_success_20260710
```

Important interpretation:

This is not a meaningful performance result because `N = 20` is tiny and the
trajectory encoding is a first toy version. However, it proves that the
end-to-end bridge works:

```text
tau2 results.json -> L2T-compatible .pkl -> Minxing run_baseline.py
```

Also note:

The first attempt used `--time-window 0 63` and failed because L2T interprets
`time_window` as physical time, not timestep index. The valid range for
`T = 64` with `dt = 0.05` was `[0.0500, 3.1500]`. Rerunning with
`--time-window 0.1 3.0` fixed the issue.

## Retail 50 Pilot and Abnormal Simulation Filtering

Retail 50 run folder:

```text
data/simulations/20260714_131548_retail_llm_agent_gpt-4o-mini_user_simulator_gpt-4o-mini
```

Command setup:

- Domain: `retail`
- Model: `gpt-4o-mini`
- Agent: `llm_agent`
- User simulator: `user_simulator`
- Num trials: 1
- Num tasks: 50
- Max concurrency: 1
- Verbose logs: true

Reported summary:

- Total simulations: 50
- Total tasks: 50
- Average reward: 0.3800
- Pass^1: 0.380
- Avg cost/conversation: $0.0072
- Read actions: 188/245 = 76.7%
- Write actions: 35/62 = 56.5%
- DB match: 22 / 46 = 47.8%
- Normal stop: 46
- Max steps: 1
- Error: 3
- Runtime: about 36 minutes

Abnormal simulations:

- Task 11: `max_steps`
- Task 21: `too_many_errors`
- Task 34: `too_many_errors`
- Task 42: `too_many_errors`

Inspection result:

- All four abnormal simulations have `reward_info` present but `db_check` is
  null and `action_checks` is null.
- Task 11 hit `max_steps` after a long loop and had two tool errors with
  "Order not found".
- Task 21 had repeated invalid tool calls involving wrong delivered/pending
  order operations and reached `too_many_errors`.
- Task 34 had repeated invalid return/exchange/modify attempts and reached
  `too_many_errors`.
- Task 42 had invalid exchange item counts, variant not found, non-pending
  modify, repeated non-delivered returns, and reached `too_many_errors`.

Decision:

For clean distribution-shift analysis, exclude all four abnormal simulations.
Retain the 46 normal-stop simulations as the clean retail subset.

Do not treat the three `too_many_errors` cases as normal model failures in the
clean dataset because `db_check` and `action_checks` are unavailable. Do not
keep task 11 in the clean subset because it lacks usable `db_check` and
`action_checks` and hit `max_steps`.

Interpretation:

The reported full-run average reward was 0.38, but the clean analysis subset
should be based on 46 normal-stop simulations. The run is useful, but it
should be described as "retail 50 pilot with 46 completed/normal-stop
simulations retained after filtering 4 abnormal simulations."

## Airline 50 Pilot and Abnormal Simulation Filtering

Airline 50 run folder:

```text
data/simulations/20260714_140540_airline_llm_agent_gpt-4o-mini_user_simulator_gpt-4o-mini
```

Abnormal simulations:

- Task 6: `max_steps`
- Task 14: `too_many_errors`
- Task 47: `max_steps`

Inspection result:

- All three abnormal simulations had `reward = 0.0`.
- All three had `reward_info` present, but `db_check`, `action_checks`,
  `nl_assertions`, `communicate_checks`, and `reward_breakdown` were null or
  incomplete.
- Task 6 hit `max_steps`. The user persistently wanted insurance added. The
  agent should have refused, but instead entered a tool loop, repeatedly
  calling `update_reservation_passengers` many times after the initial
  reservation lookup.
- Task 14 reached `too_many_errors`. The agent failed payment/rebooking
  logic, tried to use a certificate while updating, and repeatedly attempted
  bookings with wrong payment totals/splits.
- Task 47 hit `max_steps`. The user wanted a full refund for cancellation due
  to a birthday conflict. The agent incorrectly kept assuring a full refund and
  asking for confirmation, while the user kept asking for certainty, leading
  to a loop.

Decision:

For clean distribution-shift analysis, exclude all three abnormal airline
simulations. They are premature/abnormal terminations with incomplete
evaluation details, zero rewards, and no usable `db_match` or `action_checks`.

Retain the 47 normal-stop simulations as the clean airline subset.

Current clean subsets:

- Retail: retain 46 normal-stop simulations, exclude tasks 11, 21, 34, and
  42.
- Airline: retain 47 normal-stop simulations, exclude tasks 6, 14, and 47.
- Combined filtered dataset target size: 93 simulations.

## Filtered Retail-Airline L2T Dataset Smoke Test

Input tau2 runs:

- retail:
  `data/simulations/20260714_131548_retail_llm_agent_gpt-4o-mini_user_simulator_gpt-4o-mini`
- airline:
  `data/simulations/20260714_140540_airline_llm_agent_gpt-4o-mini_user_simulator_gpt-4o-mini`

Filtering:

- Kept only normal-stop simulations with usable evaluation details.
- Retail retained 46 / 50.
- Airline retained 47 / 50.
- Combined N = 93.

Generated dataset:

```text
data/processed/tau2_l2t_success_retail_airline_50_filtered_20260714.pkl
```

Dataset format:

- X shape: `(93, 12)`
- y shape: `(93,)`
- `traj["s"]` shape: `(93, 64)`
- y is task-level success: `1` if `reward_info.reward == 1.0` else `0`

L2T smoke command:

```bash
python experiment/run_baseline.py \
  --input-data /Users/xuyida/Research/llm-toolcalling-benchmarks/tau2-bench/data/processed/tau2_l2t_success_retail_airline_50_filtered_20260714.pkl \
  --models proposed_only \
  --time-window 0.1 3.0 \
  --epochs 5 \
  --batch-train 16 \
  --batch-val 16 \
  --skip-sweeps \
  --num-workers 0 \
  --output-dir results/tau2_l2t_success_retail_airline_50_filtered_20260714
```

Result:

- L2T successfully loaded the tau2-derived dataset.
- `d_x` was overridden from 2 to 12.
- Shapes were X: `(93, 12)`, trajectory: `(93, 64)`.
- Proposed model trained successfully for 5 epochs.
- Epoch 1 metrics shown:
  - train-loss = 36.175673
  - train-recon = 2.269067
  - train-div = 0.724943
  - train-acc = 0.6250
  - val-recon = 2.332024
  - val-div = 0.654115
  - val-acc = 0.6316
- Outputs saved to:
  `/Users/xuyida/Research/physics_informed_testing/share_code/results/tau2_l2t_success_retail_airline_50_filtered_20260714`

Interpretation:

This is not a final performance result. It is a smoke test proving the full
bridge works:

```text
tau2 results.json -> filtered L2T-compatible pickle -> Minxing run_baseline.py training
```

Current filtered pilot record:

- Filtered dataset:
  `data/processed/tau2_l2t_success_retail_airline_50_filtered_20260714.pkl`
- N = 93
- Retail retained 46 / 50
- Airline retained 47 / 50
- X shape = `(93, 12)`
- `traj["s"]` shape = `(93, 64)`
- `y = 1` if `reward_info.reward == 1.0` else `0`
- The full pipeline works:
  `tau2 -> filtered pkl -> L2T run_baseline`

## Clean Success-Rate Analysis for Filtered N=93 Dataset

Dataset:

```text
data/processed/tau2_l2t_success_retail_airline_50_filtered_20260714.pkl
```

Total N:

- 93

Feature names:

- `domain_retail`
- `domain_airline`
- `expected_action_count`
- `expected_read_action_count`
- `expected_write_action_count`
- `requires_db_mutation`
- `has_communication_checks`
- `has_nl_assertions`
- `has_env_assertions`
- `reward_basis_has_DB`
- `reward_basis_has_COMMUNICATE`
- `reward_basis_has_NL_ASSERTION`

Metadata fields:

- `db_match`
- `domain`
- `expected_action_count`
- `expected_read_action_count`
- `expected_write_action_count`
- `num_messages`
- `num_tool_calls`
- `reward`
- `source_result_folder`
- `task_id`
- `termination_reason`

Domain success rates:

- retail: 19/46 = 41.30%
- airline: 14/47 = 29.79%
- retail -> airline drop: 11.52 percentage points

Write-required shift:

- no expected write actions: 12/26 = 46.15%
- expected write actions / DB mutation: 21/67 = 31.34%
- no-write -> write-required drop: 14.81 percentage points

DB-mutation/write-required:

- DB-mutation/write-required: 21/67 = 31.34%
- non-write: 12/26 = 46.15%

Interpretation:

- The filtered pilot shows that retail -> airline has a performance drop, but
  it is only about 11.5 percentage points.
- This is evidence of a domain shift, but not yet the strong harmful-shift
  pattern from Minxing's example, where source success is around 90% and
  target success is reduced to around 50%.
- The write-required / DB-mutation shift appears stronger in this pilot, with
  success dropping from 46.15% to 31.34%.
- Therefore, write-required tool-use tasks may be a more promising
  harmful-shift direction than domain shift alone.
- The current `gpt-4o-mini` pilot shows moderate drops, not the strong
  90% -> 50% harmful-shift pattern.
- To find a stronger harmful-shift candidate, next add another domain and/or
  define capability-based shifts such as no-write -> write-required,
  short -> long trajectory, and few-tool-calls -> many-tool-calls.

Next steps:

- Compute clean success rates by domain after filtering.
- Inspect whether retail -> airline is a harmful shift under `P(y = 1)`.
- Define more shift groups such as read-only -> write-heavy and
  no-DB-mutation -> DB-mutation.
- Optionally run a longer L2T training/sweep after confirming the dataset
  design with Minxing.

## Candidate Harmful-Shift Analysis on Filtered N=93 Dataset

Script:

```text
scripts/analyze_shift_groups.py
```

Dataset:

```text
data/processed/tau2_l2t_success_retail_airline_50_filtered_20260714.pkl
```

Basic summary:

- total N: 93
- X shape: `(93, 12)`
- `traj["s"]` shape: `(93, 64)`
- total positive y count: 33
- overall success rate: 33/93 = 35.5%

Domain shift:

- retail: 19/46 = 41.3%
- airline: 14/47 = 29.8%
- retail -> airline drop: 11.5 percentage points

Write-required shift:

- no expected write actions: 12/26 = 46.2%
- expected write actions > 0: 21/67 = 31.3%
- no-write -> write-required drop: 14.8 percentage points

Expected write count:

- zero write: 12/26 = 46.2%
- one write: 19/43 = 44.2%
- two or more writes: 2/24 = 8.3%

Trajectory/message length shift:

- median `num_messages`: 23
- short messages: 21/48 = 43.8%
- long messages: 12/45 = 26.7%
- short -> long drop: 17.1 percentage points

Tool-call count shift:

- median `num_tool_calls`: 6
- few tool calls: 21/51 = 41.2%
- many tool calls: 12/42 = 28.6%
- few -> many drop: 12.6 percentage points

Candidate harmful-shift ranking:

1. short messages -> long messages: 17.1 pp drop
2. no-write -> write-required: 14.8 pp drop
3. few tool calls -> many tool calls: 12.6 pp drop
4. retail -> airline: 11.5 pp drop
5. low expected actions -> high expected actions: -1.0 pp drop
6. low read count -> high read count: -7.2 pp drop

Interpretation:

- The strongest harmful-shift candidate in this pilot is interaction length /
  trajectory complexity: short-message tasks have 43.8% success, while
  long-message tasks have 26.7% success.
- Write-required tasks are also much harder, especially tasks with two or more
  expected writes, where success drops to 8.3%.
- Domain shift retail -> airline exists but is weaker than
  complexity/action-type shifts.
- Therefore, the next version should treat domain shift and
  capability/complexity shift separately.
- For this pilot, the most promising harmful-shift definitions are:
  1. short trajectory -> long trajectory
  2. no-write -> write-required
  3. few tool calls -> many tool calls
  4. zero/one write -> two-or-more writes
- This still does not reproduce Minxing's example of source around 90% and
  target around 50%, because `gpt-4o-mini` has low baseline success on these
  tasks.
- However, it demonstrates the pipeline and identifies plausible harmful-shift
  directions.

## Open Questions

Questions for Minxing:

1. What threshold should define a harmful drop in success rate?
2. Is a target success rate around 50% a reasonable harmful-shift criterion,
   or should harmfulness be defined only by a source-to-target drop threshold?
3. Should source/target distributions be defined by domain, task type, or
   tool/action difficulty?
4. Should the same model be fixed while only the context/task distribution
   changes?
5. Should existing tau2 result files be converted to a separate input artifact,
   or can Minxing's code read a tau2-specific adapter directly?

Data representation questions:

1. What should `X` contain for tau2-bench: task metadata, domain, tool
   requirements, initial DB/context features, observed action counts, or a
   combination?
2. Should `y` use strict reward equality, DB match, Pass^1, or another
   task-success criterion?
3. What should `traj["s"]` contain: per-turn reward/progress state, tool-call
   sequence features, message/action embeddings, DB-state deltas, or fixed
   tabular time-series features?

## Next Steps

1. Completed: tau2-to-L2T smoke test using a toy `N = 20` tau2-derived
   dataset.
2. Completed: retail 50 pilot with 46 completed/normal-stop simulations
   retained after filtering 4 abnormal simulations.
3. Completed: airline 50 pilot with 47 completed/normal-stop simulations
   retained after filtering 3 abnormal simulations.
4. Completed: update the converter to optionally filter to normal-stop
   simulations with usable `reward_info`, `db_check`, and `action_checks`.
5. Completed: build the combined filtered dataset with 93 simulations.
6. Improve the tau2-to-L2T mapping:
   - `X` = task/context features
   - `y` = binary task success
   - `traj["s"]` = trajectory/time-series representation from tau2
     interactions
7. Construct source and target groups, for example by domain, task type, or
   tool/action difficulty.
8. Compare `P(y = 1)` across source and target groups:

   ```text
   source_success_rate = P(y = 1 | source)
   target_success_rate = P(y = 1 | target)
   drop = source_success_rate - target_success_rate
   ```

9. Increase `N` beyond the 20-row toy dataset so any L2T result is more
   meaningful.
10. Confirm with Minxing the threshold for calling a source-to-target success
   rate drop harmful.
11. Treat domain shift and write-required shift as two candidate shift
    definitions.
12. Ask Minxing whether the harmful-shift setup should focus on domain-level
    shifts, capability/action-type shifts, or both.
13. Consider running more domains or more tasks to find source groups with
    higher baseline `P(y = 1)` and target groups with stronger drops.
