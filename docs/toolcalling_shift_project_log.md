# LLM Tool-Calling Context Shift Project Log

This file is the persistent project log for the LLM tool-calling context shift project. Future Codex sessions should read this file before editing project artifacts and append chronological entries rather than overwriting or deleting previous entries.

Log rules:

- Distinguish verified facts from assumptions and proposed next steps.
- Do not claim that smoke tests are performance benchmarks.
- Do not describe synthetic API-Bank negatives as naturally occurring LLM failures.
- Preserve the distinction between task-level and API-call-level labels.
- Keep entries append-only after this initial project summary.

## 1. Research Motivation

An LLM tool-calling agent may work well when deployed at Company A, while Company B has a different business context, tools, policies, workflows, or user distribution. The naive response to any context shift is to collect new data and fine-tune or retrain the model, but fine-tuning and retraining are costly.

The research goal is to determine whether a context shift is actually harmful before retraining. Some context shifts may substantially change X or S while causing little or no change in P(Y=1). Such harmless or business-orthogonal shifts may not require retraining. The project should study harmful, harmless, and possibly beneficial shifts.

Use:

```text
Delta_Y = P_target(Y=1) - P_source(Y=1)
```

Conceptually:

- harmful shift: Delta_Y is meaningfully negative
- harmless shift: Delta_Y is approximately zero
- beneficial shift: Delta_Y is positive

No numerical threshold is frozen yet.

## 2. Current X, S, Y Formulation

X is the task and deployment context available before or during execution. It may include the user request, domain, available tools, tool schemas, policies, expected action complexity, and environment context.

S is the tool-calling interaction trajectory. It includes user messages, assistant messages, tool/API calls, arguments, outputs, errors, retries, and termination.

Y is correctness or final task success. Its exact meaning depends on `label_scope` and `source_dataset`.

Important distinction:

- tau2 uses task-level success labels.
- API-Bank pilot uses API-call-level correctness labels.

## 3. tau2-Bench Pilot

Verified facts:

- Data source: tau2-bench retail and airline.
- Source/target interpretation: retail is the source domain and airline is the target domain.
- Local task counts observed: retail 114, airline 50, telecom 2285, banking knowledge 97.
- Pilot executions: retail 50 run with 46 retained after filtering; airline 50 run with 47 retained after filtering; total retained 93.
- The 46/50 and 47/50 counts are retained simulation counts, not train/test splits.

Current tau2 X:

- 12 structured features.
- Includes domain indicators, expected action counts, read/write counts, DB mutation requirement, and evaluator-component indicators.

Current tau2 S:

- Fixed-length event sequence of length 64.
- Codes: 0 padding, 1 user message, 2 assistant text, 3 read tool call, 4 write tool call, 5 successful tool response, 6 tool error, 7 end.
- This is a coarse structural representation and does not preserve full semantics, tool names, arguments, or output content.

Current tau2 Y:

- `y = 1` only when the existing converted benchmark reward label is 1.
- Otherwise `y = 0`.
- This is a task-level label.
- It does not require exact trajectory match against a reference trajectory.

Filtering:

- Normal termination.
- Valid reward information.
- Required evaluator checks present.
- Filtering logic should remain auditable.

## 4. tau2 Sequence-Model Compatibility Test

Verified facts:

- Input: 93 samples.
- X shape: `(93, 12)`.
- S shape: `(93, 64)`.
- Configuration: model `proposed`, epochs 30, `batch_train` 16, `batch_val` 16, train/validation split 74/19, `d_x` 12, `d_o` 1, `tau` 1.0, `time_window` `[0.1, 3.0]`, `skip_sweeps` true.
- The train loader used `drop_last=True`, so train metrics were computed over 64 samples per epoch.

Observed metrics:

- `train_safety_acc` started at 0.625 and ended at 0.640625.
- Validation safety accuracy remained 0.631578947368421.
- `train_safety_y1_acc` remained 0.0.
- `train_safety_y0_acc` remained 1.0.
- `validation_safety_y1_acc` remained 0.0.
- `validation_safety_y0_acc` remained 1.0.

Interpretation:

- The model predicted every sample as `y=0`.
- Validation accuracy equals the validation majority-class rate, 12/19.
- This run verified pipeline compatibility only.
- It did not demonstrate useful predictive learning and must not be presented as a performance benchmark.

## 5. Telecom Feasibility Test

Verified facts:

- Telecom is technically runnable.
- One tested task took approximately 362 seconds.
- It encountered a TPM/rate-limit issue.
- It reached max steps.
- Reward was 0.
- Approximate agent cost was $0.095.

Conclusion: telecom is technically possible, but currently too slow, costly, and unstable for immediate scaling.

## 6. API-Bank Investigation

Verified facts:

- API-Bank contains successful reference tool-use dialogues.
- The README reports 314 evaluation dialogues, 753 evaluation API calls, 1,888 training dialogues, 2,138 APIs, and 1,000 domains.
- Local level-1/level-2 inspection found 264 dialogues, 508 API reference steps, 508 API events, and zero exceptions in the reference trajectories.
- Therefore, the released local dialogues are primarily successful references.
- No released model-prediction files were found in the repository.
- API-Bank provides X and S directly, but not naturally occurring positive and negative Y labels in the reference files.

API-Bank evaluator:

- API-call predictions can be evaluated as correct or incorrect.
- The evaluator checks API name, parameters, execution result, and reference result.
- Text responses use ROUGE-L.
- The current project uses only binary API-call correctness for the pilot.

## 7. API-Bank Synthetic Correctness Pilot

Verified facts:

- Records: 508 positive reference API calls and 508 synthetic negative API calls, for 1,016 total records.
- Balanced labels: 508 positive and 508 negative.

Negative corruption distribution:

- `missing_required_argument`: 137
- `wrong_api_name`: 140
- `wrong_argument_type`: 114
- `wrong_argument_value`: 117

Validation:

- 508/508 negatives validated as incorrect through evaluator/API correctness logic.
- Only `GetToday` supported API-name corruption only.
- Fallback count: 25.
- Sensitive values were redacted.
- Total redacted sensitive values from the first reported build: 2,872.

Important limitation:

- Synthetic negatives are not natural LLM failures.
- This pilot is suitable for schema, representation, evaluator, and pipeline development.
- It should not be used to estimate real-world harmful shifts in P(Y=1).

## 8. Unified Dataset Schema

Verified facts:

tau2:

- Records: 93.
- Labels: 0 = 60, 1 = 33.
- `label_scope`: `task_level`.
- `label_origin`: `tau2_benchmark_reward`.
- `is_synthetic`: false.
- X dimension: 12.
- S dimension: 64.

API-Bank:

- Records: 1,016.
- Labels: 0 = 508, 1 = 508.
- `label_scope`: `api_call_level`.
- Positive label origin: `reference_api_call`.
- Negative label origin: `synthetic_corruption`.
- Synthetic negatives: 508.
- Non-synthetic positive references: 508.
- X dimension: 5.
- S dimension: 4.

Combined bookkeeping totals:

- Total records across separate files: 1,109.
- Synthetic true: 508.
- Synthetic false: 601.

Important compatibility warnings:

- Task-level versus API-call-level labels.
- Benchmark outcomes versus synthetic corruptions.
- Different X dimensions.
- Different S representations.
- The datasets must not yet be treated as IID samples from one task.
- The two datasets remain in separate JSONL files.

Existing files:

- `scripts/build_unified_toolcalling_dataset.py`
- `tests/test_build_unified_toolcalling_dataset.py`
- `data/processed/unified_toolcalling_tau2.jsonl`
- `data/processed/unified_toolcalling_apibank.jsonl`
- `data/processed/unified_toolcalling_manifest.json`

Tests:

- 8 pytest tests passed.
- ruff passed.

## 9. Open Research Questions

- How should text and structured trajectory content be converted into numerical representations?
- Which information belongs in X versus S?
- Should evaluator-side features be excluded from X if unavailable at deployment time?
- How should task-level and API-call-level labels be related?
- How can real, non-synthetic failures be collected economically?
- How should harmful versus harmless shifts be defined statistically?
- What practical threshold in Delta_Y should trigger retraining?
- How should source and target groups be constructed without label leakage?
- Can business-domain changes be orthogonal to the underlying tool-use mechanism?

## 10. Immediate Next Step

The next task is to create a 200-sample numerical-representation pilot:

- tau2: all 93 records.
- API-Bank: deterministic sample of 107 records.
- Serialize raw X and S into non-leaking text.
- Embed X and S into fixed-dimensional vectors.
- Concatenate embeddings with shared structural features.
- Do not train a predictive model yet.

## 2026-07-17 — Numerical Representation Pilot

### Objective

Create a 200-sample numerical-representation pilot that converts unified tau2 and API-Bank X/S records into shared numerical arrays without training a predictive model and without introducing label leakage into model-facing text or structural features.

### Files added or changed

- Added `scripts/build_toolcalling_numerical_representation.py`.
- Added `tests/test_build_toolcalling_numerical_representation.py`.
- Added generated outputs:
  - `data/processed/toolcalling_numerical_pilot.npz`
  - `data/processed/toolcalling_numerical_pilot.jsonl`
  - `data/processed/toolcalling_numerical_pilot_manifest.json`
- Added this project log at `docs/toolcalling_shift_project_log.md`.
- Updated `pyproject.toml` to add `sentence-transformers>=3.0.0` under the existing `experiments` optional dependency group.
- Updated `uv.lock` through `uv add --optional experiments 'sentence-transformers>=3.0.0'`.

## 2026-07-17 — Tau2 Shift Uncertainty Analysis

### Objective

Add exploratory statistical uncertainty estimates for the eligible tau2 candidate shifts only, preserving the existing source/target definitions and excluding API-Bank synthetic correctness labels from harmful/harmless interpretation.

### Files added or changed

- Added `scripts/analyze_tau2_shift_uncertainty.py`.
- Added `tests/test_analyze_tau2_shift_uncertainty.py`.
- Added generated outputs:
  - `data/processed/tau2_shift_uncertainty.jsonl`
  - `data/processed/tau2_shift_uncertainty_summary.json`
  - `docs/tau2_shift_uncertainty.md`
- Applied ruff's mechanical import-spacing fix to four older shift scripts so the requested full `ruff check` command passes.

### Verified facts

- Eligible tau2 shifts analyzed: 6.
- API-Bank rows analyzed: 0.
- Bootstrap configuration: 10,000 replicates, seed 1, resampling within source and target groups separately.
- Primary confidence interval: Newcombe-Wilson difference in proportions.
- Multiple testing method: Benjamini-Hochberg adjustment across the six eligible tau2 shifts.
- Classification counts under `delta_practical` 0.05, 0.10, and 0.15 were identical: 1 `candidate_harmful` and 5 `inconclusive`.
- `tau2_zero_or_one_write_to_two_plus_writes` was the only `candidate_harmful` shift under all three thresholds.

### Validation

- `uv run --extra dev pytest tests/test_analyze_tau2_shift_uncertainty.py tests/test_build_toolcalling_shift_inventory.py -q` passed with 21 tests.
- `uv run --extra dev ruff check` passed.

### Interpretation

This analysis is exploratory and small-sample. It does not train a predictive model, does not use `y` to redefine group membership, does not imply causal effects, and does not independently authorize deployment decisions.

### Commands run

- `uv run --extra dev pytest tests/test_build_toolcalling_numerical_representation.py -q`
- `uv run --extra dev ruff check scripts/build_toolcalling_numerical_representation.py tests/test_build_toolcalling_numerical_representation.py`
- `uv add --optional experiments 'sentence-transformers>=3.0.0'`
- `uv run --extra dev pytest tests/test_build_toolcalling_numerical_representation.py -q`
- `uv run --extra dev ruff check scripts/build_toolcalling_numerical_representation.py tests/test_build_toolcalling_numerical_representation.py`
- `uv run --extra experiments python scripts/build_toolcalling_numerical_representation.py --tau2-limit 93 --apibank-limit 107 --seed 1`
- `uv run --extra experiments python - <<'PY' ...` to inspect generated NPZ shapes.

### Data counts

Verified output counts:

- Total records: 200.
- tau2 records: 93.
- API-Bank records: 107.
- tau2 labels: 0 = 60, 1 = 33.
- API-Bank labels in selected sample: 0 = 53, 1 = 54.
- Synthetic records: 53.
- Non-synthetic records: 147.
- API-Bank complete positive/negative pairs selected: 53.
- API-Bank all selected pairs complete: false, because the requested API-Bank sample size of 107 is odd.

### Representation decisions

- X text and S text are deterministic.
- tau2 X text uses domain, task id, expected action counts, structured task requirements, and an explicit note that exact tool schemas are unavailable in the unified pilot record.
- tau2 S text serializes the 0-7 event sequence into symbolic event names and records that the representation is coarse.
- API-Bank X text uses only pre-call dialogue history and available API names.
- API-Bank S text uses only the candidate API name and candidate arguments.
- API-Bank execution results are excluded from model-facing S text.
- API-Bank `has_exception` and `termination_success_signal` are marked unavailable in model-facing S structural features because they could trivially encode synthetic-negative validation status in future builds.
- tau2 reward and `db_match` metadata are excluded from model-facing text and structural features.
- Positive and negative API-Bank pair members have identical X text and X structural features when both members are selected.
- The embedding model is exactly `sentence-transformers/all-MiniLM-L6-v2`.
- The installed `sentence-transformers` package version reported in the manifest is 5.6.0.

### Output shapes

Verified NPZ contents:

- `X`: `(200, 402)`, `float32`.
- `S`: `(200, 404)`, `float32`.
- `y`: `(200,)`, `int64`.
- `sample_ids`: `(200,)`.
- `source_dataset`: `(200,)`.
- `label_scope`: `(200,)`.
- `is_synthetic`: `(200,)`, `bool`.

Dimension breakdown:

- X embedding dimension: 384.
- S embedding dimension: 384.
- X structural dimension: 18.
- S structural dimension: 20.
- Final X dimension: 402.
- Final S dimension: 404.

### Leakage audit

Verified manifest results:

- Duplicate sample IDs: none.
- NaN counts: `X = 0`, `S = 0`.
- Infinite-value counts: `X = 0`, `S = 0`.
- Model-facing leakage field-name hits: none.
- Embedding norm summaries are approximately normalized to 1.0 for both X and S embeddings.

Excluded metadata-only fields from model-facing text and structural features:

- `corruption_type`
- `is_synthetic`
- `label_origin`
- `validation_error`
- `validation_status`
- `variant`
- `y`

### Test results

Verified:

- `uv run --extra dev pytest tests/test_build_toolcalling_numerical_representation.py -q`: 11 passed, 2 warnings.
- `uv run --extra dev ruff check scripts/build_toolcalling_numerical_representation.py tests/test_build_toolcalling_numerical_representation.py`: all checks passed.

The pytest warnings were unrelated to this builder: a Python 3.13 `audioop` deprecation warning from voice utilities and an unknown pytest config option warning for `asyncio_default_fixture_loop_scope`.

### Verified findings

- The requested 200-record numerical pilot was generated successfully.
- The JSONL output stores serialized text, structural features, labels, and metadata, but does not store embeddings directly.
- The NPZ output stores the numerical arrays and required metadata arrays.
- The generated manifest records feature order, missing-feature counts, text-length summaries, duplicate IDs, NaN/inf counts, embedding norms, coarse tau2 serialization count, API-Bank pair completeness, leakage-audit decisions, compatibility warnings, and embedding model/package information.
- No classifier, sequence model, predictive accuracy, harmful/harmless shift labels, or source dataset mutations were introduced.

## 2026-07-17 — Full Numerical Dataset and Shift Inventory

### Objective

Scale the numerical representation from the 200-record pilot to all unified tau2 and API-Bank records, then build a descriptive inventory of candidate source/target context shifts without training a predictive model and without assigning final harmful/harmless labels.

### Files added or changed

- Updated `scripts/build_toolcalling_numerical_representation.py` with full-data mode and full-output file names.
- Added `scripts/build_toolcalling_shift_inventory.py`.
- Updated `tests/test_build_toolcalling_numerical_representation.py`.
- Added `tests/test_build_toolcalling_shift_inventory.py`.
- Added generated outputs:
  - `data/processed/toolcalling_numerical_full.npz`
  - `data/processed/toolcalling_numerical_full.jsonl`
  - `data/processed/toolcalling_numerical_full_manifest.json`
  - `data/processed/toolcalling_shift_inventory.jsonl`
  - `data/processed/toolcalling_shift_inventory_summary.json`
  - `docs/toolcalling_shift_inventory.md`
- Appended this entry to `docs/toolcalling_shift_project_log.md`.

### Commands run

- `uv run --extra dev pytest tests/test_build_toolcalling_numerical_representation.py tests/test_build_toolcalling_shift_inventory.py -q`
- `uv run --extra dev ruff check scripts/build_toolcalling_numerical_representation.py scripts/build_toolcalling_shift_inventory.py tests/test_build_toolcalling_numerical_representation.py tests/test_build_toolcalling_shift_inventory.py`
- `uv run --extra experiments python scripts/build_toolcalling_numerical_representation.py --full-data`
- `uv run --extra dev python scripts/build_toolcalling_shift_inventory.py`
- `uv run --extra dev python - <<'PY' ...` to inspect generated NPZ shapes, manifest counts, leakage audit, and inventory rows.

### Full data counts

Verified full numerical output counts:

- Total records: 1,109.
- tau2 records: 93.
- API-Bank records: 1,016.
- tau2 labels: 0 = 60, 1 = 33.
- API-Bank labels: 0 = 508, 1 = 508.
- Label scopes: `task_level` = 93, `api_call_level` = 1,016.
- Synthetic records: 508.
- Non-synthetic records: 601.

### Numerical output shapes

Verified NPZ contents:

- `X`: `(1109, 402)`.
- `S`: `(1109, 404)`.
- `y`: `(1109,)`.
- X embedding dimension: 384.
- S embedding dimension: 384.
- X structural dimension: 18.
- S structural dimension: 20.

### Candidate shift definitions

tau2 candidate groupings:

- `tau2_retail_to_airline`: retail domain to airline domain.
- `tau2_no_write_to_write_required`: expected write count 0 to expected write count at least 1.
- `tau2_zero_or_one_write_to_two_plus_writes`: expected write count <= 1 to expected write count >= 2.
- `tau2_few_to_many_expected_actions`: lower quartile to upper quartile of expected action count; thresholds 1.0 and 6.0.
- `tau2_short_to_long_trajectory`: lower quartile to upper quartile of trajectory length; thresholds 19.0 and 33.0.
- `tau2_few_to_many_tool_calls`: lower quartile to upper quartile of observed tool-call count; thresholds 3.0 and 9.0.

API-Bank candidate groupings:

- `api_bank_no_auth_to_auth_required`: no authentication signal to authentication-required context/API.
- `api_bank_short_to_long_dialogue_history`: lower quartile to upper quartile of dialogue-history length; thresholds 3.0 and 8.0.
- `api_bank_one_tool_to_multiple_tools_available`: one available API to multiple available APIs.
- `api_bank_few_to_many_arguments`: lower quartile to upper quartile of candidate argument count; thresholds 1.0 and 3.0.
- `api_bank_simple_call_to_multi_step_context`: previous API-call count 0 to previous API-call count at least 1.
- `api_bank_domain_or_tool_family_comparison`: recorded as failed/unsupported because unified API-Bank domain metadata is unavailable.

### Descriptive results

Inventory status: 12 candidate rows, 11 eligible, 1 failed.

tau2 descriptive results:

- `tau2_retail_to_airline`: n = 46 -> 47, delta_y = -0.1152, X distance = 4.2353, S distance = 4.6796.
- `tau2_no_write_to_write_required`: n = 26 -> 67, delta_y = -0.1481, X distance = 4.1591, S distance = 10.4401.
- `tau2_zero_or_one_write_to_two_plus_writes`: n = 69 -> 24, delta_y = -0.3659, X distance = 4.3299, S distance = 14.0699.
- `tau2_few_to_many_expected_actions`: n = 25 -> 36, delta_y = -0.0789, X distance = 8.0470, S distance = 12.6480.
- `tau2_short_to_long_trajectory`: n = 27 -> 28, delta_y = -0.2685, X distance = 4.6089, S distance = 28.4241.
- `tau2_few_to_many_tool_calls`: n = 25 -> 26, delta_y = -0.2092, X distance = 4.9653, S distance = 28.5383.

API-Bank descriptive results:

- `api_bank_no_auth_to_auth_required`: n = 472 -> 544, delta_y = 0.0000, X distance = 151.2239, S distance = 0.9240.
- `api_bank_short_to_long_dialogue_history`: n = 276 -> 350, delta_y = 0.0000, X distance = 697.2969, S distance = 1.4473.
- `api_bank_one_tool_to_multiple_tools_available`: n = 212 -> 804, delta_y = 0.0000, X distance = 265.9932, S distance = 0.4071.
- `api_bank_few_to_many_arguments`: n = 374 -> 306, delta_y = 0.1150, X distance = 294.6998, S distance = 2.8494.
- `api_bank_simple_call_to_multi_step_context`: n = 522 -> 494, delta_y = 0.0000, X distance = 451.7896, S distance = 0.7280.
- `api_bank_domain_or_tool_family_comparison`: failed with n = 0 -> 0.

### Validity distinctions

- tau2 outcome type is `real_benchmark_task_outcome`.
- API-Bank outcome type is `synthetic_api_call_correctness`.
- API-Bank delta_y values cannot be interpreted as real deployment success-rate shifts because the negative samples are synthetic corruptions and labels are balanced by construction.
- Task-level and API-call-level labels remain separate.
- The report uses descriptive terms only: candidate negative-outcome shift, candidate stable-outcome shift, and candidate positive-outcome shift.
- No final harmful, harmless, or beneficial classification was assigned.

### Leakage audit

Verified:

- Full numerical duplicate sample IDs: none.
- Full numerical NaN counts: `X = 0`, `S = 0`.
- Full numerical infinite-value counts: `X = 0`, `S = 0`.
- Model-facing leakage field-name hits: none.
- Inventory group-definition fields do not use `y`.
- Inventory group-definition fields do not use `variant`, `corruption_type`, `label_origin`, `is_synthetic`, `validation_status`, or `validation_error`.

### Test results

Verified:

- `uv run --extra dev pytest tests/test_build_toolcalling_numerical_representation.py tests/test_build_toolcalling_shift_inventory.py -q`: 20 passed, 2 warnings.
- `uv run --extra dev ruff check scripts/build_toolcalling_numerical_representation.py scripts/build_toolcalling_shift_inventory.py tests/test_build_toolcalling_numerical_representation.py tests/test_build_toolcalling_shift_inventory.py`: all checks passed.

The pytest warnings were unrelated to these builders: a Python 3.13 `audioop` deprecation warning from voice utilities and an unknown pytest config option warning for `asyncio_default_fixture_loop_scope`.

### Verified findings

- The full numerical dataset was generated without overwriting the 200-record pilot outputs.
- Full API-Bank pair completeness is 508 complete pairs and all selected pairs complete.
- tau2 coarse trajectory serialization count is 93.
- Candidate shifts are deterministic and record exact grouping rules, thresholds, source/target group sizes, label counts, y means, raw delta_y, and X/S centroid distances.
- API-Bank domain/tool-family comparison is currently unsuitable because domain metadata is not reliable in the unified API-Bank records.

### Limitations

- API-Bank negative labels are synthetic corruptions and are balanced by construction.
- tau2 and API-Bank labels are not equivalent and should not be pooled for final shift conclusions.
- tau2 sample sizes remain small for final harmful/harmless classification.
- No confidence intervals or practical significance thresholds have been applied.
- No predictive model was trained.

### Next step

Define a practical delta_y threshold and confidence-interval procedure, then apply it first to eligible tau2 candidate shifts before considering any synthetic API-Bank analyses as representation or evaluator stress tests only.

## 2026-07-17 — Tau2 Additional Sampling Plan

### Objective

Create a targeted additional-data collection plan for tau2 without running new LLM simulations, changing shift definitions, training a model, or using API-Bank synthetic outcomes.

### Files added or changed

- Added `scripts/plan_tau2_additional_sampling.py`.
- Added `tests/test_plan_tau2_additional_sampling.py`.
- Added generated outputs:
  - `data/processed/tau2_additional_sampling_plan.json`
  - `docs/tau2_additional_sampling_plan.md`
- Appended this project-log entry.

### Verified findings

- The plan includes all 6 eligible tau2 shifts and excludes API-Bank.
- Current evidence remains unchanged: `tau2_zero_or_one_write_to_two_plus_writes` is `candidate_harmful` at practical thresholds 0.05, 0.10, and 0.15; the other 5 eligible tau2 shifts are inconclusive.
- The current tau2 dataset still contains 93 retained real task-level outcomes.
- Local task-pool audit found 114 retail tasks, 50 airline tasks, 2,285 telecom tasks, and 97 banking-knowledge tasks.
- Retail has 46 retained outcomes and 68 tasks without retained outcomes; 64 retail tasks were not previously attempted in the 2026-07-14 retail run.
- Airline has 47 retained outcomes, but all 50 local airline tasks appear in the 2026-07-14 airline simulation result file.
- The proposed Stage 1 batch contains 12 unused retail tasks: 8 with two or more expected write actions, 2 with no expected write actions, and 2 low-action one-write tasks.
- Task selection uses local task definitions and prior task IDs only; it does not use observed `y`.
- Telecom remains technically runnable but currently unsuitable for large runs without a separate cost-control plan because the feasibility test was slow, rate-limited, reached max steps, and cost approximately $0.095 for one unsuccessful task.

### Planning methods

- Precision estimates use a normal approximation for difference-in-proportions 95% CI half-width targets 0.15, 0.10, and 0.05.
- Power estimates use a standard two-sided two-sample proportions normal approximation with alpha 0.05 and power 0.80 for effect sizes 0.15, 0.10, and 0.05.
- Equivalence estimates are reported as CI-screening approximations where current `delta_y` is inside the practical margin; otherwise they are explicitly unavailable rather than invented.
- All estimates are planning approximations, not guarantees.

### Test results

- `uv run pytest tests/test_plan_tau2_additional_sampling.py`: 10 passed, 2 warnings.
- `uv run ruff check scripts/plan_tau2_additional_sampling.py tests/test_plan_tau2_additional_sampling.py`: all checks passed.
- `uv run ruff check`: all checks passed.
- `uv run pytest`: collection failed because optional repo dependencies are not installed in the current environment (`a2a`, `agentify_tau_bench`, `websockets`, `rank_bm25`, `gymnasium`, and `pyaudio`). The failure occurred before running the suite and is unrelated to the added planner tests.

### Limitations

- The current tau2 outcome set is still small.
- Candidate shift definitions reuse records and are not independent.
- Observed trajectory-length and observed tool-call-count group membership cannot be assigned to unused tasks before running them.
- The proposed Stage 1 batch is intentionally conservative and should be followed by a rerun of the uncertainty analysis before any larger collection.

## 2026-07-17 — Tau2 Stage 1 Collection Preparation

### Objective

Prepare a reproducible tau2 retail Stage 1 collection workflow from the existing additional-sampling plan, generate and validate a dry-run manifest, add a dry-run-first runner, and add post-run ingestion/update tooling without executing real LLM simulations.

### Selected task IDs

Selected retail task IDs: `54`, `55`, `64`, `71`, `72`, `74`, `76`, `81`, `57`, `62`, `50`, `70`.

### Selection composition

- `two_plus_writes`: 8 tasks (`54`, `55`, `64`, `71`, `72`, `74`, `76`, `81`)
- `no_write`: 2 tasks (`57`, `62`)
- `low_action_one_write`: 2 tasks (`50`, `70`)

### Dry-run command

Manifest builder:

```text
uv run python scripts/build_tau2_stage1_manifest.py
```

Runner dry-run:

```text
uv run python scripts/run_tau2_stage1.py
```

Dry-run result: `data/processed/tau2_stage1_run_status.json` records 12 dry-run tasks, 0 completed executions, 0 observed cost, and stop reason `finished`.

### Runtime and cost controls

- Real execution requires explicit `--execute`; the runner defaults to dry-run mode.
- The runner executes one retail task at a time with `--num-trials 1`, `--max-concurrency 1`, seed `20260717`, verbose logs, and `--auto-resume`.
- Native tau2 outputs remain under `data/simulations/tau2_stage1_raw/task_<task_id>/results.json`.
- One preserved raw result copy per completed task is written under `data/processed/tau2_stage1_raw/task_<task_id>.json`.
- Status is written incrementally to `data/processed/tau2_stage1_run_status.json`.
- Optional stop controls include `--max-total-cost` and default stop-on-first-error behavior unless `--continue-on-error` is supplied.

### Files added or changed

- Added `scripts/build_tau2_stage1_manifest.py`.
- Added `scripts/run_tau2_stage1.py`.
- Added `scripts/ingest_tau2_stage1_results.py`.
- Added `tests/test_build_tau2_stage1_manifest.py`.
- Added `tests/test_run_tau2_stage1.py`.
- Added `tests/test_ingest_tau2_stage1_results.py`.
- Generated `data/processed/tau2_stage1_manifest.json`.
- Generated `docs/tau2_stage1_manifest.md`.
- Generated dry-run status `data/processed/tau2_stage1_run_status.json`.
- Appended this project-log entry.

### Tests

- `uv run --extra dev pytest tests/test_build_tau2_stage1_manifest.py tests/test_run_tau2_stage1.py tests/test_ingest_tau2_stage1_results.py -q`: 15 passed, 2 warnings.
- `uv run --extra dev ruff check scripts/build_tau2_stage1_manifest.py scripts/run_tau2_stage1.py scripts/ingest_tau2_stage1_results.py tests/test_build_tau2_stage1_manifest.py tests/test_run_tau2_stage1.py tests/test_ingest_tau2_stage1_results.py`: all checks passed.

The warnings were unrelated to the Stage 1 code: a Python 3.13 `audioop` deprecation warning from voice utilities and an unknown pytest config option warning for `asyncio_default_fixture_loop_scope`.

### Verified findings

- The manifest contains exactly 12 unique retail task IDs.
- The selected composition is exactly 8 `two_plus_writes`, 2 `no_write`, and 2 `low_action_one_write`.
- No selected task is marked previously attempted or previously retained.
- Selection is tied to the existing additional-sampling plan and local retail task definitions.
- Selection policy records that observed `y`, reward, and prior success/failure labels are not used.
- Telecom is excluded.
- The runner defaults to dry-run mode and mock-tested execution requires `--execute`.
- Resume behavior skips already preserved raw task files.
- Ingestion reuses the original tau2 conversion/filtering helpers and writes Stage 1 retained records separately.
- Stage 1 analysis mode writes versioned `_stage1` outputs and does not overwrite baseline uncertainty outputs.

### Limitations

- No real Stage 1 LLM simulations were executed in this task.
- Therefore `data/processed/tau2_stage1_retained.jsonl`, `data/processed/tau2_stage1_ingestion_summary.json`, and Stage 1 uncertainty outputs have not been produced from real outcomes yet.
- Estimated maximum LLM calls is a conservative planning bound, not an observed usage count.
- The proposed command uses the existing pilot model configuration (`gpt-4o-mini` for both agent and user); execution should be reviewed before approval.

### Next step

Review the selected task IDs and proposed real execution command. If approved, run:

```text
uv run python scripts/run_tau2_stage1.py --execute
```

## 2026-07-17 — Tau2 Stage 1 Canary Support

### Issue found

The Stage 1 runner did not accept a canary task selector. Running:

```text
uv run python scripts/run_tau2_stage1.py --task-id 54 --execute
```

failed at argument parsing with `unrecognized arguments: --task-id 54`.

### Implementation

- Added optional `--task-id TASK_ID` support to `scripts/run_tau2_stage1.py`.
- When omitted, the runner still processes the full Stage 1 manifest.
- When supplied, the runner validates the task ID against the manifest and processes only that task.
- Unknown task IDs are rejected before any task execution.
- Dry-run remains the default; real subprocess execution still requires `--execute`.
- Canary runs use the same per-task output paths as full runs: `data/processed/tau2_stage1_raw/task_<task_id>.json` and `data/simulations/tau2_stage1_raw/task_<task_id>/results.json`.
- Completed canary output is not moved or duplicated; a later full run skips the completed task automatically through the existing preserved raw-result check.

### Canary eligibility

Verified task `54` is present in `data/processed/tau2_stage1_manifest.json`, belongs to `two_plus_writes`, has 12 expected actions, 9 reads, 3 writes, requires DB mutation, is not marked previously attempted, and is not marked previously retained. No preserved raw output for task `54` was present before this dry-run-only canary check, so it is eligible for the canary.

### Dry-run result

Command run:

```text
uv run python scripts/run_tau2_stage1.py --task-id 54
```

Result: `data/processed/tau2_stage1_run_status.json` records exactly one selected task (`54`), `dry_run_count` 1, `completed_count` 0, status `dry_run`, and stop reason `finished`. No real LLM simulation was executed.

### Tests

- `uv run --extra dev pytest tests/test_run_tau2_stage1.py -q`: 9 passed, 2 warnings.
- `uv run --extra dev ruff check scripts/run_tau2_stage1.py tests/test_run_tau2_stage1.py`: all checks passed.

The warnings were unrelated to the Stage 1 runner changes: a Python 3.13 `audioop` deprecation warning from voice utilities and an unknown pytest config option warning for `asyncio_default_fixture_loop_scope`.

### Exact real canary command

```text
uv run python scripts/run_tau2_stage1.py --task-id 54 --execute
```

### Next step

If approved, run the exact real canary command above for task `54`, inspect the preserved raw result and status JSON, then run the full Stage 1 command. The later full run should skip task `54` automatically if the canary completed successfully.

## 2026-07-17 — Tau2 Stage 1 Post-Collection Analysis

### Objective

Integrate the 12 retained Stage 1 tau2 records with the original 93 tau2 records, then rebuild tau2-only numerical data, the shift inventory, uncertainty analysis, and pre/post comparison using versioned Stage 1 outputs.

### Stage 1 execution summary

Stage 1 execution and ingestion were complete before this analysis: 12 attempted, 12 completed, 12 retained, 0 filtered, with 4/12 successes and total observed cost 0.0706581. Task selection was targeted by X/task characteristics, not observed outcomes.

### Merge counts

- Baseline records: 93
- Stage 1 records: 12
- Merged records: 105
- Merged y distribution: {"0": 68, "1": 37}

### Numerical shapes

- X: (105, 402)
- S: (105, 404)
- y: (105,)

### Pre/post shift results

- `tau2_retail_to_airline`: source_n=58, target_n=47, delta_y=-0.0987, 95% CI=[-0.2687, 0.0844], d=0.05 inconclusive, d=0.10 inconclusive, d=0.15 inconclusive
- `tau2_no_write_to_write_required`: source_n=29, target_n=76, delta_y=-0.1325, 95% CI=[-0.3320, 0.0658], d=0.05 inconclusive, d=0.10 inconclusive, d=0.15 inconclusive
- `tau2_zero_or_one_write_to_two_plus_writes`: source_n=73, target_n=32, delta_y=-0.3271, 95% CI=[-0.4634, -0.1371], d=0.05 candidate_harmful, d=0.10 candidate_harmful, d=0.15 inconclusive
- `tau2_few_to_many_expected_actions`: source_n=28, target_n=39, delta_y=-0.1053, 95% CI=[-0.3263, 0.1252], d=0.05 inconclusive, d=0.10 inconclusive, d=0.15 inconclusive
- `tau2_short_to_long_trajectory`: source_n=28, target_n=35, delta_y=-0.2786, 95% CI=[-0.4833, -0.0371], d=0.05 inconclusive, d=0.10 inconclusive, d=0.15 inconclusive
- `tau2_few_to_many_tool_calls`: source_n=26, target_n=33, delta_y=-0.2191, 95% CI=[-0.4355, 0.0226], d=0.05 inconclusive, d=0.10 inconclusive, d=0.15 inconclusive

### Candidate-harmful shift result

`tau2_zero_or_one_write_to_two_plus_writes` remains candidate_harmful at all three practical thresholds after adding Stage 1 records: False. Post-Stage-1 classifications are d=0.05 `candidate_harmful`, d=0.10 `candidate_harmful`, and d=0.15 `inconclusive`.

### Classification changes

tau2_zero_or_one_write_to_two_plus_writes

### CI-width changes

CI widths narrowed for: tau2_retail_to_airline, tau2_no_write_to_write_required, tau2_zero_or_one_write_to_two_plus_writes, tau2_few_to_many_expected_actions, tau2_short_to_long_trajectory, tau2_few_to_many_tool_calls

### Verified findings

- Baseline tau2/API-Bank files were not overwritten.
- API-Bank data were not modified.
- Six tau2 shift definitions and thresholds were preserved from the baseline inventory.
- No predictive model was trained.

### Limitations

The analysis remains exploratory and relatively small. The shifts reuse records across non-independent definitions, and results should not be interpreted causally or as proof that a shift is harmful or harmless.

### Next step

Use these versioned Stage 1 results to decide whether another targeted collection stage is warranted before considering adaptation or retraining.

## 2026-07-17 — Stage 1 PR Preparation

### Scope

Prepared the branch for one coherent GitHub PR covering the tau2 L2T and Stage 1 context-shift workflow. This pass audited changed and untracked files, proposed the Git inclusion/exclusion set, added narrow ignore rules for local generated artifacts, validated the dependency and script diffs, verified core scientific invariants from generated manifests, and created `docs/pr_tau2_stage1_shift_analysis.md`.

No commit, push, PR creation, or deletion of local research outputs was performed.

### Included files

Proposed PR contents include workflow scripts, targeted tests, documentation, dependency files, `.gitignore`, the project log, `docs/pr_tau2_stage1_shift_analysis.md`, and compact machine-readable manifests/summaries needed to understand or reproduce the experiment.

The four pre-existing script diffs are mechanical Ruff/import-spacing blank-line removals only:

- `scripts/analyze_shift_groups.py`
- `scripts/build_shift_level_dataset.py`
- `scripts/build_shift_level_summary.py`
- `scripts/convert_tau2_results_to_l2t_pkl.py`

### Excluded local artifacts

Proposed local-only artifacts are binary arrays/pickles, raw Stage 1 result copies, runtime status logs, and regenerable large JSONL outputs:

- `data/processed/*.pkl`
- `data/processed/*.npz`
- `data/processed/tau2_stage1_raw/`
- `data/processed/tau2_stage1_run_status.json`
- `data/processed/toolcalling_numerical_*.jsonl`
- `data/processed/unified_toolcalling_*.jsonl`

Added these exact narrow `.gitignore` rules. An existing local `.env` and `.DS_Store` appear only in ignored-file status through pre-existing ignore rules and are not proposed for tracking. No model weight files appeared in normal changed/untracked status.

### Validation

Targeted workflow tests:

```text
uv run --extra dev pytest tests/test_build_unified_toolcalling_dataset.py tests/test_build_toolcalling_numerical_representation.py tests/test_build_toolcalling_shift_inventory.py tests/test_analyze_tau2_shift_uncertainty.py tests/test_plan_tau2_additional_sampling.py tests/test_build_tau2_stage1_manifest.py tests/test_run_tau2_stage1.py tests/test_ingest_tau2_stage1_results.py tests/test_build_tau2_stage1_analysis.py -q
```

Result: 83 passed, 2 warnings. The warnings were the existing voice `audioop` deprecation warning and an unknown pytest config option warning for `asyncio_default_fixture_loop_scope`.

Ruff over all changed Python source and test files passed.

`git diff --check` passed.

Full-repository pytest collection with only `--extra dev` failed separately because optional dependencies for unrelated suites are not installed: `a2a`, `agentify_tau_bench`, `websockets`, `rank_bm25`, `gymnasium`, and `pyaudio`. The collection check reported 895 collected tests and 13 collection errors. This is not a failure of the targeted Stage 1 workflow tests.

### Main result

Verified facts from artifacts:

- Original tau2 records: 93.
- Stage 1 retained records: 12.
- Merged tau2 records: 105.
- Merged y distribution: 68 zero and 37 one.
- Task selection uses X/task characteristics and does not use `y`.
- All six baseline tau2 shift definitions are preserved.
- `tau2_zero_or_one_write_to_two_plus_writes` is `candidate_harmful` at d=0.05 and d=0.10, and `inconclusive` at d=0.15.
- Baseline outputs were not overwritten by versioned Stage 1 outputs.

### Limitations

The Stage 1 analysis remains exploratory and small-sample. Candidate shifts reuse records across non-independent definitions. API-Bank synthetic negatives remain API-call-level synthetic correctness labels and are not evidence of natural LLM failure rates. The lockfile adds the `sentence-transformers` stack and includes Hugging Face resolver churn that should be reviewed with the dependency diff.

### Review status

Ready for human review of the proposed inclusion/exclusion set in `docs/pr_tau2_stage1_shift_analysis.md`. The branch is not staged, committed, pushed, or opened as a PR.

### Next step

Stage the proposed inclusion set, leave excluded local artifacts untracked/ignored, then commit with proposed message `feat: add tau2 stage1 shift analysis workflow` and open a PR titled `Add tau2 Stage 1 context-shift analysis workflow`.

## 2026-07-20 — BFCL Shift Analysis

### Objective

Add BFCL as a third, real evaluated tool-calling dataset to compare category-level context shifts against tau2 task-level outcomes and API-Bank evaluator-pipeline records. BFCL was added because it provides more real evaluator-labeled outcomes than the current tau2 sample while avoiding API-Bank's synthetic-negative limitation.

### Source data

- Source file: `data/processed/bfcl/bfcl_v4_non_live_1240_xy.jsonl`.
- Compact source summary: `data/processed/bfcl/bfcl_v4_non_live_1240_summary.json`.
- Records: 1,240 unique BFCL v4 non-live single-turn evaluated model outcomes.
- Model: `gpt-4o-mini-2024-07-18-FC`.
- Label scope: `test_case_level`.
- Label origin: `bfcl_evaluator`.
- Synthetic rows: false.
- Y distribution: 1,059 successes and 181 failures.
- Category counts and success rates: `simple_python` 400, 350/400 = 0.8750; `multiple` 200, 176/200 = 0.8800; `parallel` 200, 174/200 = 0.8700; `parallel_multiple` 200, 160/200 = 0.8000; `irrelevance` 240, 199/240 = 0.8292.

### X, S, Y representation

- X is stored in `x_raw` and contains BFCL prompt/context and function information.
- S is stored in `s_raw` and contains the evaluated model result.
- Y is stored in `y` as binary test-case correctness from the BFCL evaluator.
- The BFCL builder now validates that every row has `id`, `x_raw`, `s_raw`, and valid `y`.

### Candidate shifts

Primary complexity shifts:

- `bfcl_simple_python_to_multiple`
- `bfcl_simple_python_to_parallel`
- `bfcl_simple_python_to_parallel_multiple`
- `bfcl_multiple_to_parallel_multiple`
- `bfcl_parallel_to_parallel_multiple`

Behavioral/abstention shift:

- `bfcl_simple_python_to_irrelevance`

Candidate groups are defined only from BFCL `category` metadata. They do not use `y`, `s_raw`, evaluator errors, label scope, label origin, or synthetic-status fields.

### Threshold-sensitive findings

The BFCL uncertainty analysis uses Newcombe-Wilson 95% confidence intervals, deterministic bootstrap with 10,000 replicates and seed 1, two-proportion p-values, and Benjamini-Hochberg adjustment across the six BFCL shifts. Under the full-CI rule, all six shifts are inconclusive at `d=0.05`; `simple_python -> multiple` and `simple_python -> parallel` are candidate harmless at `d=0.10`; five of six shifts are candidate harmless at `d=0.15`; and no BFCL shift is candidate harmful at any tested threshold.

### Cross-dataset interpretation

BFCL, tau2, and API-Bank are complementary studies with different label scopes and are not pooled as IID rows. tau2 currently has 105 real task-level outcomes; API-Bank has 1,016 API-call-level correctness records with 508 reference positives and 508 synthetic negatives; BFCL has 1,240 real evaluated test-case-level outcomes.

### Git inclusion set

Proposed included files:

- `scripts/build_bfcl_shift_inventory.py`
- `scripts/analyze_bfcl_shift_uncertainty.py`
- `tests/test_build_bfcl_shift_inventory.py`
- `tests/test_analyze_bfcl_shift_uncertainty.py`
- `docs/bfcl_data_source_audit.md`
- `docs/bfcl_shift_inventory.md`
- `docs/bfcl_shift_uncertainty.md`
- `docs/toolcalling_cross_dataset_findings.md`
- `docs/toolcalling_shift_project_log.md`
- `data/processed/bfcl/bfcl_v4_non_live_1240_xy.jsonl`
- `data/processed/bfcl/bfcl_v4_non_live_1240_summary.json`
- `data/processed/bfcl/bfcl_v4_non_live_shift_inventory.jsonl`
- `data/processed/bfcl/bfcl_v4_non_live_shift_inventory_summary.json`
- `data/processed/bfcl/bfcl_v4_non_live_shift_uncertainty.jsonl`
- `data/processed/bfcl/bfcl_v4_non_live_shift_uncertainty_summary.json`

Explicit exclusions:

- Gorilla raw result directories.
- Gorilla score directories.
- API keys and `.env` files.
- Temporary canary files.
- Caches.
- Generated Python bytecode.

The 2.1 MB sample-level BFCL JSONL is included because it is the direct reproducibility input. The compact BFCL input summary is included. No new `.gitignore` rule was needed in this pass because the required BFCL reproducibility inputs are visible to Git, while existing ignore rules already exclude `.env`, caches, temporary files, and Python bytecode; no Gorilla raw or score directories were present in this working tree.

### Limitations

The BFCL analysis is exploratory, category-level, and non-causal. It verifies local processed artifacts and does not re-run the upstream BFCL evaluator. Candidate contrasts reuse category groups across shifts and should not be treated as independent discoveries. Results do not imply deployment safety or a retraining requirement.
