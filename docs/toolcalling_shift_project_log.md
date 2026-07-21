# LLM Tool-Calling Context Shift Project Log

Last consolidated: 2026-07-21.

This is the canonical handoff document for the LLM tool-calling context-shift
project. Future ChatGPT, Codex, and Claude sessions should read this file before
editing project artifacts.

Log rules:

- Distinguish completed work, current conclusions, and planned work.
- Do not claim that smoke tests or compatibility runs are performance
  benchmarks.
- Do not describe synthetic API-Bank negatives as naturally occurring LLM
  failures.
- Preserve label-scope distinctions across tau2, BFCL, and API-Bank.
- Preserve `Y=0` labels. Do not redefine all outcomes as `Y=1`.
- Do not pool tau2, BFCL, and API-Bank as IID samples from one task.
- Keep code and data changes out of this log unless the user explicitly asks for
  implementation work.

## 1. Project Overview

Research motivation: an LLM tool-calling agent can work at one company or
deployment setting and then be deployed into another setting with different
tools, policies, workflows, business objects, user requests, or error modes. The
naive response to any such context change is to collect new data and adapt,
fine-tune, or retrain. That can be expensive and unnecessary.

The core deployment question is a company/source-context to target-context
question: given data from a source context, can we tell whether a target context
shift is likely to change benchmark success before deciding to adapt the model?

The project goal is to identify shifts that are:

- harmful: target success is meaningfully lower than source success;
- harmless: target success is practically unchanged;
- beneficial: target success is meaningfully higher;
- inconclusive: current samples and intervals do not support one of the above.

The practical value is to avoid unnecessary adaptation or retraining when the
shift changes `X` or `S` but not the probability of success, while still
surfacing contexts where success appears to degrade.

## 2. Formal Formulation

`X` is the task or deployment context. It may include the user request, domain,
available tools, tool schemas, policies, expected action counts, environment
state, and other pre-execution context.

`S` is the LLM/tool-calling interaction trajectory. It can include user
messages, assistant messages, tool calls, arguments, tool outputs, errors,
retries, termination state, or a derived numeric trajectory representation.

`Y` is benchmark success or correctness. It is binary in the current
implementations, but its scope depends on the dataset:

- tau2: task-level benchmark success;
- BFCL: test-case-level evaluator correctness;
- API-Bank: API-call-level correctness under the current pilot construction.

The outcome shift is:

```text
Delta_Y = P_target(Y=1) - P_source(Y=1)
```

`Y` is not fixed to 1. Failures and synthetic negatives are preserved as `Y=0`
where the source artifacts define them.

Current practical thresholds are `delta_practical = 0.05, 0.10, 0.15`.
Classifications use the full 95% confidence interval for `Delta_Y`:

- `candidate_harmful`: upper 95% CI < `-d`;
- `candidate_harmless`: full 95% CI is inside `[-d, d]`;
- `candidate_beneficial`: lower 95% CI > `d`;
- `inconclusive`: all other cases.

The primary interval in current tau2 and BFCL analyses is a Newcombe-Wilson
difference-in-proportions interval. A deterministic nonparametric bootstrap
interval with 10,000 replicates and seed 1 is also recorded. Multiple testing is
adjusted with Benjamini-Hochberg across the analyzed shifts for each dataset.

These labels are exploratory statistical labels only. They do not imply
causality, deployment safety, or an automatic retraining decision.

## 3. Relationship To Minxing's Method

Minxing Zheng's method lives in the sibling repository:

```text
/Users/xuyida/Research/physics_informed_testing
```

Reference entry point:

```text
/Users/xuyida/Research/physics_informed_testing/share_code/experiment/run_baseline.py
```

What is being transferred from Minxing's method setting:

- the external-data pickle interface;
- deterministic train/validation splitting;
- existing model modes and objectives;
- the idea of using context `X` and trajectory `S` to diagnose safety or
  success-related shifts.

The abstract project formulation is `X/S/Y` for LLM tool-calling. The current
Minxing reference implementation is physics-specific and uses reconstructed
trajectory safety scores. The current bridge is therefore a compatibility
bridge, not a proof that the physics-specific thresholded reconstruction
semantics already match tool-calling success.

Exact Minxing pickle contract used by the bridge:

- top-level keys: `X`, `y`, `traj`;
- `X`: `float32` array with shape `(N, d_x)`;
- `y`: binary array with shape `(N,)`;
- `traj["s"]`: `float32` array with shape `(N, T)`;
- `X`, `y`, and `traj["s"]` must share the same first dimension;
- Minxing casts arrays to `float32`, builds one-step sequence pairs from
  `traj["s"]`, and splits with
  `numpy.random.RandomState(seed).permutation`.

Current semantic mismatch: `proposed_only` and `reconstruction_only` make
safety-style predictions by thresholding reconstructed trajectories under a time
window. tau2, BFCL, and API-Bank labels instead encode benchmark success or
correctness. This mismatch likely explains the observed one-class sequence-mode
collapses.

This does not invalidate benchmark success as `Y`. It means Minxing's current
thresholded reconstructed-trajectory score is not yet proven to be a valid
surrogate for LLM tool-calling success. Benchmark success remains the correct
outcome variable for this project; the semantic bridge from reconstruction score
to success still needs design and validation.

## 4. Dataset Inventory

### tau2

- Source repository: `/Users/xuyida/Research/llm-toolcalling-benchmarks/tau2-bench`,
  origin `https://github.com/YidaXu04/tau2-bench.git`, upstream
  `https://github.com/sierra-research/tau2-bench.git`.
- Current real shift-analysis sample count: 105 task-level outcomes after
  Stage 1, from 93 baseline retained records plus 12 Stage 1 retained records.
- Original Minxing-compatibility pickle sample count: 93 filtered retail/airline
  records in `data/processed/tau2_l2t_success_retail_airline_50_filtered_20260714.pkl`.
- X representation, unified baseline: 12 structured numeric features including
  domain indicators, expected action/read/write counts, DB mutation requirement,
  and evaluator-component indicators.
- X representation, numerical artifact: 384-dimensional text embedding plus 18
  structural features, final `X=(N,402)`.
- S representation, unified baseline: fixed-length 64-event structural
  trajectory with event codes for user, assistant, read call, write call, tool
  response, tool error, end, and padding.
- S representation, numerical artifact: 384-dimensional text embedding plus 20
  structural features, final `S=(N,404)`.
- Y definition: `y=1` only when the converted benchmark reward is 1.0;
  otherwise `y=0`.
- Label scope: `task_level`.
- Label origin: `tau2_benchmark_reward`.
- Synthetic status: non-synthetic.
- Class counts: baseline 93 has 60 failures and 33 successes; Stage 1 adds 8
  failures and 4 successes; merged 105 has 68 failures and 37 successes.
- Important limitations: small sample sizes; candidate shifts reuse records
  across definitions; current S is coarse and does not preserve full tool names,
  arguments, outputs, or message semantics; observed trajectory-length/tool-call
  groups cannot be assigned before running a task.

### BFCL

- Source repository: `/Users/xuyida/Research/llm-toolcalling-benchmarks/gorilla`,
  origin `https://github.com/ShishirPatil/gorilla.git`.
- Processed source file:
  `data/processed/bfcl/bfcl_v4_non_live_1240_xy.jsonl`.
- Compact source summary:
  `data/processed/bfcl/bfcl_v4_non_live_1240_summary.json`.
- Sample count: 1,240 BFCL v4 non-live single-turn evaluated model outcomes.
- Model: `gpt-4o-mini-2024-07-18-FC`.
- X representation: `x_raw` stores prompt/context and function information.
  L2T bridge converts this to 17 structural features from BFCL prompt and tool
  schema context only.
- S representation: `s_raw` stores evaluated model result. L2T bridge converts
  `s_raw.model_result` to a 32-step event sequence.
- Y definition: binary BFCL evaluator correctness for the test case.
- Label scope: `test_case_level`.
- Label origin: `bfcl_evaluator`.
- Synthetic status: non-synthetic.
- Class counts: 1,059 successes and 181 failures; overall accuracy 0.8540.
- Category counts: `simple_python` 400, `multiple` 200, `parallel` 200,
  `parallel_multiple` 200, `irrelevance` 240.
- Category success rates: simple 0.8750, multiple 0.8800, parallel 0.8700,
  parallel-multiple 0.8000, irrelevance 0.8292.
- Important limitations: category-level exploratory contrasts; does not rerun
  the upstream BFCL evaluator; candidate contrasts reuse category groups; no
  deployment-safety or retraining-required claim follows from these labels.
  BFCL irrelevance includes an abstention criterion, so call-vs-no-call behavior
  is part of that category's evaluator definition and should be reported
  separately from non-irrelevance tool-call competence.

### API-Bank

- Source repository: `/Users/xuyida/Research/llm-toolcalling-benchmarks/DAMO-ConvAI`,
  origin `https://github.com/YidaXu04/DAMO-ConvAI.git`, upstream
  `https://github.com/AlibabaResearch/DAMO-ConvAI.git`.
- Unified source file:
  `data/processed/unified_toolcalling_apibank.jsonl`.
- Sample count: 1,016 API-call-level records, composed of 508 reference positive
  API calls and 508 synthetic corrupted negatives.
- X representation, unified: 5 numeric features:
  `assistant_turn_count`, `available_api_count`, `history_length`,
  `previous_api_call_count`, `user_turn_count`.
- X representation, numerical/L2T: rows from
  `toolcalling_numerical_full.npz["X"][source_dataset == "api_bank"]`, final
  `X=(1016,402)`.
- S representation, unified: 4 numeric features:
  `api_name_id`, `argument_count`, `has_exception`, `output_is_success`.
- S representation, numerical/L2T: rows from
  `toolcalling_numerical_full.npz["S"][source_dataset == "api_bank"]`, final
  `S=(1016,404)`.
- Y definition: `Y=1` for correct reference API calls and `Y=0` for validated
  synthetic corrupted calls.
- Label scope: `api_call_level`.
- Label origin: 508 `reference_api_call`, 508 `synthetic_corruption`.
- Synthetic status: positives are non-synthetic references; negatives are
  synthetic corruptions.
- Class counts: 508 successes and 508 failures.
- Corruption types: `missing_required_argument` 137, `wrong_api_name` 140,
  `wrong_argument_type` 114, `wrong_argument_value` 117.
- Important limitations: negatives are not natural LLM failures; labels are
  balanced by construction; positive/negative paired records can leak across
  ordinary random splits unless split by pair/group; API-Bank deltas are not
  valid estimates of natural deployment success-rate shifts. The supervised
  diagnostic row-level split placed 160/508 pairs across train and validation,
  so pair-grouped splitting is required for clean supervised API-Bank
  diagnostics.

## 5. Tau2 Work Completed

Task/domain filtering:

- Local task counts observed earlier: retail 114, airline 50, telecom 2,285,
  banking knowledge 97.
- Initial retained records: 46 retail and 47 airline after filtering, 93 total.
- Filtering requires normal termination, valid reward information, required
  evaluator checks, and auditable conversion logic.
- Telecom was tested for feasibility: one task took about 362 seconds, hit a
  TPM/rate-limit issue, reached max steps, had reward 0, and cost about
  `0.095`. Telecom remains technically possible but unsuitable for immediate
  scaling without a cost-control plan.

Source/target setup:

- Baseline source/target contrast uses retail as source and airline as target.
- Additional tau2 shifts compare write requirement, expected action count,
  observed trajectory length, and observed tool-call count.

Numerical representation:

- Full unified numerical artifact for tau2 plus API-Bank has `X=(1109,402)`,
  `S=(1109,404)`, `y=(1109,)`.
- Embedding model is `sentence-transformers/all-MiniLM-L6-v2`; manifest package
  version is 5.6.0.
- No duplicate sample IDs, NaNs, or infinite values were reported.
- Model-facing leakage fields such as `y`, `label_origin`, `is_synthetic`,
  `variant`, `corruption_type`, `validation_status`, and `validation_error` are
  excluded from model-facing text and structural features.

Stage 1 shift analysis:

- Stage 1 selected 12 unused retail tasks by X/task characteristics, not by
  observed `Y`.
- Selected task IDs: `54`, `55`, `64`, `71`, `72`, `74`, `76`, `81`, `57`,
  `62`, `50`, `70`.
- Composition: 8 `two_plus_writes`, 2 `no_write`, 2 `low_action_one_write`.
- Execution/ingestion completed: 12 attempted, 12 completed, 12 retained, 0
  filtered.
- Stage 1 outcome distribution: 8 failures, 4 successes.
- Observed Stage 1 cost: 0.0706581.
- Merged tau2 record count: 105, with 68 failures and 37 successes.

Harmful candidate results:

- Before Stage 1, `tau2_zero_or_one_write_to_two_plus_writes` was
  `candidate_harmful` at `d=0.05`, `d=0.10`, and `d=0.15`.
- After Stage 1, the same shift has `Delta_Y=-0.3271`,
  95% CI `[-0.4634, -0.1371]`, and is `candidate_harmful` at `d=0.05` and
  `d=0.10`, but `inconclusive` at `d=0.15`.
- All other current tau2 shifts are inconclusive under the full-CI rule.
- CI widths decreased for all six tau2 shifts after Stage 1.

Uncertainty and additional-sampling analysis:

- Baseline and Stage 1 analyses use Newcombe-Wilson 95% intervals, deterministic
  bootstrap intervals with 10,000 replicates and seed 1, and Benjamini-Hochberg
  adjustment.
- Additional sampling plans were generated without running new simulations at
  the planning stage.
- The Stage 1 plan intentionally targeted the multiple-write contrast and does
  not support causal interpretation.

Previous Minxing compatibility run and collapse result:

- The original 93-sample tau2 Minxing run used `X=(93,12)`, `S=(93,64)`.
- The earlier 30-epoch `proposed` compatibility run predicted all validation
  samples as `Y=0`; validation accuracy equaled the validation majority-class
  rate 12/19 = 0.6316.
- The current 5-epoch compatibility matrix reproduces the sequence-mode
  collapse for tau2 in `proposed_only` and `reconstruction_only`.

## 6. API-Bank Work Completed

Pilot/full construction:

- Local API-Bank level-1/level-2 inspection found 264 dialogues, 508 API
  reference steps, 508 API events, and zero exceptions in reference trajectories.
- No released model-prediction files were found locally.
- The pilot creates a balanced API-call-level dataset from reference positives
  and synthetic negatives.

Positive and synthetic-negative counts:

- 508 positive reference API calls.
- 508 synthetic negative API calls.
- 1,016 total API-call-level records.

Corruption types:

- `missing_required_argument`: 137.
- `wrong_api_name`: 140.
- `wrong_argument_type`: 114.
- `wrong_argument_value`: 117.
- 508/508 negatives were validated as incorrect through evaluator/API
  correctness logic.
- `GetToday` supported API-name corruption only; fallback count was 25.
- Sensitive values were redacted; the first reported build recorded 2,872
  redacted sensitive values.

Unified/numerical representation:

- Unified API-Bank X dimension: 5.
- Unified API-Bank S dimension: 4.
- Numerical/L2T API-Bank `X=(1016,402)`, `S=(1016,404)`, `y=(1016,)`.
- Positive and negative pair members share pre-call `X` when both are selected;
  candidate-call `S` differs.

Limitations and paired-sample leakage concern:

- Synthetic negatives are not natural LLM failures.
- API-Bank is suitable for schema, representation, evaluator, and pipeline
  diagnostics, not for estimating real deployment harmful shifts.
- Random row-level train/validation splits can put paired positive/negative
  records across splits. The corrected supervised diagnostic uses grouped
  splitting by pair ID. The previous row-level split crossed 160/508 pairs.

## 7. BFCL Work Completed

BFCL version/subsets/model:

- Dataset: BFCL v4 non-live single-turn subset.
- Processed rows: 1,240.
- Model: `gpt-4o-mini-2024-07-18-FC`.
- Categories: `simple_python`, `multiple`, `parallel`, `parallel_multiple`,
  `irrelevance`.

Per-category and overall results:

- Overall: 1,059/1,240 correct, success rate 0.8540.
- `simple_python`: 350/400 correct, 0.8750.
- `multiple`: 176/200 correct, 0.8800.
- `parallel`: 174/200 correct, 0.8700.
- `parallel_multiple`: 160/200 correct, 0.8000.
- `irrelevance`: 199/240 correct, 0.8292.

Shift analysis findings:

- Six BFCL category contrasts were analyzed.
- At `d=0.05`, all six are inconclusive.
- At `d=0.10`, `bfcl_simple_python_to_multiple` and
  `bfcl_simple_python_to_parallel` are `candidate_harmless`; the other four are
  inconclusive.
- At `d=0.15`, five are `candidate_harmless`; only
  `bfcl_multiple_to_parallel_multiple` remains inconclusive.
- No BFCL shift is `candidate_harmful` or `candidate_beneficial` under the
  tested full-CI rule.
- `bfcl_simple_python_to_irrelevance` is a behavioral/abstention contrast, not a
  primary complexity shift.
- BFCL irrelevance rows should be interpreted separately because correct
  abstention is part of the evaluator definition. In the current encoding,
  correct irrelevance behavior is often free text/no function call and incorrect
  behavior is typically a function call.

Label semantics:

- `Y=1` means BFCL evaluator marked the test case correct.
- `Y=0` means incorrect under the BFCL evaluator.
- Label scope is `test_case_level`; label origin is `bfcl_evaluator`; rows are
  non-synthetic.

PR/merge status where recoverable:

- tau2 Stage 1 workflow was merged as PR #1, commit `3b09ca5`, with feature
  commit `2c7744d`.
- BFCL shift analysis was merged as PR #2, commit `00ddf01`, with feature
  commit `acf3ea4`.
- Current HEAD is `00ddf01` and branch `feat/l2t-multisource-model-bridge` is
  at the same commit as local `main`, `origin/main`, and `origin/HEAD`, before
  the current untracked L2T bridge files are committed.

## 8. L2T Bridge Work On Current Branch

Current branch:

```text
feat/l2t-multisource-model-bridge
```

Converter scripts:

- `scripts/l2t_model_bridge.py`
- `scripts/convert_bfcl_to_l2t_pkl.py`
- `scripts/convert_apibank_to_l2t_pkl.py`
- `scripts/evaluate_l2t_supervised_baselines.py`
- `scripts/run_minxing_l2t_compatibility.py`

Shared bridge utilities:

- `CONVERTER_VERSION = "l2t_model_bridge_20260720"`.
- Shared validation checks required keys, shapes, binary labels, duplicate sample
  IDs, NaN counts, infinite-value counts, class distribution, and Minxing split
  summaries.
- Shared contract summary records `X`, `y`, `traj["s"]`, and loader behavior.

Generated pickle paths:

- BFCL:
  `data/processed/l2t/bfcl/bfcl_v4_non_live_1240_l2t.pkl`.
- API-Bank:
  `data/processed/l2t/apibank/apibank_full_l2t.pkl`.
- Existing tau2 compatibility pickle:
  `data/processed/tau2_l2t_success_retail_airline_50_filtered_20260714.pkl`
  is ignored in Git and should not be committed.
- BFCL/API-Bank L2T pickles under `data/processed/l2t/**/*.pkl` are also
  ignored local derived artifacts regenerated by their converter scripts.

Manifests:

- `data/processed/l2t/bfcl/bfcl_v4_non_live_1240_l2t_manifest.json`.
- `data/processed/l2t/apibank/apibank_full_l2t_manifest.json`.
- `data/processed/l2t/diagnostics/l2t_supervised_diagnostic_summary.json`.
- `data/processed/l2t/diagnostics/l2t_supervised_diagnostic_results.csv`.
- `data/processed/l2t/diagnostics/l2t_supervised_diagnostic_best_by_view.csv`.
- `data/processed/l2t/diagnostics/minxing_compatibility_5ep/l2t_compat_comparison_5ep.csv`.
- `data/processed/l2t/diagnostics/minxing_compatibility_5ep/minxing_compatibility_summary_5ep.json`.
- `data/processed/l2t/diagnostics/minxing_compatibility_5ep/commands_configuration.json`.
- `data/processed/l2t/diagnostics/minxing_compatibility_5ep/dataset_mode_metadata.json`.

Exact X/S/y shapes and dtypes:

| Dataset | X | S / `traj["s"]` | y | Class counts |
| --- | ---: | ---: | ---: | --- |
| tau2 compatibility pickle | `(93,12)` | `(93,64)` | `(93,)` | 60 `Y=0`, 33 `Y=1` |
| BFCL L2T | `float32 (1240,17)` | `float32 (1240,32)` | `int64 (1240,)` | 181 `Y=0`, 1059 `Y=1` |
| API-Bank L2T | `float32 (1016,402)` | `float32 (1016,404)` | `int64 (1016,)` | 508 `Y=0`, 508 `Y=1` |

Validation performed:

- BFCL and API-Bank manifests report no duplicate sample IDs.
- BFCL and API-Bank manifests report NaN counts `X=0`, `traj_s=0`.
- BFCL and API-Bank manifests report infinite-value counts `X=0`, `traj_s=0`.
- BFCL manifest excludes evaluator error fields, label metadata, synthetic
  status, and `y` from model-facing arrays.
- API-Bank manifest carries forward the numerical artifact leakage audit and
  excludes label scope, label origin, synthetic flag, corruption type,
  validation status/error, and `y`.
- API-Bank converter now preserves `pair_id` as non-model metadata in
  `metadata[].pair_id` and `group_ids`; these fields are excluded from
  model-facing `X` and `traj["s"]`.
- BFCL sequence encoding canonically sorts function names and argument keys for
  deterministic output, trading away original JSON key order.

Tests added:

- `tests/test_l2t_model_bridge_converters.py`
- `tests/test_evaluate_l2t_supervised_baselines.py`
- `tests/test_run_minxing_l2t_compatibility.py`

## 9. Supervised Diagnostic Baselines

Purpose: check whether the bridge representations contain ordinary supervised
signal for `Y` under controlled classifiers. This is not a replacement for
Minxing's L2T objective.

BFCL split: deterministic row split with `seed=1`, implemented to match the
split logic observed in Minxing's referenced entry point:

```python
perm = np.random.RandomState(seed).permutation(N)
n_train = int(0.8 * N)
```

API-Bank split: deterministic grouped split by `pair_id` with `seed=1`. The
previous row-level split crossed 160/508 pairs. The grouped split has 406 train
pairs and 102 validation pairs, 812/204 rows, class counts 406/406 and 102/102,
and zero cross-split pairs.

Feature views: `X-only`, `S-only`, and `X+S`, evaluated separately by dataset
and BFCL subset.
Preprocessing is fitted on training data only through sklearn pipelines.

Best non-permuted results:

| Dataset | Subset | View | Best model | Acc | Bal acc | Macro-F1 | Confusion matrix |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| BFCL | all categories | X+S | small MLP | 0.9032 | 0.6338 | 0.6737 | `[[8,20],[4,216]]` |
| BFCL | non-irrelevance | X+S | class-weighted logistic regression | 0.7150 | 0.6606 | 0.5667 | `[[13,9],[48,130]]` |
| BFCL | irrelevance only | S-only | logistic regression | 1.0000 | 1.0000 | 1.0000 | `[[8,0],[0,40]]` |
| BFCL | irrelevance only | X+S | logistic regression | 1.0000 | 1.0000 | 1.0000 | `[[8,0],[0,40]]` |
| API-Bank | all pairs | X-only | small MLP | 0.5000 | 0.5000 | 0.4988 | `[[46,56],[46,56]]` |
| API-Bank | all pairs | S-only | random forest | 0.9461 | 0.9461 | 0.9460 | `[[93,9],[2,100]]` |
| API-Bank | all pairs | X+S | small MLP | 0.9510 | 0.9510 | 0.9509 | `[[93,9],[1,101]]` |

Repeated shuffled-label controls now use 10 deterministic trials per
dataset/subset/view/model. Selected null summaries:

| Dataset | Subset | View/model | Bal acc mean/std/min/max | Macro-F1 mean |
| --- | --- | --- | ---: | ---: |
| BFCL | all categories | X+S small MLP | 0.5087 / 0.0108 / 0.4886 / 0.5286 | 0.4975 |
| BFCL | irrelevance only | S-only logistic regression | 0.5062 / 0.0187 / 0.5000 / 0.5625 | 0.4662 |
| API-Bank | all pairs | S-only random forest | 0.4966 / 0.0425 / 0.4265 / 0.5637 | 0.4954 |
| API-Bank | all pairs | X+S small MLP | 0.5088 / 0.0255 / 0.4608 / 0.5441 | 0.5082 |

Interpretation:

- BFCL contains supervised signal, but it is split between general tool-call
  competence and irrelevance abstention behavior. Non-irrelevance results are
  now reported separately from all-category results.
- API-Bank has strong signal in `S`, not in `X`. This matches the construction:
  paired positive/negative records share pre-call `X`, while candidate-call `S`
  differs. These are synthetic negatives, not natural LLM failures.
- Corrected leakage audit reports no exact `y` or exact `1-y` columns in
  API-Bank `X` or `S`, while documenting constant/inert API-Bank structural
  slots: 12 constant `X` columns and 15 constant `S` columns in the current
  pilot.
- The supervised baselines show the bridge can expose `Y` signal. The Minxing
  sequence-mode collapse is therefore more consistent with objective/semantic
  mismatch than with a completely signal-free bridge.

Leakage concerns:

- Numeric-array field-name audits did not find explicit label metadata in
  model-facing arrays.
- Pair leakage is addressed for API-Bank supervised diagnostics with grouped
  splitting; it remains a concern for any row-level ablation or compatibility
  run that uses synthetic pairs.
- Do not use these supervised baselines as final scientific evidence for
  harmful/harmless deployment shifts.

## 10. Minxing Compatibility Study

Three existing model modes tested:

- `proposed_only`: Minxing model name `proposed`; inputs `X` and `S`; objective
  `stability_asymmetric_recon(delta=10.0) + lambda_mmd=0.01 * MMD`.
- `label_bce`: Minxing model name `label_bce`; input `X` only; binary
  cross-entropy on benchmark success label.
- `reconstruction_only`: Minxing model name `reconstruction_only`; inputs `X`
  and `S`; plain sequence reconstruction MSE with `asym_recon_delta=1` and
  `lambda_mmd=0`.

All nine 5-epoch results used `epochs=5`, `seed=1`, no sweeps, and dataset-
specific time windows. Compact table:

| Dataset | Mode | Inputs | Train/Val | Val dist | Acc | Bal acc | Macro-F1 | Y=0 recall | Y=1 recall | Confusion matrix | Collapse |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| tau2 | `proposed_only` | X+S | 74/19 | 12/7 | 0.632 | 0.500 | 0.387 | 1.000 | 0.000 | `[[12,0],[7,0]]` | Y=0 |
| tau2 | `label_bce` | X | 74/19 | 12/7 | 0.526 | 0.446 | 0.424 | 0.750 | 0.143 | `[[9,3],[6,1]]` | no |
| tau2 | `reconstruction_only` | X+S | 74/19 | 12/7 | 0.632 | 0.500 | 0.387 | 1.000 | 0.000 | `[[12,0],[7,0]]` | Y=0 |
| BFCL | `proposed_only` | X+S | 992/248 | 28/220 | 0.113 | 0.500 | 0.101 | 1.000 | 0.000 | `[[28,0],[220,0]]` | Y=0 |
| BFCL | `label_bce` | X | 992/248 | 28/220 | 0.887 | 0.500 | 0.470 | 0.000 | 1.000 | `[[0,28],[0,220]]` | Y=1 |
| BFCL | `reconstruction_only` | X+S | 992/248 | 28/220 | 0.113 | 0.500 | 0.101 | 1.000 | 0.000 | `[[28,0],[220,0]]` | Y=0 |
| API-Bank | `proposed_only` | X+S | 812/204 | 100/104 | 0.510 | 0.500 | 0.338 | 0.000 | 1.000 | `[[0,100],[0,104]]` | Y=1 |
| API-Bank | `label_bce` | X | 812/204 | 100/104 | 0.549 | 0.548 | 0.547 | 0.490 | 0.606 | `[[49,51],[41,63]]` | no |
| API-Bank | `reconstruction_only` | X+S | 812/204 | 100/104 | 0.510 | 0.500 | 0.338 | 0.000 | 1.000 | `[[0,100],[0,104]]` | Y=1 |

Format compatibility conclusion:

- All three converted pickle files satisfy Minxing's external-data contract.
- All three datasets load, produce sequence pairs, train through selected
  existing modes, and emit validation metrics under the deterministic split.

Semantic compatibility conclusion:

- Semantic compatibility is not established.
- Sequence modes use a thresholded reconstructed-trajectory safety score, while
  `Y` here is benchmark success/correctness.
- `label_bce` directly targets `Y`, but it is an X-only supervised baseline, not
  evidence for the reconstruction/MMD shift-detection method.

Collapse findings:

- tau2 sequence modes collapse to `Y=0`.
- BFCL sequence modes collapse to `Y=0`, the validation minority class.
- API-Bank sequence modes collapse to `Y=1`.
- BFCL `label_bce` collapses to `Y=1`, matching the validation majority class.

Why longer 30-epoch sequence-mode runs are not currently justified:

- The 5-epoch sequence modes already collapse under the current semantic
  interpretation.
- Longer `proposed_only` or `reconstruction_only` runs would mostly test whether
  a mismatched induced classifier remains collapsed.
- A cautious 30-epoch API-Bank `label_bce` run could answer an X-only supervised
  baseline question, but not the Minxing reconstruction/MMD harmful-shift
  question.
- The next scientific action should clarify the semantic bridge between
  tool-calling success labels and Minxing's safety score.

Maintained reproducibility runner:

- `scripts/run_minxing_l2t_compatibility.py`
- The runner imports Minxing's existing `share_code/experiment/run_baseline.py`
  at runtime, sets `sys.dont_write_bytecode = True`, and fingerprints Minxing
  source files before and after a run. It fails if imported Minxing source files
  change.

Compact artifact bundle:

```text
data/processed/l2t/diagnostics/minxing_compatibility_5ep/
```

The bundle intentionally excludes large prediction CSVs, training histories,
checkpoints, and redundant result folders.

## 11. Current Repository Structure And Responsibilities

`tau2-bench`:

- Main working repository for this project.
- Path: `/Users/xuyida/Research/llm-toolcalling-benchmarks/tau2-bench`.
- Current branch: `feat/l2t-multisource-model-bridge`.
- It is acceptable to modify this repository when asked, subject to the user's
  scope restrictions.

`DAMO-ConvAI`:

- Source/reference repository for API-Bank.
- Path: `/Users/xuyida/Research/llm-toolcalling-benchmarks/DAMO-ConvAI`.
- Should not be modified for this branch.
- Current status observed on 2026-07-21: `main...upstream/main [ahead 2]`.

`gorilla`:

- Source/reference repository for BFCL.
- Path: `/Users/xuyida/Research/llm-toolcalling-benchmarks/gorilla`.
- Should not be modified for this branch.
- Current status observed on 2026-07-21: `main...origin/main` with untracked
  BFCL local generated files:
  `berkeley-function-call-leaderboard/bfcl_canary_sample_labels.jsonl`,
  `berkeley-function-call-leaderboard/bfcl_v4_non_live_1240_summary.json`,
  `berkeley-function-call-leaderboard/bfcl_v4_non_live_1240_xy.jsonl`, and
  `berkeley-function-call-leaderboard/local_canary_backup/`.

`physics_informed_testing`:

- Minxing's reference implementation repository.
- Path: `/Users/xuyida/Research/physics_informed_testing`.
- Should not be modified unless the user explicitly authorizes Minxing-core
  changes.
- Current status observed on 2026-07-21: `main...origin/main` with untracked
  result folders for the 5-epoch compatibility matrix and
  `results/l2t_compat_comparison_5ep.csv`.

## 12. Important Files

Core scripts:

- `scripts/build_unified_toolcalling_dataset.py`
- `scripts/build_toolcalling_numerical_representation.py`
- `scripts/build_toolcalling_shift_inventory.py`
- `scripts/analyze_tau2_shift_uncertainty.py`
- `scripts/plan_tau2_additional_sampling.py`
- `scripts/build_tau2_stage1_manifest.py`
- `scripts/run_tau2_stage1.py`
- `scripts/ingest_tau2_stage1_results.py`
- `scripts/build_tau2_stage1_analysis.py`
- `scripts/build_bfcl_shift_inventory.py`
- `scripts/analyze_bfcl_shift_uncertainty.py`
- `scripts/l2t_model_bridge.py`
- `scripts/convert_bfcl_to_l2t_pkl.py`
- `scripts/convert_apibank_to_l2t_pkl.py`
- `scripts/evaluate_l2t_supervised_baselines.py`
- `scripts/run_minxing_l2t_compatibility.py`

Core tests:

- `tests/test_build_unified_toolcalling_dataset.py`
- `tests/test_build_toolcalling_numerical_representation.py`
- `tests/test_build_toolcalling_shift_inventory.py`
- `tests/test_analyze_tau2_shift_uncertainty.py`
- `tests/test_plan_tau2_additional_sampling.py`
- `tests/test_build_tau2_stage1_manifest.py`
- `tests/test_run_tau2_stage1.py`
- `tests/test_ingest_tau2_stage1_results.py`
- `tests/test_build_tau2_stage1_analysis.py`
- `tests/test_build_bfcl_shift_inventory.py`
- `tests/test_analyze_bfcl_shift_uncertainty.py`
- `tests/test_l2t_model_bridge_converters.py`
- `tests/test_evaluate_l2t_supervised_baselines.py`
- `tests/test_run_minxing_l2t_compatibility.py`

Docs and result summaries:

- `docs/toolcalling_shift_project_log.md`
- `docs/toolcalling_shift_inventory.md`
- `docs/tau2_shift_uncertainty.md`
- `docs/tau2_shift_uncertainty_stage1.md`
- `docs/tau2_additional_sampling_plan.md`
- `docs/tau2_stage1_manifest.md`
- `docs/pr_tau2_stage1_shift_analysis.md`
- `docs/bfcl_data_source_audit.md`
- `docs/bfcl_shift_inventory.md`
- `docs/bfcl_shift_uncertainty.md`
- `docs/toolcalling_cross_dataset_findings.md`
- `docs/l2t_multisource_model_bridge.md`
- `docs/l2t_supervised_diagnostic_baselines.md`
- `docs/l2t_minxing_compatibility_study.md`

Processed artifacts and manifests:

- `data/processed/unified_toolcalling_manifest.json`
- `data/processed/toolcalling_numerical_full_manifest.json`
- `data/processed/toolcalling_shift_inventory_summary.json`
- `data/processed/tau2_shift_uncertainty_summary.json`
- `data/processed/tau2_stage1_ingestion_summary.json`
- `data/processed/tau2_shift_uncertainty_stage1_summary.json`
- `data/processed/tau2_shift_stage1_comparison.json`
- `data/processed/bfcl/bfcl_v4_non_live_1240_xy.jsonl`
- `data/processed/bfcl/bfcl_v4_non_live_1240_summary.json`
- `data/processed/bfcl/bfcl_v4_non_live_shift_inventory_summary.json`
- `data/processed/bfcl/bfcl_v4_non_live_shift_uncertainty_summary.json`
- `data/processed/l2t/bfcl/bfcl_v4_non_live_1240_l2t_manifest.json`
- `data/processed/l2t/apibank/apibank_full_l2t_manifest.json`
- `data/processed/l2t/diagnostics/l2t_supervised_diagnostic_summary.json`
- `data/processed/l2t/diagnostics/l2t_supervised_diagnostic_results.csv`
- `data/processed/l2t/diagnostics/l2t_supervised_diagnostic_best_by_view.csv`
- `data/processed/l2t/diagnostics/l2t_supervised_diagnostic_permutation_summary.csv`
- `data/processed/l2t/diagnostics/minxing_compatibility_5ep/minxing_compatibility_summary_5ep.json`

Minxing reference entry point:

- `/Users/xuyida/Research/physics_informed_testing/share_code/experiment/run_baseline.py`

## 13. Validation Status

Current L2T bridge review-fix validation on 2026-07-21:

- `uv run --extra dev pytest tests/test_l2t_model_bridge_converters.py tests/test_evaluate_l2t_supervised_baselines.py tests/test_run_minxing_l2t_compatibility.py -q`
  passed: 18 passed, 2 warnings.
- Warnings were the existing `audioop` deprecation warning from
  `src/tau2/voice/utils/audio_preprocessing.py` and the existing unknown pytest
  config option warning for `asyncio_default_fixture_loop_scope`.
- `uv run --extra dev ruff check scripts/l2t_model_bridge.py scripts/convert_bfcl_to_l2t_pkl.py scripts/convert_apibank_to_l2t_pkl.py scripts/evaluate_l2t_supervised_baselines.py scripts/run_minxing_l2t_compatibility.py tests/test_l2t_model_bridge_converters.py tests/test_evaluate_l2t_supervised_baselines.py tests/test_run_minxing_l2t_compatibility.py`
  passed.
- `git diff --check` passed.
- `git status --ignored --short` shows nested L2T pickles ignored:
  `data/processed/l2t/apibank/apibank_full_l2t.pkl` and
  `data/processed/l2t/bfcl/bfcl_v4_non_live_1240_l2t.pkl`.

Prior Stage 1 validation recorded in this log:

- Targeted tau2 Stage 1 workflow tests passed: 83 passed, 2 warnings.
- Ruff over changed Python source and tests passed.
- Full-repository pytest collection with only `--extra dev` previously failed
  because optional unrelated dependencies were not installed: `a2a`,
  `agentify_tau_bench`, `websockets`, `rank_bm25`, `gymnasium`, and `pyaudio`.
  That was a collection/environment limitation, not a targeted workflow test
  failure.

Repository hygiene checks:

- Current branch has untracked L2T bridge files and generated compact artifacts.
- Ignored local artifacts include `.env`, caches, binary arrays/pickles, raw
  Stage 1 results, simulation outputs, and Python bytecode.
- No commit or push has been performed for the current L2T bridge work.

Outstanding review concerns:

- Generated L2T pickles should remain ignored local artifacts; commit only
  compact manifests, diagnostics, docs, scripts, and tests.
- API-Bank pair/group splitting is addressed for supervised diagnostics, but the
  synthetic-negative construction still limits interpretation.
- Minxing semantic compatibility remains unresolved.

## 14. Current Git State

Repository:

```text
/Users/xuyida/Research/llm-toolcalling-benchmarks/tau2-bench
```

Branch:

```text
feat/l2t-multisource-model-bridge
```

HEAD observed before this documentation edit:

```text
00ddf01 (HEAD -> feat/l2t-multisource-model-bridge, origin/main, origin/HEAD, main) Merge pull request #2 from YidaXu04/feat/bfcl-shift-analysis
```

Tracked/untracked files before this documentation edit:

- No tracked modifications were present.
- Untracked L2T bridge files were present under:
  `data/processed/l2t/`, `docs/l2t_minxing_compatibility_study.md`,
  `docs/l2t_multisource_model_bridge.md`,
  `docs/l2t_supervised_diagnostic_baselines.md`,
  five scripts, and three tests.

Tracked/untracked files after this documentation edit:

- This file is modified:
  `docs/toolcalling_shift_project_log.md`.
- The L2T bridge files remain untracked.
- Nothing has been staged, committed, or pushed in this documentation-only
  consolidation.

Sibling-repository status where relevant:

- `DAMO-ConvAI`: `main...upstream/main [ahead 2]`; do not modify.
- `gorilla`: `main...origin/main` plus untracked BFCL local generated files; do
  not modify.
- `physics_informed_testing`: `main...origin/main` plus untracked compatibility
  result folders; do not modify unless explicitly authorized.

## 15. Immediate Next Step

The immediate next step is an independent Claude review of the complete current
branch:

1. Ask Claude to review the full `feat/l2t-multisource-model-bridge` branch,
   including untracked scripts, tests, docs, manifests, and artifact-inclusion
   choices.
2. Fix critical or major findings with Codex.
3. Rerun targeted validation and `git diff --check`.
4. Stage only the intended inclusion set.
5. Commit locally.
6. Push to the user's fork.
7. Open a PR into the user's own `main`.
8. Squash merge and clean up the branch after review.

Never open a PR to upstream `sierra-research/tau2-bench` for this research
branch unless the user explicitly changes that instruction.

## 16. Open Questions And Risks

- API-Bank grouped/paired splitting: supervised diagnostics now use a grouped
  split by `pair_id`, but any row-level ablation or Minxing compatibility run
  must still be interpreted with pair leakage in mind.
- Permutation-control interpretation: supervised diagnostics now report 10-trial
  deterministic null summaries. These controls address the earlier single-run
  shuffled-label artifact but do not make synthetic API-Bank negatives natural
  LLM failures.
- Semantic validity of S representations: tau2 uses coarse structural event
  sequences, BFCL uses compact candidate-call events, and API-Bank uses fixed
  numerical S rows as Minxing `traj["s"]`.
- Limits of current Minxing implementation: sequence modes threshold
  reconstructed trajectory scores that are not yet semantically aligned with
  tool-calling success.
- Sample size and class imbalance: tau2 remains small; BFCL validation is
  strongly positive-majority; API-Bank is balanced by construction but
  synthetic.
- Dataset label scopes differ: task-level, test-case-level, and API-call-level
  labels should not be conflated.
- Artifact inclusion risk: binary pickles, NumPy archives, raw result folders,
  checkpoints, and large prediction CSVs should remain ignored local artifacts.
  The minimal reproducibility set is scripts, tests, docs, compact manifests,
  and compact diagnostic JSON/CSV outputs.

## 17. Do-Not-Do List

- Do not run long experiments yet.
- Do not modify Minxing core code yet.
- Do not pool tau2, BFCL, and API-Bank as IID rows.
- Do not redefine all `Y` values as 1.
- Do not commit raw result folders, checkpoints, large prediction CSVs, training
  histories, or redundant Minxing result directories.
- Do not modify `DAMO-ConvAI` or `gorilla`.
- Do not submit upstream PRs.
- Do not treat synthetic API-Bank negatives as natural LLM failures.
- Do not claim deployment safety, causal harm, or retraining necessity from the
  current exploratory classifications.

## 18. New-Session Handoff

Copy-paste this section into a new session:

```text
Current status:
- Repo: /Users/xuyida/Research/llm-toolcalling-benchmarks/tau2-bench
- Branch: feat/l2t-multisource-model-bridge
- HEAD before current untracked L2T bridge work: 00ddf01, merged BFCL PR #2.
- The current branch adds an L2T bridge for BFCL and API-Bank, supervised
  diagnostics, a Minxing compatibility runner, compact manifests, and docs.
- Nothing has been committed or pushed for the L2T bridge work.
- Do not modify DAMO-ConvAI, gorilla, or physics_informed_testing unless the
  user explicitly asks.

Exact next action:
- Review the implemented fixes and regenerated compact diagnostics.
- Rerun targeted validation and git diff --check if anything changes.
- Commit, push to YidaXu04/tau2-bench, open PR into the user's own main, then
  squash merge and clean up.
- Never open a PR to upstream sierra-research/tau2-bench.

Files to inspect first:
- docs/toolcalling_shift_project_log.md
- docs/l2t_multisource_model_bridge.md
- docs/l2t_supervised_diagnostic_baselines.md
- docs/l2t_minxing_compatibility_study.md
- scripts/l2t_model_bridge.py
- scripts/convert_bfcl_to_l2t_pkl.py
- scripts/convert_apibank_to_l2t_pkl.py
- scripts/evaluate_l2t_supervised_baselines.py
- scripts/run_minxing_l2t_compatibility.py
- tests/test_l2t_model_bridge_converters.py
- tests/test_evaluate_l2t_supervised_baselines.py
- tests/test_run_minxing_l2t_compatibility.py
- data/processed/l2t/** manifests and compact diagnostic summaries

Commands to check git status and run validation:
- git status --short --branch
- git status --ignored --short
- uv run --extra dev pytest tests/test_l2t_model_bridge_converters.py tests/test_evaluate_l2t_supervised_baselines.py tests/test_run_minxing_l2t_compatibility.py -q
- uv run --extra dev ruff check scripts/l2t_model_bridge.py scripts/convert_bfcl_to_l2t_pkl.py scripts/convert_apibank_to_l2t_pkl.py scripts/evaluate_l2t_supervised_baselines.py scripts/run_minxing_l2t_compatibility.py tests/test_l2t_model_bridge_converters.py tests/test_evaluate_l2t_supervised_baselines.py tests/test_run_minxing_l2t_compatibility.py
- git diff --check
```
