# Summary

This PR should present one coherent tau2 L2T and Stage 1 context-shift workflow: unified tool-calling records, numerical X/S representation, candidate shift inventory, tau2-only uncertainty analysis, additional-sampling plan, Stage 1 collection tooling, Stage 1 ingestion, and Stage 1 post-collection comparison.

No commit, push, PR creation, or deletion of local research outputs was performed during this preparation pass.

# Research motivation

The workflow asks whether a tool-calling context shift changes the task outcome rate enough to justify more data collection or adaptation. The analysis keeps tau2 real task-level outcomes separate from API-Bank synthetic API-call correctness labels. It treats Stage 1 as exploratory evidence, not a deployment decision, causal claim, or model-performance benchmark.

# X, S, Y formulation

- `X`: task and deployment context, including domain, expected action counts, tool/write requirements, and other pre-execution context features.
- `S`: tool-calling trajectory representation. For tau2 this is a coarse fixed-length event sequence plus structural features; for API-Bank it is candidate API-call context and arguments.
- `Y`: binary correctness/success label. tau2 uses task-level benchmark reward labels. API-Bank pilot labels are API-call-level correctness labels with synthetic negative corruptions.

Metadata-only fields are excluded from model-facing X or S where they could leak labels or evaluator state: `y`, `label_origin`, `is_synthetic`, `variant`, `corruption_type`, `validation_status`, and `validation_error`.

# Included workflow

The proposed PR includes scripts and tests for:

- building unified tau2/API-Bank wrapper records;
- building numerical representations with `sentence-transformers/all-MiniLM-L6-v2`;
- constructing candidate shift inventories without using `y`;
- estimating tau2-only shift uncertainty with full confidence-interval classification;
- planning a targeted tau2 Stage 1 collection;
- building a dry-run-first Stage 1 manifest and runner;
- ingesting Stage 1 results with the original tau2 conversion/filtering helpers;
- building versioned Stage 1 tau2 outputs and pre/post comparison docs through
  `scripts/build_tau2_stage1_analysis.py`, the canonical Stage 1 merge path.

The four existing shift scripts are changed only by mechanical Ruff/import-spacing blank-line removals:

- `scripts/analyze_shift_groups.py`
- `scripts/build_shift_level_dataset.py`
- `scripts/build_shift_level_summary.py`
- `scripts/convert_tau2_results_to_l2t_pkl.py`

# Data sources and label scopes

- tau2 baseline: 93 retained retail/airline records, labels 0 = 60 and 1 = 33, `label_scope = task_level`.
- tau2 Stage 1: 12 retained retail records, labels 0 = 8 and 1 = 4, selected by X/task characteristics rather than outcomes.
- tau2 merged Stage 1 dataset: 105 records, labels 0 = 68 and 1 = 37.
- API-Bank pilot: 1,016 API-call-level records, labels 0 = 508 and 1 = 508; negative labels are synthetic corruptions and are not described as natural LLM failures.

# Stage 1 collection

Stage 1 selected exactly 12 unused retail tasks from the additional-sampling plan:

- 8 tasks with two or more expected writes;
- 2 tasks with no expected writes;
- 2 low-action one-write tasks.

The manifest records deterministic selection with seed `20260717`, `selection_uses_outcome_labels = false`, and per-task `selection_uses_outcome_label = false`. The runner defaults to dry-run mode and requires `--execute` for real LLM simulations. Stage 1 raw per-task JSON copies and run status logs are treated as local-only artifacts. The manifest's batch tau2 command is illustrative; `scripts/run_tau2_stage1.py` is the executed path and launches one task per subprocess.

# Reproducibility inputs

The proposed PR includes compact result manifests, summaries, and reports. A fresh clone can inspect the reported Stage 1 outputs, but cannot fully regenerate every derived artifact from tracked files alone. Full regeneration also requires:

- local tau2 baseline inputs such as `data/processed/unified_toolcalling_tau2.jsonl` and the source `.pkl` built from gitignored `data/simulations/*` raw logs;
- local Stage 1 raw copies under `data/processed/tau2_stage1_raw/`, which remain excluded from Git because they are raw simulation outputs;
- API-Bank wrapper input from a sibling `DAMO-ConvAI/api-bank` checkout, or an equivalent explicit local path passed to the unified-dataset builder.

Those inputs are outside the proposed Git inclusion set in this PR. The tracked manifests document artifact contents and provenance, not complete fresh-clone reproducibility.

# Main results

The Stage 1 post-collection analysis preserved all six baseline tau2 shift definitions. The main threshold-sensitive result is:

| Shift | Stage 1 delta_y | 95% CI | d=0.05 | d=0.10 | d=0.15 |
|---|---:|---|---|---|---|
| `tau2_zero_or_one_write_to_two_plus_writes` | -0.3271 | [-0.4634, -0.1371] | candidate_harmful | candidate_harmful | inconclusive |

Other tau2 shifts remain inconclusive at all three thresholds. Classifications use the full 95% confidence interval: `candidate_harmful` requires the full interval to lie below `-d`; `candidate_harmless` requires the full interval to lie inside `[-d, +d]`; `candidate_beneficial` requires the full interval to lie above `+d`.

# Validation

Targeted tests:

```text
uv run --extra dev pytest tests/test_build_unified_toolcalling_dataset.py tests/test_build_toolcalling_numerical_representation.py tests/test_build_toolcalling_shift_inventory.py tests/test_analyze_tau2_shift_uncertainty.py tests/test_plan_tau2_additional_sampling.py tests/test_build_tau2_stage1_manifest.py tests/test_run_tau2_stage1.py tests/test_ingest_tau2_stage1_results.py tests/test_build_tau2_stage1_analysis.py -q
```

Result: 84 passed, 2 warnings. The warnings were the existing Python `audioop` deprecation warning from voice utilities and an unknown pytest config option warning for `asyncio_default_fixture_loop_scope`.

Ruff:

```text
uv run --extra dev ruff check scripts/analyze_shift_groups.py scripts/build_shift_level_dataset.py scripts/build_shift_level_summary.py scripts/convert_tau2_results_to_l2t_pkl.py scripts/analyze_tau2_shift_uncertainty.py scripts/build_tau2_stage1_analysis.py scripts/build_tau2_stage1_manifest.py scripts/build_toolcalling_numerical_representation.py scripts/build_toolcalling_shift_inventory.py scripts/build_unified_toolcalling_dataset.py scripts/ingest_tau2_stage1_results.py scripts/plan_tau2_additional_sampling.py scripts/run_tau2_stage1.py tests/test_analyze_tau2_shift_uncertainty.py tests/test_build_tau2_stage1_analysis.py tests/test_build_tau2_stage1_manifest.py tests/test_build_toolcalling_numerical_representation.py tests/test_build_toolcalling_shift_inventory.py tests/test_build_unified_toolcalling_dataset.py tests/test_ingest_tau2_stage1_results.py tests/test_plan_tau2_additional_sampling.py tests/test_run_tau2_stage1.py
```

Result: all checks passed.

Full-repository pytest collection:

```text
uv run --extra dev pytest --collect-only -q
```

Result: collection failed after collecting 895 tests because optional dependencies for unrelated suites are not installed: `a2a`, `agentify_tau_bench`, `websockets`, `rank_bm25`, `gymnasium`, and `pyaudio`. This is separate from the targeted workflow tests.

Scientific and implementation checks verified from artifacts:

- original 93 tau2 records preserved;
- Stage 1 adds exactly 12 retained records;
- merged total is 105;
- final y distribution is 68 zero and 37 one;
- task selection does not use `y`;
- all six shift definitions are preserved;
- API-Bank synthetic negatives are not described as natural LLM failures;
- task-level and API-call-level labels remain distinguished;
- no causal or proven claims appear in the workflow docs reviewed here;
- metadata-only fields do not leak into model-facing X or S;
- baseline outputs were not overwritten by versioned Stage 1 outputs.

# Files included

Proposed Git inclusion set:

- Dependency and repo hygiene: `.gitignore`, `pyproject.toml`, `uv.lock`.
- Source scripts:
  - `scripts/analyze_tau2_shift_uncertainty.py`
  - `scripts/build_tau2_stage1_analysis.py`
  - `scripts/build_tau2_stage1_manifest.py`
  - `scripts/build_toolcalling_numerical_representation.py`
  - `scripts/build_toolcalling_shift_inventory.py`
  - `scripts/build_unified_toolcalling_dataset.py`
  - `scripts/ingest_tau2_stage1_results.py`
  - `scripts/plan_tau2_additional_sampling.py`
  - `scripts/run_tau2_stage1.py`
  - the four mechanical Ruff-only script diffs listed above.
- Tests:
  - `tests/test_analyze_tau2_shift_uncertainty.py`
  - `tests/test_build_tau2_stage1_analysis.py`
  - `tests/test_build_tau2_stage1_manifest.py`
  - `tests/test_build_toolcalling_numerical_representation.py`
  - `tests/test_build_toolcalling_shift_inventory.py`
  - `tests/test_build_unified_toolcalling_dataset.py`
  - `tests/test_ingest_tau2_stage1_results.py`
  - `tests/test_plan_tau2_additional_sampling.py`
  - `tests/test_run_tau2_stage1.py`
- Documentation:
  - `docs/tau2_additional_sampling_plan.md`
  - `docs/tau2_shift_stage1_comparison.md`
  - `docs/tau2_shift_uncertainty.md`
  - `docs/tau2_shift_uncertainty_stage1.md`
  - `docs/tau2_stage1_manifest.md`
  - `docs/toolcalling_shift_inventory.md`
  - `docs/toolcalling_shift_project_log.md`
  - `docs/pr_tau2_stage1_shift_analysis.md`
- Compact result manifests and summaries:
  - `data/processed/tau2_additional_sampling_plan.json`
  - `data/processed/tau2_shift_stage1_comparison.json`
  - `data/processed/tau2_shift_uncertainty.jsonl`
  - `data/processed/tau2_shift_uncertainty_stage1.jsonl`
  - `data/processed/tau2_shift_uncertainty_stage1_summary.json`
  - `data/processed/tau2_shift_uncertainty_summary.json`
  - `data/processed/tau2_stage1_ingestion_summary.json`
  - `data/processed/tau2_stage1_manifest.json`
  - `data/processed/tau2_stage1_retained.jsonl`
  - `data/processed/toolcalling_numerical_full_manifest.json`
  - `data/processed/toolcalling_numerical_pilot_manifest.json`
  - `data/processed/toolcalling_numerical_tau2_stage1_manifest.json`
  - `data/processed/toolcalling_shift_inventory.jsonl`
  - `data/processed/toolcalling_shift_inventory_summary.json`
  - `data/processed/toolcalling_shift_inventory_tau2_stage1.jsonl`
  - `data/processed/toolcalling_shift_inventory_tau2_stage1_summary.json`
  - `data/processed/unified_toolcalling_manifest.json`
  - `data/processed/unified_toolcalling_tau2_stage1_manifest.json`

Audit table:

| File | Category | Proposed Git action |
|---|---|---|
| `.gitignore` | source code | include |
| `pyproject.toml` | dependency file | include |
| `uv.lock` | dependency file | include |
| `scripts/analyze_shift_groups.py` | source code | include; mechanical Ruff-only diff |
| `scripts/build_shift_level_dataset.py` | source code | include; mechanical Ruff-only diff |
| `scripts/build_shift_level_summary.py` | source code | include; mechanical Ruff-only diff |
| `scripts/convert_tau2_results_to_l2t_pkl.py` | source code | include; mechanical Ruff-only diff |
| `scripts/analyze_tau2_shift_uncertainty.py` | source code | include |
| `scripts/build_tau2_stage1_analysis.py` | source code | include |
| `scripts/build_tau2_stage1_manifest.py` | source code | include |
| `scripts/build_toolcalling_numerical_representation.py` | source code | include |
| `scripts/build_toolcalling_shift_inventory.py` | source code | include |
| `scripts/build_unified_toolcalling_dataset.py` | source code | include |
| `scripts/ingest_tau2_stage1_results.py` | source code | include |
| `scripts/plan_tau2_additional_sampling.py` | source code | include |
| `scripts/run_tau2_stage1.py` | source code | include |
| `tests/test_analyze_tau2_shift_uncertainty.py` | test | include |
| `tests/test_build_tau2_stage1_analysis.py` | test | include |
| `tests/test_build_tau2_stage1_manifest.py` | test | include |
| `tests/test_build_toolcalling_numerical_representation.py` | test | include |
| `tests/test_build_toolcalling_shift_inventory.py` | test | include |
| `tests/test_build_unified_toolcalling_dataset.py` | test | include |
| `tests/test_ingest_tau2_stage1_results.py` | test | include |
| `tests/test_plan_tau2_additional_sampling.py` | test | include |
| `tests/test_run_tau2_stage1.py` | test | include |
| `docs/tau2_additional_sampling_plan.md` | documentation | include |
| `docs/tau2_shift_stage1_comparison.md` | documentation | include |
| `docs/tau2_shift_uncertainty.md` | documentation | include |
| `docs/tau2_shift_uncertainty_stage1.md` | documentation | include |
| `docs/tau2_stage1_manifest.md` | documentation | include |
| `docs/toolcalling_shift_inventory.md` | documentation | include |
| `docs/toolcalling_shift_project_log.md` | documentation | include |
| `docs/pr_tau2_stage1_shift_analysis.md` | documentation | include |
| `data/processed/tau2_additional_sampling_plan.json` | small result manifest/summary | include |
| `data/processed/tau2_shift_stage1_comparison.json` | small result manifest/summary | include |
| `data/processed/tau2_shift_uncertainty.jsonl` | small result manifest/summary | include |
| `data/processed/tau2_shift_uncertainty_stage1.jsonl` | small result manifest/summary | include |
| `data/processed/tau2_shift_uncertainty_stage1_summary.json` | small result manifest/summary | include |
| `data/processed/tau2_shift_uncertainty_summary.json` | small result manifest/summary | include |
| `data/processed/tau2_stage1_ingestion_summary.json` | small result manifest/summary | include |
| `data/processed/tau2_stage1_manifest.json` | small result manifest/summary | include |
| `data/processed/tau2_stage1_retained.jsonl` | small result manifest/summary | include |
| `data/processed/toolcalling_numerical_full_manifest.json` | small result manifest/summary | include |
| `data/processed/toolcalling_numerical_pilot_manifest.json` | small result manifest/summary | include |
| `data/processed/toolcalling_numerical_tau2_stage1_manifest.json` | small result manifest/summary | include |
| `data/processed/toolcalling_shift_inventory.jsonl` | small result manifest/summary | include |
| `data/processed/toolcalling_shift_inventory_summary.json` | small result manifest/summary | include |
| `data/processed/toolcalling_shift_inventory_tau2_stage1.jsonl` | small result manifest/summary | include |
| `data/processed/toolcalling_shift_inventory_tau2_stage1_summary.json` | small result manifest/summary | include |
| `data/processed/unified_toolcalling_manifest.json` | small result manifest/summary | include |
| `data/processed/unified_toolcalling_tau2_stage1_manifest.json` | small result manifest/summary | include |
| `data/processed/tau2_l2t_success_retail_airline_50_filtered_20260714.pkl` | generated numerical artifact | exclude |
| `data/processed/tau2_l2t_toy_success_20260710.pkl` | generated numerical artifact | exclude |
| `data/processed/toolcalling_numerical_full.npz` | generated numerical artifact | exclude |
| `data/processed/toolcalling_numerical_pilot.npz` | generated numerical artifact | exclude |
| `data/processed/toolcalling_numerical_tau2_stage1.npz` | generated numerical artifact | exclude |
| `data/processed/toolcalling_numerical_full.jsonl` | generated numerical artifact | exclude |
| `data/processed/toolcalling_numerical_pilot.jsonl` | generated numerical artifact | exclude |
| `data/processed/toolcalling_numerical_tau2_stage1.jsonl` | generated numerical artifact | exclude |
| `data/processed/unified_toolcalling_apibank.jsonl` | generated numerical artifact | exclude |
| `data/processed/unified_toolcalling_tau2.jsonl` | generated numerical artifact | exclude |
| `data/processed/unified_toolcalling_tau2_stage1.jsonl` | generated numerical artifact | exclude |
| `data/processed/tau2_stage1_raw/task_50.json` | raw simulation output | exclude |
| `data/processed/tau2_stage1_raw/task_54.json` | raw simulation output | exclude |
| `data/processed/tau2_stage1_raw/task_55.json` | raw simulation output | exclude |
| `data/processed/tau2_stage1_raw/task_57.json` | raw simulation output | exclude |
| `data/processed/tau2_stage1_raw/task_62.json` | raw simulation output | exclude |
| `data/processed/tau2_stage1_raw/task_64.json` | raw simulation output | exclude |
| `data/processed/tau2_stage1_raw/task_70.json` | raw simulation output | exclude |
| `data/processed/tau2_stage1_raw/task_71.json` | raw simulation output | exclude |
| `data/processed/tau2_stage1_raw/task_72.json` | raw simulation output | exclude |
| `data/processed/tau2_stage1_raw/task_74.json` | raw simulation output | exclude |
| `data/processed/tau2_stage1_raw/task_76.json` | raw simulation output | exclude |
| `data/processed/tau2_stage1_raw/task_81.json` | raw simulation output | exclude |
| `data/processed/tau2_stage1_run_status.json` | runtime status/log | exclude |

# Files intentionally excluded

The following local artifacts should not be staged for this PR:

- `data/processed/*.pkl`
- `data/processed/*.npz`
- `data/processed/tau2_stage1_raw/`
- `data/processed/tau2_stage1_run_status.json`
- `data/processed/toolcalling_numerical_*.jsonl`
- `data/processed/unified_toolcalling_*.jsonl`

Exact `.gitignore` rules added:

```gitignore
# Tau2 Stage 1 local research outputs
data/processed/*.pkl
data/processed/*.npz
data/processed/tau2_stage1_raw/
data/processed/tau2_stage1_run_status.json
data/processed/toolcalling_numerical_*.jsonl
data/processed/unified_toolcalling_*.jsonl
```

An existing local `.DS_Store` and `.env` appear only in ignored-file status through pre-existing ignore rules and are not proposed for tracking. No model weight files appeared in normal changed/untracked status. The local `.venv` contains installed Hugging Face packages but is already ignored and is not proposed for tracking.

# Limitations

- Stage 1 remains small and exploratory.
- Candidate shift definitions reuse records and are not independent.
- The tau2 trajectory representation is coarse and does not preserve full tool names, arguments, outputs, or message content.
- API-Bank synthetic negatives are useful for representation/evaluator development, not for estimating real deployment harmful shifts.
- API-Bank `output_is_success` and `has_exception` are constant in the current
  unified API-Bank pilot records, so they are uninformative dead features rather
  than evidence-bearing covariates.
- Full artifact regeneration needs local-only tau2 raw logs and external
  API-Bank input as described above; tracked manifests are not a complete
  fresh-clone rebuild recipe.
- The dependency lock adds the optional `sentence-transformers` stack under the
  `experiments` extra. That extra may pull heavy Hugging Face/PyTorch/CUDA
  transitive packages when all extras are installed.
- Full-repository pytest collection needs optional extras to collect all unrelated suites.

# Follow-up work

- Decide whether another targeted tau2 collection stage is warranted for threshold-sensitive shifts.
- Add a cost-controlled plan before scaling telecom.
- Consider richer semantic trajectory representations once enough real tau2 outcomes are available.
- Run full repository tests under `uv sync --all-extras` or targeted optional-extra environments before merging if CI requires them.

## Proposed PR title

Add tau2 Stage 1 context-shift analysis workflow

## Proposed commit message

```text
feat: add tau2 stage1 shift analysis workflow
```

## GitHub-ready PR description

```markdown
## Summary

Adds the tau2 L2T and Stage 1 context-shift workflow: unified records, numerical X/S representations, candidate shift inventory, tau2-only uncertainty analysis, targeted Stage 1 collection tooling, Stage 1 ingestion, and versioned post-collection comparison docs.

## Key results

- Preserves the original 93 tau2 records and adds 12 retained Stage 1 retail records.
- Merged tau2 total is 105 records with y distribution 68 zero / 37 one.
- Preserves all six tau2 shift definitions.
- `tau2_zero_or_one_write_to_two_plus_writes` is `candidate_harmful` at d=0.05 and d=0.10, and `inconclusive` at d=0.15.
- Keeps tau2 task-level labels separate from API-Bank synthetic API-call-level labels.

## Validation

- `uv run --extra dev pytest tests/test_build_unified_toolcalling_dataset.py tests/test_build_toolcalling_numerical_representation.py tests/test_build_toolcalling_shift_inventory.py tests/test_analyze_tau2_shift_uncertainty.py tests/test_plan_tau2_additional_sampling.py tests/test_build_tau2_stage1_manifest.py tests/test_run_tau2_stage1.py tests/test_ingest_tau2_stage1_results.py tests/test_build_tau2_stage1_analysis.py -q`
  - 84 passed, 2 known warnings.
- `uv run --extra dev ruff check ...`
  - all checks passed.

## Notes

This is exploratory, small-sample analysis. It does not train a predictive model, does not make causal or proven deployment claims, and does not treat API-Bank synthetic negatives as natural LLM failures.

Full regeneration also requires local tau2 raw simulation inputs and external API-Bank source data; the tracked manifests are compact result records, not a complete fresh-clone rebuild package.
```
