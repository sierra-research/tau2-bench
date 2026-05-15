# CORAL Change Log

## 2026-05-15 Tau2 Manifest-Managed CORAL Retail Agent Registration

Status: committed on `bianca/tau2-eval-refactor`.

### Files changed

- `src/tau2/agent/manifest_bootstrap.py`
- `src/tau2/agent/promoted/__init__.py`
- `src/tau2/agent/promoted/bootstrap.py`
- `src/tau2/agent/promoted/llm_agent__retail__coral__051a161/*`
- `src/tau2/agent/promoted/manifest.json`
- `src/tau2/agent/staged/__init__.py`
- `src/tau2/agent/staged/bootstrap.py`
- `src/tau2/agent/staged/llm_agent__retail__coral_candidate__051a161/*`
- `src/tau2/agent/staged/manifest.json`
- `src/tau2/registry.py`
- `tests/test_manifest_bootstrap.py`

### Summary

- Added a manifest-driven bootstrap path for externally managed Tau2 agents.
- Registered staged and promoted manifest agents during default registry initialization.
- Vendored only the evolved CORAL retail candidate/promoted agent snapshot `051a161`.
- Removed the raw seed snapshot `a438b3e` from the staged/promoted manifests and source tree so Tau2 no longer exposes the unevolved seed as a managed candidate or promoted agent.
- Added regression coverage for:
  - manifest loading from both bare-list and object-wrapped JSON formats,
  - successful factory registration and metadata propagation,
  - duplicate-name rejection.

### Verification

- `uv run pytest tests/test_manifest_bootstrap.py tests/test_llm_utils.py tests/test_orchestrator.py tests/test_checkpoint.py tests/test_evaluation_mode.py tests/test_results_format.py tests/test_run.py -q`
  - `102 passed, 1 xfailed`
- `uv run python -m ruff check src/tau2/registry.py src/tau2/agent/manifest_bootstrap.py src/tau2/agent/staged src/tau2/agent/promoted tests/test_manifest_bootstrap.py src/tau2/utils/llm_utils.py src/tau2/orchestrator/orchestrator.py src/tau2/orchestrator/full_duplex_orchestrator.py tests/test_llm_utils.py tests/test_checkpoint.py tests/test_evaluation_mode.py tests/test_results_format.py tests/test_run.py`
  - clean
- `uv run python -c "from tau2.registry import registry; ..."`
  - confirmed `llm_agent__retail__coral_candidate__051a161` and `llm_agent__retail__coral__051a161` register
  - confirmed `llm_agent__retail__coral_candidate__a438b3e` and `llm_agent__retail__coral__a438b3e` do not register

## 2026-05-14 Tau2 External Candidate Public Task View

Status: committed on `bianca/tau2-eval-refactor`.

### Files changed

- `src/tau2/data_model/tasks.py`
- `src/tau2/runner/build.py`
- `tests/test_evaluation_mode.py`

### Summary

- Added `PublicTaskView` plus `make_public_task_view(task)` as the sanitized Tau2 task shape for external candidate factories.
- Limited the external task view to user-facing context only:
  - `id`
  - `user_scenario`
  - `ticket`
- Kept full hidden `Task` data, including `initial_state` and `evaluation_criteria`, internal to Tau2 runtime and evaluation.
- Updated `build_agent(...)` so:
  - `agent_factory_override` receives `PublicTaskView`
  - built-in and registry-resolved Tau2 agents continue to receive the full `Task`
- Added regression coverage to lock the boundary in place for external vs internal agent construction.

### Verification

- `uv run python -m ruff check src/tau2/data_model/tasks.py src/tau2/runner/build.py tests/test_evaluation_mode.py`
  - clean
- `uv run python -m pytest tests/test_evaluation_mode.py -q`
  - `8 passed`
- `uv run python -m pytest tests --ignore=tests/test_gym --ignore=tests/test_voice --ignore=tests/test_streaming -q`
  - `681 passed, 44 skipped, 1 xfailed`

### Regression validation

- Reran text benchmark regressions for `airline` and `retail` on both models.
- Results were recorded in `/Users/bj/research/self-evolving/agent_evals/results-14-05-2026-airline-retail-rerun.md`.
- No structural abnormalities were observed:
  - all four runs completed their full scored task sets,
  - no infrastructure-error exclusions,
  - no max-step terminations.
- The only notable item was run-to-run variance in scores, especially on `airline`, plus expected LiteLLM warning noise for the self-hosted GPT-OSS endpoint.

## 2026-05-14 Tau2 Responses Cost Accounting Fix

Status: committed on `bianca/tau2-eval-refactor`.

### Files changed

- `src/tau2/orchestrator/full_duplex_orchestrator.py`
- `src/tau2/orchestrator/orchestrator.py`
- `src/tau2/utils/llm_utils.py`
- `tests/test_llm_utils.py`

### Summary

- Fixed Responses-mode agent cost accounting so assistant messages no longer hardcode `cost=0.0`.
- Added a shared LiteLLM cost helper for completions and Responses calls that:
  - normalizes fine-tuned model names before pricing,
  - supports `call_type="responses"`,
  - returns `None` when pricing is unavailable instead of incorrectly reporting zero cost.
- Added optional custom token pricing passthrough via `input_cost_per_token` and `output_cost_per_token` for unmapped models such as self-hosted GPT-OSS endpoints.
- Updated aggregate cost computation to track agent and user totals independently, so unknown agent cost no longer wipes out known user-simulator cost.
- Updated orchestrator finalization to use the new per-side cost aggregation.
- Added regression coverage for:
  - Responses cost calculation on priced OpenAI models,
  - custom pricing passthrough for unmapped Responses models,
  - preserving known user cost when agent cost is unknown.

### Verification

- `uv run pytest tests/test_llm_utils.py -q`
  - `11 passed`
- `uv run pytest tests/test_orchestrator.py -q`
  - `13 passed`
- `uv run python -m ruff check src/tau2/utils/llm_utils.py src/tau2/orchestrator/orchestrator.py src/tau2/orchestrator/full_duplex_orchestrator.py tests/test_llm_utils.py`
  - clean
- `uv run pytest tests -q --ignore=tests/test_gym --ignore=tests/test_streaming/test_discrete_time_audio_native_agent.py --ignore=tests/test_streaming/test_voice_streaming_user_simulator.py --ignore=tests/test_voice`
  - `793 passed, 44 skipped, 2 xfailed, 1 xpassed, 2 failed`
  - remaining failures are audio-native streaming tests in `tests/test_streaming/test_run_streaming.py`, blocked by the missing optional `websockets` dependency in this environment

## 2026-05-14 Tau2 Evaluation-Only Execution Refactor

Status: committed on `bianca/tau2-eval-refactor`.

### Files changed

- `src/tau2/api_service/simulation_service.py`
- `src/tau2/cli.py`
- `src/tau2/data_model/evaluation.py`
- `src/tau2/data_model/simulation.py`
- `src/tau2/evaluator/evaluator.py`
- `src/tau2/evaluator/evaluator_action.py`
- `src/tau2/evaluator/evaluator_env.py`
- `src/tau2/run.py`
- `src/tau2/runner/__init__.py`
- `src/tau2/runner/batch.py`
- `src/tau2/runner/build.py`
- `src/tau2/runner/checkpoint.py`
- `src/tau2/runner/progress.py`
- `src/tau2/runner/simulation.py`
- `tests/test_checkpoint.py`
- `tests/test_evaluation_mode.py`
- `tests/test_results_format.py`
- `tests/test_run.py`

### Summary

- Added a new reward-free evaluation model layer in `src/tau2/data_model/evaluation.py`:
  - `DBEvaluation`
  - `EnvAssertionEvaluation`
  - `ActionEvaluation`
  - `CommunicateEvaluation`
  - `NLAssertionEvaluation`
  - `CheckResult`
  - `EvaluationReport`
  - `EvaluationOutcome`
  - `EvaluatedSimulation`
  - `EvaluationIndexEntry`
  - `EvaluatedResults`
- Added a generic `PostEvaluationMode` to Tau2 run config with benchmark mode preserved as the default.
- Split evaluator behavior into:
  - `evaluate_to_report(...)`
  - `compute_evaluation_outcome(...)`
  - existing benchmark-compatible `evaluate_simulation(...)`
- Implemented two score policies:
  - `evaluation_mean_v1` for evaluation-only callers
  - `tau2_reward_compatible` for benchmark compatibility
- Added external agent factory override threading through Tau2 build helpers so external callers can supply a candidate factory without modifying Tau2's built-in registry.
- Added evaluation-only runner entrypoints:
  - `run_simulation_evaluated(...)`
  - `run_single_task_evaluated(...)`
  - `run_tasks_evaluated(...)`
  - `run_domain_evaluated(...)`
- Added Tau2-owned evaluation-only checkpointing, resume, and results persistence using `EvaluatedResults`.
- Generalized progress/retry utilities so Tau2 can display reward in benchmark mode and score in evaluation-only mode.
- Kept the benchmark path intact:
  - benchmark wrappers remain reward/metric-producing by default
  - benchmark wrappers reject evaluation-only configs instead of silently changing behavior
- Added coverage for the new evaluation-only mode and updated result/checkpoint/run tests accordingly.

### Verification

- `uv run python -m pytest tests --ignore=tests/test_gym --ignore=tests/test_voice --ignore=tests/test_streaming -q`
  - `675 passed, 44 skipped, 1 xfailed`

### Regression validation

- Reran the text benchmark regressions after the refactor.
- Results were recorded in `/Users/bj/research/self-evolving/agent_evals/results-14-05-2026.md`.
- Standard reward-based benchmark outputs remained in place for the benchmark execution path.

### CORAL planning impact

- Tau2 now exposes a generic evaluation-only execution path without introducing Tau2-side CORAL-specific concepts.
- CORAL can consume `EvaluationReport` and `EvaluationOutcome` through Tau2 runner helpers while leaving reward reconstruction and benchmark metrics unused.
- External candidate agents can be injected through Tau2 build overrides without falling back to the built-in agent registry.

## 2026-05-13 Tau2 LiteLLM Responses Transport

Status: committed on `bianca/tau2-eval-refactor` as `172849c`.

### Files changed

- `src/tau2/utils/llm_utils.py`
- `tests/test_llm_utils.py`

### Summary

- Added a dedicated LiteLLM Responses path in `generate()` for `api_mode="responses"` and `use_responses_api=True`.
- Implemented `_responses_request()` around `litellm.responses(...)` instead of raw HTTP for Tau2 Responses-mode calls.
- Added Responses-specific helpers for:
  - model normalization for self-hosted GPT-OSS endpoints,
  - system/message/tool conversion,
  - tool schema conversion,
  - `previous_response_id` anchoring and fallback,
  - empty-response retry handling,
  - usage extraction and LLM-call logging,
  - GPT-OSS `tool_choice="required"` fallback to `auto`.
- Preserved the existing completions path for non-Responses models.
- Added focused regression coverage in `tests/test_llm_utils.py` for:
  - plain text Responses replies,
  - tool-call turns and follow-up turns,
  - empty-turn retry recovery,
  - generic OpenAI Responses use beyond GPT-OSS,
  - self-hosted OpenAI-compatible base handling,
  - GPT-OSS `tool_choice` normalization.

### Review summary

- No blocking issues found in this diff.
- Removed the unused `_get_responses_endpoint()` helper while reviewing the change.

### Verification

- `uv run pytest tests/test_llm_utils.py -q`
  - `8 passed`
- `uv run python -m ruff check src/tau2/utils/llm_utils.py tests/test_llm_utils.py`
  - clean

### CORAL planning impact

- The Tau2-side GPT-OSS transport rewrite to `litellm.responses(...)` is already implemented in local Tau2 changes.
- The CORAL Phase 1 plan should treat this as an existing Tau2 dependency, not a future implementation item.
