# CORAL Change Log

## 2026-05-13 Local Tau2-Bench Changes

Status: local uncommitted changes on `main`.

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
