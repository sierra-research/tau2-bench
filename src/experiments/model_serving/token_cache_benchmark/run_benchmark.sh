#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
ENV_FILE="${ENV_FILE:-$REPO_ROOT/.env}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

TOKENIZER_PATH="${TOKENIZER_PATH:-}"
MODEL_NAME="${MODEL_NAME:-model}"
API_KEY_ENV="${API_KEY_ENV:-BENCHMARK_API_KEY}"
METRICS_URL="${METRICS_URL:-}"
NETWORK_LABEL="${NETWORK_LABEL:-unspecified}"
REQUEST_EXTRA_JSON="${REQUEST_EXTRA_JSON:-}"

if [[ -z "$TOKENIZER_PATH" ]]; then
  echo "ERROR: set TOKENIZER_PATH to a tokenizer matching the served model." >&2
  exit 2
fi

if [[ -z "${BASE_URL:-}" ]]; then
  PORT="${PORT:-9019}"
  BASE_URL="http://127.0.0.1:${PORT}"
fi

SYSTEM_CACHE_MODES="${SYSTEM_CACHE_MODES:-mixed_system}"
CONCURRENCY="${CONCURRENCY:-4,8,12,16}"
SYSTEM_PROMPT_TOKENS="${SYSTEM_PROMPT_TOKENS:-4500}"
PROMPT_TOKEN_BUCKETS="${PROMPT_TOKEN_BUCKETS:-4000,6000,8000,10000,12288,16384}"
MIN_SAMPLES_PER_BUCKET="${MIN_SAMPLES_PER_BUCKET:-128}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-32256}"
MAX_TURNS_PER_USER="${MAX_TURNS_PER_USER:-80}"
CONVERSATION_POOL_MULTIPLIER="${CONVERSATION_POOL_MULTIPLIER:-7}"
SHARED_SYSTEM_RATIO="${SHARED_SYSTEM_RATIO:-0.8}"
PROMPT_WARMUP="${PROMPT_WARMUP:-1}"
EXCLUDE_FIRST_TURNS="${EXCLUDE_FIRST_TURNS:-3}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-900}"
TEMPERATURE="${TEMPERATURE:-0.7}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-64}"
MAX_CONSECUTIVE_FAILED_WAVES="${MAX_CONSECUTIVE_FAILED_WAVES:-3}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/data/exp/token_cache_benchmark/$RUN_ID}"
OUTPUT_FILE="${OUTPUT_FILE:-results.json}"
mkdir -p "$OUTPUT_DIR"

ARGS=(
  --base-url "$BASE_URL"
  --model-name "$MODEL_NAME"
  --tokenizer-path "$TOKENIZER_PATH"
  --api-key-env "$API_KEY_ENV"
  --network-label "$NETWORK_LABEL"
  --system-prompt-tokens "$SYSTEM_PROMPT_TOKENS"
  --system-cache-modes "$SYSTEM_CACHE_MODES"
  --concurrency "$CONCURRENCY"
  --prompt-token-buckets "$PROMPT_TOKEN_BUCKETS"
  --min-samples-per-bucket "$MIN_SAMPLES_PER_BUCKET"
  --max-prompt-tokens "$MAX_PROMPT_TOKENS"
  --max-turns-per-user "$MAX_TURNS_PER_USER"
  --conversation-pool-multiplier "$CONVERSATION_POOL_MULTIPLIER"
  --shared-system-ratio "$SHARED_SYSTEM_RATIO"
  --prompt-warmup "$PROMPT_WARMUP"
  --exclude-first-turns "$EXCLUDE_FIRST_TURNS"
  --timeout "$REQUEST_TIMEOUT"
  --temperature "$TEMPERATURE"
  --max-output-tokens "$MAX_OUTPUT_TOKENS"
  --max-consecutive-failed-waves "$MAX_CONSECUTIVE_FAILED_WAVES"
  --output "$OUTPUT_DIR/$OUTPUT_FILE"
)

if [[ -n "$METRICS_URL" ]]; then
  ARGS+=(--metrics-url "$METRICS_URL")
fi

if [[ "${DISABLE_METRICS:-0}" == "1" ]]; then
  ARGS+=(--disable-metrics)
fi

if [[ -n "$REQUEST_EXTRA_JSON" ]]; then
  ARGS+=(--request-extra-json "$REQUEST_EXTRA_JSON")
fi

if [[ "${SHOW_FIRST_TURN_METRICS:-0}" == "1" ]]; then
  ARGS+=(--show-first-turn-metrics)
fi

if [[ "${TRUST_REMOTE_CODE:-0}" == "1" ]]; then
  ARGS+=(--trust-remote-code)
fi

echo "Benchmark endpoint: configured (redacted; see report for sanitized URL)"
echo "Tokenizer: $TOKENIZER_PATH"
echo "Network label: $NETWORK_LABEL"
echo "Report: $OUTPUT_DIR/$OUTPUT_FILE"
if [[ "$NETWORK_LABEL" == "unspecified" ]]; then
  echo "WARNING: set NETWORK_LABEL=direct or company-vpn before comparing latency." >&2
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  exec "$PYTHON_BIN" "$SCRIPT_DIR/benchmark_token_cache_buckets.py" "${ARGS[@]}" "$@"
fi

if command -v uv >/dev/null 2>&1; then
  exec uv run --project "$SCRIPT_DIR" python \
    "$SCRIPT_DIR/benchmark_token_cache_buckets.py" "${ARGS[@]}" "$@"
fi

exec python3 "$SCRIPT_DIR/benchmark_token_cache_buckets.py" "${ARGS[@]}" "$@"
