#!/usr/bin/env bash
# banking_knowledge eval driver: golden_retrieval + a "real retrieval" config,
# in both TEXT and VOICE modes. Idempotent (--auto-resume). Safe to re-run.
#
# NOTE: tau2 auto-prefixes --save-to with data/simulations/, so pass bare paths.
# Override the real-retrieval config via REAL_CFG env (fallback if terminal_use fails):
#   REAL_CFG=openai_embeddings_reranker_grep  or  REAL_CFG=bm25_grep
set -uo pipefail
cd "$(dirname "$0")/.."

NUM_TASKS="${NUM_TASKS:-25}"
SEED="${SEED:-42}"
REAL_CFG="${REAL_CFG:-terminal_use}"
VOICE_CONC="${VOICE_CONC:-6}"
TEXT_CONC="${TEXT_CONC:-8}"

run() {
  local mode="$1" cfg="$2"
  local root extra
  if [ "$mode" = voice ]; then
    root="banking_voice"
    extra="--audio-native --audio-native-provider openai --audio-native-model gpt-realtime-2 --max-concurrency $VOICE_CONC --max-steps-seconds 600 --timeout 900"
  else
    root="banking_text"
    extra="--max-concurrency $TEXT_CONC"
  fi
  echo "=== [$(date +%H:%M:%S)] START mode=$mode cfg=$cfg tasks=$NUM_TASKS ==="
  TAU2_FORCE_LLM_COMMUNICATE_JUDGE=1 uv run tau2 run \
    --domain banking_knowledge \
    --retrieval-config "$cfg" \
    --num-tasks "$NUM_TASKS" --num-trials 1 --seed "$SEED" \
    $extra \
    --save-to "$root/$cfg" \
    --verbose-logs --auto-resume
  echo "=== [$(date +%H:%M:%S)] DONE  mode=$mode cfg=$cfg exit=$? ==="
}

# Text baseline first (fast/cheap), then voice.
run text  golden_retrieval
run text  "$REAL_CFG"
run voice golden_retrieval
run voice "$REAL_CFG"
echo "=== ALL DONE [$(date +%H:%M:%S)] ==="
