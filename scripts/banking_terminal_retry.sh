#!/usr/bin/env bash
# Retry ONLY the terminal_use phases at low concurrency (sandbox + realtime are
# fragile under load, esp. with a competing realtime run). --auto-resume keeps
# completed sims and auto-retries infrastructure_error sims (checkpoint.py:163).
set -uo pipefail
cd "$(dirname "$0")/.."

CONC="${CONC:-2}"

run() {
  local mode="$1" root extra
  if [ "$mode" = voice ]; then
    root="banking_voice"
    extra="--audio-native --audio-native-provider openai --audio-native-model gpt-realtime-2 --max-concurrency $CONC --max-steps-seconds 600 --timeout 900"
  else
    root="banking_text"
    extra="--max-concurrency $CONC"
  fi
  echo "=== [$(date +%H:%M:%S)] START $mode terminal_use (conc=$CONC) ==="
  TAU2_FORCE_LLM_COMMUNICATE_JUDGE=1 uv run tau2 run \
    --domain banking_knowledge --retrieval-config terminal_use \
    --num-tasks 25 --num-trials 1 --seed 42 \
    $extra \
    --save-to "$root/terminal_use" --verbose-logs --auto-resume
  echo "=== [$(date +%H:%M:%S)] DONE  $mode terminal_use exit=$? ==="
}

run text    # fast: recovers the 17 infra-error sims
run voice    # slow: runs the remaining 14 tasks
echo "=== RETRY ALL DONE [$(date +%H:%M:%S)] ==="
