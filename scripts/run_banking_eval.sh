#!/bin/bash
# Evaluate a set of models on tau3 banking_knowledge (bm25 retrieval) and push
# each model's rollouts to its own Supabase table. One head-to-head, same config.
#
# Usage: bash scripts/run_banking_eval.sh <num_tasks>
# Resilient: a model that errors is logged and skipped, the others still run.
set -u
cd "$(dirname "$0")/.."
N="${1:-3}"
MEM=/Users/lilyzhang/Documents/lily-memory
export OPENROUTER_API_KEY=$(grep -E '^(VITE_)?OPENROUTER_API_KEY=' "$MEM/MicDrop/.env.local" | head -1 | sed 's/^[^=]*=//' | tr -d '"'"'"' ')
export OPENAI_API_KEY=$(grep -E '^OPENAI_API_KEY=' "$MEM/GeniusTeam/genius-builder/.env" | head -1 | sed 's/^[^=]*=//' | tr -d '"'"'"' ')
export ANTHROPIC_API_KEY=$(grep -E '^ANTHROPIC_API_KEY=' "$MEM/GeniusTeam/genius-builder/.env" | head -1 | sed 's/^[^=]*=//' | tr -d '"'"'"' ')

# model | table-suffix | agent-llm-args
MODELS=(
  "openrouter/minimax/minimax-m3|m3|{\"temperature\":0,\"max_tokens\":8000}"
  "claude-opus-4-8|opus48|{\"temperature\":0,\"max_tokens\":16000}"
  "gpt-5.5|gpt55|{\"max_tokens\":16000}"
)

for entry in "${MODELS[@]}"; do
  IFS='|' read -r MODEL SUFFIX ARGS <<< "$entry"
  echo "================ $MODEL -> tau2_rollouts_$SUFFIX (n=$N) ================"
  uv run --with psycopg2-binary python scripts/rollouts_supabase.py create --suffix "$SUFFIX"
  uv run tau2 run \
    --domain banking_knowledge --retrieval-config bm25 \
    --agent-llm "$MODEL" --agent-llm-args "$ARGS" \
    --user-llm gpt-4.1 --user-llm-args '{"temperature":0}' \
    --num-tasks "$N" --num-trials 1 --max-concurrency 3 --max-steps 80 --seed 300 \
    --save-to "eval_${SUFFIX}_n${N}" 2>&1 | tail -8 \
    || { echo "!!! $MODEL run FAILED, skipping"; continue; }
  uv run --with psycopg2-binary python scripts/rollouts_supabase.py push \
    --suffix "$SUFFIX" --results "data/simulations/eval_${SUFFIX}_n${N}/results.json" \
    --retrieval-config bm25 --domain banking_knowledge \
    || echo "!!! push FAILED for $SUFFIX"
  uv run --with psycopg2-binary python scripts/rollouts_supabase.py stats --suffix "$SUFFIX"
done
echo "================ DONE ================"
