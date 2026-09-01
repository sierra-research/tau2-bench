#!/usr/bin/env bash
# Прогон banking_ru с живым DEBUG-логом в файл.
#
#   scripts/banking_ru/run.sh "bank_hard_02" [max_steps] [fast|pin] [лог]
#
# fast (по умолчанию) — OpenRouter выбирает провайдера по скорости
#   (sort=throughput, фолбэки разрешены): для отладочных прогонов.
# pin — закреплённый DigitalOcean без фолбэков: для калибровочных прогонов,
#   где важна сопоставимость (квантование у провайдеров разное).
#
# Живой прогресс:  grep -oE "Step [0-9]+" <лог> | tail -1
set -euo pipefail
cd "$(dirname "$0")/../.."
TASKS=${1:?"укажи task-ids через пробел"}
MAX_STEPS=${2:-50}
MODE=${3:-fast}
LOG=${4:-data/simulations/run_$(date +%Y%m%d_%H%M%S).log}
if [ "$MODE" = "pin" ]; then
  PROVIDER='{"only":["digitalocean"],"allow_fallbacks":false}'
else
  PROVIDER='{"sort":"throughput","allow_fallbacks":true}'
fi
ARGS='{"temperature":0.0,"top_p":1.0,"max_tokens":2048,"seed":42,"timeout":90,"input_cost_per_token":6.8e-8,"output_cost_per_token":1.68e-7,"extra_body":{"provider":'"$PROVIDER"'}}'
echo "лог: $LOG (режим $MODE, max-steps $MAX_STEPS)"
PYTHONUNBUFFERED=1 uv run tau2 run --domain banking_ru \
  --task-ids $TASKS --num-trials 1 \
  --max-steps "$MAX_STEPS" --max-errors 5 --max-concurrency 8 \
  --log-level DEBUG \
  --agent-llm openrouter/deepseek/deepseek-v4-flash --agent-llm-args "$ARGS" \
  --user-llm openrouter/deepseek/deepseek-v4-flash --user-llm-args "$ARGS" \
  > "$LOG" 2>&1
grep -E "Reward:|Termination" "$LOG" | tail -6
