#!/bin/bash
# τ²-Adv Bench Evaluation Runner
# Run this script to start the evaluation with live output

cd "$(dirname "$0")"

export OPENROUTER_API_KEY='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'

echo "=========================================="
echo "  τ²-Adv Bench Evaluation"
echo "=========================================="
echo "Starting at: $(date)"
echo ""

# Run with unbuffered output so you see results in real-time
python -u run_openrouter_eval.py --verbose 2>&1 | tee results/openrouter_eval/eval_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "=========================================="
echo "Evaluation completed at: $(date)"
echo "=========================================="
