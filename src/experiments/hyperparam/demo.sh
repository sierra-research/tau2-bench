#!/bin/bash

# Simple demo for experiments.hyperparam module
# Shows the basic commands to run hyperparameter experiments

set -e

# Navigate to src directory
cd "$(dirname "$0")/../../.."
if [ -d "src" ]; then
    cd src
else
    echo "Error: Run this from the tau2 project root"
    exit 1
fi

EXP_NAME="demo-$(date +%Y%m%d-%H%M%S)"

echo "=== tau2 Responses Sweep Demo ==="
echo ""
echo "Running experiment: $EXP_NAME"
echo "This will run a small Responses API smoke sweep with gpt-5.4-mini."
echo ""

# Run experiment
echo "# Step 1: Run experiment"
echo "python -m experiments.hyperparam.cli run-responses-sweep \\"
echo "    --exp-dir $EXP_NAME \\"
echo "    --shape ofat \\"
echo "    --llm gpt-5.4-mini \\"
echo "    --domains retail \\"
echo "    --modes default \\"
echo "    --num-tasks 3 \\"
echo "    --num-trials 1 \\"
echo "    --max-concurrency 1 \\"
echo "    --auto-resume"
echo ""

python -m experiments.hyperparam.cli run-responses-sweep \
    --exp-dir "$EXP_NAME" \
    --shape ofat \
    --llm gpt-5.4-mini \
    --domains retail \
    --modes default \
    --num-tasks 3 \
    --num-trials 1 \
    --max-concurrency 1 \
    --auto-resume

echo ""
echo "# Step 2: Inspect summary artifacts"
echo "open ../data/exp/responses/$EXP_NAME/results.csv"
echo ""

echo "=== Demo Complete ==="
echo "Results saved in: data/exp/responses/$EXP_NAME/"
echo ""
echo "Try these next:"
echo "  # Full grid on one domain:"
echo "  python -m experiments.hyperparam.cli run-responses-sweep --exp-dir grid-retail \\"
echo "      --shape grid --llm gpt-5.4-mini --domains retail --num-trials 1 --max-concurrency 1 --auto-resume"
echo ""
echo "  # Targeted smoke test for one task:"
echo "  python -m experiments.hyperparam.cli run-responses-sweep --exp-dir one-task \\"
echo "      --shape ofat --llm gpt-5.4-mini --domains retail --num-trials 1 --task-ids 0 --max-concurrency 1 --auto-resume"
