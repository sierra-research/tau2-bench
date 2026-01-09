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

echo "=== tau2 Hyperparameter Experiments Demo ==="
echo ""
echo "Running experiment: $EXP_NAME"
echo "This will take ~2-3 minutes and cost ~\$0.10"
echo ""

# Run experiment
echo "# Step 1: Run experiment"
echo "python -m experiments.hyperparam.cli run-evals \\"
echo "    --exp-dir $EXP_NAME \\"
echo "    --llms gpt-4o-mini \\"
echo "    --domains retail \\"
echo "    --modes default \\"
echo "    --num-tasks 3 \\"
echo "    --num-trials 2 \\"
echo "    --max-steps 50"
echo ""

python -m experiments.hyperparam.cli run-evals \
    --exp-dir "$EXP_NAME" \
    --llms gpt-4o-mini \
    --domains retail \
    --modes default \
    --num-tasks 3 \
    --num-trials 2 \
    --max-steps 50

echo ""
echo "# Step 2: Analyze results (optional - already done automatically)"
echo "python -m experiments.hyperparam.cli analyze-results --exp-dir $EXP_NAME"
echo ""

echo "# Step 3: View results interactively (optional)"
echo "python -m experiments.hyperparam.cli view --dir ../data/exp/$EXP_NAME"
echo ""

echo "=== Demo Complete ==="
echo "Results saved in: data/exp/$EXP_NAME/"
echo ""
echo "Try these next:"
echo "  # Multiple models:"
echo "  python -m experiments.hyperparam.cli run-evals --exp-dir multi-model \\"
echo "      --llms gpt-4o-mini claude-3-haiku-20240307 --domains retail"
echo ""
echo "  # Different domains:"
echo "  python -m experiments.hyperparam.cli run-evals --exp-dir multi-domain \\"
echo "      --llms gpt-4o-mini --domains retail airline telecom"
echo ""
echo "  # Different modes:"
echo "  python -m experiments.hyperparam.cli run-evals --exp-dir multi-mode \\"
echo "      --llms gpt-4o-mini --domains telecom --modes default no-user oracle-plan"
