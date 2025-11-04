# Eval: Hyperparameter Experiments

The `hyperparam/` module provides tools for running systematic experiments with different hyperparameters on the tau2 benchmark. It enables comprehensive evaluation across multiple LLMs, domains, and experimental modes.

## Overview

This module allows you to:
- **Run hyperparameter sweeps** across LLMs, domains, and experimental modes
- **Analyze experimental results** with comprehensive statistics and visualizations  
- **View simulation results** interactively
- **Compare model performance** across different configurations

## Quick Start

### Demo Script (Recommended)

The easiest way to test the system is with the included demo script:

```bash
# From project root
src/experiments/hyperparam/demo.sh
```

This runs a small experiment (3 tasks, 2 trials) and shows you the essential commands.

### Running Experiments

```bash
# Basic experiment with multiple LLMs and domains
python -m experiments.hyperparam.cli run-evals \
    --exp-dir my-experiment \
    --llms gpt-4.1-2025-04-14 claude-3-7-sonnet-20250219 \
    --domains retail airline telecom \
    --num-trials 3

# Experiment with specific modes and parameters
python -m experiments.hyperparam.cli run-evals \
    --exp-dir hyperparameter-sweep \
    --llms gpt-4.1-2025-04-14 \
    --domains telecom \
    --modes default oracle-plan no-user \
    --num-trials 5 \
    --max-steps 150 \
    --seed 42
```

### Analyzing Results

```bash
# Analyze completed experiment
python -m experiments.hyperparam.cli analyze-results --exp-dir my-experiment

# View specific simulations interactively
python -m experiments.hyperparam.cli view --dir path/to/simulations --only-failed
```

## CLI Commands

### `run-evals`
Runs evaluation experiments with hyperparameter combinations.

**Required Arguments:**
- `--exp-dir`: Experiment directory name (created under `data/exp/`)
- `--llms`: List of LLM models to test

**Key Optional Arguments:**
- `--domains`: Domains to test (default: retail, airline, telecom, telecom-workflow)
- `--modes`: Experimental modes (default: default, oracle-plan, no-user)
- `--num-trials`: Trials per configuration (default: 4)
- `--num-tasks`: Limit number of tasks per domain (default: all tasks)
- `--max-steps`: Maximum steps per trial (default: 200)
- `--max-concurrency`: Concurrent simulations (default: 100)
- `--seed`: Random seed (default: 300)
- `--llm-user`: User simulator model (default: gpt-4.1-2025-04-14)
- `--agent-llm-args`: JSON string of agent LLM parameters (default: temperature=0.0)
- `--user-llm-args`: JSON string of user LLM parameters (default: temperature=0.0)

### `analyze-results`
Analyzes experimental results and generates comprehensive reports.

**Arguments:**
- `--exp-dir`: Experiment directory to analyze

### `view`
Interactive viewer for simulation results.

**Arguments:**
- `--dir`: Directory containing simulation files
- `--file`: Specific simulation file to view
- `--only-failed`: Show only failed simulations
- `--only-all-failed`: Show only tasks where all trials failed

## Experimental Modes

The module supports several experimental modes:

- **`default`**: Standard agent-user interaction
- **`no-user`**: Agent operates without user simulator (solo mode)
- **`oracle-plan`**: Agent has access to ground truth task plan
- **`no-user-oracle-plan`**: Solo agent with ground truth plan

> **Note**: Oracle plan modes are only supported for telecom domains.

## Supported Domains

- **`retail`**: Customer service scenarios in retail environment
- **`airline`**: Flight booking and customer service tasks
- **`telecom`**: Telecommunications customer support
- **`telecom-workflow`**: Telecom with workflow-based task execution

## LLM Support

The module supports various LLM providers and handles special cases:

- **Reasoning models** (o1-*, o4-*): Automatically sets `reasoning_effort=high`, removes temperature
- **GPT-5 models**: Special handling for reasoning capabilities
- **Standard models**: Full parameter control (temperature, etc.)

## File Organization

Experiment results are saved with descriptive filenames:
```
{llm}_{domain}_{mode}_{user_llm}_{num_trials}trials[_{num_tasks}tasks].json
```

Example:
```
gpt-4.1-2025-04-14_retail_default_gpt-4.1-2025-04-14_4trials.json
claude-3-7-sonnet-20250219_telecom_oracle-plan_gpt-4.1-2025-04-14_3trials_50tasks.json
```

## Analysis Features

The `analyze-results` command provides:

- **Performance metrics** across all configurations
- **Statistical comparisons** between models and modes
- **Visualization plots** for result trends
- **Breakdown analysis** by domain and task type
- **Success/failure pattern analysis**
- **Detailed per-task performance reports**

## Example Workflows

### Comprehensive Model Comparison
```bash
# Run multiple models across all domains
python -m experiments.hyperparam.cli run-evals \
    --exp-dir model-comparison-2025 \
    --llms gpt-4.1-2025-04-14 claude-3-7-sonnet-20250219 o1-2024-12-17 \
    --domains retail airline telecom \
    --num-trials 5

# Analyze results
python -m experiments.hyperparam.cli analyze-results --exp-dir model-comparison-2025
```

### Quick Debug Run
```bash
# Fast test with limited scope
python -m experiments.hyperparam.cli run-evals \
    --exp-dir debug-test \
    --llms gpt-4.1-2025-04-14 \
    --domains retail \
    --modes default \
    --num-trials 1 \
    --num-tasks 5
```

### Hyperparameter Sensitivity
```bash
# Test different temperature settings
python -m experiments.hyperparam.cli run-evals \
    --exp-dir temp-sensitivity \
    --llms gpt-4.1-2025-04-14 \
    --domains telecom \
    --agent-llm-args '{"temperature": 0.7}' \
    --num-trials 3
```

## Development Notes

### Adding New Modes
To add new experimental modes, extend the `RunMode` enum in `run_eval.py` and update the agent selection logic in `make_config()`.

### Custom Analysis
For custom analysis, use `get_simulation_results()` to load experiment data programmatically:

```python
from experiments.hyperparam.run_eval import get_simulation_results
from pathlib import Path

# Load results
exp_dir = Path("data/exp/my-experiment")
results = get_simulation_results(exp_dir)

# Each result is (params_dict, Results_object)
for params, sim_results in results:
    print(f"LLM: {params['llm']}, Domain: {params['domain']}")
    # Analyze sim_results...
```

## Dependencies

The eval module requires:
- Core tau2 dependencies
- `scikit-learn` for parameter grid generation
- `matplotlib`, `pandas`, `numpy` for analysis and visualization
- `scipy` for statistical computations

## Troubleshooting

**Common Issues:**

1. **Experiment directory exists**: The system will prompt before overwriting existing experiments
2. **Invalid mode/domain combinations**: Oracle plan modes only work with telecom domains
3. **Concurrency issues**: Reduce `--max-concurrency` if experiencing rate limits
4. **Memory issues**: Use `--num-tasks` to limit scope for large experiments

For additional help, see the main tau2 documentation or run:
```bash
python -m experiments.hyperparam.cli --help
python -m experiments.hyperparam.cli run-evals --help
```
