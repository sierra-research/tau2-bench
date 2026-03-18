# Leaderboard Submission Guide

Submit your agent results to the τ-bench leaderboard at **[taubench.com](https://taubench.com)**.

## Requirements

Your submission must meet these constraints:

1. **Complete domain coverage** — include results for all core domains: `retail`, `airline`, `telecom` (note: `banking_knowledge` submissions are accepted but not required for leaderboard inclusion)
2. **Consistent model configuration** — all trajectory files must use the same agent LLM and user simulator LLM with identical arguments across all domains
3. **One result per domain** — each domain should appear exactly once
4. **All tasks completed** — run evaluation on all tasks within each domain (don't use `--task-ids` or `--num-tasks` filters)

> **Note**: Use the `base` task split (default) when evaluating your agent to ensure you're testing on the complete, standard task set consistent with the original τ-bench methodology.

## Step 1: Run Evaluations

Run your agent on all domains with consistent settings:

```bash
tau2 run --domain retail --agent-llm gpt-4.1 --user-llm gpt-4.1 --num-trials 4 --save-to my_model_retail
tau2 run --domain airline --agent-llm gpt-4.1 --user-llm gpt-4.1 --num-trials 4 --save-to my_model_airline
tau2 run --domain telecom --agent-llm gpt-4.1 --user-llm gpt-4.1 --num-trials 4 --save-to my_model_telecom
```

**Important**: Use identical `--agent-llm`, `--user-llm`, and their arguments across all runs.

## Step 2: Prepare Submission Package

```bash
tau2 submit prepare \
  data/simulations/my_model_retail \
  data/simulations/my_model_airline \
  data/simulations/my_model_telecom \
  --output ./my_submission
```

This will:
- Find `results.json` in each simulation directory
- Verify all trajectory files are valid
- Check that submission requirements are met
- Compute performance metrics (Pass^k rates)
- Prompt for required metadata (model name, organization, contact email)
- Create a structured submission directory with:
  - `submission.json` — metadata and metrics
  - `trajectories/` — your trajectory files

## Step 3: Validate Your Submission

```bash
tau2 submit validate ./my_submission
```

This verifies:
- All required files are present
- Trajectory files are valid
- Domain coverage is complete
- Model configurations are consistent

## Step 4: Submit

1. Review the generated `submission.json` file
2. Follow the submission guidelines in [web/leaderboard/public/submissions/README.md](../web/leaderboard/public/submissions/README.md) to create a Pull Request
3. Keep your `trajectories/` directory for reference

The leaderboard displays your model's Pass^k success rates (k=1,2,3,4) across all domains.

## Additional Options

### Skip verification

```bash
tau2 submit prepare \
  data/simulations/my_model_retail \
  data/simulations/my_model_airline \
  data/simulations/my_model_telecom \
  --output ./my_submission --no-verify
```

### Verify individual trajectory files

```bash
tau2 submit verify-trajs \
  data/simulations/my_model_retail \
  data/simulations/my_model_airline \
  data/simulations/my_model_telecom
```
