# Bayes@N for repeated text-agent trials

This experiment computes the Bayes@N posterior mean and standard deviation of
an agent's full-success probability over a fixed task set. It analyzes saved
tau2 `Results` without running agents or calling model APIs.

It is experimental analysis, separate from the benchmark's reported reward
and Pass^k metrics. It does not change core scoring or leaderboard submissions.

## Run the analysis

From a source checkout with the core dependencies installed:

```bash
uv sync --frozen
uv run python -m experiments.bayes_at_n \
  --results data/tau2/results/final/gpt-4.1-mini-2025-04-14_telecom_base_gpt-4.1-2025-04-14_4trials.json \
  --output data/analyses/bayes_at_n.json
```

`--results` accepts one saved JSON result or a directory containing `results.json`
and `simulations/`. The command prints JSON; `--output` also saves it to a new
file, creating parent directories if needed. Existing output files are refused.
The source result is read-only. No extra dependencies are needed for this
experiment.

The same analysis is available in Python:

```python
from pathlib import Path

from experiments.bayes_at_n import compute_bayes_at_n
from tau2.data_model.simulation import Results

results = Results.load(Path("data/simulations/my_run.json"))
summary = compute_bayes_at_n(results)
print(summary.posterior_mean, summary.posterior_std)
```

The output contains `posterior_mean`, `posterior_std`, `num_tasks`, `num_trials`,
the original `info` configuration, and each task's `success_count` and trial
outcomes. Every outcome retains its task, trial, simulation ID and seed. Values
use the 0–1 scale. Keep the source result for the complete task definitions and
trajectories.

## Supported experiment

Each row is one declared task in `Results.tasks`. Each column is one of the
`Info.num_trials` independent generations with the same agent, user simulator,
task, environment and grading configuration. Only one text half-duplex result
is analyzed at a time. Files from different configurations are not combined.

An outcome is full success according to tau2's existing `is_successful` rule:
the graded reward must be within `1e-6` of 1, including the endpoints. A finite
partial reward such as 0.5 is a failure of this binary event. This analysis does
not estimate average partial reward or interpret fractional rewards as
Bernoulli probabilities.

Every declared task must have exactly trials `0..N-1`, each with a unique
simulation ID and a recorded seed that is distinct within the task. The
analysis rejects missing tasks or trials, duplicate or unknown identities,
missing/nonfinite grades and infrastructure errors. It does not drop errors,
fill in failures or reduce N to fit an interrupted run. Other termination
reasons use the recorded finite grade, as in the existing success metric.

Task policy and language context must agree across trials and with recorded
`Info.language_tasks` when present. Voice runs, elicitation experiments and
hallucination-feedback retries are unsupported. The recorded configuration
and distinct seeds cannot prove that responses were independently generated:
keep model versions, backend state and grader behavior fixed, and avoid reused
cached responses or outcome-dependent retries. This tool does not authenticate
unrecorded changes in a saved result.

## Posterior and interpretation

This is the binary, uniform-prior variant of
[Bayes@N](https://arxiv.org/abs/2510.04265v4), with no prior observations. The
[Scorio reference](https://github.com/mohsenhariri/scorio/blob/7278f677c30396d99a0454bcbb08f5a9a53c4efb/scorio/eval/bayes.py)
also supports categorical weights and prior observations; those variants are
not included here.

For task i with s_i successes in N trials, the posterior is
`Beta(s_i + 1, N - s_i + 1)`. Its mean is `(s_i + 1)/(N + 2)` and its variance
is `(s_i + 1)(N - s_i + 1)/((N + 2)^2(N + 3))`. With M tasks, the reported mean
is the average of those task means and the reported standard deviation is
`sqrt(sum(task_variances))/M`.

The standard deviation describes posterior uncertainty in the equal-task mean
for this fixed cohort. It is not a standard error over a population of tasks,
a standard deviation of observed rewards or uncertainty in Pass^k. No credible
interval or ranking procedure is added. For outcomes `[[0, 0], [1, 1]]`, the
mean is 0.5 and the standard deviation is `sqrt(3/160)`; pooling the four
outcomes into one Beta posterior would give a different uncertainty.

Tests run offline with:

```bash
uv sync --frozen --extra dev
uv run pytest tests/test_experiments/test_bayes_at_n.py
```
