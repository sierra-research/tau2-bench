# tau2-bench Responses API fork

[Original Sierra tau2-bench README](README.upstream.md)

This fork adds an OpenAI Responses API evaluation path for tau2-bench hyperparameter sweeps. It is intended for controlled customer-facing and research-oriented experiments where calls should go directly to the OpenAI API rather than through LiteLLM.

The upstream project remains intact. Use the original README for core tau2-bench concepts, domains, and general installation guidance. Use this README for the fork-specific Responses sweep workflow.

## What This Fork Adds

- Responses API agent/user adapters for text-mode tau2-bench runs.
- `gpt-5.4-mini` as the default Responses sweep model.
- One-factor-at-a-time and full-grid hyperparameter sweeps across:
  - reasoning effort: `none`, `low`, `medium`, `high`, `xhigh`
  - verbosity: `low`, `medium`, `high`
  - hosted web search mode: `off`, `auto`, `required`
  - service tier: `default`, `priority`
- Public-pricing cost estimates for text tokens and reasoning-model hosted web search.
- Resume and cache reuse support so completed task/config combinations can be reused from previous experiment directories.
- A portable HTML report for performance, latency, cost, tokens, coverage, and task-level diagnostics.
- A focused follow-up variant suite for known Responses API comparisons:
  - cached `gpt-5.4-mini` baseline
  - `gpt-5.5` with `reasoning=low`
  - `parallel_tool_calls=true`
  - `parallel_tool_calls=false`
  - WebSocket Responses transport

## Default Run Limits

Responses exploratory sweeps now default to:

- `--max-steps 100`
- `--max-duration-seconds 900`

The 15 minute wall-clock limit is passed through to tau2 as the per-simulation `timeout`. The default is based on observed OFAT diagnostics: very long runs were mostly low-success outliers and were dominating mean latency and cost. These limits are written to the sweep `manifest.json` so reports and downstream analysis can see the run constraints.

Override either value when needed:

```bash
uv run python -m experiments.hyperparam.cli run-responses-sweep \
  --max-steps 150 \
  --max-duration-seconds 1200
```

Disable the wall-clock timeout for a specific run with:

```bash
uv run python -m experiments.hyperparam.cli run-responses-sweep --max-duration-seconds 0
```

`--timeout` is also accepted as an alias for `--max-duration-seconds`.

## Setup

```bash
git clone git@github.com:bjones-oai/tau2-bench-responses-api.git
cd tau2-bench-responses-api
uv sync --extra knowledge --extra dev
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env` or export it in your shell before running sweeps.

The Responses sweep path is:

```bash
uv run python -m experiments.hyperparam.cli run-responses-sweep
```

The legacy upstream sweep path is still present as `run-evals`, but it is not the recommended path for this fork because it uses the original multi-provider/LiteLLM plumbing.

## Smoke Test

Use a small task count first to validate credentials, outputs, and report generation:

```bash
uv run python -m experiments.hyperparam.cli run-responses-sweep \
  --exp-dir responses-smoke \
  --shape ofat \
  --llm gpt-5.4-mini \
  --domains retail \
  --modes default \
  --num-tasks 3 \
  --num-trials 1 \
  --max-concurrency 1 \
  --auto-resume
```

Build the HTML report:

```bash
uv run python -m experiments.hyperparam.cli build-responses-report \
  --exp-dir responses-smoke
```

Open:

```text
data/exp/responses/responses-smoke/index.html
```

## OFAT Sweep

This is the recommended first full exploratory run because it changes one factor at a time around the baseline:

- `reasoning=medium`
- `verbosity=medium`
- `web_search=off`
- `service_tier=default`

Example across the base splits for airline, retail, banking knowledge, and telecom:

```bash
uv run python -m experiments.hyperparam.cli run-responses-sweep \
  --exp-dir ofat-base-banking-telecom-base \
  --shape ofat \
  --llm gpt-5.4-mini \
  --domains airline retail banking_knowledge telecom \
  --modes default \
  --task-split-name base \
  --num-trials 1 \
  --max-concurrency 5 \
  --auto-resume
```

Reuse compatible completed simulations from an earlier experiment:

```bash
uv run python -m experiments.hyperparam.cli run-responses-sweep \
  --exp-dir ofat-base-banking-telecom-base-rerun \
  --shape ofat \
  --llm gpt-5.4-mini \
  --domains airline retail banking_knowledge telecom \
  --modes default \
  --task-split-name base \
  --num-trials 1 \
  --max-concurrency 5 \
  --auto-resume \
  --reuse-from-exp-dirs ofat-base-banking-telecom-base
```

## Known Variant Suite

After the baseline OFAT run exists, use the focused known-variant suite to compare only the additional variants we know how to run today. The baseline config name is intentionally unchanged, so `--reuse-from-exp-dirs` can seed those baseline rows from a prior OFAT run instead of making duplicate API calls.

```bash
uv run python -m experiments.hyperparam.cli run-responses-sweep \
  --exp-dir known-variants-base-banking-telecom-base \
  --known-variant-suite \
  --llm gpt-5.4-mini \
  --domains airline retail banking_knowledge telecom \
  --modes default \
  --task-split-name base \
  --num-trials 1 \
  --max-concurrency 5 \
  --auto-resume \
  --reuse-from-exp-dirs ofat-base-banking-telecom-base
```

The suite still uses the default exploratory run limits unless overridden: `--max-steps 100` and `--max-duration-seconds 900`.

## Outputs

Responses sweep outputs are written under:

```text
data/exp/responses/<exp-dir>/
```

Key files:

- `manifest.json`: planned configs, sweep dimensions, run limits, task split settings, and cache sources.
- `results.csv`: one committed row per completed config.
- `simulations.csv`: one row per completed simulation.
- `raw/*.json`: raw summarized results per config.
- `runs/*/results.json`: tau2 checkpoint outputs used for auto-resume.
- `index.html`: portable dashboard generated by `build-responses-report`.

Large experiment outputs are intentionally not part of the source repo. Share them separately when needed.

## Dashboard

The generated HTML report includes:

- run progress and coverage by domain/config
- OFAT coverage matrix
- pairwise performance, latency, cost, and token plots
- mean/median latency toggle
- success diagnostics by latency and message-step buckets
- task-level tables with expandable tail-latency and token metrics

Regenerate the report after a run:

```bash
uv run python -m experiments.hyperparam.cli build-responses-report \
  --exp-dir <exp-dir>
```

## Notes

- `banking_knowledge` requires the knowledge extra and the relevant retrieval setup from the upstream docs.
- Hosted web search modes only affect the Responses API agent path.
- `priority` service tier is costed using public standard pricing multipliers in the sweep estimator.
- For final defensibility, consider rerunning only timed-out tasks with a higher timeout to measure sensitivity to the default 15 minute cutoff.

[Original Sierra tau2-bench README](README.upstream.md)
