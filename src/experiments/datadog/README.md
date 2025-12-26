# Datadog LLM Observability Experiment

This directory contains the **Datadog hackathon project** - a self-contained experiment demonstrating LLM observability with tau2-bench-agent.

> **Extraction Note**: This directory is designed for eventual extraction into its own repository (`tau2-datadog-observability`). All datadog-specific code is isolated here to enable clean `git subtree split` extraction.

## Purpose

Demonstrates end-to-end LLM observability for the Google Cloud x Datadog hackathon:
- Gemini LLM traces via ddtrace + LiteLLM
- Custom tau2 evaluation metrics
- Detection rules with Case/Incident management
- Health dashboards

## Quick Start

### Prerequisites

1. **API Keys**: You'll need the following environment variables:
   - `NEBIUS_API_KEY` - Required for mock agent LLM calls
   - `DD_API_KEY` - Datadog API key (for production mode)
   - `DD_APP_KEY` - Datadog Application key (for monitor/dashboard creation)

2. **Python 3.10+** with uv package manager

### Run the Demo

```bash
# Set environment variables
export NEBIUS_API_KEY=your_nebius_key
export DD_API_KEY=your_datadog_api_key
export DD_APP_KEY=your_datadog_app_key

# Run full demo with Datadog integration
./scripts/demo.sh

# Or run a local dry-run demo (no Datadog API calls)
./scripts/demo.sh --dry-run
```

### What the Demo Does

1. **Checks Prerequisites** - Validates required API keys
2. **Creates Datadog Resources** - Sets up monitors, SLOs, and dashboard
3. **Generates Normal Traffic** - Runs 5 evaluations for baseline metrics
4. **Generates Failure Traffic** - Runs 3 failure evaluations to trigger DR-002 (Task Failure Spike) monitor
5. **Emits Metrics** - Sends all metrics to Datadog
6. **Outputs Summary** - Shows dashboard URLs and alert status

### Demo Output

After the demo completes, you can view:
- **Dashboard**: `https://app.datadoghq.com/dashboard/tau2-bench-health`
- **APM Traces**: `https://app.datadoghq.com/apm/traces?query=service:tau2-bench-agent`
- **Metrics**: `https://app.datadoghq.com/metric/explorer?query=tau2.task.reward`
- **Monitors**: `https://app.datadoghq.com/monitors/manage`

## Directory Structure

```
datadog/
├── README.md                    # This file
├── LICENSE                      # Apache-2.0 license
├── configs/
│   ├── monitors.json            # Datadog monitor definitions (5 detection rules)
│   ├── slos.json                # SLO definitions (3 SLOs)
│   ├── dashboards.json          # Dashboard JSON exports
│   └── case_templates.json      # Case management templates
├── scripts/
│   ├── demo.sh                  # End-to-end demo script
│   ├── traffic_generator.py     # Runs tau2 evaluations for telemetry
│   ├── emit_metrics.py          # Post-hoc metrics emission from JSON
│   ├── setup_datadog.py         # Creates monitors/dashboards via API
│   └── tau2_traced.py           # Wrapper to run tau2 with tracing
├── deployment/
│   ├── Dockerfile               # Cloud Run deployment
│   ├── cloudbuild.yaml          # GCP Cloud Build config
│   └── requirements.txt         # Python dependencies
└── tests/
    └── test_traffic_generator.py
```

## Enabling Datadog Tracing

The ddtrace integration is **opt-in** and does not modify tau2 core. Choose one of these methods:

### Option 1: Use the traced wrapper (recommended)

```bash
# From repository root
python -m experiments.datadog.scripts.tau2_traced run --domain mock
```

### Option 2: Use ddtrace-run (ddtrace's built-in wrapper)

```bash
# ddtrace-run auto-patches before importing
ddtrace-run tau2 run --domain mock
```

### Option 3: Environment-based auto-patch

```bash
# Set environment variables and run normally
DD_TRACE_ENABLED=true \
DD_PATCH_MODULES=litellm:true,httpx:true \
tau2 run --domain mock
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DD_API_KEY` | Yes (for agentless) | Datadog API key |
| `DD_APP_KEY` | Yes (for setup) | Datadog Application key |
| `DD_SITE` | No | Datadog site (default: datadoghq.com) |
| `DD_SERVICE` | No | Service name (default: tau2-bench-agent) |
| `DD_ENV` | No | Environment (default: development) |
| `DD_LLMOBS_ENABLED` | No | Enable LLM Observability (default: false) |
| `NEBIUS_API_KEY` | Yes | Nebius API key for mock agent LLM calls |
| `TAU2_DATA_DIR` | No | Data directory (default: ./data) |

## Detection Rules

The project includes 5 detection rules that create actionable Cases/Incidents:

| ID | Name | Trigger Condition | Action |
|----|------|-------------------|--------|
| DR-001 | High Error Rate | error_count / total > 0.2 | Create Case |
| DR-002 | Task Failure Spike (Hero) | avg:tau2.task.reward < 0.7 | Create Case |
| DR-003 | Token Cost Anomaly | token_cost > 2x baseline | Alert |
| DR-004 | Premature Termination | termination:max_errors > 10/hr | Create Incident |
| DR-005 | Latency SLO Breach | p99:duration > 60s | SLO Alert |

## Development

```bash
# Run traffic generator locally
cd src/experiments/datadog
python scripts/traffic_generator.py --dry-run

# Emit metrics from evaluation results
python scripts/emit_metrics.py --all --dry-run

# Run tests
pytest tests/
```

## Extraction to Standalone Repo

When ready to extract for hackathon submission:

```bash
# From tau2-bench-agent root
git subtree split -P src/experiments/datadog -b datadog-standalone

# Create new repo and push
gh repo create tau2-datadog-observability --public
git push origin datadog-standalone:main
```

The extracted repo will need:
1. Its own `pyproject.toml` (copy from `deployment/requirements.txt`)
2. Reference to tau2-bench-agent as a dependency
3. Updated paths in scripts

## Hackathon Submission Checklist

- [ ] Public GitHub repo with Apache-2.0 license
- [ ] README with deployment instructions
- [ ] `configs/` directory with JSON exports (monitors, SLOs, dashboard)
- [ ] Traffic generator script that produces telemetry
- [ ] demo.sh script for end-to-end demo
- [ ] Hosted Cloud Run URL (optional)
- [ ] Datadog organization name
- [ ] Screenshots of dashboard and triggered monitors

### Submission Artifacts

1. **GitHub Repository**: `https://github.com/wuTims/tau2-datadog-observability`
2. **Dashboard JSON**: `configs/dashboards.json`
3. **Monitors JSON**: `configs/monitors.json`
4. **SLOs JSON**: `configs/slos.json`

## Related Specs

- [007-datadog-project spec](../../../specs/007-datadog-project/spec.md)
- [008-gcp-integration spec](../../../specs/008-gcp-integration/spec.md)
