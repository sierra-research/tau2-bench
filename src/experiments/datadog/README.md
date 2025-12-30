# Datadog LLM Observability Experiment

This directory contains the **Datadog hackathon project** - a self-contained experiment demonstrating LLM observability with tau2-bench-agent.

## Purpose

Demonstrates end-to-end LLM observability for the Google Cloud x Datadog hackathon:
- Gemini LLM traces via ddtrace + LiteLLM
- Custom tau2 evaluation metrics
- Detection rules with Case/Incident management
- Health dashboards

## Prerequisites

1. **API Keys**:
   - `DD_API_KEY` - Datadog API key
   - `DD_APP_KEY` - Datadog Application key
   - `GEMINI_API_KEY` - Required for local mode

2. **Python 3.10+** with uv package manager

## One-Time Setup

Create Datadog monitors, SLOs, and dashboards:

```bash
export DD_API_KEY=your_datadog_api_key
export DD_APP_KEY=your_datadog_app_key

uv run python -m experiments.datadog.scripts.setup_datadog
```

---

## GCP Demo

Uses deployed agents on Google Cloud Run. No local server setup required.

### Deployed Endpoints

| Service | URL |
|---------|-----|
| tau2_agent | https://tau2-agent-676371821546.us-west2.run.app |
| simple_gemini_agent | https://simple-gemini-agent-4twyiz3sqq-wl.a.run.app |
| kimi_litellm_agent | https://kimi-litellm-agent-4twyiz3sqq-wl.a.run.app |

### Sample A2A Queries

**Health check:**
```bash
curl https://tau2-agent-676371821546.us-west2.run.app/a2a/tau2_agent/.well-known/agent-card.json
```

**Run evaluation:**
```bash
curl -X POST https://tau2-agent-676371821546.us-west2.run.app/a2a/tau2_agent \
  -H "Content-Type: application/json" \
  -H "X-User-LLM-Model: gemini/gemini-2.0-flash" \
  -H "X-User-LLM-API-Key: $GEMINI_API_KEY" \
  -d '{
    "jsonrpc": "2.0",
    "method": "message/stream",
    "params": {
      "message": {
        "messageId": "demo-001",
        "role": "user",
        "parts": [{
          "text": "Run an evaluation on the airline domain for agent at https://simple-gemini-agent-4twyiz3sqq-wl.a.run.app/a2a/simple_gemini_agent. Use 1 tasks and 1 trial(s)."
        }]
      }
    },
    "id": "1"
  }'
```

### Traffic Generation

Generate evaluation traffic across different domains:

```bash
# Airline domain - 2 tasks, 1 trial, 2 evaluations
uv run python -m experiments.datadog.scripts.traffic_generator \
  --domain airline --num-tasks 2 --num-trials 1 --count 2

# Retail domain - 1 task, 2 trials, 1 evaluation
uv run python -m experiments.datadog.scripts.traffic_generator \
  --domain retail --num-tasks 1 --num-trials 2 --count 1

# Telecom domain - 3 tasks, 1 trial, 2 evaluations
uv run python -m experiments.datadog.scripts.traffic_generator \
  --domain telecom --num-tasks 3 --num-trials 1 --count 2
```

**Trigger failure monitors (DR-002, DR-006):**
```bash
uv run python -m experiments.datadog.scripts.traffic_generator \
  --mode failure --domain airline --count 3
```

**Use multiple mock agents:**
```bash
uv run python -m experiments.datadog.scripts.traffic_generator \
  --mock-urls https://simple-gemini-agent-4twyiz3sqq-wl.a.run.app \
              https://kimi-litellm-agent-4twyiz3sqq-wl.a.run.app \
  --domain airline --count 4
```

---

## Local Demo

Starts tau2_agent and mock agent servers locally on separate ports.

### Setup

```bash
export GEMINI_API_KEY=your_gemini_key
export DD_API_KEY=your_datadog_api_key
```

### Traffic Generation

```bash
# Airline domain
uv run python -m experiments.datadog.scripts.traffic_generator \
  --local --domain airline --num-tasks 2 --num-trials 1 --count 2

# Retail domain
uv run python -m experiments.datadog.scripts.traffic_generator \
  --local --domain retail --num-tasks 1 --num-trials 2 --count 1

# Telecom domain
uv run python -m experiments.datadog.scripts.traffic_generator \
  --local --domain telecom --num-tasks 3 --num-trials 1 --count 2

# Failure mode
uv run python -m experiments.datadog.scripts.traffic_generator \
  --local --mode failure --domain airline --count 3
```

### Dry Run

Test without sending metrics to Datadog:

```bash
uv run python -m experiments.datadog.scripts.traffic_generator \
  --local --dry-run --domain airline --count 1
```

---

## Datadog URLs

After running traffic, view results at:

| Resource | URL |
|----------|-----|
| Dashboard | https://app.datadoghq.com/dashboard/tau2-bench-health |
| APM Traces | https://app.datadoghq.com/apm/traces?query=service:tau2-bench-agent |
| Metrics | https://app.datadoghq.com/metric/explorer?query=tau2.task.reward |
| Monitors | https://app.datadoghq.com/monitors/manage |

---

## Detection Rules

| ID | Name | Trigger Condition | Action |
|----|------|-------------------|--------|
| DR-001 | High Error Rate | error_count / total > 0.2 | Create Case |
| DR-002 | Task Quality Degradation | avg:tau2.task.reward < 0.5 | Create Case |
| DR-003 | Token Cost Anomaly | token_cost > 2x baseline | Alert |
| DR-004 | Premature Termination | termination:max_errors > 10/hr | Create Incident |
| DR-005 | Latency SLO Breach | p99:duration > 60s | SLO Alert |
| DR-006 | Low Task Efficiency | reward_per_turn < 0.03 | Create Case |

---

## Directory Structure

```
datadog/
├── README.md
├── LICENSE
├── configs/
│   ├── monitors.json
│   ├── slos.json
│   ├── dashboards_agents.json
│   └── dashboards_operations.json
└── scripts/
    ├── demo.py
    ├── traffic_generator.py
    ├── emit_metrics.py
    └── setup_datadog.py
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DD_API_KEY` | Yes | Datadog API key |
| `DD_APP_KEY` | Yes | Datadog Application key |
| `DD_SITE` | No | Datadog site (default: datadoghq.com) |
| `DD_SERVICE` | No | Service name (default: tau2-bench-agent) |
| `DD_ENV` | No | Environment (default: development) |
| `DD_LLMOBS_ENABLED` | No | Enable LLM Observability (default: false) |
| `TAU2_AGENT_URL` | No | GCP tau2_agent base URL |
| `MOCK_AGENT_URL` | No | GCP mock agent base URL |
| `GEMINI_API_KEY` | Local | Gemini API key |
| `NEBIUS_API_KEY` | Local | Nebius API key |
| `TAU2_DATA_DIR` | No | Data directory (default: ./data) |

## Enabling Datadog Tracing

### Option 1: ddtrace-run

```bash
ddtrace-run tau2 run --domain mock
```

### Option 2: Environment-based

```bash
DD_TRACE_ENABLED=true \
DD_PATCH_MODULES=litellm:true,httpx:true \
tau2 run --domain mock
```

---

## Related Specs

- [007-datadog-project spec](../../../specs/007-datadog-project/spec.md)
- [008-gcp-integration spec](../../../specs/008-gcp-integration/spec.md)
