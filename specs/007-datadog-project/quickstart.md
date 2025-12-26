# Quickstart: Datadog LLM Observability Hackathon Project

**Feature Branch**: `007-datadog-project`
**Date**: 2025-12-24

## Prerequisites

### Required Accounts
- [ ] **Datadog Account** - Free trial at https://www.datadoghq.com/free-datadog-trial/
- [ ] **Google Cloud Account** - For Gemini API access
- [ ] **GitHub Account** - For hackathon repo submission

### Required API Keys
- [ ] `DD_API_KEY` - Datadog API key (from Datadog → Organization Settings → API Keys)
- [ ] `DD_APP_KEY` - Datadog Application key (for monitor/dashboard creation)
- [ ] `GOOGLE_API_KEY` - Gemini Developer API key (from https://makersuite.google.com/app/apikey)

### Required Tools
- Python 3.10+
- pip or uv (package manager)
- Docker (for Cloud Run deployment)
- gcloud CLI (for GCP deployment)

## Local Development Setup

### 1. Install Dependencies

```bash
# Clone the repository
git clone https://github.com/wuTims/tau2-bench-agent.git
cd tau2-bench-agent

# Checkout the feature branch
git checkout 007-datadog-project

# Install with Datadog extras
pip install -e ".[datadog]"

# Or using uv
uv pip install -e ".[datadog]"
```

### 2. Configure Environment

Create a `.env` file in the repository root:

```bash
# Datadog Configuration
DD_TRACE_ENABLED=true
DD_LLMOBS_ENABLED=true
DD_SERVICE=tau2-bench-agent
DD_ENV=development
DD_API_KEY=your_datadog_api_key
DD_SITE=datadoghq.com  # or datadoghq.eu, us3.datadoghq.com, etc.

# Gemini Configuration
GOOGLE_API_KEY=your_gemini_api_key
TAU2_AGENT_MODEL=gemini-2.0-flash

# Application Configuration
TAU2_DATA_DIR=./data
LOG_LEVEL=INFO
```

### 3. Run Local Evaluation with Tracing

The ddtrace integration is opt-in and does not modify tau2 core. Choose one of these methods:

```bash
# Load environment
export $(cat .env | xargs)

# Option 1: Use the traced wrapper (recommended)
python -m experiments.datadog.scripts.tau2_traced run --domain mock

# Option 2: Use ddtrace-run (ddtrace's built-in wrapper)
ddtrace-run tau2 run --domain mock

# Option 3: Environment-based (requires DD_PATCH_MODULES)
DD_PATCH_MODULES=litellm:true,httpx:true tau2 run --domain mock

# Verify traces appear in Datadog APM
open "https://app.datadoghq.com/apm/traces?query=service:tau2-bench-agent"
```

### 4. Emit Custom Metrics (Post-Evaluation)

```bash
# After evaluation completes, emit metrics from stored JSON
python src/experiments/datadog/scripts/emit_metrics.py

# Verify metrics appear in Datadog
open "https://app.datadoghq.com/metric/explorer?query=tau2.task.reward"
```

## Datadog Setup

### 1. Create Monitors

```bash
# Set up API keys
export DD_API_KEY=your_api_key
export DD_APP_KEY=your_app_key

# Create monitors from JSON
python src/experiments/datadog/scripts/setup_datadog.py --monitors
```

### 2. Create Dashboard

```bash
# Create dashboard from JSON
python src/experiments/datadog/scripts/setup_datadog.py --dashboard
```

### 3. Create SLOs

```bash
# Create SLOs from JSON
python src/experiments/datadog/scripts/setup_datadog.py --slos
```

### 4. Run Full Setup

```bash
# Create all Datadog resources at once
python src/experiments/datadog/scripts/setup_datadog.py --all
```

## Demo Workflow

### 1. Generate Normal Traffic

```bash
# Run traffic generator for baseline metrics
python src/experiments/datadog/scripts/traffic_generator.py --mode normal --tasks 10
```

### 2. Trigger Detection Rules

```bash
# Run failure mode to trigger monitors
python src/experiments/datadog/scripts/traffic_generator.py --mode failure --tasks 5

# This will:
# - Generate low-reward tasks (triggers DR-002)
# - Cause MAX_ERRORS terminations (triggers DR-004)
# - Simulate high latency (triggers DR-005)
```

### 3. Verify Case Creation

```bash
# Check for created cases
open "https://app.datadoghq.com/cases"
```

### 4. View Dashboard

```bash
# Open the tau2-bench dashboard
open "https://app.datadoghq.com/dashboard/tau2-bench-health"
```

## Cloud Run Deployment

### 1. Build Docker Image

```bash
cd src/experiments/datadog

# Build the image
docker build -t tau2-datadog-observability .

# Test locally
docker run -e DD_API_KEY=$DD_API_KEY -e GOOGLE_API_KEY=$GOOGLE_API_KEY \
  tau2-datadog-observability
```

### 2. Deploy to Cloud Run

```bash
# Authenticate with GCP
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Deploy
gcloud run deploy tau2-datadog-observability \
  --source . \
  --platform managed \
  --region us-central1 \
  --timeout 3600 \
  --set-env-vars DD_API_KEY=$DD_API_KEY,GOOGLE_API_KEY=$GOOGLE_API_KEY,DD_TRACE_ENABLED=true,DD_LLMOBS_ENABLED=true
```

## Hackathon Submission

### 1. Extract to Standalone Repo

```bash
# Extract datadog directory with history
git subtree split -P src/experiments/datadog -b datadog-standalone

# Create new repo and push
gh repo create tau2-datadog-observability --public
git push origin datadog-standalone:main
```

### 2. Export Datadog Configurations

```bash
# Export all configurations to JSON
python src/experiments/datadog/scripts/setup_datadog.py --export
```

### 3. Submission Checklist

- [ ] Public GitHub repo with Apache-2.0 license
- [ ] README with deployment instructions
- [ ] `datadog/` directory with JSON exports
- [ ] Traffic generator script
- [ ] Hosted Cloud Run URL
- [ ] Datadog organization name

## Troubleshooting

### No Traces Appearing

1. Verify `DD_TRACE_ENABLED=true`
2. Check API key is valid: `curl -X GET "https://api.datadoghq.com/api/v1/validate" -H "DD-API-KEY: $DD_API_KEY"`
3. Check logs for ddtrace initialization

### No LLM Observability Data

1. Verify `DD_LLMOBS_ENABLED=true`
2. Check LiteLLM is being patched: `from ddtrace import patch; patch(litellm=True)`
3. Verify Gemini API calls are working

### Monitors Not Triggering

1. Ensure metrics are being emitted: check Metrics Explorer
2. Verify monitor queries match metric names and tags
3. Check threshold values in monitor definition

### Cloud Run Timeout

1. Reduce `--num-tasks` to 20 or fewer
2. Check task duration in APM traces
3. Consider using async evaluation (003-async-evaluation)

## Useful Links

- [Datadog LLM Observability Docs](https://docs.datadoghq.com/llm_observability/)
- [ddtrace Python Docs](https://ddtrace.readthedocs.io/)
- [Gemini Developer API](https://ai.google.dev/tutorials/python_quickstart)
- [Cloud Run Docs](https://cloud.google.com/run/docs)
