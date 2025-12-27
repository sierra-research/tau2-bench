# Quickstart: tau2_agent GCP Deployment

This guide covers deploying tau2_agent to Google Cloud Run with BYOK (Bring Your Own Key) support.

## Prerequisites

- GCP project with billing enabled
- `gcloud` CLI installed and authenticated
- Docker installed (for local testing)
- Python 3.10+

## 1. Enable Required APIs

```bash
gcloud services enable \
    artifactregistry.googleapis.com \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    secretmanager.googleapis.com
```

## 2. Set Up Secrets

```bash
# Create the Gemini API key secret
# Get your key from https://aistudio.google.com/
echo -n "AIza..." | gcloud secrets create google-api-key \
    --data-file=- \
    --replication-policy="automatic"
```

## 3. Create Service Account

```bash
PROJECT_ID=$(gcloud config get-value project)

# Create service account
gcloud iam service-accounts create tau2-agent-sa \
    --display-name="tau2-agent Service Account"

# Grant secret access
gcloud secrets add-iam-policy-binding google-api-key \
    --member="serviceAccount:tau2-agent-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
```

## 4. Deploy to Cloud Run

```bash
cd tau2_agent/docker_setup

gcloud run deploy tau2-agent \
    --source . \
    --region us-west2 \
    --platform managed \
    --allow-unauthenticated \
    --port 8001 \
    --memory 2Gi \
    --cpu 2 \
    --timeout 3600 \
    --concurrency 10 \
    --min-instances 0 \
    --max-instances 10 \
    --service-account tau2-agent-sa@${PROJECT_ID}.iam.gserviceaccount.com \
    --set-env-vars "TAU2_AGENT_MODEL=gemini-2.0-flash,LOG_LEVEL=INFO" \
    --set-secrets "GOOGLE_API_KEY=google-api-key:latest"
```

## 5. Get Service URL

```bash
SERVICE_URL=$(gcloud run services describe tau2-agent \
    --region us-west2 \
    --format 'value(status.url)')
echo "Service URL: ${SERVICE_URL}"
```

## 6. Test the Deployment

```bash
# Replace with your actual values
SERVICE_URL="https://tau2-agent-xxx.run.app"
USER_MODEL="gpt-4o"
USER_API_KEY="sk-..."  # Your OpenAI API key

curl -X POST "${SERVICE_URL}/a2a/tau2_agent" \
    -H "Content-Type: application/json" \
    -H "X-User-LLM-Model: ${USER_MODEL}" \
    -H "X-User-LLM-API-Key: ${USER_API_KEY}" \
    -d '{
        "jsonrpc": "2.0",
        "id": "test-1",
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"text": "List available domains"}]
            }
        }
    }'
```

## Usage

### Run an Evaluation

```bash
curl -X POST "${SERVICE_URL}/a2a/tau2_agent" \
    -H "Content-Type: application/json" \
    -H "X-User-LLM-Model: gpt-4o" \
    -H "X-User-LLM-API-Key: ${OPENAI_API_KEY}" \
    -d '{
        "jsonrpc": "2.0",
        "id": "eval-1",
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{
                    "text": "Run evaluation on mock domain for http://my-agent:8001/a2a/agent with 5 tasks"
                }]
            }
        }
    }'
```

### Supported LLM Providers

| Provider | Model String | API Key Env Var |
|----------|--------------|-----------------|
| OpenAI | `gpt-4o`, `gpt-4o-mini` | `OPENAI_API_KEY` |
| Anthropic | `claude-3-5-sonnet-20241022` | `ANTHROPIC_API_KEY` |
| Google | `gemini/gemini-2.0-flash` | `GEMINI_API_KEY` |

### Limits

| Parameter | Limit | Reason |
|-----------|-------|--------|
| `num_tasks` | Max 30 | Cloud Run 60-min timeout |
| `num_trials` | Max 3 | Multiplies execution time |

## Local Development

```bash
# Set environment variables
export TAU2_AGENT_MODEL="gemini-2.0-flash"
export GOOGLE_API_KEY="AIza..."

# Run locally
cd tau2_agent
python -m tau2_agent.server
```

## Troubleshooting

### View Logs

```bash
gcloud run services logs read tau2-agent --region us-west2
```

### Check Service Status

```bash
gcloud run services describe tau2-agent --region us-west2
```

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| 400: Missing header | BYOK headers not provided | Add `X-User-LLM-Model` and `X-User-LLM-API-Key` headers |
| 401: Auth failed | Invalid API key | Check your LLM provider API key |
| 504: Timeout | Evaluation too long | Reduce `num_tasks` to ≤30 |

## Cost Estimate

| Component | Monthly Cost |
|-----------|--------------|
| Cloud Run (scale to zero) | $1-10 |
| Gemini orchestrator | $5-20 |
| User simulator | $0 (client pays) |
| **Total Server Cost** | **$6-30** |
