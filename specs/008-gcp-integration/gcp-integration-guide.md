# GCP Integration Guide for tau2_agent

This guide provides comprehensive implementation details for deploying tau2_agent to Google Cloud Platform (GCP). It covers SDK setup, CLI configuration, authentication patterns, and deployment best practices.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [GCP SDK and CLI Setup](#gcp-sdk-and-cli-setup)
3. [Authentication Patterns](#authentication-patterns)
4. [Secret Management](#secret-management)
5. [Cloud Run Deployment](#cloud-run-deployment)
6. [Configuration Files](#configuration-files)
7. [LiteLLM Gemini Integration](#litellm-gemini-integration)
8. [Environment Variables](#environment-variables)
9. [Deployment Scripts](#deployment-scripts)
10. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Tools

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.10+ | Runtime (per tau2-bench pyproject.toml) |
| gcloud CLI | 511.0.0+ | GCP resource management |
| Docker | Latest | Container building |
| uv/pip | Latest | Python dependency management |

### Required GCP APIs

Enable these APIs before deployment:

```bash
gcloud services enable \
    artifactregistry.googleapis.com \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    secretmanager.googleapis.com
```

### Python Dependencies

```
google-cloud-secret-manager>=2.18.0
litellm>=1.80.0
httpx>=0.28.0
```

---

## GCP SDK and CLI Setup

### Installation

```bash
# Install gcloud CLI (Linux/macOS)
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Initialize with your account
gcloud init
```

### Configuration Profiles

gcloud uses named configurations stored in `~/.config/gcloud/configurations/`. Create project-specific configurations:

```bash
# Create a configuration for tau2-agent deployment
gcloud config configurations create tau2-agent
gcloud config configurations activate tau2-agent

# Set project and region defaults
gcloud config set project YOUR_PROJECT_ID
gcloud config set run/region us-central1
gcloud config set run/platform managed
```

### Configuration via Environment Variables

Environment variables override file-based configuration:

```bash
# Core settings
export CLOUDSDK_CORE_PROJECT=your-project-id
export CLOUDSDK_RUN_REGION=us-central1

# Compute settings (if needed)
export CLOUDSDK_COMPUTE_ZONE=us-central1-a
```

### Viewing Current Configuration

```bash
# List all configurations
gcloud config configurations list

# Show active configuration
gcloud config list

# Export configuration
gcloud config configurations describe tau2-agent
```

---

## Authentication Patterns

### Application Default Credentials (ADC) - Recommended

ADC provides automatic credential discovery. Priority order:

1. `GOOGLE_APPLICATION_CREDENTIALS` environment variable (service account key file)
2. User credentials from `gcloud auth application-default login`
3. Attached service account (on GCP compute resources)
4. Compute Engine/Cloud Run default service account

**Local Development:**

```bash
# Set up ADC for local development
gcloud auth application-default login

# Verify credentials
gcloud auth application-default print-access-token
```

**Cloud Run:** Uses the attached service account automatically (ADC).

### Service Account Setup

For Cloud Run deployment, create a dedicated service account:

```bash
# Create service account
gcloud iam service-accounts create tau2-agent-sa \
    --display-name="tau2-agent Service Account"

# Grant necessary roles
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:tau2-agent-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:tau2-agent-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/logging.logWriter"
```

### API Key Authentication (Gemini Developer API)

For the tau2_agent orchestrator LLM, use the Gemini Developer API with an API key:

1. Get API key from [Google AI Studio](https://aistudio.google.com/)
2. Store in Secret Manager (see next section)
3. Reference in Cloud Run deployment

**Important:** This is distinct from OAuth/service account authentication. The Gemini Developer API uses simple API key auth, not IAM.

---

## Secret Management

### Creating Secrets

```bash
# Create the GOOGLE_API_KEY secret for Gemini
echo -n "AIza..." | gcloud secrets create google-api-key \
    --data-file=- \
    --replication-policy="automatic"

# Create optional service API keys (for restricting access)
echo -n "service-key-1,service-key-2" | gcloud secrets create service-api-keys \
    --data-file=- \
    --replication-policy="automatic"
```

### Granting Access to Cloud Run

```bash
# Get the Cloud Run service account
SERVICE_ACCOUNT="tau2-agent-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com"

# Grant access to read secrets
gcloud secrets add-iam-policy-binding google-api-key \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/secretmanager.secretAccessor"
```

### Version Management Best Practices

**Pin to specific versions** rather than using `latest`:

```bash
# Add a new version
echo -n "new-api-key" | gcloud secrets versions add google-api-key --data-file=-

# List versions
gcloud secrets versions list google-api-key

# Deploy with specific version (recommended)
--set-secrets="GOOGLE_API_KEY=google-api-key:2"

# Or use latest (not recommended for production)
--set-secrets="GOOGLE_API_KEY=google-api-key:latest"
```

### Python SDK for Secret Manager

When direct API access is needed (not environment variable injection):

```python
from google.cloud import secretmanager

def get_secret(project_id: str, secret_id: str, version: str = "latest") -> str:
    """Retrieve a secret from Secret Manager."""
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version}"
    response = client.access_secret_version(name=name)
    return response.payload.data.decode("UTF-8")

# Usage
api_key = get_secret("my-project", "google-api-key", "2")
```

---

## Cloud Run Deployment

### Deployment Options Comparison

| Method | Use Case | Complexity |
|--------|----------|------------|
| `gcloud run deploy --source .` | Quick deployments, source code | Low |
| `gcloud run deploy --image` | Pre-built images, CI/CD | Medium |
| `gcloud run services replace service.yaml` | Full control, GitOps | High |

### Source-Based Deployment (Simplest)

```bash
cd tau2_agent/docker_setup

gcloud run deploy tau2-agent \
    --source . \
    --region us-central1 \
    --platform managed \
    --allow-unauthenticated \
    --port 8001 \
    --memory 2Gi \
    --cpu 2 \
    --timeout 3600 \
    --concurrency 10 \
    --min-instances 0 \
    --max-instances 10 \
    --set-env-vars "TAU2_AGENT_MODEL=gemini-2.0-flash,LOG_LEVEL=INFO" \
    --set-secrets "GOOGLE_API_KEY=google-api-key:latest"
```

### Image-Based Deployment (Recommended for Production)

```bash
# Build and push image
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/tau2-agent:v1.0.0

# Deploy specific image version
gcloud run deploy tau2-agent \
    --image gcr.io/YOUR_PROJECT_ID/tau2-agent:v1.0.0 \
    --region us-central1 \
    --platform managed \
    --allow-unauthenticated \
    --port 8001 \
    --memory 2Gi \
    --cpu 2 \
    --timeout 3600 \
    --concurrency 10 \
    --min-instances 0 \
    --max-instances 10 \
    --service-account tau2-agent-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com \
    --set-env-vars "TAU2_AGENT_MODEL=gemini-2.0-flash,LOG_LEVEL=INFO" \
    --set-secrets "GOOGLE_API_KEY=google-api-key:latest"
```

### Key Deployment Flags

| Flag | Value | Purpose |
|------|-------|---------|
| `--timeout` | 3600 | 60-minute max for long evaluations |
| `--concurrency` | 10 | Requests per container instance |
| `--min-instances` | 0 | Scale to zero when idle |
| `--max-instances` | 10 | Cost control |
| `--memory` | 2Gi | Memory allocation |
| `--cpu` | 2 | CPU allocation |
| `--port` | 8001 | Container port (matches ADK default) |

---

## Configuration Files

### Cloud Run Service YAML (service.yaml)

For GitOps-style deployments, define the service declaratively:

```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: tau2-agent
  labels:
    cloud.googleapis.com/location: us-central1
  annotations:
    run.googleapis.com/ingress: all
    run.googleapis.com/description: "tau2-bench evaluation service"
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "0"
        autoscaling.knative.dev/maxScale: "10"
        run.googleapis.com/cpu-throttling: "false"
    spec:
      containerConcurrency: 10
      timeoutSeconds: 3600
      serviceAccountName: tau2-agent-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com
      containers:
        - name: tau2-agent
          image: gcr.io/YOUR_PROJECT_ID/tau2-agent:latest
          ports:
            - containerPort: 8001
              name: http1
          env:
            - name: TAU2_AGENT_MODEL
              value: "gemini-2.0-flash"
            - name: LOG_LEVEL
              value: "INFO"
            - name: PORT
              value: "8001"
            - name: GOOGLE_API_KEY
              valueFrom:
                secretKeyRef:
                  key: latest
                  name: google-api-key
          resources:
            limits:
              cpu: "2"
              memory: 2Gi
```

**Deploy with YAML:**

```bash
gcloud run services replace service.yaml --region us-central1
```

### Dockerfile Configuration

Update `tau2_agent/docker_setup/Dockerfile`:

```dockerfile
FROM python:3.10-slim

# Non-root user for security
RUN useradd -m -u 1000 tau2user

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set ownership
RUN chown -R tau2user:tau2user /app

USER tau2user

# Cloud Run uses PORT environment variable
ENV PORT=8001
EXPOSE 8001

# Start ADK server
CMD ["sh", "-c", "adk api_server --a2a . --port $PORT --host 0.0.0.0"]
```

### Environment File (.env.example)

Document required environment variables:

```bash
# Server-side configuration
TAU2_AGENT_MODEL=gemini-2.0-flash
GOOGLE_API_KEY=                    # Set via Secret Manager
LOG_LEVEL=INFO
PORT=8001

# Optional: Service access restriction
SERVICE_API_KEYS=                  # Comma-separated list

# GCP Configuration (for local development)
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
```

---

## LiteLLM Gemini Integration

### Environment Setup

```bash
# For Gemini Developer API (recommended for tau2_agent)
export GEMINI_API_KEY="AIza..."

# For Vertex AI (alternative, enterprise use)
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
export VERTEXAI_PROJECT="your-project-id"
export VERTEXAI_LOCATION="us-central1"
```

### Model String Formats

| API | Model String | Environment Variable |
|-----|--------------|---------------------|
| Gemini Developer API | `gemini/gemini-2.0-flash` | `GEMINI_API_KEY` |
| Vertex AI | `vertex_ai/gemini-2.0-flash` | ADC or `GOOGLE_APPLICATION_CREDENTIALS` |

### Python Usage

```python
from litellm import completion
import os

# Gemini Developer API
os.environ["GEMINI_API_KEY"] = "your-api-key"

response = completion(
    model="gemini/gemini-2.0-flash",
    messages=[{"role": "user", "content": "Hello"}],
    # Optional parameters
    temperature=1.0,  # Gemini defaults to 1.0
    max_tokens=1024,
)

# With explicit API key (for BYOK)
response = completion(
    model="gpt-4o",  # Client's model
    messages=[{"role": "user", "content": "Hello"}],
    api_key="client-provided-key",  # Passed from headers
)
```

### Important Gotchas

1. **Temperature defaults:** Gemini models default to `temperature=1.0`. Values below 1.0 may cause errors on some models.

2. **Model prefixes:** Always use `gemini/` prefix for Gemini Developer API to avoid confusion with Vertex AI.

3. **API key source:** LiteLLM reads from `GEMINI_API_KEY` environment variable by default, or accepts `api_key` parameter per-call.

---

## Environment Variables

### Server Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `TAU2_AGENT_MODEL` | No | `gemini-2.0-flash` | Model for tau2_agent orchestrator |
| `GOOGLE_API_KEY` | Yes | - | Gemini API key (via Secret Manager) |
| `PORT` | No | 8001 | Server port (Cloud Run sets this) |
| `LOG_LEVEL` | No | INFO | Logging verbosity |
| `SERVICE_API_KEYS` | No | - | Comma-separated service keys for access control |

### Client-Provided (via HTTP Headers)

| Header | Required | Description |
|--------|----------|-------------|
| `X-User-LLM-Model` | Yes | LiteLLM model identifier (e.g., `gpt-4o`) |
| `X-User-LLM-API-Key` | Yes | Client's API key for their chosen provider |

### GCP-Specific Variables

| Variable | Purpose |
|----------|---------|
| `GOOGLE_CLOUD_PROJECT` | GCP project ID |
| `GOOGLE_CLOUD_LOCATION` | Default region |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to service account key (local dev only) |
| `K_SERVICE` | Cloud Run service name (auto-set) |
| `K_REVISION` | Cloud Run revision name (auto-set) |

---

## Deployment Scripts

### deploy.sh

```bash
#!/bin/bash
set -e

# Configuration
PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-your-project-id}"
REGION="${GOOGLE_CLOUD_LOCATION:-us-central1}"
SERVICE_NAME="tau2-agent"
IMAGE_TAG="${1:-latest}"

echo "Deploying tau2-agent to Cloud Run..."

# Build image
echo "Building container image..."
gcloud builds submit \
    --tag "gcr.io/${PROJECT_ID}/${SERVICE_NAME}:${IMAGE_TAG}" \
    --project "${PROJECT_ID}"

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
    --image "gcr.io/${PROJECT_ID}/${SERVICE_NAME}:${IMAGE_TAG}" \
    --region "${REGION}" \
    --platform managed \
    --allow-unauthenticated \
    --port 8001 \
    --memory 2Gi \
    --cpu 2 \
    --timeout 3600 \
    --concurrency 10 \
    --min-instances 0 \
    --max-instances 10 \
    --set-env-vars "TAU2_AGENT_MODEL=gemini-2.0-flash,LOG_LEVEL=INFO" \
    --set-secrets "GOOGLE_API_KEY=google-api-key:latest" \
    --project "${PROJECT_ID}"

# Get service URL
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
    --region "${REGION}" \
    --project "${PROJECT_ID}" \
    --format 'value(status.url)')

echo "Deployment complete!"
echo "Service URL: ${SERVICE_URL}"
```

### setup-secrets.sh

```bash
#!/bin/bash
set -e

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-your-project-id}"
SERVICE_ACCOUNT="tau2-agent-sa@${PROJECT_ID}.iam.gserviceaccount.com"

echo "Setting up Secret Manager secrets..."

# Create secrets (interactive - prompts for values)
echo "Enter your Gemini API key:"
read -s GEMINI_KEY
echo -n "${GEMINI_KEY}" | gcloud secrets create google-api-key \
    --data-file=- \
    --replication-policy="automatic" \
    --project="${PROJECT_ID}" 2>/dev/null || \
    echo -n "${GEMINI_KEY}" | gcloud secrets versions add google-api-key \
    --data-file=- \
    --project="${PROJECT_ID}"

# Grant access
gcloud secrets add-iam-policy-binding google-api-key \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/secretmanager.secretAccessor" \
    --project="${PROJECT_ID}"

echo "Secrets configured successfully!"
```

### test-deployment.sh

```bash
#!/bin/bash

SERVICE_URL="${1:-https://tau2-agent-xxxxx.run.app}"
USER_MODEL="${2:-gpt-4o}"
USER_API_KEY="${3}"

if [ -z "${USER_API_KEY}" ]; then
    echo "Usage: ./test-deployment.sh <service-url> <model> <api-key>"
    exit 1
fi

echo "Testing tau2-agent deployment..."

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
                "parts": [{"text": "Run a quick health check"}]
            }
        }
    }'
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Cold start > 30s | Large container image | Reduce dependencies, use slim base image |
| Secret access denied | Missing IAM binding | Grant `secretmanager.secretAccessor` role |
| 60-min timeout reached | Evaluation too long | Reduce `num_tasks` to max 30 |
| LiteLLM auth error | Wrong model prefix | Use `gemini/` for Developer API, `vertex_ai/` for Vertex |
| Port binding error | Wrong PORT env | Ensure `PORT=8001` matches Dockerfile |

### Debugging Commands

```bash
# View Cloud Run logs
gcloud run services logs read tau2-agent --region us-central1

# Stream logs in real-time
gcloud run services logs tail tau2-agent --region us-central1

# Check service status
gcloud run services describe tau2-agent --region us-central1

# Test locally with Docker
docker run -p 8001:8001 \
    -e GOOGLE_API_KEY="your-key" \
    -e TAU2_AGENT_MODEL="gemini-2.0-flash" \
    gcr.io/your-project/tau2-agent:latest
```

### Validating Secret Access

```bash
# List secrets the service account can access
gcloud secrets list --project YOUR_PROJECT_ID \
    --filter="replication.automatic"

# Test secret retrieval
gcloud secrets versions access latest --secret=google-api-key
```

---

## References

- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Secret Manager Best Practices](https://cloud.google.com/secret-manager/docs/best-practices)
- [gcloud CLI Reference](https://cloud.google.com/sdk/gcloud/reference)
- [LiteLLM Gemini Integration](https://docs.litellm.ai/docs/providers/gemini)
- [ADK Cloud Run Deployment](https://google.github.io/adk-docs/deploy/cloud-run/)
- [Cloud Run YAML Reference](https://cloud.google.com/run/docs/reference/yaml/v1)
- [Python Optimization for Cloud Run](https://cloud.google.com/run/docs/tips/python)
