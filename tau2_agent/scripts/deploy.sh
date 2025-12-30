#!/usr/bin/env bash
# =============================================================================
# deploy.sh - Deploy tau2_agent to Google Cloud Run
# =============================================================================
#
# This script deploys tau2_agent to Cloud Run with credentials middleware support.
#
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - GCP project with billing enabled
#   - Required APIs enabled (run setup-secrets.sh first)
#   - Secret 'google-api-key' created in Secret Manager
#   - Secrets 'dd-api-key' and 'dd-site' created in Secret Manager (Datadog)
#   - Service account 'tau2-agent-sa' created with secret access
#
# Usage:
#   ./deploy.sh                    # Deploy to default project
#   ./deploy.sh --project my-proj  # Deploy to specific project
#   ./deploy.sh --region us-west1  # Deploy to specific region
#   ./deploy.sh --service my-agent # Deploy with custom service name
#
# =============================================================================

set -euo pipefail

# Default configuration
DEFAULT_REGION="us-west2"
DEFAULT_SERVICE_NAME="tau2-agent"

# Parse command line arguments
REGION="${DEFAULT_REGION}"
SERVICE_NAME="${DEFAULT_SERVICE_NAME}"
PROJECT_ID=""
DRY_RUN=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --project)
            PROJECT_ID="$2"
            shift 2
            ;;
        --region)
            REGION="$2"
            shift 2
            ;;
        --service)
            SERVICE_NAME="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --project PROJECT_ID  GCP project ID (default: current project)"
            echo "  --region REGION       Cloud Run region (default: ${DEFAULT_REGION})"
            echo "  --service NAME        Service name (default: ${DEFAULT_SERVICE_NAME})"
            echo "  --dry-run             Print commands without executing"
            echo "  --help, -h            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Get project ID if not specified
if [[ -z "${PROJECT_ID}" ]]; then
    PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
    if [[ -z "${PROJECT_ID}" ]]; then
        echo "Error: No project ID specified and no default project set"
        echo "Run: gcloud config set project YOUR_PROJECT_ID"
        exit 1
    fi
fi

echo "=== tau2_agent Cloud Run Deployment ==="
echo "Project:      ${PROJECT_ID}"
echo "Region:       ${REGION}"
echo "Service:      ${SERVICE_NAME}"
echo ""

# Get script directory and repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DOCKER_SETUP_DIR="${REPO_ROOT}/tau2_agent/docker_setup"

# Verify we're in the right place
if [[ ! -f "${DOCKER_SETUP_DIR}/Dockerfile" ]]; then
    echo "Error: Dockerfile not found at ${DOCKER_SETUP_DIR}/Dockerfile"
    exit 1
fi

# Check prerequisites
echo "Checking prerequisites..."

# Verify required secrets exist
if ! gcloud secrets describe google-api-key --project="${PROJECT_ID}" &>/dev/null; then
    echo "Error: Secret 'google-api-key' not found"
    echo "Run setup-secrets.sh first to create the secret"
    exit 1
fi
echo "  ✓ Secret 'google-api-key' exists"

# Verify Datadog secrets exist (required for LLM Observability)
if ! gcloud secrets describe dd-api-key --project="${PROJECT_ID}" &>/dev/null; then
    echo "Error: Secret 'dd-api-key' not found"
    echo "Run setup-secrets.sh first to create the Datadog secrets"
    exit 1
fi
echo "  ✓ Secret 'dd-api-key' exists"

if ! gcloud secrets describe dd-site --project="${PROJECT_ID}" &>/dev/null; then
    echo "Error: Secret 'dd-site' not found"
    echo "Run setup-secrets.sh first to create the Datadog secrets"
    exit 1
fi
echo "  ✓ Secret 'dd-site' exists"

# Verify service account exists
SA_EMAIL="tau2-agent-sa@${PROJECT_ID}.iam.gserviceaccount.com"
if ! gcloud iam service-accounts describe "${SA_EMAIL}" --project="${PROJECT_ID}" &>/dev/null; then
    echo "Error: Service account '${SA_EMAIL}' not found"
    echo "Run setup-secrets.sh first to create the service account"
    exit 1
fi
echo "  ✓ Service account exists"

echo ""
echo "Deploying from: ${REPO_ROOT}"

# Image name (Artifact Registry format - gcr.io is deprecated)
# To migrate existing images from gcr.io, run:
#   gcloud artifacts docker upgrade migrate --project=${PROJECT_ID}
ARTIFACT_REGION="${ARTIFACT_REGION:-${REGION}}"
ARTIFACT_REPO="${ARTIFACT_REPO:-tau2-agent}"
IMAGE="${ARTIFACT_REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REPO}/tau2-agent:latest"

if [[ -n "${DRY_RUN}" ]]; then
    echo ""
    echo "Dry run - would execute:"
    echo "1. gcloud builds submit --config cloudbuild.yaml --project ${PROJECT_ID}"
    echo "2. gcloud run deploy ${SERVICE_NAME} --image ${IMAGE} ..."
    exit 0
fi

# Step 1: Build container image using cloudbuild.yaml
echo ""
echo "Step 1: Building container image..."
gcloud builds submit \
    --config "${REPO_ROOT}/cloudbuild.yaml" \
    --project "${PROJECT_ID}" \
    "${REPO_ROOT}"

# Step 2: Deploy to Cloud Run
echo ""
echo "Step 2: Deploying to Cloud Run..."

# Secrets from Secret Manager; ENV vars use Dockerfile defaults
SECRETS_STRING="GOOGLE_API_KEY=google-api-key:latest,DD_API_KEY=dd-api-key:latest,DD_SITE=dd-site:latest"

gcloud run deploy "${SERVICE_NAME}" \
    --image "${IMAGE}" \
    --region "${REGION}" \
    --project "${PROJECT_ID}" \
    --platform managed \
    --allow-unauthenticated \
    --port 8001 \
    --memory 2Gi \
    --cpu 2 \
    --timeout 3600 \
    --concurrency 10 \
    --min-instances 0 \
    --max-instances 10 \
    --service-account "${SA_EMAIL}" \
    --set-secrets "${SECRETS_STRING}"

echo ""
echo "=== Deployment Complete ==="

# Get service URL
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
    --region "${REGION}" \
    --project "${PROJECT_ID}" \
    --format 'value(status.url)')

echo ""
echo "Service URL: ${SERVICE_URL}"

# Save endpoint to .env file for demo scripts
ENV_FILE="${REPO_ROOT}/.env"
if [[ -f "${ENV_FILE}" ]]; then
    # Update existing TAU2_AGENT_URL or append
    if grep -q "^TAU2_AGENT_URL=" "${ENV_FILE}"; then
        sed -i "s|^TAU2_AGENT_URL=.*|TAU2_AGENT_URL=${SERVICE_URL}|" "${ENV_FILE}"
        echo "  ✓ Updated TAU2_AGENT_URL in .env"
    else
        echo "" >> "${ENV_FILE}"
        echo "# GCP deployed tau2_agent endpoint (auto-set by deploy.sh)" >> "${ENV_FILE}"
        echo "TAU2_AGENT_URL=${SERVICE_URL}" >> "${ENV_FILE}"
        echo "  ✓ Added TAU2_AGENT_URL to .env"
    fi
else
    echo "# GCP deployed endpoints (auto-set by deploy.sh)" > "${ENV_FILE}"
    echo "TAU2_AGENT_URL=${SERVICE_URL}" >> "${ENV_FILE}"
    echo "  ✓ Created .env with TAU2_AGENT_URL"
fi

echo ""
echo "Test with:"
echo "  curl -X POST '${SERVICE_URL}/a2a/tau2_agent' \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -H 'X-User-LLM-Model: gpt-4o' \\"
echo "    -H 'X-User-LLM-API-Key: \${OPENAI_API_KEY}' \\"
echo "    -d '{\"jsonrpc\":\"2.0\",\"id\":\"test\",\"method\":\"message/send\",\"params\":{\"message\":{\"role\":\"user\",\"parts\":[{\"text\":\"List available domains\"}]}}}'"
echo ""
echo "Or run: ./test-deployment.sh ${SERVICE_URL}"
