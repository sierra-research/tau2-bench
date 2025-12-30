#!/usr/bin/env bash
# =============================================================================
# deploy.sh - Deploy simple_gemini_agent to Google Cloud Run
# =============================================================================
#
# This script deploys simple_gemini_agent (mock agent) to Cloud Run.
#
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - Required secrets in Secret Manager (google-api-key, dd-api-key, dd-site)
#
# Usage:
#   ./deploy.sh                    # Deploy to default project
#   ./deploy.sh --project my-proj  # Deploy to specific project
#
# =============================================================================

set -euo pipefail

# Default configuration
DEFAULT_REGION="us-west2"
SERVICE_NAME="simple-gemini-agent"

# Parse command line arguments
REGION="${DEFAULT_REGION}"
PROJECT_ID=""

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
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --project PROJECT_ID  GCP project ID (default: current project)"
            echo "  --region REGION       Cloud Run region (default: ${DEFAULT_REGION})"
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
        exit 1
    fi
fi

# Get script directory and repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "=== simple_gemini_agent Cloud Run Deployment ==="
echo "Project: ${PROJECT_ID}"
echo "Region:  ${REGION}"
echo "Service: ${SERVICE_NAME}"
echo ""

# Run Cloud Build
echo "Building and deploying via Cloud Build..."
gcloud builds submit \
    --config "${SCRIPT_DIR}/cloudbuild.yaml" \
    --project "${PROJECT_ID}" \
    "${REPO_ROOT}"

# Get service URL
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
    --region "${REGION}" \
    --project "${PROJECT_ID}" \
    --format 'value(status.url)')

echo ""
echo "=== Deployment Complete ==="
echo "Service URL: ${SERVICE_URL}"

# Save endpoint to .env file for demo scripts
ENV_FILE="${REPO_ROOT}/.env"
if [[ -f "${ENV_FILE}" ]]; then
    # Update existing MOCK_AGENT_URL or append
    if grep -q "^MOCK_AGENT_URL=" "${ENV_FILE}"; then
        sed -i "s|^MOCK_AGENT_URL=.*|MOCK_AGENT_URL=${SERVICE_URL}|" "${ENV_FILE}"
        echo "  ✓ Updated MOCK_AGENT_URL in .env"
    else
        echo "" >> "${ENV_FILE}"
        echo "# GCP deployed mock agent endpoint (auto-set by deploy.sh)" >> "${ENV_FILE}"
        echo "MOCK_AGENT_URL=${SERVICE_URL}" >> "${ENV_FILE}"
        echo "  ✓ Added MOCK_AGENT_URL to .env"
    fi
else
    echo "# GCP deployed endpoints (auto-set by deploy.sh)" > "${ENV_FILE}"
    echo "MOCK_AGENT_URL=${SERVICE_URL}" >> "${ENV_FILE}"
    echo "  ✓ Created .env with MOCK_AGENT_URL"
fi

echo ""
echo "Test with:"
echo "  curl '${SERVICE_URL}/a2a/simple_gemini_agent/.well-known/agent-card.json'"
