#!/usr/bin/env bash
#
# setup-secrets.sh - Set up GCP Secret Manager for tau2_agent deployment
#
# This script creates the google-api-key secret and grants access to the
# tau2-agent-sa service account.
#
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - GCP project selected (gcloud config set project PROJECT_ID)
#   - Secret Manager API enabled
#
# Usage:
#   ./setup-secrets.sh [GOOGLE_API_KEY]
#
#   If GOOGLE_API_KEY is not provided, the script will prompt for it.
#
# Example:
#   ./setup-secrets.sh AIza...
#   # or
#   ./setup-secrets.sh  # prompts for key

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
if ! command -v gcloud &> /dev/null; then
    log_error "gcloud CLI not found. Please install: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Get project ID
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
if [[ -z "${PROJECT_ID}" ]]; then
    log_error "No GCP project set. Run: gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

log_info "Using GCP project: ${PROJECT_ID}"

# Enable Secret Manager API if not already enabled
log_info "Ensuring Secret Manager API is enabled..."
gcloud services enable secretmanager.googleapis.com --quiet || {
    log_warn "Could not enable Secret Manager API. It may already be enabled."
}

# Get or prompt for API key
GOOGLE_API_KEY="${1:-}"
if [[ -z "${GOOGLE_API_KEY}" ]]; then
    log_info "Get your Gemini API key from: https://aistudio.google.com/"
    echo -n "Enter your Google API key (Gemini): "
    read -rs GOOGLE_API_KEY
    echo
fi

if [[ -z "${GOOGLE_API_KEY}" ]]; then
    log_error "API key cannot be empty"
    exit 1
fi

# Helper function to create or update a secret
create_or_update_secret() {
    local secret_name="$1"
    local secret_value="$2"

    if gcloud secrets describe "${secret_name}" --project="${PROJECT_ID}" &>/dev/null; then
        log_info "Secret '${secret_name}' exists. Adding new version..."
        echo -n "${secret_value}" | gcloud secrets versions add "${secret_name}" \
            --project="${PROJECT_ID}" \
            --data-file=-
    else
        log_info "Creating secret '${secret_name}'..."
        echo -n "${secret_value}" | gcloud secrets create "${secret_name}" \
            --project="${PROJECT_ID}" \
            --replication-policy="automatic" \
            --data-file=-
    fi
}

# Create or update the Google API key secret
create_or_update_secret "google-api-key" "${GOOGLE_API_KEY}"
log_info "Google API key secret created/updated successfully"

# Optionally create Datadog secrets for LLM Observability
echo ""
log_info "Datadog secrets are optional but enable LLM Observability in GCP."
echo -n "Do you want to configure Datadog secrets? (y/N): "
read -r SETUP_DATADOG

if [[ "${SETUP_DATADOG}" =~ ^[Yy]$ ]]; then
    # Get Datadog API key
    log_info "Get your Datadog API key from: https://app.datadoghq.com/organization-settings/api-keys"
    echo -n "Enter your Datadog API key: "
    read -rs DD_API_KEY
    echo

    if [[ -n "${DD_API_KEY}" ]]; then
        create_or_update_secret "dd-api-key" "${DD_API_KEY}"
        log_info "Datadog API key secret created/updated successfully"
    else
        log_warn "Datadog API key not provided, skipping"
    fi

    # Get Datadog site (optional, has default)
    echo -n "Enter your Datadog site (default: datadoghq.com): "
    read -r DD_SITE
    DD_SITE="${DD_SITE:-datadoghq.com}"

    create_or_update_secret "dd-site" "${DD_SITE}"
    log_info "Datadog site secret created/updated successfully"
fi

# Create service account if it doesn't exist
SA_NAME="tau2-agent-sa"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

if ! gcloud iam service-accounts describe "${SA_EMAIL}" &>/dev/null; then
    log_info "Creating service account '${SA_NAME}'..."
    gcloud iam service-accounts create "${SA_NAME}" \
        --display-name="tau2-agent Service Account" \
        --description="Service account for tau2_agent Cloud Run deployment"
else
    log_info "Service account '${SA_NAME}' already exists"
fi

# Grant secret access to service account
log_info "Granting secret access to service account..."

# Helper function to grant secret access
grant_secret_access() {
    local secret_name="$1"
    if gcloud secrets describe "${secret_name}" --project="${PROJECT_ID}" &>/dev/null; then
        gcloud secrets add-iam-policy-binding "${secret_name}" \
            --project="${PROJECT_ID}" \
            --member="serviceAccount:${SA_EMAIL}" \
            --role="roles/secretmanager.secretAccessor" \
            --quiet
        log_info "Granted access to '${secret_name}'"
    fi
}

grant_secret_access "google-api-key"
grant_secret_access "dd-api-key"
grant_secret_access "dd-site"

log_info "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Deploy to Cloud Run using: ./deploy.sh"
echo "  2. The deployment will automatically inject secrets from Secret Manager"
echo ""
echo "Service account: ${SA_EMAIL}"
echo "Secrets configured:"
echo "  - google-api-key (required)"
if gcloud secrets describe "dd-api-key" --project="${PROJECT_ID}" &>/dev/null 2>&1; then
    echo "  - dd-api-key (Datadog LLM Observability)"
    echo "  - dd-site (Datadog site)"
fi
