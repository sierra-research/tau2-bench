#!/bin/bash
#
# Datadog LLM Observability Demo Script
#
# This script runs a complete demo of the tau2-bench Datadog integration:
# 1. Checks prerequisites (API keys)
# 2. Creates Datadog resources (monitors, SLOs, dashboard)
# 3. Generates normal traffic to produce baseline metrics
# 4. Generates failure traffic to trigger DR-002 (Task Failure Spike) monitor
# 5. Emits metrics to Datadog
# 6. Outputs summary with dashboard URL
#
# Usage:
#   ./demo.sh                    # Full demo with Datadog API calls
#   ./demo.sh --dry-run          # Local demo without Datadog API calls
#   ./demo.sh --help             # Show help
#
# Environment Variables:
#   DD_API_KEY      - Required (except --dry-run): Datadog API key
#   DD_APP_KEY      - Required (except --dry-run): Datadog Application key
#   DD_SITE         - Optional: Datadog site (default: datadoghq.com)
#   NEBIUS_API_KEY  - Required: Nebius API key for mock agent LLM calls
#   TAU2_DATA_DIR   - Optional: Data directory (default: ./data)
#
# Copyright 2024 Timothy Wu
# SPDX-License-Identifier: Apache-2.0
#

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find project root by looking for pyproject.toml
find_project_root() {
    local dir="$1"
    while [[ "$dir" != "/" ]]; do
        if [[ -f "$dir/pyproject.toml" ]]; then
            echo "$dir"
            return 0
        fi
        dir="$(dirname "$dir")"
    done
    return 1
}

PROJECT_ROOT="$(find_project_root "$SCRIPT_DIR")"
if [[ -z "$PROJECT_ROOT" ]]; then
    echo "Error: Could not find project root (no pyproject.toml found)"
    exit 1
fi

# Default values
DRY_RUN=false
NORMAL_COUNT=5
FAILURE_COUNT=3
VERBOSE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# Helper Functions
# ============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_section() {
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
}

show_help() {
    cat << EOF
Datadog LLM Observability Demo Script

Usage: ./demo.sh [OPTIONS]

Options:
    --dry-run       Run demo locally without Datadog API calls
    --normal N      Number of normal evaluations (default: 5)
    --failure N     Number of failure evaluations (default: 3)
    --verbose       Enable verbose output
    --help          Show this help message

Environment Variables:
    DD_API_KEY      Required (except --dry-run): Datadog API key
    DD_APP_KEY      Required (except --dry-run): Datadog Application key
    DD_SITE         Optional: Datadog site (default: datadoghq.com)
    NEBIUS_API_KEY  Required: Nebius API key for mock agent LLM calls
    TAU2_DATA_DIR   Optional: Data directory (default: ./data)

Examples:
    # Full demo with Datadog
    ./demo.sh

    # Local dry-run demo
    ./demo.sh --dry-run

    # Custom evaluation counts
    ./demo.sh --normal 10 --failure 5

EOF
}

# ============================================================================
# Argument Parsing
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --normal)
            NORMAL_COUNT="$2"
            shift 2
            ;;
        --failure)
            FAILURE_COUNT="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# ============================================================================
# Step 1: Prerequisite Check
# ============================================================================

log_section "Step 1: Checking Prerequisites"

check_env_var() {
    local var_name="$1"
    local required="$2"
    local value="${!var_name}"

    if [[ -z "$value" ]]; then
        if [[ "$required" == "true" ]]; then
            log_error "$var_name is not set"
            return 1
        else
            log_warning "$var_name is not set (optional)"
            return 0
        fi
    else
        # Mask the value for security
        local masked="${value:0:4}****${value: -4}"
        log_success "$var_name is set: $masked"
        return 0
    fi
}

PREREQ_FAILED=false

# NEBIUS_API_KEY is always required (for mock agent LLM calls)
if ! check_env_var "NEBIUS_API_KEY" "true"; then
    PREREQ_FAILED=true
fi

# DD_API_KEY and DD_APP_KEY are required unless --dry-run
if [[ "$DRY_RUN" == "false" ]]; then
    if ! check_env_var "DD_API_KEY" "true"; then
        PREREQ_FAILED=true
    fi
    if ! check_env_var "DD_APP_KEY" "true"; then
        PREREQ_FAILED=true
    fi
    # DD_SITE is optional
    check_env_var "DD_SITE" "false"
else
    log_info "Dry-run mode: Skipping Datadog API key checks"
    check_env_var "DD_API_KEY" "false"
    check_env_var "DD_APP_KEY" "false"
fi

# Check Python is available
if ! command -v python &> /dev/null; then
    log_error "Python is not installed or not in PATH"
    PREREQ_FAILED=true
else
    log_success "Python is available: $(python --version)"
fi

# Check uv is available
if ! command -v uv &> /dev/null; then
    log_warning "uv is not installed. Will use python directly."
else
    log_success "uv is available: $(uv --version)"
fi

if [[ "$PREREQ_FAILED" == "true" ]]; then
    log_error "Prerequisite check failed. Please set the required environment variables."
    exit 1
fi

log_success "All prerequisites satisfied!"

# ============================================================================
# Step 2: Create Datadog Resources
# ============================================================================

log_section "Step 2: Creating Datadog Resources"

if [[ "$DRY_RUN" == "true" ]]; then
    log_info "Dry-run mode: Skipping Datadog resource creation"
else
    log_info "Creating monitors, SLOs, and dashboard..."

    cd "$PROJECT_ROOT"

    if ! uv run python -m experiments.datadog.scripts.setup_datadog --all; then
        log_warning "Datadog resource creation had issues, but continuing with demo..."
    else
        log_success "Datadog resources created successfully!"
    fi
fi

# ============================================================================
# Step 3: Generate Normal Traffic
# ============================================================================

log_section "Step 3: Generating Normal Traffic"

log_info "Running $NORMAL_COUNT normal evaluations..."

cd "$PROJECT_ROOT"

TRAFFIC_ARGS="--count $NORMAL_COUNT --mode normal"
if [[ "$DRY_RUN" == "true" ]]; then
    TRAFFIC_ARGS="$TRAFFIC_ARGS --dry-run"
fi
if [[ "$VERBOSE" == "true" ]]; then
    TRAFFIC_ARGS="$TRAFFIC_ARGS --log-level DEBUG"
fi

if ! uv run python -m experiments.datadog.scripts.traffic_generator $TRAFFIC_ARGS; then
    log_warning "Some normal evaluations failed, but continuing..."
else
    log_success "Normal traffic generation completed!"
fi

# ============================================================================
# Step 4: Generate Failure Traffic
# ============================================================================

log_section "Step 4: Generating Failure Traffic (Trigger DR-002)"

log_info "Running $FAILURE_COUNT failure evaluations to trigger Task Failure Spike monitor..."

TRAFFIC_ARGS="--count $FAILURE_COUNT --mode failure"
if [[ "$DRY_RUN" == "true" ]]; then
    TRAFFIC_ARGS="$TRAFFIC_ARGS --dry-run"
fi
if [[ "$VERBOSE" == "true" ]]; then
    TRAFFIC_ARGS="$TRAFFIC_ARGS --log-level DEBUG"
fi

# Note: traffic_generator.py already calls emit_metrics.py at the end
if ! uv run python -m experiments.datadog.scripts.traffic_generator $TRAFFIC_ARGS; then
    log_warning "Some failure evaluations failed, but this is expected..."
else
    log_success "Failure traffic generation completed!"
fi

# ============================================================================
# Step 5: Emit Final Metrics
# ============================================================================

log_section "Step 5: Emitting Final Metrics"

log_info "Emitting all stored evaluation metrics to Datadog..."

EMIT_ARGS="--all"
if [[ "$DRY_RUN" == "true" ]]; then
    EMIT_ARGS="$EMIT_ARGS --dry-run"
fi

if ! uv run python -m experiments.datadog.scripts.emit_metrics $EMIT_ARGS; then
    log_warning "Metrics emission had issues, but continuing..."
else
    log_success "Metrics emitted successfully!"
fi

# ============================================================================
# Step 6: Summary
# ============================================================================

log_section "Step 6: Demo Complete!"

# Count evaluations stored
DATA_DIR="${TAU2_DATA_DIR:-./data}"
EVAL_COUNT=$(find "$DATA_DIR/evaluations" -name "*.json" 2>/dev/null | wc -l | tr -d ' ' || echo "0")

echo ""
log_info "Demo Summary:"
echo "  - Normal evaluations requested: $NORMAL_COUNT"
echo "  - Failure evaluations requested: $FAILURE_COUNT"
echo "  - Total evaluations stored: $EVAL_COUNT"
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
    log_info "Dry-run mode: No Datadog API calls were made"
    echo ""
    echo "To run with Datadog integration:"
    echo "  export DD_API_KEY=your_api_key"
    echo "  export DD_APP_KEY=your_app_key"
    echo "  ./demo.sh"
else
    DD_SITE="${DD_SITE:-datadoghq.com}"
    DASHBOARD_URL="https://app.${DD_SITE}/dashboard/tau2-bench-health"
    APM_URL="https://app.${DD_SITE}/apm/traces?query=service:tau2-bench-agent"
    METRICS_URL="https://app.${DD_SITE}/metric/explorer?query=tau2.task.reward"
    MONITORS_URL="https://app.${DD_SITE}/monitors/manage"

    log_info "Datadog URLs:"
    echo ""
    echo "  Dashboard:  $DASHBOARD_URL"
    echo "  APM Traces: $APM_URL"
    echo "  Metrics:    $METRICS_URL"
    echo "  Monitors:   $MONITORS_URL"
    echo ""
    log_info "Check the monitors page for DR-002 (Task Failure Spike) alerts!"
fi

log_success "Demo completed successfully!"
