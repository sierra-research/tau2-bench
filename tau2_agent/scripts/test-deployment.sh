#!/usr/bin/env bash
# =============================================================================
# test-deployment.sh - Verify tau2_agent Cloud Run deployment
# =============================================================================
#
# This script tests a deployed tau2_agent service with various verification
# steps to ensure proper functionality.
#
# Prerequisites:
#   - Service deployed to Cloud Run
#   - Valid LLM API key (OpenAI, Anthropic, or Gemini)
#
# Usage:
#   ./test-deployment.sh SERVICE_URL
#   ./test-deployment.sh https://tau2-agent-xxx.run.app
#
# Environment Variables:
#   USER_LLM_MODEL    - Model to use (default: gpt-4o)
#   USER_LLM_API_KEY  - API key for the model (required for full test)
#
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default configuration
USER_LLM_MODEL="${USER_LLM_MODEL:-gpt-4o}"

# Parse arguments
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 SERVICE_URL [OPTIONS]"
    echo ""
    echo "Arguments:"
    echo "  SERVICE_URL    The Cloud Run service URL"
    echo ""
    echo "Environment Variables:"
    echo "  USER_LLM_MODEL     Model to use (default: gpt-4o)"
    echo "  USER_LLM_API_KEY   API key for the model (required for full test)"
    echo ""
    echo "Examples:"
    echo "  $0 https://tau2-agent-xxx.run.app"
    echo "  USER_LLM_API_KEY=sk-... $0 https://tau2-agent-xxx.run.app"
    exit 1
fi

SERVICE_URL="${1}"

# Remove trailing slash if present
SERVICE_URL="${SERVICE_URL%/}"

echo "=== tau2_agent Deployment Verification ==="
echo "Service URL: ${SERVICE_URL}"
echo "Test Model:  ${USER_LLM_MODEL}"
echo ""

# Track test results
TESTS_PASSED=0
TESTS_FAILED=0

# Helper function to run a test
run_test() {
    local test_name="$1"
    local expected_status="$2"
    shift 2
    local curl_args=("$@")

    echo -n "Testing: ${test_name}... "

    # Run curl and capture both status and body
    HTTP_RESPONSE=$(curl -s -w "\n%{http_code}" "${curl_args[@]}" 2>&1)
    HTTP_STATUS=$(echo "${HTTP_RESPONSE}" | tail -n1)
    HTTP_BODY=$(echo "${HTTP_RESPONSE}" | sed '$d')

    if [[ "${HTTP_STATUS}" == "${expected_status}" ]]; then
        echo -e "${GREEN}PASS${NC} (HTTP ${HTTP_STATUS})"
        ((TESTS_PASSED++)) || true
        return 0
    else
        echo -e "${RED}FAIL${NC} (Expected ${expected_status}, got ${HTTP_STATUS})"
        echo "  Response: ${HTTP_BODY:0:200}"
        ((TESTS_FAILED++)) || true
        return 1
    fi
}

# -----------------------------------------------------------------------------
# Test 1: Health check - Agent card endpoint
# -----------------------------------------------------------------------------
echo ""
echo "--- Basic Connectivity Tests ---"

run_test "Agent card endpoint" "200" \
    "${SERVICE_URL}/a2a/tau2_agent/.well-known/agent-card.json"

# -----------------------------------------------------------------------------
# Test 2: Missing credential headers - should return 400
# -----------------------------------------------------------------------------
echo ""
echo "--- Credential Header Validation Tests ---"

run_test "Missing credential headers (expect 400)" "400" \
    -X POST "${SERVICE_URL}/a2a/tau2_agent" \
    -H "Content-Type: application/json" \
    -d '{"jsonrpc":"2.0","id":"test-1","method":"message/send","params":{"message":{"role":"user","parts":[{"text":"test"}]}}}'

# -----------------------------------------------------------------------------
# Test 3: Missing API key header - should return 400
# -----------------------------------------------------------------------------
run_test "Missing API key header (expect 400)" "400" \
    -X POST "${SERVICE_URL}/a2a/tau2_agent" \
    -H "Content-Type: application/json" \
    -H "X-User-LLM-Model: ${USER_LLM_MODEL}" \
    -d '{"jsonrpc":"2.0","id":"test-2","method":"message/send","params":{"message":{"role":"user","parts":[{"text":"test"}]}}}'

# -----------------------------------------------------------------------------
# Test 4: Missing model header - should return 400
# -----------------------------------------------------------------------------
run_test "Missing model header (expect 400)" "400" \
    -X POST "${SERVICE_URL}/a2a/tau2_agent" \
    -H "Content-Type: application/json" \
    -H "X-User-LLM-API-Key: dummy-key" \
    -d '{"jsonrpc":"2.0","id":"test-3","method":"message/send","params":{"message":{"role":"user","parts":[{"text":"test"}]}}}'

# -----------------------------------------------------------------------------
# Test 5: Full request with credential headers (if API key provided)
# -----------------------------------------------------------------------------
echo ""
echo "--- Functional Tests ---"

if [[ -n "${USER_LLM_API_KEY:-}" ]]; then
    echo "API key provided, running functional test..."

    # Test with valid headers - expect 200 OK
    echo -n "Testing: Full request with credential headers... "

    RESPONSE=$(curl -s -w "\n%{http_code}" \
        -X POST "${SERVICE_URL}/a2a/tau2_agent" \
        -H "Content-Type: application/json" \
        -H "X-User-LLM-Model: ${USER_LLM_MODEL}" \
        -H "X-User-LLM-API-Key: ${USER_LLM_API_KEY}" \
        -d '{"jsonrpc":"2.0","id":"test-4","method":"message/send","params":{"message":{"role":"user","parts":[{"text":"List available domains"}]}}}' \
        2>&1)

    HTTP_STATUS=$(echo "${RESPONSE}" | tail -n1)
    HTTP_BODY=$(echo "${RESPONSE}" | sed '$d')

    if [[ "${HTTP_STATUS}" == "200" ]]; then
        echo -e "${GREEN}PASS${NC} (HTTP ${HTTP_STATUS})"
        ((TESTS_PASSED++)) || true

        # Check if response contains expected A2A structure
        if echo "${HTTP_BODY}" | grep -q '"jsonrpc"'; then
            echo "  ✓ Response is valid JSON-RPC"
        fi
        if echo "${HTTP_BODY}" | grep -q '"result"'; then
            echo "  ✓ Response contains result field"
        fi
    else
        echo -e "${RED}FAIL${NC} (HTTP ${HTTP_STATUS})"
        echo "  Response: ${HTTP_BODY:0:500}"
        ((TESTS_FAILED++)) || true
    fi
else
    echo -e "${YELLOW}SKIP${NC}: Set USER_LLM_API_KEY to run functional test"
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "=== Test Summary ==="
echo -e "Passed: ${GREEN}${TESTS_PASSED}${NC}"
echo -e "Failed: ${RED}${TESTS_FAILED}${NC}"
echo ""

if [[ ${TESTS_FAILED} -eq 0 ]]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed.${NC}"
    exit 1
fi
