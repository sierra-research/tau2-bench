# tau2_agent - Evaluation Service Agent

ADK agent that exposes tau2-bench evaluation capabilities via A2A protocol.

## Purpose

This agent is the **evaluator** - it accepts requests to evaluate other A2A-compatible agents against tau2-bench domains. It is NOT a target agent for evaluation itself.

## Deployment Options

### Local Development

```bash
# Set environment variables
export TAU2_AGENT_MODEL="gemini-2.0-flash"
export GOOGLE_API_KEY="AIza..."  # Your Gemini API key

# Run the server
python -m tau2_agent.server
```

### Google Cloud Run (Production)

See [quickstart.md](../specs/008-gcp-integration/quickstart.md) for full deployment instructions.

**Quick Deploy:**

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
    --min-instances 0 \
    --max-instances 10 \
    --set-env-vars "TAU2_AGENT_MODEL=gemini-2.0-flash,LOG_LEVEL=INFO" \
    --set-secrets "GOOGLE_API_KEY=google-api-key:latest"
```

## User LLM Credentials

Clients provide their own LLM API keys for the user simulator via HTTP headers. This separates server orchestration costs from client evaluation costs.

### Required Headers

| Header | Description | Example |
|--------|-------------|---------|
| `X-User-LLM-Model` | LiteLLM model identifier for user simulator | `gpt-4o`, `claude-3-5-sonnet-20241022` |
| `X-User-LLM-API-Key` | API key for the LLM provider | `sk-...` (OpenAI), `sk-ant-...` (Anthropic) |

### Optional Headers

| Header | Description |
|--------|-------------|
| `X-Request-ID` | Correlation ID for request tracing |
| `Authorization` | Service authentication (when `SERVICE_API_KEYS` is configured) |

### Supported LLM Providers

| Provider | Model String | API Key Format |
|----------|--------------|----------------|
| OpenAI | `gpt-4o`, `gpt-4o-mini` | `sk-...` |
| Anthropic | `claude-3-5-sonnet-20241022` | `sk-ant-...` |
| Google | `gemini/gemini-2.0-flash` | `AIza...` |

## Usage

### Verify Agent Card

```bash
curl http://localhost:8001/a2a/tau2_agent/.well-known/agent-card.json | jq
```

### Request an Evaluation (with credential headers)

```bash
curl -X POST http://localhost:8001/a2a/tau2_agent/ \
  -H "Content-Type: application/json" \
  -H "X-User-LLM-Model: gpt-4o" \
  -H "X-User-LLM-API-Key: sk-..." \
  -d '{
    "jsonrpc": "2.0",
    "method": "message/send",
    "params": {
      "message": {
        "messageId": "eval-001",
        "role": "user",
        "parts": [{"text": "Evaluate the agent at http://localhost:8002/a2a/simple_nebius_agent on the mock domain with 3 tasks"}]
      }
    },
    "id": "req-001"
  }'
```

### List Available Domains

```bash
curl -X POST http://localhost:8001/a2a/tau2_agent/ \
  -H "Content-Type: application/json" \
  -H "X-User-LLM-Model: gpt-4o" \
  -H "X-User-LLM-API-Key: sk-..." \
  -d '{
    "jsonrpc": "2.0",
    "method": "message/send",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"text": "List available domains"}]
      }
    },
    "id": "req-002"
  }'
```

## Available Tools

### run_tau2_evaluation

Execute tau2-bench evaluation of a conversational agent.

**Parameters:**
- `domain`: Evaluation domain (airline, retail, telecom, mock)
- `agent_endpoint`: A2A endpoint of the agent to evaluate
- `user_llm`: LLM model for user simulator (default: gpt-4o)
- `num_trials`: Number of trials per task (default: 1, max: 3)
- `num_tasks`: Number of tasks to evaluate (optional, max: 30)
- `task_ids`: Specific task IDs to run (optional)

### list_domains

List all available tau2-bench evaluation domains.

### get_evaluation_results

Retrieve detailed results from a completed evaluation.

## Limits

These limits ensure evaluations complete within Cloud Run's 60-minute timeout:

| Parameter | Limit | Reason |
|-----------|-------|--------|
| `num_tasks` | Max 30 | ~2 min/task = 60 min max |
| `num_trials` | Max 3 | Multiplies execution time |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8001` | Server port (Cloud Run sets this) |
| `HOST` | `0.0.0.0` | Server bind address |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `TAU2_AGENT_MODEL` | `gemini-2.0-flash` | Orchestrator LLM model |
| `GOOGLE_API_KEY` | - | Gemini API key for orchestrator |
| `SERVICE_API_KEYS` | - | Optional comma-separated service auth keys |

## Architecture

```
tau2_agent (Evaluator)
    |
    +-- Credentials Middleware --> Extract X-User-LLM-* headers
    |
    +-- A2A Protocol --> Target Agent (e.g., simple_nebius_agent)
    |
    +-- tau2-bench evaluation framework
            |
            +-- Domain tasks (airline, retail, telecom, mock)
            +-- User simulator (uses client's LLM API key)
            +-- Metrics collection
```

## Error Responses

| HTTP Code | Error Code | Description |
|-----------|------------|-------------|
| 400 | `MISSING_HEADER` | Required credential header not provided |
| 400 | `LIMIT_EXCEEDED` | num_tasks > 30 or num_trials > 3 |
| 401 | `INVALID_AUTH` | Service authentication failed |
| 401 | `USER_LLM_AUTH_FAILED` | Client's LLM API key is invalid |
| 500 | `EVALUATION_FAILED` | Unexpected evaluation error |

## Notes

- This agent orchestrates evaluations, not receives them
- Target agents being evaluated must be A2A-compatible
- Evaluations can take several minutes depending on task count
- API keys are never logged (security requirement)
