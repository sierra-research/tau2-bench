# 008: GCP Integration - tau2_agent Cloud Deployment

## Overview

Deploy tau2_agent as a hosted evaluation service on Google Cloud Platform, allowing external clients to run tau2-bench evaluations without managing infrastructure.

## Clarifications

### Session 2025-12-25

- Q: When the client's LLM provider returns a rate limit error, timeout, or transient failure mid-evaluation, what should happen? → A: Fail the entire evaluation and return an error response (no partial results)
- Q: Should this GCP deployment include observability instrumentation (metrics, tracing)? → A: Defer to 007-datadog-project; this spec focuses on deployment infrastructure only
- Q: How should API versioning be handled for future breaking changes? → A: No versioning initially; add `/v2/` prefix only when breaking changes are needed
- Q: What availability level is expected for the hosted service? → A: Best effort, no formal SLA (cold starts and occasional failures acceptable)
- Q: How should "Evaluation results returned correctly" be validated? → A: Manual verification - run one evaluation and visually inspect the response structure

## Problem Statement

Currently, tau2_agent runs locally with LLM credentials stored in environment variables. For a public-hosted evaluation service:

1. **Cost Attribution**: Server operator should not pay for all LLM usage
2. **Provider Flexibility**: Clients should choose their preferred LLM provider for user simulation
3. **Scalability**: Service should scale with demand and cost-effectively idle when unused

## Architecture

### LLM Roles in tau2-bench Evaluation

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        tau2-bench Evaluation Flow                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐ │
│  │  tau2_agent LLM  │     │ User Simulator   │     │  Agent Under     │ │
│  │  (Orchestrator)  │     │     LLM          │     │   Evaluation     │ │
│  ├──────────────────┤     ├──────────────────┤     ├──────────────────┤ │
│  │ Purpose:         │     │ Purpose:         │     │ Purpose:         │ │
│  │ - Orchestrate    │     │ - Simulate user  │     │ - Handle tasks   │ │
│  │   evaluation     │     │   interactions   │     │ - Being tested   │ │
│  │ - Analyze traces │     │ - Generate       │     │                  │ │
│  │ - Report results │     │   realistic      │     │                  │ │
│  │                  │     │   requests       │     │                  │ │
│  ├──────────────────┤     ├──────────────────┤     ├──────────────────┤ │
│  │ Cost: Server     │     │ Cost: CLIENT     │     │ Cost: External   │ │
│  │ (default Gemini) │     │ (BYOK required)  │     │ (not our concern)│ │
│  └──────────────────┘     └──────────────────┘     └──────────────────┘ │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | LLM Provider | Cost Bearer | Notes |
|-----------|--------------|-------------|-------|
| **tau2_agent** | Default Gemini (gemini-2.0-flash) | Server operator | Low cost, predictable usage |
| **User Simulator** | Client-specified | Client (BYOK) | High usage, varies per evaluation |
| **Agent Under Evaluation** | External | External party | Hosted separately, not managed |

## Functional Requirements

### FR-1: BYOK for User Simulator LLM

Clients MUST provide LLM credentials for the user simulator via HTTP headers:

```http
POST /a2a/tau2_agent
Headers:
  X-User-LLM-Model: gpt-4o              # litellm model identifier
  X-User-LLM-API-Key: sk-...            # API key for the model provider
  Content-Type: application/json
```

**Supported Providers** (via litellm):
- `gpt-4o`, `gpt-4o-mini` (OpenAI)
- `gemini/gemini-2.0-flash`, `gemini/gemini-2.0-pro` (Google)
- `claude-3-5-sonnet-20241022` (Anthropic)
- Any litellm-supported model

### FR-2: Server-Side Agent LLM

The tau2_agent orchestrator uses a server-configured default model:
- **Default**: `gemini-2.0-flash` via Gemini Developer API
- **Configured via**: Server environment variable `TAU2_AGENT_MODEL`
- **API Key**: Server-managed `GOOGLE_API_KEY` environment variable

### FR-3: Request Validation

The service MUST validate incoming requests:

| Condition | Response |
|-----------|----------|
| Missing `X-User-LLM-Model` header | 400 Bad Request |
| Missing `X-User-LLM-API-Key` header | 400 Bad Request |
| Invalid API key (LLM call fails) | 401 Unauthorized |

### FR-4: Service Authentication (Optional)

Optional API key authentication to restrict access:

```http
Authorization: Bearer <service-api-key>
```

If enabled, requests without valid service API key receive 401 Unauthorized.

## Non-Functional Requirements

### NFR-1: Deployment Target

- **Platform**: Google Cloud Run
- **Region**: us-central1 (or configurable)
- **Scaling**: 0-10 instances (scale to zero when idle)

### NFR-2: Cost Targets

| Component | Target Monthly Cost |
|-----------|---------------------|
| Cloud Run compute | $1-10 |
| tau2_agent LLM (Gemini) | $5-20 |
| User Simulator LLM | $0 (client pays) |
| **Total Server Cost** | **$6-30/month** |

### NFR-3: Performance

- Cold start: < 30 seconds
- **Request timeout: 60 minutes** (Cloud Run hard limit)
- Concurrent evaluations: Limited by Cloud Run instances

### NFR-3.1: Benchmark Execution Time Constraints

**Cloud Run imposes a 60-minute maximum request timeout.** This limits which benchmarks can be run synchronously.

#### Benchmark Timing Analysis

| Domain | Total Tasks | Time per Task | Full Domain Time | Supported? |
|--------|-------------|---------------|------------------|------------|
| Mock | 9 | ~40s | ~6 min | ✅ Full |
| Airline | 50 | ~64s avg | ~53 min | ⚠️ Risky (borderline) |
| Retail | 114 | ~69s avg | ~2-3 hours | ❌ Task limit required |
| Telecom | 2,285 | ~191s avg | ~20+ days | ❌ Task limit required |

#### Enforced Limits

To ensure requests complete within Cloud Run's timeout:

| Parameter | Limit | Rationale |
|-----------|-------|-----------|
| `num_tasks` | Max 30 | ~30-40 min execution, safe margin |
| `num_trials` | Max 3 | Multiplies execution time |

Requests exceeding limits receive 400 Bad Request with explanation.

#### Supported Use Cases

| Use Case | Supported | Notes |
|----------|-----------|-------|
| Quick smoke test (1-5 tasks) | ✅ | Any domain |
| Mock domain (full) | ✅ | 9 tasks, ~6 min |
| Airline sample (10-30 tasks) | ✅ | Subset of 50 |
| Retail sample (10-30 tasks) | ✅ | Subset of 114 |
| Telecom sample (10-30 tasks) | ✅ | Subset of 2,285 |
| Full domain evaluation | ❌ | Requires local execution or async (future) |

#### Future Enhancement: Async Evaluation

For full domain evaluations, a future async pattern could:
1. Accept request, return `job_id` immediately
2. Run evaluation in background (Cloud Run Jobs or separate compute)
3. Client polls for completion
4. Return results when ready

This is **out of scope** for initial deployment but documented as future work.

### NFR-4: Availability

- **SLA**: Best effort, no formal uptime guarantees
- **Cold starts**: Acceptable (scale-to-zero prioritizes cost over latency)
- **Failure handling**: Clients should retry on transient errors
- **Rationale**: Service provides benchmarking convenience, not production-critical infrastructure

### NFR-5: Security

- HTTPS only (Cloud Run default)
- API keys never logged
- Non-root container user (existing in Dockerfile)

## Technical Design

### Request Flow

```
Client
   │
   │ POST /a2a/tau2_agent
   │ Headers: X-User-LLM-Model, X-User-LLM-API-Key
   │
   ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Cloud Run                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ 1. BYOK Middleware                                        │  │
│  │    - Validate X-User-LLM-Model header present             │  │
│  │    - Validate X-User-LLM-API-Key header present           │  │
│  │    - Store in request context                              │  │
│  └───────────────────────────┬───────────────────────────────┘  │
│                              ▼                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ 2. ADK A2A Handler                                        │  │
│  │    - Process JSON-RPC request                              │  │
│  │    - tau2_agent uses server's Gemini for orchestration    │  │
│  └───────────────────────────┬───────────────────────────────┘  │
│                              ▼                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ 3. Evaluation Execution                                    │  │
│  │    - User simulator uses client's LLM (from headers)      │  │
│  │    - Agent under eval called via A2A                       │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Files to Modify

**All changes isolated to `tau2_agent/` - no modifications to core `src/tau2/` required.**

tau2 already supports `llm_args_user={"api_key": ...}` via `RunConfig`. We just need to pass the user's API key from headers instead of environment variables.

| File | Changes |
|------|---------|
| `tau2_agent/agent.py` | Use server-configured Gemini model (remove env-var switching) |
| `tau2_agent/tools/run_tau2_evaluation.py` | Read user LLM API key from request context instead of `os.getenv()` |

### Files to Create

| File | Purpose |
|------|---------|
| `tau2_agent/middleware.py` | Extract `X-User-LLM-Model` and `X-User-LLM-API-Key` headers |
| `tau2_agent/context.py` | Request context (contextvars) for passing LLM config through call stack |

### No Changes Required

| File | Reason |
|------|--------|
| `src/tau2/run.py` | Already accepts `llm_args_user` with API key |
| `src/tau2/utils/llm_utils.py` | litellm already supports per-call `api_key` |
| `src/tau2/data_model/simulation.py` | `RunConfig` already has `llm_args_user` field |

### Environment Variables

```bash
# Server-side configuration
TAU2_AGENT_MODEL=gemini-2.0-flash          # Model for tau2_agent orchestrator
GOOGLE_API_KEY=AIza...                      # API key for server's Gemini usage
SERVICE_API_KEYS=key1,key2                  # Optional: restrict service access
LOG_LEVEL=INFO
PORT=8001
```

### Docker Configuration

Use existing `tau2_agent/docker_setup/` with modifications:

```dockerfile
# Add to existing Dockerfile
ENV PORT=8001
ENV TAU2_AGENT_MODEL=gemini-2.0-flash

# Custom entrypoint enables BYOK middleware (see research.md DEC-006)
# ADK's adk api_server CLI doesn't expose middleware hooks
CMD ["python", "-m", "tau2_agent.server"]
```

## API Contract

**Versioning**: No version prefix initially (`/a2a/tau2_agent`). Future breaking changes will use `/v2/` prefix while maintaining `/a2a/` as v1 for backward compatibility.

### Request

```http
POST /a2a/tau2_agent
Content-Type: application/json
X-User-LLM-Model: gpt-4o
X-User-LLM-API-Key: sk-...

{
  "jsonrpc": "2.0",
  "id": "1",
  "method": "message/send",
  "params": {
    "message": {
      "role": "user",
      "parts": [{"text": "Run evaluation on mock domain for http://agent:8000/a2a/my_agent"}]
    }
  }
}
```

### Response

Standard A2A JSON-RPC response with evaluation results.

### Error Responses

| Status | Condition | Body |
|--------|-----------|------|
| 400 | Missing required header | `{"error": "Missing X-User-LLM-Model header"}` |
| 401 | Invalid service API key | `{"error": "Invalid authorization"}` |
| 401 | Invalid user LLM API key | `{"error": "User LLM authentication failed"}` |
| 500 | Internal error | `{"error": "Internal server error"}` |

## Deployment

### Initial Deployment

```bash
# Build and push
cd tau2_agent/docker_setup
gcloud builds submit --tag gcr.io/PROJECT_ID/tau2-agent

# Deploy
gcloud run deploy tau2-agent \
  --image gcr.io/PROJECT_ID/tau2-agent \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8001 \
  --set-env-vars "TAU2_AGENT_MODEL=gemini-2.0-flash" \
  --set-secrets "GOOGLE_API_KEY=google-api-key:latest"
```

### Secret Management

Store `GOOGLE_API_KEY` in Google Secret Manager, reference in Cloud Run deployment.

## Edge Cases & Failure Handling

### LLM Provider Failures

When the client's LLM provider (user simulator) encounters errors mid-evaluation:

| Failure Type | Behavior |
|--------------|----------|
| Rate limit (429) | Fail entire evaluation, return error |
| Timeout | Fail entire evaluation, return error |
| Authentication error | Fail entire evaluation, return 401 |
| Transient errors | Fail entire evaluation, return error |

**No partial results are returned.** If any task fails due to LLM provider issues, the entire evaluation fails. Rationale:
- Client controls execution volume and provider choice
- Partial results may produce misleading benchmark scores
- Simpler implementation without checkpoint/resume logic

Error response includes:
- Tasks completed before failure
- Failure reason and affected task index
- Suggestion to reduce `num_tasks` or check provider status

## Out of Scope

- Agent Under Evaluation hosting (external responsibility)
- Multi-tenant isolation (single shared service)
- Usage metering/billing (future feature)
- Rate limiting (future feature)
- Persistent evaluation history (uses existing file-based storage)
- **Full domain evaluations** (Cloud Run 60-min timeout constraint)
  - Retail (114 tasks) and Telecom (2,285 tasks) require local execution
  - Hosted service supports sampled evaluations (max 30 tasks)
- **Async evaluation pattern** (future enhancement for full domains)
- **Observability instrumentation** (metrics, tracing, structured logging) - deferred to 007-datadog-project

## Success Criteria

1. [ ] tau2_agent deployed to Cloud Run
2. [ ] Client can run evaluation by providing user LLM credentials in headers
3. [ ] Server uses Gemini for orchestration without client input
4. [ ] Missing headers return appropriate 400 errors
5. [ ] Invalid API keys return 401 errors
6. [ ] Evaluation results returned correctly via A2A protocol
   - Validation: Manual verification - run one mock domain evaluation and inspect response contains valid JSON-RPC structure with evaluation metrics
