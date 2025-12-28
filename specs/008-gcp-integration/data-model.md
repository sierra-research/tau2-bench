# Data Model: 008-GCP Integration

**Feature**: tau2_agent Cloud Run Deployment with BYOK
**Date**: 2025-12-26

## Entities

### BYOKContext

Request-scoped context containing client-provided LLM credentials.

```python
@dataclass
class BYOKContext:
    """BYOK credentials extracted from request headers."""

    user_llm_model: str
    """LiteLLM model identifier (e.g., 'gpt-4o', 'claude-3-5-sonnet-20241022')"""

    user_llm_api_key: str
    """API key for the user's LLM provider (never logged)"""

    request_id: str | None = None
    """Optional correlation ID for tracing"""
```

**Storage**: In-memory via Python `contextvars` (request-scoped, not persisted)

### EvaluationLimits

Validation constraints for GCP deployment.

```python
@dataclass
class EvaluationLimits:
    """Limits enforced for Cloud Run deployment."""

    MAX_TASKS: ClassVar[int] = 30
    """Maximum tasks per evaluation (Cloud Run 60-min timeout)"""

    MAX_TRIALS: ClassVar[int] = 3
    """Maximum trials per task"""

    TIMEOUT_SECONDS: ClassVar[int] = 3600
    """Cloud Run request timeout (60 minutes)"""
```

**Storage**: Constants in `tau2_agent/config.py`

### ServerConfig

Server-side configuration from environment.

```python
@dataclass
class ServerConfig:
    """Server configuration loaded from environment."""

    tau2_agent_model: str = "gemini-2.0-flash"
    """Model for tau2_agent orchestrator LLM"""

    google_api_key: str | None = None
    """Gemini API key (from Secret Manager in production)"""

    port: int = 8001
    """Server port (Cloud Run sets PORT env var)"""

    log_level: str = "INFO"
    """Logging verbosity"""

    service_api_keys: list[str] | None = None
    """Optional: comma-separated keys for service access control"""
```

**Storage**: Environment variables, injected by Cloud Run

---

## Validation Rules

### Header Validation (Middleware)

| Field | Rule | Error Response |
|-------|------|----------------|
| `X-User-LLM-Model` | Required, non-empty string | 400: `{"error": "Missing X-User-LLM-Model header"}` |
| `X-User-LLM-API-Key` | Required, non-empty string | 400: `{"error": "Missing X-User-LLM-API-Key header"}` |

### Evaluation Parameter Validation (Tool)

| Field | Rule | Error Response |
|-------|------|----------------|
| `num_tasks` | Optional, 1-30 if provided | 400: `{"error": "num_tasks must be between 1 and 30"}` |
| `num_trials` | Optional, 1-3 if provided | 400: `{"error": "num_trials must be between 1 and 3"}` |
| `domain` | Required, must be valid domain | 400: `{"error": "Invalid domain: {domain}. Must be one of [...]"}` |
| `agent_endpoint` | Required, valid URL format | 400: `{"error": "Invalid agent_endpoint URL"}` |

### LLM Credential Validation (Runtime)

| Condition | Response |
|-----------|----------|
| LLM call fails with auth error | 401: `{"error": "User LLM authentication failed"}` |
| LLM call times out | 504: `{"error": "LLM request timeout"}` |
| LLM call rate limited | 429: `{"error": "LLM rate limited, retry later"}` |

---

## State Transitions

### Request Lifecycle

```
┌─────────────┐
│  Received   │ ← HTTP request arrives
└──────┬──────┘
       │
       ▼
┌─────────────┐     Missing Headers
│  Validate   │─────────────────────→ 400 Bad Request
│   Headers   │
└──────┬──────┘
       │ Valid
       ▼
┌─────────────┐     Invalid Service Key
│  Auth Check │─────────────────────→ 401 Unauthorized
│ (optional)  │
└──────┬──────┘
       │ Valid
       ▼
┌─────────────┐
│ Set Context │ ← Store BYOK in contextvars
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ ADK Handler │ ← A2A protocol processing
└──────┬──────┘
       │
       ▼
┌─────────────┐     Exceeds Limits
│  Validate   │─────────────────────→ 400 Bad Request
│   Params    │
└──────┬──────┘
       │ Valid
       ▼
┌─────────────┐     LLM Auth Failed
│  Execute    │─────────────────────→ 401 Unauthorized
│ Evaluation  │
└──────┬──────┘
       │ Success
       ▼
┌─────────────┐
│  Response   │ ← A2A JSON-RPC response
└─────────────┘
```

### Context Variable Lifecycle

```
Request Start
     │
     ▼
┌─────────────────────┐
│ token = var.set(val)│ ← Middleware sets context
└──────────┬──────────┘
           │
           ▼
    ┌──────────────┐
    │   Handler    │
    │  Execution   │ ← Context readable anywhere in call stack
    └──────┬───────┘
           │
           ▼
┌─────────────────────┐
│   var.reset(token)  │ ← Middleware cleans up (finally block)
└─────────────────────┘
           │
           ▼
    Request End
```

---

## Entity Relationships

```
┌─────────────────────────────────────────────────────────────────┐
│                         HTTP Request                             │
│  Headers:                                                        │
│    X-User-LLM-Model ────────┐                                   │
│    X-User-LLM-API-Key ──────┼──→ BYOKContext                    │
│    Authorization (opt) ─────┘      │                            │
└─────────────────────────────────────┼───────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                      BYOK Middleware                             │
│  1. Extract headers                                              │
│  2. Validate presence                                            │
│  3. Set contextvars                                              │
│  4. Continue to handler                                          │
│  5. Reset contextvars (finally)                                  │
└─────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                   run_tau2_evaluation Tool                       │
│  Reads: BYOKContext from contextvars                            │
│  Applies: EvaluationLimits validation                           │
│  Uses: ServerConfig for orchestrator LLM                        │
│  Passes: user_llm_api_key to RunConfig.llm_args_user            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Configuration Sources

| Config Item | Source | Default | Env Var |
|-------------|--------|---------|---------|
| Orchestrator Model | Server env | `gemini-2.0-flash` | `TAU2_AGENT_MODEL` |
| Orchestrator API Key | Secret Manager | - | `GOOGLE_API_KEY` |
| Server Port | Cloud Run | `8001` | `PORT` |
| Log Level | Server env | `INFO` | `LOG_LEVEL` |
| Service API Keys | Server env | None (open) | `SERVICE_API_KEYS` |
| User LLM Model | Request header | - | `X-User-LLM-Model` |
| User LLM API Key | Request header | - | `X-User-LLM-API-Key` |

---

## Security Constraints

1. **API Keys Never Logged**: All logging must sanitize `user_llm_api_key` and `google_api_key`
2. **Headers Not Echoed**: Error responses must not include header values
3. **Context Cleanup**: contextvars must be reset in finally blocks to prevent leakage
4. **HTTPS Only**: Cloud Run enforces HTTPS by default
5. **Non-Root Container**: Dockerfile uses `tau2agent` user (UID 1000)
