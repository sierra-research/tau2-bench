# Research: 008-GCP Integration

**Feature**: tau2_agent Cloud Run Deployment with BYOK
**Date**: 2025-12-26
**Branch**: 008-gcp-integration

## Dependencies

### Runtime Dependencies

| Package | Version | Purpose | Registry |
|---------|---------|---------|----------|
| google-cloud-secret-manager | >=2.18.0 | GCP Secret Manager access | [PyPI](https://pypi.org/project/google-cloud-secret-manager/) |
| litellm | >=1.65.0 | Multi-provider LLM abstraction | [PyPI](https://pypi.org/project/litellm/) |
| httpx | >=0.28.0 | Async HTTP client | [PyPI](https://pypi.org/project/httpx/) |
| google-adk[a2a] | >=1.18.0 | Google ADK with A2A support | [PyPI](https://pypi.org/project/google-adk/) |
| pydantic | >=2.0.0 | Data validation | Already in pyproject.toml |

### Development Dependencies

| Package | Version | Purpose | Registry |
|---------|---------|---------|----------|
| pytest | >=8.3.5 | Testing framework | Already in pyproject.toml |
| pytest-asyncio | >=0.24.0 | Async test support | Already in pyproject.toml |

### Version Constraints

- **google-cloud-secret-manager**: Latest is 2.26.0 (2025-12-18). Supports Python 3.7-3.14. Use >=2.18.0 for stability.
- **litellm**: Already at >=1.65.0 in pyproject.toml. Gemini support stable since 1.40+.
- **google-adk**: Already at >=1.18.0. `get_fast_api_app()` pattern available since 1.15.0.

---

## Decision Registry

### DEC-001: ADK Middleware Integration Pattern

**Decision**: Create custom Python entrypoint that imports `get_fast_api_app()` from ADK and adds BYOK middleware before running uvicorn.

**Pattern**: `app.add_middleware(BYOKMiddleware)`

**Verify In**: `tau2_agent/server.py` or custom entrypoint file

**Rationale**: ADK's `adk api_server` command doesn't expose middleware hooks. However, the underlying `get_fast_api_app()` function returns a standard FastAPI app that can be modified before running. This is the documented pattern for customizing ADK servers.

**Alternatives Rejected**:
- **Modify ADK source**: Violates maintainability; not our code
- **Request body injection**: Mixes auth with business logic; violates ADR-004
- **Separate proxy server**: Adds latency and complexity

**Verification Points**:
- [ ] Custom server.py imports `get_fast_api_app()`
- [ ] Middleware added via `app.add_middleware()`
- [ ] Headers extracted before A2A handler processes request

---

### DEC-002: Request Context via Python contextvars

**Decision**: Use Python `contextvars` module to pass BYOK credentials from middleware to user simulator code deep in call stack.

**Pattern**:
```python
from contextvars import ContextVar
user_llm_model: ContextVar[str | None] = ContextVar('user_llm_model', default=None)
user_llm_api_key: ContextVar[str | None] = ContextVar('user_llm_api_key', default=None)
```

**Verify In**: `tau2_agent/context.py`

**Rationale**: Per ADR-007, contextvars is async-safe, request-scoped, and doesn't require modifying intermediate function signatures. This matches FastAPI/Starlette's async model.

**Alternatives Rejected**:
- **Thread-local storage**: Not async-safe; could cause race conditions
- **Parameter passing**: Requires modifying many function signatures in call chain
- **Global variables**: Not request-scoped; race conditions

**Verification Points**:
- [ ] ContextVars declared at module level in `tau2_agent/context.py`
- [ ] Middleware sets context via `token = var.set(value)`
- [ ] Context reset in finally block: `var.reset(token)`
- [ ] `run_tau2_evaluation.py` reads from context instead of environment

---

### DEC-003: BYOK Header Extraction and Validation

**Decision**: Use Starlette's `BaseHTTPMiddleware` to extract `X-User-LLM-Model` and `X-User-LLM-API-Key` headers, validate presence, and return 400 if missing.

**Pattern**:
```python
class BYOKMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        model = request.headers.get("x-user-llm-model")
        api_key = request.headers.get("x-user-llm-api-key")
        if not model or not api_key:
            return JSONResponse(status_code=400, content={"error": "Missing BYOK headers"})
        # Set context and continue...
```

**Verify In**: `tau2_agent/middleware.py`

**Rationale**: FastAPI/Starlette middleware pattern is well-documented and integrates seamlessly with ADK's FastAPI server. Headers are case-insensitive per HTTP spec.

**Alternatives Rejected**:
- **FastAPI dependencies**: Would require modifying ADK's route handlers
- **A2A context_builder**: Only applies to A2A SDK, not full request flow

**Verification Points**:
- [ ] Middleware class inherits `BaseHTTPMiddleware`
- [ ] Headers accessed via `request.headers.get()` (case-insensitive)
- [ ] 400 response for missing headers before hitting handler
- [ ] API keys never logged (use loguru with sanitization)

---

### DEC-004: Server-Configured Gemini Model

**Decision**: Use environment variable `TAU2_AGENT_MODEL` (default: `gemini-2.0-flash`) with `GOOGLE_API_KEY` from Secret Manager for tau2_agent orchestrator LLM.

**Pattern**:
```python
model = os.getenv("TAU2_AGENT_MODEL", "gemini-2.0-flash")
# litellm uses GEMINI_API_KEY env var
```

**Verify In**: `tau2_agent/agent.py` (modify `create_model()` function)

**Rationale**: Per ADR-003, Gemini Developer API is simpler than Vertex AI for initial deployment. LiteLLM automatically reads `GEMINI_API_KEY` or accepts `api_key` parameter.

**Alternatives Rejected**:
- **Vertex AI**: More complex auth (IAM), ~15% cost markup
- **Hard-coded model**: Not configurable for different deployments

**Verification Points**:
- [ ] `create_model()` reads `TAU2_AGENT_MODEL` env var
- [ ] Falls back to `gemini-2.0-flash` if not set
- [ ] Removes Nebius-specific logic (simplified for GCP)
- [ ] `GOOGLE_API_KEY` used by litellm for Gemini API

---

### DEC-005: Task Limit Validation

**Decision**: Validate `num_tasks <= 30` and `num_trials <= 3` in `run_tau2_evaluation.py`, return 400 error if exceeded.

**Pattern**:
```python
MAX_TASKS = 30
MAX_TRIALS = 3
if num_tasks and num_tasks > MAX_TASKS:
    raise ValueError(f"num_tasks must be <= {MAX_TASKS}")
```

**Verify In**: `tau2_agent/tools/run_tau2_evaluation.py`

**Rationale**: Per ADR-000, Cloud Run has 60-minute timeout. With ~2 minutes per task, 30 tasks provides safe margin (~30-40 min execution).

**Alternatives Rejected**:
- **No limits**: Would cause timeout failures for Retail/Telecom domains
- **Async job pattern**: Too complex for initial deployment
- **Cloud Run Jobs**: Different deployment model

**Verification Points**:
- [ ] `MAX_TASKS = 30` constant defined
- [ ] `MAX_TRIALS = 3` constant defined
- [ ] Validation happens before evaluation starts
- [ ] Error message explains limit and rationale

---

### DEC-006: Custom Server Entrypoint

**Decision**: Create `tau2_agent/server.py` that composes the FastAPI app with middleware instead of using `adk api_server` directly.

**Pattern**:
```python
# tau2_agent/server.py
from google.adk.cli.fast_api import get_fast_api_app
from tau2_agent.middleware import BYOKMiddleware

app = get_fast_api_app(agents_dir=".", web=False)
app.add_middleware(BYOKMiddleware)

# Entry point for Docker
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8001)))
```

**Verify In**: `tau2_agent/server.py`

**Rationale**: ADK's `adk api_server` CLI doesn't support custom middleware. Creating our own entrypoint gives full control while still using ADK's FastAPI app construction.

**Alternatives Rejected**:
- **Fork ADK**: Maintenance burden; violates separation
- **Wrap with nginx**: Adds latency; more moving parts
- **Patch at runtime**: Fragile; could break on ADK updates

**Verification Points**:
- [ ] `server.py` imports `get_fast_api_app` from ADK
- [ ] Middleware registered before app runs
- [ ] Dockerfile CMD points to `python -m tau2_agent.server`
- [ ] PORT environment variable respected for Cloud Run

---

### DEC-007: Cloud Run Service Configuration

**Decision**: Deploy with Cloud Run managed platform using configuration from gcp-integration-guide.md: 2Gi memory, 2 CPU, 60-min timeout, 0-10 instances.

**Pattern**: YAML service definition or gcloud CLI flags

**Verify In**: `tau2_agent/docker_setup/service.yaml` or deploy script

**Rationale**: Per spec NFR-1 and NFR-2:
- Scale to zero for cost optimization ($1-10/month compute target)
- 60-min timeout for long evaluations
- 2Gi memory for LLM operations

**Alternatives Rejected**:
- **GKE**: $70+/month base cost; operational overhead
- **Cloud Functions**: Not suitable for long-running operations
- **App Engine**: Less flexible scaling

**Verification Points**:
- [ ] `--timeout 3600` (60 minutes)
- [ ] `--memory 2Gi --cpu 2`
- [ ] `--min-instances 0 --max-instances 10`
- [ ] `--allow-unauthenticated` (BYOK model handles costs)
- [ ] `--set-secrets GOOGLE_API_KEY=google-api-key:latest`

---

### DEC-008: Secret Manager Integration

**Decision**: Store `GOOGLE_API_KEY` in GCP Secret Manager, inject as environment variable at Cloud Run deployment time.

**Pattern**: `--set-secrets "GOOGLE_API_KEY=google-api-key:latest"`

**Verify In**: Deploy script and/or service.yaml

**Rationale**: Per gcp-integration-guide.md, Secret Manager provides:
- Secure storage with IAM access control
- Automatic injection into Cloud Run environment
- Version management for key rotation

**Alternatives Rejected**:
- **Hard-coded in image**: Security risk; can't rotate
- **Environment variable in gcloud deploy**: Visible in CLI history
- **Secret file mount**: More complex for single key

**Verification Points**:
- [ ] Secret created: `gcloud secrets create google-api-key`
- [ ] Service account has `secretmanager.secretAccessor` role
- [ ] Cloud Run deployment uses `--set-secrets` flag
- [ ] Local development uses `.env` file (not committed)

---

## Research Notes

### ADK FastAPI Integration

The ADK `get_fast_api_app()` function from `google.adk.cli.fast_api` returns a fully constructed FastAPI application. Key observations:

1. **Already adds CORSMiddleware** when `allow_origins` specified - proving middleware can be added
2. **Returns standard FastAPI app** - all FastAPI patterns apply
3. **Supports web=False** for API-only mode (no UI)
4. **Uses agents_dir** to discover agents - we point to `tau2_agent/` directory

### Context Variable Lifecycle

For async request handling with contextvars:

```python
# Middleware
token_model = user_llm_model.set(model_value)
token_key = user_llm_api_key.set(key_value)
try:
    response = await call_next(request)
    return response
finally:
    user_llm_model.reset(token_model)
    user_llm_api_key.reset(token_key)
```

**Critical**: Always reset in finally block to prevent memory leaks between requests.

### LiteLLM Gemini Model Strings

Per gcp-integration-guide.md:

| API | Model String | Env Var |
|-----|--------------|---------|
| Gemini Developer API | `gemini/gemini-2.0-flash` | `GEMINI_API_KEY` |
| Vertex AI | `vertex_ai/gemini-2.0-flash` | ADC |

For tau2_agent orchestrator, use Gemini Developer API with `GOOGLE_API_KEY` mapped to `GEMINI_API_KEY`.

---

## Open Items

All NEEDS CLARIFICATION items from Technical Context have been resolved:

| Item | Resolution |
|------|------------|
| How to add middleware to ADK | Use `get_fast_api_app()` + custom `server.py` |
| Context passing pattern | Python `contextvars` module |
| Secret storage | GCP Secret Manager with `--set-secrets` injection |
| Gemini model configuration | `GEMINI_API_KEY` env var for litellm |
| Task limit enforcement | Validate in tool before evaluation starts |

---

## References

- [ADK FastAPI Source](https://github.com/google/adk-python) - `get_fast_api_app()` pattern
- [Python contextvars docs](https://docs.python.org/3/library/contextvars.html)
- [FastAPI Middleware](https://fastapi.tiangolo.com/tutorial/middleware/)
- [LiteLLM Gemini Integration](https://docs.litellm.ai/docs/providers/gemini)
- [Cloud Run Deployment](https://cloud.google.com/run/docs)
- [Secret Manager Best Practices](https://cloud.google.com/secret-manager/docs/best-practices)
