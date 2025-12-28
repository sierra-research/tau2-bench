# ADR: GCP Integration Architecture Decisions

## ADR-000: Synchronous Execution with Task Limits (Critical Constraint)

### Status
Accepted

### Context
Cloud Run has a **60-minute maximum request timeout**. Tau2 benchmark execution times vary significantly:

| Domain | Tasks | Avg Time/Task | Full Domain |
|--------|-------|---------------|-------------|
| Mock | 9 | ~40s | ~6 min |
| Airline | 50 | ~64s | ~53 min |
| Retail | 114 | ~69s | ~2-3 hours |
| Telecom | 2,285 | ~191s | ~20+ days |

**Problem**: Full domain evaluations for Retail and Telecom will timeout. Even Airline is borderline.

### Decision
**Enforce `num_tasks` limit (max 30) and `num_trials` limit (max 3) per request.**

Reject requests exceeding these limits with 400 Bad Request.

### Alternatives Considered

| Alternative | Pros | Cons |
|-------------|------|------|
| **Task limits (chosen)** | Simple, works now, predictable | Can't run full domains |
| Async job pattern | Full domains possible | Complex, requires job queue, polling API |
| Cloud Run Jobs | No timeout limit | Different deployment model, more complex |
| Compute Engine | No limits | Always-on costs, operational overhead |

### Rationale
For initial deployment, simplicity wins:
- Task sampling is statistically valid for benchmarking
- 30 tasks provides meaningful signal
- Full domain runs can be done locally
- Async pattern can be added later if needed

### Consequences
- **Supported**: Quick evaluations, sampled benchmarks (10-30 tasks)
- **Not Supported**: Full domain evaluations (must run locally)
- **Documentation**: Must clearly communicate this limitation
- **Future**: May add async pattern for full evaluations

---

## ADR-001: Cloud Run as Deployment Platform

### Status
Accepted

### Context
tau2_agent needs to be deployed as a hosted service on Google Cloud Platform. Options considered:
- Cloud Run (serverless containers)
- Cloud Functions (serverless functions)
- GKE (Kubernetes)
- Compute Engine (VMs)
- App Engine (PaaS)

### Decision
**Use Cloud Run** for deploying tau2_agent.

### Rationale

| Service | Pros | Cons |
|---------|------|------|
| **Cloud Run** | Scale to zero, pay-per-use, container-based (matches existing Docker setup), simple deployment, HTTPS by default | Cold starts (5-25s), 60min request timeout |
| Cloud Functions | Simpler for single functions | Not suitable for complex agent with multiple tools |
| GKE | Sub-second cold starts, full control | High base cost (~$70/month), operational overhead |
| Compute Engine | Always-on, no cold start | Manual scaling, always paying |
| App Engine | Managed | Less flexible than Cloud Run |

**Key factors:**
1. Existing Docker setup in `tau2_agent/docker_setup/` maps directly to Cloud Run
2. Variable traffic pattern (scale to zero when unused)
3. Low cost target ($5-30/month)
4. 300s evaluation timeout fits within Cloud Run's 60min limit

### Consequences
- Accept cold start latency (mitigate with minimum instances if needed)
- Use existing Dockerfile with minimal modifications

---

## ADR-002: Split LLM Cost Model (Orchestrator vs User Simulator)

### Status
Accepted

### Context
tau2-bench evaluations involve multiple LLM calls:
1. **tau2_agent (orchestrator)**: Interprets requests, orchestrates evaluation, analyzes results
2. **User simulator**: Generates synthetic user messages during evaluation (high token usage)
3. **Agent under evaluation**: The target being tested (external)

All LLM costs could be:
- Server-paid (simple, but expensive at scale)
- Client-paid via BYOK (complex, but scalable)
- Split based on responsibility

### Decision
**Split cost model:**
- tau2_agent orchestrator: Server pays (default Gemini)
- User simulator: Client pays (BYOK required)

### Rationale

| Component | Token Usage | Predictability | Who Benefits |
|-----------|-------------|----------------|--------------|
| tau2_agent | Low (orchestration) | High | Service operator |
| User simulator | High (generates all user turns) | Variable | Client running evaluation |

**User simulator is the primary cost driver** because:
- Runs once per simulated user turn
- Multiple turns per task, multiple tasks per evaluation
- Client controls evaluation volume

**tau2_agent orchestrator cost is manageable** because:
- Single call to interpret request
- Single call to format results
- Predictable, low volume

### Consequences
- Clients MUST provide `X-User-LLM-Model` and `X-User-LLM-API-Key` headers
- Server operator pays ~$5-20/month for orchestration (Gemini Flash is cheap)
- No surprise bills for server operator from runaway evaluations

---

## ADR-003: Gemini Developer API vs Vertex AI

### Status
Accepted (for initial deployment)

### Context
Google offers two ways to access Gemini models:
1. **Gemini Developer API** (ai.google.dev) - Simple API key auth
2. **Vertex AI** (cloud.google.com) - Enterprise features, IAM auth

### Decision
**Start with Gemini Developer API** for tau2_agent orchestrator.

### Rationale

| Aspect | Gemini Developer API | Vertex AI |
|--------|---------------------|-----------|
| Authentication | Simple API key | Service account + IAM |
| Pricing | Same base price | ~15% markup for enterprise |
| SLA | None | Yes |
| Compliance | Limited | HIPAA, SOC 2, etc. |
| Setup complexity | Low | Medium |

**For initial deployment:**
- Simple API key is sufficient
- No enterprise compliance requirements yet
- Lower cost

**Migration path to Vertex AI** when:
- Need SLA guarantees
- Enterprise customers require compliance certifications
- Volume warrants negotiated pricing

### Consequences
- Store `GOOGLE_API_KEY` in Secret Manager
- Use `gemini/gemini-2.0-flash` model string (litellm format)
- Document migration path to Vertex AI

---

## ADR-004: BYOK via HTTP Headers (not request body)

### Status
Accepted

### Context
Clients need to provide LLM credentials for user simulator. Options:
1. HTTP headers (`X-User-LLM-Model`, `X-User-LLM-API-Key`)
2. Request body parameters
3. Query parameters
4. Separate authentication endpoint

### Decision
**Use HTTP headers** for BYOK credentials.

### Rationale

| Method | Pros | Cons |
|--------|------|------|
| **Headers** | Separates auth from payload, works with any body format, easy to strip in logs | Limited header size |
| Body | Part of JSON-RPC payload | Mixes auth with business logic, harder to validate early |
| Query params | Simple | Visible in URLs/logs, security risk |
| Separate endpoint | Most secure | Complex, requires session management |

**Headers are ideal because:**
- A2A protocol uses JSON-RPC body - keep it clean
- Middleware can validate before parsing body
- Easy to sanitize from logs
- Standard pattern (like `Authorization` header)

### Consequences
- Create middleware to extract headers before ADK processes request
- Document header requirements in API contract
- Ensure headers are not logged

---

## ADR-005: No Service Authentication Initially

### Status
Accepted (with optional implementation path)

### Context
Should the tau2_agent service require authentication to access?

Options:
1. Open access (anyone can call)
2. Service API key (`Authorization: Bearer <key>`)
3. Google IAP (Identity-Aware Proxy)
4. OAuth2

### Decision
**Start with open access**, implement optional service API key.

### Rationale

**Open access acceptable because:**
- BYOK model means clients pay for their own LLM usage
- Server cost is predictable (~$5-20/month regardless of usage)
- No sensitive data stored
- Abuse limited by client's own API key costs

**Optional service API key for:**
- Restricting to known clients
- Basic access control
- Future billing/metering

### Consequences
- Default: `--allow-unauthenticated` on Cloud Run
- Optional: `SERVICE_API_KEYS` env var to enable authentication
- Future: Add proper auth if abuse becomes an issue

---

## ADR-006: Isolate Changes to tau2_agent (No Core tau2 Modifications)

### Status
Accepted

### Context
Initial design proposed modifying core tau2 files (`src/tau2/run.py`, `src/tau2/utils/llm_utils.py`) to support per-request API keys. This would create coupling between GCP deployment concerns and the core benchmark framework.

### Decision
**Keep all GCP integration changes isolated to `tau2_agent/`.**

tau2 already supports dynamic API keys via `RunConfig.llm_args_user`:

```python
# Already works in tau2_agent/tools/run_tau2_evaluation.py
config = RunConfig(
    llm_user=user_llm,
    llm_args_user={"api_key": user_api_key},  # Pass user's key
    ...
)
```

### Files Changed

| Scope | Files | Changes |
|-------|-------|---------|
| **tau2_agent/** | `agent.py` | Server Gemini model |
| **tau2_agent/** | `tools/run_tau2_evaluation.py` | Read API key from context |
| **tau2_agent/** | `middleware.py` (new) | Extract headers |
| **tau2_agent/** | `context.py` (new) | Request context |
| **src/tau2/** | None | No changes |

### Rationale
- **Separation of concerns**: GCP deployment is a `tau2_agent` concern, not core tau2
- **Maintainability**: Core tau2 can be updated independently
- **Already supported**: tau2's `RunConfig` already accepts `llm_args_user` with API key
- **Minimal changes**: Only 2 files modified, 2 files created

### Consequences
- Clean separation between benchmark framework and agent wrapper
- Future tau2 updates won't conflict with GCP integration
- All BYOK logic contained in one place (`tau2_agent/`)

---

## ADR-007: Request Context for Passing BYOK Credentials

### Status
Accepted

### Context
BYOK credentials extracted from headers need to reach the user simulator code deep in the call stack:

```
HTTP Request → Middleware → ADK → tau2_agent → run_tau2_evaluation → user simulator
```

Options:
1. Thread-local storage
2. Async context variables (`contextvars`)
3. Pass through function parameters
4. Global/environment variables

### Decision
**Use Python `contextvars`** for request context.

### Rationale

| Method | Pros | Cons |
|--------|------|------|
| **contextvars** | Async-safe, scoped to request, clean | Requires Python 3.7+ |
| Thread-local | Simple | Not async-safe |
| Parameters | Explicit | Requires modifying many function signatures |
| Globals | Easy | Not request-scoped, race conditions |

**contextvars is ideal because:**
- ADK/FastAPI is async - need async-safe context
- Automatically scoped to request lifecycle
- No need to modify intermediate function signatures
- Standard library (Python 3.7+)

### Consequences
- Create `tau2_agent/context.py` with context variable definitions
- Set context in middleware, read in user simulator code
- Document context variable usage for future maintainers
