# =============================================================================
# Stage 1: Builder - Build wheels for dependencies
# =============================================================================
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && rm -rf /var/lib/apt/lists/*

# Copy and build minimal runtime dependencies
COPY tau2_agent/docker_setup/requirements.txt ./
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# =============================================================================
# Stage 2: Runtime - Minimal production image
# =============================================================================
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TAU2_DATA_DIR=/app/data \
    PORT=8001 \
    LOG_LEVEL=INFO \
    TAU2_AGENT_MODEL=gemini-2.0-flash \
    PYTHONPATH=/app/agents:/app/src \
    AGENTS_DIR=/app/agents \
    # Datadog LLM Observability (agentless mode)
    DD_TRACE_ENABLED=true \
    DD_LLMOBS_ENABLED=true \
    DD_LLMOBS_AGENTLESS_ENABLED=true \
    DD_LLMOBS_ML_APP=tau2-bench-agent \
    DD_SERVICE=tau2-bench-agent \
    DD_ENV=dev \
    # Disable httpx distributed tracing to avoid async generator conflicts
    DD_HTTPX_DISTRIBUTED_TRACING=false \
    # Force Gemini API instead of Vertex AI (ADK auto-detects GCP environment)
    GOOGLE_GENAI_USE_VERTEXAI=FALSE

# Create non-root user
RUN groupadd --gid 1000 tau2agent && \
    useradd --uid 1000 --gid tau2agent --shell /bin/bash --create-home tau2agent

WORKDIR /app

# Install pre-built wheels
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels

# Copy application source
COPY --chown=tau2agent:tau2agent src/ ./src/
COPY --chown=tau2agent:tau2agent data/tau2/domains/ ./data/tau2/domains/
COPY --chown=tau2agent:tau2agent data/tau2/user_simulator/ ./data/tau2/user_simulator/
COPY --chown=tau2agent:tau2agent pyproject.toml README.md ./

# Create agents directory and copy tau2_agent directly into it
# (ADK validates symlinks don't resolve outside base directory)
RUN mkdir -p /app/agents
COPY --chown=tau2agent:tau2agent tau2_agent/ ./agents/tau2_agent/

# Create .adk directory inside agents for ADK artifact storage
# ADK looks for .adk relative to AGENTS_DIR
RUN mkdir -p /app/agents/.adk && chown tau2agent:tau2agent /app/agents/.adk

# Install tau2 package (no deps - already installed)
RUN pip install -e . --no-deps

# Create data directories with proper permissions
# - simulations/: tau2 run.py saves simulation results here
# - evaluations/: EvaluationStore completed evaluations
# - sessions/: EvaluationStore in-progress sessions
# - logs/: EvaluationStore structured event logs
# - tau2/evals/: Legacy evals directory for volume mount
RUN mkdir -p /app/data/simulations \
             /app/data/evaluations \
             /app/data/sessions \
             /app/data/logs \
             /app/data/tau2/evals && \
    chown -R tau2agent:tau2agent /app/data

USER tau2agent

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8001/a2a/tau2_agent/.well-known/agent-card.json', timeout=5).raise_for_status()" || exit 1

CMD ["python", "-m", "tau2_agent.server"]
