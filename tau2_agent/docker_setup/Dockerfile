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
    TAU2_AGENT_MODEL=gemini-2.0-flash

# Create non-root user
RUN groupadd --gid 1000 tau2agent && \
    useradd --uid 1000 --gid tau2agent --shell /bin/bash --create-home tau2agent

WORKDIR /app

# Install pre-built wheels
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels

# Copy application source
COPY --chown=tau2agent:tau2agent src/ ./src/
COPY --chown=tau2agent:tau2agent tau2_agent/ ./tau2_agent/
COPY --chown=tau2agent:tau2agent data/tau2/domains/ ./data/tau2/domains/
COPY --chown=tau2agent:tau2agent data/tau2/user_simulator/ ./data/tau2/user_simulator/
COPY --chown=tau2agent:tau2agent pyproject.toml README.md ./

# Install tau2 package (no deps - already installed)
RUN pip install -e . --no-deps

# Create evals directory for volume mount
RUN mkdir -p /app/data/tau2/evals && chown tau2agent:tau2agent /app/data/tau2/evals

# Create .adk directory for ADK artifact storage
RUN mkdir -p /app/.adk && chown tau2agent:tau2agent /app/.adk

# Create agents/ directory with symlink to tau2_agent
# This ensures /list-apps only returns valid agents (not data/, src/, etc.)
RUN mkdir -p /app/agents && \
    ln -s ../tau2_agent /app/agents/tau2_agent && \
    chown -h tau2agent:tau2agent /app/agents /app/agents/tau2_agent

USER tau2agent

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8001/a2a/tau2_agent/.well-known/agent-card.json', timeout=5).raise_for_status()" || exit 1

CMD ["python", "-m", "tau2_agent.server"]
