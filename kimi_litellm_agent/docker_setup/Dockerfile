# =============================================================================
# Kimi LiteLLM Agent - ADK agent using Kimi K2 Thinking via Nebius TokenFactory
# =============================================================================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8002 \
    PYTHONPATH=/app/agents \
    AGENTS_DIR=/app/agents \
    # AgentBeats compatibility: serve agent card at root path
    AGENTBEATS_MODE=true \
    # Datadog LLM Observability (agentless mode)
    DD_LLMOBS_ENABLED=1 \
    DD_LLMOBS_AGENTLESS_ENABLED=1 \
    DD_LLMOBS_ML_APP=tau2-bench-agent \
    DD_SERVICE=kimi-litellm-agent \
    DD_ENV=dev

WORKDIR /app

# Install curl for agentbeats health checks
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Install ADK, LiteLLM, and dependencies
RUN pip install --no-cache-dir \
    google-adk[a2a]>=1.18.0 \
    litellm>=1.65.0 \
    httpx>=0.28.0 \
    uvicorn>=0.34.0 \
    loguru>=0.7.3 \
    ddtrace>=2.0.0

# Create agents directory and copy agent source directly into it
# (ADK validates symlinks don't resolve outside base directory)
RUN mkdir -p /app/agents
COPY kimi_litellm_agent/ ./agents/kimi_litellm_agent/

# Copy shared utilities (card_url_utils, middleware, server_utils)
COPY shared_utils/ ./agents/shared_utils/

# Create .adk directory for ADK artifact storage
RUN mkdir -p /app/.adk

EXPOSE 8002

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8002/.well-known/agent-card.json', timeout=5).raise_for_status()" || exit 1

# AgentBeats-compatible entrypoint: accepts --host, --port, --card-url args
# CARD_URL env var is read by server.py to update agent.json URL at startup
ENTRYPOINT ["python", "-m", "kimi_litellm_agent.server"]
CMD ["--host", "0.0.0.0"]
