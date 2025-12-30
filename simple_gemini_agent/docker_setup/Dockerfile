# =============================================================================
# Simple Gemini Agent - Minimal ADK agent for tau2-bench evaluation
# =============================================================================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8001 \
    PYTHONPATH=/app/agents \
    AGENTS_DIR=/app/agents \
    # Datadog LLM Observability (agentless mode)
    DD_LLMOBS_ENABLED=1 \
    DD_LLMOBS_AGENTLESS_ENABLED=1 \
    DD_LLMOBS_ML_APP=tau2-bench-agent \
    DD_SERVICE=simple-gemini-agent \
    DD_ENV=dev

WORKDIR /app

# Install ADK and dependencies (match tau2_agent versions)
RUN pip install --no-cache-dir \
    google-adk[a2a]>=1.18.0 \
    google-genai>=1.0.0 \
    httpx>=0.28.0 \
    uvicorn>=0.34.0 \
    loguru>=0.7.3 \
    ddtrace>=2.0.0

# Create agents directory and copy agent source directly into it
# (ADK validates symlinks don't resolve outside base directory)
RUN mkdir -p /app/agents
COPY simple_gemini_agent/ ./agents/simple_gemini_agent/

# Create .adk directory for ADK artifact storage
RUN mkdir -p /app/.adk

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8001/a2a/simple_gemini_agent/.well-known/agent-card.json', timeout=5).raise_for_status()" || exit 1

# Run server using get_fast_api_app (like tau2_agent)
CMD ["python", "-m", "simple_gemini_agent.server"]
