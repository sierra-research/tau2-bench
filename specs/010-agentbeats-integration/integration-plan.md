# AgentBeats Integration Plan

## Overview

This document outlines the integration plan for deploying tau2-bench as an AgentBeats-compatible evaluation service. It references the detailed technical analysis in [integration-analysis.md](./integration-analysis.md).

## Background

### Analysis Summary

From [integration-analysis.md](./integration-analysis.md#functional-equivalence-verification):

> Both implementations are functionally equivalent in their isolation mechanism. The primary difference is the mechanism, not the isolation model.

**Key Finding**: The current `tau2_agent` with `run_tau2_evaluation` tool provides the same evaluation isolation as the gym-based `agentified-tau-bench` implementation. No modifications to core evaluation logic are required.

### Agent Roles

| AgentBeats Role | Our Implementation | Purpose |
|-----------------|-------------------|---------|
| Green Agent | `tau2_agent` | Assessment manager - orchestrates evaluations |
| Purple Agent | `kimi_litellm_agent` | Target agent being evaluated |

---

## Integration Components

### 1. Container Orchestration

**Purpose**: Deploy both agents as containerized services that can communicate via A2A protocol.

**Components**:
- Docker Compose configuration for multi-agent deployment
- Health checks for agent readiness
- Shared network for inter-agent communication
- Environment variable configuration for credentials

**Reference Services**:
- `tau2_agent` on port 8001 (green agent)
- `kimi_litellm_agent` on port 8002 (purple agent)

### 2. Evaluation Trigger

**Purpose**: External script/service to initiate evaluations after agents are running.

**Responsibilities**:
- Wait for agent readiness via health checks
- Send evaluation request to green agent with purple agent endpoint
- Pass user simulator credentials via headers
- Collect and report results

**Credential Flow** (from [integration-analysis.md](./integration-analysis.md#credential-handling)):
```
Environment Variables → Evaluation Script → X-User-LLM-* Headers → tau2_agent middleware
```

### 3. A2A Communication Utilities

**Purpose**: Reusable utilities for agent communication and health checking.

**Functions**:
- Agent card discovery
- Health/readiness polling
- A2A message sending with custom headers

**Reference**: `experiments/agentify_tau_bench/utils/a2a_utils.py`

---

## Integration Points

### tau2_agent (Green Agent)

**Existing Capabilities** (no modifications needed):
- A2A protocol support via ADK
- `run_tau2_evaluation` tool for evaluation orchestration
- Credential middleware for `X-User-LLM-*` headers
- Context ID handling via `A2AAgent` (see [analysis](./integration-analysis.md#a2a-run_domain-approach))

**Agent Card Endpoint**:
```
GET /a2a/tau2_agent/.well-known/agent-card.json
```

### kimi_litellm_agent (Purple Agent)

**Existing Capabilities** (no modifications needed):
- A2A protocol support via ADK
- Context ID handling via ADK framework
- LiteLLM integration for Nebius TokenFactory

**Agent Card Endpoint**:
```
GET /a2a/kimi_litellm_agent/.well-known/agent-card.json
```

### Evaluation Flow

```
1. Container Startup
   ├─ tau2-agent starts on port 8001
   └─ kimi-litellm-agent starts on port 8002

2. Health Check
   ├─ Poll tau2-agent agent card endpoint
   └─ Poll kimi-litellm-agent agent card endpoint

3. Evaluation Request
   ├─ Send A2A message to tau2-agent
   ├─ Include purple agent URL in message
   └─ Include credentials in X-User-LLM-* headers

4. Evaluation Execution
   ├─ tau2_agent invokes run_tau2_evaluation tool
   ├─ A2AAgent communicates with kimi_litellm_agent
   └─ Results returned via A2A response
```

---

## Configuration

### Environment Variables

**Green Agent (tau2_agent)**:
| Variable | Required | Description |
|----------|----------|-------------|
| `PORT` | Yes | Server port (default: 8001) |
| `GOOGLE_API_KEY` | Yes | For user simulator LLM |
| `DD_TRACE_ENABLED` | No | Datadog tracing (default: false) |

**Purple Agent (kimi_litellm_agent)**:
| Variable | Required | Description |
|----------|----------|-------------|
| `PORT` | Yes | Server port (default: 8002) |
| `NEBIUS_API_KEY` | Yes | For Kimi K2 model |

**Evaluation Trigger**:
| Variable | Required | Description |
|----------|----------|-------------|
| `GREEN_AGENT_URL` | No | Green agent endpoint (default: http://localhost:8001/a2a/tau2_agent) |
| `PURPLE_AGENT_URL` | No | Purple agent endpoint (default: http://localhost:8002/a2a/kimi_litellm_agent) |
| `USER_LLM_MODEL` | No | User simulator model (default: gemini-2.0-flash) |
| `GOOGLE_API_KEY` | Yes | Passed to green agent via headers |

---

## Files to Create

| File | Purpose |
|------|---------|
| `docker-compose.agentbeats.yaml` | Container orchestration for both agents |
| `scripts/agentbeats_evaluate.py` | Evaluation trigger script |
| `src/tau2/agentbeats/__init__.py` | Module initialization |
| `src/tau2/agentbeats/a2a_utils.py` | A2A communication utilities |

## Existing Files (No Modifications)

| File | Role |
|------|------|
| `tau2_agent/server.py` | Green agent server |
| `tau2_agent/tools/run_tau2_evaluation.py` | Evaluation tool |
| `tau2_agent/middleware.py` | Credential handling |
| `kimi_litellm_agent/server.py` | Purple agent server |
| `src/tau2/agent/a2a_agent.py` | A2A protocol adapter |

---

## Usage

### Local Development

```bash
# Start agents
docker compose -f docker-compose.agentbeats.yaml up -d

# Verify readiness
docker compose -f docker-compose.agentbeats.yaml ps

# Run evaluation
GOOGLE_API_KEY=<key> python scripts/agentbeats_evaluate.py --domain mock

# Stop agents
docker compose -f docker-compose.agentbeats.yaml down
```

### AgentBeats Competition

For competition deployment, the purple agent URL will be provided externally. The green agent (tau2_agent) can evaluate any A2A-compliant agent:

```bash
PURPLE_AGENT_URL=<competition-agent-url> python scripts/agentbeats_evaluate.py
```

---

## References

- [Integration Analysis](./integration-analysis.md) - Detailed technical analysis
- [A2A Integration Spec](../001-a2a-integration/spec.md) - A2A protocol specification
- [Experimental Implementation](../../src/experiments/agentify_tau_bench/README.md) - Reference implementation
