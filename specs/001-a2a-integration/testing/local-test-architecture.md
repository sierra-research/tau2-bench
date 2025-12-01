# Local A2A Test Architecture

## Overview

This document describes the architecture for local testing of A2A agents using a simple example that wraps the Nebius API with meta-llama/Meta-Llama-3.1-8B-Instruct.

## Purpose

Provide a minimal, easy-to-understand example of:
1. Creating an ADK agent that wraps an LLM API
2. Exposing the agent via A2A protocol on localhost
3. Evaluating the agent using tau2-bench's A2A client
4. Running the complete flow in a single command

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                       Local Test Environment                      │
│                                                                   │
│  ┌──────────────────┐         ┌──────────────────────────────┐ │
│  │  tau2-bench      │         │  Simple Nebius ADK Agent     │ │
│  │  Test Runner     │         │  (localhost:8001)            │ │
│  │                  │         │                              │ │
│  │  ┌────────────┐  │  HTTP   │  ┌────────────────────────┐ │ │
│  │  │  A2AAgent  │──┼────────>│  │  A2A HTTP Server      │ │ │
│  │  │  (Client)  │  │  A2A    │  │  (adk web --a2a)      │ │ │
│  │  └────────────┘  │ Protocol│  └────────────────────────┘ │ │
│  │                  │         │            │                 │ │
│  │  ┌────────────┐  │         │            v                 │ │
│  │  │  Domain    │  │         │  ┌────────────────────────┐ │ │
│  │  │  Simulator │  │         │  │  LlmAgent (ADK)       │ │ │
│  │  └────────────┘  │         │  │  + Nebius config      │ │ │
│  │                  │         │  └────────────────────────┘ │ │
│  └──────────────────┘         │            │                 │ │
│                               │            v                 │ │
└───────────────────────────────┼────────────────────────────────┘
                                │
                                v
                    ┌────────────────────────┐
                    │   Nebius API           │
                    │   meta-llama/          │
                    │   Meta-Llama-3.1-8B    │
                    └────────────────────────┘
```

## Components

### 1. Simple Nebius ADK Agent

**Location**: `simple_nebius_agent/agent.py`

**Responsibilities**:
- Wrap Nebius Llama 3.1 8B API with ADK's LlmAgent
- Provide minimal agent configuration (no tools)
- Convert to A2A protocol using `.to_a2a()`
- Serve on localhost:8001

**Key Features**:
- Single file implementation (~50 lines)
- Uses LiteLLM for Nebius API compatibility
- No external tools - conversation only
- Easy to understand and modify

### 2. Test Scripts

#### `run_simple_agent.sh`
**Purpose**: Start the simple agent server

**Behavior**:
- Checks for NEBIUS_API_KEY environment variable
- Starts ADK server on port 8001
- Enables A2A protocol
- Provides health check endpoint
- Logs to console for debugging

#### `test_simple_agent.sh`
**Purpose**: Run tau2-bench evaluation against the agent

**Behavior**:
- Waits for agent to be ready (health check)
- Runs tau2-bench with A2A agent type
- Uses mock domain for quick testing
- Reports results and metrics

#### `test_simple_agent_e2e.py`
**Purpose**: Automated pytest-based E2E test

**Behavior**:
- Manages agent lifecycle (startup/shutdown)
- Runs multiple evaluation scenarios
- Validates metrics and results
- Cleans up on failure

### 3. Test Configuration

**Location**: `tests/test_simple_local/`

**Components**:
- `conftest.py`: Pytest fixtures
  - `simple_agent_server`: Manages agent lifecycle
  - `wait_for_agent`: Health check polling
  - `nebius_api_configured`: Skip if no API key
- `test_simple_local.py`: Test cases
  - Test agent discovery
  - Test single-turn conversation
  - Test multi-turn conversation
  - Test error handling

## Port Allocation

- **8000**: Main tau2_agent (from tau2_agent/)
- **8001**: Simple test agent (this architecture)
- **8002+**: Available for additional test agents

## Prerequisites

1. **Python Environment**:
   - Python 3.10+
   - Dependencies installed: `pip install -e .`

2. **API Keys**:
   - `NEBIUS_API_KEY`: Required for Nebius API access
   - `NEBIUS_API_BASE`: Optional (defaults to Nebius endpoint)

3. **Network**:
   - Port 8001 available
   - Internet access for Nebius API calls

## Setup Steps

### Quick Start (Single Command)

```bash
# 1. Set up environment
export NEBIUS_API_KEY="your-api-key"

# 2. Run the test
./specs/001-a2a-integration/scripts/test_simple_agent.sh
```

This will:
1. Start the simple agent on localhost:8001
2. Wait for it to be ready
3. Run a tau2-bench evaluation
4. Display results
5. Clean up

### Manual Setup (Step by Step)

```bash
# 1. Set up environment
export NEBIUS_API_KEY="your-api-key"
export NEBIUS_API_BASE="https://api.tokenfactory.nebius.com/v1/"  # Optional

# 2. Start the agent
./specs/001-a2a-integration/scripts/run_simple_agent.sh

# 3. In another terminal, run the evaluation
tau2 run mock \
  --agent a2a_agent \
  --agent-a2a-endpoint http://localhost:8001/a2a/simple_nebius_agent \
  --num-trials 1

# 4. Stop the agent (Ctrl+C in first terminal)
```

### Pytest-Based Testing

```bash
# Run with pytest (automated lifecycle management)
pytest tests/test_local_eval/ -v

# Run with detailed logging
pytest tests/test_local_eval/ -v -s --log-cli-level=DEBUG
```

## Testing Workflow

### Development Cycle

1. **Modify Agent**: Edit `simple_nebius_agent/agent.py`
2. **Test Locally**: Run `./test_simple_agent.sh`
3. **Validate**: Check metrics and conversation quality
4. **Iterate**: Adjust configuration and retry

### Debugging

1. **Agent Not Starting**:
   - Check NEBIUS_API_KEY is set
   - Verify port 8001 is available: `lsof -i :8001`
   - Check logs for errors

2. **Evaluation Failing**:
   - Verify agent is accessible: `curl http://localhost:8001/a2a/simple_nebius_agent/.well-known/agent-card.json`
   - Check tau2-bench logs: `--a2a-debug`
   - Validate Nebius API is responding

3. **Unexpected Results**:
   - Review conversation logs
   - Check token usage and costs
   - Compare with baseline (OpenAI GPT-4 agent)

## Extension Points

### Adding Custom Agents

To create your own test agent:

1. Copy `simple_nebius_agent/` to `my_custom_agent/`
2. Modify `agent.py`:
   - Change agent name
   - Adjust model/configuration
   - Add tools if needed
3. Update scripts to use new port (8002)
4. Run tests

### Adding More Domains

To test with different tau2-bench domains:

```bash
# Airline domain (requires user LLM)
tau2 run airline \
  --agent a2a_agent \
  --agent-a2a-endpoint http://localhost:8001/a2a/simple_nebius_agent \
  --user-llm claude-3-haiku-20240307 \
  --num-trials 3

# Retail domain (requires user LLM)
tau2 run retail \
  --agent a2a_agent \
  --agent-a2a-endpoint http://localhost:8001/a2a/simple_nebius_agent \
  --user-llm claude-3-haiku-20240307 \
  --num-trials 3
```

### Adding Authentication

To test with authentication:

1. Modify agent to require bearer token
2. Update test scripts to include `--auth-token`
3. Validate 401 errors for missing/invalid tokens

## Success Metrics

### Functional Validation
- ✅ Agent starts successfully
- ✅ Agent card is accessible
- ✅ A2A message/send endpoint works
- ✅ Conversation completes without errors
- ✅ Metrics are collected

### Performance Baselines
- **Startup Time**: < 5 seconds
- **First Response**: < 2 seconds (after API call)
- **Token Usage**: ~100-500 tokens per turn
- **Error Rate**: 0% for valid requests

### Cost Considerations
- **Nebius Llama 3.1 8B**: ~$0.20 per 1M tokens
- **Single Evaluation (mock domain)**: ~$0.001
- **Full Test Suite**: ~$0.01

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Port already in use | Another agent running | `lsof -i :8001` and kill process |
| 401 Unauthorized | Missing/invalid API key | Verify NEBIUS_API_KEY is set |
| Agent not responding | Startup still in progress | Wait 5-10 seconds, check health |
| Evaluation timeout | Network/API issues | Check internet and Nebius API status |
| Import errors | Missing dependencies | `pip install -e .` |

### Logs

- **Agent Logs**: stdout from `run_simple_agent.sh`
- **Evaluation Logs**: tau2-bench logs (use `--log-level DEBUG`)
- **HTTP Logs**: Enable httpx logging in pytest fixtures

## Comparison with Full Implementation

| Feature | Simple Agent (This) | tau2_agent (Full) |
|---------|---------------------|------------------------|
| Purpose | Testing/Demo | Production Service |
| Complexity | ~50 lines | ~500+ lines |
| Tools | None | 3 tools (run_tau2, list_domains, get_results) |
| Port | 8001 | 8000 |
| Model | Nebius Llama 3.1 8B | Gemini 2.0 Flash |
| Use Case | Quick local testing | Remote evaluation service |

## Next Steps

After validating the simple local test:

1. **Scale Up**: Use the full tau2_agent for production
2. **Add Haiku**: Test with Claude Haiku for comparison
3. **Performance Testing**: Add benchmarking scripts
4. **CI Integration**: Add to GitHub Actions workflow
5. **Documentation**: Update main README with this workflow

## References

- [ADK Documentation](https://github.com/google/adk-toolkit)
- [A2A Protocol Specification](https://anthropic.com/a2a)
- [tau2-bench Documentation](https://github.com/tau-bench/tau2-bench)
- [Nebius AI API](https://nebius.ai/)
- [Research Notes](../research.md)
