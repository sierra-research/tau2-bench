# Simple Nebius Agent

A minimal ADK agent that wraps the Nebius Qwen3-30B API for local A2A protocol testing.

## Purpose

This agent provides a simple example of:
- Configuring ADK with a custom OpenAI-compatible API (Nebius)
- Exposing an agent via A2A protocol on localhost
- Testing tau2-bench's A2A client against a real agent

## Prerequisites

1. **Python Environment**:
   ```bash
   pip install -e .
   ```

2. **Nebius API Key**:
   - Sign up at https://tokenfactory.nebius.com/
   - Get your API key
   - Set environment variable:
     ```bash
     export NEBIUS_API_KEY="your-api-key-here"
     ```

## Quick Start

### 1. Start the Agent Server

```bash
# Set your API key
export NEBIUS_API_KEY="your-api-key-here"

# Start the agent on port 8001 (from project root)
adk api_server --a2a . --port 8001
```

The agent will be available at `http://localhost:8001/a2a/simple_nebius_agent`

### 2. Verify Agent is Running

Check the agent card:
```bash
curl http://localhost:8001/a2a/simple_nebius_agent/.well-known/agent-card.json | jq
```

Expected response:
```json
{
  "name": "simple_nebius_agent",
  "description": "A simple agent using Nebius Qwen3-30B for testing A2A protocol",
  "url": "http://localhost:8001/a2a/simple_nebius_agent",
  "version": "1.0.0",
  "capabilities": {
    "streaming": false
  }
}
```

### 3. Test with tau2-bench

Run a simple evaluation (mock domain):
```bash
tau2 run mock \
  --agent a2a_agent \
  --agent-a2a-endpoint http://localhost:8001/a2a/simple_nebius_agent \
  --num-trials 1
```

For real domains (requires user simulator):
```bash
tau2 run airline \
  --agent a2a_agent \
  --agent-a2a-endpoint http://localhost:8001/a2a/simple_nebius_agent \
  --user-llm claude-3-haiku-20240307 \
  --num-trials 1 \
  --num-tasks 5
```

## Configuration

### Environment Variables

- **NEBIUS_API_KEY** (required): Your Nebius API key
- **NEBIUS_API_BASE** (optional): API base URL
  - Default: `https://api.tokenfactory.nebius.com/v1/`

### Agent Parameters

You can modify the agent behavior by editing [agent.py](agent.py):

```python
# Change the model
llm_model = LiteLlm(
    model="nebius/Qwen/Qwen3-30B-A3B-Thinking-2507",  # Larger model
    api_base=api_base,
    api_key=api_key,
)

# Change the instruction
agent = LlmAgent(
    model=llm_model,
    name="simple_nebius_agent",
    description="Custom description",
    instruction="Custom system prompt here",
)
```

## Port Configuration

By default, this agent runs on port 8001 to avoid conflicts with:
- Port 8000: Main tau2_agent
- Port 8002+: Available for additional test agents

To use a different port:
```bash
adk api_server --a2a . --port 8002
```

## Testing

### Manual Testing

1. **Test Agent Discovery**:
   ```bash
   curl http://localhost:8001/a2a/simple_nebius_agent/.well-known/agent-card.json | jq
   ```

2. **Test Message Sending** (using A2A protocol):
   ```bash
   curl -X POST http://localhost:8001/a2a/simple_nebius_agent \
     -H "Content-Type: application/json" \
     -d '{
       "jsonrpc": "2.0",
       "method": "message/send",
       "id": "test-123",
       "params": {
         "message": {
           "messageId": "msg-123",
           "role": "user",
           "parts": [{"kind": "text", "text": "Hello, how are you?"}]
         }
       }
     }'
   ```

### Automated Testing

Run the automated test suite:
```bash
# Run local evaluation tests
pytest tests/test_local_eval/ -v

# Run with detailed logging
pytest tests/test_local_eval/ -v -s --log-cli-level=DEBUG
```

Or use the convenience script:
```bash
./specs/001-a2a-integration/scripts/test_simple_agent.sh
```

## Architecture

```
+--------------------------------------------+
|  Simple Nebius Agent (localhost:8001)      |
|                                            |
|  +--------------------------------------+  |
|  |  A2A HTTP Server                     |  |
|  |  (adk api_server --a2a)              |  |
|  +------------------+-------------------+  |
|                     |                      |
|  +------------------v-------------------+  |
|  |  LlmAgent (ADK)                      |  |
|  |  - Name: simple_nebius_agent         |  |
|  |  - No tools (conversation only)      |  |
|  +------------------+-------------------+  |
|                     |                      |
|  +------------------v-------------------+  |
|  |  LiteLlm Wrapper                     |  |
|  |  - Model: Qwen3-30B                  |  |
|  |  - API: Nebius                       |  |
|  +--------------------------------------+  |
+--------------------------------------------+
                     |
                     | HTTP Request
                     v
       +------------------------+
       |  Nebius API            |
       |  Qwen/                 |
       |  Qwen3-30B-A3B         |
       +------------------------+
```

## Comparison with Full tau2_agent

| Feature | simple_nebius_agent | tau2_agent |
|---------|---------------------|-----------------|
| Purpose | Testing/Demo | Production Service |
| Complexity | ~50 lines | 500+ lines |
| Tools | None | 3 tools |
| Port | 8001 | 8000 |
| Model | Nebius Qwen3-30B | Gemini 2.0 Flash |
| Use Case | Quick local testing | Remote evaluation |

## Troubleshooting

### Agent Won't Start

**Error**: `ValueError: NEBIUS_API_KEY environment variable is required`

**Solution**: Set your API key:
```bash
export NEBIUS_API_KEY="your-api-key-here"
```

### Port Already in Use

**Error**: `Address already in use`

**Solution**: Check what's using the port and kill it:
```bash
lsof -i :8001
kill <PID>
```

Or use a different port:
```bash
adk api_server --a2a . --port 8002
```

### Agent Not Responding

**Check 1**: Wait for startup (5-10 seconds)
```bash
# Health check
curl http://localhost:8001/a2a/simple_nebius_agent/.well-known/agent-card.json
```

**Check 2**: Verify API key is valid
```bash
# Test Nebius API directly
curl https://api.tokenfactory.nebius.com/v1/models \
  -H "Authorization: Bearer $NEBIUS_API_KEY"
```

### Import Errors

**Error**: `ModuleNotFoundError: No module named 'google.adk'`

**Solution**: Install dependencies:
```bash
pip install -e .
```

## Cost Considerations

- **Model**: Qwen/Qwen3-30B-A3B-Thinking-2507
- **Provider**: Nebius
- **Estimated Cost**: ~$0.20 per 1M tokens
- **Typical Evaluation**: ~$0.001 per task (mock domain)

## Next Steps

After testing the simple agent:

1. **Scale Up**: Use the full `tau2_agent` for production evaluations
2. **Add Tools**: Extend the agent with custom tools
3. **Different Models**: Try Claude Haiku or GPT-4
4. **More Domains**: Test with airline, retail, or telecom domains

## References

- [Local Test Architecture](../specs/001-a2a-integration/testing/local-test-architecture.md)
- [ADK Documentation](https://adk.google.dev/)
- [A2A Protocol](https://a2a-protocol.org/)
- [Nebius AI](https://tokenfactory.nebius.com/)
