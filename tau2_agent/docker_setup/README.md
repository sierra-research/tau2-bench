# Docker Setup for tau2_agent

Run tau2_agent in a Docker container.

## Quick Start

```bash
# 1. Build and start
docker-compose up -d

# 2. Test the agent
curl http://localhost:8001/a2a/tau2_agent/.well-known/agent-card.json
```

## Configuration

Set API keys in `.env` file at project root:
```bash
NEBIUS_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here  # optional
ANTHROPIC_API_KEY=your_key_here  # optional
```

## Endpoints

### From Host Machine
- Agent Card: `http://localhost:8001/a2a/tau2_agent/.well-known/agent-card.json`
- A2A Endpoint: `http://localhost:8001/a2a/tau2_agent`

### From Inside a Container (e.g., dev container)
- Agent Card: `http://host.docker.internal:8001/a2a/tau2_agent/.well-known/agent-card.json`
- A2A Endpoint: `http://host.docker.internal:8001/a2a/tau2_agent`

## Test the Agent

### Get Agent Card
```bash
# From host machine
curl http://localhost:8001/a2a/tau2_agent/.well-known/agent-card.json | python3 -m json.tool

# From inside a container
curl http://host.docker.internal:8001/a2a/tau2_agent/.well-known/agent-card.json | python3 -m json.tool
```

### Send a Message
```bash
# From host machine
curl -X POST http://localhost:8001/a2a/tau2_agent \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "request-1",
    "method": "message/send",
    "params": {
      "message": {
        "messageId": "msg-1",
        "role": "user",
        "parts": [{"text": "List available evaluation domains"}]
      }
    }
  }' | python3 -m json.tool

# From inside a container
curl -X POST http://host.docker.internal:8001/a2a/tau2_agent \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "request-1",
    "method": "message/send",
    "params": {
      "message": {
        "messageId": "msg-1",
        "role": "user",
        "parts": [{"text": "List available evaluation domains"}]
      }
    }
  }' | python3 -m json.tool
```

### Message Structure

```json
{
  "jsonrpc": "2.0",
  "id": "unique-request-id",
  "method": "message/send",
  "params": {
    "message": {
      "messageId": "unique-message-id",
      "role": "user",
      "parts": [{"text": "Your message here"}]
    }
  }
}
```

For multi-turn conversations, include `contextId` from previous response:
```json
{
  "params": {
    "message": {
      "messageId": "msg-2",
      "role": "user",
      "contextId": "previous-context-id",
      "parts": [{"text": "Follow-up message"}]
    }
  }
}
```

## Commands

```bash
# Start
docker-compose up -d

# Stop
docker-compose down

# View logs
docker-compose logs -f

# Rebuild
docker-compose up -d --build
```

## Troubleshooting

### 404 Error When Agent Tries to Contact Another Agent

**Error:**
```text
A2AError: Message send failed with status 404 (HTTP 404)
```

**Common Causes:**

1. **Port Conflict**
   - Multiple services using the same port
   - **Solution**: Use different ports for different agents (e.g., tau2_agent on 8001, other agents on 8002+)

2. **Incorrect Host Binding**
   - ADK server bound to `127.0.0.1` (localhost only) instead of `0.0.0.0`
   - **Solution**: Start ADK server with `--host 0.0.0.0`:
     ```bash
     adk api_server --a2a . --port 8002 --host 0.0.0.0
     ```

3. **Network Configuration**
   - Using `localhost` instead of `host.docker.internal` when connecting from containers
   - **Solution**: From inside containers, use `host.docker.internal` to reach host services

**Verification:**

Check port binding shows `0.0.0.0`:
```bash
ss -tlnp | grep :8002
# Should show: LISTEN 0 2048 0.0.0.0:8002 0.0.0.0:*
```

Test connectivity from the container:
```bash
docker exec tau2-agent python3 -c "
import httpx
r = httpx.get('http://host.docker.internal:8002/a2a/agent_name/.well-known/agent-card.json', timeout=5)
print(f'Status: {r.status_code}')
"
# Should print: Status: 200
```

### Non-Critical Errors in Logs

These errors can be safely ignored:

#### 1. Unclosed client session
```text
ERROR - base_events.py:1785 - Unclosed client session
```
- Minor memory leak from Google ADK's aiohttp client
- Does not affect functionality

#### 2. LLM cost tracking error
```text
ERROR | tau2.utils.llm_utils:get_response_cost:97 - litellm.BadRequestError
```
- Cost tracking fails for non-standard model names (e.g., Nebius models)
- Results in `avg_agent_cost: null` but evaluation proceeds normally

### Container Won't Start

**Error:** `Address already in use`

**Solution:** Check what's using port 8001:
```bash
docker ps | grep 8001
lsof -i :8001
```

Kill conflicting process or change port in `docker-compose.yml`.
