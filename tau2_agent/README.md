# tau2_agent - Evaluation Service Agent

ADK agent that exposes tau2-bench evaluation capabilities via A2A protocol.

## Purpose

This agent is the **evaluator** - it accepts requests to evaluate other A2A-compatible agents against tau2-bench domains. It is NOT a target agent for evaluation itself.

## Usage

### Start the A2A Server

```bash
# From project root
adk api_server --a2a . --port 8001
```

### Verify Agent Card

```bash
curl http://localhost:8001/a2a/tau2_agent/.well-known/agent-card.json | jq
```

### Request an Evaluation

```bash
curl -X POST http://localhost:8001/a2a/tau2_agent/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "message/send",
    "params": {
      "message": {
        "messageId": "eval-001",
        "role": "user",
        "parts": [{"text": "Evaluate the agent at http://localhost:8002/a2a/simple_nebius_agent on the mock domain with 3 tasks"}]
      }
    },
    "id": "req-001"
  }'
```

## Available Tools

### run_tau2_evaluation

Execute tau2-bench evaluation of a conversational agent.

**Parameters:**
- `domain`: Evaluation domain (airline, retail, telecom, mock)
- `agent_endpoint`: A2A endpoint of the agent to evaluate
- `user_llm`: LLM model for user simulator (default: gpt-4o)
- `num_trials`: Number of trials per task (default: 1)
- `num_tasks`: Number of tasks to evaluate (optional)
- `task_ids`: Specific task IDs to run (optional)

### list_domains

List all available tau2-bench evaluation domains.

### get_evaluation_results

Retrieve detailed results from a completed evaluation.

## Architecture

```
tau2_agent (Evaluator)
    │
    ├─► A2A Protocol ─► Target Agent (e.g., simple_nebius_agent)
    │
    └─► tau2-bench evaluation framework
            │
            ├─► Domain tasks (airline, retail, telecom)
            ├─► User simulator
            └─► Metrics collection
```

## Notes

- This agent orchestrates evaluations, not receives them
- Target agents being evaluated must be A2A-compatible
- Evaluations can take several minutes depending on task count
