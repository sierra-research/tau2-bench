# AgentBeats Integration Analysis

## Overview

This document provides a comprehensive analysis of the tau2-bench agent evaluation architecture, comparing the current A2A `run_domain` implementation with the gym-based `agentified-tau-bench` experimental implementation. The goal is to verify functional equivalence and identify integration points for AgentBeats competition deployment.

## Table of Contents

1. [Architecture Comparison](#architecture-comparison)
2. [Context ID Isolation Analysis](#context-id-isolation-analysis)
3. [Message Flow Analysis](#message-flow-analysis)
4. [Agent Role Definitions](#agent-role-definitions)
5. [Key Implementation Details](#key-implementation-details)
6. [Functional Equivalence Verification](#functional-equivalence-verification)

---

## Architecture Comparison

### Current A2A `run_domain` Implementation

**Entry Point**: `tau2_agent/tools/run_tau2_evaluation.py`

```
External Client → tau2_agent (ADK Server)
                      ↓
              run_tau2_evaluation tool
                      ↓
              tau2.run.run_domain()
                      ↓
              A2AAgent (src/tau2/agent/a2a_agent.py)
                      ↓
              Remote A2A Agent (purple agent)
```

**Key Files**:
- `tau2_agent/tools/run_tau2_evaluation.py` - Tool implementation
- `src/tau2/run.py` - `run_domain()` orchestration
- `src/tau2/agent/a2a_agent.py` - A2A protocol adapter
- `src/tau2/a2a/client.py` - HTTP client for A2A messages

### Gym-Based `agentified-tau-bench` Implementation

**Entry Point**: `experiments/agentify_tau_bench/launcher.py`

```
Launcher → Green Agent (TauGreenAgentExecutor)
               ↓
           gymnasium.make(TAU_BENCH_ENV_ID)
               ↓
           ask_agent_to_solve()
               ↓
           White Agent (GeneralWhiteAgentExecutor)
```

**Key Files**:
- `experiments/agentify_tau_bench/launcher.py` - Process orchestration
- `experiments/agentify_tau_bench/green_agent/agent.py` - Assessment manager
- `experiments/agentify_tau_bench/white_agent/agent.py` - Target agent
- `experiments/agentify_tau_bench/utils/a2a_utils.py` - A2A utilities

---

## Context ID Isolation Analysis

### A2A `run_domain` Approach

**Source**: `src/tau2/agent/a2a_agent.py`

#### State Initialization (lines 145-168)
```python
def get_init_state(
    self,
    message_history: list[Message] | None = None,
) -> A2AAgentState:
    """
    Get the initial state of the agent.
    Returns:
        Fresh A2AAgentState with no context_id (will be set on first response)
    """
    return A2AAgentState(
        context_id=None,  # <-- Fresh context per task
        conversation_history=message_history or [],
        agent_card=None,
        request_count=0,
    )
```

#### Context Persistence (lines 236-270)
```python
# After receiving response from A2A agent
if state.context_id is None and new_context_id is not None:
    logger.trace("A2A context_id lifecycle: New context created by agent")
elif state.context_id == new_context_id:
    logger.trace("A2A context_id lifecycle: Context persisted across turns")
elif state.context_id != new_context_id:
    logger.warning("A2A context_id lifecycle: Context changed unexpectedly")

# Update state with new context
new_state = A2AAgentState(
    context_id=new_context_id or state.context_id,  # <-- Preserve across turns
    conversation_history=new_conversation_history,
    request_count=state.request_count + 1,
)
```

### Gym-Based Approach

**Source**: `experiments/agentify_tau_bench/green_agent/agent.py`

#### Per-Task Context Initialization (lines 122-128)
```python
async def ask_agent_to_solve(
    white_agent_url: str,
    env: gym.Env,
    max_retries: int = 3,
) -> Optional[SimulationRun]:
    terminated = False
    context_id = None  # <-- Fresh context per task
    observation, info = env.reset()
```

#### Context Consistency Assertion (lines 216-221)
```python
if context_id is None:
    context_id = res_result.context_id  # <-- Set on first response
else:
    assert context_id == res_result.context_id, (
        "Context ID should remain the same in a conversation"  # <-- Validate
    )
```

### Purple Agent Context Handling

**Source**: `experiments/agentify_tau_bench/white_agent/agent.py`

```python
class GeneralWhiteAgentExecutor(AgentExecutor):
    def __init__(self):
        self.ctx_id_to_messages = {}  # <-- Per-context conversation history

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        if context.context_id not in self.ctx_id_to_messages:
            self.ctx_id_to_messages[context.context_id] = []  # <-- New context
        messages = self.ctx_id_to_messages[context.context_id]  # <-- Isolate by context
        # ... process with isolated message history
```

### `kimi_litellm_agent` Context Handling

**Source**: `kimi_litellm_agent/server.py`

Uses ADK's `get_fast_api_app(a2a=True)` which automatically handles context_id through the ADK framework's built-in session management.

---

## Message Flow Analysis

### A2A `run_domain` Flow

```
1. run_tau2_evaluation.run_async()
   ├─ Get credentials from context variables (X-User-LLM-* headers)
   ├─ Create evaluation session in store
   └─ Call _execute()

2. _execute()
   ├─ Validate domain
   ├─ Create RunConfig(agent="a2a_agent", llm_agent=endpoint_url)
   └─ Call run_domain(config) in ThreadPoolExecutor

3. run_domain() → run_tasks()
   └─ For each task:
       └─ Orchestrator.simulate_task()
           └─ A2AAgent.generate_next_message() for each turn
               ├─ Translate tau2 message → A2A format
               ├─ Send via A2AClient.send_message()
               ├─ Parse response (5 formats supported)
               └─ Update state with context_id
```

### Gym-Based Flow

```
1. launcher.launch_evaluation()
   ├─ Start green agent process (port 9001)
   ├─ Start white agent process (port 9002)
   └─ Send evaluation request to green agent

2. TauGreenAgentExecutor.execute()
   ├─ Parse white_agent_url and env_config from tags
   └─ For each task (via asyncio.gather):
       └─ run_one_task(task_id)
           ├─ gym.make(TAU_BENCH_ENV_ID, **config)
           └─ ask_agent_to_solve()

3. ask_agent_to_solve()
   ├─ env.reset() → get initial observation
   └─ Loop until terminated:
       ├─ a2a_send_message(white_agent_url, message, context_id)
       ├─ Parse response tags for action JSON
       └─ env.step(action) → next observation
```

---

## Agent Role Definitions

### AgentBeats Terminology

| Role | Description | Current Implementation |
|------|-------------|----------------------|
| **Green Agent** | Assessment manager that orchestrates evaluations | `tau2_agent` with `run_tau2_evaluation` tool |
| **Purple/White Agent** | Target agent being evaluated | `kimi_litellm_agent` or any A2A-compliant agent |
| **User Simulator** | Simulates customer interactions | Built into tau2-bench, configurable via LLM |

### Role Responsibilities

**Green Agent (tau2_agent)**:
- Receives evaluation requests via A2A protocol
- Creates evaluation sessions and tracks progress
- Invokes tau2-bench evaluation engine
- Reports results back to caller

**Purple Agent (kimi_litellm_agent)**:
- Receives task messages via A2A protocol
- Maintains conversation context per `context_id`
- Responds with tool calls or direct responses
- Has no knowledge it's being evaluated

---

## Key Implementation Details

### Credential Handling

**A2A `run_domain`** (`tau2_agent/middleware.py`):
```python
# Context variables for request-scoped credentials
user_llm_model: ContextVar[str | None]
user_llm_api_key: ContextVar[str | None]

# Middleware extracts from headers
X-User-LLM-Model: gemini-2.0-flash
X-User-LLM-API-Key: <api-key>
```

**Gym-based** (`launcher.py`):
```python
# Passed directly in env_config
task_config = {
    "user_llm": "openrouter/openai/gpt-4o",
    "user_llm_args": {"temperature": 0.0},
}
```

### A2A Message Format

**Request**:
```json
{
  "jsonrpc": "2.0",
  "id": "<uuid>",
  "method": "message/send",
  "params": {
    "message": {
      "messageId": "<uuid>",
      "role": "user",
      "parts": [{"text": "<content>"}],
      "contextId": "<session_id>"
    }
  }
}
```

**Response Parsing** (`src/tau2/a2a/client.py`):
Supports 5 formats:
1. Google ADK: `result.artifacts[].parts[].text`
2. Standard A2A: `result.parts[].text`
3. ADK streaming: `result.status.message.parts[].text`
4. Legacy wrapper: `result.message.parts[].text`
5. History-based: Last agent message from `result.history[]`

### Concurrency Models

| Implementation | Concurrency Mechanism | Limit |
|----------------|----------------------|-------|
| A2A `run_domain` | `ThreadPoolExecutor` | 10 workers |
| Gym-based | `asyncio.Semaphore` | 2 concurrent tasks |

---

## Functional Equivalence Verification

### Isolation Comparison

| Aspect | A2A `run_domain` | Gym-based |
|--------|-----------------|-----------|
| **Context Init** | `context_id=None` in `get_init_state()` | `context_id = None` per task |
| **First Message** | Sent without context_id | Sent without context_id |
| **Context Receipt** | From `new_context_id` in response | From `res_result.context_id` |
| **Persistence** | `context_id=new_context_id or state.context_id` | Assert equality, reuse |
| **Task Isolation** | Fresh `A2AAgentState` per task | Fresh `context_id=None` per task |
| **State Storage** | In `A2AAgentState` dataclass | In local variable |

### Conclusion

**Both implementations are functionally equivalent** in their isolation mechanism:

1. **Per-task isolation**: Both create fresh context for each task
2. **Multi-turn persistence**: Both preserve context across turns within a task
3. **Purple agent dependency**: Both rely on the purple agent to honor `context_id`
4. **No cross-contamination**: No shared state between concurrent task evaluations

The primary difference is the **mechanism**, not the **isolation model**:
- A2A `run_domain`: Uses `run_domain()` orchestrator with `A2AAgent`
- Gym-based: Uses Gymnasium `step()` API with threading

### Verified Behaviors

| Behavior | A2A `run_domain` | Gym-based | Status |
|----------|-----------------|-----------|--------|
| Fresh context per task | `get_init_state()` returns `context_id=None` | `context_id = None` in `ask_agent_to_solve()` | EQUIVALENT |
| Context persisted across turns | `new_context_id or state.context_id` | Assert and reuse | EQUIVALENT |
| Purple agent honors context | ADK handles automatically | `ctx_id_to_messages` dict | EQUIVALENT |
| Concurrent task isolation | Fresh `A2AAgentState` per task | Fresh env and context per task | EQUIVALENT |

---

## Summary

The current `tau2_agent` implementation with `run_tau2_evaluation` tool and `A2AAgent` provides **functionally equivalent isolation** to the gym-based `agentified-tau-bench` implementation. Both:

1. Initialize with `context_id=None` for each new task evaluation
2. Preserve context across turns within a task
3. Rely on the purple agent (e.g., `kimi_litellm_agent`) to honor `context_id` for session management
4. Prevent state leakage between concurrent task evaluations

This validates that the current A2A-based implementation is suitable for AgentBeats integration without requiring modifications to the core evaluation logic.
