# ADR: GreenExecutor for AgentBeats Integration

**Status**: Proposed
**Date**: 2026-01-10
**Author**: Claude Code

## Context

### Problem Statement

The tau2_agent currently uses ADK's `LlmAgent` which:
1. Receives evaluation requests via A2A protocol at `POST /a2a/tau2_agent`
2. Uses an LLM orchestrator to interpret requests and call tools
3. The `run_tau2_evaluation` tool returns structured data, but the LLM summarizes results as **markdown TextPart**

**Issue**: The agentbeats-client (v1.0.0) extracts results from **DataPart** artifacts only. When it receives markdown text, `json.loads()` fails and the results array remains empty:

```json
{
  "participants": {"agent": "019b9515-..."},
  "results": []  // Empty because TextPart contains markdown, not JSON
}
```

### Research References

1. **RDI-Foundation/agentbeats-tutorial** (`scenarios/tau2/tau2_evaluator.py`)
   - Reference implementation of GreenAgent/GreenExecutor pattern
   - Uses `TaskUpdater.add_artifact()` with both TextPart and DataPart
   - Key code: https://github.com/RDI-Foundation/agentbeats-tutorial

2. **RDI-Foundation/agentbeats-debate-leaderboard** (`results/*.json`)
   - Shows expected result format with populated `results` array
   - Example: `{"participants": {...}, "results": [{"winner": "...", "detail": {...}}]}`

3. **agentbeats-client source** (`src/agentbeats/client_cli.py`)
   ```python
   for artifact in artifacts:
       _, data_parts = parse_parts(artifact.parts)
       all_data_parts.extend(data_parts)

   def parse_parts(parts):
       for part in parts:
           if isinstance(part.root, DataPart):
               data_parts.append(part.root.data)  # <-- This is what we need
           elif isinstance(part.root, TextPart):
               data_item = json.loads(part.root.text)  # Fails on markdown
   ```

4. **Google ADK A2A Executor** (`a2a/executor/a2a_agent_executor.py`)
   - Creates artifacts from `task_result_aggregator.task_status_message.parts`
   - These parts come from LLM response, hence TextPart with markdown

## Decision

### Chosen Approach: Separate GreenExecutor Route

Add a separate `Tau2GreenExecutor` that implements the A2A `AgentExecutor` interface directly, bypassing the LLM orchestrator.

**Dual-route architecture**:
| Path | Handler | Purpose |
|------|---------|---------|
| `POST /` | tau2_green | AgentBeats evaluations (returns DataPart) |
| `GET /.well-known/agent-card.json` | tau2_green card | Root-based A2A discovery |
| `POST /a2a/tau2_agent` | LlmAgent | Local testing, natural language |
| `POST /a2a/tau2_green` | tau2_green | Explicit path (same as root) |

### Design Rationale

#### Why Separate Route (not middleware/detection)?

| Approach | Complexity | Risk | Maintainability |
|----------|------------|------|-----------------|
| **Separate route** | Low | None | High - clear separation |
| Entry routing middleware | Medium-High | Medium | Medium - mixing concerns |
| Custom ADK executor wrapper | High | High - ADK internals | Low - may break on updates |

The A2A protocol uses SSE streaming with TaskStatusUpdateEvent and TaskArtifactUpdateEvent. Implementing this in middleware would duplicate what `AgentExecutor` already provides.

#### Why Root-Based Routing?

- Eliminates `agent_name` config requirement for submitters
- Follows agentbeats convention (root-based discovery)
- `generate_compose.py` already supports empty `agent_name` (returns root path)

#### Why Reuse RunTau2Evaluation?

- All evaluation logic already implemented and tested
- Credential handling via env vars already works
- Metrics emission, error handling, store persistence all included
- Zero code duplication

### Alternatives Considered

1. **Modify LlmAgent response**: Intercept tool results before LLM summarizes
   - **Rejected**: ADK doesn't expose clean hook for this; tightly coupled to internals

2. **Post-process artifacts**: Add DataPart after LlmAgent completes
   - **Rejected**: Tool result not available after LLM processing

3. **Custom after_model_callback**: Capture structured data in callback
   - **Rejected**: Callback sees LLM response, not raw tool output

## Implementation Plan

### Phase 1: Create GreenExecutor (tau2-bench-agent)

**File**: `tau2_agent/green_executor.py`

```python
"""AgentBeats-compatible evaluation executor with DataPart results."""

from pydantic import BaseModel
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Part, TaskState, TextPart
from a2a.utils import new_agent_text_message, new_task

from tau2_agent.tools.run_tau2_evaluation import RunTau2Evaluation


class EvalConfig(BaseModel):
    """Evaluation configuration from agentbeats scenario."""
    domain: str
    num_tasks: int | None = None
    num_trials: int = 1
    task_ids: list[str] | None = None


class EvalRequest(BaseModel):
    """Request format from agentbeats-client."""
    participants: dict[str, str]  # {"agent": "http://..."}
    config: EvalConfig


class Tau2GreenAgent:
    """Direct evaluation executor for AgentBeats."""

    async def run_eval(self, request: EvalRequest, updater: TaskUpdater) -> None:
        """Execute evaluation and return structured results."""
        agent_endpoint = request.participants.get("agent")
        if not agent_endpoint:
            raise ValueError("Missing 'agent' in participants")

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"Starting evaluation: domain={request.config.domain}, "
                f"num_tasks={request.config.num_tasks}"
            )
        )

        # Reuse existing evaluation tool logic
        tool = RunTau2Evaluation(name="run_tau2_evaluation", description="")

        args = {
            "domain": request.config.domain,
            "agent_endpoint": agent_endpoint,
            "num_trials": request.config.num_trials,
        }
        if request.config.num_tasks:
            args["num_tasks"] = request.config.num_tasks
        if request.config.task_ids:
            args["task_ids"] = request.config.task_ids

        result = await tool.run_async(args=args, tool_context=None)

        # Check for errors
        if "error" in result:
            raise ValueError(f"{result['error']}: {result.get('message', '')}")

        # Format human-readable summary
        summary = result.get("summary", {})
        total = summary.get("total_simulations", 0)
        successful = summary.get("successful_simulations", 0)
        avg_reward = summary.get("avg_reward", 0)

        summary_text = f"""Evaluation Results
Domain: {request.config.domain}
Tasks: {summary.get('total_tasks', 0)}
Pass Rate: {successful}/{total} ({avg_reward:.1%})
Avg Agent Cost: ${summary.get('avg_agent_cost', 0):.4f}"""

        # Add artifact with BOTH TextPart (human) and DataPart (structured)
        await updater.add_artifact(
            parts=[
                Part(root=TextPart(text=summary_text)),
                Part(root=DataPart(data=result)),
            ],
            name="evaluation_results",
        )


class Tau2GreenExecutor(AgentExecutor):
    """A2A executor that wraps Tau2GreenAgent for agentbeats compatibility."""

    def __init__(self):
        self.agent = Tau2GreenAgent()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute evaluation request and stream results."""
        # Parse EvalRequest from A2A message
        request_text = context.get_user_input()
        request = EvalRequest.model_validate_json(request_text)

        # Create task and send initial event
        task = new_task(context.message)
        await event_queue.enqueue_event(task)

        # Run evaluation with TaskUpdater for streaming
        updater = TaskUpdater(event_queue, task.id, task.context_id)

        try:
            await self.agent.run_eval(request, updater)
            await updater.complete()
        except Exception as e:
            await updater.failed(
                new_agent_text_message(f"Evaluation failed: {e}")
            )
            raise

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        """Cancel is not supported for evaluations."""
        raise NotImplementedError("Cancellation not supported")
```

### Phase 2: Modify Server (tau2-bench-agent)

**File**: `tau2_agent/server.py` (modifications)

```python
# Add imports
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentCapabilities

from tau2_agent.green_executor import Tau2GreenExecutor


def create_green_agent_card(base_url: str) -> AgentCard:
    """Create agent card for the green executor route."""
    return AgentCard(
        name="tau2_green",
        description="AgentBeats-compatible tau2 evaluation (structured results)",
        url=base_url,
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=True),
    )


def create_app():
    project_root = Path(__file__).resolve().parent.parent
    agents_dir = os.getenv("AGENTS_DIR", str(project_root / "agents"))

    # Existing ADK app with LlmAgent at /a2a/tau2_agent
    app = get_fast_api_app(agents_dir=agents_dir, web=False, a2a=True)

    # Create green executor for agentbeats
    card_url = os.getenv("CARD_URL", "http://localhost:8001")
    green_executor = Tau2GreenExecutor()
    green_handler = DefaultRequestHandler(
        agent_executor=green_executor,
        task_store=InMemoryTaskStore(),
    )
    green_app = A2AStarletteApplication(
        agent_card=create_green_agent_card(card_url),
        http_handler=green_handler,
    )
    green_starlette = green_app.build()

    # Mount at explicit path
    app.mount("/a2a/tau2_green", green_starlette)

    # Root-based routes for agentbeats (no agent_name config needed)
    @app.get("/.well-known/agent-card.json")
    async def root_agent_card():
        return create_green_agent_card(card_url).model_dump()

    # Mount green executor at root for POST /
    app.mount("/", green_starlette)

    # Middleware
    from tau2_agent.middleware import CredentialsMiddleware
    app.add_middleware(CredentialsMiddleware)

    return app
```

### Phase 3: Update Leaderboard (tau2-bench-agent-leaderboard)

**File**: `scenario.toml`

```toml
[green_agent]
agentbeats_id = "019b950f-0070-7aa0-9135-085aab814ed7"
# No agent_name needed - uses root endpoint automatically
env = { USER_LLM_MODEL = "nebius/Qwen/Qwen3-235B-A22B-Thinking-2507", ... }

[[participants]]
agentbeats_id = "019b9515-47bd-7e80-8ad3-86c33d0175c9"
name = "agent"
agent_name = "kimi_litellm_agent"
env = { NEBIUS_API_KEY = "${NEBIUS_API_KEY}" }

[config]
domain = "airline"
num_tasks = 5
```

## Test Scenarios

### Scenario 1: Unit Test - EvalRequest Parsing

```python
def test_eval_request_parsing():
    """Test that EvalRequest parses correctly from JSON."""
    json_data = {
        "participants": {"agent": "http://localhost:9009/a2a/test"},
        "config": {"domain": "mock", "num_tasks": 2}
    }
    request = EvalRequest.model_validate(json_data)
    assert request.participants["agent"] == "http://localhost:9009/a2a/test"
    assert request.config.domain == "mock"
    assert request.config.num_tasks == 2
    assert request.config.num_trials == 1  # default
```

### Scenario 2: Unit Test - GreenAgent Execution

```python
@pytest.mark.asyncio
async def test_green_agent_run_eval(mock_updater, mock_tool):
    """Test that GreenAgent returns DataPart with structured results."""
    agent = Tau2GreenAgent()
    request = EvalRequest(
        participants={"agent": "http://localhost:9009"},
        config=EvalConfig(domain="mock", num_tasks=1)
    )

    await agent.run_eval(request, mock_updater)

    # Verify add_artifact was called with DataPart
    mock_updater.add_artifact.assert_called_once()
    parts = mock_updater.add_artifact.call_args[1]["parts"]

    assert len(parts) == 2
    assert isinstance(parts[0].root, TextPart)
    assert isinstance(parts[1].root, DataPart)
    assert "status" in parts[1].root.data
    assert "summary" in parts[1].root.data
```

### Scenario 3: Integration Test - A2A Protocol

```python
@pytest.mark.asyncio
async def test_green_executor_a2a_response():
    """Test full A2A request/response cycle."""
    async with AsyncClient(app=create_app(), base_url="http://test") as client:
        response = await client.post(
            "/a2a/tau2_green",
            json={
                "jsonrpc": "2.0",
                "method": "message/send",
                "id": "test-1",
                "params": {
                    "message": {
                        "parts": [{
                            "text": json.dumps({
                                "participants": {"agent": "http://mock:9009"},
                                "config": {"domain": "mock", "num_tasks": 1}
                            })
                        }]
                    }
                }
            }
        )

        # SSE response should contain DataPart artifact
        events = parse_sse_response(response)
        completed_event = find_event(events, state="completed")
        assert completed_event is not None

        artifact = completed_event.artifacts[0]
        data_part = find_data_part(artifact.parts)
        assert data_part is not None
        assert "status" in data_part.data
```

### Scenario 4: End-to-End Test - Leaderboard

```bash
# Start services
cd /home/ubuntu/workspace/tau2-bench-agent-leaderboard
docker compose up -d

# Wait for health
docker compose exec green-agent curl -f http://localhost:9009/.well-known/agent-card.json

# Run evaluation
docker compose up agentbeats-client

# Verify results
cat output/results.json | jq '.results | length'
# Expected: > 0
```

## Test Instructions

### Prerequisites

```bash
cd /home/ubuntu/workspace/tau2-bench-agent
uv sync
```

### Running Unit Tests

```bash
# Create test file first (see test scenarios above)
uv run pytest tests/test_green_executor.py -v
```

### Running Local Integration Test

```bash
# Terminal 1: Start tau2_agent with green executor
cd /home/ubuntu/workspace/tau2-bench-agent
CARD_URL=http://localhost:9009 uv run python -m tau2_agent.server --port 9009

# Terminal 2: Start a mock purple agent
cd /home/ubuntu/workspace/tau2-bench-agent
uv run python -m simple_nebius_agent.server --port 9010

# Terminal 3: Send test request
curl -X POST http://localhost:9009/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "message/send",
    "id": "test-1",
    "params": {
      "message": {
        "parts": [{
          "text": "{\"participants\":{\"agent\":\"http://localhost:9010/a2a/simple_nebius_agent\"},\"config\":{\"domain\":\"mock\",\"num_tasks\":1}}"
        }]
      }
    }
  }'

# Should see SSE events with DataPart in artifact
```

### Running Full Leaderboard Test

```bash
cd /home/ubuntu/workspace/tau2-bench-agent-leaderboard

# Update scenario.toml to remove agent_name from green_agent
# Generate compose file
python generate_compose.py --scenario scenario.toml

# Create .env with credentials
cp .env.example .env
# Edit .env with NEBIUS_API_KEY

# Run full stack
docker compose up --abort-on-container-exit

# Check results
cat output/results.json | jq '.'
# Expected: {"participants": {...}, "results": [{...structured data...}]}
```

## Consequences

### Positive

- **Zero impact on existing LlmAgent**: Still available at `/a2a/tau2_agent`
- **No config changes for submitters**: Root-based routing eliminates `agent_name` requirement
- **Reuses battle-tested code**: `RunTau2Evaluation` handles all complexity
- **Clean separation**: GreenExecutor is isolated, easy to test and maintain
- **Follows established pattern**: Same approach as RDI-Foundation examples

### Negative

- **Two execution paths**: Must maintain both LlmAgent and GreenExecutor
- **Slight complexity in server.py**: Additional mounting logic

### Neutral

- **New dependency on A2A server types**: Already a transitive dependency via ADK
- **Docker image size**: Minimal impact (~1 file added)

## Files Changed

| File | Change Type | Description |
|------|-------------|-------------|
| `tau2_agent/green_executor.py` | Create | GreenExecutor and Tau2GreenAgent |
| `tau2_agent/server.py` | Modify | Add green executor mounting |
| `tests/test_green_executor.py` | Create | Unit tests |
| `tau2-bench-agent-leaderboard/scenario.toml` | Modify | Remove `agent_name` |
