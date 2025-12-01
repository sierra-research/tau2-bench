# Research: A2A Protocol Integration for tau2-bench via ADK

**Date**: 2025-11-24
**Feature**: A2A Protocol Integration using Google ADK
**Branch**: `001-a2a-integration`
**Last Updated**: 2025-11-24 (Corrected API references)

---

## ⚠️ Corrections Applied (2025-11-24)

The following critical corrections have been made to this research document:

### 1. ADK CLI Flag Corrected
- **❌ Previous:** `adk web tau2_agent/ --enable-a2a`
- **✅ Correct:** `adk web --a2a tau2_agent/`
- **Impact:** All command examples updated (lines 298, 652, 723)

### 2. tau2-bench Python API Corrected
- **❌ Previous:** `from tau2.orchestrator import run_simulation`, `from tau2.domains import load_domain`
- **✅ Correct:** `from tau2.run import run_domain, load_tasks, get_tasks, get_options`
- **✅ Correct:** `from tau2.data_model.simulation import RunConfig` for configuration
- **✅ Results structure:** Returns `Results` object with `timestamp`, `info`, `tasks`, `simulations` fields
- **Note:** tau2-bench DOES provide a Python API, but with different function names than initially assumed
- **Impact:** Tool implementations updated with correct RunConfig and Results handling

### 3. A2A Client API Corrected
- **❌ Previous:** `from a2a_sdk import A2AClient` with `A2AClient(endpoint)`
- **✅ Correct:** `from a2a.client.client_factory import ClientFactory` with `await ClientFactory.connect(endpoint)`
- **✅ Correct Message construction:** `Part(root=TextPart(text="..."))` not `TextPart(text="...")`
- **Impact:** A2AAgent implementation completely rewritten to extend LocalAgent and use correct BaseAgent interface

### 4. Message Format Clarification
- A2A protocol uses camelCase in JSON-RPC wire format (messageId, contextId)
- Protocol Buffers use snake_case (message_id, context_id)
- Examples show JSON-RPC format (lines 610-621)

---

## Overview

This document consolidates research findings for implementing A2A (Agent-to-Agent) protocol support to enable tau2-bench as an evaluator agent. The architecture uses **Google Agent Development Kit (ADK)** as the agent framework, with **A2A protocol** for agent-to-agent communication, and **tau2-bench** as the evaluation backend.

**Key Insight**: We're building an ADK-based agent that exposes tau2-bench evaluation capabilities via A2A protocol, allowing other agents to request evaluations.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    External A2A Clients                       │
│         (Other agents requesting evaluations)                 │
└─────────────────┬───────────────────────────────────────────┘
                  │ A2A Protocol
                  │ (message/send, agent-card)
┌─────────────────▼───────────────────────────────────────────┐
│              ADK Agent Framework                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  ADK FastAPI Server (with a2a-sdk integration)         │ │
│  │  - Serves /.well-known/agent-card.json                 │ │
│  │  - Handles A2A message/send endpoint                   │ │
│  │  - Translates A2A ↔ ADK Events (automatic)            │ │
│  └────────────────────┬───────────────────────────────────┘ │
│                       │                                        │
│  ┌────────────────────▼───────────────────────────────────┐ │
│  │  Tau2EvalAgent (LlmAgent)                              │ │
│  │  - Orchestrates evaluation requests                    │ │
│  │  - Uses LLM for request parsing/response generation    │ │
│  └────────────────────┬───────────────────────────────────┘ │
│                       │                                        │
│  ┌────────────────────▼───────────────────────────────────┐ │
│  │  Tau2 Evaluation Tools (ADK BaseTool)                  │ │
│  │  - run_tau2_evaluation                                 │ │
│  │  - get_evaluation_results                              │ │
│  │  - list_domains                                        │ │
│  │  - list_available_tasks                                │ │
│  └────────────────────┬───────────────────────────────────┘ │
└───────────────────────┼────────────────────────────────────┘
                        │
┌───────────────────────▼────────────────────────────────────┐
│              tau2-bench Core System                          │
│  - SimulationOrchestrator                                   │
│  - Domain Environments (airline, retail, telecom)           │
│  - Evaluation Metrics & Results Storage                     │
└─────────────────────────────────────────────────────────────┘
```

### Decision

Use **ADK's built-in A2A support** (via `a2a-sdk` optional dependency) instead of implementing custom HTTP client layers. ADK automatically handles:
- A2A protocol translation (A2A Messages ↔ ADK Events)
- Session/context management (A2A `context_id` ↔ ADK `session_id`)
- Agent card serving
- HTTP transport layer

### Rationale

ADK already has mature A2A integration through the `a2a-sdk>=0.3.4` package. Reimplementing HTTP clients and message translation duplicates existing functionality and creates maintenance burden. Using ADK's built-in support ensures protocol compliance and reduces implementation complexity.

---

## 1. A2A Protocol Overview (For Reference)

### Message Structure

**A2A Messages** follow this structure:
- **message_id**: Unique UUID identifier (required)
- **role**: "user" (client→agent) or "agent" (agent→client) (required)
- **parts**: Array of Part objects (TextPart, DataPart, FilePart) (required)
- **context_id**: Optional session identifier for multi-turn conversations
- **metadata**: Optional structured metadata (Struct)

**Part Types**:
- **TextPart**: Plain text content (`{text: "..."`)
- **DataPart**: Structured JSON data (`{data: {...}}`) - used for tool calls
- **FilePart**: File reference with metadata (out of scope for Phase 1)

### Agent Discovery

A2A uses agent card discovery via standard `.well-known` URI:
- Endpoint: `{agent_endpoint}/.well-known/agent-card.json`
- Method: HTTP GET
- Format: JSON document (AgentCard)

**AgentCard** contains:
- `name`, `description`, `url`: Basic metadata
- `capabilities.streaming`: Boolean indicating streaming support
- `securitySchemes`, `security`: Authentication requirements
- `skills`: Optional list of agent capabilities (informational)

### Tool Calling in A2A

**Important**: A2A protocol does NOT standardize tool calling. Tool execution is delegated to complementary protocols like MCP. For our integration, ADK handles tool calling internally, and we expose tau2-bench evaluation functions as ADK tools.

---

## 2. ADK Integration Architecture

### Decision

Implement tau2-bench evaluator as an **ADK LlmAgent** with tau2-bench evaluation functions exposed as **ADK BaseTool** implementations.

### Implementation Approach

**Agent Definition** (`agent.py`):

```python
from google.adk.agents import LlmAgent
from google.adk.tools import BaseTool
from typing import Any
import asyncio

# tau2-bench Tool Implementations

class RunTau2Evaluation(BaseTool):
    """Tool to run tau2-bench agent evaluation"""

    name = "run_tau2_evaluation"
    description = """
    Run a tau2-bench evaluation of a conversational agent.

    Parameters:
    - domain: Evaluation domain (airline, retail, telecom, mock)
    - agent_endpoint: A2A endpoint of agent to evaluate (e.g., https://agent.example.com)
    - user_llm: LLM model for user simulator (e.g., gpt-4o)
    - num_trials: Number of trials per task (default: 1)
    - num_tasks: Number of tasks to evaluate (default: all)
    - task_ids: Optional list of specific task IDs to run
    """

    async def __call__(
        self,
        tool_context,
        domain: str,
        agent_endpoint: str,
        user_llm: str = "gpt-4o",
        num_trials: int = 1,
        num_tasks: int | None = None,
        task_ids: list[str] | None = None
    ) -> dict[str, Any]:
        """Execute tau2-bench evaluation"""
        from tau2.run import run_domain
        from tau2.data_model.simulation import RunConfig

        # Create run configuration
        config = RunConfig(
            domain=domain,
            agent="a2a_agent",  # Use A2A client implementation
            user="user_simulator",
            task_ids=task_ids,
            llm_agent=agent_endpoint,  # A2A agent endpoint
            llm_args_agent={},
            llm_user=user_llm,
            llm_args_user={},
            num_trials=num_trials,
            max_steps=50,
            max_errors=10,
            save_to=None,
            llm_review=False,
            max_concurrency=3,
        )

        # Run evaluations
        results = run_domain(config)

        # Extract metrics from Results object
        # Results contains: timestamp, info, tasks, simulations
        total_simulations = len(results.simulations)
        successful_sims = sum(1 for sim in results.simulations if sim.success)
        success_rate = successful_sims / total_simulations if total_simulations > 0 else 0.0

        return {
            "status": "completed",
            "timestamp": results.timestamp,
            "summary": {
                "total_simulations": total_simulations,
                "total_tasks": len(results.tasks),
                "success_rate": success_rate,
                "successful_simulations": successful_sims,
            },
            "tasks": [{"task_id": task.id, "name": task.name} for task in results.tasks]
        }


class GetEvaluationResults(BaseTool):
    """Retrieve results from a completed evaluation"""

    name = "get_evaluation_results"
    description = "Get detailed results from a tau2-bench evaluation by evaluation_id"

    async def __call__(self, tool_context, evaluation_id: str) -> dict[str, Any]:
        """Load evaluation results from storage"""
        # tau2-bench Results are returned directly from run_domain()
        # This tool would need to load from saved results if save_to was specified
        # For now, return guidance to use run_tau2_evaluation which returns results directly
        return {
            "error": "Results retrieval not yet implemented",
            "message": "Use run_tau2_evaluation tool which returns results directly. " +
                      "For persisted results, tau2-bench saves to JSON files which can be loaded manually."
        }


class ListDomains(BaseTool):
    """List available tau2-bench evaluation domains"""

    name = "list_domains"
    description = "List all available tau2-bench evaluation domains and their descriptions"

    async def __call__(self, tool_context) -> dict[str, Any]:
        """Return available domains"""
        return {
            "domains": [
                {
                    "name": "airline",
                    "description": "Airline customer service (flights, bookings, cancellations)",
                    "num_tasks": 45
                },
                {
                    "name": "retail",
                    "description": "Retail e-commerce (orders, returns, exchanges)",
                    "num_tasks": 39
                },
                {
                    "name": "telecom",
                    "description": "Telecommunications support (technical issues, billing)",
                    "num_tasks": 50
                },
                {
                    "name": "mock",
                    "description": "Simple test domain for development",
                    "num_tasks": 5
                }
            ]
        }


# Define the ADK Agent

root_agent = LlmAgent(
    name="tau2_agent",
    model="gemini-2.0-flash-exp",
    instruction="""
    You are a conversational agent evaluation service powered by tau2-bench.

    You can evaluate other conversational agents across multiple customer service domains:
    - airline: Flight booking, modifications, cancellations
    - retail: Product orders, returns, exchanges
    - telecom: Technical support, billing issues
    - mock: Simple test scenarios

    When a user requests an evaluation:
    1. Clarify the evaluation parameters (domain, agent endpoint, number of tasks)
    2. Use run_tau2_evaluation tool to execute the evaluation
    3. Provide clear, actionable feedback on agent performance
    4. Offer to retrieve detailed results using get_evaluation_results

    Be helpful in explaining evaluation metrics and suggesting improvements.
    """,
    description="Agent evaluation service using tau2-bench framework across airline, retail, and telecom domains",
    tools=[
        RunTau2Evaluation(),
        GetEvaluationResults(),
        ListDomains()
    ]
)
```

**ADK Configuration** (`__init__.py`):

```python
from . import agent
```

### Rationale

This approach:
- **Leverages ADK's strengths**: Agent orchestration, LLM integration, tool management, session persistence
- **Uses ADK's A2A support**: No custom HTTP client needed
- **Clean separation**: tau2-bench remains independent backend, ADK provides agent interface
- **Flexible**: Can extend with more tools (task listing, domain info, result analysis)

---

## 3. A2A Integration via ADK

### Decision

Use ADK's built-in A2A protocol support via the `a2a-sdk` optional dependency and ADK's FastAPI server.

### Implementation Approach

**Install ADK with A2A support**:

```bash
pip install google-adk[a2a]
```

This installs `a2a-sdk>=0.3.4` which provides:
- A2A protocol message handling
- Agent card generation
- A2A endpoint integration with FastAPI

**Start ADK server with A2A enabled**:

```bash
adk web --a2a tau2_agent/
```

Or programmatically:

```python
from google.adk.cli.fast_api import get_fast_api_app

app = get_fast_api_app(
    agents_dir="./tau2_agent",
    web=False,  # Disable web UI for production
    session_service_uri="sqlite:///tau2_sessions.db",
    artifact_service_uri="gs://tau2-artifacts",  # Optional: GCS storage
)

# ADK automatically serves:
# - /.well-known/agent-card.json (agent discovery)
# - A2A protocol endpoints (when a2a-sdk is installed)
```

### Context Management

**A2A context_id ↔ ADK session_id mapping**:

ADK's FastAPI server automatically maps:
- A2A's `context_id` → ADK's `session_id`
- First message with no `context_id` → Create new session
- Subsequent messages with `context_id` → Resume existing session

This mapping is handled internally by the `a2a-sdk` integration in ADK.

### Rationale

Using ADK's built-in A2A support:
- **Eliminates custom HTTP code**: No need for httpx, asyncio.run() wrappers
- **Automatic protocol compliance**: ADK team maintains A2A spec compatibility
- **Production-ready**: Handles authentication, error cases, streaming (future)
- **Consistent with ADK patterns**: Session management, event streaming, tool calling all work as expected

---

## 4. tau2-bench Backend Integration

### Decision

Integrate tau2-bench as **backend evaluation tools** called by ADK agent, not as a parallel agent framework.

### Architecture Pattern

```
A2A Client Request
    ↓
ADK FastAPI Server (a2a-sdk handles protocol)
    ↓
ADK Runner.run_async(session_id, message)
    ↓
Tau2EvalAgent.run_async(invocation_context)
    ↓
RunTau2Evaluation.__call__(domain, agent_endpoint, ...)
    ↓
tau2.run.run_domain(...)
    ↓
tau2 SimulationOrchestrator
    ↓
Evaluation Results → Return to agent → ADK Event → A2A Response
```

### Implementation Considerations

**A2A Client Agent for tau2-bench**:

tau2-bench needs an A2A client implementation to evaluate A2A-enabled agents. This requires:

```python
# tau2/agent/a2a_agent.py
import uuid
from a2a.client.client_factory import ClientFactory
from a2a.types import Message, Part, TextPart, Role
from tau2.agent.base_agent import BaseAgent, LocalAgent
from tau2.data_model.base import AgentState, AssistantMessage, ValidAgentInputMessage

class A2AAgent(LocalAgent):
    """tau2-bench agent that communicates via A2A protocol"""

    def __init__(self, llm: str, llm_args: dict, tools: list, domain_policy: str):
        """
        Initialize A2A agent.

        Args:
            llm: A2A agent endpoint URL (reusing llm parameter for endpoint)
            llm_args: Additional configuration (unused for A2A)
            tools: Available tools for the domain
            domain_policy: Domain-specific policy
        """
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.endpoint = llm  # A2A endpoint passed via llm parameter
        self.client = None
        self.context_id = None

    async def _ensure_client(self):
        """Initialize A2A client if not already done"""
        if self.client is None:
            self.client = await ClientFactory.connect(self.endpoint)

    def get_init_state(self, message_history: list | None = None) -> AgentState:
        """Get initial agent state"""
        return AgentState(message_history=message_history or [])

    async def generate_next_message(
        self,
        message: ValidAgentInputMessage,
        state: AgentState
    ) -> tuple[AssistantMessage, AgentState]:
        """Send message to A2A agent and get response"""
        await self._ensure_client()

        # Convert message to A2A format
        user_message = Message(
            message_id=str(uuid.uuid4()),
            role=Role.user,
            parts=[Part(root=TextPart(text=message.content))],
            context_id=self.context_id
        )

        # Send message and collect response
        response_text = []
        async for event in self.client.send_message(user_message):
            if isinstance(event, Message):
                # Update context for next turn
                self.context_id = event.context_id
                # Extract text from parts
                for part in event.parts:
                    if hasattr(part.root, 'text'):
                        response_text.append(part.root.text)

        # Create assistant message in tau2-bench format
        assistant_message = AssistantMessage(
            role="assistant",
            content="\n".join(response_text),
            tool_calls=[]  # A2A tool calls would need additional parsing
        )

        # Update state
        new_state = AgentState(
            message_history=state.message_history + [message, assistant_message]
        )

        return assistant_message, new_state
```

This allows tau2-bench to evaluate agents that use A2A protocol.

### Rationale

Keeping tau2-bench as backend tools:
- **Preserves tau2-bench architecture**: No breaking changes to existing system
- **Clear responsibility**: ADK handles agent interface, tau2-bench handles evaluation logic
- **Reusable**: Same tau2-bench backend can be used via CLI, API, or ADK agent
- **Testable**: Can test tau2-bench independently of ADK/A2A layers

---

## 5. Session and State Management

### Decision

Use **ADK's native session management** for conversation state, mapping A2A's `context_id` to ADK's `session_id`.

### ADK Session Architecture

ADK provides:
- **InvocationContext**: Execution state container with session, services, agent states
- **SessionService**: Persistence layer (in-memory, database, Vertex AI)
- **Runner**: Orchestrates agent execution with session retrieval/storage

**Session lifecycle**:
1. A2A client sends message with `context_id`
2. ADK's a2a-sdk maps `context_id` → `session_id`
3. Runner retrieves session from SessionService
4. Agent executes with InvocationContext containing session
5. Runner persists events back to SessionService
6. ADK's a2a-sdk includes `context_id` in response

### Implementation

**No custom state management needed**. ADK handles this automatically:

```python
# ADK automatically:
# 1. Maps context_id from A2A message to session_id
# 2. Retrieves/creates session via SessionService
# 3. Populates InvocationContext with session
# 4. Agent can access session via tool_context
# 5. Returns context_id in A2A response
```

**Accessing session in tools**:

```python
class RunTau2Evaluation(BaseTool):
    async def __call__(self, tool_context, domain: str, agent_endpoint: str, ...):
        # Access session if needed
        session = tool_context.session
        user_id = session.user_id

        # Store evaluation metadata in session state
        tool_context.session.state["last_evaluation"] = {
            "domain": domain,
            "agent_endpoint": agent_endpoint,
            "timestamp": datetime.now().isoformat()
        }

        # Run evaluation...
```

### Rationale

ADK's session management:
- **Integrated**: Works seamlessly with A2A protocol mapping
- **Flexible**: Supports multiple backends (in-memory, SQLite, PostgreSQL, Vertex AI)
- **Production-ready**: Handles concurrent sessions, state persistence, cleanup
- **No reinvention**: Leverages ADK's mature state management system

---

## 6. Error Handling and Logging

### Decision

Use **ADK's built-in error handling** and **loguru** for structured logging.

### Implementation Approach

**Error handling in tools**:

```python
from loguru import logger

class RunTau2Evaluation(BaseTool):
    async def __call__(self, tool_context, domain: str, agent_endpoint: str, ...):
        try:
            # Validate inputs
            if domain not in ["airline", "retail", "telecom", "mock"]:
                raise ValueError(f"Invalid domain: {domain}")

            # Run evaluation
            from tau2.run import run_domain
            results = run_domain(config)

            # Calculate success rate from Results object
            total_simulations = len(results.simulations)
            successful = sum(1 for sim in results.simulations if sim.success)
            success_rate = successful / total_simulations if total_simulations > 0 else 0.0

            logger.info(
                "Evaluation completed",
                domain=domain,
                agent_endpoint=agent_endpoint,
                success_rate=success_rate
            )

            return {
                "status": "completed",
                "timestamp": results.timestamp,
                "summary": {
                    "total_simulations": total_simulations,
                    "success_rate": success_rate,
                    "total_tasks": len(results.tasks)
                }
            }

        except ValueError as e:
            logger.error("Invalid evaluation parameters", error=str(e))
            raise

        except Exception as e:
            logger.error(
                "Evaluation failed",
                domain=domain,
                agent_endpoint=agent_endpoint,
                error=str(e),
                exc_info=True
            )
            raise
```

**ADK automatically handles**:
- Wrapping exceptions in appropriate A2A error responses
- Logging request/response cycles
- Tracing execution through InvocationContext

### Rationale

- **Consistency**: Uses same logging infrastructure as ADK
- **Structured logs**: loguru provides JSON-structured logs for production monitoring
- **Error propagation**: ADK translates Python exceptions to A2A protocol errors
- **Observability**: Integration with ADK's OpenTelemetry support for distributed tracing

---

## 7. Testing Strategy

### Decision

Use **ADK's testing patterns** for agent testing, plus **a2a-sdk testing utilities** for protocol validation.

### Implementation Approach

**Unit tests for tools**:

```python
import pytest
from unittest.mock import AsyncMock, patch
from tau2_agent.tools import RunTau2Evaluation

@pytest.mark.asyncio
async def test_run_tau2_evaluation_success():
    """Test successful evaluation execution"""
    tool = RunTau2Evaluation()

    # Mock tau2-bench backend with correct Results structure
    mock_simulation = Mock(success=True)
    mock_task = Mock(id="task-1", name="Test Task")
    mock_results = Mock(
        timestamp="2025-11-24T10:00:00Z",
        simulations=[mock_simulation] * 17,  # 17 successful simulations
        tasks=[mock_task]
    )

    with patch("tau2.run.run_domain", return_value=mock_results):
        result = await tool(
            tool_context=MockToolContext(),
            domain="airline",
            agent_endpoint="https://agent.example.com",
            user_llm="gpt-4o"
        )

    assert result["status"] == "completed"
    assert result["timestamp"] == "2025-11-24T10:00:00Z"
    assert result["summary"]["success_rate"] == 1.0  # All 17 simulations successful
    assert result["summary"]["total_simulations"] == 17
```

**Integration tests with A2A protocol**:

```python
import pytest
from google.adk.cli.fast_api import get_fast_api_app
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_a2a_agent_card():
    """Test agent card is served correctly"""
    app = get_fast_api_app(agents_dir="./tau2_agent")

    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/.well-known/agent-card.json")

        assert response.status_code == 200
        card = response.json()
        assert card["name"] == "tau2_agent"
        assert "run_tau2_evaluation" in str(card["skills"])

@pytest.mark.asyncio
async def test_a2a_message_send():
    """Test A2A message/send endpoint"""
    app = get_fast_api_app(agents_dir="./tau2_agent")

    async with AsyncClient(app=app, base_url="http://test") as client:
        # Mock A2A message
        a2a_message = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": "msg-001",
                    "role": "user",
                    "parts": [{"text": "What domains can you evaluate?"}]
                }
            },
            "id": "req-001"
        }

        response = await client.post("/", json=a2a_message)

        assert response.status_code == 200
        result = response.json()
        assert "result" in result
        assert "airline" in result["result"]["message"]["parts"][0]["text"]
```

### Rationale

- **Standard patterns**: Uses ADK's established testing patterns
- **Fast feedback**: Unit tests run without network/external dependencies
- **Protocol compliance**: Integration tests verify A2A protocol behavior
- **Isolation**: Mock tau2-bench backend for agent testing

---

## 8. Deployment and Operations

### Decision

Deploy as **containerized ADK application** using ADK's deployment tools.

### Implementation Approach

**Local development**:

```bash
# Start development server with A2A support
adk web --a2a tau2_agent/ --port 8000

# Test agent card
curl http://localhost:8000/.well-known/agent-card.json
```

**Production deployment - Cloud Run**:

```bash
# ADK provides deployment command
adk deploy cloud_run tau2_agent/ \
    --project tau2-eval-prod \
    --region us-central1 \
    --session-service-uri postgresql://... \
    --artifact-service-uri gs://tau2-artifacts
```

**Production deployment - Vertex AI Agent Engine**:

```bash
adk deploy agent_engine tau2_agent/ \
    --project tau2-eval-prod \
    --region us-central1
```

### Configuration

**Environment variables**:

```bash
# ADK configuration
GOOGLE_CLOUD_PROJECT=tau2-eval-prod
GOOGLE_CLOUD_LOCATION=us-central1

# LLM API keys (for ADK agent and tau2-bench user simulator)
GOOGLE_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...

# tau2-bench specific
TAU2_DATA_DIR=/data/tau2
TAU2_RESULTS_DIR=/data/tau2/simulations

# Session persistence
SESSION_SERVICE_URI=postgresql://user:pass@host:5432/tau2_sessions
ARTIFACT_SERVICE_URI=gs://tau2-artifacts
```

### Rationale

- **Production-ready**: ADK provides tested deployment patterns
- **Scalable**: Cloud Run auto-scales based on traffic
- **Observable**: Built-in logging, metrics, tracing
- **Managed**: Vertex AI Agent Engine handles infrastructure

---

## Implementation Checklist

### Phase 1: Core ADK Agent

- [ ] Install ADK with A2A support: `pip install google-adk[a2a]`
- [ ] Create agent.py with Tau2EvalAgent (LlmAgent)
- [ ] Implement RunTau2Evaluation tool (BaseTool)
- [ ] Implement GetEvaluationResults tool
- [ ] Implement ListDomains tool
- [ ] Configure agent card metadata
- [ ] Test local agent execution: `adk run tau2_agent/`

### Phase 2: A2A Integration

- [ ] Start ADK server with A2A: `adk web --a2a tau2_agent/`
- [ ] Verify agent card endpoint: `/.well-known/agent-card.json`
- [ ] Test A2A message/send with external client
- [ ] Verify context_id → session_id mapping
- [ ] Test multi-turn conversations via A2A

### Phase 3: tau2-bench Backend Integration

- [ ] Implement A2AAgent extending LocalAgent in tau2-bench (`tau2/agent/a2a_agent.py`)
- [ ] Register A2AAgent with tau2-bench registry as "a2a_agent"
- [ ] Integrate a2a.client.client_factory.ClientFactory for evaluating A2A agents
- [ ] Implement generate_next_message() and get_init_state() methods per BaseAgent interface
- [ ] Test evaluation flow: ADK agent → tau2-bench → A2A target agent
- [ ] Verify Results object structure (timestamp, info, tasks, simulations)

### Phase 4: Testing & Documentation

- [ ] Unit tests for each tool
- [ ] Integration tests for A2A protocol
- [ ] End-to-end evaluation test
- [ ] Agent card documentation
- [ ] API usage examples

### Phase 5: Deployment

- [ ] Configure production session/artifact services
- [ ] Deploy to Cloud Run or Vertex AI
- [ ] Set up monitoring and logging
- [ ] Performance testing

---

## Open Questions

**Resolved**:
- ✅ Use ADK or custom framework? → **ADK (built-in A2A support)**
- ✅ How to handle A2A protocol? → **ADK's a2a-sdk handles automatically**
- ✅ Session management? → **ADK's SessionService with context_id mapping**
- ✅ How to integrate tau2-bench? → **As ADK BaseTool implementations**

**To be determined during implementation**:
- Authentication strategy for A2A clients (API keys, OAuth2, etc.)
- Rate limiting for evaluation requests
- Concurrent evaluation limits
- Result retention policy
- Streaming evaluation progress (Phase 2)

---

## References

### A2A Protocol
- [A2A Protocol Specification](https://a2a-protocol.org/latest/specification/)
- [Agent Discovery - A2A Protocol](https://a2a-protocol.org/latest/topics/agent-discovery/)
- [A2A and MCP Integration](https://a2a-protocol.org/dev/topics/a2a-and-mcp/)

### Google ADK
- [ADK Python GitHub](https://github.com/google/adk-python)
- [ADK Documentation](https://adk.google.dev/)
- [ADK A2A Integration](https://github.com/google/adk-python#a2a-support)
- [ADK Tool Development](https://adk.google.dev/tools/)
- [ADK Deployment Guide](https://adk.google.dev/deployment/)

### tau2-bench
- [tau2-bench GitHub](https://github.com/sierra-research/tau2-bench)
- [tau2-bench Domains](https://github.com/sierra-research/tau2-bench/tree/main/data/tau2/domains)
- [tau2-bench Agent Interface](https://github.com/sierra-research/tau2-bench/tree/main/src/tau2/agent)
