# Orchestrator Module

This module provides orchestrators for managing simulations between agents, users, and environments. Each orchestrator offers different communication patterns and capabilities.

## Overview

| Orchestrator | Communication Mode | Tool Execution | Participant Interface | Primary Use Case |
|--------------|-------------------|----------------|----------------------|------------------|
| `Orchestrator` | Half-duplex (turn-based) | Synchronous | `generate_next_message()` | Standard benchmarking |
| `FullDuplexOrchestrator` | Full-duplex (streaming) | Synchronous | `get_next_chunk()` | Real-time streaming |

---

## 1. Orchestrator (Half-Duplex)

The standard orchestrator for turn-based communication.

### Communication Pattern

```
Agent ──message──> User ──message──> Agent ──tool_call──> Environment ──result──> Agent
```

Each participant takes turns sending complete messages. Only one party "speaks" at a time.

### Participant Interface

```python
class MyAgent(HalfDuplexAgent):
    def generate_next_message(
        self, 
        message: ValidAgentInputMessage, 
        state: AgentState
    ) -> tuple[AssistantMessage, AgentState]:
        """Generate a complete response to the incoming message."""
        ...
```

### Tool Execution

- **Synchronous**: Tool calls block until complete
- **Immediate**: Results returned in the same step

### Trajectory Structure

```python
trajectory: list[Message]  # Flat list of messages
```

### Compatible Classes

| Role | Classes |
|------|---------|
| Agent | `LLMAgent`, `LLMGTAgent`, `LLMSoloAgent` |
| User | `UserSimulator`, `DummyUser` |

### Usage

```python
from tau2.orchestrator.orchestrator import Orchestrator

orchestrator = Orchestrator(
    domain="airline",
    agent=LLMAgent(tools=tools, domain_policy=policy, llm="gpt-4"),
    user=UserSimulator(llm="gpt-4", instructions=instructions, tools=user_tools),
    environment=environment,
    task=task,
    max_steps=100,
    seed=42,
)
result = orchestrator.run()
```

---

## 2. FullDuplexOrchestrator

Orchestrator for real-time streaming communication where both parties can "speak" simultaneously.

### Communication Pattern

```
     Tick 0          Tick 1          Tick 2
Agent: [chunk0] ───> [chunk1] ───> [chunk2] ───>
User:  [chunk0] ───> [chunk1] ───> [chunk2] ───>
```

Both agent and user generate chunks each tick. Chunks can overlap (simultaneous speech).

### Participant Interface

```python
class MyStreamingAgent(FullDuplexAgent):
    def get_next_chunk(
        self,
        state: StreamingState,
        participant_chunk: Optional[Message] = None,
        tool_results: Optional[EnvironmentMessage] = None,
    ) -> tuple[AssistantMessage, StreamingState]:
        """Generate the next chunk based on incoming chunk and tool results."""
        ...
```

### Tool Execution

- **Synchronous**: Tool calls block and execute immediately
- **Within tick**: Results returned before tick completes

### Trajectory Structure

```python
ticks: list[Tick]  # Tick-grouped events

class Tick(BaseModel):
    tick_id: int
    timestamp: str
    agent_chunk: Optional[AssistantMessage] = None
    user_chunk: Optional[UserMessage] = None
    agent_tool_calls: list[ToolCall] = []
    user_tool_calls: list[ToolCall] = []
    agent_tool_results: list[ToolMessage] = []
    user_tool_results: list[ToolMessage] = []
    user_transcript: Optional[str] = None
    tick_duration_seconds: Optional[float] = None     # Configured tick duration
    wall_clock_duration_seconds: Optional[float] = None  # Actual wall clock time
```

### Compatible Classes

| Role | Classes |
|------|---------|
| Agent | `DiscreteTimeAudioNativeAgent` |
| User | `VoiceStreamingUserSimulator` |

### Usage

```python
from tau2.orchestrator.full_duplex_orchestrator import FullDuplexOrchestrator

orchestrator = FullDuplexOrchestrator(
    domain="airline",
    agent=DiscreteTimeAudioNativeAgent(tools=tools, domain_policy=policy, tick_duration_ms=1000, provider="openai"),
    user=VoiceStreamingUserSimulator(tools=user_tools, instructions=instructions, llm="gpt-4"),
    environment=environment,
    task=task,
    tick_duration_seconds=0.1,  # Optional: real-time pacing
)
result = orchestrator.run()
```

---

## Choosing the Right Orchestrator

| Scenario | Recommended Orchestrator |
|----------|-------------------------|
| Standard benchmarking, simple evaluation | `Orchestrator` |
| Real-time voice/streaming demo | `FullDuplexOrchestrator` |

---

## Output Types Comparison

| Orchestrator | Participant Output | Tool Calls |
|--------------|-------------------|------------|
| `Orchestrator` | `Message` | Embedded in message (`msg.tool_calls`) |
| `FullDuplexOrchestrator` | `Message` | Embedded in message |

---

## Migration Guide

### From `Orchestrator` to `FullDuplexOrchestrator`

1. Replace `LLMAgent` with `DiscreteTimeAudioNativeAgent`
2. Replace `UserSimulator` with `VoiceStreamingUserSimulator`
3. Configure the agent with provider, tick duration, etc. via its constructor (when using the orchestrator directly) or via `AudioNativeConfig` (when using the runner/CLI)

