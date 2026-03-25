# Agent Developer Guide

> **Quick start:** See `examples/agents/` for runnable code you can copy and modify.

## Overview

Agents are the system-under-test in tau2 evals. An agent interacts with a simulated user and uses environment tools (API endpoints) to resolve customer service tasks. The eval framework scores the agent on whether it takes the correct actions and follows domain policy.

There are two communication protocols, each with a corresponding base class:

- **Half-Duplex** (`HalfDuplexAgent`) -- Turn-based text conversations. One party speaks at a time.
- **Full-Duplex** (`FullDuplexAgent`) -- Tick-based streaming (e.g., voice). Both parties can speak simultaneously.

## Architecture

```
HalfDuplexParticipant          FullDuplexParticipant
  (base/participant.py)          (base/participant.py)
        |                              |
  HalfDuplexAgent               FullDuplexAgent
  (base_agent.py)               (base_agent.py)
        |                              |
     LLMAgent              DiscreteTimeAudioNativeAgent
  (llm_agent.py)      (discrete_time_audio_native_agent.py)
```

### HalfDuplexAgent

For turn-based agents. You must implement:

- `get_init_state(message_history) -> StateType` -- Return the initial agent state.
- `generate_next_message(message, state) -> (AssistantMessage, StateType)` -- Given a user or tool message and the current state, produce the next assistant message.

Input types: `UserMessage | ToolMessage | MultiToolMessage`
Output type: `AssistantMessage`

### FullDuplexAgent

For streaming / tick-based agents (e.g., voice). You must implement:

- `get_init_state(message_history) -> StateType` -- Return the initial agent state.
- `get_next_chunk(state, participant_chunk=None, tool_results=None) -> (AssistantMessage, StateType)` -- Process one tick of input and produce one tick of output.

Input types: `UserMessage | ToolMessage | MultiToolMessage`
Output type: `AssistantMessage`

### Shared constructor contract

Both `HalfDuplexAgent` and `FullDuplexAgent` expect these constructor arguments:

```python
def __init__(self, tools: list[Tool], domain_policy: str):
```

- `tools` -- The environment tools (API endpoints) the agent can call.
- `domain_policy` -- The policy text the agent must follow.

If your agent uses an LLM, mix in `LLMConfigMixin` (from `tau2.agent.base.llm_config`) which adds `llm: str` and `llm_args: dict` parameters.

## Core Agent Types

### LLMAgent (text evals, half-duplex)

`LLMAgent` is the standard agent for text-based evals. It wraps an LLM with tool-calling support in a turn-based loop.

- **Base classes:** `LLMConfigMixin`, `HalfDuplexAgent`
- **State:** `LLMAgentState` (holds `system_messages` and `messages`)
- **Constructor:** `tools`, `domain_policy`, `llm`, `llm_args`

```python
from tau2.agent import LLMAgent

# Create the agent
agent = LLMAgent(
    tools=env.get_tools(),
    domain_policy=env.get_policy(),
    llm="gpt-4o",
    llm_args={"temperature": 0.7},
)

# Initialize state
state = agent.get_init_state()

# Turn-based loop (simplified)
response, state = agent.generate_next_message(user_msg, state)
```

### DiscreteTimeAudioNativeAgent (voice evals, full-duplex)

`DiscreteTimeAudioNativeAgent` is the agent for voice evals. It connects to an audio-native API (OpenAI Realtime, Gemini Live, xAI) and exchanges audio in discrete time ticks.

- **Base class:** `FullDuplexAgent`
- **State:** `DiscreteTimeAgentState` (tracks tick count, audio bytes, message history)
- **Constructor:** `tools`, `domain_policy`, plus provider-specific options (`tick_duration_ms`, `provider`, `model`, `modality`, etc.)

```python
from tau2.agent.discrete_time_audio_native_agent import DiscreteTimeAudioNativeAgent

# OpenAI Realtime (default provider)
agent = DiscreteTimeAudioNativeAgent(
    tools=env.get_tools(),
    domain_policy=env.get_policy(),
    tick_duration_ms=1000,
)

# Gemini Live
agent = DiscreteTimeAudioNativeAgent(
    tools=env.get_tools(),
    domain_policy=env.get_policy(),
    tick_duration_ms=1000,
    provider="gemini",
)

# Initialize state (connects to the API)
state = agent.get_init_state()

# Tick-based loop (driven by the orchestrator)
response, state = agent.get_next_chunk(
    state, participant_chunk=incoming_user_chunk, tool_results=pending_tool_results
)
```

## How to Develop a New Agent

### Step 1: Choose the right base class

| Eval type | Base class | Key method |
|-----------|-----------|------------|
| Text (turn-based) | `HalfDuplexAgent` | `generate_next_message()` |
| Voice (tick-based) | `FullDuplexAgent` | `get_next_chunk()` |

### Step 2: Define a state class

Your agent needs a state object to track conversation history and any internal data between turns/ticks. Use a Pydantic `BaseModel`:

```python
from pydantic import BaseModel
from tau2.data_model.message import SystemMessage, Message

class MyAgentState(BaseModel):
    system_messages: list[SystemMessage]
    messages: list[Message]
    # ... any additional fields your agent needs
```

For full-duplex agents, you can extend `StreamingState` (from `tau2.agent.base.streaming`) which includes tick-tracking fields out of the box.

### Step 3: Implement the agent

Here is a minimal half-duplex agent skeleton:

```python
from typing import List, Optional
from tau2.agent.base.llm_config import LLMConfigMixin
from tau2.agent.base_agent import HalfDuplexAgent, ValidAgentInputMessage
from tau2.data_model.message import AssistantMessage, EnvironmentMessage, Message, SystemMessage
from tau2.environment.tool import Tool

class MyAgent(LLMConfigMixin, HalfDuplexAgent["MyAgentState"]):
    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        llm: str,
        llm_args: Optional[dict] = None,
    ):
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
        )

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> "MyAgentState":
        return MyAgentState(
            system_messages=[SystemMessage(role="system", content="...")],
            messages=message_history or [],
        )

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: "MyAgentState"
    ) -> tuple[AssistantMessage, "MyAgentState"]:
        # Your logic here: call LLM, process tool results, etc.
        ...
```

And a minimal full-duplex agent skeleton:

```python
from typing import List, Optional, Tuple
from tau2.agent.base_agent import FullDuplexAgent, ValidAgentInputMessage
from tau2.data_model.message import AssistantMessage, EnvironmentMessage, Message
from tau2.environment.tool import Tool

class MyStreamingAgent(FullDuplexAgent["MyStreamingState"]):
    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        # ... provider-specific params
    ):
        super().__init__(tools=tools, domain_policy=domain_policy)

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> "MyStreamingState":
        # Initialize state and connect to any external APIs
        ...

    def get_next_chunk(
        self,
        state: "MyStreamingState",
        participant_chunk: Optional[ValidAgentInputMessage] = None,
        tool_results: Optional[EnvironmentMessage] = None,
    ) -> Tuple[Optional[AssistantMessage], "MyStreamingState"]:
        # Process one tick: extract input, call API, build response
        ...
```

### Step 4: Provide a factory function and register the agent

Agents are integrated into the eval framework via **factory functions**. A factory encapsulates all construction logic, following the same pattern as domain factories (`get_environment()`). The factory signature is:

```python
def create_agent(tools, domain_policy, **kwargs):
    """
    Factory function called by the eval framework.

    Args:
        tools: Environment tools (API endpoints) the agent can call.
        domain_policy: Policy text the agent must follow.
        **kwargs: Additional arguments from the CLI/config:
            - llm (str): LLM model name (from --agent-llm)
            - llm_args (dict): Additional LLM arguments
            - task (Task): The current task being evaluated
    """
    return MyAgent(tools=tools, domain_policy=domain_policy, ...)
```

There are two ways to register an agent, depending on whether it is a core or community contribution.

**Core agents** register a factory directly in `src/tau2/registry.py`:

```python
from tau2.agent.my_agent import create_my_agent

registry.register_agent_factory(create_my_agent, "my_agent")
```

The name you pass (e.g., `"my_agent"`) is what you will use on the CLI with `--agent`.

## Understanding the Environment

To develop an agent for a specific domain, you first need to understand the domain's policy and available tools. Start by running the environment server for your target domain:

```bash
tau2 domain <domain>
```

This will start a server and automatically open your browser to the API documentation page (ReDoc). Here you can:
- Review the available tools (API endpoints) for the domain
- Understand the policy requirements and constraints
- Test API calls directly through the documentation interface

## Testing Your Agent

Run an eval with your agent using the CLI:

```bash
tau2 run \
  --domain <domain> \
  --agent my_agent \
  --agent-llm <llm_name> \
  --user-llm <llm_name> \
  ...
```

See `tau2 run --help` or `docs/cli-reference.md` for the full list of options.
