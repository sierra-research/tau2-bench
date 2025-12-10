# Agentified Tau-Bench: A Standardized Agent Evaluation Platform

> **Developed in relation to the [AgentX-AgentBeats Competition](https://rdi.berkeley.edu/agentx-agentbeats) hosted by Berkeley RDI**

This project demonstrates how to "agentify" benchmarks by transforming [Tau-Bench](https://github.com/sierra-research/tau2-bench) into an agent-based service using the [A2A (Agent-to-Agent)](https://a2a.ai/) protocol. It ports the original [AgentBeats example](https://github.com/agentbeats/agentify-example-tau-bench) to tau2-bench, enabling any A2A-compatible agent to be automatically evaluated without custom integration code.

## Overview

**Agentification** transforms a traditional benchmark into an agent-based service. Instead of requiring each agent to implement benchmark-specific code, the benchmark itself becomes an agent that can interact with other agents through a standard protocol.

### Why Agentify Benchmarks?

As AI systems evolve from simple models to complex **agentic systems** capable of reasoning, taking actions, and interacting with environments, traditional evaluation approaches fall short:

**The Problem:**
- 🔧 **Integration Friction**: Each benchmark requires substantial agent modification and custom adapters
- 📊 **Fragmented Ecosystem**: No unified way to compare agents across benchmarks
- 🔄 **Poor Reproducibility**: Stateful interactions and varying configurations lead to inconsistent results
- 🔍 **Discovery Challenges**: Finding and running the right benchmarks is time-consuming

**The Solution - Agentification:**
- 🔌 **Universal Compatibility**: Build your agent once, test it anywhere using the A2A protocol
- 🏛️ **Standardized Interface**: Benchmarks expose a common API, agents speak a common language
- 🎯 **Automated Evaluation**: The benchmark orchestrates the entire test lifecycle
- 🌐 **Interoperability**: Agents and benchmarks from different teams work together seamlessly

By agentifying Tau-Bench, we enable a future where agent evaluation is as simple as sending a message—no integration code, no custom adapters, just standardized, reproducible assessment.

### Architecture

The system uses two types of agents:

- **Green Agent** 🟢: The assessment manager that administers Tau-Bench evaluations
  - Receives requests specifying which agent to test and what tasks to run
  - Sets up the Tau-Bench environment with the specified configuration
  - Orchestrates the conversation between the benchmark and the target agent
  - Evaluates responses and reports results

- **White Agent** ⚪: The target agent being tested
  - Can be any agent that implements the A2A protocol
  - Receives task instructions and responds with tool calls or messages
  - Operates without knowledge of being benchmarked

### How It Works

1. Send a message to the **green agent** with:
   - The URL of the white agent to test
   - The benchmark configuration (domain, task_id, etc.)

2. The green agent:
   - Instantiates a Tau-Bench environment
   - Forwards user messages to the white agent
   - Collects the white agent's responses
   - Evaluates performance using Tau-Bench's scoring system
   - Returns the results

3. The white agent simply responds to messages using its tools, unaware it's being evaluated.

## Installation

```bash
# Install dependencies
uv sync
```

## Configuration

Create a `.env` file with your API keys:

```bash
OPENAI_API_KEY=your_key_here
```

## Usage

### Option 1: Launch Complete Evaluation (Quickstart)

Run both agents and execute a full evaluation cycle:

```bash
uv run tau-bench-agent launch
```

This will:
1. Start the green agent on port 9001
2. Start the white agent on port 9002
3. Send a test task to the green agent
4. Display the evaluation results

### Option 2: Run Agents Separately

Start agents in separate terminals for more control:

**Terminal 1 - Start the green agent:**
```bash
uv run tau-bench-agent green
```

**Terminal 2 - Start the white agent:**
```bash
uv run tau-bench-agent white
```

**Terminal 3 - Send a benchmark request:**
```python
import asyncio
from agentify_tau_bench.utils import a2a_send_message

async def benchmark():
    message = """
    <white_agent_url>http://localhost:9002</white_agent_url>
    <env_config>
    {
      "domain": "retail",
      "task_id": "1",
      "max_steps": 100,
      "user_llm": "openai/gpt-4o",
      "user_llm_args": {"temperature": 0.0}
    }
    </env_config>
    """
    response = await a2a_send_message("http://localhost:9001", message)
    print(response)

asyncio.run(benchmark())
```

## Project Structure

```
src/agentify_tau_bench/
├── green_agent/       # Assessment manager agent
│   ├── agent.py       # Green agent implementation
│   └── tau_green_agent.toml
├── white_agent/       # Target agent being tested
│   └── agent.py       # White agent implementation
├── utils/             # Utility functions
│   ├── utils.py       # Tag parsing
│   └── a2a_utils.py   # A2A communication helpers
├── launcher.py        # Evaluation coordinator
└── main.py            # CLI entry point

tests/
├── test_green_agent.py
├── test_white_agent.py
├── test_launcher.py
└── test_utils.py
```

## Development

### Run Tests

```bash
# Run all tests
make test

# Run tests with verbose output
uv run pytest tests -v
```

### Linting and Formatting

```bash
# Check code quality
make lint

# Auto-format code
make format

# Run both linting and tests
make check
```

### Clean Up

```bash
# Remove cache files
make clean
```

## Available Commands

The `tau-bench-agent` command (defined in `pyproject.toml`) provides these subcommands:

- `uv run tau-bench-agent green` - Start the green agent (assessment manager)
- `uv run tau-bench-agent white` - Start the white agent (target being tested)
- `uv run tau-bench-agent launch` - Launch complete evaluation workflow

## Benefits of Agentification

1. **Standardization**: Any A2A-compatible agent can be tested without modification
2. **Separation of Concerns**: The benchmark and agent implementations are independent
3. **Reusability**: The same green agent can test multiple white agents
4. **Scalability**: Agents can be distributed across different machines/services
5. **Language Agnostic**: White agents can be implemented in any language supporting A2A

## Acknowledgments

This project builds upon the [agentify-example-tau-bench](https://github.com/agentbeats/agentify-example-tau-bench) created by the AgentBeats team for the Berkeley RDI Agentic AI MOOC.

## Related Links

- [AgentX-AgentBeats Competition](https://rdi.berkeley.edu/agentx-agentbeats) - Berkeley RDI's competition for advancing agentic AI evaluation
- [Original AgentBeats Example](https://github.com/agentbeats/agentify-example-tau-bench) - The original tau-bench v1 implementation
- [Tau-Bench v2](https://github.com/sierra-research/tau2-bench) - The benchmark this project uses
- [A2A Protocol](https://a2a.ai/) - Agent-to-Agent communication protocol
- [A2A SDK](https://github.com/a2a-protocol/a2a-sdk) - Python SDK for building A2A agents
