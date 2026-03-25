# AGENTS.md — src/tau2/agent/

> See `README.md` for the full developer guide with code examples.
> See `base/README.md` for protocol classes, streaming, and voice mixins.

## Rules for Working in This Directory

### Choose the Right Base Class

| Communication mode | Base class | Key method |
|-------------------|-----------|------------|
| Half-duplex (turn-based text) | `HalfDuplexAgent` | `generate_next_message()` |
| Full-duplex (streaming/voice) | `FullDuplexAgent` | `get_next_chunk()` |

### Constructor Contract

Both base classes MUST accept: `__init__(self, tools: list[Tool], domain_policy: str)`.
For LLM-powered agents, add `LLMConfigMixin` which provides `llm: str` and `llm_args: dict`.

### MRO (Mixin Inheritance Order) Is Critical

Python MRO determines which `__init__` runs. Follow this exact order:

```python
class MyAgent(
    LLMConfigMixin,           # 1. Config mixins first
    VoiceMixin,               # 2. Capability mixins second
    AudioChunkingMixin,       # 2. (continued)
    FullDuplexAgent[StateType] # 3. Protocol base class last
):
```

Getting this wrong causes subtle bugs where `__init__` arguments silently disappear.

### Message Constraints

- A message MUST have either `content` (text) OR `tool_calls`, NEVER both simultaneously.
- `AssistantMessage.is_tool_call()` returns `True` when `tool_calls` is non-empty.
- Tool calls are atomic — never chunk them in streaming agents.

### State Classes

- Use Pydantic `BaseModel` for all state classes.
- For full-duplex agents, extend `StreamingState` (from `base/streaming.py`).
- State composition uses multiple inheritance: `class MyState(BaseState, StreamingState, VoiceState)`.

### Registration

Every agent needs a factory function and registry entry:

```python
# Factory signature:
def create_agent(tools, domain_policy, **kwargs):
    # kwargs may include: llm, llm_args, task, audio_native_config, etc.
    return MyAgent(tools=tools, domain_policy=domain_policy, ...)

# In src/tau2/registry.py:
registry.register_agent_factory(create_my_agent, "my_agent")
```

The name passed to `register_agent_factory` is what users pass via `--agent` on the CLI.

### Stop Signals

- Half-duplex: `is_stop()` classmethod checks for stop conditions.
- Full-duplex (`DiscreteTimeAudioNativeAgent`): detects `transfer_to_human_agents` tool calls (appends `###STOP###` to message content), then `is_stop()` checks for `###STOP###` in `message.content`.

### Testing

- Unit tests: `tests/test_agent.py`
- Streaming agent tests: `tests/test_streaming/`
- Audio native agent tests: `tests/test_streaming/test_discrete_time_audio_native_agent.py`
