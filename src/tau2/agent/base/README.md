# Base Agent Components

This directory contains the foundational building blocks for creating conversation participants (agents and users) with support for half-duplex, full-duplex streaming, and voice communication.

## Table of Contents

1. [File Overview](#file-overview)
2. [Protocol Classes](#protocol-classes)
3. [Streaming Capabilities](#streaming-capabilities)
4. [Voice Capabilities](#voice-capabilities)
5. [LLM Configuration](#llm-configuration)
6. [Combining Capabilities](#combining-capabilities)
7. [Usage Examples](#usage-examples)

---

## File Overview

| File | Purpose |
|------|---------|
| `participant.py` | Protocol base classes (`HalfDuplexParticipant`, `FullDuplexParticipant`) |
| `streaming.py` | Streaming state and mixins (`StreamingMixin`, `AudioChunkingMixin`) |
| `voice.py` | Voice state and mixin (`VoiceMixin`) |
| `llm_config.py` | Shared LLM configuration (`LLMConfigMixin`) |

---

## Protocol Classes

### `participant.py`

Defines two protocol base classes representing different communication modes:

### `HalfDuplexParticipant[InputMessageType, OutputMessageType, StateType]`

Turn-based communication where participants take turns speaking.

```python
class HalfDuplexParticipant(ABC, Generic[InputMessageType, OutputMessageType, StateType]):
    
    @abstractmethod
    def generate_next_message(
        self, message: InputMessageType, state: StateType
    ) -> tuple[OutputMessageType, StateType]:
        """Generate a complete response to an input message."""
    
    @abstractmethod
    def get_init_state(self, message_history: Optional[list[Message]] = None) -> StateType:
        """Get initial state."""
    
    def stop(self, message=None, state=None) -> None:
        """Stop the participant (cleanup)."""
    
    @classmethod
    def is_stop(cls, message: OutputMessageType) -> bool:
        """Check if message indicates conversation should stop."""
    
    def set_seed(self, seed: int):
        """Set random seed for reproducibility."""
```

**Use for:** Text chat, simple voice assistants, any turn-based interaction.

---

### `FullDuplexParticipant[InputMessageType, OutputMessageType, StateType]`

Streaming communication where both parties can communicate simultaneously.

```python
class FullDuplexParticipant(ABC, Generic[InputMessageType, OutputMessageType, StateType]):
    
    @abstractmethod
    def get_next_chunk(
        self,
        state: StateType,
        participant_chunk: Optional[InputMessageType] = None,
        tool_results: Optional[EnvironmentMessage] = None,
    ) -> Tuple[Optional[OutputMessageType], StateType]:
        """Process incoming chunk, optionally produce outgoing chunk."""
    
    @abstractmethod
    def get_init_state(self, message_history: Optional[list[Message]] = None) -> StateType:
        """Get initial state."""
    
    def stop(self, participant_chunk=None, state=None, tool_results=None) -> None:
        """Stop the participant (cleanup)."""
    
    @classmethod
    def is_stop(cls, message: OutputMessageType) -> bool:
        """Check if message indicates conversation should stop."""
    
    def set_seed(self, seed: int):
        """Set random seed for reproducibility."""
```

**Use for:** Real-time voice, streaming responses, interruptible conversations.

---

### Voice Protocol Extensions

For participants that need voice capabilities:

| Class | Location | Adds |
|-------|----------|------|
| `VoiceParticipantMixin` | `participant.py` | Abstract `transcribe_voice()`, `synthesize_voice()` |
| `HalfDuplexVoiceAgent` | `base_agent.py` | Combines `VoiceParticipantMixin` + `HalfDuplexAgent` |
| `FullDuplexVoiceAgent` | `base_agent.py` | Combines `VoiceParticipantMixin` + `FullDuplexAgent` |

---

## Streaming Capabilities

### `streaming.py`

Provides state management and implementation mixins for streaming communication.

### `StreamingState[InputMessageType, OutputMessageType]`

Base state class for streaming participants.

```python
class StreamingState(BaseModel, Generic[InputMessageType, OutputMessageType]):
    # Tick-based conversation history
    ticks: list[ParticipantTick[InputMessageType, OutputMessageType]] = []
    
    # Chunk buffers
    input_turn_taking_buffer: list[InputMessageType] = []
    output_streaming_queue: list[OutputMessageType] = []
    
    # Turn-taking timing
    time_since_last_talk: int = 0
    time_since_last_other_talk: int = 0
    tick_count: int = 0
```

**Key Methods:**
- `is_talking` → Check if currently producing output
- `input_total_speech_duration()` → Total input speech received
- `input_ongoing_speech_duration()` → Current continuous speech duration
- `input_interrupt()` → Detect if being interrupted
- `get_linearized_messages()` → Convert tick buffer to message list for LLM

---

### `StreamingMixin[InputMessageType, OutputMessageType, StateType]`

Abstract mixin that implements `get_next_chunk()`.

**Provides:**
- Complete `get_next_chunk()` implementation
- Chunk buffer management
- Turn-taking framework

**Requires Subclasses to Implement:**

```python
@abstractmethod
def _next_turn_taking_action(self, state: StateType) -> TurnTakingAction:
    """Decide next action: 'stop_talking', 'keep_talking', 'generate_message', 'wait', 'backchannel'"""

@abstractmethod
def _perform_turn_taking_action(self, state: StateType, action: TurnTakingAction) 
    -> Tuple[OutputMessageType, StateType]:
    """Execute the action and return next chunk + updated state"""

@abstractmethod
def _create_chunk_messages(self, full_message: OutputMessageType) -> list[OutputMessageType]:
    """Split a full message into chunks"""

@abstractmethod
def speech_detection(self, chunk: InputMessageType) -> bool:
    """Check if the chunk contains speech audio"""

@abstractmethod
def _process_tool_result(self, tool_result: EnvironmentMessage, state: StateType)
    -> Tuple[OutputMessageType, StateType]:
    """Process a tool result and return the next chunk + updated state"""

@abstractmethod
def _emit_waiting_chunk(self, state: StateType)
    -> Tuple[OutputMessageType, StateType]:
    """Emit a chunk while waiting for tool results"""
```

---

### `AudioChunkingMixin[InputMessageType, OutputMessageType, StateType]`

Extends `StreamingMixin` with audio-based chunking. Inherits `chunk_size` from `StreamingMixin`.

```python
def __init__(self, *args, **kwargs)
```

> **Note:** `chunk_size` is defined on `StreamingMixin.__init__(*args, chunk_size: int = 50, **kwargs)` and is passed through via `super()`.

---

### Turn-Taking Policy Helper

```python
def basic_turn_taking_policy(
    state: StateType,
    wait_to_respond_threshold_other: int = 2,
    wait_to_respond_threshold_self: int = 4,
    yield_threshold_when_interrupted: Optional[int] = None,
    yield_threshold_when_interrupting: Optional[int] = None,
    backchannel_min_threshold: Optional[int] = None,
    backchannel_max_threshold: Optional[int] = None,
    backchannel_poisson_rate: Optional[float] = None,
    tick_duration_seconds: float = 0.05,
    # ... additional callback and config params
) -> tuple[BasicActionType, str]
```

**Parameters:**
- `wait_to_respond_threshold_other`: Minimum time to wait since OTHER last spoke before generating a response. Both this AND `wait_to_respond_threshold_self` must be satisfied.
- `wait_to_respond_threshold_self`: Minimum time to wait since SELF last spoke before generating a response. Both this AND `wait_to_respond_threshold_other` must be satisfied.
- `yield_threshold_when_interrupted`: How long self keeps speaking when interrupted by other. If None, cannot be interrupted.
- `yield_threshold_when_interrupting`: How long self keeps speaking when self interrupts other.

**Returns:** Tuple of `(BasicActionType, str)` — action and info message.

---

## Voice Capabilities

### `voice.py`

Provides state and mixin for voice communication.

### `VoiceState`

State class for voice participants.

```python
class VoiceState(BaseModel):
    noise_generator: BackgroundNoiseGenerator  # Required
```

---

### `VoiceMixin[InputMessageType, OutputMessageType, VoiceStateType]`

Implements voice transcription and synthesis.

```python
def __init__(self, *args, voice_settings: VoiceSettings = VoiceSettings(), **kwargs)

def transcribe_voice(self, message: InputMessageType) -> InputMessageType
def synthesize_voice(
    self,
    message: OutputMessageType,
    state: VoiceStateType,
    effects_turn_idx: int = 0,
    add_background_noise: bool = True,
    add_burst_noise: bool = True,
    add_telephony_format: bool = True,
    add_channel_effects: bool = True,
    apply_speech_effects: bool = True,
) -> OutputMessageType
```

**Features:**
- Uses configured transcription service (Deepgram, OpenAI Whisper)
- Uses configured synthesis service (ElevenLabs, OpenAI TTS)
- Optionally saves audio files and transcripts
- Adds background noise during synthesis if configured

---

## LLM Configuration

### `llm_config.py`

Shared LLM configuration for any LLM-powered participant.

```python
class LLMConfigMixin:
    """Used by both agents and user simulators."""
    
    def __init__(self, *args, llm: str, llm_args: Optional[dict] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm = llm
        self.llm_args = deepcopy(llm_args) if llm_args is not None else {}
    
    def set_seed(self, seed: int):
        """Set the seed for the LLM."""
        self.llm_args["seed"] = seed
```

---

## Combining Capabilities

### Inheritance Order

When combining mixins, follow this order (Python MRO):

```
1. Config mixins (LLMConfigMixin, AudioNativeConfigMixin)
2. Capability mixins (VoiceMixin, AudioChunkingMixin)
3. Protocol base class (FullDuplexAgent, HalfDuplexUser)
```

### State Composition

Combine state classes using multiple inheritance:

```python
# Streaming only
class MyStreamingState(MyBaseState, StreamingState[InputMsg, OutputMsg]):
    pass

# Voice only
class MyVoiceState(MyBaseState, VoiceState):
    pass

# Both streaming and voice
class MyVoiceStreamingState(MyBaseState, StreamingState[InputMsg, OutputMsg], VoiceState):
    pass
```

---

## Usage Examples

### Example 1: Audio Streaming Agent

```python
from tau2.agent.base_agent import FullDuplexAgent
from tau2.agent.base.streaming import AudioChunkingMixin, StreamingState
from tau2.agent.base.llm_config import LLMConfigMixin

# Define state
class MyAgentStreamingState(LLMAgentState, StreamingState[UserMessage, AssistantMessage]):
    """Combines base agent state with streaming capabilities."""

# Specialize chunking mixin for your message types
class MyAgentAudioChunkingMixin(
    AudioChunkingMixin[UserMessage, AssistantMessage, MyAgentStreamingState]
):
    pass

# Create concrete agent
class MyAudioStreamingAgent(
    LLMConfigMixin,
    MyAgentAudioChunkingMixin,
    FullDuplexAgent[MyAgentStreamingState],
):
    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        llm: str,
        llm_args: Optional[dict] = None,
        chunk_size: int = 50,
    ):
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
            chunk_size=chunk_size,
        )
    
    def get_init_state(self, message_history=None) -> MyAgentStreamingState:
        return MyAgentStreamingState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history or [],
            input_turn_taking_buffer=[],
            output_streaming_queue=[],
        )
    
    def _next_turn_taking_action(self, state: MyAgentStreamingState) -> TurnTakingAction:
        action, _ = basic_turn_taking_policy(state, wait_to_respond_threshold_other=2)
        return TurnTakingAction(action=action, info="")
    
    def _perform_turn_taking_action(
        self, state: MyAgentStreamingState, action: TurnTakingAction
    ):
        if action.action == "generate_message":
            # Merge buffered input and generate response
            merged_input = merge_homogeneous_chunks(state.input_turn_taking_buffer)
            response = self._generate_response(merged_input, state)
            state.input_turn_taking_buffer = []
            
            # Split into chunks
            chunks = self._create_chunk_messages(response)
            state.output_streaming_queue.extend(chunks)
            return state.output_streaming_queue.pop(0), state
            
        elif action.action == "keep_talking":
            return state.output_streaming_queue.pop(0), state
            
        elif action.action == "stop_talking":
            state.output_streaming_queue = []
            return None, state
            
        else:  # wait
            return None, state
```

---

### Example 2: Voice Streaming Agent

```python
from tau2.agent.base_agent import FullDuplexVoiceAgent
from tau2.agent.base.streaming import AudioChunkingMixin, StreamingState
from tau2.agent.base.voice import VoiceMixin, VoiceState
from tau2.agent.base.llm_config import LLMConfigMixin

# Define state (combines streaming + voice)
class MyVoiceStreamingState(
    LLMAgentState,
    StreamingState[UserMessage, AssistantMessage],
    VoiceState,
):
    pass

# Specialize mixins
class MyAudioChunkingMixin(
    AudioChunkingMixin[UserMessage, AssistantMessage, MyVoiceStreamingState]
):
    pass

# Create concrete agent
class MyVoiceStreamingAgent(
    LLMConfigMixin,
    VoiceMixin[UserMessage, AssistantMessage, MyVoiceStreamingState],
    MyAudioChunkingMixin,
    FullDuplexVoiceAgent[MyVoiceStreamingState],
):
    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        llm: str,
        llm_args: Optional[dict] = None,
        voice_settings: VoiceSettings = VoiceSettings(),
        chunk_size: int = 50,
    ):
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
            voice_settings=voice_settings,
            chunk_size=chunk_size,
        )
    
    def get_init_state(self, message_history=None) -> MyVoiceStreamingState:
        return MyVoiceStreamingState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history or [],
            input_turn_taking_buffer=[],
            output_streaming_queue=[],
            noise_generator=BackgroundNoiseGenerator(sample_rate=8000),
        )
    
    # ... implement _next_turn_taking_action and _perform_turn_taking_action
```

---

### Example 3: Half-Duplex LLM Agent

```python
from tau2.agent.base_agent import HalfDuplexAgent
from tau2.agent.base.llm_config import LLMConfigMixin

class MyHalfDuplexAgent(
    LLMConfigMixin,
    HalfDuplexAgent[LLMAgentState],
):
    def __init__(
        self,
        tools: list[Tool],
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
    
    def get_init_state(self, message_history=None) -> LLMAgentState:
        return LLMAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history or [],
        )
    
    def generate_next_message(
        self, message: ValidAgentInputMessage, state: LLMAgentState
    ) -> tuple[AssistantMessage, LLMAgentState]:
        # Build messages for LLM
        messages = state.system_messages + state.messages + [message]
        
        # Call LLM
        response = generate(model=self.llm, messages=messages, **self.llm_args)
        
        # Update state
        state.messages.append(message)
        state.messages.append(response)
        
        return response, state
```

---

## Summary

| Component | Purpose | Key Methods/Fields |
|-----------|---------|-------------------|
| `HalfDuplexParticipant` | Turn-based protocol | `generate_next_message()` |
| `FullDuplexParticipant` | Streaming protocol | `get_next_chunk()` |
| `StreamingMixin` | Streaming implementation | `get_next_chunk()`, turn-taking |
| `AudioChunkingMixin` | Audio chunking | `_create_chunk_messages()` |
| `VoiceMixin` | Voice transcription/synthesis | `transcribe_voice()`, `synthesize_voice()` |
| `LLMConfigMixin` | LLM configuration | `llm`, `llm_args`, `set_seed()` |
| `StreamingState` | Streaming data | `input_turn_taking_buffer`, `output_streaming_queue` |
| `VoiceState` | Voice data | `noise_generator` (required) |

**To create a new participant:**

1. Choose protocol: `HalfDuplexAgent`/`FullDuplexAgent` (or User equivalents)
2. Add capabilities: `LLMConfigMixin`, `VoiceMixin`, `AudioChunkingMixin`, etc.
3. Define state class: Combine base state with capability states
4. Implement required methods: `get_init_state()`, and either `generate_next_message()` or turn-taking methods
5. Follow inheritance order: Config mixins → Capability mixins → Protocol base
