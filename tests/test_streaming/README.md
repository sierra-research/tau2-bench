# Streaming Components Test Suite

## Overview

Test suite for production streaming, chunking, linearization, and audio-native agent components.

> **Note:** Tests for the experimental text/voice streaming agents (`TextStreamingLLMAgent`,
> `VoiceStreamingLLMAgent`) and the text streaming user simulator (`TextStreamingUserSimulator`)
> have been removed.

## Test Files

### `test_discrete_time_audio_native_agent.py` - DiscreteTimeAudioNativeAgent Tests

Unit tests for the discrete-time audio native agent:
- ✅ Tick-based audio exchange
- ✅ State management (DiscreteTimeAgentState)
- ✅ Tool call handling
- ✅ Speech detection
- ✅ Audio extraction from user messages
- ✅ Response creation
- ✅ Configuration options

### `test_voice_streaming_user_simulator.py` - VoiceStreamingUserSimulator Tests

Unit tests for the voice streaming user simulator:
- ✅ Audio chunk output (receives text, sends audio)
- ✅ State management with VoiceState
- ✅ Turn-taking logic
- ✅ Interruption and backchanneling parameters
- ✅ Speech detection
- ✅ Voice settings validation

### `test_linearization.py` - Tick Linearization Tests

Tests conversion of tick-based conversation history into linear message sequences:
- ✅ Basic linearization (no overlap)
- ✅ Overlapping speech handling
- ✅ Tool message ordering
- ✅ Different linearization strategies
- ✅ `integration_ticks` parameter

### `test_chunking.py` - Chunking and Merging Tests

Tests message chunking (splitting) and merging:
- ✅ Text chunking (by chars, by words)
- ✅ Audio chunking (by samples)
- ✅ Chunk metadata and cost distribution
- ✅ Chunking + merging as inverse operations
- ✅ `audio_script_gold` template handling

### `test_streaming_integration.py` - Integration & Backward Compatibility

Tests backward compatibility and mode validation:
- ✅ Orchestrator defaults to HALF_DUPLEX
- ✅ FULL_DUPLEX requires streaming components
- ✅ Message creation backward compatible

### `test_run_streaming.py` - Streaming Run Tests

Tests for running streaming evaluations end-to-end.

## Running Tests

### Run All Streaming Tests

```bash
pytest tests/test_streaming/ -v
```

### Run Specific Test File

```bash
# Audio native agent tests
pytest tests/test_streaming/test_discrete_time_audio_native_agent.py -v

# Voice user simulator tests
pytest tests/test_streaming/test_voice_streaming_user_simulator.py -v

# Linearization tests
pytest tests/test_streaming/test_linearization.py -v

# Chunking tests
pytest tests/test_streaming/test_chunking.py -v

# Integration tests
pytest tests/test_streaming/test_streaming_integration.py -v
```

## Directory Structure

```
tests/test_streaming/
├── linearization_fixtures.py                    # Shared fixtures for linearization tests
├── test_chunking.py                             # Chunking/merging logic
├── test_discrete_time_audio_native_agent.py     # DiscreteTimeAudioNativeAgent
├── test_linearization.py                        # Tick linearization logic
├── test_run_streaming.py                        # Streaming run tests
├── test_streaming_integration.py                # Backward compatibility
└── test_voice_streaming_user_simulator.py       # VoiceStreamingUserSimulator unit tests
```

## What These Tests Verify

### Core Functionality ✅
- ✅ Tick linearization (overlap handling, tool message ordering)
- ✅ Text and audio chunking/merging
- ✅ Discrete-time audio native agent
- ✅ Voice streaming user simulator
- ✅ Backward compatibility maintained

### Special Cases ✅
- ✅ Tool calls not chunked
- ✅ Empty messages handled
- ✅ Cost/usage on final chunk only
- ✅ State serialization structure
