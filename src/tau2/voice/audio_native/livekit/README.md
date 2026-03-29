# LiveKit Cascaded Voice Provider

This module implements a cascaded STT → LLM → TTS voice pipeline using LiveKit's plugin ecosystem.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CascadedVoiceProvider (provider.py)              │
│                                                                     │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐            │
│  │  Deepgram    │   │   OpenAI/    │   │  Deepgram/   │            │
│  │  STT + VAD   │──▶│  Anthropic   │──▶│  ElevenLabs  │            │
│  │              │   │     LLM      │   │     TTS      │            │
│  └──────────────┘   └──────────────┘   └──────────────┘            │
│         │                  │                  │                     │
│         ▼                  ▼                  ▼                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    CascadedEvent stream                      │   │
│  │  (SPEECH_STARTED, TRANSCRIPT_*, LLM_*, TTS_*, TOOL_CALL)    │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│              LiveKitCascadedAdapter (discrete_time_adapter.py)      │
│                                                                     │
│  • Tick-based interface for simulation                              │
│  • Audio buffering for tick alignment                               │
│  • Event → TickResult mapping                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Intricate Decisions for Cascaded Pipeline

This section documents key design decisions, their rationale, and available tuning knobs.

### 1. Turn Endpointing (When to Trigger LLM)

**Decision**: Rely on Deepgram's integrated VAD with optional additional silence padding.

**Rationale**: Deepgram's VAD is trained on conversational data and handles edge cases (hesitations, filled pauses) better than naive energy-based VAD. We expose an additional silence buffer for experiments requiring more conservative endpointing.

| Knob | Location | Default | Description |
|------|----------|---------|-------------|
| `endpointing_ms` | `DeepgramSTTConfig` | 25ms | Deepgram's silence threshold before finalizing transcript |
| `vad_events` | `DeepgramSTTConfig` | True | Enable VAD events (speech_started/speech_ended) |
| `additional_silence_ms` | `TurnTakingConfig` | 0ms | Extra wait after Deepgram's endpoint before triggering LLM |
| `min_transcript_chars` | `TurnTakingConfig` | 1 | Minimum characters to trigger LLM (filters "um", "uh") |

**Example**: For slower-paced conversations, increase `endpointing_ms` to 100-200ms.

---

### 2. Interruption / Barge-in Handling

**Decision**: Detect barge-in via VAD events; cancel TTS playback and pending LLM generation.

**Rationale**: Users expect immediate responsiveness when they interrupt. The agent should stop speaking and listen.

| Knob | Location | Default | Description |
|------|----------|---------|-------------|
| `allow_interruptions` | `TurnTakingConfig` | True | Whether barge-in is allowed during TTS |
| `interruption_threshold_ms` | `TurnTakingConfig` | 200ms | Minimum speech duration to trigger interrupt |

**Implementation**: When `SPEECH_STARTED` is received during `ProviderState.SPEAKING`:
1. Cancel TTS stream
2. Discard buffered audio
3. Emit `INTERRUPTED` event
4. Transition to `LISTENING` state

---

### 3. Sentence Chunking for TTS

**Decision**: Use LiveKit TTS plugin's built-in sentence detection via `stream.push_text()` / `stream.end_input()`.

**Rationale**: LiveKit plugins handle sentence boundary detection internally, optimizing for natural prosody breaks. We stream the full LLM response and let the TTS plugin chunk appropriately.

| Knob | Location | Default | Description |
|------|----------|---------|-------------|
| (built-in) | LiveKit TTS plugins | - | Automatic sentence/phrase chunking |

**Future**: If finer control is needed, we could intercept LLM tokens and manually chunk at sentence boundaries before pushing to TTS.
