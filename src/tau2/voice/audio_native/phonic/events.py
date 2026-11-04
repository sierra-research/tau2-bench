"""Event types for the Phonic speech-to-speech WebSocket API.

The Phonic SDK already returns typed payloads; the adapter dispatches on each
payload's ``type`` string. Key event types:
- Audio output: audio_chunk (base64 mu-law 8 kHz, with transcript text)
- Function calls: tool_call
- VAD events: user_started_speaking / user_finished_speaking
- Agent speech: assistant_started_speaking / assistant_finished_speaking
- Session: conversation_created, error

Reference: https://docs.phonic.co/docs/build/with-web-sockets/live_websocket
"""

AUDIO_CHUNK = "audio_chunk"
TOOL_CALL = "tool_call"
USER_STARTED_SPEAKING = "user_started_speaking"
USER_FINISHED_SPEAKING = "user_finished_speaking"
ASSISTANT_STARTED_SPEAKING = "assistant_started_speaking"
ASSISTANT_FINISHED_SPEAKING = "assistant_finished_speaking"
CONVERSATION_CREATED = "conversation_created"
ERROR = "error"
