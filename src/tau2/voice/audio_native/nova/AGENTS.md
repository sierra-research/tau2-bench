# AGENTS.md — src/tau2/voice/audio_native/nova/

> See `README.md` for full architecture, event types, audio conversion, and usage examples.
> See parent `../AGENTS.md` for general audio native provider rules.

## AWS Documentation (Primary Source)

Always check the official AWS documentation first when working on Nova Sonic:

- **Getting started**: https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-getting-started.html
- **Code examples**: https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-code-examples.html
- **System prompts**: https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-system-prompts.html
- **Core concepts**: https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-core-concepts.html
- **Event lifecycle**: https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-event-lifecycle.html
- **Event flow**: https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-event-flow.html
- **Input events**: https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-input-events.html
- **Output events**: https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-output-events.html
- **Barge-in**: https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-barge-in.html
- **Turn-taking**: https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-turn-taking.html
- **Tool configuration**: https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-tool-configuration.html
- **Async tools**: https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-async-tools.html

If AWS docs don't have the answer, use web search or the LiveKit `aws_sdk_bedrock_runtime` implementation as a secondary reference (located in `.venv/lib/python3.12/site-packages/livekit/plugins/aws/experimental/realtime/`).

## Nova-Specific Gotchas

### SDK

- Use `aws_sdk_bedrock_runtime` (Smithy-based SDK), NOT boto3 for the streaming API.
- The `receive()` method **blocks at C-level** and does not respect asyncio timeouts. Handle receive in background tasks within the same event loop.

### Event Format

- Use `contentName` (NOT `contentId`) for content identification.
- Set `interactive: true` on content that should trigger responses.
- System prompts also need `interactive: true`.

### Audio Format

- **Input**: 16kHz PCM16 mono (what we send to Nova)
- **Output**: 24kHz PCM16 mono (what Nova sends back)
- Conversion to/from telephony (8kHz μ-law) is handled by `audio_utils.py`.

### VAD / Turn Detection

- Nova Sonic only supports server-side VAD. MANUAL mode is NOT supported.
- **CRITICAL**: After sending speech audio, send ~1 second of silence to trigger VAD end-of-turn detection. Do not immediately end the audio content block after speech — let VAD detect the turn end naturally.
- Keep the audio stream open for continuous conversation.

### SPECULATIVE vs FINAL

- Nova generates speculative content that may be revised. Only process content from `FINAL` generation stage. SPECULATIVE content must be logged and ignored.
- On interruption (barge-in), clear `_final_content_ids` since the new response will have new IDs.
