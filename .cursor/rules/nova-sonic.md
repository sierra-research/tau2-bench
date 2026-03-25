# Amazon Nova Sonic Integration Guidelines

The authoritative guide for Nova Sonic-specific rules lives in:

**`src/tau2/voice/audio_native/nova/AGENTS.md`**

Refer to that file for:
- AWS documentation links
- SDK quirks (C-level blocking, Smithy vs boto3)
- Event format gotchas (`contentName`, `interactive: true`)
- VAD / turn detection rules (silence padding)
- SPECULATIVE vs FINAL generation handling
