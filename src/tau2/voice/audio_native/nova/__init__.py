"""Amazon Nova Sonic audio-native integration.

Nova 2 Sonic provides real-time speech-to-speech capabilities through
Amazon Bedrock's bidirectional streaming API.

Key features:
- Input: LPCM 16kHz audio (requires conversion from G.711 μ-law for telephony)
- Output: LPCM 24kHz audio (requires conversion to G.711 μ-law for telephony)
- AWS SigV4 authentication via boto3
- Tool/function calling support
- Barge-in (interruption) support with server-side VAD
- 1M token context window

Reference: AWS Bedrock Nova Sonic documentation
https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-getting-started.html
"""

from tau2.voice.audio_native.nova.discrete_time_adapter import DiscreteTimeNovaAdapter
from tau2.voice.audio_native.nova.events import (
    NovaSonicEvent,
    parse_nova_event,
)
from tau2.voice.audio_native.nova.provider import NovaSonicProvider, NovaVADConfig

__all__ = [
    "DiscreteTimeNovaAdapter",
    "NovaSonicEvent",
    "NovaVADConfig",
    "NovaSonicProvider",
    "parse_nova_event",
]
