"""Core voice synthesis (TTS) functions."""

import os
import threading

from dotenv import load_dotenv

from tau2.config import DEFAULT_TTS_MAX_CONCURRENCY
from tau2.data_model.audio import AudioData
from tau2.data_model.voice import ElevenLabsTTSConfig
from tau2.utils.retry import tts_retry
from tau2.voice.utils.elevenlabs_utils import tts_elevenlabs

load_dotenv()

ProviderConfig = ElevenLabsTTSConfig


def _tts_concurrency_limit() -> int:
    """Process-wide TTS concurrency ceiling; see DEFAULT_TTS_MAX_CONCURRENCY."""
    raw = os.environ.get("TAU2_TTS_MAX_CONCURRENCY")
    if raw is None:
        return DEFAULT_TTS_MAX_CONCURRENCY
    try:
        value = int(raw)
    except ValueError as e:
        raise ValueError(
            f"TAU2_TTS_MAX_CONCURRENCY must be an integer, got {raw!r}"
        ) from e
    if value < 1:
        raise ValueError(f"TAU2_TTS_MAX_CONCURRENCY must be >= 1, got {value}")
    return value


# Bounds provider calls in flight across EVERY synthesis path in the process —
# the streaming user simulator, the legacy half-duplex simulator, the CLI, and
# OutOfTurnSpeechGenerator's thread pool. This module is the one choke point all
# four pass through, which is why the limit lives here rather than at each fan-out
# site: capping a single thread pool would leave the other three unbounded.
#
# Read once at import so every caller shares one semaphore.
_TTS_SEMAPHORE = threading.BoundedSemaphore(_tts_concurrency_limit())


@tts_retry
def synthesize_voice(
    text: str,
    provider: str,
    provider_config: ProviderConfig,
) -> AudioData:
    """Synthesize voice from text using the specified configuration.

    Serialized against DEFAULT_TTS_MAX_CONCURRENCY: providers cap concurrent
    requests per subscription and answer 429 over it, which costs a caller
    utterance rather than just time.
    """
    if provider != "elevenlabs":
        # Raised before acquiring: an unsupported provider is a config error,
        # not work, and must not queue behind in-flight synthesis.
        raise ValueError(f"Unsupported synthesis provider: {provider}")

    # Acquired INSIDE the retried function, so a backing-off attempt releases
    # its slot while it sleeps. Holding across the retry would let a few
    # rate-limited calls idle the whole allowance and stall every other caller.
    with _TTS_SEMAPHORE:
        audio_data = tts_elevenlabs(text=text, config=provider_config)

    if not audio_data.format.is_pcm16:
        raise ValueError(
            f"TTS must output PCM_S16LE, got {audio_data.format.encoding}. "
            "Configure the provider to use PCM output format."
        )

    return audio_data
