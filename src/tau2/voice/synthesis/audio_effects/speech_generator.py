# Copyright Sierra
"""Out-of-turn speech generator for voice synthesis."""

import random
from concurrent.futures import Future
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Optional

from loguru import logger

from tau2.config import DEFAULT_TELEPHONY_RATE
from tau2.data_model.audio import AudioData
from tau2.data_model.audio_effects import UserSpeechInsert
from tau2.data_model.voice import SynthesisConfig
from tau2.data_model.voice_personas import get_elevenlabs_voice_id
from tau2.voice.synthesis.audio_effects.effects import apply_constant_muffling
from tau2.voice.synthesis.synthesize import synthesize_voice
from tau2.voice.utils.audio_preprocessing import resample_audio
from tau2.voice.utils.probability import poisson_should_trigger

from .noise_generator import BackgroundNoiseGenerator, create_background_noise_generator


@dataclass(frozen=True)
class _SpeechCacheKey:
    """Content-addressed identity for one generated speech insert."""

    provider: str
    provider_config: str
    voice_id: str
    target_sample_rate: int
    text: str
    is_muffled: bool


_speech_cache: dict[_SpeechCacheKey, Future[Optional[AudioData]]] = {}
_speech_cache_lock = Lock()


def clear_out_of_turn_speech_cache() -> None:
    """Clear process-cached out-of-turn speech clips."""
    with _speech_cache_lock:
        _speech_cache.clear()


def _provider_config_cache_key(provider_config: Optional[object]) -> str:
    """Serialize provider settings without retaining API key material."""
    if provider_config is None:
        return "none"
    model_dump_json = getattr(provider_config, "model_dump_json", None)
    if callable(model_dump_json):
        return model_dump_json(exclude={"api_key"})
    provider_type = type(provider_config)
    return (
        f"{provider_type.__module__}.{provider_type.__qualname__}:{id(provider_config)}"
    )


class OutOfTurnSpeechGenerator:
    """Load process-cached out-of-turn speech for one simulator state."""

    def __init__(
        self,
        voice_id: str,
        provider: str = "elevenlabs",
        provider_config: Optional[object] = None,
        target_sample_rate: int = DEFAULT_TELEPHONY_RATE,
        seed: int = 42,
        speech_rate: float = 0.0,
    ):
        self.voice_id = voice_id
        self.provider = provider
        self.provider_config = provider_config
        self.target_sample_rate = target_sample_rate
        self.generated_audio: dict[str, AudioData] = {}

        self.speech_rate = speech_rate
        self.trigger_rng = random.Random(seed)
        self.selection_rng = random.Random(seed + 1)
        self.items: list[UserSpeechInsert] = []

    def should_trigger(self, chunk_duration_sec: float) -> Optional[UserSpeechInsert]:
        """Check if an out-of-turn speech effect should trigger for this chunk."""
        if not self.items or self.speech_rate <= 0:
            return None

        if poisson_should_trigger(
            self.speech_rate, chunk_duration_sec, self.trigger_rng
        ):
            return self.selection_rng.choice(self.items)
        return None

    def generate_all(self, items: list[UserSpeechInsert]) -> None:
        """Load all items from the process cache, generating cache misses."""
        if not items:
            return

        self.items = items
        logger.info(f"OutOfTurnSpeechGenerator: loading {len(items)} items")

        # Materialize misses sequentially. Batch-level max_concurrency already
        # parallelizes simulations; another executor here multiplied provider
        # concurrency by up to ten per simulation.
        for item in items:
            item, audio = self._get_or_generate_item(item)
            if audio is not None:
                self.generated_audio[item.text] = audio

        logger.info(
            f"OutOfTurnSpeechGenerator: loaded {len(self.generated_audio)} items"
        )

    def _generate_item(
        self, item: UserSpeechInsert
    ) -> tuple[UserSpeechInsert, Optional[AudioData]]:
        """Generate a single item."""
        try:
            audio = synthesize_voice(
                text=item.text,
                provider=self.provider,
                provider_config=self.provider_config,
            )
            if audio.format.sample_rate != self.target_sample_rate:
                audio = resample_audio(audio, self.target_sample_rate)
            if item.is_muffled:
                audio = apply_constant_muffling(audio)
            return (item, audio)
        except Exception as e:
            logger.warning(f"Failed to generate '{item.text}': {e}")
            return (item, None)

    def _cache_key(self, item: UserSpeechInsert) -> _SpeechCacheKey:
        """Build the cache identity for one fully processed insert."""
        return _SpeechCacheKey(
            provider=self.provider,
            provider_config=_provider_config_cache_key(self.provider_config),
            voice_id=self.voice_id,
            target_sample_rate=self.target_sample_rate,
            text=item.text,
            is_muffled=item.is_muffled,
        )

    def _get_or_generate_item(
        self, item: UserSpeechInsert
    ) -> tuple[UserSpeechInsert, Optional[AudioData]]:
        """Return one cached clip, single-flighting concurrent cache misses."""
        cache_key = self._cache_key(item)
        with _speech_cache_lock:
            future = _speech_cache.get(cache_key)
            is_generator = future is None
            if future is None:
                future = Future()
                _speech_cache[cache_key] = future

        if not is_generator:
            return item, future.result()

        try:
            _, audio = self._generate_item(item)
        except BaseException as error:
            with _speech_cache_lock:
                if _speech_cache.get(cache_key) is future:
                    _speech_cache.pop(cache_key)
            future.set_exception(error)
            raise

        if audio is None:
            # Do not make a transient provider failure sticky for later tasks.
            with _speech_cache_lock:
                if _speech_cache.get(cache_key) is future:
                    _speech_cache.pop(cache_key)
        future.set_result(audio)
        return item, audio

    def get_audio(self, item: UserSpeechInsert) -> Optional[AudioData]:
        """Get pre-generated audio for an item."""
        return self.generated_audio.get(item.text)

    def has_audio(self, item: UserSpeechInsert) -> bool:
        """Check if audio has been generated for an item."""
        return item.text in self.generated_audio


def create_streaming_audio_generators(
    synthesis_config: SynthesisConfig,
    persona_name: str,
    sample_rate: int,
    background_noise_file: Optional[Path] = None,
) -> tuple[BackgroundNoiseGenerator, Optional[OutOfTurnSpeechGenerator]]:
    """Create audio generators for streaming mode."""
    source_config = synthesis_config.source_effects_config
    speech_config = synthesis_config.speech_effects_config

    noise_generator = create_background_noise_generator(
        config=source_config,
        sample_rate=sample_rate,
        background_noise_file=background_noise_file,
    )

    out_of_turn_speech_generator = None
    out_of_turn_items = speech_config.get_out_of_turn_speech_inserts()

    if out_of_turn_items:
        voice_id = get_elevenlabs_voice_id(persona_name)
        provider_config_with_voice = deepcopy(synthesis_config.provider_config)
        provider_config_with_voice.voice_id = voice_id
        out_of_turn_speech_generator = OutOfTurnSpeechGenerator(
            voice_id=voice_id,
            provider=synthesis_config.provider,
            provider_config=provider_config_with_voice,
            target_sample_rate=sample_rate,
        )
        out_of_turn_speech_generator.generate_all(items=out_of_turn_items)

    return noise_generator, out_of_turn_speech_generator
