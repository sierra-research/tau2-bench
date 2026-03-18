# Copyright Sierra
"""Out-of-turn speech generator for voice synthesis."""

import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
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


class OutOfTurnSpeechGenerator:
    """Pre-generates out-of-turn speech (vocal tics, non-directed phrases)."""

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
        """Generate all items concurrently. Non-directed phrases are muffled."""
        if not items:
            return

        self.items = items
        logger.info(f"OutOfTurnSpeechGenerator: loading {len(items)} items")

        with ThreadPoolExecutor(max_workers=min(len(items), 10)) as executor:
            futures = [executor.submit(self._generate_item, item) for item in items]

            for future in as_completed(futures):
                item, audio = future.result()
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
