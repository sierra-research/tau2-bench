# Copyright Sierra
"""Text manipulation utilities for speech effects.

These functions modify text before TTS synthesis.
"""

import random

from loguru import logger

from tau2.data_model.audio_effects import UserSpeechInsert
from tau2.voice_config import MIN_WORDS_FOR_VOCAL_TICS


def insert_speech_text(
    text: str,
    speech_insert: UserSpeechInsert,
    rng: random.Random,
    min_words: int = MIN_WORDS_FOR_VOCAL_TICS,
    in_turn: bool = True,
) -> str:
    """Insert speech text at a random position.

    For in-turn insertion, only vocal tics are appropriate. Non-directed phrases
    should only be used out-of-turn (pre-rendered audio during silence).
    """
    if in_turn and speech_insert.type == "non_directed_phrase":
        logger.warning(
            f"Non-directed phrase '{speech_insert.text}' used in-turn. "
            "Non-directed phrases should only be used out-of-turn."
        )

    words = text.split()
    if len(words) < min_words:
        return text

    insert_position = rng.randint(1, len(words) - 1)
    words.insert(insert_position, speech_insert.text)
    return " ".join(words)
