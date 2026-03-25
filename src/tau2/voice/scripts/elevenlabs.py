#!/usr/bin/env python3
"""
Simple CLI script to run ElevenLabs TTS synthesis with audio playback.

Usage:
    python elevenlabs.py "Hello, this is a test"
    python elevenlabs.py "Hello, world" --voice-id "avQFHuQU7IjJf0u5MMBq" --no-audio-tags
"""

import argparse
import sys

import pyaudio
from loguru import logger

from tau2.data_model.audio import AudioData
from tau2.voice.utils.elevenlabs_utils import (
    ElevenLabsTTSConfig,
    VoiceSettings,
    tts_elevenlabs,
)
from tau2.voice_config import (
    ELEVENLABS_AUDIO_TAGS_PROBABILITY as DEFAULT_ELEVENLABS_AUDIO_TAGS_PROBABILITY,
)


def play_audio(audio_data: AudioData) -> None:
    """Play AudioData using pyaudio."""
    p = pyaudio.PyAudio()

    # Get format from audio data
    if audio_data.format.sample_width == 1:
        pyaudio_format = pyaudio.paInt8
    elif audio_data.format.sample_width == 2:
        pyaudio_format = pyaudio.paInt16
    elif audio_data.format.sample_width == 4:
        pyaudio_format = pyaudio.paInt32
    else:
        logger.error(f"Unsupported sample width: {audio_data.format.sample_width}")
        p.terminate()
        sys.exit(1)

    stream = p.open(
        format=pyaudio_format,
        channels=audio_data.format.channels,
        rate=audio_data.format.sample_rate,
        output=True,
    )

    try:
        logger.info("Playing audio...")
        stream.write(audio_data.data)
        logger.info("Playback complete!")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


def main():
    parser = argparse.ArgumentParser(
        description="Generate speech using ElevenLabs TTS and play it back",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument("text", type=str, help="Text to synthesize")

    # Voice configuration
    parser.add_argument(
        "--voice-id",
        type=str,
        default=None,
        help="Voice ID to use (if not provided, uses random voice)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="eleven_v3",
        help="Model ID to use (default: eleven_v3)",
    )

    # Audio tags configuration
    parser.add_argument(
        "--no-audio-tags",
        action="store_true",
        help="Disable audio tags (coughs, sneezes, etc.)",
    )
    parser.add_argument(
        "--audio-tags-probability",
        type=float,
        default=DEFAULT_ELEVENLABS_AUDIO_TAGS_PROBABILITY,
        help=f"Probability of inserting audio tags (default: {DEFAULT_ELEVENLABS_AUDIO_TAGS_PROBABILITY})",
    )

    # Voice settings
    parser.add_argument(
        "--stability",
        type=float,
        default=0.5,
        help="Voice stability (0.0-1.0, default: 0.5)",
    )
    parser.add_argument(
        "--similarity-boost",
        type=float,
        default=0.75,
        help="Similarity boost (0.0-1.0, default: 0.75)",
    )
    parser.add_argument(
        "--style", type=float, default=0.0, help="Style (0.0-1.0, default: 0.0)"
    )

    # Seed for reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (optional)",
    )

    # API key
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="ElevenLabs API key (defaults to ELEVENLABS_API_KEY env var)",
    )

    args = parser.parse_args()

    voice_settings = VoiceSettings(
        stability=args.stability,
        similarity_boost=args.similarity_boost,
        style=args.style,
        use_speaker_boost=False,
    )

    config = ElevenLabsTTSConfig(
        model_id=args.model_id,
        voice_id=args.voice_id,
        api_key=args.api_key,
        voice_settings=voice_settings,
        insert_audio_tags=not args.no_audio_tags,
        audio_tags_probability=args.audio_tags_probability,
        seed=args.seed,
    )

    try:
        logger.info(f"Synthesizing: '{args.text}'")
        logger.info(
            f"Model: {config.model_id}, Voice ID: {config.voice_id or 'random'}"
        )

        # Generate speech
        audio_data = tts_elevenlabs(
            text=args.text,
            config=config,
        )

        logger.info(
            f"Generated audio: Duration: {audio_data.duration:.2f}s, Format: {audio_data.format}"
        )

        # Play audio
        play_audio(audio_data)

    except Exception as e:
        logger.error(f"Error during synthesis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
