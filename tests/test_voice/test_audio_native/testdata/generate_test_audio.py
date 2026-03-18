#!/usr/bin/env python3
"""
Generate test audio files for audio native provider testing using ElevenLabs TTS.

These audio files are shared across all audio native provider tests (Nova, Gemini, Deepgram, etc.).

Usage:
    python tests/test_voice/test_audio_native/testdata/generate_test_audio.py

Output format: WAV, 16kHz, 16-bit, mono (compatible with most provider inputs)
"""

import os
import sys
import wave

# Add src to path
sys.path.insert(0, "src")

from tau2.data_model.voice import ElevenLabsTTSConfig
from tau2.data_model.voice_personas import get_elevenlabs_voice_id
from tau2.voice.utils.elevenlabs_utils import tts_elevenlabs

# Test utterances to generate
TEST_UTTERANCES = [
    ("hello", "Hello"),
    ("hi_how_are_you", "Hi, how are you?"),
    ("yes", "Yes"),
    ("no", "No"),
    ("thank_you", "Thank you very much"),
    ("order_status", "I'd like to check my order status please"),
    ("help_me", "Can you help me with something?"),
    # Tool call trigger - specific order ID for function calling tests
    ("check_order_12345", "Can you check the status of my order number 12345?"),
]

# Output directory (same as this script's location)
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_test_audio():
    """Generate test audio files using ElevenLabs."""

    # Check for API key
    if not os.getenv("ELEVENLABS_API_KEY"):
        print("❌ ELEVENLABS_API_KEY environment variable not set")
        print("   Set it with: export ELEVENLABS_API_KEY=your_key")
        return False

    print(f"Output directory: {OUTPUT_DIR}")

    # Use a specific voice (Lisa Brenner)
    voice_id = get_elevenlabs_voice_id("lisa_brenner")

    # Configure ElevenLabs - output is PCM 16kHz which matches most provider inputs
    config = ElevenLabsTTSConfig(
        voice_id=voice_id,
        model_id="eleven_v3",
        insert_audio_tags=False,
    )

    print(f"Using voice ID: {voice_id}")
    print(f"Output format: WAV 16kHz 16-bit mono")
    print()

    generated = []

    for filename, text in TEST_UTTERANCES:
        output_path = os.path.join(OUTPUT_DIR, f"{filename}.wav")

        print(f"Generating: '{text}'...")

        try:
            # Generate audio using ElevenLabs
            audio_data = tts_elevenlabs(text, config)

            # The audio is raw PCM, wrap it in a WAV file
            raw_pcm = audio_data.data

            # Write as WAV file
            with wave.open(output_path, "wb") as wav:
                wav.setnchannels(1)  # mono
                wav.setsampwidth(2)  # 16-bit
                wav.setframerate(16000)  # 16kHz
                wav.writeframes(raw_pcm)

            duration_ms = len(raw_pcm) / 32  # 32 bytes per ms at 16kHz 16-bit
            print(
                f"   ✅ Saved: {output_path} ({len(raw_pcm)} bytes, {duration_ms:.0f}ms)"
            )
            generated.append(output_path)

        except Exception as e:
            print(f"   ❌ Error: {e}")

    print()
    print(f"Generated {len(generated)} audio files in {OUTPUT_DIR}")

    return len(generated) > 0


if __name__ == "__main__":
    success = generate_test_audio()
    sys.exit(0 if success else 1)
