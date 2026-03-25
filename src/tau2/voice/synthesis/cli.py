#!/usr/bin/env python3
# Copyright Sierra
"""
Simple synthesis test using the voice synthesis API

Usage:
    # Test with default settings (random voice, background noise)
    python src/tau2/voice/synthesis/cli.py synthesize "Hello, world!"

    # Test with playback instead of saving
    python src/tau2/voice/synthesis/cli.py synthesize "Hello, world!" --play

    # Test with custom output file
    python src/tau2/voice/synthesis/cli.py synthesize "Hello, world!" --output greeting.wav

    # Test with specific voice ID
    python src/tau2/voice/synthesis/cli.py synthesize "Hello, world!" --voice-id avQFHuQU7IjJf0u5MMBq

    # List available configuration
    python src/tau2/voice/synthesis/cli.py list-config

Note: Output is in telephony format (WAV μ-law 8kHz) with optional background noise
"""

import os
import random
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from elevenlabs import VoiceSettings
from rich.console import Console
from rich.table import Table
from typer import Option, Typer

from tau2.config import DEFAULT_PCM_SAMPLE_RATE
from tau2.data_model.audio_effects import SourceEffectsConfig
from tau2.data_model.voice import ElevenLabsTTSConfig
from tau2.data_model.voice_personas import ALL_PERSONAS
from tau2.voice.synthesis.audio_effects.noise_generator import (
    apply_background_noise,
    create_background_noise_generator,
)
from tau2.voice.synthesis.synthesize import synthesize_voice
from tau2.voice.utils.audio_io import play_audio, save_wav_file
from tau2.voice_config import BACKGROUND_NOISE_CONTINUOUS_DIR

load_dotenv()

app = Typer()
console = Console()


@app.command()
def list_config() -> None:
    """List synthesis configuration and check API key availability"""
    table = Table(title="Voice Synthesis Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Default Value", style="green")
    table.add_column("Status", style="yellow")

    # Check API key
    api_key_status = "✓ Found" if os.getenv("ELEVENLABS_API_KEY") else "✗ Missing"

    config_info = [
        ("Provider", "elevenlabs", "Supported"),
        ("Voice IDs", "Random selection", f"{len(ALL_PERSONAS)} voices available"),
        ("Default Model", "eleven_v3", "Available"),
        ("Output Format", "PCM 16kHz → μ-law 8kHz", "Telephony format"),
        ("Background Noise", "SNR 15dB ± 3dB drift", "Continuous ambient sounds"),
        ("Burst Noise", "SNR -5 to +10 dB", "Short sound effects"),
        ("Audio Tags (v3)", "Enabled by default (20%)", "[cough], [sneeze], [sniffle]"),
        ("API Key", "ELEVENLABS_API_KEY", api_key_status),
    ]

    for setting, value, status in config_info:
        table.add_row(setting, value, status)

    console.print(table)

    if "✗" in api_key_status:
        console.print("\n[red]⚠️  ELEVENLABS_API_KEY not found in environment[/red]")
        console.print("Set it with: export ELEVENLABS_API_KEY='your-api-key'")


@app.command()
def synthesize(
    text: str,
    play: bool = Option(
        False,
        "--play",
        "-p",
        help="Play audio instead of saving to file",
    ),
    output: Optional[str] = Option(
        None,
        "--output",
        "-o",
        help="Output audio file path (WAV format, μ-law 8kHz). If not provided and --play is False, saves to output.wav",
    ),
    voice_id: Optional[str] = Option(
        None,
        "--voice-id",
        "-v",
        help="Voice ID for synthesis (if not provided, randomly selects from available voices)",
    ),
    model_id: str = Option(
        "eleven_v3", "--model-id", "-m", help="Model ID for synthesis"
    ),
    api_key: Optional[str] = Option(
        None, "--api-key", help="API key (if not provided, uses environment variable)"
    ),
    no_background_noise: bool = Option(
        False, "--no-background-noise", help="Disable background noise mixing"
    ),
    no_burst_noise: bool = Option(
        False,
        "--no-burst-noise",
        help="Disable burst noise overlay (short sounds like door closing, baby crying)",
    ),
    no_audio_tags: bool = Option(
        False,
        "--no-audio-tags",
        help="Disable audio tags insertion for v3 models ([cough], [sneeze], [sniffle])",
    ),
    snr_db: float = Option(
        15.0,
        "--snr-db",
        help="Background noise SNR in dB (higher = quieter noise)",
    ),
    burst_prob: float = Option(
        0.3,
        "--burst-prob",
        help="Probability of adding burst noise (0.0-1.0)",
        min=0.0,
        max=1.0,
    ),
    tags_prob: float = Option(
        0.2,
        "--tags-prob",
        help="Probability of adding audio tags (0.0-1.0)",
        min=0.0,
        max=1.0,
    ),
    stability: float = Option(
        0.5,
        "--stability",
        help="Voice stability (0.0-1.0)",
        min=0.0,
        max=1.0,
    ),
    similarity_boost: float = Option(
        0.75,
        "--similarity-boost",
        help="Voice similarity boost (0.0-1.0)",
        min=0.0,
        max=1.0,
    ),
    style: float = Option(
        0.0,
        "--style",
        help="Voice style (0.0-1.0)",
        min=0.0,
        max=1.0,
    ),
    seed: Optional[int] = Option(
        None,
        "--seed",
        help="Random seed for reproducibility",
    ),
) -> None:
    """Synthesize text to speech using ElevenLabs API with background noise and effects"""

    # Check API key availability if not provided
    if not api_key and not os.getenv("ELEVENLABS_API_KEY"):
        console.print("[red]Error: ELEVENLABS_API_KEY not found in environment[/red]")
        console.print("Either set the environment variable or use --api-key option")
        raise SystemExit(1)

    # Select a random voice if none was provided
    if voice_id is None:
        persona = random.choice(list(ALL_PERSONAS.values()))
        voice_id = persona.elevenlabs_voice_id
        console.print(
            f"[cyan]Selected persona:[/cyan] {persona.display_name} ({persona.short_description})"
        )

    # Create voice settings
    voice_settings = VoiceSettings(
        stability=stability,
        similarity_boost=similarity_boost,
        style=style,
        use_speaker_boost=False,
    )

    # Create provider config (ElevenLabs)
    provider_config = ElevenLabsTTSConfig(
        model_id=model_id,
        voice_id=voice_id,
        api_key=api_key,
        voice_settings=voice_settings,
        insert_audio_tags=not no_audio_tags,
        audio_tags_probability=tags_prob,
        seed=seed,
    )

    # Create source effects config (background noise + burst noise)
    source_effects_config = SourceEffectsConfig(
        enable_background_noise=not no_background_noise,
        noise_snr_db=snr_db,
        enable_burst_noise=not no_burst_noise,
        burst_noise_events_per_minute=burst_prob
        * 60,  # Convert probability to events/min
    )

    # Determine output path
    output_path = None
    if not play:
        output_path = Path(output if output else "output.wav")
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Display what we're doing
    console.print(
        f"[cyan]Synthesizing text:[/cyan] {text[:50]}{'...' if len(text) > 50 else ''}"
    )
    console.print(f"[cyan]Voice ID:[/cyan] {voice_id}")
    console.print(f"[cyan]Model:[/cyan] {model_id}")
    console.print(
        f"[cyan]Background Noise:[/cyan] {'Enabled' if not no_background_noise else 'Disabled'} (SNR={snr_db:.1f}dB)"
    )
    console.print(
        f"[cyan]Burst Noise:[/cyan] {'Enabled' if not no_burst_noise else 'Disabled'} (prob={burst_prob:.1%})"
    )
    if "v3" in model_id.lower():
        console.print(
            f"[cyan]Audio Tags:[/cyan] {'Enabled' if not no_audio_tags else 'Disabled'} (prob={tags_prob:.1%})"
        )

    # Create background noise generator with actual noise file
    background_noise_generator = None
    if not no_background_noise:
        noise_files = list(BACKGROUND_NOISE_CONTINUOUS_DIR.glob("*.wav"))
        if noise_files:
            noise_file = random.choice(noise_files)
            console.print(f"[cyan]Noise file:[/cyan] {noise_file.name}")
            background_noise_generator = create_background_noise_generator(
                config=source_effects_config,
                sample_rate=DEFAULT_PCM_SAMPLE_RATE,
                background_noise_file=noise_file,
            )

    # Synthesize
    try:
        with console.status("Synthesizing audio..."):
            audio_data = synthesize_voice(
                text=text,
                provider="elevenlabs",
                provider_config=provider_config,
            )

            if background_noise_generator is not None:
                audio_data = apply_background_noise(
                    audio=audio_data,
                    noise_generator=background_noise_generator,
                )

            if output_path:
                audio_data.audio_path = output_path
                save_wav_file(audio_data, str(output_path))
    except Exception as e:
        console.print(f"[red]✗ Synthesis failed:[/red] {e}")
        raise SystemExit(1)

    console.print(f"[cyan]Duration:[/cyan] {audio_data.duration:.2f} seconds")
    console.print(f"[cyan]Format:[/cyan] {audio_data.format}")

    if play:
        play_audio(audio_data)
    else:
        console.print(f"[green]✓ Audio saved to:[/green] {audio_data.audio_path}")


if __name__ == "__main__":
    app()
