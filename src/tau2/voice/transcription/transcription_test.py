#!/usr/bin/env python3
# Copyright Sierra
"""
Simple transcription test using the unified transcription API

Usage:
    # Test with default model (whisper-1)
    python -m tau2.voice.transcription_test transcribe audio.mp3

    # Test with specific model
    python -m tau2.voice.transcription_test transcribe audio.mp3 --model nova-2
    python -m tau2.voice.transcription_test transcribe audio.mp3 --model gpt-4o-transcribe

    # Test with language option
    python -m tau2.voice.transcription_test transcribe audio.mp3 --model nova-2 --language en-US

    # List available models
    python -m tau2.voice.transcription_test list-models
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from typer import Option, Typer

from tau2.data_model.voice import TranscriptionConfig, TranscriptionModel
from tau2.voice.transcription.transcribe import transcribe_audio

load_dotenv()

app = Typer()
console = Console()


@app.command()
def list_models() -> None:
    """List all available transcription models"""
    table = Table(title="Available Transcription Models")
    table.add_column("Model", style="cyan")
    table.add_column("Provider", style="green")
    table.add_column("API Type", style="yellow")
    table.add_column("Required Env Var", style="red")

    models_info = [
        ("nova-2", "Deepgram", "REST", "DEEPGRAM_API_KEY"),
        ("nova-3", "Deepgram", "REST", "DEEPGRAM_API_KEY"),
        ("whisper-1", "OpenAI", "REST", "OPENAI_API_KEY"),
        ("gpt-4o-transcribe", "OpenAI", "WebSocket", "OPENAI_API_KEY"),
        ("gpt-4o-mini-transcribe", "OpenAI", "WebSocket", "OPENAI_API_KEY"),
    ]

    for model, provider, api_type, env_var in models_info:
        # Check if API key is available
        key_status = "✓" if os.getenv(env_var) else "✗"
        table.add_row(f"{model} {key_status}", provider, api_type, env_var)

    console.print(table)
    console.print("\n✓ = API key found, ✗ = API key missing")


@app.command()
def transcribe(
    audio_file: str,
    model: TranscriptionModel = Option(
        "whisper-1", "--model", "-m", help="Transcription model to use"
    ),
    language: Optional[str] = Option(
        None,
        "--language",
        "-l",
        help="Language code (e.g., 'en-US' for Deepgram, 'en' for OpenAI)",
    ),
) -> None:
    """Transcribe audio file using unified transcription API"""

    # Check if file exists
    audio_path = Path(audio_file)
    if not audio_path.exists():
        console.print(f"[red]Error: File {audio_file} not found[/red]")
        raise SystemExit(1)

    # Check API key availability
    if model in ["nova-2", "nova-3"]:
        if not os.getenv("DEEPGRAM_API_KEY"):
            console.print("[red]Error: DEEPGRAM_API_KEY not found in environment[/red]")
            raise SystemExit(1)
    else:
        if not os.getenv("OPENAI_API_KEY"):
            console.print("[red]Error: OPENAI_API_KEY not found in environment[/red]")
            raise SystemExit(1)

    # Create config
    config = TranscriptionConfig(model=model, language=language)

    # Transcribe
    with console.status(f"Transcribing with {model}..."):
        result = transcribe_audio(audio_path, config)

    # Handle results
    if result.error:
        console.print(f"[red]Error: {result.error}[/red]")
        raise SystemExit(1)

    # Display transcript
    console.print("\n[bold]Transcript:[/bold]")
    console.print(result.transcript)

    if result.confidence is not None:
        console.print(f"\n[cyan]Confidence:[/cyan] {result.confidence:.2%}")


if __name__ == "__main__":
    app()
