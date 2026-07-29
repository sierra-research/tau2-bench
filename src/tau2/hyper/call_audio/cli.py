#!/usr/bin/env python3
# Copyright Sierra
"""Render hyper-tau phone-call transcripts into call-recording audio.

Usage:
    # Preview which cases would render and how they're cast
    python -m tau2.hyper.call_audio.cli list-cases data/tau2/hyper/sops/airline

    # Write a casting manifest for hand-editing (voice per case)
    python -m tau2.hyper.call_audio.cli cast data/tau2/hyper/sops/airline \
        --output data/tau2/hyper/call_audio/casting.json

    # Render every phone-call case under a directory (mock = no API key)
    python -m tau2.hyper.call_audio.cli render data/tau2/hyper/sops/airline \
        --output-dir data/tau2/hyper/call_audio --mock

    # Real render with ElevenLabs (needs ELEVENLABS_API_KEY)
    python -m tau2.hyper.call_audio.cli render <case_or_dir> \
        --output-dir data/tau2/hyper/call_audio \
        --manifest data/tau2/hyper/call_audio/casting.json
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from typer import Option, Typer

from tau2.hyper.call_audio.casting import CastingManifest, default_casting
from tau2.hyper.call_audio.renderer import RenderSettings, render_call
from tau2.hyper.call_audio.transcript_parser import (
    find_phone_call_transcripts,
    is_phone_call_file,
    parse_call_transcript,
)

load_dotenv()

app = Typer()
console = Console()


def _collect_cases(target: Path) -> list[Path]:
    if target.is_file():
        if not is_phone_call_file(target):
            console.print(
                f"[red]{target} is not a phone-call record "
                "(needs a 'Channel: phone call' header)[/red]"
            )
            raise SystemExit(1)
        return [target]
    cases = find_phone_call_transcripts(target)
    if not cases:
        console.print(f"[red]No phone-call case_*.md files under {target}[/red]")
        raise SystemExit(1)
    return cases


@app.command()
def list_cases(
    target: Path,
    root: Optional[Path] = Option(
        None, help="Casting root for stable seeds (default: target directory)"
    ),
) -> None:
    """List phone-call cases with their default casting."""
    cases = _collect_cases(target)
    casting_root = root or (target if target.is_dir() else target.parent)

    table = Table(title=f"Phone-call cases under {target}")
    table.add_column("Case", style="cyan")
    table.add_column("Turns", justify="right")
    table.add_column("Console", justify="right")
    table.add_column("Agent voice", style="green")
    table.add_column("Customer voice", style="yellow")

    for path in cases:
        transcript = parse_call_transcript(path)
        casting = default_casting(path, casting_root)
        table.add_row(
            casting.case_path,
            str(len(transcript.spoken_turns)),
            str(len(transcript.console_events)),
            casting.agent_voice,
            casting.customer_voice,
        )
    console.print(table)


@app.command()
def cast(
    target: Path,
    output: Path = Option(..., "--output", "-o", help="Manifest JSON path"),
    root: Optional[Path] = Option(
        None, help="Casting root for stable seeds (default: target directory)"
    ),
) -> None:
    """Write a default casting manifest for hand-editing."""
    cases = _collect_cases(target)
    casting_root = root or (target if target.is_dir() else target.parent)
    manifest = CastingManifest.build(cases, casting_root)
    manifest.save(output)
    console.print(f"[green]Wrote casting for {len(cases)} cases to {output}[/green]")


@app.command()
def render(
    target: Path,
    output_dir: Path = Option(..., "--output-dir", "-o"),
    manifest_path: Optional[Path] = Option(
        None, "--manifest", help="Casting manifest (default: deterministic casting)"
    ),
    mock: bool = Option(
        False, "--mock", help="Render sine-tone stand-ins instead of calling TTS"
    ),
    no_background_noise: bool = Option(False, "--no-background-noise"),
    snr_db: float = Option(20.0, "--snr-db", help="Background noise SNR in dB"),
    no_telephony: bool = Option(
        False, "--no-telephony", help="Keep 16kHz PCM instead of telephony 8kHz"
    ),
    audio_tags: bool = Option(
        False, "--audio-tags", help="Let ElevenLabs insert [cough]-style tags"
    ),
    limit: Optional[int] = Option(None, "--limit", help="Render at most N cases"),
    overwrite: bool = Option(False, "--overwrite"),
) -> None:
    """Render phone-call cases to WAV call recordings."""
    if not mock and not os.getenv("ELEVENLABS_API_KEY"):
        console.print(
            "[red]ELEVENLABS_API_KEY not set — use --mock or provide a key[/red]"
        )
        raise SystemExit(1)

    cases = _collect_cases(target)
    if limit is not None:
        cases = cases[:limit]

    manifest = CastingManifest.load(manifest_path) if manifest_path else None
    casting_root = (
        Path(manifest.root)
        if manifest
        else (target if target.is_dir() else target.parent)
    )

    settings = RenderSettings(
        background_noise=not no_background_noise,
        noise_snr_db=snr_db,
        telephony=not no_telephony,
        insert_audio_tags=audio_tags,
        mock=mock,
    )

    rendered, skipped = 0, 0
    for path in cases:
        casting = (manifest.lookup(path) if manifest else None) or default_casting(
            path, casting_root
        )
        output_path = output_dir / Path(casting.case_path).with_suffix(".wav")
        if output_path.exists() and not overwrite:
            skipped += 1
            continue
        transcript = parse_call_transcript(path)
        render_call(
            transcript,
            casting,
            settings,
            output_path,
            turn_cache_dir=output_dir / ".turn_cache" / Path(casting.case_path).with_suffix("")
            if not mock
            else None,
        )
        rendered += 1

    console.print(
        f"[green]Rendered {rendered} calls to {output_dir}[/green]"
        + (f" [dim]({skipped} already existed, skipped)[/dim]" if skipped else "")
    )


if __name__ == "__main__":
    app()
