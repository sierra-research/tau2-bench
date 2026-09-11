#!/usr/bin/env python3
# Copyright Sierra
"""Create ElevenLabs voices for all τ-bench personas via the Voice Design API.

This script automates the voice setup described in docs/voice-personas.md.
It generates a voice for each persona using the ElevenLabs Voice Design API,
saves it to your ElevenLabs account, and prints the environment variables
you need to add to your .env file.

Usage:
    # Create all 7 voices
    python -m tau2.voice.scripts.setup_voices

    # Create only the 2 control personas (for quick testing)
    python -m tau2.voice.scripts.setup_voices --complexity control

    # Create only the 5 regular personas
    python -m tau2.voice.scripts.setup_voices --complexity regular

    # Dry run — show what would be created without calling the API
    python -m tau2.voice.scripts.setup_voices --dry-run

    # Preview voices before saving (plays audio)
    python -m tau2.voice.scripts.setup_voices --preview

Requirements:
    - ELEVENLABS_API_KEY set in your environment or .env file
    - uv sync --extra voice
"""

import argparse
import sys
import time

from dotenv import load_dotenv
from loguru import logger

from tau2.voice.utils.voice_design import (
    create_voice_from_preview,
    design_voice_preview,
)

load_dotenv()

SAMPLE_TEXT = (
    "Hi, I'm calling about an issue with my account. "
    "I tried logging in earlier today but it keeps saying my password is wrong. "
    "I'm pretty sure I haven't changed it recently, "
    "so I'm not sure what's going on. "
    "Could you help me figure this out?"
)

# Voice Design parameters matching the docs/voice-personas.md recommendations.
# UI percentage → API value mapping:
#   loudness: UI 0–100% maps to API -1.0–1.0  (75% → 0.5)
#   guidance_scale: UI 0–100% maps to API 0–100  (38% → 38)
VOICE_DESIGN_LOUDNESS = 0.5  # 75% in the ElevenLabs UI
VOICE_DESIGN_GUIDANCE_SCALE = 38  # 38% in the ElevenLabs UI
VOICE_DESIGN_MODEL = "eleven_ttv_v3"
VOICE_DESIGN_MODELS = ["eleven_ttv_v3", "eleven_multilingual_ttv_v2"]
VOICE_DESIGN_SEED = 42
VOICE_NAME_PREFIX = "tau2"


def _get_existing_tau2_voices(client) -> dict[str, list[tuple[str, str]]]:
    """Fetch voices from the account and return tau2-prefixed ones.

    Returns:
        Mapping of voice_name → list of (voice_id, voice_name) for voices
        whose name starts with the tau2 prefix.
    """
    existing: dict[str, list[tuple[str, str]]] = {}
    try:
        response = client.voices.get_all()
        for voice in response.voices:
            if voice.name and voice.name.startswith(f"{VOICE_NAME_PREFIX}_"):
                existing.setdefault(voice.name, []).append((voice.voice_id, voice.name))
    except Exception as e:
        print(f"  WARNING: Could not fetch existing voices: {e}")
    return existing


def setup_voices(
    complexity: str = "all",
    dry_run: bool = False,
    preview: bool = False,
    seed: int = VOICE_DESIGN_SEED,
    model: str = VOICE_DESIGN_MODEL,
    force: bool = False,
) -> dict[str, str]:
    """Create ElevenLabs voices for τ-bench personas.

    Args:
        complexity: Which persona group to create — "control", "regular", or "all".
        dry_run: If True, print what would be created without calling the API.
        preview: If True, play each voice preview before saving.
        seed: Random seed for reproducible voice generation (same seed +
            same prompt = same voice).
        model: ElevenLabs TTV model to use for voice generation.
        force: If True, skip confirmation when voices already exist.

    Returns:
        Mapping of persona_name → voice_id for all created voices.
    """
    from elevenlabs import ElevenLabs

    from tau2.data_model.voice_personas import (
        ALL_PERSONAS,
        CONTROL_PERSONAS,
        REGULAR_PERSONAS,
    )

    if complexity == "control":
        personas = CONTROL_PERSONAS
    elif complexity == "regular":
        personas = REGULAR_PERSONAS
    else:
        personas = list(ALL_PERSONAS.values())

    if dry_run:
        print("\n=== DRY RUN — no API calls will be made ===\n")

        client = ElevenLabs()
        existing_voices = _get_existing_tau2_voices(client)
        if existing_voices:
            print(f"Existing tau2 voices in your account:\n")
            for voice_name, entries in existing_voices.items():
                for voice_id, _ in entries:
                    print(f"  {voice_name} (id: {voice_id})")
            print()
        else:
            print("No existing tau2 voices found in your account.\n")

        print("Would create voices for the following personas:\n")
        for p in personas:
            env_key = f"TAU2_VOICE_ID_{p.name.upper()}"
            voice_name = f"{VOICE_NAME_PREFIX}_{p.name}"
            conflict = voice_name in existing_voices
            print(f"  {p.display_name} ({p.name})")
            print(f"    Complexity: {p.complexity}")
            print(f"    Env var: {env_key}")
            print(f"    Voice name: {voice_name}")
            if conflict:
                existing_ids = [vid for vid, _ in existing_voices[voice_name]]
                print(f"    *** ALREADY EXISTS (id: {', '.join(existing_ids)}) ***")
            print(f"    Prompt: {p.prompt[:80]}...")
            print()
        print(f"Voice Design settings:")
        print(f"  Model: {model}")
        print(f"  Loudness: {VOICE_DESIGN_LOUDNESS} (75% in UI)")
        print(f"  Guidance scale: {VOICE_DESIGN_GUIDANCE_SCALE} (38% in UI)")
        print(f"  Seed: {seed}")
        return {}

    client = ElevenLabs()
    created_voices: dict[str, str] = {}

    print("\nChecking for existing tau2 voices...")
    existing_voices = _get_existing_tau2_voices(client)

    # "replace" = delete old + create new; "duplicate" = keep old + create new
    conflict_action: str | None = "replace" if force else None

    if existing_voices:
        print(f"\nFound {len(existing_voices)} existing tau2 voice(s):")
        for voice_name, entries in existing_voices.items():
            for voice_id, _ in entries:
                print(f"  {voice_name} (id: {voice_id})")

        conflicting = [
            p for p in personas if f"{VOICE_NAME_PREFIX}_{p.name}" in existing_voices
        ]
        if conflicting and not force:
            names = ", ".join(p.display_name for p in conflicting)
            print(f"\nThe following voices already exist: {names}")
            print("Options:")
            print("  [r]eplace  — delete existing voice(s), create new")
            print("  [d]uplicate — keep existing, create additional")
            print("  [s]kip     — skip existing, only create missing")
            print("  [a]bort    — exit without changes")
            answer = input("Choice [r/d/s/a]: ").strip().lower()
            if answer in ("r", "replace"):
                conflict_action = "replace"
            elif answer in ("d", "duplicate"):
                conflict_action = "duplicate"
            elif answer in ("s", "skip"):
                conflict_action = "skip"
            else:
                print("Aborted.")
                return {}
    else:
        print("  No existing tau2 voices found.")

    print(f"\nCreating {len(personas)} voice(s) via ElevenLabs Voice Design API...\n")
    print(f"  Model: {model}")
    print(f"  Loudness: {VOICE_DESIGN_LOUDNESS} (75% in UI)")
    print(f"  Guidance scale: {VOICE_DESIGN_GUIDANCE_SCALE} (38% in UI)")
    print(f"  Seed: {seed}")
    print()

    for i, persona in enumerate(personas, 1):
        voice_name = f"{VOICE_NAME_PREFIX}_{persona.name}"
        print(f"[{i}/{len(personas)}] Creating voice: {persona.display_name}")
        print(f"  Description: {persona.short_description}")

        if voice_name in existing_voices:
            existing_ids = [vid for vid, _ in existing_voices[voice_name]]
            print(
                f"  Voice '{voice_name}' already exists (id: {', '.join(existing_ids)})"
            )
            if conflict_action == "skip":
                print("  Skipped (already exists).")
                print()
                continue
            if conflict_action == "replace":
                for old_id, _ in existing_voices[voice_name]:
                    try:
                        client.voices.delete(voice_id=old_id)
                        print(f"  Deleted existing voice: {old_id}")
                    except Exception as e:
                        print(f"  WARNING: Failed to delete {old_id}: {e}")
            else:
                print("  Creating duplicate.")

        try:
            selected_preview = design_voice_preview(
                client,
                voice_description=persona.prompt,
                audition_text=SAMPLE_TEXT,
                model_id=model,
                loudness=VOICE_DESIGN_LOUDNESS,
                guidance_scale=VOICE_DESIGN_GUIDANCE_SCALE,
                seed=seed,
            )

            if selected_preview is None:
                print(f"  ERROR: No previews returned for {persona.display_name}")
                continue

            print(f"  Using preview: {selected_preview.generated_voice_id}")

            if preview:
                _play_preview(selected_preview)

            voice_id = create_voice_from_preview(
                client,
                voice_name=voice_name,
                short_description=persona.short_description,
                generated_voice_id=selected_preview.generated_voice_id,
            )

            created_voices[persona.name] = voice_id
            print(f"  Saved as voice_id: {voice_id}")
            print()

            if i < len(personas):
                time.sleep(1)

        except Exception as e:
            print(f"  ERROR: Failed to create voice for {persona.display_name}: {e}")
            print()
            continue

    return created_voices


def _play_preview(preview) -> None:
    """Play a voice preview through speakers."""
    import base64

    try:
        from elevenlabs.play import play

        audio_bytes = base64.b64decode(preview.audio_base_64)
        print("  Playing preview...")
        play(audio_bytes)
    except ImportError:
        print("  (skipping playback — elevenlabs.play not available)")
    except Exception as e:
        print(f"  (playback failed: {e})")


def print_env_block(created_voices: dict[str, str]) -> None:
    """Print the env var block to paste into .env."""
    if not created_voices:
        return

    print("=" * 60)
    print("Add the following to your .env file:")
    print("=" * 60)
    print()
    for name, voice_id in created_voices.items():
        env_key = f"TAU2_VOICE_ID_{name.upper()}"
        print(f"{env_key}={voice_id}")
    print()
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Create ElevenLabs voices for τ-bench personas.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create all 7 voices
  python -m tau2.voice.scripts.setup_voices

  # Create only the 2 control personas (for quick testing)
  python -m tau2.voice.scripts.setup_voices --complexity control

  # Create only the 5 regular personas
  python -m tau2.voice.scripts.setup_voices --complexity regular

  # See what would be created without calling the API
  python -m tau2.voice.scripts.setup_voices --dry-run

  # Preview each voice before saving
  python -m tau2.voice.scripts.setup_voices --preview

See docs/voice-personas.md for the full setup guide.
""",
    )
    parser.add_argument(
        "--complexity",
        choices=["all", "control", "regular"],
        default="all",
        help="Which persona group to create: 'control' (2 American accents), "
        "'regular' (5 diverse accents), or 'all' (default).",
    )
    parser.add_argument(
        "--model",
        choices=VOICE_DESIGN_MODELS,
        default=VOICE_DESIGN_MODEL,
        help=f"ElevenLabs TTV model (default: {VOICE_DESIGN_MODEL}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=VOICE_DESIGN_SEED,
        help=f"Random seed for reproducible voice generation (default: {VOICE_DESIGN_SEED}). "
        "Same seed + same prompt = same voice.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be created without calling the API.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Play each voice preview before saving.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompts when voices already exist.",
    )

    args = parser.parse_args()
    logger.configure(handlers=[{"sink": sys.stderr, "level": "WARNING"}])

    created_voices = setup_voices(
        complexity=args.complexity,
        dry_run=args.dry_run,
        preview=args.preview,
        seed=args.seed,
        model=args.model,
        force=args.force,
    )

    if created_voices:
        print_env_block(created_voices)
        print(
            f"Done! Created {len(created_voices)} voice(s). "
            "Copy the lines above into your .env file."
        )
    elif not args.dry_run:
        print("\nNo voices were created. Check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
