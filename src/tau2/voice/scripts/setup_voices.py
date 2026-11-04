#!/usr/bin/env python3
# Copyright Sierra
"""Create ElevenLabs caller voices for language-pack personas via Voice Design.

This script automates the voice setup described in docs/voice-personas.md.
It designs a voice for each caller persona of the selected language packs from
the prompts in ``data/tau2/voice_design/<lang>.yaml``, saves it to your
ElevenLabs account, and prints the ``TAU2_VOICE_ID_<LANG>_<PERSONA>``
environment variables to add to your .env file.

Usage:
    # Create the English caller voices
    python -m tau2.voice.scripts.setup_voices

    # Create the caller voices for every voice-v2 language
    python -m tau2.voice.scripts.setup_voices --language all

    # Dry run — show what would be created without calling the API
    python -m tau2.voice.scripts.setup_voices --language es --language hi --dry-run

    # Preview voices before saving (plays audio)
    python -m tau2.voice.scripts.setup_voices --preview

Requirements:
    - ELEVENLABS_API_KEY set in your environment or .env file
    - uv sync --extra voice
"""

import argparse
import sys
import time

import yaml
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel

from tau2.data_model.voice import (
    LanguageVoiceDesign,
    PersonaVoiceDesign,
    VoiceCreateRequest,
    VoiceDesignRequest,
)
from tau2.multilingual.runtime import load_pack, voice_id_env_key
from tau2.utils import DATA_DIR
from tau2.voice.elevenlabs_operations import create_voice, design_voice

load_dotenv()

# Voice Design parameters matching the docs/voice-personas.md recommendations.
# UI percentage → API value mapping:
#   loudness: UI 0–100% maps to API -1.0–1.0  (75% → 0.5)
#   guidance_scale: UI 0–100% maps to API 0–100  (38% → 38)
VOICE_DESIGN_LOUDNESS = 0.5  # 75% in the ElevenLabs UI
VOICE_DESIGN_GUIDANCE_SCALE = 38  # 38% in the ElevenLabs UI
VOICE_DESIGN_MODELS = ["eleven_ttv_v3", "eleven_multilingual_ttv_v2"]
VOICE_DESIGN_SEED = 42
VOICE_NAME_PREFIX = "tau2"
DEFAULT_LANGUAGE = "en"
ALL_LANGUAGES = "all"


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


class PersonaDesignTarget(BaseModel):
    """One pack persona to design, with the names it is saved and exported under."""

    language: str
    persona_id: str
    design: PersonaVoiceDesign

    @property
    def voice_name(self) -> str:
        """Account voice name; the language prefix keeps packs from colliding."""
        return f"{VOICE_NAME_PREFIX}_{self.language}_{self.persona_id}"

    @property
    def env_key(self) -> str:
        """The runtime override variable for this persona's voice."""
        return voice_id_env_key(self.language, self.persona_id)

    @property
    def account_description(self) -> str:
        """Short account label for the saved voice."""
        return f"tau2 {self.language} caller {self.persona_id} ({self.design.gender})"


def released_voice_languages() -> list[str]:
    """Languages of the released voice benchmark, in manifest order."""
    from tau2.runner.presets import load_manifest

    manifest, _ = load_manifest("voice")
    return list(dict.fromkeys(cell.language for cell in manifest.cells))


def resolve_languages(requested: list[str] | None) -> list[str]:
    """Expand ``--language`` values: none means English, ``all`` means voice-v2."""
    if not requested:
        return [DEFAULT_LANGUAGE]
    languages: list[str] = []
    for language in requested:
        expanded = (
            released_voice_languages() if language == ALL_LANGUAGES else [language]
        )
        for lang in expanded:
            if lang not in languages:
                languages.append(lang)
    return languages


def load_voice_design(language: str) -> LanguageVoiceDesign:
    """Read a language's design prompts; they must cover exactly its pack personas."""
    pack = load_pack(language)  # validates the identifier before the path join
    path = DATA_DIR / "tau2" / "voice_design" / f"{language}.yaml"
    if not path.is_file():
        raise ValueError(f"no voice design prompts for language {language!r}")
    design = LanguageVoiceDesign.model_validate(yaml.safe_load(path.read_text()))
    if design.language != language:
        raise ValueError(f"{path.name} declares language {design.language!r}")
    if set(design.personas) != set(pack.personas):
        raise ValueError(
            f"{path.name} personas {sorted(design.personas)} differ from the "
            f"{language} pack personas {sorted(pack.personas)}"
        )
    for persona_id, persona in design.personas.items():
        if persona.gender != pack.personas[persona_id].gender:
            raise ValueError(f"{path.name} gender differs for persona {persona_id!r}")
    return design


def design_targets(languages: list[str]) -> list[PersonaDesignTarget]:
    """Every persona to design for the selected languages, in pack order."""
    return [
        PersonaDesignTarget(language=language, persona_id=persona_id, design=design)
        for language in languages
        for persona_id, design in load_voice_design(language).personas.items()
    ]


def setup_voices(
    languages: list[str] | None = None,
    dry_run: bool = False,
    preview: bool = False,
    seed: int = VOICE_DESIGN_SEED,
    model: str | None = None,
    force: bool = False,
) -> dict[str, str]:
    """Create ElevenLabs voices for the selected language packs' personas.

    Args:
        languages: Language pack codes; None selects English only.
        dry_run: If True, print what would be created without calling the API.
        preview: If True, play each voice preview before saving.
        seed: Random seed for reproducible voice generation (same seed +
            same prompt = same voice).
        model: ElevenLabs TTV model override; None uses each persona's model.
        force: If True, skip confirmation when voices already exist.

    Returns:
        Mapping of ``TAU2_VOICE_ID_<LANG>_<PERSONA>`` → voice_id for all
        created voices.
    """
    from elevenlabs import ElevenLabs

    personas = design_targets(languages or [DEFAULT_LANGUAGE])

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
            print(f"  {p.persona_id} ({p.language}, {p.design.gender})")
            print(f"    Env var: {p.env_key}")
            print(f"    Voice name: {p.voice_name}")
            if p.voice_name in existing_voices:
                existing_ids = [vid for vid, _ in existing_voices[p.voice_name]]
                print(f"    *** ALREADY EXISTS (id: {', '.join(existing_ids)}) ***")
            print(f"    Model: {model or p.design.model}")
            print(f"    Prompt: {p.design.prompt[:80]}...")
            print()
        print(f"Voice Design settings:")
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

        conflicting = [p for p in personas if p.voice_name in existing_voices]
        if conflicting and not force:
            names = ", ".join(p.voice_name for p in conflicting)
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
    if model:
        print(f"  Model (override): {model}")
    print(f"  Loudness: {VOICE_DESIGN_LOUDNESS} (75% in UI)")
    print(f"  Guidance scale: {VOICE_DESIGN_GUIDANCE_SCALE} (38% in UI)")
    print(f"  Seed: {seed}")
    print()

    for i, persona in enumerate(personas, 1):
        voice_name = persona.voice_name
        print(f"[{i}/{len(personas)}] Creating voice: {voice_name}")
        print(f"  Model: {model or persona.design.model}")

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
            previews = design_voice(
                client,
                VoiceDesignRequest(
                    description=persona.design.prompt,
                    text=persona.design.audition_text,
                    model=model or persona.design.model,
                    loudness=VOICE_DESIGN_LOUDNESS,
                    guidance_scale=VOICE_DESIGN_GUIDANCE_SCALE,
                    seed=seed,
                ),
            )

            if not previews:
                print(f"  ERROR: No previews returned for {voice_name}")
                continue

            selected_preview = previews[0]
            print(
                f"  Generated {len(previews)} preview(s), "
                f"using first: {selected_preview.generated_voice_id}"
            )

            if preview:
                _play_preview(selected_preview)

            voice_id = create_voice(
                client,
                VoiceCreateRequest(
                    name=voice_name,
                    description=persona.account_description,
                    generated_voice_id=selected_preview.generated_voice_id,
                ),
            )

            created_voices[persona.env_key] = voice_id
            print(f"  Saved as voice_id: {voice_id}")
            print()

            if i < len(personas):
                time.sleep(1)

        except Exception as e:
            print(f"  ERROR: Failed to create voice for {voice_name}: {e}")
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
    for env_key, voice_id in created_voices.items():
        print(f"{env_key}={voice_id}")
    print()
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Create ElevenLabs caller voices for language-pack personas.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create the English caller voices
  python -m tau2.voice.scripts.setup_voices

  # Create the caller voices for every voice-v2 language
  python -m tau2.voice.scripts.setup_voices --language all

  # Only Spanish and Hindi
  python -m tau2.voice.scripts.setup_voices --language es --language hi

  # See what would be created without calling the API
  python -m tau2.voice.scripts.setup_voices --dry-run

  # Preview each voice before saving
  python -m tau2.voice.scripts.setup_voices --preview

See docs/voice-personas.md for the full setup guide.
""",
    )
    parser.add_argument(
        "--language",
        action="append",
        metavar="CODE",
        help="Language pack to create caller voices for (repeatable); "
        f"'{ALL_LANGUAGES}' selects every voice-v2 language "
        f"(default: {DEFAULT_LANGUAGE}).",
    )
    parser.add_argument(
        "--model",
        choices=VOICE_DESIGN_MODELS,
        default=None,
        help="ElevenLabs TTV model override (default: each persona's model "
        "from data/tau2/voice_design/<lang>.yaml).",
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
        languages=resolve_languages(args.language),
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
