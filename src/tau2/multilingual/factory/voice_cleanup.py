# Copyright Sierra
"""Delete rejected ElevenLabs voices (audition losers, superseded redesigns).

Programmatic core behind ``tau2 factory delete-voices``. Every deletion is
guarded: a voice_id pinned by any language pack is refused, so the sweep can
never orphan a live persona. Errors are isolated per voice — a voice that is
already gone from the account (or fails to delete) never aborts the rest.
"""

from __future__ import annotations

import yaml
from loguru import logger
from pydantic import BaseModel, Field

from tau2.multilingual.factory.voice_generation import _pack_path


class VoiceDeletionResult(BaseModel):
    """Outcome of one voice deletion attempt."""

    voice_id: str
    status: str = Field(
        description="'deleted' | 'dry_run' | 'skipped_pinned' | 'error'"
    )
    detail: str = ""


def pinned_voice_ids() -> set[str]:
    """Every voice_id currently pinned by any language pack persona."""
    multilingual_root = _pack_path("es").parent.parent
    ids: set[str] = set()
    for pack in sorted(multilingual_root.glob("*/pack.yaml")):
        data = yaml.safe_load(pack.read_text())
        for persona in (data.get("personas") or {}).values():
            vid = str((persona or {}).get("voice_id") or "").strip()
            if vid:
                ids.add(vid)
    return ids


def delete_voices(
    voice_ids: list[str],
    *,
    dry_run: bool = False,
    client=None,
) -> list[VoiceDeletionResult]:
    """Delete the given ElevenLabs voices, refusing any pack-pinned voice_id."""
    if client is None:
        from elevenlabs import ElevenLabs

        client = ElevenLabs()

    pinned = pinned_voice_ids()
    results: list[VoiceDeletionResult] = []
    for vid in voice_ids:
        if vid in pinned:
            results.append(
                VoiceDeletionResult(
                    voice_id=vid,
                    status="skipped_pinned",
                    detail="pinned by a language pack — refusing to delete",
                )
            )
            logger.warning(f"refusing to delete pinned voice {vid}")
            continue
        if dry_run:
            results.append(VoiceDeletionResult(voice_id=vid, status="dry_run"))
            continue
        try:
            client.voices.delete(voice_id=vid)
        except Exception as e:  # noqa: BLE001 — per-voice isolation
            results.append(
                VoiceDeletionResult(voice_id=vid, status="error", detail=str(e))
            )
            logger.error(f"failed to delete voice {vid}: {e}")
            continue
        results.append(VoiceDeletionResult(voice_id=vid, status="deleted"))
        logger.info(f"deleted voice {vid}")
    return results
