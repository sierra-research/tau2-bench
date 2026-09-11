# Copyright Sierra
"""Fail-closed boundary for unavailable user-channel provider judging.

The deterministic user-channel panel is a valid inventory of stored audio and
intended scripts.  It does not define an approved provider-judge contract.  This
module intentionally contains no prompt, model configuration, or provider call
path.  Keeping the unavailable command explicit prevents a stale or improvised
rubric from silently creating paper evidence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict


class UserAudioJudgingUnavailableError(RuntimeError):
    """Raised before any provider operation can be configured or launched."""


class UserAudioJudgingAvailability(BaseModel):
    """Typed, stable explanation of the intentionally unavailable operation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["unavailable"] = "unavailable"
    provider_calls_permitted: Literal[False] = False
    reason: Literal["no_approved_user_audio_judge_contract"] = (
        "no_approved_user_audio_judge_contract"
    )


AVAILABILITY = UserAudioJudgingAvailability()


def run_user_audio_quality_unavailable(*, manifest_path: Path | None = None) -> None:
    """Fail before reading audio, constructing a request, or calling a provider."""

    del manifest_path
    raise UserAudioJudgingUnavailableError(
        "user-channel provider judging is unavailable: there is no approved "
        "judge contract; the deterministic user-audio panel remains usable"
    )


def run_user_audio_quality_cli(args) -> None:
    """CLI handler for the explicit fail-closed unavailable operation."""

    run_user_audio_quality_unavailable(manifest_path=getattr(args, "manifest", None))
