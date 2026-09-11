# Copyright Sierra
"""Fail-closed tests for unavailable user-channel provider judging."""

from pathlib import Path

import pytest

from tau2.judges import user_audio_quality as module


def test_unavailable_contract_contains_no_provider_configuration() -> None:
    assert module.AVAILABILITY.status == "unavailable"
    assert module.AVAILABILITY.provider_calls_permitted is False
    assert module.AVAILABILITY.reason == "no_approved_user_audio_judge_contract"
    assert not hasattr(module, "FIDELITY_PROMPT")
    assert not hasattr(module, "INTONATION_PROMPT")
    assert not hasattr(module, "run_user_audio_quality")


def test_unavailable_operation_fails_before_reading_manifest(tmp_path: Path) -> None:
    missing = tmp_path / "not-read.json"
    with pytest.raises(
        module.UserAudioJudgingUnavailableError,
        match="no approved judge contract",
    ):
        module.run_user_audio_quality_unavailable(manifest_path=missing)
