# Copyright Sierra
"""Tests for the LiveKit OpenAI voice config.

The cascaded (LiveKit) voice path must accept ``reasoning_effort="none"`` so it
can match the codebase-wide gpt-5 default, while still rejecting unknown effort
values.
"""

from __future__ import annotations

import pytest


class TestLiveKitReasoningEffortNone:
    def test_config_accepts_none(self):
        from tau2.voice.audio_native.livekit.config import OpenAILLMConfig

        cfg = OpenAILLMConfig(model="gpt-5.4-mini", reasoning_effort="none")
        assert cfg.reasoning_effort == "none"

    def test_config_rejects_unknown_effort(self):
        from pydantic import ValidationError

        from tau2.voice.audio_native.livekit.config import OpenAILLMConfig

        with pytest.raises(ValidationError):
            OpenAILLMConfig(model="gpt-5.4-mini", reasoning_effort="ultra")
