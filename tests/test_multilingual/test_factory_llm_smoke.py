# Copyright Sierra
"""Factory LLM smoke tests (REAL provider calls — opt-in only).

Everything here is marked ``llm_smoke`` (registered in pyproject.toml) and
additionally gated on ``TAU2_FACTORY_LLM_SMOKE=1``, so the suite is skipped by
default, in CI, and under ``-m "not llm_smoke"``. Run explicitly with:

    TAU2_FACTORY_LLM_SMOKE=1 uv run pytest tests/test_multilingual -m llm_smoke

Skeleton: it pins the one LLM gateway the factory is allowed to use
(``tau2.utils.llm_utils``). Factory PRs extend this file with real-model
spot checks of their prompts (draft quality, translation invariants, etc.).
"""

import os

import pytest

LLM_SMOKE_ENABLED = os.environ.get("TAU2_FACTORY_LLM_SMOKE") == "1"
SMOKE_MODEL = os.environ.get("TAU2_FACTORY_LLM_SMOKE_MODEL", "gpt-4.1-mini")

pytestmark = [
    pytest.mark.llm_smoke,
    pytest.mark.skipif(
        not LLM_SMOKE_ENABLED,
        reason="real-LLM smoke test; set TAU2_FACTORY_LLM_SMOKE=1 to run",
    ),
]


def test_llm_gateway_round_trip():
    """The factory's LLM gateway answers and the JSON extractor parses it."""
    import json

    from tau2.data_model.message import SystemMessage, UserMessage
    from tau2.utils.llm_utils import extract_json_from_llm_response, generate

    response = generate(
        model=SMOKE_MODEL,
        messages=[
            SystemMessage(
                role="system",
                content="Reply with ONLY a JSON object: "
                '{"language": "<ISO 639-1 code of the user message>"}',
            ),
            UserMessage(role="user", content="नमस्ते, मुझे रिफंड चाहिए।"),
        ],
        call_name="factory_smoke_language_probe",
    )
    assert response.content
    payload = json.loads(extract_json_from_llm_response(response.content))
    assert payload.get("language") == "hi"
