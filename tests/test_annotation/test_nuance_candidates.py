# Copyright Sierra
"""Fixed-prompt nuance-candidate stage (LLM mocked at the generate() seam)."""

import json
from types import SimpleNamespace

import pytest

import tau2.annotation.nuance_candidates as nc
from tau2.annotation.models import AuditStatus
from tau2.annotation.nuance_candidates import (
    LANGUAGE_HINTS,
    NUANCE_CANDIDATES_CALL_NAME,
    NUANCE_CANDIDATES_PROMPT,
    generate_nuance_candidates,
    load_nuance_candidates,
    nuance_candidates_prompt,
    nuance_prompt_sha256,
    write_nuance_candidates,
)
from tau2.annotation.sheets.audit import AUDIT_TABS

PAYLOAD = {
    "nuances": [
        {
            "category": "Register & formality",
            "title": "aap vs tum drift",
            "ai_likely_does": "slips into tum",
            "native_does": "keeps aap with a stranger agent ('Aap bataiye...')",
            "severity": "3",  # stringly severity coerced by the validator
        },
        {
            "category": "Grammatical gender agreement",
            "title": "karta vs karti",
            "ai_likely_does": "uses karta hoon for a female speaker",
            "native_does": "karti hoon (feminine first-person agreement)",
            "severity": 2,
        },
    ]
}


def _mock_generate(monkeypatch, content: str):
    calls: list[dict] = []

    def fake_generate(model, messages, call_name=None, **kwargs):
        calls.append(
            {
                "model": model,
                "messages": messages,
                "call_name": call_name,
                "kwargs": kwargs,
            }
        )
        return SimpleNamespace(content=content)

    monkeypatch.setattr(nc, "generate", fake_generate)
    return calls


def test_prompt_is_fixed_and_filled():
    prompt = nuance_candidates_prompt("Hindi", LANGUAGE_HINTS["hi"])
    assert "{{LANG}}" not in prompt and "{{HINTS}}" not in prompt
    assert "senior Hindi linguist" in prompt
    assert "aap/tum/tu" in prompt  # hints landed
    # The template itself is versioned + hashed for provenance.
    assert "{{LANG}}" in NUANCE_CANDIDATES_PROMPT
    assert len(nuance_prompt_sha256()) == 64


def test_hints_cover_every_audit_language():
    assert set(LANGUAGE_HINTS) == set(AUDIT_TABS)


def test_generate_parses_reply_through_seam(monkeypatch):
    calls = _mock_generate(monkeypatch, json.dumps(PAYLOAD))
    candidates = generate_nuance_candidates("hi", model="fake-model")
    assert calls[0]["call_name"] == NUANCE_CANDIDATES_CALL_NAME
    assert calls[0]["model"] == "fake-model"
    assert "senior Hindi linguist" in calls[0]["messages"][0].content
    assert len(candidates) == 2
    assert candidates[0].severity == 3  # coerced from "3"
    row = candidates[0].to_row()
    assert row.status is AuditStatus.AUTO
    assert row.nuance == "aap vs tum drift"
    assert row.ai_does == "slips into tum"


def test_generate_tolerates_fenced_reply(monkeypatch):
    _mock_generate(monkeypatch, "```json\n" + json.dumps(PAYLOAD) + "\n```")
    assert len(generate_nuance_candidates("hi", model="fake-model")) == 2


def test_generate_garbage_reply_is_loud(monkeypatch):
    _mock_generate(monkeypatch, "I could not think of any nuances, sorry!")
    with pytest.raises(ValueError):
        generate_nuance_candidates("hi", model="fake-model")


def test_generate_unknown_language_is_loud():
    with pytest.raises(ValueError, match="unknown audit language"):
        generate_nuance_candidates("xx", model="fake-model")


def test_write_and_load_round_trip_with_provenance(monkeypatch, tmp_path):
    _mock_generate(monkeypatch, json.dumps(PAYLOAD))
    candidates = generate_nuance_candidates("hi", model="fake-model")
    path = write_nuance_candidates("hi", candidates, tmp_path, model="fake-model")
    assert path == tmp_path / "hi.json"
    payload = json.loads(path.read_text())
    # Audit-builder wire shape preserved + provenance keys added.
    assert payload["nuances"][0]["title"] == "aap vs tum drift"
    assert payload["prompt_sha256"] == nuance_prompt_sha256()
    assert payload["prompt_version"] == nc.NUANCE_CANDIDATES_PROMPT_VERSION
    assert payload["model"] == "fake-model"
    loaded = load_nuance_candidates(path)
    assert loaded == candidates


def test_load_accepts_legacy_candidates_without_provenance(tmp_path):
    path = tmp_path / "hi.json"
    path.write_text(json.dumps(PAYLOAD))

    loaded = load_nuance_candidates(path)

    assert [candidate.title for candidate in loaded] == [
        "aap vs tum drift",
        "karta vs karti",
    ]
    assert loaded[0].severity == 3
