"""Banking dependent prose never discloses locally reused candidate values."""

import json

import pytest

from tau2.domains.banking_knowledge.localization_tokens import BankingReferenceTokens
from tau2.multilingual.factory.generation_config import FactoryModelConfig


def test_banking_reference_tokens_keep_candidates_local_and_restore_both_forms():
    payload = {
        "domain": "banking_knowledge",
        "entries": [{"id": "prose", "source": "Call Ada at ada@example.com."}],
        "completed_mappings": [
            {
                "id": "name",
                "category": "identity_name",
                "source": "Ada",
                "forms": {
                    "native": "नाम",
                    "romanized": "Naam",
                    "english": "Donor private explanation",
                },
            },
            {
                "id": "email",
                "category": "contact",
                "source": "ada@example.com",
                "forms": {
                    "native": "naam@example.es",
                    "romanized": "naam@example.es",
                    "english": "Derived from donor",
                },
            },
        ],
    }
    tokens = BankingReferenceTokens.protect(payload)
    serialized = json.dumps(payload, ensure_ascii=False)
    for secret in [
        "नाम",
        "Naam",
        "naam@example.es",
        "Donor private",
        "Derived from donor",
    ]:
        assert secret not in serialized
    name, email = payload["completed_mappings"]
    response = {
        "entries": [
            {
                "id": "prose",
                "forms": {
                    "native": f"Call {name['forms']['native']} at {email['forms']['native']}.",
                    "romanized": f"Call {name['forms']['romanized']} at {email['forms']['romanized']}.",
                    "english": "Mapped identity only.",
                },
                "operation": "adapt",
                "notes": "",
            }
        ]
    }
    restored = tokens.restore(response)
    assert restored["entries"][0]["forms"]["native"] == "Call नाम at naam@example.es."
    assert (
        restored["entries"][0]["forms"]["romanized"] == "Call Naam at naam@example.es."
    )
    assert "LOCALIZED" not in json.dumps(restored)


def test_banking_reference_tokens_reject_unknown_or_wrong_form_tokens():
    payload = {
        "completed_mappings": [
            {
                "id": "name",
                "forms": {"native": "नाम", "romanized": "Naam", "english": "Local"},
            }
        ]
    }
    tokens = BankingReferenceTokens.protect(payload)
    wrong = payload["completed_mappings"][0]["forms"]["romanized"]
    for value in [wrong, "__LOCALIZED_UNKNOWN_N__"]:
        with pytest.raises(ValueError, match="reference token"):
            tokens.restore(
                {
                    "entries": [
                        {
                            "forms": {
                                "native": value,
                                "romanized": "literal",
                                "english": "",
                            },
                            "notes": "",
                        }
                    ]
                }
            )


def test_real_banking_batch_masks_outgoing_payload_and_restores_checkpoint(
    tmp_path, monkeypatch
):
    from pathlib import Path
    from types import SimpleNamespace

    from tau2.multilingual.factory import database_localization as drafting
    from tau2.multilingual.factory.localization_inventory import inventory_localization
    from tau2.multilingual.localization import (
        LocalizationEntry,
        LocalizedForms,
        RegionalSettings,
    )

    source = inventory_localization(
        "banking_knowledge", Path(__file__).resolve().parents[2] / "data"
    )
    entry = next(
        e for e in source.entries if e.category == "prose" and isinstance(e.source, str)
    )
    name = next(e for e in source.entries if e.category == "identity_name")
    completed = LocalizationEntry(
        **name.model_dump(),
        forms=LocalizedForms(
            native="PRIVATE_NATIVE_नाम",
            romanized="PRIVATE_LATIN_NAAM",
            english="PRIVATE_REASON",
        ),
        operation="adapt",
        notes="Not provider context",
    )
    request = drafting.RegionalDraftRequest(
        source=source,
        settings=RegionalSettings(
            language="hi",
            country_code="IN",
            currency_code="USD",
            currency_scale="1",
            time_zone="America/New_York",
            name_order="given_first",
            notes=[],
        ),
        revision="test",
        generation=FactoryModelConfig(model="test", reasoning_effort="low"),
    )
    batch = drafting.BatchInput(
        request_sha256="a" * 64, entries=[entry], completed=[completed]
    )
    seen = []

    def generate(**kwargs):
        payload = json.loads(kwargs["messages"][1].content)
        seen.append(payload)
        serialized = json.dumps(
            [m.content for m in kwargs["messages"]], ensure_ascii=False
        )
        assert "PRIVATE_" not in serialized
        assert kwargs["response_format"]["json_schema"]["strict"] is True
        assert (
            kwargs["response_format"]["json_schema"]["schema"]
            == payload["response_schema"]
        )
        forms = payload["completed_mappings"][0]["forms"]
        if len(seen) == 1:
            return SimpleNamespace(content=json.dumps({"entries": []}))
        assert (
            '"required_entry_ids": ["' + entry.id + '"]'
            in kwargs["messages"][-1].content
        )
        assert '"required_count": 1' in kwargs["messages"][-1].content
        return SimpleNamespace(
            content=json.dumps(
                {
                    "entries": {
                        entry.id: {
                            "forms": {
                                "native": entry.source + " " + forms["native"],
                                "romanized": entry.source + " " + forms["romanized"],
                                "english": "Test only.",
                            },
                            "operation": "adapt",
                            "notes": "",
                        }
                    }
                }
            )
        )

    monkeypatch.setattr(drafting, "generate", generate)
    result = drafting._generate_batch(request, batch, tmp_path)
    assert result[0].forms.native.endswith("PRIVATE_NATIVE_नाम")
    assert result[0].forms.romanized.endswith("PRIVATE_LATIN_NAAM")
    assert drafting._generate_batch(request, batch, tmp_path) == result
    assert len(seen) == 2
    artifact = drafting.BatchArtifact.model_validate_json(
        next(tmp_path.glob("*.json")).read_text()
    )
    assert artifact.response_schema_version == "banking-string-entries-v2"


def test_banking_response_schema_requires_every_exact_key_without_extra_rows():
    from pydantic import ValidationError

    from tau2.domains.banking_knowledge.localization_tokens import (
        banking_response_model,
    )

    model = banking_response_model(["prose:a", "contact:b"])
    value = {
        "forms": {"native": "test", "romanized": "test", "english": "test"},
        "operation": "preserve",
        "notes": "",
    }
    valid = {"entries": {"prose:a": value, "contact:b": value}}
    assert model.model_validate(valid).model_dump() == valid
    schema = model.model_json_schema()
    entry_schema = schema["$defs"]["BankingRequestedEntries"]
    assert set(entry_schema["required"]) == {"prose:a", "contact:b"}
    assert entry_schema["additionalProperties"] is False
    for invalid in [
        {"entries": {"prose:a": value}},
        {"entries": {**valid["entries"], "other": value}},
        {
            "entries": {
                "prose:a": {**value, "forms": {**value["forms"], "native": 123}},
                "contact:b": value,
            }
        },
    ]:
        with pytest.raises(ValidationError):
            model.model_validate(invalid)
