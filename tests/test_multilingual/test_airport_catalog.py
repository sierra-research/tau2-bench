"""Airport city authoring stays bound to the reviewed code map."""

import json
from types import SimpleNamespace

import pytest

from tau2.multilingual.airport_catalog import AirportCatalog
from tau2.multilingual.factory import airport_catalog as authoring
from tau2.multilingual.factory.generation_config import FactoryModelConfig
from tau2.multilingual.factory.localization_artifact import RegionalMapArtifact
from tau2.utils.artifacts import content_digest


def _map():
    return RegionalMapArtifact.model_validate(
        {
            "revision": "zh-airline-reviewed",
            "settings": {
                "language": "zh",
                "country_code": "CN",
                "currency_code": "CNY",
                "currency_scale": "7",
                "time_zone": "Asia/Shanghai",
                "name_order": "family_first",
            },
            "source": {
                "domain": "airline",
                "documents": [
                    {
                        "id": "database",
                        "kind": "database",
                        "source_path": "db.json",
                        "sha256": "b" * 64,
                    }
                ],
            },
            "entries": [
                {
                    "id": f"airport:{source}",
                    "category": "airport",
                    "source": source,
                    "forms": {
                        "native": target,
                        "romanized": target,
                        "english": explanation,
                    },
                    "operation": "adapt",
                    "notes": "Reviewed mapping; keep this code.",
                }
                for source, target, explanation in [
                    ("JFK", "PVG", "Shanghai Pudong airport."),
                    ("EWR", "SHA", "Shanghai Hongqiao airport."),
                ]
            ],
            "provenance": {
                "model": "fixture",
                "model_args": {},
                "prompt_version": "fixture",
                "call_name": "fixture",
                "input_sha256": "c" * 64,
            },
        }
    )


def _reply():
    return {
        "airports": [
            {
                "source_iata": source,
                "iata": target,
                "native_city": "上海",
                "romanized_city": "Shanghai",
                "english_city": "Shanghai",
                "notes": "City served by this airport.",
            }
            for source, target in [("EWR", "SHA"), ("JFK", "PVG")]
        ]
    }


def _settings(**changes):
    return authoring.AirportCatalogDraftRequest(
        revision="zh-airports-1",
        generation=FactoryModelConfig(
            model="fixture-model", reasoning_effort="minimal"
        ),
        **changes,
    )


def test_generate_and_reuse_exact_reviewed_input(monkeypatch, tmp_path):
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(content=json.dumps(_reply()))

    monkeypatch.setattr(authoring, "generate", generate)
    proposal = _map()
    before = proposal.model_dump_json()
    output = tmp_path / "catalog.json"
    catalog = authoring.generate_airport_catalog(proposal, _settings(), output)
    assert len(calls) == 1
    assert calls[0]["call_name"] == "factory_airport_city_catalog"
    assert calls[0]["model"] == "fixture-model"
    assert calls[0]["reasoning_effort"] == "minimal"
    assert calls[0]["messages"][0].content == authoring.AIRPORT_CATALOG_PROMPT
    payload = json.loads(calls[0]["messages"][1].content)
    assert {row["iata"] for row in payload["airports"]} == {"SHA", "PVG"}
    assert catalog.map_sha256 == content_digest(proposal)
    assert catalog.provenance.prompt_version == authoring.AIRPORT_CATALOG_PROMPT_VERSION
    assert proposal.model_dump_json() == before
    assert AirportCatalog.model_validate_json(output.read_text()) == catalog
    assert authoring.generate_airport_catalog(proposal, _settings(), output) == catalog
    assert len(calls) == 1
    with pytest.raises(FileExistsError, match="inputs changed"):
        authoring.generate_airport_catalog(
            proposal,
            _settings().model_copy(
                update={
                    "generation": FactoryModelConfig(
                        model="changed", reasoning_effort="minimal"
                    )
                }
            ),
            output,
        )


def test_prompt_revision_requires_new_output_path(monkeypatch, tmp_path):
    monkeypatch.setattr(
        authoring,
        "generate",
        lambda **kw: SimpleNamespace(content=json.dumps(_reply())),
    )
    proposal = _map()
    output = tmp_path / "catalog.json"
    authoring.generate_airport_catalog(proposal, _settings(), output)
    before = output.read_bytes()
    monkeypatch.setattr(authoring, "AIRPORT_CATALOG_PROMPT_VERSION", "new-prompt")
    monkeypatch.setattr(
        authoring, "generate", lambda **kw: pytest.fail("unexpected model call")
    )
    with pytest.raises(FileExistsError, match="inputs changed"):
        authoring.generate_airport_catalog(proposal, _settings(), output)
    assert output.read_bytes() == before


@pytest.mark.parametrize(
    "change",
    ["missing", "duplicate", "different_code", "blank_city", "same_size_wrong_pair"],
)
def test_generated_coverage_must_match_reviewed_codes(monkeypatch, tmp_path, change):
    value = _reply()
    if change == "missing":
        value["airports"].pop()
    elif change == "duplicate":
        value["airports"].append(value["airports"][0])
    elif change == "different_code":
        value["airports"][0]["iata"] = "PEK"
    elif change == "blank_city":
        value["airports"][0]["native_city"] = "   "
    else:
        value["airports"][0]["source_iata"], value["airports"][1]["source_iata"] = (
            "JFK",
            "EWR",
        )
    monkeypatch.setattr(
        authoring, "generate", lambda **kw: SimpleNamespace(content=json.dumps(value))
    )
    proposal = _map()
    with pytest.raises(ValueError):
        authoring.generate_airport_catalog(proposal, _settings(), tmp_path / "bad.json")
    assert not (tmp_path / "bad.json").exists()
