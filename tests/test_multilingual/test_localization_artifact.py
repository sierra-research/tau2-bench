"""Compact maps rehydrate only against their exact inventoried source."""

import json
from pathlib import Path

import pytest

from tau2.multilingual.factory import localization_artifact as artifacts
from tau2.multilingual.factory.localization_inventory import inventory_localization
from tau2.multilingual.localization import (
    GenerationProvenance,
    LocalizationEntry,
    LocalizationMap,
    LocalizedForms,
    RegionalSettings,
    json_content_digest,
)


@pytest.fixture
def draft(monkeypatch):
    source = inventory_localization("telecom", Path("data").resolve())
    settings = RegionalSettings(
        language="zh",
        country_code="CN",
        currency_code="CNY",
        currency_scale="7",
        time_zone="Asia/Shanghai",
        name_order="family_first",
    )
    proposal = LocalizationMap(
        revision="compact-fixture-v1",
        settings=settings,
        source=source,
        entries=[
            artifacts.automatic_localization(entry, settings)
            or LocalizationEntry(
                **entry.model_dump(),
                forms=LocalizedForms(
                    native=entry.source,
                    romanized=entry.source,
                    english="Unchanged test fixture only.",
                ),
                operation="preserve",
                notes="Not an authored regional proposal.",
            )
            for entry in source.entries
        ],
        provenance=GenerationProvenance(
            model="fixture",
            model_args={},
            prompt_version="fixture-v1",
            call_name="fixture",
            input_sha256="0" * 64,
        ),
    )
    monkeypatch.setattr(
        artifacts, "inventory_localization", lambda *args: source.model_copy(deep=True)
    )
    return proposal


def test_compact_roundtrip_has_no_source_snapshots_and_is_idempotent(draft, tmp_path):
    path = tmp_path / "map.json"
    assert artifacts.write_localization_map(draft, path)
    before = path.read_bytes()
    payload = json.loads(before)
    assert all("content" not in document for document in payload["source"]["documents"])
    assert all(
        set(entry) == {"id", "category", "source", "forms", "operation", "notes"}
        for entry in payload["entries"]
    )
    assert len(payload["entries"]) < len(draft.entries)
    assert artifacts.read_localization_map(path) == draft
    assert not artifacts.write_localization_map(draft, path)
    assert path.read_bytes() == before


def refreshed_source(draft, monkeypatch, *, changed_text=False):
    from tau2.multilingual.factory import localization_inventory as inventory

    documents = [d.model_copy(deep=True) for d in draft.source.documents]
    evidence = next(d for d in documents if d.kind == "tool_text" and d.content)
    evidence.content[0]["line"] += 100
    if changed_text:
        evidence.content[0]["text"] += " changed arithmetic"
    evidence.sha256 = json_content_digest(evidence.content)
    monkeypatch.setattr(inventory, "_source_documents", lambda *args: documents)
    return inventory.inventory_localization("telecom", Path("data"))


@pytest.mark.parametrize("changed_text", [False, True])
def test_explicit_source_refresh_preserves_authored_values_and_generation(
    draft, monkeypatch, tmp_path, changed_text
):
    from tau2.utils.artifacts import content_digest

    old = tmp_path / "old.json"
    artifacts.write_localization_map(draft, old)
    previous = artifacts.RegionalMapArtifact.model_validate_json(old.read_text())
    source = refreshed_source(draft, monkeypatch, changed_text=changed_text)
    monkeypatch.setattr(artifacts, "inventory_localization", lambda *args: source)
    with pytest.raises(ValueError, match="manifest"):
        artifacts.read_localization_map(old)
    assert "source_refresh" not in previous.model_dump()
    assert "source_refresh" not in draft.model_dump()
    result = artifacts.refresh_localization_source(previous, source, "rebound")
    assert result.provenance == draft.provenance
    assert result.source_refresh.version == "regional-source-evidence-v1"
    assert result.source_refresh.parent_artifact_sha256 == content_digest(previous)
    assert result.source_refresh.previous_source_sha256 == content_digest(
        previous.source
    )
    assert result.source_refresh.source_sha256 == content_digest(source)
    new = tmp_path / "new.json"
    artifacts.write_localization_map(result, new)
    written = artifacts.RegionalMapArtifact.model_validate_json(new.read_text())
    assert written.entries == previous.entries
    assert artifacts.read_localization_map(new) == result
    assert not artifacts.write_localization_map(result, new)


@pytest.mark.parametrize("drift", ["data", "policy", "old_hash", "revision"])
def test_source_refresh_rejects_changes_beyond_code_evidence(
    draft, monkeypatch, tmp_path, drift
):
    path = tmp_path / "old.json"
    artifacts.write_localization_map(draft, path)
    before = path.read_bytes()
    previous = artifacts.RegionalMapArtifact.model_validate_json(path.read_text())
    source = refreshed_source(draft, monkeypatch)
    revision = "rebound"
    if drift == "data":
        doc = next(d for d in source.documents if d.kind == "database")
        doc.content["customers"][0]["full_name"] = "Changed"
        doc.sha256 = json_content_digest(doc.content)
    elif drift == "policy":
        doc = next(d for d in source.documents if d.kind == "policy")
        doc.content += " Different rule."
        doc.sha256 = json_content_digest(doc.content)
    elif drift == "old_hash":
        previous.source.documents[0].sha256 = "f" * 64
    elif drift == "revision":
        revision = previous.revision
    with pytest.raises(ValueError):
        artifacts.refresh_localization_source(previous, source, revision)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "mutation",
    [
        "hash",
        "path",
        "kind",
        "missing_document",
        "extra_document",
        "duplicate_document",
    ],
)
def test_compact_map_refuses_changed_source_manifest(draft, tmp_path, mutation):
    path = tmp_path / "map.json"
    artifacts.write_localization_map(draft, path)
    payload = json.loads(path.read_text())
    rows = payload["source"]["documents"]
    if mutation == "hash":
        rows[0]["sha256"] = "f" * 64
    elif mutation == "path":
        rows[0]["source_path"] = "different.json"
    elif mutation == "kind":
        rows[0]["kind"] = "policy"
    elif mutation == "missing_document":
        rows.pop()
    else:
        rows.append(dict(rows[0]))
        if mutation == "extra_document":
            rows[-1]["id"] = "extra"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="source|document|manifest"):
        artifacts.read_localization_map(path)


@pytest.mark.parametrize(
    "mutation", ["missing", "extra", "duplicate", "automatic", "source", "category"]
)
def test_compact_map_requires_exact_authored_entries(draft, tmp_path, mutation):
    path = tmp_path / "map.json"
    artifacts.write_localization_map(draft, path)
    payload = json.loads(path.read_text())
    rows = payload["entries"]
    if mutation == "missing":
        rows.pop()
    elif mutation == "source":
        rows[0]["source"] = "Stale source"
    elif mutation == "category":
        rows[0]["category"] = "other"
    else:
        rows.append(dict(rows[0]))
        if mutation == "extra":
            rows[-1]["id"] = "extra"
        elif mutation == "automatic":
            rows[-1]["id"] = next(
                entry.id
                for entry in draft.source.entries
                if entry.category == "protocol"
            )
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="entries|entry|authored"):
        artifacts.read_localization_map(path)


def test_loader_uses_explicit_data_root_and_rejects_changed_content(
    draft, tmp_path, monkeypatch
):
    path = tmp_path / "map.json"
    artifacts.write_localization_map(draft, path)
    seen = []

    def stale_inventory(domain, data_root):
        seen.append((domain, data_root))
        source = draft.source.model_copy(deep=True)
        source.documents[0].content["customers"][0]["full_name"] = "Changed"
        source.documents[0].sha256 = json_content_digest(source.documents[0].content)
        return source

    monkeypatch.setattr(artifacts, "inventory_localization", stale_inventory)
    root = tmp_path / "explicit-data"
    with pytest.raises(ValueError, match="source|manifest"):
        artifacts.read_localization_map(path, root)
    assert seen == [("telecom", root)]


def test_write_cannot_silently_drop_modified_automatic_forms(draft, tmp_path):
    row = next(entry for entry in draft.entries if entry.category == "tool_text")
    row.forms.native = "Changed code evidence"
    with pytest.raises(ValueError):
        artifacts.write_localization_map(draft, tmp_path / "map.json")


def test_unknown_automatic_rules_are_rejected(draft, tmp_path):
    path = tmp_path / "map.json"
    artifacts.write_localization_map(draft, path)
    payload = json.loads(path.read_text())
    payload["automatic_rules_version"] = "future-version"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        artifacts.read_localization_map(path)


def test_regional_build_keeps_original_tool_evidence_and_rejects_data_drift(
    draft, tmp_path, monkeypatch
):
    from tau2.data_model.language import TaskMetadata
    from tau2.multilingual.factory import localization_publish as publish
    from tau2.utils.artifacts import content_digest

    path = tmp_path / "map.json"
    artifacts.write_localization_map(draft, path)
    artifact = artifacts.RegionalMapArtifact.model_validate_json(path.read_text())
    tasks = next(d.content for d in draft.source.documents if d.kind == "tasks")
    metadata = {
        task["id"]: TaskMetadata(canonical_task_id=task["id"]) for task in tasks
    }
    changed_code = draft.source.model_copy(deep=True)
    evidence = next(
        d for d in changed_code.documents if d.kind == "tool_text" and d.content
    )
    evidence.content[0]["text"] += " regional runtime implementation"
    evidence.sha256 = json_content_digest(evidence.content)
    monkeypatch.setattr(publish, "inventory_localization", lambda *args: changed_code)
    bundle = publish.build_regional_bundle(
        artifact, draft.source, metadata, Path("data")
    )
    assert bundle.localization_map_sha256 == content_digest(artifact)
    assert "reviewed_by" not in bundle.model_dump_json()
    assert bundle.source_document_sha256[evidence.id] != evidence.sha256
    assert bundle.tasks[0].id == tasks[0]["id"]
    assert bundle.settings.currency_code == "CNY"
    changed_data = changed_code.model_copy(deep=True)
    policy = next(d for d in changed_data.documents if d.kind == "policy")
    policy.content += "\nChanged canonical obligations."
    policy.sha256 = json_content_digest(policy.content)
    monkeypatch.setattr(publish, "inventory_localization", lambda *args: changed_data)
    with pytest.raises(ValueError, match="source database, tasks or policies changed"):
        publish.build_regional_bundle(artifact, draft.source, metadata, Path("data"))
