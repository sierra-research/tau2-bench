# Copyright Sierra
"""Regression tests for the private-to-public validation evidence projection."""

import csv
import json
from pathlib import Path

import pytest

from tau2.judges.nativeness.corrected_paper_rebind import (
    VALIDATION_DROPPED_FIELDS,
    ValidationEvidenceProjectionConfig,
    ValidationEvidenceProjectionContract,
    prove_sanitized_validation_evidence,
)
from tau2.judges.nativeness.paper_trial import (
    ValidationEvidenceIdentity,
    _sha256_value,
)
from tau2.judges.nativeness.validation import sha256_file
from tau2.judges.validation_archive import ValidationRow


def _write_csv(path: Path, fields: list[str], row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


def _archive_file(path: Path, relative: str, *, rows: int | None = None) -> dict:
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "rows": rows,
    }


def _inventory(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != root / "manifest.json"
    }


def _public_row(*, agent_utterance: str = "hola") -> dict[str, str]:
    return {
        "row_id": "0123456789abcdef0123",
        "language": "es",
        "measure_id": "naturalness",
        "source_factor_ids": '["natural_word_choice"]',
        "source_sets": '["held_out"]',
        "evaluation_level": "utterance",
        "label_origin": "direct_utterance",
        "simulation_id": "simulation",
        "task_id": "task",
        "utterance_id": "utterance",
        "utterance_alignment": "agent_turn",
        "agent_turn_index": "0",
        "agent_turn_id": "turn",
        "agent_utterance": agent_utterance,
        "stored_reference_preview": "",
        "stored_reference_alignment": "",
        "preceding_customer_utterance": "",
        "audio_start_seconds": "",
        "audio_end_seconds": "",
        "judge_quote": "",
        "human_label": "pass",
        "judge_label": "pass",
        "judge_reason": "ok",
        "agreement": "true",
        "judge_model": "fixture",
        "prompt_version": "fixture-v1",
        "rubric_version": "fixture-v1",
    }


def _write_projection_fixture(
    tmp_path: Path,
    *,
    sanitized_agent_utterance: str = "hola",
    extra_sanitized_file: bool = False,
):
    reference_root = tmp_path / "validation-repo"
    source_root = reference_root / "archives/private/validations"
    sanitized_root = reference_root / "archives/public/validations"
    leaf = Path("utterance_level/naturalness/es")
    source_leaf = source_root / leaf
    sanitized_leaf = sanitized_root / leaf

    source_public = _public_row()
    sanitized_public = _public_row(agent_utterance=sanitized_agent_utterance)
    public_fields = list(ValidationRow.model_fields)
    private_values = {
        "source_row_ids": '["source-row"]',
        "source_artifacts": json.dumps(
            [
                {
                    "role": "validation_source_manifest",
                    "path": "provenance/utterance/naturalness/es.json",
                    "sha256": "a" * 64,
                    "revision": "fixture-private-v1",
                }
            ],
            separators=(",", ":"),
        ),
        "human_note": "private reviewer note",
    }
    source_fields = []
    for field in public_fields:
        source_fields.append(field)
        if field == "source_factor_ids":
            source_fields.extend(("source_row_ids", "source_artifacts"))
        if field == "human_label":
            source_fields.append("human_note")
    source_row = {
        field: source_public.get(field, private_values.get(field, ""))
        for field in source_fields
    }
    _write_csv(source_leaf / "validation.csv", source_fields, source_row)
    _write_csv(
        sanitized_leaf / "validation.csv",
        public_fields,
        sanitized_public,
    )

    for root in (source_root, sanitized_root):
        metrics = root / leaf / "metrics.csv"
        _write_csv(metrics, ["metric", "value"], {"metric": "f1", "value": "1"})
        (root / "index.csv").parent.mkdir(parents=True, exist_ok=True)
        (root / "index.csv").write_text("language,measure\nes,naturalness\n")
        (root / "prompts.json").write_text('{"prompt":"same"}\n')
        prompt = root / "prompt_sources/naturalness/es.json"
        prompt.parent.mkdir(parents=True, exist_ok=True)
        prompt.write_text('{"prompt":"same"}\n')
    (source_root / "README.md").write_text("Values are row provenance.\n")
    (sanitized_root / "README.md").write_text("Values identify sampling strata.\n")
    provenance = source_root / "provenance/utterance/naturalness/es.json"
    provenance.parent.mkdir(parents=True, exist_ok=True)
    provenance.write_text('{"private":"source"}\n')
    if extra_sanitized_file:
        (sanitized_root / "extra.txt").write_text("not approved\n")

    def write_leaf_manifest(root: Path, schema: str) -> None:
        validation = root / leaf / "validation.csv"
        metrics = root / leaf / "metrics.csv"
        manifest = {
            "schema_version": schema,
            "evaluation_level": "utterance",
            "measure_id": "naturalness",
            "language": "es",
            "validation": _archive_file(validation, "validation.csv", rows=1),
            "metrics": _archive_file(metrics, "metrics.csv", rows=1),
            "rows": 1,
            "metric_rows": 1,
        }
        (root / leaf / "manifest.json").write_text(json.dumps(manifest, indent=2))

    write_leaf_manifest(source_root, "tau-multi-validation-leaf-v1")
    write_leaf_manifest(sanitized_root, "tau-multi-validation-leaf-v2")

    def write_root_manifest(root: Path, schema: str) -> None:
        leaf_manifest = root / leaf / "manifest.json"
        manifest = {
            "schema_version": schema,
            "index": _archive_file(root / "index.csv", "index.csv", rows=1),
            "readme": _archive_file(root / "README.md", "README.md"),
            "partitions": [
                {
                    "evaluation_level": "utterance",
                    "measure_id": "naturalness",
                    "language": "es",
                    "manifest": _archive_file(
                        leaf_manifest, (leaf / "manifest.json").as_posix()
                    ),
                }
            ],
            "rows": 1,
            "metric_rows": 1,
        }
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    write_root_manifest(source_root, "tau-multi-validations-v1")
    write_root_manifest(sanitized_root, "tau-multi-validations-v2")

    source_inventory = _inventory(source_root)
    sanitized_inventory = _inventory(sanitized_root)
    projection = [
        {
            "partition": (leaf / "validation.csv").as_posix(),
            "row": source_public,
        }
    ]
    contract = ValidationEvidenceProjectionContract(
        source_manifest_sha256=sha256_file(source_root / "manifest.json"),
        sanitized_manifest_sha256=sha256_file(sanitized_root / "manifest.json"),
        source_inventory_sha256=_sha256_value(source_inventory),
        sanitized_inventory_sha256=_sha256_value(sanitized_inventory),
        public_projection_sha256=_sha256_value(projection),
        partitions=1,
        rows=1,
        metric_rows=1,
        prompt_source_files=1,
        invariant_files=4,
        removed_provenance_files=1,
        source_files=len(source_inventory),
        sanitized_files=len(sanitized_inventory),
    )
    prior = ValidationEvidenceIdentity(
        path="historical/human_annotations/validations",
        manifest_sha256=sha256_file(source_root / "manifest.json"),
        files=source_inventory,
    )
    config = ValidationEvidenceProjectionConfig(
        validation_repo_root=reference_root,
        source_root=source_root,
        sanitized_root=sanitized_root,
    )
    return config, prior, contract


def _stub_semantic_archive_validation(monkeypatch) -> None:
    from tau2.judges.nativeness import corrected_paper_rebind

    monkeypatch.setattr(
        corrected_paper_rebind,
        "read_validation_archive",
        lambda _root: ([object()], [object()], [object()]),
    )


def test_sanitized_validation_projection_is_exact_and_portable(tmp_path, monkeypatch):
    _stub_semantic_archive_validation(monkeypatch)
    config, prior, contract = _write_projection_fixture(tmp_path)

    proof = prove_sanitized_validation_evidence(
        config, prior_evidence=prior, contract=contract
    )

    assert proof.prior_evidence == prior
    assert proof.sanitized_evidence.path == "archives/public/validations"
    assert proof.dropped_fields == VALIDATION_DROPPED_FIELDS
    assert proof.source_projection_sha256 == proof.sanitized_projection_sha256
    assert proof.counts.partitions == 1
    assert proof.counts.rows == 1
    assert proof.counts.removed_provenance_files == 1


def test_sanitized_validation_projection_rejects_modified_public_field(
    tmp_path, monkeypatch
):
    _stub_semantic_archive_validation(monkeypatch)
    config, prior, contract = _write_projection_fixture(
        tmp_path, sanitized_agent_utterance="adios"
    )

    with pytest.raises(ValueError, match="change a public field"):
        prove_sanitized_validation_evidence(
            config, prior_evidence=prior, contract=contract
        )


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_sanitized_validation_projection_rejects_file_inventory_drift(
    tmp_path, monkeypatch, mutation
):
    _stub_semantic_archive_validation(monkeypatch)
    config, prior, contract = _write_projection_fixture(
        tmp_path, extra_sanitized_file=mutation == "extra"
    )
    if mutation == "missing":
        (config.sanitized_root / "utterance_level/naturalness/es/metrics.csv").unlink()
        sanitized_inventory = _inventory(config.sanitized_root)
        contract = contract.model_copy(
            update={
                "sanitized_inventory_sha256": _sha256_value(sanitized_inventory),
                "sanitized_files": len(sanitized_inventory),
            }
        )

    pattern = "extra file" if mutation == "extra" else "regular file"
    with pytest.raises(ValueError, match=pattern):
        prove_sanitized_validation_evidence(
            config, prior_evidence=prior, contract=contract
        )


def test_sanitized_validation_projection_rejects_unsafe_root(tmp_path, monkeypatch):
    _stub_semantic_archive_validation(monkeypatch)
    config, prior, contract = _write_projection_fixture(tmp_path)
    unsafe = config.model_copy(
        update={
            "validation_repo_root": config.validation_repo_root / "archives/private"
        }
    )

    with pytest.raises(ValueError, match="portable reference root"):
        prove_sanitized_validation_evidence(
            unsafe, prior_evidence=prior, contract=contract
        )
