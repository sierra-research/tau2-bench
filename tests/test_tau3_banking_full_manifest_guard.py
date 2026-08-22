"""Offline integrity tests for the active tau3 banking Modal order fixture."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS_DIR = REPO_ROOT / "reproduction" / "tau3_banking"
if str(HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HARNESS_DIR))

import run as reproduction_run  # noqa: E402


def _config() -> dict:
    return json.loads((HARNESS_DIR / "reference.json").read_text(encoding="utf-8"))


def _manifest() -> dict:
    return json.loads(
        (HARNESS_DIR / "full_shell_order_manifest.json").read_text(encoding="utf-8")
    )


def _write_forged_manifest(tmp_path: Path, config: dict, manifest: dict) -> Path:
    path = tmp_path / "full_shell_order_manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fixture = config["reproduction_transport"]["sandbox_order_manifest_integrity"]
    fixture["file_sha256"] = reproduction_run.digest_file(path)
    return path


def test_committed_full_manifest_matches_every_pinned_integrity_field():
    manifest = reproduction_run.verify_modal_order_manifest(_config())

    assert manifest["schema_version"] == 1
    assert manifest["entry_count"] == 699
    assert manifest["document_count"] == 698
    assert manifest["order_sha256"] == (
        "ddb11f1a583e408079c136805c786f6e53903afb3dad46047c69a06b3b01b6f3"
    )
    assert manifest["corpus_export_sha256"] == (
        "395ccccab4cf1eeebcefc10307431c3f0525ac790bb159ff2fa7abbc01bd6199"
    )


def test_full_manifest_byte_tampering_fails_before_paid_execution(tmp_path):
    config = _config()
    manifest = _manifest()
    manifest["filenames"][0], manifest["filenames"][1] = (
        manifest["filenames"][1],
        manifest["filenames"][0],
    )
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(reproduction_run.RunGuardError, match="file SHA-256"):
        reproduction_run.verify_modal_order_manifest(config, path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("entry_count", 700, "entry count"),
        ("order_sha256", "0" * 64, "order SHA-256"),
        ("corpus_export_sha256", "0" * 64, "corpus checksum"),
    ),
)
def test_forged_file_digest_cannot_bypass_loaded_manifest_bindings(
    tmp_path, field, value, message
):
    config = copy.deepcopy(_config())
    manifest = _manifest()
    manifest[field] = value
    path = _write_forged_manifest(tmp_path, config, manifest)

    with pytest.raises(reproduction_run.RunGuardError, match=message):
        reproduction_run.verify_modal_order_manifest(config, path)


def test_malformed_integrity_metadata_fails_closed():
    config = _config()
    fixture = config["reproduction_transport"]["sandbox_order_manifest_integrity"]
    fixture.pop("corpus_export_sha256")

    with pytest.raises(reproduction_run.RunGuardError, match="metadata is malformed"):
        reproduction_run.verify_modal_order_manifest(config)
