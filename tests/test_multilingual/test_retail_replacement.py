import hashlib
import json
from pathlib import Path

from tau2.paper.retail_replacement import load_retail_replacement_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = (
    REPO_ROOT / "papers/tau-multilingual/reproduction/retail_name_role_replacement.json"
)


def test_checked_in_retail_replacement_manifest_is_exact_and_portable() -> None:
    manifest = load_retail_replacement_manifest(MANIFEST_PATH)

    assert len(manifest.cells) == 14
    assert sum(cell.replacement.calls for cell in manifest.cells) == 1_100
    assert sum(cell.modality == "voice" for cell in manifest.cells) == 10
    assert sum(cell.modality == "text" for cell in manifest.cells) == 4
    assert {cell.language for cell in manifest.cells} == {"ko", "zh"}
    assert manifest.unchanged_active_cells == 112
    assert manifest.exclusions.included_infrastructure_failures == 0
    assert manifest.human_validation_preserved_unchanged
    assert not manifest.old_archive_uploaded
    assert _sha256(REPO_ROOT / manifest.previous_audit.path) == (
        manifest.previous_audit.sha256
    )
    assert all(
        cell.archive_path == f"validation_runs/{cell.language}/{cell.canonical_path}"
        for cell in manifest.cells
    )
    assert manifest.artifacts.latency_analysis.path == (
        "papers/tau-multilingual/reproduction/latency.json"
    )
    assert manifest.artifacts.experience_analysis.path == (
        "papers/tau-multilingual/reproduction/validation_runs/"
        "intermediate-trial0-results-binding-v1/"
        "tau_multilingual_experience_without_fluency_2026-09-18.json"
    )
    assert "/private/tmp" not in MANIFEST_PATH.read_text()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fingerprint(cells: list[dict[str, object]]) -> str:
    lines = [
        f"{cell['results_path']}\t{cell['sha256']}\t{cell['rows']}\n"
        for cell in sorted(cells, key=lambda item: str(item["results_path"]))
    ]
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def test_manifest_binds_current_audit_and_portable_artifacts() -> None:
    manifest = load_retail_replacement_manifest(MANIFEST_PATH)
    audit_path = REPO_ROOT / manifest.replacement_audit.path
    audit = json.loads(audit_path.read_text())
    current = audit["voice_cells"] + audit["text_cells"]
    replacement_keys = {
        (cell.modality, cell.language, "retail", cell.system): cell
        for cell in manifest.cells
    }
    current_by_key = {
        (cell["cohort"], cell["language"], cell["domain"], cell["system"]): cell
        for cell in current
    }

    assert _sha256(audit_path) == manifest.replacement_audit.sha256
    assert audit["artifact_id"] == manifest.replacement_audit.artifact_id
    for key, replacement in replacement_keys.items():
        cell = current_by_key[key]
        assert cell["results_path"] == replacement.canonical_path
        assert cell["sha256"] == replacement.replacement.results_sha256
        assert cell["rows"] == replacement.replacement.calls
        assert tuple(cell["trials"]) == replacement.replacement.trials
        assert cell["trial_success"] == replacement.replacement.trial_success
        assert cell["task_ids_sha256"] == replacement.replacement.task_ids_sha256
        assert cell["git_commit"] == replacement.replacement.run_git_commit

    for artifact in (
        manifest.artifacts.prompt_archive,
        manifest.artifacts.experience_analysis,
        manifest.artifacts.latency_analysis,
        manifest.artifacts.naturalness_bootstrap,
        manifest.artifacts.previous_experience_analysis,
        manifest.artifacts.human_validation,
    ):
        assert _sha256(REPO_ROOT / artifact.path) == artifact.sha256

    assert (
        "/private/tmp"
        not in (REPO_ROOT / manifest.artifacts.experience_analysis.path).read_text()
    )

    current_replacements = [
        cell
        for cell in current
        if (cell["cohort"], cell["language"], cell["domain"], cell["system"])
        in replacement_keys
    ]
    unchanged = [
        cell
        for cell in current
        if (cell["cohort"], cell["language"], cell["domain"], cell["system"])
        not in replacement_keys
    ]
    previous = []
    for cell in current:
        key = (
            cell["cohort"],
            cell["language"],
            cell["domain"],
            cell["system"],
        )
        if key not in replacement_keys:
            previous.append(cell)
            continue
        archived = replacement_keys[key].previous
        previous.append(
            {
                **cell,
                "sha256": archived.results_sha256,
                "rows": archived.calls,
            }
        )

    fingerprints = manifest.fingerprints
    assert len(unchanged) == manifest.unchanged_active_cells
    assert _fingerprint(unchanged) == fingerprints.unchanged_cells
    assert _fingerprint(current_replacements) == fingerprints.replacement_cells
    assert _fingerprint(current) == fingerprints.replacement_full_cohort
    assert _fingerprint(previous) == fingerprints.previous_full_cohort
    assert (
        _fingerprint([cell for cell in current if cell["cohort"] == "text"])
        == fingerprints.replacement_text
    )
    assert (
        _fingerprint([cell for cell in previous if cell["cohort"] == "text"])
        == fingerprints.previous_text
    )
    assert (
        _fingerprint(
            [
                cell
                for cell in current
                if cell["cohort"] == "voice"
                and cell["system"] != "xai_provider_default"
            ]
        )
        == fingerprints.replacement_repeated_voice
    )
    assert (
        _fingerprint(
            [
                cell
                for cell in previous
                if cell["cohort"] == "voice"
                and cell["system"] != "xai_provider_default"
            ]
        )
        == fingerprints.previous_repeated_voice
    )
    assert (
        _fingerprint(
            [
                cell
                for cell in current
                if cell["cohort"] == "voice"
                and cell["system"] == "xai_provider_default"
            ]
        )
        == fingerprints.replacement_xai_voice
    )
    assert (
        _fingerprint(
            [
                cell
                for cell in previous
                if cell["cohort"] == "voice"
                and cell["system"] == "xai_provider_default"
            ]
        )
        == fingerprints.previous_xai_voice
    )
