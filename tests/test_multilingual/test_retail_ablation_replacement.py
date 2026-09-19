import hashlib
import json
from pathlib import Path

from tau2.paper.retail_replacement import (
    load_retail_ablation_replacement_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = (
    REPO_ROOT
    / "papers/tau-multilingual/reproduction/retail_ablation_name_role_replacement.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_checked_in_ablation_replacement_is_exact_and_portable() -> None:
    manifest = load_retail_ablation_replacement_manifest(MANIFEST_PATH)

    assert manifest.active_replacement_cells == 4
    assert manifest.unchanged_active_ablation_cells == 4
    assert manifest.accepted_calls == 120
    assert manifest.previous_cells_archived
    assert not manifest.old_archive_uploaded
    assert manifest.human_validation_preserved_unchanged
    assert manifest.exclusions.model_dump() == {
        "hallucination_discard_records": 14,
        "infrastructure_error_artifacts": 5,
        "superseded_complete_artifacts": 5,
        "incomplete_retry_artifacts": 66,
        "unindexed_artifact_directories": 90,
        "orphan_simulation_files": 0,
        "included_infrastructure_failures": 0,
    }
    assert all(
        cell.archive_path == f"validation_runs/zh/{cell.canonical_path}"
        for cell in manifest.cells
    )
    assert len({cell.replacement.task_ids_sha256 for cell in manifest.cells}) == 1
    assert {cell.replacement.calls for cell in manifest.cells} == {30}
    assert {cell.replacement.trials for cell in manifest.cells} == {(0,)}
    assert "/private/tmp" not in MANIFEST_PATH.read_text()


def test_ablation_replacement_binds_exact_old_and_new_results() -> None:
    manifest = load_retail_ablation_replacement_manifest(MANIFEST_PATH)
    observed = {
        (cell.condition, cell.system): (
            cell.previous.results_sha256,
            cell.replacement.results_sha256,
            cell.previous.trial_success["0"],
            cell.replacement.trial_success["0"],
        )
        for cell in manifest.cells
    }

    assert observed == {
        ("source_english_identity", "openai_xhigh"): (
            "6470b5253796ac579ab9909bbac2f29f45734134cd43f92c789e726b78de0a16",
            "4557e7d91428ab1728b4278ab4b6b6fc45654f685f2fb049541592fe4a96c822",
            14 / 30,
            6 / 30,
        ),
        ("source_english_identity", "gemini_high"): (
            "a1b18eb65038ba6ba5ba11bba290f0d0215f91c0fc7f90833ec72536fce134a3",
            "737bcbb82e9eaa64c53411d5ee5591060a74fd7c6081533384f6720c323ecd0d",
            16 / 30,
            16 / 30,
        ),
        ("native_script_database", "openai_xhigh"): (
            "d6e97b5acba85f503e26e33f696001b02c3c74106479d96cc6e4f77d7c89925e",
            "3e4d9e41adaa057c197832ee6cbf252866d128207ad8d07c5de77b2e6d18f612",
            14 / 30,
            16 / 30,
        ),
        ("native_script_database", "gemini_high"): (
            "ea8ec2ba35d62df94ff219d5e35a369af84b11aeeea475f328355bd2c91cd028",
            "519e802abc700a48217f01349241c23c9a933f60f1534b8ffe05890396639459",
            18 / 30,
            16 / 30,
        ),
    }


def test_ablation_replacement_binds_prompt_and_transcript_archives() -> None:
    manifest = load_retail_ablation_replacement_manifest(MANIFEST_PATH)
    prompt_path = REPO_ROOT / manifest.prompt_archive.path
    prompt_payload = json.loads(prompt_path.read_text())
    prompt_cells = {
        cell["results_path"]: cell
        for cell in prompt_payload["cells"]
        if cell["results_path"] in {item.canonical_path for item in manifest.cells}
    }

    assert _sha256(prompt_path) == manifest.prompt_archive.sha256
    assert prompt_payload["artifact_id"] == manifest.prompt_archive.artifact_id
    assert len(prompt_payload["cells"]) == manifest.prompt_archive.cells
    assert sum(len(cell["simulations"]) for cell in prompt_payload["cells"]) == (
        manifest.prompt_archive.simulations
    )
    assert len(prompt_payload["objects"]) == manifest.prompt_archive.objects
    assert manifest.prompt_archive.retained_cells == 130
    assert manifest.prompt_archive.replaced_cells == 4
    assert manifest.prompt_archive.previous_objects == 2811
    assert manifest.prompt_archive.objects_added == 66
    assert manifest.prompt_archive.objects_removed == 64
    assert len(prompt_cells) == 4
    for cell in manifest.cells:
        frozen = prompt_cells[cell.canonical_path]
        assert frozen["results_sha256"] == cell.replacement.results_sha256
        assert frozen["prompt_profile_sha256"] == cell.prompt_profile_sha256
        assert len(frozen["simulations"]) == cell.replacement.calls

    source_verification = prompt_path.with_name("source_verification.json")
    assert _sha256(source_verification) == (
        manifest.prompt_archive.source_verification_sha256
    )
    source_payload = json.loads(source_verification.read_text())
    assert source_payload["ok"]
    assert source_payload["problems"] == []
    assert source_payload["checked_cells"] == manifest.prompt_archive.cells
    assert source_payload["checked_simulations"] == manifest.prompt_archive.simulations

    transcript_path = REPO_ROOT / manifest.transcript_archive.path
    transcript_payload = json.loads(transcript_path.read_text())
    assert _sha256(transcript_path) == manifest.transcript_archive.sha256
    assert transcript_payload["rows"] == manifest.transcript_archive.rows
    assert transcript_payload["transcript_sha256"] == (
        manifest.transcript_archive.transcript_sha256
    )
    source_runs = {
        run["results_path"]: run for run in transcript_payload["source_runs"]
    }
    for cell in manifest.cells:
        assert source_runs[cell.canonical_path]["results_sha256"] == (
            cell.replacement.results_sha256
        )
