# Copyright Sierra
"""Sparse corrected-paper analysis repository construction."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import pytest

import tau2.judges.corrected_paper_overlay as overlay
from tau2.data_model.simulation import TaskSubsetInfo
from tau2.judges.cli import add_judges_args
from tau2.judges.corrected_paper_overlay import (
    CorrectedAnalysisOverlayConfig,
    CorrectedAnalysisOverlayManifest,
    OverlayCellIdentity,
    OverlayFileIdentity,
    build_corrected_analysis_overlay,
)

SHA = "a" * 64


def _fixture_plan(
    tmp_path: Path,
) -> tuple[CorrectedAnalysisOverlayConfig, overlay._OverlayPlan]:
    output = tmp_path / "analysis-repo"
    links: list[tuple[Path, Path]] = []
    cells: list[OverlayCellIdentity] = []
    legacy_slugs = sorted(overlay.REPEATED_SYSTEM_SLUGS)
    for index in range(90):
        source = tmp_path / "sources" / f"cell-{index}"
        source.mkdir(parents=True)
        (source / "results.json").write_text(f'{{"cell": {index}}}\n')
        relative = Path("data/simulations/main") / f"cell-{index}"
        links.append((relative, source))
        corrected = index >= 80
        legacy = index < len(legacy_slugs)
        cells.append(
            OverlayCellIdentity(
                language="en" if legacy else f"l{index}",
                domain="telecom" if legacy else "retail",
                system_slug=legacy_slugs[index] if legacy else f"s{index}",
                source_kind=("corrected_composition" if corrected else "canonical"),
                source_cell_path=str(source),
                source_results_sha256=SHA,
                canonical_results_path=str(source / "results.json"),
                canonical_results_sha256=SHA,
                target_cell_relative_path=relative.as_posix(),
                trial_0_calls=50,
                source_trials=(0,) if corrected else (0, 1),
                task_frame_sha256=SHA,
                simulation_ids_sha256=SHA,
                nonjudge_hashes_verified=corrected,
                task_subset_provenance=(
                    "legacy_null_verified_against_pinned_xai" if legacy else "pinned"
                ),
            )
        )
    file_payloads = {
        Path("data/analysis/pt.csv"): b"pt\n",
        Path("data/analysis/hi.csv"): b"hi\n",
        Path("data/analysis/final.csv"): b"final\n",
        Path("data/analysis/final.json"): b"{}\n",
    }
    roles = (
        "pt_gender",
        "hi_gender",
        "hybrid_final_window",
        "hybrid_final_window_manifest",
    )
    files = tuple(
        OverlayFileIdentity(
            role=role,
            source_path=str(tmp_path / f"{role}.source"),
            source_sha256=SHA,
            target_relative_path=relative.as_posix(),
            installed_sha256=overlay._sha256_bytes(payload),
            rows=1,
            transformation=(
                "stable_artifact_name"
                if role == "hybrid_final_window_manifest"
                else "byte_copy"
            ),
        )
        for role, (relative, payload) in zip(roles, file_payloads.items(), strict=True)
    )
    manifest = CorrectedAnalysisOverlayManifest(
        schema_version=overlay.OVERLAY_SCHEMA_VERSION,
        created_at="2026-09-18T00:00:00+00:00",
        canonical_repo_root=str(tmp_path / "canonical"),
        output_repo_root=str(output),
        compose_report_path=str(tmp_path / "compose_report.json"),
        compose_report_sha256=SHA,
        composition_cohort_fingerprint_sha256=SHA,
        cells=tuple(cells),
        files=files,
        cells_total=90,
        canonical_cells=80,
        corrected_cells=10,
        trial_0_calls=4_500,
        corrected_nonjudge_verified_calls=500,
        legacy_null_task_subset_cells=tuple(
            sorted("/".join(key) for key in overlay.LEGACY_NULL_TASK_SUBSET_KEYS)
        ),
        cohort_fingerprint_sha256=SHA,
    )
    plan = overlay._OverlayPlan(
        manifest=manifest,
        links=tuple(links),
        files=tuple(file_payloads.items()),
    )
    config = CorrectedAnalysisOverlayConfig(
        canonical_repo_root=tmp_path / "canonical",
        quality_workspace=tmp_path / "quality",
        delivery_workspace=tmp_path / "delivery",
        modal_workspace=tmp_path / "modal",
        composed_workspace=tmp_path / "composed",
        final_window_sidecar=tmp_path / "hybrid.csv",
        output_repo_root=output,
    )
    return config, plan


def test_overlay_build_is_atomic_idempotent_and_detects_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, plan = _fixture_plan(tmp_path)
    monkeypatch.setattr(overlay, "_plan_overlay", lambda *_args, **_kwargs: plan)

    first = build_corrected_analysis_overlay(config)
    second = build_corrected_analysis_overlay(config)

    assert first == second == plan.manifest
    assert all((config.output_repo_root / path).is_symlink() for path, _ in plan.links)
    target, source = plan.links[-1]
    assert (config.output_repo_root / target).resolve() == source.resolve()

    sidecar = config.output_repo_root / plan.files[0][0]
    sidecar.write_text("drift\n")
    with pytest.raises(ValueError, match="sidecar drifted"):
        build_corrected_analysis_overlay(config)


def test_corrected_cells_require_nonjudge_verification() -> None:
    with pytest.raises(ValueError, match="non-judge hashes"):
        OverlayCellIdentity(
            language="ko",
            domain="retail",
            system_slug="openai_minimal",
            source_kind="corrected_composition",
            source_cell_path="/source",
            source_results_sha256=SHA,
            canonical_results_path="/canonical/results.json",
            canonical_results_sha256=SHA,
            target_cell_relative_path="data/results/cell",
            trial_0_calls=50,
            source_trials=(0,),
            task_frame_sha256=SHA,
            simulation_ids_sha256=SHA,
            nonjudge_hashes_verified=False,
            task_subset_provenance="pinned",
        )


def test_legacy_null_subset_exception_is_exact_path_and_exact_four_keys(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical" / "en_telecom_openai_minimal/results.json"

    assert (
        overlay._task_subset_provenance(
            None,
            path=canonical,
            language="en",
            domain="telecom",
            system_slug="openai_minimal",
            corrected=False,
            canonical_results_path=canonical,
        )
        == "legacy_null_verified_against_pinned_xai"
    )

    rejected = (
        {"system_slug": "xai_provider_default"},
        {"language": "ko"},
        {"domain": "retail"},
        {"corrected": True},
        {"path": tmp_path / "copy" / "results.json"},
    )
    base = {
        "path": canonical,
        "language": "en",
        "domain": "telecom",
        "system_slug": "openai_minimal",
        "corrected": False,
        "canonical_results_path": canonical,
    }
    for update in rejected:
        with pytest.raises(ValueError, match="task-subset metadata drifted"):
            overlay._task_subset_provenance(None, **(base | update))

    pinned = TaskSubsetInfo(
        name="telecom_50",
        size=50,
        tasks_scored=50,
        frame_task_set="telecom",
        frame_size=114,
        frame_digest=SHA,
        strategy="stratified",
        seed=42,
    )
    assert overlay._task_subset_provenance(pinned, **base) == "pinned"
    for update in (
        {"name": "retail_50"},
        {"size": 49},
        {"tasks_scored": 49},
    ):
        malformed = pinned.model_copy(update=update)
        with pytest.raises(ValueError, match="task-subset metadata drifted"):
            overlay._task_subset_provenance(malformed, **base)


@pytest.mark.parametrize(
    "source_kind,language,domain,system_slug",
    [
        ("canonical", "en", "telecom", "xai_provider_default"),
        ("corrected_composition", "en", "telecom", "openai_minimal"),
        ("canonical", "ko", "telecom", "openai_minimal"),
    ],
)
def test_cell_model_rejects_legacy_provenance_outside_closed_exception(
    source_kind: str, language: str, domain: str, system_slug: str
) -> None:
    with pytest.raises(ValueError, match="restricted to the four canonical"):
        OverlayCellIdentity(
            language=language,
            domain=domain,
            system_slug=system_slug,
            source_kind=source_kind,
            source_cell_path="/source",
            source_results_sha256=SHA,
            canonical_results_path="/canonical/results.json",
            canonical_results_sha256=SHA,
            target_cell_relative_path="data/results/cell",
            trial_0_calls=50,
            source_trials=(0,),
            task_frame_sha256=SHA,
            simulation_ids_sha256=SHA,
            nonjudge_hashes_verified=source_kind == "corrected_composition",
            task_subset_provenance="legacy_null_verified_against_pinned_xai",
        )


def test_legacy_null_subset_frame_must_equal_pinned_xai(tmp_path: Path) -> None:
    evidence = overlay._CellEvidence(
        trials=(0, 1),
        trial_0_ids=frozenset(f"sim-{index}" for index in range(50)),
        trial_0_task_ids=frozenset(f"task-{index}" for index in range(50)),
        trial_0_tasks=frozenset(f"task-{index}" for index in range(50)),
        task_frame_sha256=SHA,
        simulation_ids_sha256=SHA,
        task_subset_provenance="legacy_null_verified_against_pinned_xai",
    )

    overlay._validate_legacy_null_frame(
        evidence,
        pinned_xai_task_ids=evidence.trial_0_task_ids,
        path=tmp_path / "results.json",
    )
    with pytest.raises(ValueError, match="differs from pinned xAI"):
        overlay._validate_legacy_null_frame(
            evidence,
            pinned_xai_task_ids=frozenset({"different"}),
            path=tmp_path / "results.json",
        )


def test_overlay_manifest_rejects_any_legacy_exception_inventory_drift(
    tmp_path: Path,
) -> None:
    _config, plan = _fixture_plan(tmp_path)
    payload = plan.manifest.model_dump(mode="json")
    legacy = next(
        cell
        for cell in payload["cells"]
        if cell["task_subset_provenance"] == "legacy_null_verified_against_pinned_xai"
    )
    legacy["task_subset_provenance"] = "pinned"

    with pytest.raises(ValueError, match="legacy null task-subset inventory"):
        CorrectedAnalysisOverlayManifest.model_validate(payload)


def test_final_window_source_identity_combines_csv_and_manifest(tmp_path: Path) -> None:
    canonical_csv = tmp_path / "canonical.csv"
    canonical_manifest = canonical_csv.with_suffix(".json")
    canonical_csv.write_text("sim_id\ncall-1\n")
    canonical_manifest.write_text('{"filter_version":"v1"}\n')

    raw_csv_sha256 = overlay.sha256_file(canonical_csv)
    expected = overlay._sha256_value(
        {
            "csv": raw_csv_sha256,
            "manifest": overlay.sha256_file(canonical_manifest),
        }
    )

    assert overlay._sidecar_identity_sha256(canonical_csv) == expected
    assert overlay._sidecar_identity_sha256(canonical_csv) != raw_csv_sha256

    canonical_manifest.write_text('{"filter_version":"v2"}\n')
    assert overlay._sidecar_identity_sha256(canonical_csv) != expected


def test_corrected_analysis_overlay_cli_is_explicit() -> None:
    parser = ArgumentParser()
    add_judges_args(parser)
    args = parser.parse_args(
        [
            "corrected-paper-suite",
            "analysis-overlay",
            "--canonical-repo-root",
            "/canonical",
            "--quality-workspace",
            "/quality",
            "--delivery-workspace",
            "/delivery",
            "--modal-workspace",
            "/modal",
            "--composed-workspace",
            "/composed",
            "--final-window-sidecar",
            "/hybrid.csv",
            "--output-repo-root",
            "/overlay",
        ]
    )

    assert args.func.__name__ == "build_corrected_analysis_overlay"
    assert args.output_repo_root == Path("/overlay")
