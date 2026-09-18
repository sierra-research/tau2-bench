# Copyright Sierra
"""Corrected-cohort adapter for the frozen τ-Multilingual naturalness judge."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from tau2.data_model.voice import SpeechEnvironment
from tau2.judges.nativeness.corrected_paper_promotion import (
    NaturalnessPromotionConfig,
    NaturalnessPromotionReport,
    _build_code_migration_bridge,
    _build_validation_migration_bridge,
    _validate_paid_judge_specs,
    promote_replacement_only_trial0_naturalness,
)
from tau2.judges.nativeness.corrected_paper_rebind import (
    VALIDATION_DROPPED_FIELDS,
    NaturalnessActiveInventory,
    NaturalnessRebindConfig,
    NaturalnessRebindContract,
    RebindInputReference,
    ValidationEvidenceProjectionConfig,
    ValidationEvidenceProjectionCounts,
    ValidationEvidenceProjectionProof,
    rebind_promoted_trial0_naturalness,
)
from tau2.judges.nativeness.corrected_paper_trial import (
    CORRECTED_COHORT_SCHEMA_VERSION,
    CorrectedCohortContract,
    CorrectedTrialRunConfig,
    ReplacementOnlyTrialRunConfig,
    _validate_production_replacement_root,
    build_corrected_cohort_lock,
    build_replacement_only_cohort_lock,
    corrected_manifest_fingerprint,
    prepare_corrected_trial0_naturalness,
    prepare_replacement_only_trial0_naturalness,
    run_corrected_trial0_naturalness,
    run_replacement_only_trial0_naturalness,
    transcript_sha256,
)
from tau2.judges.nativeness.harness import build_agent_turns
from tau2.judges.nativeness.paper_trial import (
    CANONICAL_VALIDATION_PATH,
    CohortContract,
    TrialCodeProvenance,
    TrialRunConfig,
    ValidationEvidenceIdentity,
    _sha256_value,
    frozen_judge_spec,
    run_trial0_naturalness,
)
from tau2.judges.nativeness.validation import sha256_file
from test_judges.conftest import hi_factor_checks, hi_sim, make_hi_results
from test_judges.test_paper_naturalness import _passing_check


def _experience_fingerprint(sources: list[dict]) -> str:
    payload = "\n".join(
        f"{row['path']}\t{row['sha256']}\t{row['trial_0_calls']}" for row in sources
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _write_validation_bridge_fixture(tmp_path: Path):
    languages = ("es", "pt", "hi", "ko", "zh")
    root = tmp_path / "validation-repo" / CANONICAL_VALIDATION_PATH
    legacy_files: dict[str, str] = {}
    current_files: dict[str, str] = {}
    for index, language in enumerate(languages):
        hashes = {
            "dev_dataset": f"{index * 5 + 1:064x}",
            "dev_result": f"{index * 5 + 2:064x}",
            "prompt": hashlib.sha256(f"{language}-prompt".encode()).hexdigest(),
            "test_dataset": f"{index * 5 + 4:064x}",
            "test_result": f"{index * 5 + 5:064x}",
        }
        legacy_files.update(
            {
                f"{language}/dev/dataset.json": hashes["dev_dataset"],
                f"{language}/dev/results.json": hashes["dev_result"],
                f"{language}/prompt.json": hashes["prompt"],
                f"{language}/test/dataset.json": hashes["test_dataset"],
                f"{language}/test/results.json": hashes["test_result"],
            }
        )
        prompt_relative = f"prompt_sources/naturalness/{language}.json"
        prompt_path = root / prompt_relative
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(f"{language}-prompt")
        current_files[prompt_relative] = sha256_file(prompt_path)
        provenance_relative = f"provenance/utterance/naturalness/{language}.json"
        provenance_path = root / provenance_relative
        provenance_path.parent.mkdir(parents=True, exist_ok=True)
        provenance_path.write_text(
            json.dumps(
                {
                    "schema_version": "tau-multi-validation-provenance-v1",
                    "language": language,
                    "measure_id": "naturalness",
                    "evaluation_level": "utterance",
                    "source_revision": "fixture-consensus",
                    "source_files": [
                        {
                            "role": "naturalness_validation_results",
                            "source_path": f"sources/{language}/results.json",
                            "sha256": hashes["dev_result"],
                            "revision": "naturalness-v16-rubric-v20",
                        },
                        {
                            "role": "naturalness_validation_results",
                            "source_path": (
                                f"sources/{language}/naturalness_validation_results.json"
                            ),
                            "sha256": hashes["test_result"],
                            "revision": "naturalness-v16-rubric-v20",
                        },
                    ],
                },
                indent=2,
            )
        )
        current_files[provenance_relative] = sha256_file(provenance_path)
    legacy = ValidationEvidenceIdentity(
        path=(
            "data/simulations/paper_runs/tau-multi/validations/utterance_naturalness"
        ),
        manifest_sha256="a" * 64,
        files=legacy_files,
    )
    current = ValidationEvidenceIdentity(
        path=CANONICAL_VALIDATION_PATH.as_posix(),
        manifest_sha256="b" * 64,
        files=current_files,
    )
    return root.parents[5], legacy, current


def test_validation_migration_bridge_pins_exact_legacy_inputs(tmp_path):
    validation_repo, legacy, current = _write_validation_bridge_fixture(tmp_path)

    bridge = _build_validation_migration_bridge(
        validation_repo_root=validation_repo,
        bootstrap_evidence=legacy,
        replacement_evidence=current,
        current_evidence=current,
        bootstrap_manifest_sha256="c" * 64,
        bootstrap_identity_sha256="d" * 64,
        replacement_manifest_sha256="e" * 64,
        replacement_identity_sha256="f" * 64,
    )

    assert bridge is not None
    assert bridge.legacy_evidence == legacy
    assert bridge.current_evidence == current
    assert len(bridge.languages) == 5
    assert {row.language for row in bridge.languages} == {
        "es",
        "pt",
        "hi",
        "ko",
        "zh",
    }


def test_validation_migration_bridge_rejects_tampered_result_provenance(tmp_path):
    validation_repo, legacy, current = _write_validation_bridge_fixture(tmp_path)
    provenance_path = (
        validation_repo
        / CANONICAL_VALIDATION_PATH
        / "provenance/utterance/naturalness/ko.json"
    )
    payload = json.loads(provenance_path.read_text())
    payload["source_files"][1]["sha256"] = "9" * 64
    provenance_path.write_text(json.dumps(payload, indent=2))
    current = current.model_copy(
        update={
            "files": {
                **current.files,
                "provenance/utterance/naturalness/ko.json": sha256_file(
                    provenance_path
                ),
            }
        }
    )

    with pytest.raises(ValueError, match="legacy dev/test result hashes"):
        _build_validation_migration_bridge(
            validation_repo_root=validation_repo,
            bootstrap_evidence=legacy,
            replacement_evidence=current,
            current_evidence=current,
            bootstrap_manifest_sha256="c" * 64,
            bootstrap_identity_sha256="d" * 64,
            replacement_manifest_sha256="e" * 64,
            replacement_identity_sha256="f" * 64,
        )


def test_validation_migration_bridge_rejects_tampered_prompt(tmp_path):
    validation_repo, legacy, current = _write_validation_bridge_fixture(tmp_path)
    prompt_path = (
        validation_repo
        / CANONICAL_VALIDATION_PATH
        / "prompt_sources/naturalness/zh.json"
    )
    prompt_path.write_text("tampered")

    with pytest.raises(ValueError, match="prompt does not pin legacy hash"):
        _build_validation_migration_bridge(
            validation_repo_root=validation_repo,
            bootstrap_evidence=legacy,
            replacement_evidence=current,
            current_evidence=current,
            bootstrap_manifest_sha256="c" * 64,
            bootstrap_identity_sha256="d" * 64,
            replacement_manifest_sha256="e" * 64,
            replacement_identity_sha256="f" * 64,
        )


def test_validation_migration_bridge_rejects_replacement_archive_drift(tmp_path):
    validation_repo, legacy, current = _write_validation_bridge_fixture(tmp_path)

    with pytest.raises(ValueError, match="replacement validation evidence"):
        _build_validation_migration_bridge(
            validation_repo_root=validation_repo,
            bootstrap_evidence=legacy,
            replacement_evidence=current.model_copy(
                update={"manifest_sha256": "0" * 64}
            ),
            current_evidence=current,
            bootstrap_manifest_sha256="c" * 64,
            bootstrap_identity_sha256="d" * 64,
            replacement_manifest_sha256="e" * 64,
            replacement_identity_sha256="f" * 64,
        )


def _code_provenance(label: str) -> TrialCodeProvenance:
    source_files = {f"code:{label}.py": hashlib.sha256(label.encode()).hexdigest()}
    return TrialCodeProvenance(
        code_root=f"/tmp/{label}",
        data_root=f"/tmp/{label}/data",
        git_commit=hashlib.sha1(label.encode()).hexdigest(),
        git_dirty=False,
        source_fingerprint_sha256=_sha256_value(source_files),
        source_files=source_files,
    )


def test_code_migration_bridge_pins_legacy_replacement_and_current_provenance():
    legacy = _code_provenance("legacy")
    replacement = _code_provenance("current")
    current = replacement.model_copy(
        update={"code_root": "/tmp/current-checkout", "git_commit": "f" * 40}
    )

    bridge = _build_code_migration_bridge(
        bootstrap_invocation_codes=[legacy] * 5,
        bootstrap_runtime_source_fingerprint_sha256=(legacy.source_fingerprint_sha256),
        replacement_code=replacement,
        current_code=current,
        bootstrap_manifest_sha256="a" * 64,
        bootstrap_identity_sha256="b" * 64,
        replacement_manifest_sha256="c" * 64,
        replacement_identity_sha256="d" * 64,
    )

    assert bridge is not None
    assert bridge.legacy_code == legacy
    assert bridge.replacement_code == replacement
    assert bridge.current_code == current


def test_code_migration_bridge_rejects_mixed_legacy_invocations():
    legacy = _code_provenance("legacy")
    current = _code_provenance("current")
    mixed = legacy.model_copy(update={"git_commit": "0" * 40})

    with pytest.raises(ValueError, match="bootstrap invocations disagree"):
        _build_code_migration_bridge(
            bootstrap_invocation_codes=[legacy, legacy, mixed, legacy, legacy],
            bootstrap_runtime_source_fingerprint_sha256=(
                legacy.source_fingerprint_sha256
            ),
            replacement_code=current,
            current_code=current,
            bootstrap_manifest_sha256="a" * 64,
            bootstrap_identity_sha256="b" * 64,
            replacement_manifest_sha256="c" * 64,
            replacement_identity_sha256="d" * 64,
        )


def test_paid_judge_spec_bridge_rejects_changed_spec():
    current = {language: frozen_judge_spec(language) for language in ("ko", "zh")}
    changed = {
        **current,
        "ko": current["ko"].model_copy(update={"prompt_version": "changed"}),
    }

    with pytest.raises(ValueError, match="bootstrap judge identity drifted"):
        _validate_paid_judge_specs(
            current_judges=current,
            bootstrap_judges=changed,
            replacement_judges=current,
            replacement_languages=("ko", "zh"),
        )


def _write_source(
    root: Path,
    relative: str,
    *,
    language: str,
    simulation_id: str,
    task_id: str,
    calls: int = 1,
) -> Path:
    destination = root / relative
    simulations = []
    for index in range(calls):
        simulation = hi_sim(
            simulation_id if index == 0 else f"{simulation_id}-{index}",
            task_id if index == 0 else f"{task_id}-{index}",
            checks=hi_factor_checks(),
            language=language,
        )
        if language != "en":
            simulation = simulation.model_copy(
                update={"speech_environment": SpeechEnvironment(language=language)}
            )
        simulations.append(simulation)
    run = make_hi_results(
        destination.parent.parent,
        simulations,
        name=destination.parent.name,
    )
    assert run / "results.json" == destination
    return run / "results.json"


def _locked_source(
    evidence_root: Path,
    relative: str,
    *,
    language: str,
    domain: str,
    system_slug: str,
    origin: str,
) -> dict:
    results_path = evidence_root / relative
    raw_results = json.loads(results_path.read_text())
    calls = []
    for entry in raw_results["simulation_index"]:
        if entry["trial"] != 0:
            continue
        sim_path = results_path.parent / "simulations" / f"{entry['id']}.json"
        raw = sim_path.read_bytes()
        simulation = hi_sim(
            entry["id"],
            entry["task_id"],
            checks=hi_factor_checks(),
            language=language,
        ).model_validate_json(raw)
        calls.append(
            {
                "simulation_id": simulation.id,
                "task_id": str(simulation.task_id),
                "trial": 0,
                "simulation_sha256": hashlib.sha256(raw).hexdigest(),
                "transcript_sha256": transcript_sha256(build_agent_turns(simulation)),
            }
        )
    return {
        "path": relative,
        "results_sha256": sha256_file(results_path),
        "language": language,
        "domain": domain,
        "system_slug": system_slug,
        "origin": origin,
        "calls": calls,
    }


def _write_fixture(tmp_path: Path, monkeypatch, *, calls_per_root: int = 1):
    """Build a small analogue with equal canonical and corrected slices."""
    from tau2.judges.nativeness import paper_trial

    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    systems = ("s1", "s2")
    old_contract = CohortContract(
        languages=("en", "ko", "zh"),
        domains=("airline", "retail"),
        systems=("S1", "S2"),
        system_slugs=systems,
        trial=0,
        calls_per_root=calls_per_root,
    )
    old_sources = []
    old_paths: dict[tuple[str, str, str], str] = {}
    for language in old_contract.languages:
        for domain in old_contract.domains:
            for system in systems:
                relative = (
                    "data/simulations/paper_runs/tau-multi/main_runs/"
                    f"{language}_{domain}_{system}/{language}_{domain}_{system}/"
                    "results.json"
                )
                path = _write_source(
                    repo,
                    relative,
                    language=language,
                    simulation_id=f"old-{language}-{domain}-{system}",
                    task_id=f"task-{domain}",
                    calls=calls_per_root,
                )
                old_paths[(language, domain, system)] = relative
                old_sources.append(
                    {
                        "path": relative,
                        "sha256": sha256_file(path),
                        "trial_0_calls": calls_per_root,
                    }
                )
    experience = {
        "instrument": "tau-multilingual-utterance-experience",
        "instrument_version": "fixture-v1",
        "cohort": {
            "calls": old_contract.calls,
            "calls_per_system": (
                len(old_contract.languages) * len(old_contract.domains) * calls_per_root
            ),
            "languages": list(old_contract.languages),
            "domains": list(old_contract.domains),
            "systems": list(old_contract.systems),
            "trial": 0,
        },
        "provenance": {
            "cohort_fingerprint_sha256": _experience_fingerprint(old_sources),
            "results_files": old_sources,
        },
    }
    experience_path = repo / "papers/tau-multilingual/reproduction/experience.json"
    experience_path.parent.mkdir(parents=True)
    experience_path.write_text(json.dumps(experience, indent=2))
    monkeypatch.setattr(
        paper_trial,
        "_judge_turn",
        lambda language, turn: _passing_check(language, turn),
    )
    bootstrap = tmp_path / "bootstrap"
    run_trial0_naturalness(
        TrialRunConfig(repo_root=repo, output_root=bootstrap, max_concurrency=2),
        contract=old_contract,
    )

    corrected_contract = CorrectedCohortContract(
        languages=("ko", "zh"),
        domains=("airline", "retail"),
        system_slugs=systems,
        replacement_languages=("ko", "zh"),
        replacement_domain="retail",
        trial=0,
        calls_per_root=calls_per_root,
    )
    locked_sources = []
    for language in corrected_contract.languages:
        for domain in corrected_contract.domains:
            for system in systems:
                replacement = domain == "retail"
                if replacement:
                    relative = (
                        "corrected/"
                        f"{language}_{domain}_{system}/{language}_{domain}_{system}/"
                        "results.json"
                    )
                    _write_source(
                        repo,
                        relative,
                        language=language,
                        simulation_id=f"new-{language}-{domain}-{system}",
                        task_id=f"task-{domain}",
                        calls=calls_per_root,
                    )
                else:
                    relative = old_paths[(language, domain, system)]
                locked_sources.append(
                    _locked_source(
                        repo,
                        relative,
                        language=language,
                        domain=domain,
                        system_slug=system,
                        origin="corrected" if replacement else "canonical",
                    )
                )
    manifest = {
        "schema_version": CORRECTED_COHORT_SCHEMA_VERSION,
        "instrument": "tau-multilingual-corrected-naturalness-cohort",
        "cohort": {
            "calls": corrected_contract.calls,
            "roots": corrected_contract.roots,
            "languages": list(corrected_contract.languages),
            "domains": list(corrected_contract.domains),
            "system_slugs": list(corrected_contract.system_slugs),
            "trial": 0,
            "calls_per_root": calls_per_root,
        },
        "sources": locked_sources,
    }
    manifest["cohort_fingerprint_sha256"] = corrected_manifest_fingerprint(
        manifest["cohort"], manifest["sources"]
    )
    manifest_path = tmp_path / "corrected-experience.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return repo, manifest_path, bootstrap, corrected_contract


def test_corrected_preflight_hard_gates_reuse_and_pending_counts(tmp_path, monkeypatch):
    repo, manifest, bootstrap, contract = _write_fixture(tmp_path, monkeypatch)

    report = prepare_corrected_trial0_naturalness(
        repo_root=repo,
        experience_manifest=manifest,
        evidence_root=repo,
        bootstrap_from=bootstrap,
        contract=contract,
    )

    assert report.counts.roots == 8
    assert report.counts.calls == 8
    assert report.counts.reused_roots == 4
    assert report.counts.reused_calls == 4
    assert report.counts.pending_roots == 4
    assert report.counts.pending_calls == 4
    assert report.counts.pending_utterances == 4
    assert report.pending_calls_by_language_system == {
        "ko": {"s1": 1, "s2": 1},
        "zh": {"s1": 1, "s2": 1},
    }
    assert len(report.bootstrap.call_artifacts) == 4


def test_corrected_lock_builder_stages_closed_hybrid_evidence(tmp_path, monkeypatch):
    repo, source_manifest, bootstrap, contract = _write_fixture(tmp_path, monkeypatch)
    source_payload = json.loads(source_manifest.read_text())
    replacements = [
        repo / row["path"]
        for row in source_payload["sources"]
        if row["origin"] == "corrected"
    ]
    evidence = tmp_path / "staged-evidence"
    manifest = tmp_path / "staged-experience.json"

    built = build_corrected_cohort_lock(
        canonical_experience=(
            repo / "papers/tau-multilingual/reproduction/experience.json"
        ),
        canonical_evidence_root=repo,
        replacement_results=replacements,
        evidence_root=evidence,
        experience_manifest=manifest,
        contract=contract,
    )

    assert built.cohort.roots == 8
    assert len(list(evidence.rglob("results.json"))) == 8
    assert len(list(evidence.rglob("simulations/*.json"))) == 8
    assert all(
        row.path.startswith("data/simulations/paper_runs/tau-multi/main_runs/")
        for row in built.sources
    )
    corrected_row = next(row for row in built.sources if row.origin == "corrected")
    actual_corrected = next(
        path
        for path in replacements
        if path.parent.name
        == f"{corrected_row.language}_{corrected_row.domain}_{corrected_row.system_slug}"
    )
    assert (
        evidence / corrected_row.path
    ).stat().st_ino != actual_corrected.stat().st_ino
    assert (
        build_corrected_cohort_lock(
            canonical_experience=(
                repo / "papers/tau-multilingual/reproduction/experience.json"
            ),
            canonical_evidence_root=repo,
            replacement_results=replacements,
            evidence_root=evidence,
            experience_manifest=manifest,
            contract=contract,
        )
        == built
    )
    report = prepare_corrected_trial0_naturalness(
        repo_root=repo,
        experience_manifest=manifest,
        evidence_root=evidence,
        bootstrap_from=bootstrap,
        contract=contract,
    )
    assert report.counts.reused_calls == 4
    assert report.counts.pending_calls == 4

    with pytest.raises(ValueError, match="replacement source grid"):
        build_corrected_cohort_lock(
            canonical_experience=(
                repo / "papers/tau-multilingual/reproduction/experience.json"
            ),
            canonical_evidence_root=repo,
            replacement_results=replacements[:-1],
            evidence_root=tmp_path / "bad-evidence",
            experience_manifest=tmp_path / "bad-experience.json",
            contract=contract,
        )


def test_production_replacement_paths_reject_smoke_text_and_ablation_roots():
    leaf = "ko_retail_openai_xhigh/results.json"
    for wrapper in (
        "smoke_runs/retail_name_roles_v1_20260917_3provider",
        "multilingual_text_retail_name_roles_v1_korean_retail",
        "retail_identity_ablation_korean_retail",
    ):
        with pytest.raises(ValueError, match="not smoke/text/ablation"):
            _validate_production_replacement_root(
                Path("/tmp") / wrapper / leaf,
                "ko",
                "openai_xhigh",
            )


def test_corrected_preflight_rejects_source_or_bootstrap_identity_drift(
    tmp_path, monkeypatch
):
    repo, manifest, bootstrap, contract = _write_fixture(tmp_path, monkeypatch)
    payload = json.loads(manifest.read_text())
    corrected = next(row for row in payload["sources"] if row["origin"] == "corrected")
    corrected["calls"][0]["transcript_sha256"] = "0" * 64
    payload["cohort_fingerprint_sha256"] = corrected_manifest_fingerprint(
        payload["cohort"], payload["sources"]
    )
    manifest.write_text(json.dumps(payload, indent=2))
    with pytest.raises(ValueError, match="transcript hash mismatch"):
        prepare_corrected_trial0_naturalness(
            repo_root=repo,
            experience_manifest=manifest,
            evidence_root=repo,
            bootstrap_from=bootstrap,
            contract=contract,
        )

    repo, manifest, bootstrap, contract = _write_fixture(
        tmp_path / "second", monkeypatch
    )
    artifact = next(
        path
        for path in (bootstrap / "simulations").rglob("*.json")
        if json.loads(path.read_text())["domain"] == "airline"
    )
    raw = json.loads(artifact.read_text())
    raw["source_simulation_sha256"] = "f" * 64
    artifact.write_text(json.dumps(raw, indent=2))
    with pytest.raises(ValueError, match="bootstrap .* mismatch"):
        prepare_corrected_trial0_naturalness(
            repo_root=repo,
            experience_manifest=manifest,
            evidence_root=repo,
            bootstrap_from=bootstrap,
            contract=contract,
        )


def test_corrected_preflight_rejects_bootstrap_input_digest_drift(
    tmp_path, monkeypatch
):
    repo, manifest, bootstrap, contract = _write_fixture(tmp_path, monkeypatch)
    artifact = next(
        path
        for path in (bootstrap / "simulations").rglob("*.json")
        if json.loads(path.read_text())["domain"] == "airline"
        and json.loads(path.read_text())["utterances"]
    )
    raw = json.loads(artifact.read_text())
    raw["utterances"][0]["input_sha256"] = "0" * 64
    artifact.write_text(json.dumps(raw, indent=2))

    with pytest.raises(ValueError, match="utterance mismatch"):
        prepare_corrected_trial0_naturalness(
            repo_root=repo,
            experience_manifest=manifest,
            evidence_root=repo,
            bootstrap_from=bootstrap,
            contract=contract,
        )


def test_corrected_preflight_rejects_bootstrap_aggregate_drift(tmp_path, monkeypatch):
    repo, manifest, bootstrap, contract = _write_fixture(tmp_path, monkeypatch)
    artifact = next(
        path
        for path in (bootstrap / "simulations").rglob("*.json")
        if json.loads(path.read_text())["domain"] == "airline"
        and json.loads(path.read_text())["utterances"]
    )
    raw = json.loads(artifact.read_text())
    raw["check"]["outcome"] = "fail"
    artifact.write_text(json.dumps(raw, indent=2))

    with pytest.raises(ValueError, match="aggregate mismatch"):
        prepare_corrected_trial0_naturalness(
            repo_root=repo,
            experience_manifest=manifest,
            evidence_root=repo,
            bootstrap_from=bootstrap,
            contract=contract,
        )


def test_corrected_run_judges_only_replacements_and_resumes(tmp_path, monkeypatch):
    from tau2.judges.nativeness import paper_trial

    repo, manifest, bootstrap, contract = _write_fixture(tmp_path, monkeypatch)
    source_bytes = {
        path: path.read_bytes()
        for path in repo.rglob("*.json")
        if "bootstrap" not in path.parts
    }
    calls = {"count": 0}

    def judge(language, turn):
        calls["count"] += 1
        return _passing_check(language, turn)

    monkeypatch.setattr(paper_trial, "_judge_turn", judge)
    out = tmp_path / "corrected-output"
    config = CorrectedTrialRunConfig(
        repo_root=repo,
        experience_manifest=manifest,
        evidence_root=repo,
        bootstrap_from=bootstrap,
        output_root=out,
        max_concurrency=2,
    )
    first = run_corrected_trial0_naturalness(config, contract=contract)
    assert first.status == "complete"
    assert first.counts.reused_calls == 4
    assert first.counts.judged_utterances == 4
    assert first.counts.complete_calls == 8
    assert calls["count"] == 4
    assert len(list((out / "simulations").rglob("*.json"))) == 8
    assert len(list((out / "utterances").rglob("*.json"))) == 4
    assert all(path.read_bytes() == content for path, content in source_bytes.items())

    monkeypatch.setattr(
        paper_trial,
        "_judge_turn",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("completed corrected output must be reused")
        ),
    )
    second = run_corrected_trial0_naturalness(config, contract=contract)
    assert second.invocations[-1].judged_utterances == 0
    assert second.invocations[-1].reused_utterances == 4
    assert len(second.invocations) == 2


def test_replacement_only_lock_preflight_and_resumable_run(tmp_path, monkeypatch):
    from tau2.judges.nativeness import paper_trial

    contract = CorrectedCohortContract(
        languages=("ko", "zh"),
        domains=("retail",),
        system_slugs=("s1", "s2"),
        replacement_languages=("ko", "zh"),
        replacement_domain="retail",
        trial=0,
        calls_per_root=1,
    )
    sources = tmp_path / "sources"
    replacements = []
    for language in contract.languages:
        for system in contract.system_slugs:
            relative = f"{language}_retail_{system}/results.json"
            replacements.append(
                _write_source(
                    sources,
                    relative,
                    language=language,
                    simulation_id=f"new-{language}-{system}",
                    task_id=f"task-{language}-{system}",
                )
            )
    evidence = tmp_path / "replacement-evidence"
    manifest_path = tmp_path / "replacement-manifest.json"
    manifest = build_replacement_only_cohort_lock(
        replacement_results=replacements,
        evidence_root=evidence,
        experience_manifest=manifest_path,
        contract=contract,
    )
    assert manifest.cohort.roots == 4
    assert manifest.cohort.calls == 4
    assert {row.origin for row in manifest.sources} == {"corrected"}
    assert len(list(evidence.rglob("simulations/*.json"))) == 4

    report = prepare_replacement_only_trial0_naturalness(
        repo_root=tmp_path,
        experience_manifest=manifest_path,
        evidence_root=evidence,
        contract=contract,
    )
    assert report.bootstrap is None
    assert report.counts.reused_calls == 0
    assert report.counts.pending_calls == 4

    calls = {"count": 0}

    def judge(language, turn):
        calls["count"] += 1
        return _passing_check(language, turn)

    monkeypatch.setattr(paper_trial, "_judge_turn", judge)
    output = tmp_path / "replacement-output"
    config = ReplacementOnlyTrialRunConfig(
        repo_root=tmp_path,
        experience_manifest=manifest_path,
        evidence_root=evidence,
        output_root=output,
        max_concurrency=2,
    )
    first = run_replacement_only_trial0_naturalness(config, contract=contract)
    assert first.status == "complete"
    assert first.identity.bootstrap is None
    assert first.counts.calls == 4
    assert first.counts.complete_calls == 4
    assert calls["count"] == 4

    monkeypatch.setattr(
        paper_trial,
        "_judge_turn",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("completed replacement output must be reused")
        ),
    )
    second = run_replacement_only_trial0_naturalness(
        config.model_copy(update={"processes": 8, "max_concurrency": 8}),
        contract=contract,
    )
    assert second.invocations[-1].judged_utterances == 0
    assert second.invocations[-1].reused_utterances == 4
    assert second.invocations[-1].processes == 8


def test_replacement_only_output_promotes_without_judging(tmp_path, monkeypatch):
    from tau2.judges.nativeness import paper_trial
    from tau2.judges.nativeness.paper_trial import TrialRunManifest

    repo, hybrid_manifest, bootstrap, hybrid_contract = _write_fixture(
        tmp_path, monkeypatch
    )
    hybrid_payload = json.loads(hybrid_manifest.read_text())
    replacements = [
        repo / row["path"]
        for row in hybrid_payload["sources"]
        if row["origin"] == "corrected"
    ]
    replacement_contract = CorrectedCohortContract(
        languages=hybrid_contract.replacement_languages,
        domains=(hybrid_contract.replacement_domain,),
        system_slugs=hybrid_contract.system_slugs,
        replacement_languages=hybrid_contract.replacement_languages,
        replacement_domain=hybrid_contract.replacement_domain,
        trial=0,
        calls_per_root=hybrid_contract.calls_per_root,
    )
    replacement_evidence = tmp_path / "replacement-evidence"
    replacement_manifest = tmp_path / "replacement-experience.json"
    build_replacement_only_cohort_lock(
        replacement_results=replacements,
        evidence_root=replacement_evidence,
        experience_manifest=replacement_manifest,
        contract=replacement_contract,
    )
    replacement_output = tmp_path / "replacement-output"
    run_replacement_only_trial0_naturalness(
        ReplacementOnlyTrialRunConfig(
            repo_root=repo,
            experience_manifest=replacement_manifest,
            evidence_root=replacement_evidence,
            output_root=replacement_output,
            max_concurrency=2,
        ),
        contract=replacement_contract,
    )
    replacement_source_hashes = {
        path: sha256_file(path.parent / "simulations" / f"{row['id']}.json")
        for path in replacements
        for row in json.loads(path.read_text())["simulation_index"]
        if row["trial"] == 0
    }
    # Model the independent generic-judge composition that occurs after the
    # paid naturalness replay.  It changes full-file identity only; the ordered
    # agent turns consumed by naturalness remain byte-for-byte equivalent.
    for path in replacements:
        row = next(
            row
            for row in json.loads(path.read_text())["simulation_index"]
            if row["trial"] == 0
        )
        simulation_path = path.parent / "simulations" / f"{row['id']}.json"
        payload = json.loads(simulation_path.read_text())
        payload["quality_info"] = {
            "score": None,
            "factor_checks": [],
            "rubric_version": "fixture-quality-v1",
            "metrics_version": "fixture-metrics-v1",
        }
        simulation_path.write_text(json.dumps(payload, indent=2))
    assert all(
        sha256_file(path.parent / "simulations" / f"{row['id']}.json")
        != replacement_source_hashes[path]
        for path in replacements
        for row in [
            next(
                row
                for row in json.loads(path.read_text())["simulation_index"]
                if row["trial"] == 0
            )
        ]
    )

    hybrid_evidence = tmp_path / "hybrid-evidence"
    promoted_hybrid_manifest = tmp_path / "promoted-hybrid-experience.json"
    build_corrected_cohort_lock(
        canonical_experience=(
            repo / "papers/tau-multilingual/reproduction/experience.json"
        ),
        canonical_evidence_root=repo,
        replacement_results=replacements,
        evidence_root=hybrid_evidence,
        experience_manifest=promoted_hybrid_manifest,
        contract=hybrid_contract,
    )
    replacement_bytes = {
        path: path.read_bytes() for path in replacement_output.rglob("*.json")
    }
    bootstrap_bytes = {path: path.read_bytes() for path in bootstrap.rglob("*.json")}
    monkeypatch.setattr(
        paper_trial,
        "_judge_turn",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("promotion must not invoke the judge")
        ),
    )

    output = tmp_path / "promoted-output"
    config = NaturalnessPromotionConfig(
        repo_root=repo,
        experience_manifest=promoted_hybrid_manifest,
        evidence_root=hybrid_evidence,
        bootstrap_experience=(
            repo / "papers/tau-multilingual/reproduction/experience.json"
        ),
        bootstrap_from=bootstrap,
        replacement_from=replacement_output,
        output_root=output,
    )
    report = promote_replacement_only_trial0_naturalness(
        config,
        contract=hybrid_contract,
        replacement_contract=replacement_contract,
    )

    assert report.counts.calls == 8
    assert report.counts.canonical_calls == 4
    assert report.counts.replacement_calls == 4
    assert len(report.artifacts) == 8
    assert sum(row.origin == "replacement" for row in report.artifacts) == 4
    promoted = TrialRunManifest.model_validate_json(
        (output / "manifest.json").read_text()
    )
    assert promoted.status == "complete"
    assert promoted.counts.complete_simulations == 8
    assert promoted.counts.error_simulations == 0
    assert len(promoted.identity.all_sources) == 12
    assert len(promoted.identity.selected_sources) == 8
    assert promoted.identity.experience_sha256 == sha256_file(
        output / "hybrid_experience.json"
    )
    corrected_sources = [
        source
        for source in promoted.identity.selected_sources
        if source.domain == "retail"
    ]
    assert all(
        source.results_path.startswith(
            "data/simulations/paper_runs/tau-multi/main_runs/"
        )
        for source in corrected_sources
    )
    assert all(
        path.read_bytes() == content for path, content in replacement_bytes.items()
    )
    assert all(
        path.read_bytes() == content for path, content in bootstrap_bytes.items()
    )
    assert (
        promote_replacement_only_trial0_naturalness(
            config,
            contract=hybrid_contract,
            replacement_contract=replacement_contract,
        )
        == report
    )

    # Full-file hash drift is allowed only when the semantic judge input is
    # unchanged.  A transcript mutation must still reject the paid artifact.
    changed_results = replacements[0]
    changed_row = next(
        row
        for row in json.loads(changed_results.read_text())["simulation_index"]
        if row["trial"] == 0
    )
    changed_simulation = (
        changed_results.parent / "simulations" / f"{changed_row['id']}.json"
    )
    changed_payload = json.loads(changed_simulation.read_text())
    changed_payload["messages"][0]["content"] += " changed"
    changed_simulation.write_text(json.dumps(changed_payload, indent=2))
    bad_evidence = tmp_path / "bad-hybrid-evidence"
    bad_manifest = tmp_path / "bad-hybrid-experience.json"
    build_corrected_cohort_lock(
        canonical_experience=(
            repo / "papers/tau-multilingual/reproduction/experience.json"
        ),
        canonical_evidence_root=repo,
        replacement_results=replacements,
        evidence_root=bad_evidence,
        experience_manifest=bad_manifest,
        contract=hybrid_contract,
    )
    with pytest.raises(ValueError, match="replacement utterance input"):
        promote_replacement_only_trial0_naturalness(
            config.model_copy(
                update={
                    "experience_manifest": bad_manifest,
                    "evidence_root": bad_evidence,
                    "output_root": tmp_path / "bad-promoted-output",
                }
            ),
            contract=hybrid_contract,
            replacement_contract=replacement_contract,
        )


def test_corrected_cli_is_explicit_and_canonical_cli_is_unchanged():
    from argparse import ArgumentParser

    from tau2.config import DEFAULT_TAU_MULTI_NATURALNESS_CONCURRENCY
    from tau2.judges.cli import add_judges_args

    parser = ArgumentParser()
    add_judges_args(parser)
    canonical = parser.parse_args(
        ["tau-multi-naturalness", "run", "--out", "/tmp/canonical"]
    )
    assert canonical.func.__name__ == "run_tau_multi_naturalness_trial"
    assert canonical.max_concurrency == DEFAULT_TAU_MULTI_NATURALNESS_CONCURRENCY

    lock = parser.parse_args(
        [
            "tau-multi-naturalness",
            "corrected-lock",
            "--canonical-experience",
            "/tmp/canonical-experience.json",
            "--canonical-evidence-root",
            "/tmp/canonical-evidence",
            "--replacement-results",
            "/tmp/ko-retail/results.json",
            "--evidence-root",
            "/tmp/corrected-evidence",
            "--experience-manifest",
            "/tmp/corrected-experience.json",
            "--dry-run",
        ]
    )
    assert lock.func.__name__ == "build_corrected_tau_multi_naturalness_lock"
    assert lock.dry_run is True

    corrected = parser.parse_args(
        [
            "tau-multi-naturalness",
            "corrected-prepare",
            "--experience-manifest",
            "/tmp/experience.json",
            "--evidence-root",
            "/tmp/evidence",
            "--bootstrap-from",
            "/tmp/canonical-output",
        ]
    )
    assert corrected.func.__name__ == "prepare_corrected_tau_multi_naturalness_trial"

    promote = parser.parse_args(
        [
            "tau-multi-naturalness",
            "corrected-promote",
            "--experience-manifest",
            "/tmp/hybrid.json",
            "--evidence-root",
            "/tmp/evidence",
            "--bootstrap-from",
            "/tmp/bootstrap",
            "--bootstrap-experience",
            "/tmp/bootstrap-experience.json",
            "--replacement-from",
            "/tmp/replacement",
            "--validation-repo-root",
            "/tmp/validation-repo",
            "--out",
            "/tmp/promoted",
        ]
    )
    assert promote.func.__name__ == "promote_corrected_tau_multi_naturalness"
    assert promote.validation_repo_root == Path("/tmp/validation-repo")

    rebind = parser.parse_args(
        [
            "tau-multi-naturalness",
            "corrected-rebind",
            "--active-results-root",
            "/tmp/active",
            "--sidecar-from",
            "/tmp/promoted",
            "--validation-repo-root",
            "/tmp/release",
            "--source-validation-evidence-root",
            "/tmp/private-v1",
            "--sanitized-validation-evidence-root",
            "/tmp/release/data/versioned/validations",
            "--out",
            "/tmp/rebound",
        ]
    )
    assert rebind.func.__name__ == "rebind_corrected_tau_multi_naturalness"
    assert rebind.active_results_root == Path("/tmp/active")
    assert rebind.validation_repo_root == Path("/tmp/release")
    assert rebind.source_validation_evidence_root == Path("/tmp/private-v1")
    assert rebind.sanitized_validation_evidence_root == Path(
        "/tmp/release/data/versioned/validations"
    )
    incomplete_rebind = parser.parse_args(
        [
            "tau-multi-naturalness",
            "corrected-rebind",
            "--active-results-root",
            "/tmp/active",
            "--sidecar-from",
            "/tmp/promoted",
            "--source-validation-evidence-root",
            "/tmp/private-v1",
            "--out",
            "/tmp/rebound",
        ]
    )
    with pytest.raises(ValueError, match="required together"):
        incomplete_rebind.func(incomplete_rebind)


def _write_rebind_fixture(
    tmp_path: Path,
    monkeypatch,
    *,
    calls_per_root: int = 1,
    reverse_active_index: bool = False,
):
    """Build a 12-root analogue whose four retail headers need rebinding."""
    repo, hybrid_manifest, bootstrap, hybrid_contract = _write_fixture(
        tmp_path, monkeypatch, calls_per_root=calls_per_root
    )
    hybrid_payload = json.loads(hybrid_manifest.read_text())
    replacements = [
        repo / row["path"]
        for row in hybrid_payload["sources"]
        if row["origin"] == "corrected"
    ]
    replacement_contract = CorrectedCohortContract(
        languages=hybrid_contract.replacement_languages,
        domains=(hybrid_contract.replacement_domain,),
        system_slugs=hybrid_contract.system_slugs,
        replacement_languages=hybrid_contract.replacement_languages,
        replacement_domain=hybrid_contract.replacement_domain,
        trial=0,
        calls_per_root=hybrid_contract.calls_per_root,
    )
    replacement_evidence = tmp_path / "replacement-evidence"
    replacement_manifest = tmp_path / "replacement-experience.json"
    build_replacement_only_cohort_lock(
        replacement_results=replacements,
        evidence_root=replacement_evidence,
        experience_manifest=replacement_manifest,
        contract=replacement_contract,
    )
    replacement_output = tmp_path / "replacement-output"
    run_replacement_only_trial0_naturalness(
        ReplacementOnlyTrialRunConfig(
            repo_root=repo,
            experience_manifest=replacement_manifest,
            evidence_root=replacement_evidence,
            output_root=replacement_output,
            max_concurrency=2,
        ),
        contract=replacement_contract,
    )
    hybrid_evidence = tmp_path / "hybrid-evidence"
    promoted_hybrid_manifest = tmp_path / "promoted-hybrid-experience.json"
    build_corrected_cohort_lock(
        canonical_experience=(
            repo / "papers/tau-multilingual/reproduction/experience.json"
        ),
        canonical_evidence_root=repo,
        replacement_results=replacements,
        evidence_root=hybrid_evidence,
        experience_manifest=promoted_hybrid_manifest,
        contract=hybrid_contract,
    )
    active_root = tmp_path / "active"
    promoted = active_root / "judge_outputs/promoted"
    promote_replacement_only_trial0_naturalness(
        NaturalnessPromotionConfig(
            repo_root=repo,
            experience_manifest=promoted_hybrid_manifest,
            evidence_root=hybrid_evidence,
            bootstrap_experience=(
                repo / "papers/tau-multilingual/reproduction/experience.json"
            ),
            bootstrap_from=bootstrap,
            replacement_from=replacement_output,
            output_root=promoted,
        ),
        contract=hybrid_contract,
        replacement_contract=replacement_contract,
    )

    from tau2.judges.nativeness.paper_trial import TrialRunManifest

    promoted_manifest = TrialRunManifest.model_validate_json(
        (promoted / "manifest.json").read_text()
    )
    affected = {
        source.results_path
        for source in promoted_manifest.identity.selected_sources
        if source.domain == "retail"
    }
    current_hashes: dict[str, str] = {}
    for logical in affected:
        source = hybrid_evidence / logical
        destination = active_root / logical
        destination.parent.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source.parent, destination.parent)
        destination.chmod(0o644)
        if reverse_active_index:
            payload = json.loads(destination.read_text())
            payload["simulation_index"] = list(reversed(payload["simulation_index"]))
            destination.write_text(json.dumps(payload, indent=2) + "\n")
        else:
            destination.write_bytes(destination.read_bytes() + b"\n")
        current_hashes[logical] = sha256_file(destination)
    current_sources = tuple(
        source.model_copy(update={"sha256": current_hashes[source.path]})
        if source.path in current_hashes
        else source
        for source in promoted_manifest.identity.all_sources
    )
    audit_path = repo / "papers/tau-multilingual/reproduction/active-audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text('{"fixture":"active-audit"}\n')
    receipt_path = repo / "papers/tau-multilingual/reproduction/replacement.json"
    receipt_path.write_text('{"fixture":"replacement"}\n')
    inventory = NaturalnessActiveInventory(
        audit=RebindInputReference(
            path=audit_path.relative_to(repo).as_posix(),
            sha256=sha256_file(audit_path),
            identity="audit",
        ),
        replacement=RebindInputReference(
            path=receipt_path.relative_to(repo).as_posix(),
            sha256=sha256_file(receipt_path),
            identity="replacement",
        ),
        sources=current_sources,
        affected_paths=tuple(sorted(affected)),
    )
    contract = NaturalnessRebindContract(
        languages=("en", "ko", "zh"),
        domains=("airline", "retail"),
        system_slugs=("s1", "s2"),
        affected_languages=("ko", "zh"),
        affected_domain="retail",
        calls_per_root=calls_per_root,
    )
    output = tmp_path / "rebound"
    config = NaturalnessRebindConfig(
        repo_root=repo,
        active_audit=audit_path,
        replacement_manifest=receipt_path,
        active_results_root=active_root,
        sidecar_from=promoted,
        output_root=output,
    )
    return config, contract, inventory, promoted_manifest


def test_promoted_sidecar_rebinds_only_exact_current_headers(tmp_path, monkeypatch):
    from tau2.judges.nativeness import paper_trial
    from tau2.judges.nativeness.paper_trial import TrialRunManifest

    config, contract, inventory, before = _write_rebind_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        paper_trial,
        "_judge_turn",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("canonical rebind must never invoke the judge")
        ),
    )
    report = rebind_promoted_trial0_naturalness(
        config, contract=contract, inventory=inventory
    )

    assert report.counts.all_roots == 12
    assert report.counts.selected_roots == 8
    assert report.counts.calls == 8
    assert report.counts.rebound_roots == 4
    assert report.counts.rebound_calls == 4
    assert report.counts.copied_calls == 4
    assert len(report.sources) == 4
    assert sum(row.operation == "rebound" for row in report.artifacts) == 4
    assert all(
        row.input_sha256 == row.output_sha256
        for row in report.artifacts
        if row.operation == "copied"
    )
    assert all(
        row.input_sha256 != row.output_sha256
        for row in report.artifacts
        if row.operation == "rebound"
    )
    receipt_text = (config.output_root / "rebind.json").read_text()
    assert "/private/" not in receipt_text
    assert "/Users/" not in receipt_text
    assert report.source_sidecar.path == "judge_outputs/promoted"
    assert report.active_audit.path.startswith("papers/")
    assert report.replacement_manifest.path.startswith("papers/")
    after = TrialRunManifest.model_validate_json(
        (config.output_root / "manifest.json").read_text()
    )
    assert after.aggregate == before.aggregate
    assert after.counts == before.counts
    assert after.identity.work_fingerprint_sha256 != (
        before.identity.work_fingerprint_sha256
    )
    assert {
        source.results_path: source.results_sha256
        for source in after.identity.selected_sources
    } == {
        source.path: source.sha256
        for source in inventory.sources
        if source.path in {row.results_path for row in before.identity.selected_sources}
    }
    assert (
        rebind_promoted_trial0_naturalness(
            config, contract=contract, inventory=inventory
        )
        == report
    )


def test_rebind_replaces_inherited_validation_identity_with_proven_projection(
    tmp_path, monkeypatch
):
    from tau2.judges.nativeness import corrected_paper_rebind
    from tau2.judges.nativeness.paper_trial import TrialRunManifest

    config, contract, inventory, _before = _write_rebind_fixture(tmp_path, monkeypatch)
    prior = ValidationEvidenceIdentity(
        path="historical/human_annotations/validations",
        manifest_sha256="1" * 64,
        files={"private.csv": "2" * 64},
    )
    sanitized = ValidationEvidenceIdentity(
        path="data/versioned/human_annotations/validations",
        manifest_sha256="3" * 64,
        files={"public.csv": "4" * 64},
    )
    source_manifest_path = config.sidecar_from / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    source_manifest["identity"]["validation_evidence"] = prior.model_dump(mode="json")
    source_manifest["identity_sha256"] = _sha256_value(source_manifest["identity"])
    source_manifest_path.write_text(json.dumps(source_manifest, indent=2) + "\n")
    source_promotion_path = config.sidecar_from / "promotion.json"
    source_promotion = json.loads(source_promotion_path.read_text())
    source_promotion["output_manifest_sha256"] = sha256_file(source_manifest_path)
    source_promotion["output_identity_sha256"] = source_manifest["identity_sha256"]
    source_promotion_path.write_text(json.dumps(source_promotion, indent=2) + "\n")

    proof = ValidationEvidenceProjectionProof(
        schema_version="tau-multi-naturalness-validation-evidence-projection-v1",
        prior_evidence=prior,
        sanitized_evidence=sanitized,
        dropped_fields=VALIDATION_DROPPED_FIELDS,
        source_inventory_sha256="5" * 64,
        sanitized_inventory_sha256="6" * 64,
        source_projection_sha256="7" * 64,
        sanitized_projection_sha256="7" * 64,
        invariant_inventory_sha256="8" * 64,
        removed_provenance_inventory_sha256="9" * 64,
        counts=ValidationEvidenceProjectionCounts(
            partitions=1,
            rows=1,
            metric_rows=1,
            prompt_source_files=1,
            invariant_files=4,
            removed_provenance_files=1,
            source_files=8,
            sanitized_files=7,
        ),
    )

    def prove(_config, *, prior_evidence, contract):
        assert prior_evidence == prior
        return proof

    monkeypatch.setattr(
        corrected_paper_rebind, "prove_sanitized_validation_evidence", prove
    )
    projection_config = ValidationEvidenceProjectionConfig(
        validation_repo_root=tmp_path / "release",
        source_root=tmp_path / "private-v1",
        sanitized_root=tmp_path / "release/data/versioned",
    )
    config = config.model_copy(
        update={"validation_evidence_projection": projection_config}
    )

    report = rebind_promoted_trial0_naturalness(
        config, contract=contract, inventory=inventory
    )

    output_manifest = TrialRunManifest.model_validate_json(
        (config.output_root / "manifest.json").read_text()
    )
    assert output_manifest.identity.validation_evidence == sanitized
    assert report.validation_evidence_projection == proof
    receipt = (config.output_root / "rebind.json").read_text()
    assert "historical/human_annotations/validations" in receipt
    assert "data/versioned/human_annotations/validations" in receipt
    assert str(tmp_path) not in receipt


def test_rebind_work_fingerprint_uses_current_affected_index_order(
    tmp_path, monkeypatch
):
    from tau2.judges.nativeness.paper_trial import (
        TrialRunManifest,
        TrialSimulationArtifact,
    )

    config, contract, inventory, before = _write_rebind_fixture(
        tmp_path,
        monkeypatch,
        calls_per_root=2,
        reverse_active_index=True,
    )
    report = rebind_promoted_trial0_naturalness(
        config, contract=contract, inventory=inventory
    )
    after = TrialRunManifest.model_validate_json(
        (config.output_root / "manifest.json").read_text()
    )
    assert report.counts.calls == 16
    assert report.counts.rebound_calls == 8

    output_artifacts = [
        TrialSimulationArtifact.model_validate_json(path.read_text())
        for path in (config.output_root / "simulations").rglob("*.json")
    ]
    by_source_and_id = {
        (row.source.results_path, row.simulation_id): row for row in output_artifacts
    }
    source_promotion = NaturalnessPromotionReport.model_validate_json(
        (config.sidecar_from / "promotion.json").read_text()
    )
    old_ids_by_cell: dict[tuple[str, str, str], list[str]] = {}
    for row in source_promotion.artifacts:
        old_ids_by_cell.setdefault(
            (row.language, row.domain, row.system_slug), []
        ).append(row.simulation_id)

    ordered: list[TrialSimulationArtifact] = []
    affected = set(inventory.affected_paths)
    for source in after.identity.selected_sources:
        if source.results_path in affected:
            results = json.loads(
                (config.active_results_root / source.results_path).read_text()
            )
            simulation_ids = [
                row["id"] for row in results["simulation_index"] if row["trial"] == 0
            ]
        else:
            simulation_ids = old_ids_by_cell[
                (source.language, source.domain, source.system_slug)
            ]
        ordered.extend(
            by_source_and_id[(source.results_path, simulation_id)]
            for simulation_id in simulation_ids
        )
    reconstructed = _sha256_value(
        {
            "simulations": [
                (
                    row.source.results_path,
                    row.simulation_id,
                    row.source_simulation_sha256,
                )
                for row in ordered
            ],
            "utterances": [
                utterance.input_sha256
                for row in ordered
                for utterance in row.utterances
            ],
        }
    )
    assert len(ordered) == 16
    assert after.identity.work_fingerprint_sha256 == reconstructed
    assert after.identity.work_fingerprint_sha256 != (
        before.identity.work_fingerprint_sha256
    )


def test_promoted_sidecar_rebind_rejects_simulation_or_unaffected_hash_drift(
    tmp_path, monkeypatch
):
    config, contract, inventory, _before = _write_rebind_fixture(tmp_path, monkeypatch)
    simulation = next(config.active_results_root.rglob("simulations/*.json"))
    simulation.chmod(0o644)
    simulation.write_bytes(simulation.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="simulation bytes differ"):
        rebind_promoted_trial0_naturalness(
            config, contract=contract, inventory=inventory
        )
    assert not config.output_root.exists()

    config, contract, inventory, before = _write_rebind_fixture(
        tmp_path / "second", monkeypatch
    )
    selected = {row.results_path for row in before.identity.selected_sources}
    unaffected = next(
        row
        for row in inventory.sources
        if row.path in selected and row.path not in inventory.affected_paths
    )
    drifted_sources = tuple(
        row.model_copy(update={"sha256": "f" * 64})
        if row.path == unaffected.path
        else row
        for row in inventory.sources
    )
    drifted = inventory.model_copy(update={"sources": drifted_sources})
    with pytest.raises(ValueError, match="differences do not equal"):
        rebind_promoted_trial0_naturalness(config, contract=contract, inventory=drifted)
    assert not config.output_root.exists()
