"""Recorded matrix plans remain trustworthy when previewed and resumed."""

import shutil

import pytest

from tau2.data_model.benchmark import BenchmarkPlan
from tau2.data_model.run_config import AudioNativeConfig
from tau2.data_model.simulation import VoiceRunConfig
from tau2.runner import presets
from tau2.runner.presets import BenchmarkRequest, resolve_configs


def sample(frame, root, *, languages=None, task_ids=None):
    return resolve_configs(
        frame,
        VoiceRunConfig(
            task_ids=task_ids or ["0"], audio_native_config=AudioNativeConfig()
        ),
        root,
        domains=["retail"],
        languages=languages or ["en"],
    )


def write_plan(root, plan):
    root.mkdir()
    (root / "benchmark-frame.json").write_text(plan.frame.model_dump_json())
    (root / "benchmark-plan.json").write_text(plan.model_dump_json())


def test_plan_rejects_duplicate_config_with_another_selected_cell_missing(
    published_frame, tmp_path
):
    plan = sample(published_frame, tmp_path / "matrix", languages=["en", "pt"])
    payload = plan.model_dump()
    payload["configs"][1] = payload["configs"][0]
    with pytest.raises(ValueError):
        BenchmarkPlan.model_validate(payload)


@pytest.mark.parametrize("change", ["user_model", "fingerprint", "conditions"])
def test_plan_rejects_changed_effective_settings_before_execution(
    published_frame, tmp_path, change
):
    payload = sample(published_frame, tmp_path / "matrix").model_dump()
    config = payload["configs"][0]
    if change == "user_model":
        config["llm_user"] = "different-caller"
    elif change == "fingerprint":
        config["benchmark"]["condition_fingerprint"] = "f" * 64
    else:
        config["benchmark"]["conditions"]["llm_user"] = "different-caller"
    with pytest.raises(ValueError):
        BenchmarkPlan.model_validate(payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("language", "pt"),
        ("domain", "airline"),
        ("frame_revision", "other"),
        ("binding_revision", "other"),
    ],
)
def test_plan_rejects_mismatched_recorded_cell_identity(
    published_frame, tmp_path, field, value
):
    payload = sample(published_frame, tmp_path / "matrix").model_dump()
    payload["configs"][0]["benchmark"][field] = value
    with pytest.raises(ValueError):
        BenchmarkPlan.model_validate(payload)


def test_resume_from_copied_matrix_rebinds_every_cell_to_new_root_without_writes(
    published_frame, tmp_path
):
    original = tmp_path / "old"
    copied = tmp_path / "new"
    write_plan(original, sample(published_frame, original))
    shutil.copytree(original, copied)
    before = {
        str(path): path.read_bytes()
        for root in (original, copied)
        for path in root.rglob("*")
        if path.is_file()
    }
    requested = sample(published_frame, copied, languages=["pt"])
    prepared = presets.prepare_plan(requested, copied)
    assert [config.language for config in prepared.configs] == ["en", "pt"]
    assert all(
        config.save_to == str(copied / f"{config.language}-retail")
        for config in prepared.configs
    )
    after = {
        str(path): path.read_bytes()
        for root in (original, copied)
        for path in root.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_public_resolver_returns_cumulative_scope_for_preview(
    published_frame, tmp_path, monkeypatch
):
    root = tmp_path / "matrix"
    write_plan(root, sample(published_frame, root, task_ids=["0"]))
    monkeypatch.setattr(presets, "load_frame", lambda *a, **kw: published_frame)
    monkeypatch.setattr(presets, "verify_frame", lambda *a, **kw: None)
    before = (root / "benchmark-plan.json").read_bytes()
    resolved = presets.resolve_benchmark(
        BenchmarkRequest(
            output=root,
            template=VoiceRunConfig(
                task_ids=["1"], audio_native_config=AudioNativeConfig()
            ),
            languages=["en", "pt"],
            domains=["retail"],
        )
    )
    assert sum(len(config.task_ids) for config in resolved.configs) == 3
    assert resolved.configs[0].benchmark.selected_task_ids == ["0", "1"]
    assert resolved.configs[1].benchmark.selected_task_ids == ["1"]
    assert (root / "benchmark-plan.json").read_bytes() == before


def test_mismatched_canonical_to_local_mapping_fails_before_writes(
    published_frame, tmp_path
):
    root = tmp_path / "matrix"
    payload = sample(published_frame, root).model_dump()
    payload["configs"][0]["task_ids"] = ["retail-localized-9"]
    with pytest.raises(ValueError):
        presets.prepare_plan(BenchmarkPlan.model_validate(payload), root)
    assert not root.exists()


def test_serialized_benchmark_plan_cannot_opt_out_of_regional_scenarios(
    published_frame, tmp_path
):
    import hashlib
    import json

    payload = sample(published_frame, tmp_path / "matrix").model_dump(mode="json")
    config = payload["configs"][0]
    config["localize_db"] = False
    config["benchmark"]["condition_fingerprint"] = hashlib.sha256(
        json.dumps(
            config["benchmark"]["conditions"], sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="require regional"):
        BenchmarkPlan.model_validate(payload)
