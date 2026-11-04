"""Offline benchmark planning and saved-result validation on the language runtime."""

import json

import pytest

from tau2.data_model.cascaded import CascadedConfigRequest

pytest.importorskip("tau2.multilingual.runtime", reason="PR5 stacks on PR2 runtime")

from tau2.data_model.run_config import AudioNativeConfig
from tau2.data_model.simulation import VoiceRunConfig
from tau2.multilingual.runtime import load_language
from tau2.runner.presets import binding_revision, resolve_configs

# Package name for the conftest `submitter_file` metadata.
PACKAGE_NAME = "fixture_test_2026-10-02"


@pytest.mark.parametrize("provider", ("openai", "livekit"))
def test_resolves_ordinary_configs_with_exact_ids_and_approved_defaults(
    published_frame, tmp_path, provider
):
    template = VoiceRunConfig(
        audio_native_config=AudioNativeConfig(
            provider=provider,
            **(
                {"model": "gpt-realtime-2", "reasoning_effort": "minimal"}
                if provider == "openai"
                else {}
            ),
        )
    )
    plan = resolve_configs(published_frame, template, tmp_path / "results")
    assert not plan.partial and len(plan.configs) == 18
    assert all(type(config) is VoiceRunConfig for config in plan.configs)
    assert len({config.benchmark.condition_fingerprint for config in plan.configs}) == 1
    for config in plan.configs:
        assert config.benchmark.conditions.voice_instruction_prompt_version == (
            "voice-instructions-v2"
        )
        assert "voice_instruction_prompt_version" not in VoiceRunConfig.model_fields
        assert "voice_instruction_version" not in AudioNativeConfig.model_fields
        assert config.communicate_scoring.scorer == "semantic"
        judge = config.communicate_scoring.judge
        assert judge.model == "gpt-5.5"
        assert judge.model_args == {"reasoning_effort": "low"}
        assert config.llm_user == "openai/gpt-5.5"
        assert config.llm_args_user == {"reasoning_effort": "low"}
        assert config.audio_native_config.max_steps_seconds == 2400
        assert config.audio_native_config.voice == (
            "marin" if provider == "openai" else None
        )
        assert len(config.task_ids) == 50 and config.num_tasks is None
        assert config.task_ids[0] == f"{config.domain}-localized-0"
        assert config.language_artifacts.tasks_sha256
        assert config.auto_resume
    assert template.benchmark is None


def test_shipped_frame_matches_current_runtime_and_packs():
    from tau2.runner.presets import load_frame, verify_frame

    frame = load_frame()
    verify_frame(frame)
    for cell in frame.cells:
        assert cell.binding_revision == binding_revision(
            load_language(cell.language, cell.domain, mode="voice")
        )


@pytest.mark.parametrize(
    ("module", "name", "mode"),
    [
        ("tau2.user.guidelines", "GUIDELINE_TEMPLATE_VERSION", "text"),
        ("tau2.user.guidelines", "GUIDELINE_TEMPLATE_VERSION", "voice"),
        ("tau2.backchannel", "BACKCHANNEL_PROMPT_VERSION", "voice"),
    ],
)
def test_binding_revision_tracks_scaffold_versions(monkeypatch, module, name, mode):
    import importlib

    loaded = load_language("en", "airline", mode=mode)
    before = binding_revision(loaded)
    monkeypatch.setattr(importlib.import_module(module), name, "changed")
    assert binding_revision(loaded) != before


def test_text_binding_ignores_voice_only_versions(monkeypatch):
    import tau2.backchannel

    loaded = load_language("en", "airline", mode="text")
    before = binding_revision(loaded)
    monkeypatch.setattr(tau2.backchannel, "BACKCHANNEL_PROMPT_VERSION", "changed")
    assert binding_revision(loaded) == before


def test_language_bound_template_requires_explicit_filter(published_frame, tmp_path):
    template = VoiceRunConfig(language="es", audio_native_config=AudioNativeConfig())
    with pytest.raises(ValueError, match="language-bound template"):
        resolve_configs(published_frame, template, tmp_path)
    # The default language is every template's unbound language.
    english = VoiceRunConfig(language="en", audio_native_config=AudioNativeConfig())
    assert len(resolve_configs(published_frame, english, tmp_path).configs) == 18


def test_null_cascaded_config_uses_each_language_preset(published_frame, tmp_path):
    template = VoiceRunConfig.model_validate(
        {"audio_native_config": {"provider": "livekit", "cascaded_request": None}}
    )
    plan = resolve_configs(published_frame, template, tmp_path)
    assert len(plan.configs) == 18
    assert len({config.benchmark.condition_fingerprint for config in plan.configs}) == 1


@pytest.mark.parametrize("provider", ["openai", "livekit"])
def test_all_languages_preserve_explicit_conditions(
    published_frame, tmp_path, provider
):
    template = VoiceRunConfig(
        llm_user="fixture-customer",
        llm_args_user={"temperature": 0.3},
        audio_native_config=AudioNativeConfig(
            provider=provider,
            max_steps_seconds=90,
            **(
                {"voice": "cedar", "reasoning_effort": "low"}
                if provider == "openai"
                else {
                    "cascaded_request": CascadedConfigRequest(preset="openai-thinking")
                }
            ),
        ),
    )
    before = template.model_dump_json()
    plan = resolve_configs(published_frame, template, tmp_path)
    assert len(plan.configs) == 18
    assert template.model_dump_json() == before
    for config in plan.configs:
        assert config.llm_user == "fixture-customer"
        assert config.llm_args_user == {"temperature": 0.3}
        assert config.audio_native_config.max_steps_seconds == 90
        if provider == "openai":
            assert config.audio_native_config.voice == "cedar"
        else:
            cascade = config.audio_native_config.cascaded_config
            assert cascade.llm.model == "gpt-5.2"
            assert cascade.tts.language == config.language
            assert cascade.stt.language == {"pt": "pt-BR", "zh": "zh-CN"}.get(
                config.language, config.language
            )


def test_livekit_shared_conditions_preserve_real_component_differences(
    published_frame, tmp_path
):
    from tau2.data_model.cascaded import resolve_cascaded_config

    base = resolve_cascaded_config(config=None, preset_name="default", language="en")
    fingerprints = []
    for change in (None, "voice", "model", "inference", "region"):
        cascade = base.model_copy(deep=True)
        if change == "voice":
            cascade.tts.voice_id = "different-voice"
        if change == "model":
            cascade.llm.model = "different-model"
        if change == "inference":
            cascade.tts.language = None
        if change == "region":
            cascade.stt.language = "en-GB"
        config = VoiceRunConfig(
            audio_native_config=AudioNativeConfig(
                provider="livekit",
                cascaded_request=CascadedConfigRequest(config=cascade),
            )
        )
        plan = resolve_configs(
            published_frame, config, tmp_path, languages=["en"], domains=["retail"]
        )
        fingerprints.append(plan.configs[0].benchmark.condition_fingerprint)
        assert plan.configs[0].audio_native_config.cascaded_config == cascade
    assert len(set(fingerprints)) == len(fingerprints)


def test_filtered_execution_keeps_full_denominator_and_rejects_changed_bindings(
    published_frame, tmp_path
):
    template = VoiceRunConfig(audio_native_config=AudioNativeConfig())
    plan = resolve_configs(
        published_frame,
        template,
        tmp_path / "results",
        languages=["en"],
        domains=["retail"],
    )
    assert plan.partial and len(plan.configs) == 1 and len(plan.frame.cells) == 18
    with pytest.raises(ValueError, match="official matrix"):
        resolve_configs(
            published_frame, template, tmp_path / "results", languages=["fr"]
        )
    published_frame.cells[0].binding_revision = "changed"
    with pytest.raises(ValueError, match="Frozen binding"):
        resolve_configs(
            published_frame, template, tmp_path / "results", languages=["es"]
        )


def test_saved_submission_and_controller_cli_use_ordinary_configs(
    published_frame, tmp_path, monkeypatch
):
    import tau2.runner.batch
    from tau2.data_model.semantic_communication import (
        CommunicateScoring,
        SemanticCommunicateSettings,
    )
    from tau2.data_model.simulation import (
        Results,
        RewardInfo,
        SimulationRun,
        TerminationReason,
    )
    from tau2.metrics.benchmark import export_submission
    from tau2.runner.helpers import get_info
    from tau2.runner.presets import ExecutionSettings, execute_plan

    template = VoiceRunConfig(
        audio_native_config=AudioNativeConfig(reasoning_effort="minimal"),
        communicate_scoring=CommunicateScoring(
            judge=SemanticCommunicateSettings(model="fixture", model_args={})
        ),
    )
    root = tmp_path / "matrix"
    plan = resolve_configs(
        published_frame, template, root, languages=["en"], domains=["retail"]
    )
    called = []

    def runner(configs, **kwargs):
        called.append((configs, kwargs))
        return {}

    monkeypatch.setattr(tau2.runner.batch, "run_domains", runner)
    execute_plan(plan, root, ExecutionSettings(workers=3), dry_run=True)
    assert not called and not root.exists()
    execute_plan(plan, root, ExecutionSettings(workers=3), dry_run=False)
    assert len(called) == 1 and called[0][1]["workers"] == 3
    config = plan.configs[0]
    loaded = load_language("en", "retail", mode="voice")
    info = get_info(config)
    context = info.language_tasks[loaded.tasks[0].id]
    sim = SimulationRun(
        id="one",
        task_id=loaded.tasks[0].id,
        trial=0,
        start_time="a",
        end_time="b",
        duration=1,
        termination_reason=TerminationReason.USER_STOP,
        reward_info=RewardInfo(reward=1),
        language_context=context,
    )
    path = root / "en-retail"
    path.mkdir(parents=True, exist_ok=True)
    Results(info=info, tasks=loaded.tasks, simulations=[sim]).save(path, format="dir")
    submission = export_submission(root, published_frame)
    assert (
        submission.summary.scored_trials == 1
        and submission.summary.missing_trials == 899
    )
    assert submission.summary.score is None
    with pytest.raises(ValueError, match="incomplete"):
        export_submission(root, published_frame, require_complete=True)
    result = Results.load(path)
    result.info.user_info.llm = "tampered"
    result.save(path, format="dir")
    with pytest.raises(ValueError, match="Effective saved settings"):
        export_submission(root, published_frame)


@pytest.mark.parametrize(
    "provider, errored", (("openai", False), ("livekit", False), ("openai", True))
)
def test_complete_saved_matrix_submission_auto_detection(
    published_frame, tmp_path, monkeypatch, provider, errored, submitter_file
):
    import argparse

    from tau2.data_model.semantic_communication import (
        CommunicateScoring,
        SemanticCommunicateSettings,
    )
    from tau2.data_model.simulation import (
        Results,
        RewardInfo,
        SimulationRun,
        TerminationReason,
    )
    from tau2.metrics.benchmark import (
        BenchmarkSubmission,
        export_submission,
        load_submission,
    )
    from tau2.runner.helpers import get_info

    template = VoiceRunConfig(
        audio_native_config=AudioNativeConfig(provider=provider),
        communicate_scoring=CommunicateScoring(
            judge=SemanticCommunicateSettings(model="fixture", model_args={})
        ),
    )
    root = tmp_path / "complete"
    plan = resolve_configs(published_frame, template, root)
    for config in plan.configs:
        loaded = load_language(config.language, config.domain, mode="voice")
        info = get_info(config)
        # Infrastructure errors on two en-airline calls (one would have passed)
        # leave the denominator instead of blocking the package.
        failed = errored and (config.language, config.domain) == ("en", "airline")
        sims = [
            SimulationRun(
                id=f"sim-{index}",
                task_id=task.id,
                trial=0,
                start_time="a",
                end_time="b",
                duration=1,
                termination_reason=TerminationReason.INFRASTRUCTURE_ERROR
                if failed and index < 2
                else TerminationReason.USER_STOP,
                reward_info=None
                if failed and index < 2
                else RewardInfo(reward=float(index < 25)),
                language_context=info.language_tasks[task.id],
            )
            for index, task in enumerate(loaded.tasks)
        ]
        Results(info=info, tasks=loaded.tasks, simulations=sims).save(
            root / f"{config.language}-{config.domain}" / "results.json", format="dir"
        )
    submission = export_submission(root, published_frame, require_complete=True)
    if errored:
        summary = submission.summary
        assert summary.complete and summary.error_trials == 2
        assert summary.scored_trials == 898 and summary.expected_trials == 900
        en_airline = next(
            cell
            for cell in summary.cells
            if (cell.language, cell.domain) == ("en", "airline")
        )
        assert en_airline.score == pytest.approx(23 / 48)
        assert summary.score == pytest.approx((17 * 0.5 + 23 / 48) / 18)
    else:
        assert (
            submission.summary.score == 0.5 and submission.summary.scored_trials == 900
        )
    assert (
        BenchmarkSubmission.model_validate_json(submission.model_dump_json())
        == submission
    )
    corrupt = submission.model_dump()
    corrupt["summary"]["score"] = 1
    with pytest.raises(ValueError, match="summary"):
        corrupt_path = tmp_path / "corrupt-submission.json"
        corrupt_path.write_text(json.dumps(corrupt))
        load_submission(corrupt_path)

    from tau2.judges import submission as judge_submission
    from tau2.runner import presets
    from tau2.scripts.leaderboard.cli import add_submit_args

    monkeypatch.setattr(presets, "load_frame", lambda *a, **kw: published_frame)
    monkeypatch.setattr(presets, "verify_frame", lambda *a, **kw: None)
    monkeypatch.setattr(presets, "preset_names", lambda: ["voice"])
    # Reward export only; judged Voice V2 packages: test_judged_submission.py.
    monkeypatch.setattr(judge_submission, "JUDGED_BENCHMARKS", frozenset())
    (root / "benchmark-frame.json").write_text(published_frame.model_dump_json())
    submit_parser = argparse.ArgumentParser()
    add_submit_args(submit_parser)
    prepare = submit_parser.parse_args(
        [
            "prepare",
            str(root),
            "--output",
            str(tmp_path / "submission"),
            "--metadata-file",
            str(submitter_file),
        ]
    )
    prepare.func(prepare)
    exported = tmp_path / "submission" / PACKAGE_NAME
    validate = submit_parser.parse_args(
        ["validate", str(exported), "--results", str(root)]
    )
    validate.func(validate)
    missing_evidence = submit_parser.parse_args(["validate", str(exported)])
    with pytest.raises(ValueError, match="requires --results"):
        missing_evidence.func(missing_evidence)
    payload = json.loads((exported / "benchmark-submission.json").read_text())
    payload["sources"]["en/airline"] = "a" * 64
    (exported / "benchmark-submission.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="original saved result evidence"):
        validate.func(validate)


@pytest.fixture(params=("openai", "livekit"))
def saved_benchmark_cell(published_frame, tmp_path, request):
    """Ordinary saved cell paired with the authoritative installed artifact fixture."""
    from tau2.data_model.simulation import (
        Results,
        RewardInfo,
        SimulationRun,
        TerminationReason,
    )
    from tau2.runner.helpers import get_info

    root = tmp_path / "saved-audit"
    config = resolve_configs(
        published_frame,
        VoiceRunConfig(audio_native_config=AudioNativeConfig(provider=request.param)),
        root,
        languages=["en"],
        domains=["retail"],
    ).configs[0]
    loaded = load_language("en", "retail", mode="voice")
    info = get_info(config)
    task = loaded.tasks[0]
    simulation = SimulationRun(
        id="audit",
        task_id=task.id,
        trial=0,
        start_time="a",
        end_time="b",
        duration=1,
        termination_reason=TerminationReason.USER_STOP,
        reward_info=RewardInfo(reward=1),
        language_context=info.language_tasks[task.id],
    )
    result = Results(info=info, tasks=loaded.tasks, simulations=[simulation])
    path = root / "en-retail"
    result.save(path / "results.json", format="dir")
    return path, result


@pytest.mark.parametrize(
    "defect",
    [
        "missing-info",
        "stale-info",
        "missing-condition",
        "stale-condition",
        "both-stale",
    ],
)
def test_saved_cell_requires_observed_current_shared_prompt_revision(
    published_frame, saved_benchmark_cell, defect
):
    import hashlib

    from tau2.metrics.benchmark import read_task_trials

    path, _ = saved_benchmark_cell
    metadata_path = path / "results.json"
    metadata = json.loads(metadata_path.read_text())
    info = metadata["info"]
    conditions = info["benchmark"]["conditions"]
    field = "voice_instruction_prompt_version"
    if defect == "missing-info":
        info.pop(field, None)
    elif defect == "stale-info":
        info[field] = "voice-instructions-v1"
    elif defect == "missing-condition":
        conditions.pop(field, None)
    elif defect == "stale-condition":
        conditions[field] = "voice-instructions-v1"
    else:
        info[field] = conditions[field] = "voice-instructions-v1"
    # A self-consistent digest is insufficient: the observed and expected
    # shared prompt revisions must also match. Legacy absence is never inferred.
    info["benchmark"]["condition_fingerprint"] = hashlib.sha256(
        json.dumps(conditions, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(
        ValueError,
        match="shared voice prompt revision|voice_instruction_prompt_version",
    ):
        read_task_trials(path, published_frame)


@pytest.mark.parametrize(
    "field", ["complexity", "steps", "agent", "user", "retrieval", "model"]
)
def test_saved_cell_rejects_effective_setting_drift(
    published_frame, saved_benchmark_cell, field
):
    from tau2.metrics.benchmark import read_task_trials

    path, result = saved_benchmark_cell
    if field == "complexity":
        result.info.speech_complexity = "control"
    elif field == "steps":
        result.info.max_steps = 1
    elif field == "agent":
        result.info.agent_info.implementation = "different-agent"
    elif field == "user":
        result.info.user_info.implementation = "different-user"
    elif field == "retrieval":
        result.info.retrieval_config = "unexpected"
    elif field == "model":
        result.info.agent_info.llm = "openai:different-model"
    result.save(path, format="dir")
    with pytest.raises(ValueError, match="Effective saved settings"):
        read_task_trials(path, published_frame)


@pytest.mark.parametrize("language", [None, "es"])
def test_saved_cell_rejects_missing_or_wrong_observed_provider_language(
    published_frame, saved_benchmark_cell, language
):
    from tau2.metrics.benchmark import read_task_trials

    path, result = saved_benchmark_cell
    result.info.audio_native_config.language = language
    result.save(path, format="dir")
    with pytest.raises(ValueError, match="Provider language"):
        read_task_trials(path, published_frame)


def test_saved_task_self_consistent_hash_cannot_replace_frozen_task(
    published_frame, saved_benchmark_cell
):
    import hashlib

    from tau2.metrics.benchmark import read_task_trials

    path, result = saved_benchmark_cell
    task = result.tasks[0]
    task.user_scenario.instructions = "An easier replacement task."
    result.info.language_tasks[task.id].task_sha256 = hashlib.sha256(
        json.dumps(
            task.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    result.simulations[0].language_context = result.info.language_tasks[task.id]
    result.save(path, format="dir")
    with pytest.raises(ValueError, match="authoritative"):
        read_task_trials(path, published_frame)


def test_saved_context_cannot_replace_reviewed_persona(
    published_frame, saved_benchmark_cell
):
    from tau2.metrics.benchmark import read_task_trials

    path, result = saved_benchmark_cell
    result.info.language_tasks[
        result.tasks[0].id
    ].persona.locale = "an unreviewed caller background"
    result.simulations[0].language_context = result.info.language_tasks[
        result.tasks[0].id
    ]
    result.save(path, format="dir")
    with pytest.raises(ValueError, match="authoritative"):
        read_task_trials(path, published_frame)


def test_saved_context_keeps_the_submitters_own_caller_voice(
    published_frame, saved_benchmark_cell
):
    """Own caller voices (TAU2_VOICE_ID_*) change delivery, not the persona."""
    from tau2.metrics.benchmark import read_task_trials

    path, result = saved_benchmark_cell
    result.info.language_tasks[
        result.tasks[0].id
    ].persona.voice.voice_id = "submitters-own-voice"
    result.simulations[0].language_context = result.info.language_tasks[
        result.tasks[0].id
    ]
    result.save(path, format="dir")
    assert read_task_trials(path, published_frame)[0].reward == 1


def test_saved_validation_requires_pinned_assets_but_allows_relocation(
    published_frame, saved_benchmark_cell, tmp_path, monkeypatch
):
    import shutil

    import tau2.utils
    from tau2.metrics.benchmark import read_task_trials

    path, _ = saved_benchmark_cell
    relocated = tmp_path / "relocated"
    shutil.copytree(
        tau2.utils.DATA_DIR / "tau2",
        relocated / "tau2",
    )
    monkeypatch.setattr(tau2.utils, "DATA_DIR", relocated)
    assert read_task_trials(path, published_frame)[0].reward == 1
    monkeypatch.setattr(tau2.utils, "DATA_DIR", tmp_path / "missing-source")
    with pytest.raises(ValueError, match="authoritative.*TAU2_DATA_DIR"):
        read_task_trials(path, published_frame)


@pytest.mark.parametrize(
    "defect", ["missing-info", "stale-info", "missing-condition", "both-stale"]
)
def test_saved_cell_requires_current_domain_lookup_contract(
    published_frame, saved_benchmark_cell, defect
):
    import hashlib

    from tau2.metrics.benchmark import read_task_trials

    path, _ = saved_benchmark_cell
    metadata_path = path / "results.json"
    payload = json.loads(metadata_path.read_text())
    info = payload["info"]
    conditions = info["benchmark"]["conditions"]
    if defect == "missing-info":
        info.pop("domain_tools_version")
    elif defect == "stale-info":
        info["domain_tools_version"] = "exact-v1"
    elif defect == "missing-condition":
        conditions.pop("domain_tools_version")
    else:
        info["domain_tools_version"] = conditions["domain_tools_version"] = "exact-v1"
    info["benchmark"]["condition_fingerprint"] = hashlib.sha256(
        json.dumps(conditions, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    metadata_path.write_text(json.dumps(payload))
    with pytest.raises(
        ValueError,
        match="domain tools revision|domain_tools_version|condition fingerprint",
    ):
        read_task_trials(path, published_frame)


def test_legacy_results_remain_readable_without_domain_tools_revision(
    saved_benchmark_cell,
):
    from tau2.data_model.simulation import Results

    path, _ = saved_benchmark_cell
    payload = json.loads((path / "results.json").read_text())
    payload["info"].pop("domain_tools_version")
    payload["info"]["benchmark"]["conditions"].pop("domain_tools_version")
    result = Results.model_validate(payload)
    assert result.info.domain_tools_version is None
    assert result.info.benchmark.conditions.domain_tools_version is None
