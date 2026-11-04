"""Pre-manifest banking and entity release cells become voice-v2 domain packages.

A synthetic export on the real published frames: each cell records the released
binding, as the 2026-10-02 calls did, but no ``info.benchmark``. No model is
called and nothing is judged.
"""

import argparse
import json
import shutil
from pathlib import Path

import pytest

from tau2.data_model.simulation import RewardInfo, SimulationRun
from tau2.runner.pass3 import CROSSED_CONDITIONS
from tau2.runner.presets import load_cell_language, load_frame
from tau2.scripts.leaderboard.benchmark_package import (
    HISTORICAL_EXPORT_DIRS,
    HISTORICAL_PASS3_DIRS,
    SubmitterMetadata,
    load_package,
)
from tau2.scripts.leaderboard.voice_v2_publication import (
    ACCEPTED_RUNTIME_RULES,
    build_historical_package,
)

SUBMITTER = {
    "model_name": "Offline Fixture 1.0",
    "model_organization": "Test Org",
    "submitting_organization": "Test",
    "submission_date": "2026-10-02",
    "contact_info": {"email": "review@example.org"},
}
NAME = "offline-fixture-1-0_test-org_2026-10-02"  # plus _banking / _entity
DOMAINS = {"banking": "banking_knowledge", "entity": "intake"}


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _cell_results(language: str, domain: str, task_ids: list[str]) -> tuple:
    """A cell's recorded info, task list and index; localized IDs off English."""
    loaded = load_cell_language(language, domain, "voice")
    pins = loaded.artifact_pins.model_dump(mode="json")
    if pins["runtime_rules_sha256"] is not None:
        # Record a verified older digest, as the release's non-English calls do.
        pins["runtime_rules_sha256"] = sorted(ACCEPTED_RUNTIME_RULES[domain])[0]
    local = {
        task_id: task_id if language == "en" else f"{task_id}_{language}"
        for task_id in task_ids
    }
    info = {
        "benchmark": None,
        "git_commit": "0" * 40,
        "seed": 42,
        "language": language,
        "pack_extension": loaded.pack_extension,
        "environment_info": {"domain_name": domain},
        "agent_info": {"llm": "offline-fixture"},
        # Historical provider values the current parser rejects stay strings.
        "audio_native_config": {
            "provider": "openai_live",
            "model": "offline-fixture",
            "language": language,
            "reasoning_effort": "high",
        },
        "user_info": {"llm": "offline-caller"},
        "num_trials": 1,
        "speech_complexity": "regular",
        "channel_effects_mode": "regular",
        "speech_effects_mode": "regular",
        "elicitation_prompt_version": f"prompts-{language}",
        "language_artifacts": pins,
        "language_tasks": {
            local[task_id]: {
                "language": language,
                "mode": "voice",
                "pack_extension": loaded.pack_extension,
                "pack_revision": loaded.pack.revision,
                "domain_revision": loaded.binding.revision,
                **pins,
                "task_selection": {
                    "task_set_name": None,
                    "task_split_name": "base",
                    "task_kwargs": loaded.selection.task_kwargs,
                },
                "canonical_task_id": task_id,
            }
            for task_id in task_ids
        },
    }
    return info, [{"id": local[task_id]} for task_id in task_ids], local


def _pass3_reward(condition: str, position: int) -> float:
    """Regular passes odd positions; heavy channel also fails every third."""
    if condition == "regular":
        return float(position % 2)
    return 0.0 if condition == "channel_heavy" and position % 3 == 0 else 1.0


def _write_export(source: Path, benchmark: str, error_task: str | None) -> None:
    for cell in load_frame(benchmark).cells:
        language, domain = cell.language, cell.domain
        info, tasks, local = _cell_results(language, domain, cell.canonical_task_ids)
        # Entity cells also hold the other two crossed Pass^3 conditions.
        conditions = (
            CROSSED_CONDITIONS if domain == "intake" else CROSSED_CONDITIONS[:1]
        )
        for condition in conditions:
            _write_condition(source, cell, info, tasks, local, condition, error_task)


def _write_condition(source, cell, info, tasks, local, condition, error_task):
    """One condition's results and simulations; the regular one is the frame's."""
    language, domain = cell.language, cell.domain
    regular = condition.label == "regular"
    directory = (
        source
        / "trajectories"
        / language
        / (
            HISTORICAL_EXPORT_DIRS[domain]
            if regular
            else HISTORICAL_PASS3_DIRS[condition.label]
        )
    )
    if not regular:
        info = {
            **info,
            "seed": condition.seed,
            "pass3_condition": condition.label,
            "channel_effects_mode": condition.channel_effects_mode,
            "speech_effects_mode": condition.speech_effects_mode,
        }
    index = []
    for position, task_id in enumerate(cell.canonical_task_ids):
        errored = (language, task_id) == ("es", error_task)
        reward = None if errored else _pass3_reward(condition.label, position)
        termination = "infrastructure_error" if errored else "agent_stop"
        simulation_id = f"{language}-{task_id}"
        simulation = SimulationRun(
            id=simulation_id,
            task_id=local[task_id],
            trial=0,
            timestamp="2026-10-02T00:00:00",
            start_time="2026-10-02T00:00:00",
            end_time="2026-10-02T00:00:01",
            duration=1,
            termination_reason=termination,
            reward_info=None if errored else RewardInfo(reward=reward),
        )
        _write_json(
            directory / "simulations" / f"{simulation_id}.json",
            simulation.model_dump(mode="json"),
        )
        index.append(
            {
                "id": simulation_id,
                "task_id": local[task_id],
                "trial": 0,
                "reward": reward,
                "termination_reason": termination,
            }
        )
    _write_json(
        directory / "results.json",
        {"info": info, "tasks": tasks, "simulation_index": index},
    )


@pytest.fixture(scope="module", params=sorted(DOMAINS))
def release_export(request, tmp_path_factory):
    """One provider's export of a domain frame, plus its submitter metadata."""
    root = tmp_path_factory.mktemp(request.param)
    source = root / "export"
    _write_export(source, request.param, error_task=None)
    metadata = root / "metadata.json"
    _write_json(metadata, SUBMITTER)
    return request.param, source, metadata


def _submit(*argv):
    from tau2.scripts.leaderboard.cli import add_submit_args

    parser = argparse.ArgumentParser()
    add_submit_args(parser)
    args = parser.parse_args([str(value) for value in argv])
    args.func(args)


def _build(benchmark: str, source: Path):
    return build_historical_package(
        source, None, SubmitterMetadata.model_validate(SUBMITTER), benchmark
    )


def _edit_cell(source: Path, benchmark: str, language: str, tmp_path: Path, edit):
    """Copy the export and apply ``edit`` to one cell's recorded results."""
    copy = tmp_path / "export"
    shutil.copytree(source, copy)
    path = (
        copy
        / "trajectories"
        / language
        / HISTORICAL_EXPORT_DIRS[DOMAINS[benchmark]]
        / "results.json"
    )
    data = json.loads(path.read_text())
    edit(data)
    path.write_text(json.dumps(data))
    return copy


def test_domain_export_round_trips_as_a_package(release_export, tmp_path):
    benchmark, source, metadata = release_export
    output = tmp_path / "out"
    _submit(
        "prepare",
        source,
        "--historical-voice-v2",
        "--benchmark",
        benchmark,
        "--metadata-file",
        metadata,
        "--output",
        output,
    )
    name = f"{NAME}_{benchmark}"
    assert [path.name for path in output.iterdir()] == [name]
    package = load_package(output / name)
    assert package.name == name
    frame = load_frame(benchmark)
    assert package.submission.frame == frame
    assert package.diagnostics is None
    # Call quality is defined for the core voice-v2 frame only.
    assert package.quality is None
    assert not (output / name / "benchmark-quality.json").exists()
    assert package.metadata.evidence == "historical-voice-v2-2026-10-02"
    assert (package.metadata.agent_provider, package.metadata.agent_model) == (
        "openai_live",
        "offline-fixture",
    )
    summary = package.submission.summary
    owed = sum(len(cell.canonical_task_ids) for cell in frame.cells)
    assert summary.complete and summary.scored_trials == owed
    assert summary.condition_fingerprint is not None
    assert {trial.binding_revision for trial in package.submission.trials} == {
        cell.binding_revision for cell in frame.cells
    }
    assert set(package.submission.sources) == {
        f"{cell.language}/{cell.domain}" for cell in frame.cells
    }
    # Entity's leaderboard score is crossed Pass^3; banking carries none.
    pass3_file = output / name / "benchmark-pass3.json"
    if benchmark == "entity":
        tasks = len(frame.cells[0].canonical_task_ids)
        passed = sum(
            _pass3_reward("regular", position) == 1.0
            and _pass3_reward("channel_heavy", position) == 1.0
            for position in range(tasks)
        )
        assert pass3_file.exists()
        assert package.pass3.passed_all == passed * len(frame.cells)
        assert package.pass3.score == pytest.approx(passed / tasks)
    else:
        assert package.pass3 is None and not pass3_file.exists()
    # Validation reads the frame from the package; no --benchmark needed.
    _submit("validate", output / name, "--historical-voice-v2", "--results", source)
    with pytest.raises(ValueError, match="packages"):
        other = "entity" if benchmark == "banking" else "banking"
        _submit(
            "validate",
            output / name,
            "--historical-voice-v2",
            "--benchmark",
            other,
            "--results",
            source,
        )


def test_accepted_digest_is_an_older_verified_one(release_export):
    benchmark, source, _ = release_export
    domain = DOMAINS[benchmark]
    recorded = json.loads(
        (
            source / "trajectories/es" / HISTORICAL_EXPORT_DIRS[domain] / "results.json"
        ).read_text()
    )["info"]["language_artifacts"]["runtime_rules_sha256"]
    released = load_cell_language("es", domain, "voice").artifact_pins
    assert recorded in ACCEPTED_RUNTIME_RULES[domain]
    assert recorded != released.runtime_rules_sha256
    assert _build(benchmark, source).submission.summary.complete


def test_unknown_runtime_rules_digest_is_rejected(release_export, tmp_path):
    benchmark, source, _ = release_export

    def unknown(data):
        for entry in data["info"]["language_tasks"].values():
            entry["runtime_rules_sha256"] = "0" * 64

    copy = _edit_cell(source, benchmark, "ko", tmp_path, unknown)
    with pytest.raises(ValueError, match="unverified runtime_rules_sha256"):
        _build(benchmark, copy)


def test_english_cells_must_match_exactly(release_export, tmp_path):
    benchmark, source, _ = release_export
    domain = DOMAINS[benchmark]

    def accepted_but_unexpected(data):
        rules = sorted(ACCEPTED_RUNTIME_RULES[domain])[0]
        data["info"]["language_artifacts"]["runtime_rules_sha256"] = rules

    copy = _edit_cell(source, benchmark, "en", tmp_path, accepted_but_unexpected)
    with pytest.raises(ValueError, match="unverified runtime_rules_sha256"):
        _build(benchmark, copy)


@pytest.mark.parametrize(
    "field", ["tasks_sha256", "localization_revision", "domain_revision"]
)
def test_artifact_mismatch_is_rejected(release_export, tmp_path, field):
    benchmark, source, _ = release_export

    def changed(data):
        entry = next(iter(data["info"]["language_tasks"].values()))
        entry[field] = "changed"

    copy = _edit_cell(source, benchmark, "hi", tmp_path, changed)
    with pytest.raises(ValueError, match=f"binding differs.*{field}"):
        _build(benchmark, copy)


def test_cell_artifact_record_is_checked(release_export, tmp_path):
    benchmark, source, _ = release_export

    def changed(data):
        data["info"]["language_artifacts"]["scenario_sha256"] = "1" * 64

    copy = _edit_cell(source, benchmark, "zh", tmp_path, changed)
    with pytest.raises(ValueError, match="binding differs.*scenario_sha256"):
        _build(benchmark, copy)


def test_task_set_mismatch_is_rejected(release_export, tmp_path):
    benchmark, source, _ = release_export

    def off_subset(data):
        first = data["tasks"][0]["id"]
        data["info"]["language_tasks"][first]["canonical_task_id"] = "not_in_subset"

    copy = _edit_cell(source, benchmark, "pt", tmp_path, off_subset)
    with pytest.raises(ValueError, match="task set differs from the frozen subset"):
        _build(benchmark, copy)


def test_mixed_recorded_conditions_are_rejected(release_export, tmp_path):
    benchmark, source, _ = release_export

    def changed(data):
        data["info"]["audio_native_config"]["reasoning_effort"] = "low"

    copy = _edit_cell(source, benchmark, "es", tmp_path, changed)
    with pytest.raises(ValueError, match="mixes recorded run conditions"):
        _build(benchmark, copy)


def test_non_regular_condition_is_rejected(release_export, tmp_path):
    benchmark, source, _ = release_export

    def heavy(data):
        data["info"]["channel_effects_mode"] = "heavy"

    copy = _edit_cell(source, benchmark, "es", tmp_path, heavy)
    with pytest.raises(ValueError, match="regular speech condition"):
        _build(benchmark, copy)


def _edit_condition(source: Path, label: str, language: str, tmp_path: Path, edit):
    """Copy the export and apply ``edit`` to one non-regular Pass^3 condition."""
    copy = tmp_path / "export"
    shutil.copytree(source, copy)
    path = copy / "trajectories" / language / HISTORICAL_PASS3_DIRS[label]
    data = json.loads((path / "results.json").read_text())
    edit(data)
    (path / "results.json").write_text(json.dumps(data))
    return copy


def test_pass3_condition_must_match_the_catalog(release_export, tmp_path):
    benchmark, source, _ = release_export
    if benchmark != "entity":
        pytest.skip("Pass^3 scores the entity frame only")

    def wrong_seed(data):
        data["info"]["seed"] = 42

    copy = _edit_condition(source, "speech_heavy", "es", tmp_path, wrong_seed)
    with pytest.raises(ValueError, match="Pass\\^3 catalog"):
        _build(benchmark, copy)


def test_pass3_condition_must_share_run_settings(release_export, tmp_path):
    benchmark, source, _ = release_export
    if benchmark != "entity":
        pytest.skip("Pass^3 scores the entity frame only")

    def other_caller(data):
        data["info"]["user_info"] = {"llm": "another-caller"}

    copy = _edit_condition(source, "channel_heavy", "ko", tmp_path, other_caller)
    with pytest.raises(ValueError, match="differ from regular"):
        _build(benchmark, copy)


def test_pass3_edit_fails_validation(release_export, tmp_path):
    benchmark, source, metadata = release_export
    if benchmark != "entity":
        pytest.skip("Pass^3 scores the entity frame only")
    output = tmp_path / "out"
    _submit(
        "prepare",
        source,
        "--historical-voice-v2",
        "--benchmark",
        benchmark,
        "--metadata-file",
        metadata,
        "--output",
        output,
    )
    path = output / f"{NAME}_{benchmark}" / "benchmark-pass3.json"
    data = json.loads(path.read_text())
    data["cells"][0]["tasks"][1]["passes"][1] = False
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Pass\\^3 counts differ"):
        load_package(path.parent)


def test_domain_frames_take_no_judges(release_export, tmp_path):
    benchmark, source, metadata = release_export
    with pytest.raises(ValueError, match="takes no --judges-root"):
        _submit(
            "prepare",
            source,
            "--historical-voice-v2",
            "--benchmark",
            benchmark,
            "--judges-root",
            tmp_path,
            "--metadata-file",
            metadata,
            "--output",
            tmp_path / "out",
        )
    with pytest.raises(ValueError, match="--benchmark voice, banking, entity"):
        _submit(
            "prepare",
            source,
            "--historical-voice-v2",
            "--benchmark",
            "text",
            "--metadata-file",
            metadata,
            "--output",
            tmp_path / "out",
        )


def test_unscored_call_is_an_explicit_error(tmp_path):
    """An infrastructure error stays an explicit error trial, outside the score."""
    frame = load_frame("banking")
    task = frame.cells[0].canonical_task_ids[3]
    _write_export(tmp_path, "banking", error_task=task)
    summary = _build("banking", tmp_path).submission.summary
    owed = sum(len(cell.canonical_task_ids) for cell in frame.cells)
    assert (summary.error_trials, summary.scored_trials) == (1, owed - 1)
    assert summary.complete


def test_review_requires_the_suffixed_package_name(release_export, tmp_path):
    """Review dry runs (no upload) accept only the frame-suffixed directory."""
    from rich.console import Console

    from tau2.scripts.leaderboard.review_submission import _review_package

    benchmark, source, metadata = release_export
    output = tmp_path / "out"
    _submit(
        "prepare",
        source,
        "--historical-voice-v2",
        "--benchmark",
        benchmark,
        "--metadata-file",
        metadata,
        "--output",
        output,
    )
    args = argparse.Namespace(
        verify_only=False, historical_voice_v2=True, judges_root=[], upload=False
    )
    console = Console(quiet=True)
    _review_package(output / f"{NAME}_{benchmark}", source, args, console)
    unsuffixed = output / NAME
    (output / f"{NAME}_{benchmark}").rename(unsuffixed)
    with pytest.raises(ValueError, match=f"must be named {NAME}_{benchmark}"):
        _review_package(unsuffixed, source, args, console)
