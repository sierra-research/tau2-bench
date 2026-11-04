"""Official V2 aggregation must never silently lose missing/error coverage."""

import pytest

from tau2.data_model.benchmark import (
    BenchmarkFrame,
    BenchmarkSubmission,
    CellFrame,
    TaskTrial,
    TrialStatus,
)
from tau2.metrics.benchmark import (
    describe_reward_gaps,
    summarize,
    validate_submission,
)


@pytest.fixture
def frame():
    return BenchmarkFrame(
        benchmark_version="voice-v2",
        revision="fixture-v1",
        mode="voice",
        trials=[0, 1],
        cells=[
            CellFrame(
                language=language,
                domain=domain,
                canonical_task_ids=[str(i) for i in range(50)],
                binding_revision=f"{language}-{domain}-v1",
            )
            for language in ("en", "es", "pt", "hi", "zh", "ko")
            for domain in ("airline", "retail", "telecom")
        ],
    )


def result(cell, task="0", trial=0, reward=1.0, status=TrialStatus.SCORED):
    return TaskTrial(
        language=cell.language,
        domain=cell.domain,
        canonical_task_id=task,
        trial=trial,
        condition_fingerprint="a" * 64,
        binding_revision=cell.binding_revision,
        status=status,
        reward=reward,
    )


def test_complete_matrix_averages_trials_tasks_and_cells(frame):
    rows = [
        result(cell, task, trial, reward=float(i == 0))
        for i, cell in enumerate(frame.cells)
        for task in cell.canonical_task_ids
        for trial in frame.trials
    ]
    report = summarize(frame, rows)
    assert report.complete
    assert report.score == pytest.approx(1 / 18)
    assert report.by_language["en"].score == pytest.approx(1 / 3)
    assert report.by_domain["airline"].score == pytest.approx(1 / 6)
    assert report.by_language["es"].score == 0
    assert report.expected_trials == report.scored_trials == 1800
    assert report.missing_trials == report.error_trials == 0


def test_partial_observed_score_does_not_pool_unequal_trial_counts(frame):
    rows = [
        result(frame.cells[0], "0", 0, 0),
        result(frame.cells[0], "0", 1, 0),
        result(frame.cells[0], "1", 0, 1),
        result(frame.cells[1], "0", 0, 1),
        result(frame.cells[2], "0", 0, None, TrialStatus.ERROR),
    ]
    report = summarize(frame, rows)
    assert report.score is None
    assert not report.complete
    assert report.observed_score == pytest.approx(0.75)
    assert report.by_language["en"].score is None
    assert report.by_domain["airline"].score is None
    assert report.cells[0].observed_score == 0.5
    assert report.error_trials == 1
    assert report.missing_trials == 1795
    assert len(report.cells) == 18
    assert report.cells[-1].observed_score is None


@pytest.mark.parametrize(
    "change", ["condition", "binding", "task", "trial", "duplicate"]
)
def test_incompatible_or_ambiguous_results_are_rejected(frame, change):
    row = result(frame.cells[0])
    other = result(frame.cells[0], "1")
    if change == "condition":
        other.condition_fingerprint = "b" * 64
    elif change == "binding":
        other.binding_revision = "different-pack"
    elif change == "task":
        other.canonical_task_id = "unknown"
    elif change == "trial":
        other.trial = 10
    else:
        other = row
    with pytest.raises(ValueError):
        summarize(frame, [row, other])


def test_error_cannot_carry_reward_and_unscored_remains_incomplete(frame):
    with pytest.raises(ValueError):
        result(frame.cells[0], reward=1, status=TrialStatus.ERROR)
    report = summarize(
        frame, [result(frame.cells[0], reward=None, status=TrialStatus.UNSCORED)]
    )
    assert report.unscored_trials == 1
    assert report.score is None


def test_frame_rejects_duplicate_cells_and_inconsistent_domains(frame):
    with pytest.raises(ValueError, match="unique"):
        BenchmarkFrame(
            benchmark_version="voice-v2",
            revision="bad",
            mode="voice",
            trials=[0],
            cells=frame.cells + [frame.cells[0]],
        )
    with pytest.raises(ValueError, match="every released domain"):
        BenchmarkFrame(
            benchmark_version="voice-v2",
            revision="bad",
            mode="voice",
            trials=[0],
            cells=frame.cells[:-1],
        )


def test_official_run_requires_a_separately_pinned_semantic_judge(frame):
    from tau2.data_model.benchmark import BenchmarkRunInfo, encode_conditions
    from tau2.data_model.run_config import AudioNativeConfig
    from tau2.data_model.semantic_communication import (
        CommunicateScoring,
        SemanticCommunicateSettings,
    )
    from tau2.data_model.simulation import TextRunConfig, VoiceRunConfig

    config = TextRunConfig(
        communicate_scoring=CommunicateScoring(scorer="substring"),
        benchmark=BenchmarkRunInfo(
            benchmark_version="voice-v2",
            frame_revision=frame.revision,
            language="en",
            domain="airline",
            binding_revision="fixture",
            canonical_task_ids=frame.cells[0].canonical_task_ids,
            selected_task_ids=frame.cells[0].canonical_task_ids,
            trials=[0],
            condition_fingerprint="a" * 64,
            conditions=encode_conditions(
                VoiceRunConfig(language="en", audio_native_config=AudioNativeConfig())
            ),
        ),
    )
    with pytest.raises(ValueError, match="pinned semantic"):
        config.validate()
    config.communicate_scoring = CommunicateScoring(
        judge=SemanticCommunicateSettings(model="explicit-test-judge", model_args={})
    )
    config.validate()


def full_cell(frame, cell, errors=(), reward=lambda task, trial: 1.0, skip=()):
    """Every owed trial of one cell; ``errors``/``skip`` hold (task, trial) keys."""
    return [
        result(cell, task, trial, None, TrialStatus.ERROR)
        if (task, trial) in errors
        else result(cell, task, trial, reward(task, trial))
        for task in cell.canonical_task_ids
        for trial in frame.trials
        if (task, trial) not in skip
    ]


def full_matrix(frame, **first_cell):
    rows = full_cell(frame, frame.cells[0], **first_cell)
    for cell in frame.cells[1:]:
        rows += full_cell(frame, cell)
    return rows


def test_infrastructure_errors_leave_the_denominator(frame):
    # Task 0 errors on both trials and drops out; task 1 keeps one scored trial.
    errors = {("0", 0), ("0", 1), ("1", 0)}
    report = summarize(
        frame,
        full_matrix(
            frame,
            errors=errors,
            reward=lambda task, trial: float(task in {"0", "1"} or trial == 0),
        ),
    )
    cell = report.cells[0]
    assert cell.complete and report.complete
    assert (cell.scored_trials, cell.error_trials, cell.missing_trials) == (97, 3, 0)
    # Task 1 scores 1 from its one scored trial; tasks 2..49 score 0.5 each.
    assert cell.score == pytest.approx((1 + 48 * 0.5) / 49)
    assert report.score == pytest.approx((cell.score + 17) / 18)
    assert report.by_language["en"].complete
    assert report.by_language["en"].error_trials == 3
    assert report.error_trials == 3
    assert describe_reward_gaps(report) == []


def test_all_errored_cell_never_counts(frame):
    cell = frame.cells[0]
    errors = {
        (task, trial) for task in cell.canonical_task_ids for trial in frame.trials
    }
    report = summarize(frame, full_matrix(frame, errors=errors))
    assert not report.cells[0].complete and report.cells[0].score is None
    assert not report.complete and report.score is None
    assert report.by_language["en"].score is None
    gaps = describe_reward_gaps(report)
    assert gaps[0].startswith("Rewards: 100 infrastructure-errored calls")
    assert gaps[1] == "  en-airline: no scored call, 100 errored (excluded)"


def test_missing_and_unscored_still_block_beside_errors(frame):
    missing = summarize(frame, full_matrix(frame, errors={("0", 0)}, skip={("1", 0)}))
    assert not missing.cells[0].complete and missing.score is None
    assert missing.cells[0].missing_trials == 1
    assert describe_reward_gaps(missing) == [
        "Rewards: 1/1800 calls lack a scored reward (1 missing, 0 unscored).",
        "Rewards: 1 infrastructure-errored calls are excluded from the score, "
        "not counted as gaps.",
        "  en-airline: 1 missing, 1 errored (excluded)",
    ]
    rows = full_matrix(frame, errors={("0", 0)}, skip={("1", 0)})
    rows.append(result(frame.cells[0], "1", 0, None, TrialStatus.UNSCORED))
    unscored = summarize(frame, rows)
    assert not unscored.complete and unscored.cells[0].unscored_trials == 1
    assert describe_reward_gaps(unscored)[-1] == (
        "  en-airline: 1 unscored, 1 errored (excluded)"
    )


def test_submission_with_errors_validates_round_trip(frame):
    rows = full_matrix(frame, errors={("0", 0), ("3", 1)})
    submission = BenchmarkSubmission(
        frame=frame, trials=rows, summary=summarize(frame, rows), sources={}
    )
    loaded = BenchmarkSubmission.model_validate_json(submission.model_dump_json())
    validate_submission(loaded)
    assert loaded.summary.complete and loaded.summary.error_trials == 2
    forged = loaded.model_copy(deep=True)
    forged.summary.error_trials = 0
    with pytest.raises(ValueError, match="summary does not match"):
        validate_submission(forged)
