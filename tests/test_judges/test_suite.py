"""Offline suite orchestration cannot mix contracts or silently call judges."""

import json
import queue
import threading
import time
from pathlib import Path

import pytest

from tau2.data_model.judges import JudgeInput, Measure, MeasureVerdict, PlanAction
from tau2.judges import suite
from tau2.judges.catalog import load_catalog
from tau2.judges.suite import (
    JudgeExecutionConfig,
    JudgeJob,
    JudgeShard,
    JudgeWorkerMessage,
    execute_suite,
    plan_suite,
    read_verdicts,
)


def inputs():
    return [
        JudgeInput(
            run_id="run",
            simulation_id=f"sim-{trial}",
            trial=trial,
            language="pt",
            agent_gender="female",
            input_fingerprint="a" * 64,
            has_audio=True,
        )
        for trial in (0, 1)
    ]


def passing_backend(item, contract, measure):
    return MeasureVerdict.model_validate(
        {
            "units": [
                {
                    "unit_id": "call",
                    "opportunity": True,
                    "violation": False,
                    "evidence": "supported",
                }
            ]
        }
    )


def test_production_catalog_contains_only_selected_judges():
    catalog = load_catalog()
    assert {contract.contract_id for contract in catalog.contracts} == {
        pair.contract_id for pair in catalog.pairs
    }


def test_dry_run_calls_nothing_and_writes_nothing(tmp_path):
    def forbidden(*args):
        pytest.fail("dry run reached the backend")

    path = tmp_path / "verdicts.json"
    plan = execute_suite(
        inputs(),
        load_catalog(),
        path,
        only=[Measure.NATURALNESS],
        dry_run=True,
        backend=forbidden,
    )
    assert len(plan.items) == 2
    assert all(item.action == PlanAction.MISSING for item in plan.items)
    assert not path.exists()


def test_resume_reuses_matching_contract_and_all_trials(tmp_path):
    path = tmp_path / "verdicts.json"
    first = execute_suite(
        inputs(),
        load_catalog(),
        path,
        only=[Measure.NATURALNESS],
        backend=passing_backend,
    )
    assert len(first.items) == 2
    resumed = execute_suite(
        inputs(),
        load_catalog(),
        path,
        only=[Measure.NATURALNESS],
        backend=lambda *_: pytest.fail("cache miss"),
    )
    assert all(item.action == PlanAction.REUSE for item in resumed.items)
    selected = plan_suite(
        inputs(), load_catalog(), [], only=[Measure.NATURALNESS], trials=[1]
    )
    assert [item.input.trial for item in selected.items] == [1]


def test_preserve_errors_fills_gaps_without_retrying_recorded_failure(tmp_path):
    path = tmp_path / "verdicts.json"

    def fail(*_args):
        raise RuntimeError("recorded failure")

    execute_suite(
        inputs()[:1], load_catalog(), path, only=[Measure.NATURALNESS], backend=fail
    )
    before = path.read_bytes()
    execute_suite(
        inputs()[:1],
        load_catalog(),
        path,
        only=[Measure.NATURALNESS],
        retry_errors=False,
        backend=lambda *_args: pytest.fail("recorded error was retried"),
    )
    assert path.read_bytes() == before


def test_judge_worker_bounds_in_flight_calls(monkeypatch, tmp_path):
    selected = [
        inputs()[0].model_copy(update={"simulation_id": f"sim-{index}"})
        for index in range(6)
    ]
    catalog = load_catalog()
    plan = plan_suite(selected, catalog, [], only=[Measure.NATURALNESS])
    lock = threading.Lock()
    active = peak = 0

    class FakeSavedRun:
        def __init__(self, results):
            self.results_path = results
            self.metadata = self.tasks = self.language = self.run_id = None

        def load(self, _item):
            return None

    class FakeJudge:
        def evaluate(self, _evidence, _measure):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.02)
            with lock:
                active -= 1
            return passing_backend(None, None, None)

    monkeypatch.setattr(suite, "SavedRun", FakeSavedRun)
    monkeypatch.setattr(suite.Results, "_detect_format", lambda _path: "dir")
    monkeypatch.setattr(suite, "create_judge", lambda _contract: FakeJudge())
    output = queue.Queue()
    shard = JudgeShard(
        results=tmp_path,
        catalog=catalog,
        jobs=[JudgeJob(item=item) for item in plan.items],
        concurrency=3,
    )
    suite._run_judge_shard(shard.model_dump_json(), output)
    messages = [
        JudgeWorkerMessage.model_validate_json(output.get_nowait())
        for _ in range(len(plan.items))
    ]
    assert all(message.verdict is not None for message in messages)
    assert peak == 3
    assert JudgeExecutionConfig(processes=8, concurrency=10).processes == 8


def test_saved_naturalness_v16_loads_but_requires_rejudging(tmp_path):
    path = tmp_path / "verdicts.json"
    catalog = load_catalog()
    execute_suite(
        inputs(), catalog, path, only=[Measure.NATURALNESS], backend=passing_backend
    )
    recorded = json.loads(path.read_text())
    for row in recorded:
        assert (
            row["identity"]["contract_id"]
            == "naturalness-v17-rubric-v21-language-specific-draft2"
        )
        assert row["identity"]["prompt_version"] == "v17"
        row["identity"].update(
            contract_id="naturalness-v16-rubric-v20",
            implementation="naturalness-v16",
            prompt_version="v16",
            catalog_revision="validated-measures-v2",
        )
        row["verdict"]["units"][0]["evidence"] = json.dumps(
            {
                "factor_id": "natural_word_choice",
                "opportunity": True,
                "violated": False,
                "severity": 0,
                "reasoning": "Natural",
                "quote": "",
            }
        )
    path.write_text(json.dumps(recorded))
    before = path.read_bytes()
    loaded = read_verdicts(path)
    assert len(loaded) == 2
    assert all(row.identity.implementation == "naturalness-v16" for row in loaded)
    plan = execute_suite(
        inputs(),
        catalog,
        path,
        only=[Measure.NATURALNESS],
        dry_run=True,
        backend=lambda *_: pytest.fail("Planning must not call the judge"),
    )
    assert all(item.action == PlanAction.STALE for item in plan.items)
    assert path.read_bytes() == before


@pytest.mark.parametrize("changed", ["input", "model", "args", "prompt", "rubric"])
def test_cache_invalidates_changed_contract_or_input(tmp_path, changed):
    path = tmp_path / "verdicts.json"
    catalog = load_catalog()
    execute_suite(
        inputs(), catalog, path, only=[Measure.NATURALNESS], backend=passing_backend
    )
    changed_inputs = inputs()
    contract = next(
        c
        for c in catalog.contracts
        if c.contract_id == "naturalness-v17-rubric-v21-language-specific-draft2"
    )
    if changed == "input":
        changed_inputs[0].input_fingerprint = "b" * 64
    elif changed == "model":
        contract.model = "changed"
    elif changed == "args":
        contract.model_args = {"reasoning_effort": "low"}
    elif changed == "prompt":
        contract.prompt.prompt_version = "changed"
    else:
        contract.prompt.rubric_version = "changed"
    plan = execute_suite(
        changed_inputs, catalog, path, only=[Measure.NATURALNESS], dry_run=True
    )
    assert plan.items[0].action == PlanAction.STALE


def test_force_and_unsupported_pairs_and_gender_scope(tmp_path):
    path = tmp_path / "verdicts.json"
    execute_suite(
        inputs(),
        load_catalog(),
        path,
        only=[Measure.NATURALNESS],
        backend=passing_backend,
    )
    force = execute_suite(
        inputs(),
        load_catalog(),
        path,
        only=[Measure.NATURALNESS],
        force=True,
        dry_run=True,
    )
    assert all(item.action == PlanAction.FORCE for item in force.items)
    item = inputs()[0].model_copy(update={"language": "en"})
    assert (
        plan_suite([item], load_catalog(), [], only=[Measure.NATURALNESS])
        .items[0]
        .action
        == PlanAction.MISSING
    )
    item = inputs()[0].model_copy(update={"agent_gender": "male"})
    assert (
        plan_suite([item], load_catalog(), [], only=[Measure.GENDER_AGREEMENT])
        .items[0]
        .action
        == PlanAction.NOT_APPLICABLE
    )
    item = inputs()[0].model_copy(update={"has_audio": False})
    assert (
        plan_suite([item], load_catalog(), [], only=[Measure.SPEECH_FIDELITY])
        .items[0]
        .action
        == PlanAction.MISSING_INPUT
    )


def test_malformed_verdict_is_persisted_error_and_retried(tmp_path):
    path = tmp_path / "verdicts.json"
    execute_suite(
        inputs(),
        load_catalog(),
        path,
        only=[Measure.NATURALNESS],
        backend=lambda *_: '{"units": []}',
    )
    rows = json.loads(path.read_text())
    assert len(rows) == 2 and all(row["error"] for row in rows)
    plan = execute_suite(
        inputs(), load_catalog(), path, only=[Measure.NATURALNESS], dry_run=True
    )
    assert all(item.action == PlanAction.RETRY_ERROR for item in plan.items)
    execute_suite(
        inputs(),
        load_catalog(),
        path,
        only=[Measure.NATURALNESS],
        backend=passing_backend,
    )
    assert all(row["error"] is None for row in json.loads(path.read_text()))


def test_catalog_only_contains_passing_measures_and_exact_models():
    catalog = load_catalog()
    assert len(catalog.pairs) == 27
    assert any(
        p.language == "en" and p.measure == Measure.NATURALNESS for p in catalog.pairs
    )
    assert not any(
        p.language in {"ko", "zh"} and p.measure == Measure.NAME_ADDRESS
        for p in catalog.pairs
    )
    assert next(
        c for c in catalog.contracts if c.contract_id.startswith("naturalness-")
    ).model_args == {"reasoning_effort": "xhigh"}
    assert next(
        c for c in catalog.contracts if c.contract_id.startswith("quality-")
    ).model_args == {"reasoning_effort": "high"}


def test_production_judges_have_no_annotation_imports():
    import ast

    root = Path(__file__).resolve().parents[2] / "src" / "tau2" / "judges"
    for path in root.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("tau2.annotation")
            elif isinstance(node, ast.Import):
                assert not any(
                    alias.name.startswith("tau2.annotation") for alias in node.names
                )


def test_read_only_summary_distinguishes_missing_error_and_not_applicable(tmp_path):
    from tau2.judges.suite import read_verdicts
    from tau2.judges.summary import summarize_suite

    source = inputs() + [
        inputs()[0].model_copy(update={"simulation_id": "unselected", "language": "fr"})
    ]
    path = tmp_path / "verdicts.json"
    execute_suite(
        source[:1],
        load_catalog(),
        path,
        only=[Measure.NATURALNESS],
        backend=lambda *_: "malformed",
    )
    stored = read_verdicts(path)
    before = path.read_bytes()
    report = summarize_suite(
        plan_suite(source, load_catalog(), stored, only=[Measure.NATURALNESS]), stored
    )[0]
    assert report.error_calls == 1
    assert report.not_judged_calls == 1
    assert report.not_applicable_calls == 1
    assert report.applicable_calls == 2 and not report.complete
    assert report.observed_pass_fraction is None
    assert before == path.read_bytes()


def test_audio_only_changes_leave_text_verdict_cache_valid():
    from tau2.data_model.judges import MeasureVerdict, StoredVerdict, UnitVerdict

    catalog = load_catalog()
    source = JudgeInput(
        run_id="r",
        simulation_id="s",
        trial=0,
        language="es",
        agent_gender="female",
        input_fingerprint="a" * 64,
        has_audio=True,
        audio_fingerprint="b" * 64,
    )
    plan = plan_suite(
        [source], catalog, [], only=[Measure.NATURALNESS, Measure.SPEECH_FIDELITY]
    )
    saved = [
        StoredVerdict(
            identity=item.identity,
            verdict=MeasureVerdict(
                units=[
                    UnitVerdict(
                        unit_id="0", opportunity=True, violation=False, evidence="fine"
                    )
                ]
            ),
        )
        for item in plan.items
    ]
    changed = source.model_copy(update={"audio_fingerprint": "c" * 64})
    updated = plan_suite(
        [changed], catalog, saved, only=[Measure.NATURALNESS, Measure.SPEECH_FIDELITY]
    )
    assert [item.action for item in updated.items] == [
        PlanAction.REUSE,
        PlanAction.STALE,
    ]


def test_legacy_transport_verdicts_are_stale(tmp_path):
    path = tmp_path / "verdicts.json"
    execute_suite(
        inputs(),
        load_catalog(),
        path,
        only=[Measure.NATURALNESS],
        backend=passing_backend,
    )
    recorded = json.loads(path.read_text())
    for row in recorded:
        del row["identity"]["transport_revision"]
    path.write_text(json.dumps(recorded))
    plan = plan_suite(
        inputs(), load_catalog(), read_verdicts(path), only=[Measure.NATURALNESS]
    )
    assert all(item.action == PlanAction.STALE for item in plan.items)


def test_incompatible_saved_diagnostics_ask_for_rejudging(tmp_path):
    path = tmp_path / "run.diagnostics.json"
    path.write_text(json.dumps([{"identity": {"input": {"audio_verified": True}}}]))
    with pytest.raises(ValueError, match="re-judge"):
        read_verdicts(path)


def test_partial_results_report_coverage_and_resume_without_losing_successes(tmp_path):
    from tau2.data_model.judges import UnitError
    from tau2.judges.summary import summarize_suite

    path = tmp_path / "verdicts.json"
    catalog = load_catalog()

    def partial(*args):
        result = passing_backend(*args)
        result.unit_errors = [
            UnitError(unit_id="bad", kind="response", error="Incomplete JSON")
        ]
        return result

    execute_suite(inputs(), catalog, path, only=[Measure.NATURALNESS], backend=partial)
    stored = read_verdicts(path)
    plan = plan_suite(inputs(), catalog, stored, only=[Measure.NATURALNESS])
    assert all(i.action == PlanAction.RETRY_ERROR for i in plan.items)
    report = summarize_suite(plan, stored)[0]
    assert report.error_calls == report.partial_calls == report.error_units == 2
    assert report.pass_units == 2 and report.judged_calls == 0 and not report.complete
    assert report.no_opportunity_units == 0
    before = path.read_bytes()
    with pytest.raises(ValueError, match="resume backend"):
        execute_suite(
            inputs(), catalog, path, only=[Measure.NATURALNESS], backend=passing_backend
        )
    assert path.read_bytes() == before

    def broken(*args):
        raise RuntimeError("Lost connection")

    execute_suite(
        inputs(),
        catalog,
        path,
        only=[Measure.NATURALNESS],
        backend=passing_backend,
        resume_backend=broken,
    )
    failed = read_verdicts(path)
    assert all(row.verdict.units == stored[0].verdict.units for row in failed)
    assert all("Lost connection" in row.verdict.unit_errors[0].error for row in failed)

    execute_suite(
        inputs(),
        catalog,
        path,
        only=[Measure.NATURALNESS],
        backend=passing_backend,
        resume_backend=lambda item, contract, measure, previous: MeasureVerdict(
            units=previous.units
        ),
    )
    failed = read_verdicts(path)
    assert all(row.verdict.units == stored[0].verdict.units for row in failed)
    assert all("unit coverage" in row.verdict.unit_errors[0].error for row in failed)

    def resume(item, contract, measure, previous):
        assert len(previous.units) == len(previous.unit_errors) == 1
        return MeasureVerdict(
            units=[
                *previous.units,
                previous.units[0].model_copy(update={"unit_id": "bad"}),
            ]
        )

    execute_suite(
        inputs(),
        catalog,
        path,
        only=[Measure.NATURALNESS],
        backend=lambda *_: pytest.fail("full retry"),
        resume_backend=resume,
    )
    stored = read_verdicts(path)
    report = summarize_suite(
        plan_suite(inputs(), catalog, stored, only=[Measure.NATURALNESS]), stored
    )[0]
    assert report.complete and report.pass_units == 4 and report.error_units == 0


def _errored_verdict(*errors):
    verdict = passing_backend(None, None, None)
    verdict.unit_errors = list(errors)
    return verdict


def test_failed_replies_are_asked_again_in_the_same_run(tmp_path):
    """A judge that loops until its output limit on one unit usually succeeds
    when asked again; without a retry that unit left its whole call unjudged
    until someone re-ran the suite."""
    from tau2.data_model.judges import UnitError

    failed = UnitError(unit_id="bad", kind="response", error="output token limit")
    replies = iter([_errored_verdict(failed), None])
    asked = []

    def resume(item, contract, measure, previous):
        asked.append([error.unit_id for error in previous.unit_errors])
        reply = next(replies)
        if reply is not None:
            return reply
        return MeasureVerdict(
            units=[
                *previous.units,
                previous.units[0].model_copy(update={"unit_id": "bad"}),
            ]
        )

    path = tmp_path / "verdicts.json"
    execute_suite(
        inputs()[:1],
        load_catalog(),
        path,
        only=[Measure.NATURALNESS],
        backend=lambda *args: _errored_verdict(failed),
        resume_backend=resume,
        unit_retries=2,
    )

    [stored] = read_verdicts(path)
    assert asked == [["bad"], ["bad"]]
    assert stored.verdict.unit_errors == []
    assert [u.unit_id for u in stored.verdict.units] == ["call", "bad"]
    assert JudgeExecutionConfig().unit_retries == 2


def test_input_errors_and_exhausted_retries_stay_recorded(tmp_path):
    from tau2.data_model.judges import UnitError

    missing = UnitError(unit_id="live", kind="input", error="No transcript")
    failed = UnitError(unit_id="bad", kind="response", error="output token limit")
    path = tmp_path / "verdicts.json"
    execute_suite(
        inputs()[:1],
        load_catalog(),
        path,
        only=[Measure.NATURALNESS],
        backend=lambda *args: _errored_verdict(missing),
        resume_backend=lambda *args: pytest.fail("an input error was retried"),
        unit_retries=2,
    )
    assert read_verdicts(path)[0].verdict.unit_errors == [missing]

    attempts = []

    def still_failing(item, contract, measure, previous):
        attempts.append(1)
        return previous

    def lost_connection(*args):
        raise RuntimeError("Lost connection")

    for resume, expected in ((still_failing, 2), (lost_connection, 0)):
        attempts.clear()
        execute_suite(
            inputs()[:1],
            load_catalog(),
            path,
            only=[Measure.NATURALNESS],
            force=True,
            backend=lambda *args: _errored_verdict(failed),
            resume_backend=resume,
            unit_retries=2,
        )
        # The unit stays an error for a later suite run; the call's successful
        # units are kept either way.
        [stored] = read_verdicts(path)
        assert len(attempts) == expected
        assert stored.error is None
        assert stored.verdict.unit_errors == [failed]
        assert [u.unit_id for u in stored.verdict.units] == ["call"]


def test_unjudged_unit_cannot_also_be_a_pass():
    from tau2.data_model.judges import UnitError

    unit = passing_backend(None, None, None).units[0]
    with pytest.raises(ValueError, match="Duplicate"):
        MeasureVerdict(
            units=[unit],
            unit_errors=[
                UnitError(unit_id=unit.unit_id, kind="input", error="No reference")
            ],
        )
    with pytest.raises(ValueError, match="Empty"):
        MeasureVerdict(units=[])


def _live_input_rows(tmp_path, backends):
    """Store one naturalness verdict per input, using one backend per trial."""
    path = tmp_path / "verdicts.json"
    execute_suite(
        inputs(),
        load_catalog(),
        path,
        only=[Measure.NATURALNESS],
        backend=lambda item, *rest: backends[item.trial](item, *rest),
    )
    stored = read_verdicts(path)
    plan = plan_suite(inputs(), load_catalog(), stored, only=[Measure.NATURALNESS])
    return plan, stored


def _with_errors(*kinds):
    from tau2.data_model.judges import UnitError

    def backend(*args):
        result = passing_backend(*args)
        result.unit_errors = [
            UnitError(unit_id=f"u{index}", kind=kind, error="No transcript")
            for index, kind in enumerate(kinds)
        ]
        return result

    return backend


def _live_input_call_error(*_):
    from tau2.judges.run_evidence import LIVE_INPUT_CALL_ERROR

    raise ValueError(f"{LIVE_INPUT_CALL_ERROR}: segment 3 has no transcript")


def test_missing_live_transcripts_are_excluded_not_blocking(tmp_path):
    from tau2.judges.submission import CellDiagnostics, _cell_rows
    from tau2.judges.summary import summarize_suite

    plan, stored = _live_input_rows(
        tmp_path, [_with_errors("input"), _live_input_call_error]
    )
    assert stored[1].error.startswith("ValueError: Call-level judgment")
    report = summarize_suite(plan, stored)[0]
    assert report.complete
    assert (report.judged_calls, report.input_excluded_calls) == (1, 1)
    assert report.error_calls == 0 and report.partial_calls == 1
    assert report.error_units == report.input_error_units == 1
    assert report.pass_units == 1 and report.observed_pass_fraction == 1
    fingerprints, _ = _cell_rows(plan.items, stored)
    cell = CellDiagnostics(
        language="pt", domain="retail", summary=[report], verdicts=fingerprints
    )
    assert CellDiagnostics.model_validate_json(cell.model_dump_json()) == cell
    forged = cell.model_dump()
    forged["verdicts"][0]["input_error_units"] = 0
    with pytest.raises(ValueError, match="does not match verdict fingerprints"):
        CellDiagnostics.model_validate(forged)


def test_other_judge_errors_still_block_beside_live_input(tmp_path):
    from tau2.judges.summary import summarize_suite

    plan, stored = _live_input_rows(
        tmp_path, [_with_errors("input", "response"), _live_input_call_error]
    )
    report = summarize_suite(plan, stored)[0]
    assert not report.complete
    assert (report.judged_calls, report.error_calls) == (0, 1)
    assert report.input_excluded_calls == 1 and report.input_error_units == 1

    def api_failure(*_):
        raise ValueError("RateLimitError: try again")

    plan, stored = _live_input_rows(tmp_path / "api", [passing_backend, api_failure])
    report = summarize_suite(plan, stored)[0]
    assert not report.complete and report.error_calls == 1
    assert report.input_excluded_calls == 0


def test_measure_with_every_call_excluded_is_incomplete(tmp_path):
    from tau2.data_model.judges import UnitError
    from tau2.judges.summary import summarize_suite

    def no_units(*_):
        return MeasureVerdict(
            units=[],
            unit_errors=[UnitError(unit_id="1", kind="input", error="No transcript")],
        )

    plan, stored = _live_input_rows(tmp_path, [no_units, _live_input_call_error])
    report = summarize_suite(plan, stored)[0]
    assert report.input_excluded_calls == 2 and report.judged_calls == 0
    assert not report.complete


def _fidelity_rows(tmp_path, backends):
    """Store one speech-fidelity verdict per input, using one backend per trial."""
    path = tmp_path / "verdicts.json"
    execute_suite(
        inputs(),
        load_catalog(),
        path,
        only=[Measure.SPEECH_FIDELITY],
        backend=lambda item, *rest: backends[item.trial](item, *rest),
    )
    stored = read_verdicts(path)
    plan = plan_suite(inputs(), load_catalog(), stored, only=[Measure.SPEECH_FIDELITY])
    return plan, stored


def _output_limit(units=True, finish_reason="length"):
    from tau2.data_model.judges import UnitError

    def backend(*args):
        result = passing_backend(*args)
        if not units:
            result = MeasureVerdict.model_construct(units=[], unit_errors=[])
        result.unit_errors = [
            UnitError(
                unit_id="921",
                kind="response",
                error="ValueError: Judge response reached its output token limit",
                finish_reason=finish_reason,
            )
        ]
        return result

    return backend


def test_fidelity_output_limit_is_left_out_not_blocking(tmp_path):
    from tau2.judges.submission import CellDiagnostics, _cell_rows
    from tau2.judges.summary import summarize_suite

    plan, stored = _fidelity_rows(
        tmp_path, [_output_limit(), _output_limit(units=False)]
    )
    report = summarize_suite(plan, stored)[0]
    assert report.complete and report.error_calls == 0
    assert (report.judged_calls, report.partial_calls) == (2, 1)
    assert report.error_units == report.output_limit_units == 2
    fingerprints, _ = _cell_rows(plan.items, stored)
    assert [row.output_limit_units for row in fingerprints] == [1, 1]
    cell = CellDiagnostics(
        language="pt", domain="retail", summary=[report], verdicts=fingerprints
    )
    assert CellDiagnostics.model_validate_json(cell.model_dump_json()) == cell
    forged = cell.model_dump()
    forged["verdicts"][0]["output_limit_units"] = 0
    with pytest.raises(ValueError, match="does not match verdict fingerprints"):
        CellDiagnostics.model_validate(forged)
    # Packages without such clips serialize exactly as before.
    plan, stored = _fidelity_rows(tmp_path / "clean", [passing_backend] * 2)
    clean = summarize_suite(plan, stored)[0]
    assert "output_limit_units" not in clean.model_dump_json()
    fingerprints, _ = _cell_rows(plan.items, stored)
    assert "output_limit_units" not in fingerprints[0].model_dump_json()


def test_other_fidelity_response_errors_still_block(tmp_path):
    from tau2.judges.summary import summarize_suite

    plan, stored = _fidelity_rows(
        tmp_path, [_output_limit(finish_reason=None), passing_backend]
    )
    report = summarize_suite(plan, stored)[0]
    assert not report.complete and report.error_calls == 1
    assert report.output_limit_units == 0
    # The same limit on another measure is still a judge error.
    plan, stored = _live_input_rows(tmp_path / "nat", [_output_limit()] * 2)
    report = summarize_suite(plan, stored)[0]
    assert not report.complete and report.error_calls == 2


def test_output_limit_clips_are_capped_per_package():
    from types import SimpleNamespace

    from tau2.judges.submission import output_limit_overage

    def package(limited, judged):
        row = SimpleNamespace(simulation_id="sim", trial=0, output_limit_units=limited)
        summary = SimpleNamespace(
            measure="speech_fidelity",
            pass_units=judged - limited,
            fail_units=0,
            output_limit_units=limited,
        )
        cell = SimpleNamespace(
            language="es", domain="airline", summary=[summary], verdicts=[row]
        )
        return SimpleNamespace(cells=[cell])

    assert output_limit_overage(package(0, 900)) == []
    assert output_limit_overage(package(2, 2000)) == []
    lines = output_limit_overage(package(3, 2000))
    assert "3 of 2000" in lines[0]
    assert lines[1] == "  es-airline sim trial 0: 3 clip(s)"
