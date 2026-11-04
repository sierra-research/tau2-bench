"""Reviewed diagnostic definitions are reusable without transferring validation."""

import argparse
import json
from types import SimpleNamespace

import pytest

from tau2.judges.authoring import compile_judge, draft_judge
from tau2.judges.authoring_models import CompileJudgeRequest
from tau2.judges.validation import production_catalog
from tau2.multilingual.factory.judge_authoring import add_judge_authoring_args
from tau2.utils.artifacts import ReviewApproval, content_digest


def parser():
    value = argparse.ArgumentParser()
    add_judge_authoring_args(
        value.add_subparsers(dest="factory_command", required=True)
    )
    return value


def test_register_list_adapt_new_language_without_validation(
    candidate, tmp_path, monkeypatch, capsys
):
    cli = parser()
    entry = tmp_path / "catalog" / "promise-accuracy-ro.json"
    args = cli.parse_args(
        [
            "register-judge",
            "--candidate",
            str(tmp_path / "compiled.json"),
            "--output",
            str(entry),
        ]
    )
    args.func(args)
    before = entry.read_bytes()
    args.func(args)
    assert entry.read_bytes() == before
    args = cli.parse_args(["list-judges", "--catalog-entry", str(entry)])
    capsys.readouterr()
    args.func(args)
    listed = json.loads(capsys.readouterr().out)
    assert (
        len([r for r in listed["entries"] if r["source"] == "authoring_catalog"]) == 31
    )
    custom = next(
        r for r in listed["entries"] if r["diagnostic_id"] == "promise_accuracy"
    )
    assert custom["status"] == "unvalidated" and custom["language"] == "ro"
    assert any(r["status"] == "validated" for r in listed["entries"])
    request = candidate.authoring.request.model_copy(
        update={
            "mode": "adapt_diagnostic",
            "language": "sv",
            "variety": "Swedish in Sweden",
            "source_contract_id": candidate.contract.contract_id,
            "source_language": "ro",
            "source_measure": "promise_accuracy",
        }
    )
    (tmp_path / "request.json").write_text(request.model_dump_json())
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        reference = json.loads(kwargs["messages"][1].content)["reference"]
        assert reference["source"] == "reviewed_candidate"
        assert reference["source_sha256"] == content_digest(candidate)
        return SimpleNamespace(content=candidate.authoring.criteria.model_dump_json())

    monkeypatch.setattr("tau2.judges.authoring.generate", generate)
    args = cli.parse_args(
        [
            "draft-judge",
            "--request",
            str(tmp_path / "request.json"),
            "--catalog-entry",
            str(entry),
            "--output",
            str(tmp_path / "swedish.json"),
        ]
    )
    args.func(args)
    args.func(args)
    assert len(calls) == 1
    from tau2.judges.authoring_models import JudgeDraft

    draft = JudgeDraft.model_validate_json((tmp_path / "swedish.json").read_text())
    result = compile_judge(
        CompileJudgeRequest(
            draft=draft,
            review=ReviewApproval(
                reviewed_by="Swedish reviewer", content_sha256=content_digest(draft)
            ),
            revision="sv-v1",
        ),
        tmp_path / "swedish-candidate.json",
    )
    assert result.status == "unvalidated" and result.scope.language == "sv"
    assert not any(
        p.measure == "promise_accuracy" for p in production_catalog([]).pairs
    )
    with pytest.raises(ValueError):
        production_catalog([entry])


def test_catalog_entry_tampering_collisions_and_stale_reference(
    candidate, tmp_path, monkeypatch
):
    from tau2.judges.authoring_registry import load_authoring_entries, register_judge

    first = tmp_path / "first.json"
    register_judge(candidate, first)
    tampered = candidate.model_dump()
    tampered["contract"]["model"] = "unreviewed-model"
    (tmp_path / "tampered.json").write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="reviewed authoring"):
        load_authoring_entries([tmp_path / "tampered.json"])
    with pytest.raises(ValueError, match="Duplicate"):
        load_authoring_entries([first, first])
    entries = load_authoring_entries([first])
    with pytest.raises(ValueError, match="adaptation"):
        draft_judge(
            candidate.authoring.request,
            tmp_path / "collision.json",
            catalog_entries=entries,
        )
    monkeypatch.setattr(
        "tau2.judges.authoring.generate",
        lambda **kw: SimpleNamespace(
            content=candidate.authoring.criteria.model_dump_json()
        ),
    )
    request = candidate.authoring.request.model_copy(
        update={
            "mode": "adapt_diagnostic",
            "source_contract_id": candidate.contract.contract_id,
            "source_language": "ro",
            "source_measure": "promise_accuracy",
            "language": "sv",
        }
    )
    draft_judge(request, tmp_path / "draft.json", catalog_entries=entries)
    changed = candidate.authoring.model_copy(deep=True)
    changed.criteria.negative_examples = "Changed reviewed requirement."
    replacement = compile_judge(
        CompileJudgeRequest(
            draft=changed,
            review=ReviewApproval(
                reviewed_by="human", content_sha256=content_digest(changed)
            ),
            revision=candidate.revision,
        ),
        tmp_path / "replacement.json",
    )
    with pytest.raises(ValueError, match="Stale"):
        draft_judge(request, tmp_path / "draft.json", catalog_entries=[replacement])
    with pytest.raises(ValueError, match="Duplicate"):
        load_authoring_entries([first, tmp_path / "replacement.json"])


def test_contributed_catalog_discovery_and_immutable_registration(
    candidate, tmp_path, monkeypatch
):
    import tau2.utils
    from tau2.judges.authoring_registry import (
        list_authoring_catalog,
        load_authoring_entries,
        register_judge,
    )

    shipped_data = tau2.utils.DATA_DIR
    monkeypatch.setattr("tau2.utils.DATA_DIR", tmp_path / "data")
    directory = tmp_path / "data/tau2/judge_catalog"
    assert not any(
        row.source == "reviewed_candidate" for row in list_authoring_catalog().entries
    )
    entry = directory / "promise-accuracy-ro-v1.json"
    register_judge(candidate, entry)
    assert load_authoring_entries() == [candidate]
    assert any(
        row.diagnostic_id == "promise_accuracy"
        for row in list_authoring_catalog().entries
    )
    with pytest.raises(ValueError, match="Duplicate"):
        register_judge(candidate, directory / "same-contract-another-file.json")
    request = candidate.authoring.request.model_copy(
        update={
            "mode": "adapt_diagnostic",
            "language": "sv",
            "source_language": "ro",
            "source_measure": "promise_accuracy",
            "source_contract_id": candidate.contract.contract_id,
        }
    )
    monkeypatch.setattr(
        "tau2.judges.authoring.generate",
        lambda **kw: SimpleNamespace(
            content=candidate.authoring.criteria.model_dump_json()
        ),
    )
    draft = draft_judge(request, tmp_path / "draft.json")
    assert draft.reference.source == "reviewed_candidate"
    assert draft.reference.source_review == candidate.review
    entry.unlink()
    compiled = compile_judge(
        CompileJudgeRequest(
            draft=draft,
            revision="v1",
            review=ReviewApproval(
                reviewed_by="Swedish reviewer", content_sha256=content_digest(draft)
            ),
        ),
        tmp_path / "sv.json",
    )
    assert compiled.scope.language == "sv" and compiled.status == "unvalidated"
    assert not any(pair.language == "sv" for pair in production_catalog([]).pairs)
    # Execute the adapted custom ID with actual saved-result preparation and no provider.
    from tau2.data_model.message import AssistantMessage
    from tau2.data_model.simulation import (
        Results,
        RewardInfo,
        SimulationRun,
        TerminationReason,
        TextRunConfig,
    )
    from tau2.data_model.tasks import Task
    from tau2.judges.candidates import run_candidate
    from tau2.judges.run_evidence import progress_path
    from tau2.runner.helpers import get_info

    # The run itself resolves against the shipped packs and domain data.
    with monkeypatch.context() as shipped:
        shipped.setattr("tau2.utils.DATA_DIR", shipped_data)
        info = get_info(TextRunConfig(domain="retail"))
    info.language = "sv"
    task = Task(id="fixture", user_scenario={"instructions": "Caller scenario."})
    simulation = SimulationRun(
        id="call",
        task_id=task.id,
        start_time="start",
        end_time="end",
        duration=1,
        termination_reason=TerminationReason.USER_STOP,
        trial=0,
        agent_voice_gender="female",
        reward_info=RewardInfo(reward=1),
        messages=[AssistantMessage(role="assistant", content="Jag kan hjälpa dig.")],
    )
    results = Results(info=info, tasks=[task], simulations=[simulation])
    path = tmp_path / "saved.json"
    results.save(path, format="json")
    before = path.read_bytes()
    calls = []
    monkeypatch.setattr(
        "tau2.judges.criteria.generate",
        lambda **kw: (
            calls.append(kw)
            or SimpleNamespace(
                content='{"opportunity":true,"violation":false,"evidence":"Fixture only."}'
            )
        ),
    )
    executed = run_candidate(path, compiled, tmp_path / "candidate-run")
    assert len(calls) == 1 and len(executed.predictions.rows) == 1
    assert executed.predictions.rows[0].language == "sv"
    assert path.read_bytes() == before and not progress_path(path).exists()


def test_original_new_diagnostic_resumes_after_registration(
    candidate, tmp_path, monkeypatch
):
    from tau2.judges.authoring_registry import register_judge

    monkeypatch.setattr("tau2.utils.DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(
        "tau2.judges.authoring.generate",
        lambda **kw: pytest.fail("Resume must not generate"),
    )
    output = tmp_path / "original-draft.json"
    output.write_text(candidate.authoring.model_dump_json())
    before = output.read_bytes()
    assert draft_judge(candidate.authoring.request, output) == candidate.authoring
    register_judge(candidate, tmp_path / "data/tau2/judge_catalog/original.json")
    assert draft_judge(candidate.authoring.request, output) == candidate.authoring
    assert output.read_bytes() == before
    with pytest.raises(ValueError, match="adaptation"):
        draft_judge(candidate.authoring.request, tmp_path / "new-draft.json")
    changed = candidate.authoring.request.model_copy(
        update={"agent_requirements": "Changed input"}
    )
    with pytest.raises(ValueError, match="Stale"):
        draft_judge(changed, output)
    # Existing unversioned or source-bearing new-diagnostic artifacts cannot bypass checks.
    bad = candidate.authoring.model_dump()
    bad["prompt_version"] = "judge-authoring-obsolete"
    output.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        draft_judge(candidate.authoring.request, output)
    bad = candidate.authoring.model_dump()
    bad["reference"] = {
        "source": "authoring_catalog",
        "source_revision": "fixture",
        "source_sha256": "a" * 64,
        "language": None,
        "criterion": candidate.contract.nativeness_criteria["ro"][0].model_dump(),
    }
    output.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="source-free"):
        draft_judge(candidate.authoring.request, output)
