"""Manifest provenance source audit: resolve, relocate, replace, miss.

Provenance records a run's IDENTITY next to its path, so these tests are
written in terms of runs rather than path strings: the same run under a new
path must be repaired, and a DIFFERENT run at the recorded path must be
caught — which no amount of path matching can do.
"""

import json
import shutil

import pytest

from tau2.annotation.provenance import BuildFilters, BuildRecord, RecordedRun
from tau2.annotation.source_audit import (
    SourceStatus,
    audit_sources,
    format_report,
)
from tau2.data_model.run_identity import compute_run_identity


def _write_run(path, *, timestamp="2026-07-20T00:00:00+00:00", sim_ids=("s1", "s2")):
    """A dir-format run: results.json metadata + one file per simulation."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "simulations").mkdir(exist_ok=True)
    (path / "results.json").write_text(
        json.dumps(
            {
                "timestamp": timestamp,
                "simulation_index": [
                    {"id": sim_id, "task_id": 1, "trial": 0} for sim_id in sim_ids
                ],
            }
        )
    )
    return path


def _run_dir(root, preset, run, **kwargs):
    return _write_run(root / preset / run, **kwargs)


def _identity(path):
    return compute_run_identity(path)


def _recorded(path, *, identity_from=None):
    """A build record source: the path, plus the identity of the run read.

    ``identity_from`` lets a test record a run's identity against a path the
    run no longer occupies (the whole point of the audit).
    """
    return RecordedRun(
        path=str(path), identity=compute_run_identity(identity_from or path)
    )


def _manifest(root, name, recorded, *, extra_builds=None):
    d = root / name
    d.mkdir(parents=True)
    builds = [
        BuildRecord(
            results=list(recorded),
            filters=BuildFilters(),
            timestamp="2026-07-23T00:00:00+00:00",
        )
    ]
    builds.extend(extra_builds or [])
    payload = {
        "kind": "voice_review",
        "batch_id": name,
        "provenance": {
            "builds": [b.model_dump(mode="json") for b in builds],
            "rubric_version": "3.0",
        },
    }
    (d / "manifest.json").write_text(json.dumps(payload))
    return d / "manifest.json"


def _builds(manifest):
    return json.loads(manifest.read_text())["provenance"]["builds"]


@pytest.fixture
def tree(tmp_path):
    sims = tmp_path / "simulations"
    anns = tmp_path / "annotations"
    sims.mkdir()
    anns.mkdir()
    return anns, sims


def test_untouched_run_resolves(tree):
    anns, sims = tree
    run = _run_dir(sims, "multilingual_v1_korean_telecom", "full_korean_telecom")
    _manifest(anns, "ko_review", [_recorded(run)])

    report = audit_sources(anns, sims)

    assert [c.status for c in report.checks] == [SourceStatus.RESOLVED]
    assert report.count(SourceStatus.RELOCATED) == 0


def test_a_source_outside_the_searched_tree_still_resolves(tmp_path, tree):
    """The recorded path is asked FIRST, not the index.

    A packet can be built from a run that does not live under
    --simulations-root at all (an archive volume, another tree). Resolving
    through the index alone would find no candidate and slander a perfectly
    healthy source as REPLACED.
    """
    anns, sims = tree
    outside = _write_run(tmp_path / "elsewhere" / "run_a")
    manifest = _manifest(anns, "m", [_recorded(outside)])

    report = audit_sources(anns, sims, apply=True)

    assert [c.status for c in report.checks] == [SourceStatus.RESOLVED]
    # Nothing to say means nothing written.
    assert "source_audit" not in json.loads(manifest.read_text())["provenance"]


def test_a_relative_and_absolute_spelling_of_the_same_run_resolve(tree, monkeypatch):
    """RESOLVED compares runs, not path strings."""
    anns, sims = tree
    run = _run_dir(sims, "preset_a", "run_a")
    monkeypatch.chdir(sims.parent)
    relative = run.relative_to(sims.parent)
    _manifest(anns, "m", [RecordedRun(path=str(relative), identity=_identity(run))])

    report = audit_sources(anns, sims)

    assert [c.status for c in report.checks] == [SourceStatus.RESOLVED]


def test_moved_run_is_found_by_identity(tree):
    """The tree was reorganised; the run is the same run."""
    anns, sims = tree
    original = _run_dir(sims, "multilingual_v1_korean_telecom", "full_korean_telecom")
    manifest = _manifest(anns, "ko_review", [_recorded(original)])
    moved = sims / "multilingual" / "multilingual_v1_korean_telecom"
    moved.parent.mkdir(parents=True)
    shutil.move(str(original.parent), str(moved))

    report = audit_sources(anns, sims)

    (check,) = report.checks
    assert check.status is SourceStatus.RELOCATED
    assert check.current_path == str(moved / "full_korean_telecom")
    # Reporting alone must not touch the manifest.
    assert _builds(manifest)[0]["results"][0]["path"] == str(original)


def test_moved_run_is_found_even_when_renamed(tree):
    """Identity does not depend on the run's name — only on what it holds."""
    anns, sims = tree
    original = _run_dir(sims, "preset_a", "run_a")
    manifest = _manifest(anns, "m", [_recorded(original)])
    renamed = sims / "archive" / "some_other_name"
    renamed.parent.mkdir(parents=True)
    shutil.move(str(original), str(renamed))

    audit_sources(anns, sims, apply=True)

    assert _builds(manifest)[0]["results"][0]["path"] == str(renamed)


def test_apply_rewrites_relocated_and_records_the_prior_path(tree):
    anns, sims = tree
    original = _run_dir(sims, "multilingual_v1_korean_telecom", "full_korean_telecom")
    manifest = _manifest(anns, "ko_review", [_recorded(original)])
    run_id = compute_run_identity(original).run_id
    moved = sims / "multilingual" / "multilingual_v1_korean_telecom"
    moved.parent.mkdir(parents=True)
    shutil.move(str(original.parent), str(moved))
    moved_run = moved / "full_korean_telecom"

    audit_sources(anns, sims, apply=True)

    provenance = json.loads(manifest.read_text())["provenance"]
    assert provenance["builds"][0]["results"][0]["path"] == str(moved_run)
    # The identity travels with the record — it is what made the repair safe.
    assert provenance["builds"][0]["results"][0]["identity"]["run_id"] == run_id
    (record,) = provenance["source_audit"]
    assert record["repaired"] == [
        {
            "run_id": run_id,
            "previous_path": str(original),
            "current_path": str(moved_run),
        }
    ]
    assert record["losses"] == []
    assert record["git_sha"] and record["created_at"]
    # The rest of the manifest survives the rewrite.
    assert provenance["rubric_version"] == "3.0"


def test_relocation_keeps_a_recorded_results_json_shape(tree):
    """Matrix builds record the results.json FILE; the repair must keep it."""
    anns, sims = tree
    original = _run_dir(sims, "preset_a", "run_a")
    manifest = _manifest(anns, "m", [_recorded(original / "results.json")])
    moved = sims / "nested" / "run_a"
    moved.parent.mkdir(parents=True)
    shutil.move(str(original), str(moved))

    audit_sources(anns, sims, apply=True)

    assert _builds(manifest)[0]["results"][0]["path"] == str(moved / "results.json")


def test_deleted_run_is_recorded_missing_not_rewritten(tree):
    anns, sims = tree
    gone = _run_dir(sims, "multilingual_v1_arabic_telecom", "smoke_arabic_telecom")
    manifest = _manifest(anns, "ar_review", [_recorded(gone)])
    shutil.rmtree(gone)

    report = audit_sources(anns, sims, apply=True)

    (check,) = report.checks
    assert check.status is SourceStatus.MISSING
    assert check.current_path is None
    provenance = json.loads(manifest.read_text())["provenance"]
    # The path it actually read stays put; the loss is recorded beside it.
    assert provenance["builds"][0]["results"][0]["path"] == str(gone)
    (loss,) = provenance["source_audit"][0]["losses"]
    assert (loss["status"], loss["path"]) == ("missing", str(gone))
    assert provenance["source_audit"][0]["repaired"] == []


def test_regenerated_run_at_the_same_path_is_detected(tree):
    """THE case a path audit cannot see: same path, different run.

    The packet was built from one run; that run was deleted and re-executed
    to the same --save-to. The path resolves, so path matching calls it fine.
    """
    anns, sims = tree
    run = _run_dir(sims, "preset_a", "run_a", sim_ids=("s1", "s2"))
    manifest = _manifest(anns, "m", [_recorded(run)])
    original_id = compute_run_identity(run).run_id
    shutil.rmtree(run)
    _write_run(run, timestamp="2026-07-25T00:00:00+00:00", sim_ids=("s9", "s10"))
    new_id = compute_run_identity(run).run_id
    assert new_id != original_id

    report = audit_sources(anns, sims, apply=True)

    (check,) = report.checks
    assert check.status is SourceStatus.REPLACED
    assert (check.recorded_run_id, check.found_run_id) == (original_id, new_id)
    provenance = json.loads(manifest.read_text())["provenance"]
    # A replaced source is never rewritten — there is nothing truthful to
    # rewrite it to; the manifest keeps the path it read and names the loss.
    assert provenance["builds"][0]["results"][0]["path"] == str(run)
    assert provenance["builds"][0]["results"][0]["identity"]["run_id"] == original_id
    (loss,) = provenance["source_audit"][0]["losses"]
    assert (loss["status"], loss["run_id"], loss["found_run_id"]) == (
        "replaced",
        original_id,
        new_id,
    )
    assert "REPLACED" in format_report(report)


def test_a_run_regenerated_in_place_while_the_original_survives_relocates(tree):
    """Identity-first: the run that was READ wins over the path that was read."""
    anns, sims = tree
    run = _run_dir(sims, "preset_a", "run_a", sim_ids=("s1", "s2"))
    manifest = _manifest(anns, "m", [_recorded(run)])
    archived = sims / "archive" / "run_a"
    archived.parent.mkdir(parents=True)
    shutil.copytree(run, archived)
    shutil.rmtree(run)
    _write_run(run, timestamp="2026-07-25T00:00:00+00:00", sim_ids=("s9",))

    report = audit_sources(anns, sims, apply=True)

    (check,) = report.checks
    assert check.status is SourceStatus.RELOCATED
    assert _builds(manifest)[0]["results"][0]["path"] == str(archived)


def test_apply_is_idempotent(tree):
    """A second pass has nothing new to say, so it must write nothing."""
    anns, sims = tree
    original = _run_dir(sims, "multilingual_v1_korean_telecom", "full_korean_telecom")
    gone = _run_dir(sims, "multilingual_v1_arabic_telecom", "smoke_arabic_telecom")
    manifest = _manifest(anns, "ko_review", [_recorded(original), _recorded(gone)])
    shutil.rmtree(gone)
    moved = sims / "multilingual" / "multilingual_v1_korean_telecom"
    moved.parent.mkdir(parents=True)
    shutil.move(str(original.parent), str(moved))

    audit_sources(anns, sims, apply=True)
    first = manifest.read_text()
    audit_sources(anns, sims, apply=True)

    # Byte-identical: no second record, no fresh timestamp.
    assert manifest.read_text() == first
    assert len(json.loads(first)["provenance"]["source_audit"]) == 1


def test_apply_is_idempotent_over_a_replaced_source(tree):
    """A permanent replacement must not append an identical record forever."""
    anns, sims = tree
    run = _run_dir(sims, "preset_a", "run_a")
    manifest = _manifest(anns, "m", [_recorded(run)])
    shutil.rmtree(run)
    _write_run(run, timestamp="2026-07-25T00:00:00+00:00", sim_ids=("s9",))

    audit_sources(anns, sims, apply=True)
    first = manifest.read_text()
    audit_sources(anns, sims, apply=True)

    assert manifest.read_text() == first


def test_a_second_replacement_is_recorded(tree):
    """Idempotence keys on WHAT was found, so a fresh replacement is seen."""
    anns, sims = tree
    run = _run_dir(sims, "preset_a", "run_a")
    manifest = _manifest(anns, "m", [_recorded(run)])
    shutil.rmtree(run)
    _write_run(run, timestamp="2026-07-25T00:00:00+00:00", sim_ids=("s9",))
    audit_sources(anns, sims, apply=True)

    shutil.rmtree(run)
    _write_run(run, timestamp="2026-07-26T00:00:00+00:00", sim_ids=("s11",))
    audit_sources(anns, sims, apply=True)

    records = json.loads(manifest.read_text())["provenance"]["source_audit"]
    assert len(records) == 2
    assert (
        records[0]["losses"][0]["found_run_id"]
        != (records[1]["losses"][0]["found_run_id"])
    )


def test_a_later_loss_appends_one_more_record(tree):
    """Idempotence must not swallow a source that goes missing afterwards."""
    anns, sims = tree
    doomed = _run_dir(sims, "preset_a", "run_a")
    already_gone = _run_dir(sims, "preset_b", "already_gone", sim_ids=("z1",))
    manifest = _manifest(anns, "m", [_recorded(doomed), _recorded(already_gone)])
    shutil.rmtree(already_gone)

    audit_sources(anns, sims, apply=True)
    shutil.rmtree(doomed)
    audit_sources(anns, sims, apply=True)

    records = json.loads(manifest.read_text())["provenance"]["source_audit"]
    assert len(records) == 2
    assert [loss["path"] for loss in records[0]["losses"]] == [str(already_gone)]
    # Only the newly-observed loss, not a restatement of the first.
    assert [loss["path"] for loss in records[1]["losses"]] == [str(doomed)]


def test_ambiguous_identity_is_not_guessed(tree):
    """Two copies of the same run — repairing would pick one blind."""
    anns, sims = tree
    original = _run_dir(sims, "multilingual_v1_hindi_telecom", "full_hindi_telecom")
    manifest = _manifest(anns, "hi_review", [_recorded(original)])
    for copy in ("a", "b"):
        shutil.copytree(original, sims / copy / "full_hindi_telecom")
    shutil.rmtree(original)

    report = audit_sources(anns, sims, apply=True)

    (check,) = report.checks
    assert check.status is SourceStatus.AMBIGUOUS
    assert sorted(check.candidates) == [
        str(sims / "a" / "full_hindi_telecom"),
        str(sims / "b" / "full_hindi_telecom"),
    ]
    assert _builds(manifest)[0]["results"][0]["path"] == str(original)


def test_a_different_run_at_the_same_relative_path_does_not_match(tree):
    """<preset>/<run> is not identity: a same-named run is a different run."""
    anns, sims = tree
    original = _run_dir(sims, "multilingual_v1_korean_telecom", "full_run")
    _manifest(anns, "ko_review", [_recorded(original)])
    shutil.rmtree(original)
    _run_dir(
        sims / "multilingual",
        "multilingual_v1_korean_telecom",
        "full_run",
        timestamp="2026-07-25T00:00:00+00:00",
        sim_ids=("other",),
    )

    report = audit_sources(anns, sims)

    assert report.checks[0].status is SourceStatus.MISSING


def test_every_build_and_source_is_checked(tree):
    """--append packets carry several builds; all of them get audited."""
    anns, sims = tree
    live = _run_dir(sims, "preset_a", "run_a")
    gone_b = _run_dir(sims, "preset_b", "run_b", sim_ids=("b1",))
    gone_c = _run_dir(sims, "preset_c", "run_c", sim_ids=("c1",))
    extra = BuildRecord(
        results=[_recorded(gone_c)],
        filters=BuildFilters(),
        timestamp="2026-07-24T00:00:00+00:00",
    )
    _manifest(
        anns,
        "appended",
        [_recorded(live), _recorded(gone_b)],
        extra_builds=[extra],
    )
    shutil.rmtree(gone_b)
    shutil.rmtree(gone_c)

    report = audit_sources(anns, sims)

    assert len(report.checks) == 3
    assert [c.build_index for c in report.checks] == [0, 0, 1]
    assert report.count(SourceStatus.RESOLVED) == 1
    assert report.count(SourceStatus.MISSING) == 2


def test_report_is_serialisable_and_summarised(tree):
    anns, sims = tree
    live = _run_dir(sims, "preset_a", "run_a")
    gone = _run_dir(sims, "p", "gone", sim_ids=("g1",))
    _manifest(anns, "m", [_recorded(live), _recorded(gone)])
    shutil.rmtree(gone)

    report = audit_sources(anns, sims)

    assert json.loads(report.model_dump_json())["applied"] is False
    summary = format_report(report)
    assert "resolved  1" in summary
    assert "missing   1" in summary


def test_manifest_without_builds_is_skipped(tree):
    anns, sims = tree
    d = anns / "sheet_family"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"provenance": {"rubric": "x"}}))

    report = audit_sources(anns, sims, apply=True)

    assert report.checks == []
    # Nothing to repair means nothing written.
    assert (
        "source_audit"
        not in json.loads((d / "manifest.json").read_text())["provenance"]
    )


def test_one_unreadable_run_does_not_abort_the_sweep(tree):
    """Fail loud and LOCAL: a hand-edited results.json under the run tree is
    skipped with a warning, not allowed to kill the whole audit."""
    anns, sims = tree
    live = _run_dir(sims, "preset_a", "run_a")
    broken = sims / "preset_b" / "broken"
    broken.mkdir(parents=True)
    (broken / "results.json").write_text('{"simulation_index": [{"task_id": 1}]}')
    truncated = sims / "preset_c" / "truncated"
    truncated.mkdir(parents=True)
    (truncated / "results.json").write_text("{not json")
    _manifest(anns, "m", [_recorded(live)])

    report = audit_sources(anns, sims, apply=True)

    assert [c.status for c in report.checks] == [SourceStatus.RESOLVED]


def test_a_recorded_path_holding_unreadable_metadata_is_replaced(tree):
    """The source is gone even though something is still sitting there."""
    anns, sims = tree
    run = _run_dir(sims, "preset_a", "run_a")
    manifest = _manifest(anns, "m", [_recorded(run)])
    (run / "results.json").write_text("{not json")

    report = audit_sources(anns, sims, apply=True)

    (check,) = report.checks
    assert check.status is SourceStatus.REPLACED
    assert check.found_run_id is None
    (loss,) = json.loads(manifest.read_text())["provenance"]["source_audit"][0][
        "losses"
    ]
    assert loss["found_run_id"] is None


def test_a_manifest_that_does_not_validate_is_reported_not_fatal(tree):
    """Packets are regenerable: a stale provenance shape is a loud rebuild,
    and it must not abort the sweep over every other manifest."""
    anns, sims = tree
    live = _run_dir(sims, "preset_a", "run_a")
    _manifest(anns, "good", [_recorded(live)])
    stale = anns / "stale"
    stale.mkdir()
    # The pre-identity shape: a bare list of path strings.
    (stale / "manifest.json").write_text(
        json.dumps({"provenance": {"builds": [{"results": [str(live)]}]}})
    )

    report = audit_sources(anns, sims, apply=True)

    assert [c.status for c in report.checks] == [SourceStatus.RESOLVED]
    (problem,) = report.unreadable
    assert problem.manifest == str(stale / "manifest.json")
    assert "UNREADABLE" in format_report(report)
    # An unreadable manifest is left exactly as found.
    assert "source_audit" not in (stale / "manifest.json").read_text()
