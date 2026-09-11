# Copyright Sierra
"""The ``tau2 judges`` subcommand surface.

Post-hoc judging over stored results. Nativeness judges from the stored
transcripts; delivery judges from the stored AUDIO — the in-memory
``audio_content`` the inline judge consumes is never serialized, but
full-duplex runs save a stereo ``both.wav`` with the agent on the right
channel, and ``delivery.disk_audio`` slices it per utterance by tick span.
Half-duplex runs have no tick-aligned timeline and are unsupported loudly.

- ``tau2 judges rejudge``  — run the nativeness judge (and, with
                             ``--delivery``, the delivery judge from disk
                             audio) over stored results, reusing complete
                             stored verdicts and filling gaps by default
                             (``--rejudge`` forces a full re-judge), and
                             write the updated results back in their
                             original storage format.
- ``tau2 judges export``   — emit typed JSONL verdict records (the judges →
                             annotation interface). Any verdicts it had to
                             pay for are persisted back to the results so the
                             cost is never wasted.
- ``tau2 judges suite``    — one-pass annotation-calibrated judging
                             (nativeness + quality + delivery) over stored
                             results: a multiprocess text phase then an audio
                             phase, gap-filling by default, trial 0 only by
                             default, with a typed suite-level provenance
                             record per results root.
- ``tau2 judges tau-multi-naturalness`` — prepare or replay the exact combined
                             judge over its canonical paper cohort.
- ``tau2 judges tau-multi-validation`` — verify the canonical nested human-
                             annotation archive without external calls.
- ``tau2 judges legacy``   — demoted judge verbs (``conversation``,
                             ``quality``): fully functional, explicitly
                             invoked, out of the standard judging flow.
"""

import argparse
import json
from pathlib import Path

from tau2.config import (
    DEFAULT_JUDGE_STREAM_CONCURRENCY,
    DEFAULT_JUDGE_SUITE_AUDIO_PROCESSES,
    DEFAULT_JUDGE_SUITE_TEXT_PROCESSES,
    DEFAULT_REJUDGE_DELIVERY_SAMPLE_RATE,
    DEFAULT_TAU_MULTI_NATURALNESS_CONCURRENCY,
)


def add_judges_args(parser: argparse.ArgumentParser) -> None:
    """Attach the judges sub-subcommands to the ``tau2 judges`` parser."""
    judges_sub = parser.add_subparsers(
        dest="judges_command", help="Judges commands", required=True
    )

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "results",
        nargs="+",
        help="Results dir(s) or results.json file(s)",
    )
    common.add_argument(
        "--rejudge",
        action="store_true",
        help="Force a full re-judge of every sim. Default: reuse stored "
        "nativeness verdicts when they are complete (every judge factor "
        "present with a usable, non-DEFERRED/non-ERROR outcome) AND from the "
        "current judge prompt version; judge only the gaps and stale sims.",
    )
    common.add_argument(
        "--judge-model",
        default=None,
        help="Model id for the nativeness judge (default: config default)",
    )
    common.add_argument(
        "--judge-args",
        default=None,
        help="JSON dict of extra args for each judge call, e.g. "
        '\'{"reasoning_effort": "low"}\' (default: config default)',
    )
    common.add_argument(
        "--max-concurrency",
        type=int,
        default=DEFAULT_JUDGE_STREAM_CONCURRENCY,
        help="Concurrent judge calls (I/O-bound); default "
        f"{DEFAULT_JUDGE_STREAM_CONCURRENCY}",
    )

    rejudge_parser = judges_sub.add_parser(
        "rejudge",
        parents=[common],
        help="Run the nativeness judge over stored results (filling verdict "
        "gaps by default; --rejudge forces a full re-judge) and save the "
        "updated results in their original format",
    )
    rejudge_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Where to save the re-judged results (single input only). "
        "Default: overwrite each input in place.",
    )
    rejudge_parser.add_argument(
        "--delivery",
        action="store_true",
        help="Also run the delivery judge from disk audio (both.wav right "
        "channel sliced per utterance) on full-duplex sims lacking stored "
        "delivery verdicts (--rejudge re-judges all).",
    )
    rejudge_parser.add_argument(
        "--delivery-only",
        action="store_true",
        help="Stream stored nativeness verdicts through untouched (never "
        "re-judged) so only the delivery axis is (re-)run; requires "
        "--delivery. Combine with --rejudge for a forced delivery-only "
        "re-judge.",
    )
    rejudge_parser.add_argument(
        "--delivery-sample-rate",
        type=float,
        default=DEFAULT_REJUDGE_DELIVERY_SAMPLE_RATE,
        help="Fraction of conversations the delivery judge covers (default "
        f"{DEFAULT_REJUDGE_DELIVERY_SAMPLE_RATE} — backfill judges everything)",
    )
    rejudge_parser.add_argument(
        "--delivery-model",
        default=None,
        help="Model id for the delivery judge (default: config default)",
    )
    rejudge_parser.set_defaults(func=run_judges_rejudge)

    adopt_parser = judges_sub.add_parser(
        "adopt-verdicts",
        help="Copy current-prompt-version judge verdicts from judged sims "
        "under --from into results whose verdicts are stale or missing, "
        "matched by sim id (never invokes a judge). Use before rejudge so a "
        "corpus inherits verdicts already paid for on a materialized subset.",
    )
    adopt_parser.add_argument(
        "--from",
        dest="src_root",
        type=Path,
        required=True,
        help="Root scanned recursively for dir-format sims "
        "(simulations/<sim_id>.json) to adopt verdicts from",
    )
    adopt_parser.add_argument(
        "results",
        nargs="+",
        help="Destination results dir(s) or results.json file(s), updated in place",
    )
    adopt_parser.set_defaults(func=run_judges_adopt_verdicts)

    recheck_gender = judges_sub.add_parser(
        "recheck-gender",
        help="Re-derive ONLY the deterministic gender_agreement verdicts on "
        "stored runs with the current checker version (no LLM calls): "
        "snapshots the prior verdicts per root, patches precheck-FAIL sims "
        "in place, recomputes the nativeness aggregates, and prints a "
        "before/after fail-count summary. Idempotent — a re-run changes "
        "nothing and writes no snapshot.",
    )
    recheck_gender.add_argument(
        "roots",
        nargs="+",
        type=Path,
        help="Run root dir(s) containing dir-format cells "
        "(<root>/<cell>/simulations/*.json)",
    )
    recheck_gender.set_defaults(func=run_judges_recheck_gender)

    export_parser = judges_sub.add_parser(
        "export",
        parents=[common],
        help="Export typed JSONL verdict records (quality + nativeness + "
        "delivery + factor rubrics) from results; verdicts judged along the way are "
        "persisted back to the results",
    )
    export_parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for quality_verdicts.jsonl, "
        "nativeness_verdicts.jsonl, delivery_verdicts.jsonl and factor_rubrics.jsonl",
    )
    export_parser.set_defaults(func=run_judges_export)

    lexical = judges_sub.add_parser(
        "lexical-stats",
        help="Deterministic run-level lexical-diversity report (MATTR over "
        "pooled agent tokens); within-language comparisons only",
    )
    lexical.add_argument("results", nargs="+", help="Results dir(s) or JSON file(s)")
    lexical.add_argument(
        "--window",
        type=int,
        default=None,
        help="MATTR window in tokens (default: module default)",
    )
    lexical.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write the JSON report(s) here instead of stdout",
    )
    lexical.set_defaults(func=run_judges_lexical_stats)

    suite = judges_sub.add_parser(
        "suite",
        help="One-pass annotation-calibrated judging (nativeness + quality + "
        "delivery) over stored results: text phase then audio phase, "
        "multiprocess, gap-filling by default, trial 0 only by default; "
        "writes a suite provenance record per results root",
    )
    suite.add_argument(
        "results",
        nargs="+",
        help="Results dir(s) or results.json file(s) — paths only",
    )
    suite.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_JUDGE_STREAM_CONCURRENCY,
        help="Per-process simulations in flight (I/O-bound judge calls); "
        f"default {DEFAULT_JUDGE_STREAM_CONCURRENCY}",
    )
    suite.add_argument(
        "--text-processes",
        type=int,
        default=DEFAULT_JUDGE_SUITE_TEXT_PROCESSES,
        help="Worker processes for the nativeness + quality phase; default "
        f"{DEFAULT_JUDGE_SUITE_TEXT_PROCESSES}",
    )
    suite.add_argument(
        "--audio-processes",
        type=int,
        default=DEFAULT_JUDGE_SUITE_AUDIO_PROCESSES,
        help="Worker processes for the delivery phase (audio payloads are "
        f"heavier); default {DEFAULT_JUDGE_SUITE_AUDIO_PROCESSES}",
    )
    suite.add_argument(
        "--rejudge",
        action="store_true",
        help="Force a full re-judge instead of filling verdict gaps",
    )
    suite.add_argument(
        "--only",
        default=None,
        help="Comma-separated judge families to run: any of "
        "nativeness,quality,delivery (default: all three)",
    )
    suite.add_argument(
        "--trials",
        default="0",
        help="Comma-separated trial indices to judge, or 'all' for every "
        "trial; sims outside the selection are not judged at all (default: 0 "
        "— judge one trial per task)",
    )
    suite.set_defaults(func=run_judges_suite)

    paper_naturalness = judges_sub.add_parser(
        "tau-multi-naturalness",
        help="Frozen combined-utterance-naturalness validation and trial-0 replay",
    )
    paper_naturalness_sub = paper_naturalness.add_subparsers(
        dest="tau_multi_naturalness_command", required=True
    )
    paper_prepare = paper_naturalness_sub.add_parser(
        "prepare",
        help="Offline preflight: hash all 90 roots and report the exact trial-0 work",
    )
    paper_prepare.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository containing the frozen Experience and validation artifacts",
    )
    paper_prepare.add_argument(
        "--evidence-root",
        type=Path,
        help="Detached frozen tau-multi result bundle (contains main_runs/)",
    )
    paper_prepare.set_defaults(func=prepare_tau_multi_naturalness_trial)

    paper_run = paper_naturalness_sub.add_parser(
        "run",
        help="Run/resume the frozen judge over the exact Experience-manifest "
        "trial-0 cohort without modifying source results",
    )
    paper_run.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository containing the frozen Experience and validation artifacts",
    )
    paper_run.add_argument(
        "--evidence-root",
        type=Path,
        help="Detached frozen tau-multi result bundle (contains main_runs/)",
    )
    paper_run.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Separate output root for incremental utterance/call artifacts",
    )
    paper_run.add_argument(
        "--max-concurrency",
        type=int,
        default=DEFAULT_TAU_MULTI_NATURALNESS_CONCURRENCY,
        help="Maximum in-flight text judge calls (default: "
        f"{DEFAULT_TAU_MULTI_NATURALNESS_CONCURRENCY})",
    )
    paper_run.set_defaults(func=run_tau_multi_naturalness_trial)

    validation_archive = judges_sub.add_parser(
        "tau-multi-validation",
        help="Offline verification of the canonical human-annotation archive",
    )
    validation_archive.add_argument(
        "root", type=Path, help="Human-annotation archive root"
    )
    validation_archive.set_defaults(func=validate_tau_multi_validation_archive)

    legacy_parser = judges_sub.add_parser(
        "legacy",
        help="Demoted judge verbs (conversation, quality) — fully usable, "
        "explicitly invoked, out of the standard judging flow",
    )
    add_judges_legacy_args(legacy_parser)


def add_judges_legacy_args(parser: argparse.ArgumentParser) -> None:
    """Attach the legacy sub-subcommands to the ``tau2 judges legacy`` parser.

    Demoted verbs: fully functional, explicitly invoked, out of the standard
    judging flow. ``conversation`` (the EVA-ported conversation-judge suite;
    composites in shadow_scores only) and ``quality`` (the universal binary
    quality rubric) live here; the per-utterance delivery judge — the
    canonical audio judge — runs through ``rejudge --delivery``.
    """
    legacy_sub = parser.add_subparsers(
        dest="judges_legacy_command", help="Legacy judges commands", required=True
    )

    from tau2.judges.conversation.cli import add_conversation_args

    add_conversation_args(legacy_sub)

    quality = legacy_sub.add_parser(
        "quality",
        help="Compute the universal binary quality rubric over stored results",
    )
    quality.add_argument("results", nargs="+", help="Results dir(s) or JSON file(s)")
    quality.add_argument(
        "--llm-judge",
        action="store_true",
        help="Also run the task-aware semantic factors; deterministic-only by default",
    )
    quality.add_argument("--model", default=None, help="Semantic judge model override")
    quality.add_argument("--model-args", default=None, help="Semantic judge JSON args")
    quality.add_argument("-o", "--output", type=Path, default=None)
    quality.set_defaults(func=run_judges_quality)


def _judge_settings(args):
    """Build NativenessJudgeSettings from CLI overrides (config defaults else)."""
    from tau2.data_model.simulation import NativenessJudgeSettings

    kwargs: dict = {"llm_judge": True}
    if args.judge_model:
        kwargs["model"] = args.judge_model
    if args.judge_args:
        model_args = json.loads(args.judge_args)
        if not isinstance(model_args, dict):
            raise SystemExit("--judge-args must be a JSON object")
        kwargs["model_args"] = model_args
    return NativenessJudgeSettings(**kwargs)


def _delivery_settings(args):
    """Build DeliveryJudgeSettings from rejudge CLI overrides (None w/o --delivery)."""
    from tau2.data_model.simulation import DeliveryJudgeSettings

    if not getattr(args, "delivery", False):
        return None
    kwargs: dict = {"sample_rate": args.delivery_sample_rate}
    if args.delivery_model:
        kwargs["model"] = args.delivery_model
    return DeliveryJudgeSettings(**kwargs)


def parse_suite_families(value):
    """Parse ``--only`` into a family set (None => all three)."""
    from tau2.judges.suite import ALL_FAMILIES, SuiteFamily

    if value is None:
        return ALL_FAMILIES
    names = [name.strip() for name in value.split(",") if name.strip()]
    try:
        families = frozenset(SuiteFamily(name) for name in names)
    except ValueError:
        valid = ",".join(sorted(f.value for f in SuiteFamily))
        raise SystemExit(f"--only takes a comma-separated subset of: {valid}")
    if not families:
        raise SystemExit("--only must select at least one judge family")
    return families


def parse_suite_trials(value):
    """Parse ``--trials`` into a trial-index set (None => every trial)."""
    if value.strip().lower() == "all":
        return None
    try:
        trials = frozenset(int(part) for part in value.split(",") if part.strip())
    except ValueError:
        raise SystemExit("--trials takes comma-separated trial indices, or 'all'")
    if not trials:
        raise SystemExit("--trials must select at least one trial, or 'all'")
    return trials


def run_judges_suite(args) -> None:
    """Run the annotation-calibrated judge set (see ``tau2.judges.suite``)."""
    from tau2.judges.suite import SuiteConfig, run_suite

    config = SuiteConfig(
        results=[Path(path) for path in args.results],
        concurrency=args.concurrency,
        text_processes=args.text_processes,
        audio_processes=args.audio_processes,
        rejudge=args.rejudge,
        only=parse_suite_families(args.only),
        trials=parse_suite_trials(args.trials),
    )
    for record in run_suite(config):
        print(f"suite -> {record.results_path}")
        for family, prov in record.families.items():
            touches = prov.touches
            line = (
                f"  {family}: {len(touches.judged)} judged, "
                f"{len(touches.reused)} reused, {len(touches.skipped)} skipped"
            )
            if touches.unavailable:
                line += f", {len(touches.unavailable)} unavailable"
            print(line)


def run_tau_multi_naturalness_trial(args) -> None:
    """Run/resume the canonical non-English trial-0 combined judge."""
    from tau2.judges.nativeness.paper_trial import (
        TrialRunConfig,
        run_trial0_naturalness,
    )

    manifest = run_trial0_naturalness(
        TrialRunConfig(
            repo_root=args.repo_root,
            evidence_root=args.evidence_root,
            output_root=args.out,
            max_concurrency=args.max_concurrency,
        )
    )
    print(manifest.model_dump_json(indent=2))


def prepare_tau_multi_naturalness_trial(args) -> None:
    """Verify and size the canonical trial-0 replay without LLM calls."""
    from tau2.judges.nativeness.paper_trial import prepare_trial0_naturalness

    report = prepare_trial0_naturalness(
        args.repo_root, evidence_root=args.evidence_root
    )
    print(report.model_dump_json(indent=2))


def validate_tau_multi_validation_archive(args) -> None:
    """Validate the frozen human-annotation archive without external calls."""
    from tau2.judges.validation_archive import validate_archive

    report = validate_archive(args.root)
    print(report.model_dump_json(indent=2))


def run_judges_quality(args) -> None:
    """Compute and persist the universal quality axis over stored results."""
    from tau2.data_model.simulation import QualityJudgeSettings, Score
    from tau2.scripts.evaluate_trajectories import rescore_results

    if args.output is not None and len(args.results) > 1:
        raise SystemExit("--output only works with a single results input")
    settings_data: dict = {"llm_judge": args.llm_judge}
    if args.model:
        settings_data["model"] = args.model
    if args.model_args:
        model_args = json.loads(args.model_args)
        if not isinstance(model_args, dict):
            raise SystemExit("--model-args must be a JSON object")
        settings_data["model_args"] = model_args
    settings = QualityJudgeSettings(**settings_data)
    for path in args.results:
        results, _stats = rescore_results(
            Path(path),
            scores={Score.QUALITY},
            quality_judge=settings,
            output=args.output,
        )
        print(f"quality -> {args.output or path} ({len(results.simulations)} calls)")


def run_judges_lexical_stats(args) -> None:
    """Print (or save) the deterministic lexical-diversity report per input."""
    from tau2.data_model.simulation import Results
    from tau2.metrics.lexical_diversity import (
        DEFAULT_MATTR_WINDOW,
        compute_lexical_diversity,
    )

    window = args.window or DEFAULT_MATTR_WINDOW
    reports = {}
    for path in args.results:
        report = compute_lexical_diversity(Results.load(Path(path)), window=window)
        reports[str(path)] = report.model_dump()
    payload = json.dumps(reports, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.write_text(payload)
        print(f"lexical-stats -> {args.output}")
    else:
        print(payload)


def run_judges_rejudge(args) -> None:
    from loguru import logger

    from tau2.data_model.simulation import SimulationRun
    from tau2.judges.export import (
        JudgeStreamStats,
        iter_judged_sims_detailed,
        save_judged_results,
    )

    if args.output is not None and len(args.results) > 1:
        raise SystemExit("--output only works with a single results input")

    settings = _judge_settings(args)
    delivery_settings = _delivery_settings(args)
    delivery_only = getattr(args, "delivery_only", False)
    if delivery_only and delivery_settings is None:
        raise SystemExit("--delivery-only requires --delivery")
    for path in args.results:
        path = Path(path)
        logger.info(f"rejudging {path} ...")
        stats = JudgeStreamStats()
        judged: dict[str, SimulationRun] = {}
        for item in iter_judged_sims_detailed(
            path,
            reuse_existing=not args.rejudge,
            concurrency=args.max_concurrency,
            settings=settings,
            delivery_settings=delivery_settings,
            judge_nativeness=not delivery_only,
            stats=stats,
        ):
            judged[item.sim.id] = item.sim
        out = save_judged_results(path, judged, output=args.output)
        logger.info(
            f"rejudge {path}: {stats.judged} judged, {stats.reused} reused, "
            f"{stats.skipped} skipped -> {out}"
        )
        if delivery_settings is not None:
            logger.info(
                f"rejudge {path}: delivery {stats.delivery_judged} judged, "
                f"{stats.delivery_reused} reused, "
                f"{stats.delivery_unavailable} unavailable"
            )
        if stats.skipped:
            logger.warning(
                f"rejudge {path}: skipped sims (kept as stored): "
                f"{stats.skipped_sim_ids}"
            )
        if stats.delivery_unavailable:
            logger.warning(
                f"rejudge {path}: delivery unavailable (no/mono both.wav or "
                f"timeline mismatch): {stats.delivery_unavailable_sim_ids}"
            )


def run_judges_recheck_gender(args) -> None:
    """Deterministically re-derive gender_agreement verdicts on stored runs."""
    from loguru import logger

    from tau2.judges.nativeness.gender_recheck import recheck_gender_root

    for root in args.roots:
        report = recheck_gender_root(Path(root))
        logger.info(
            f"recheck-gender {report.root}: {report.cells} cell(s), "
            f"{report.sims_scanned} sim(s) with a stored gender check, "
            f"{report.candidates} precheck-FAIL candidate(s); "
            f"gender fails {report.fails_before} -> {report.fails_after}, "
            f"{report.patched} sim file(s) patched"
        )
        if report.skipped:
            logger.warning(
                f"recheck-gender {report.root}: {report.skipped} candidate(s) "
                "skipped (missing task / unresolvable language / no hybrid "
                "gender factor) — left as stored"
            )
        if report.snapshot_path:
            logger.info(
                f"recheck-gender {report.root}: prior verdicts snapshotted to "
                f"{report.snapshot_path}"
            )
        else:
            logger.info(
                f"recheck-gender {report.root}: nothing to patch; no snapshot written"
            )


def run_judges_adopt_verdicts(args) -> None:
    from loguru import logger

    from tau2.judges.export import adopt_stored_verdicts

    for path in args.results:
        stats = adopt_stored_verdicts(args.src_root, Path(path))
        logger.info(
            f"adopt-verdicts {path}: {stats.matched}/{stats.dest_sims} matched, "
            f"{stats.nativeness_adopted} nativeness adopted, "
            f"{stats.delivery_adopted} delivery adopted"
        )


def run_judges_export(args) -> None:
    from loguru import logger

    from tau2.data_model.simulation import Results, SimulationRun
    from tau2.judges.export import (
        DeliveryVerdict,
        JudgeStreamStats,
        NativenessVerdict,
        QualityVerdict,
        bucket_rollup,
        delivery_verdicts,
        factor_rubrics_for,
        iter_judged_sims_detailed,
        nativeness_verdicts,
        quality_verdicts,
        render_bucket_rollup,
        save_judged_results,
        write_verdicts_jsonl,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    settings = _judge_settings(args)
    languages: set[str] = set()
    nat_records: list[NativenessVerdict] = []
    del_records: list[DeliveryVerdict] = []
    quality_records: list[QualityVerdict] = []
    stats = JudgeStreamStats()

    # One streaming pass over the (large, audio-bearing) sims per input; only
    # the small typed records — plus any sims the judge actually (re-)ran on,
    # which must be persisted back so the paid verdicts are never wasted —
    # are held in memory.
    for path in args.results:
        path = Path(path)
        logger.info(f"loading {path} ...")
        judged: dict[str, SimulationRun] = {}
        for item in iter_judged_sims_detailed(
            path,
            reuse_existing=not args.rejudge,
            concurrency=args.max_concurrency,
            settings=settings,
            stats=stats,
        ):
            sim = item.sim
            if sim.nativeness_info is not None and sim.nativeness_info.language:
                languages.add(sim.nativeness_info.language)
            nat_records.extend(nativeness_verdicts(sim))
            del_records.extend(delivery_verdicts(sim))
            if item.was_judged:
                judged[sim.id] = sim
        if judged:
            saved = save_judged_results(path, judged)
            logger.info(f"persisted {len(judged)} freshly judged sims back to {saved}")
        for stored_sim in Results.iter_simulations(path):
            quality_records.extend(quality_verdicts(stored_sim))

    # Deterministic artifacts: rows sorted by a stable key, not judge
    # completion order.
    nat_records.sort(key=lambda r: (r.sim_id, r.factor_id))
    del_records.sort(key=lambda r: (r.sim_id, r.utterance_idx))
    quality_records.sort(key=lambda r: (r.sim_id, r.factor_id))
    rubrics = [r for lang in sorted(languages) for r in factor_rubrics_for(lang)]
    rubrics.sort(key=lambda r: (r.language, r.factor_id))

    write_verdicts_jsonl(nat_records, out_dir / "nativeness_verdicts.jsonl")
    write_verdicts_jsonl(del_records, out_dir / "delivery_verdicts.jsonl")
    write_verdicts_jsonl(quality_records, out_dir / "quality_verdicts.jsonl")
    write_verdicts_jsonl(rubrics, out_dir / "factor_rubrics.jsonl")

    rollup = bucket_rollup(nat_records, quality_records, del_records)
    write_verdicts_jsonl(rollup, out_dir / "bucket_rollup.jsonl")
    logger.info("calibration-bucket rollup (call-level any-FAIL):")
    for line in render_bucket_rollup(rollup).splitlines():
        logger.info(line)

    logger.info(
        f"export: {stats.judged} judged, {stats.reused} reused, "
        f"{stats.skipped} skipped -> {out_dir}"
    )
    if stats.skipped:
        logger.warning(
            f"export: skipped sims (no verdicts emitted): {stats.skipped_sim_ids}"
        )
