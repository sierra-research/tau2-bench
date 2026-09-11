# Copyright Sierra
"""Package-owned registration for the ``tau2 annotate`` pipeline verbs.

Each exporter writes a provenance-bearing artifact; ``ingest`` validates the
artifact kind and dispatches to its typed row model. The translation-review
export survives its retired producing loop: it reads archived rows, and filled
sheets remain ingestible.
"""

import argparse
import json
from pathlib import Path

from tau2.annotation.localization_sheet import DEFAULT_LOCALIZATION_SHEET_DIR
from tau2.annotation.source_audit import (
    DEFAULT_ANNOTATIONS_ROOT,
    DEFAULT_SIMULATIONS_ROOT,
)
from tau2.config import (
    DEFAULT_ANNOTATION_MAX_TRANS_CHARS,
    DEFAULT_ANNOTATION_PRECISION_BAR,
    DEFAULT_CALIBRATION_ADJUDICATION_FACTOR_CAP,
    DEFAULT_CALIBRATION_CALLS_PER_LANGUAGE,
    DEFAULT_CALIBRATION_CONTROL_FRACTION,
    DEFAULT_CALIBRATION_SEED,
    DEFAULT_FEATURE_LONG_SILENCE_SECONDS,
    DEFAULT_JUDGE_STREAM_CONCURRENCY,
    DEFAULT_MULTILINGUAL_DOMAIN,
)


def add_annotate_args(parser: argparse.ArgumentParser) -> None:
    """Attach the annotate sub-subcommands to the ``tau2 annotate`` parser."""
    annotate_sub = parser.add_subparsers(
        dest="annotate_command", help="Annotation commands", required=True
    )

    # --- nativeness --------------------------------------------------------
    nativeness_parser = annotate_sub.add_parser(
        "nativeness",
        help="Export nativeness-judge calibration sheets from results",
    )
    nativeness_parser.add_argument(
        "results", nargs="+", help="Results dir(s) or results.json file(s)"
    )
    nativeness_parser.add_argument(
        "--mode",
        choices=["precision", "cold", "both"],
        default="precision",
        help="precision: one row per judge FAIL; cold: blind wide sheet; "
        "both: one judge pass, cold family + derived precision sheet",
    )
    nativeness_parser.add_argument(
        "--out",
        type=Path,
        default=Path("nativeness_sheets"),
        help="Output stem (family CSVs + manifest land beside it)",
    )
    nativeness_parser.add_argument(
        "--rejudge",
        action="store_true",
        help="Force re-judging even when complete stored verdicts exist "
        "(default: reuse stored verdicts, mirroring the judges CLIs)",
    )
    nativeness_parser.add_argument(
        "--max-trans-chars",
        type=int,
        default=DEFAULT_ANNOTATION_MAX_TRANS_CHARS,
        help="Transcript truncation per row "
        f"(default {DEFAULT_ANNOTATION_MAX_TRANS_CHARS})",
    )
    nativeness_parser.add_argument(
        "--max-concurrency",
        type=int,
        default=DEFAULT_JUDGE_STREAM_CONCURRENCY,
        help="Concurrent judge calls (I/O-bound); default "
        f"{DEFAULT_JUDGE_STREAM_CONCURRENCY}",
    )
    nativeness_parser.set_defaults(func=run_annotate_nativeness)

    # --- workbook ----------------------------------------------------------
    workbook_parser = annotate_sub.add_parser(
        "workbook",
        help="Build formatted .xlsx workbooks from exported nativeness sheet "
        "CSVs (nativeness_precision / nativeness_cold families only)",
    )
    workbook_parser.add_argument(
        "csv", nargs="+", type=Path, help="Exported sheet CSV(s)"
    )
    workbook_parser.add_argument(
        "--publish-to",
        type=Path,
        default=None,
        help="Also copy each built .xlsx here under a readable name "
        "(e.g. the Drive-mount calibration folder); Drive sync uploads it",
    )
    workbook_parser.set_defaults(func=run_annotate_workbook)

    # --- localization-sheet ------------------------------------------------
    localization_parser = annotate_sub.add_parser(
        "localization-sheet",
        help="Side-by-side task review workbook: the English source plus each "
        "language's arm as it will actually run, with an approve/deny "
        "dropdown and a notes cell per language",
    )
    localization_parser.add_argument(
        "--domain", required=True, help="Task domain, e.g. retail"
    )
    localization_parser.add_argument(
        "--langs",
        required=True,
        help="Comma-separated language codes in column order; the first is "
        "the reference column (e.g. en,es,pt,ko,zh,hi)",
    )
    localization_parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_LOCALIZATION_SHEET_DIR,
        help="Output directory (default %(default)s)",
    )
    localization_parser.add_argument(
        "--only-subset",
        action="store_true",
        help="Only the domain's canonical task subset (the fixed set a full "
        "run scores); default is every task, with those ones marked",
    )
    localization_parser.add_argument(
        "--publish-to",
        type=Path,
        default=None,
        help="Also copy the workbook here (e.g. the Drive-mount review "
        "folder); Drive sync uploads it with its dropdowns intact",
    )
    localization_parser.set_defaults(func=run_annotate_localization_sheet)

    # --- audit -------------------------------------------------------------
    audit_parser = annotate_sub.add_parser(
        "audit",
        help="Nativeness Audit sheet bodies (and --generate nuance candidates)",
    )
    audit_parser.add_argument(
        "--lang",
        default="all",
        help="ISO code, or 'all' for every registered audit tab (default)",
    )
    audit_parser.add_argument(
        "--blind",
        action="store_true",
        help="Build the Round-1 (blind) bodies instead of the verify round",
    )
    audit_parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate AI nuance candidates first (fixed versioned prompt "
        "through the generate() seam), then build the bodies",
    )
    audit_parser.add_argument(
        "--model",
        default=None,
        help="Override the nuance-candidate generator model",
    )
    audit_parser.set_defaults(func=run_annotate_audit)

    # --- source-audit ------------------------------------------------------
    source_audit_parser = annotate_sub.add_parser(
        "source-audit",
        help="Check the runs recorded in packet provenance still sit where "
        "they were read (and are still the SAME run); --apply rewrites the "
        "ones that merely moved",
    )
    source_audit_parser.add_argument(
        "--annotations-root",
        type=Path,
        default=DEFAULT_ANNOTATIONS_ROOT,
        help="Manifest tree to scan (default %(default)s)",
    )
    source_audit_parser.add_argument(
        "--simulations-root",
        type=Path,
        default=DEFAULT_SIMULATIONS_ROOT,
        help="Run tree to search for relocated runs (default %(default)s)",
    )
    source_audit_parser.add_argument(
        "--apply",
        action="store_true",
        help="Rewrite relocated paths in place (default: report only). A run "
        "that is gone, replaced, or duplicated is recorded as such, never "
        "rewritten",
    )
    source_audit_parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Also write the audit report JSON here",
    )
    source_audit_parser.set_defaults(func=run_annotate_source_audit)

    # --- packets -----------------------------------------------------------
    packets_parser = annotate_sub.add_parser(
        "packets",
        help="Build a standalone HTML annotation packet (conversation pages + "
        "audio + index + manifest) from results",
    )
    packets_parser.add_argument(
        "--form",
        choices=["error_analysis", "user_realism", "voice_review"],
        required=True,
        help="Per-conversation annotation form: error findings, realism "
        "Likert ratings, or the combined voice_review pass",
    )
    packets_parser.add_argument(
        "--batch-name",
        required=True,
        help="Batch name (also the default output directory name)",
    )
    packets_parser.add_argument(
        "--no-zip",
        action="store_true",
        help="Skip the <packet>.zip. The zip is how a packet SHIPS — Google "
        "Drive syncs one archive in minutes and a 1000-file tree in hours — "
        "so this is for local iteration, not for a packet you intend to send",
    )
    packets_parser.add_argument(
        "--results",
        type=Path,
        nargs="+",
        default=None,
        help="Results dir(s) or results.json file(s)",
    )
    packets_parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Packet directory (default: data/annotations/<batch-name>)",
    )
    packets_parser.add_argument(
        "--append",
        action="store_true",
        help="Add new sims to an existing packet (dedupes by sim id against "
        "the manifest; the form must match)",
    )
    packets_parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle simulations before --max-items (seeded random sample)",
    )
    packets_parser.add_argument(
        "--seed", type=int, default=42, help="Seed for --shuffle (default 42)"
    )
    packets_parser.add_argument(
        "--max-items", type=int, default=None, help="Cap on exported simulations"
    )
    packets_parser.add_argument(
        "--filter-reward",
        default=None,
        help='Reward filter, e.g. "< 1", "== 0", ">= 0.5"',
    )
    packets_parser.add_argument(
        "--filter-tasks",
        nargs="+",
        default=None,
        help="Keep only these task ids",
    )
    packets_parser.set_defaults(func=run_annotate_packets)

    # --- prompt-bed-packet ---------------------------------------------------
    prompt_bed_parser = annotate_sub.add_parser(
        "prompt-bed-packet",
        help="Build one language's prompt + background-bed review packet "
        "(the exact runtime user-sim prompts, the personas' voice samples, "
        "plus every locale's beds)",
    )
    prompt_bed_parser.add_argument("--lang", required=True, help="ISO 639-1 code")
    prompt_bed_parser.add_argument(
        "--domain",
        default=DEFAULT_MULTILINGUAL_DOMAIN,
        help="Domain (default: %(default)s)",
    )
    prompt_bed_parser.add_argument(
        "--task-index",
        type=int,
        default=0,
        help="Index into the language's main run-arm task set (default 0)",
    )
    prompt_bed_parser.add_argument(
        "--task-set",
        default=None,
        help="Render a specific registered task set of this domain instead "
        "of the language's main run-arm set — the pre-run review gate for "
        "variant sets (e.g. retail_hi_identity_native shows the native "
        "spell-out payloads and the flipped agent DB clause)",
    )
    prompt_bed_parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output root; the packet lands at <out>/<lang>/ "
        "(default data/annotations/prompt_bed_review)",
    )
    prompt_bed_parser.add_argument(
        "--no-voice-samples",
        dest="voice_samples",
        action="store_false",
        help="Skip rendering the personas' pinned-voice audio samples "
        "(ElevenLabs TTS — the packet's one online step); the packet then "
        "builds fully offline with no voices section",
    )
    prompt_bed_parser.set_defaults(func=run_annotate_prompt_bed_packet)

    # --- calibration-packets ----------------------------------------------
    calibration_parser = annotate_sub.add_parser(
        "calibration-packets",
        help="Judge-calibration wave packets: --mode annotate draws the "
        "defect-enriched cohort from stored judge results and builds the "
        "blind rater packet (frame + coverage sidecar land OUTSIDE it); "
        "--mode adjudicate builds the owner-facing packet from EITHER the "
        "annotate build's --sidecar (judge-vs-raters with --filled CSVs, "
        "judge-only review without) OR an independent per-factor draw over "
        "--results (min(--per-factor-cap, #flagged) calls per judge factor)",
    )
    calibration_parser.add_argument(
        "--mode",
        choices=["annotate", "adjudicate"],
        default="annotate",
        help="Which packet to build (default %(default)s)",
    )
    calibration_parser.add_argument(
        "--results",
        type=Path,
        nargs="+",
        default=None,
        help="Results dirs / results.json files with stored judge results "
        "(annotate: the blind-wave corpus; adjudicate: the per-factor draw "
        "corpus — mutually exclusive with --sidecar)",
    )
    calibration_parser.add_argument(
        "--lang", default=None, help="ISO 639-1 code of the wave / draw"
    )
    calibration_parser.add_argument(
        "--conversation-artifact",
        type=Path,
        default=None,
        help="Stored conversation-judge artifact; required to see the "
        "LLM-quality verdicts on corpora that store them in the artifact "
        "rather than on the sims (quality strata stay empty without it)",
    )
    calibration_parser.add_argument(
        "--n-calls",
        type=int,
        default=None,
        help="annotate: cohort size (default "
        f"{DEFAULT_CALIBRATION_CALLS_PER_LANGUAGE})",
    )
    calibration_parser.add_argument(
        "--control-fraction",
        type=float,
        default=None,
        help="annotate: judge-clean control share (default "
        f"{DEFAULT_CALIBRATION_CONTROL_FRACTION})",
    )
    calibration_parser.add_argument(
        "--draw",
        choices=["enriched", "random"],
        default="enriched",
        help="annotate: cohort selection rule — 'enriched' (default) is the "
        "defect-enriched draw over stored judge verdicts; 'random' is the "
        "judge-blind uniform draw for the RECALL arm (never conditioned on "
        "judge outputs; --control-fraction is ignored)",
    )
    calibration_parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_CALIBRATION_SEED,
        help="Draw seed, both modes (default %(default)s)",
    )
    calibration_parser.add_argument(
        "--per-factor-cap",
        type=int,
        default=None,
        help="adjudicate --results: max flagged calls drawn per judge factor "
        f"(default {DEFAULT_CALIBRATION_ADJUDICATION_FACTOR_CAP}; capped, "
        "never padded — a factor with fewer flagged calls is taken whole)",
    )
    calibration_parser.add_argument(
        "--decision-row-cap",
        type=int,
        default=None,
        help="adjudicate: max PRESENTED decision rows per judge factor "
        "(seeded sample; fidelity stratified by finding category; default "
        "from CalibrationAdjudicationOptions)",
    )
    calibration_parser.add_argument(
        "--show-judge",
        action="store_true",
        help="annotate: the TWO-PHASE pass — each call is answered blind "
        "first (judge findings hidden), then a lock reveals the judge's "
        "positive findings beneath each question for revision "
        "(banner-marked, batch branded _judge_visible; exports carry both "
        "layers: pre-reveal rows join the blind recall pool, post-reveal "
        "rows score verified precision in calibration-agreement; the "
        "default build stays blind)",
    )
    calibration_parser.add_argument(
        "--sidecar",
        type=Path,
        default=None,
        help="adjudicate: the annotate build's coverage sidecar (pinned "
        "cohort — mutually exclusive with --results)",
    )
    calibration_parser.add_argument(
        "--filled",
        type=Path,
        nargs="+",
        default=None,
        help="adjudicate: returned rater CSVs (rubric-answer and/or "
        "nativeness-label exports); omit for a judge-only review view of "
        "the draw (judge verdicts rendered, no rater columns)",
    )
    calibration_parser.add_argument(
        "--watermark",
        default=None,
        help="adjudicate: banner rendered across the page (e.g. to mark a "
        "synthetic-label sample)",
    )
    calibration_parser.add_argument(
        "--batch-name", default=None, help="Packet label and default folder"
    )
    calibration_parser.add_argument(
        "--out", type=Path, default=None, help="Output packet directory"
    )
    calibration_parser.add_argument(
        "--no-zip", action="store_true", help="Skip the lossless shipping ZIP"
    )
    calibration_parser.set_defaults(func=run_annotate_calibration_packets)

    # --- recall-corpus -----------------------------------------------------
    recall_parser = annotate_sub.add_parser(
        "recall-corpus",
        help="Materialize a judge-blind RANDOM cohort of stored calls as a "
        "standalone subset corpus (per-cell dir-format results + linked "
        "audio) so judges and packets run over exactly the recall-arm "
        "sample; fresh verdicts land on the subset, never the source",
    )
    recall_parser.add_argument(
        "--results",
        type=Path,
        nargs="+",
        required=True,
        help="Source results dirs / results.json files (the full corpus)",
    )
    recall_parser.add_argument(
        "--lang", required=True, help="ISO 639-1 code of the cohort"
    )
    recall_parser.add_argument(
        "--n-calls",
        type=int,
        default=None,
        help="Cohort size (default: the owner-sized wave default)",
    )
    recall_parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_CALIBRATION_SEED,
        help="Draw seed (default %(default)s)",
    )
    recall_parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Corpus root to create (refused when it already exists)",
    )
    recall_parser.set_defaults(func=run_annotate_recall_corpus)

    # --- feature-table -------------------------------------------------------
    feature_table_parser = annotate_sub.add_parser(
        "feature-table",
        help="Extract the deterministic per-call feature "
        "table (Tier-1 features from tick timing, turn-taking, and tool flow) "
        "from run results",
    )
    feature_table_parser.add_argument(
        "results", nargs="+", type=Path, help="Results dir(s) or results.json file(s)"
    )
    feature_table_parser.add_argument(
        "--out",
        type=Path,
        default=Path("feature_table.json"),
        help="Feature-table JSON artifact path (default %(default)s)",
    )
    feature_table_parser.add_argument(
        "--langs",
        nargs="+",
        default=None,
        help="Restrict to these ISO 639-1 codes (default: every language found)",
    )
    feature_table_parser.add_argument(
        "--domain", default=None, help="Restrict to one domain"
    )
    feature_table_parser.add_argument(
        "--long-silence-threshold",
        type=float,
        default=DEFAULT_FEATURE_LONG_SILENCE_SECONDS,
        help="Seconds of mutual silence counted as a long silence "
        "(default %(default)s)",
    )
    feature_table_parser.add_argument(
        "--max-sims",
        type=int,
        default=None,
        help="Cap on extracted sims (dry runs / smokes)",
    )
    feature_table_parser.set_defaults(func=run_annotate_feature_table)

    # --- ingest ------------------------------------------------------------
    ingest_parser = annotate_sub.add_parser(
        "ingest",
        help="Ingest ANY filled annotation sheet (dispatches on the artifact "
        "manifest's kind)",
    )
    ingest_parser.add_argument("filled", type=Path, help="Annotator-filled CSV")
    ingest_parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Manifest path, when the filled CSV was renamed/moved away from "
        "its export family",
    )
    ingest_parser.add_argument(
        "--second",
        type=Path,
        default=None,
        help="A second annotator's filled cold CSV (same rows) for "
        "inter-annotator kappa",
    )
    ingest_parser.add_argument(
        "--precision-bar",
        type=float,
        default=None,
        help=f"LIVE/shadow precision bar (default {DEFAULT_ANNOTATION_PRECISION_BAR})",
    )
    ingest_parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional metrics CSV out (nativeness sheets only)",
    )
    ingest_parser.set_defaults(func=run_annotate_ingest)

    # --- translation-review ---------------------------------------------------
    translation_review_parser = annotate_sub.add_parser(
        "translation-review",
        help="Export the translation-review sheet family (native annotators "
        "vs the retired translation pipeline's stored rows). Task translation "
        "is retired, so this reads archived rows rather than producing new "
        "ones; filled sheets ingest via `tau2 annotate ingest`",
    )
    translation_review_parser.add_argument(
        "--lang", required=True, help="ISO 639-1 code"
    )
    translation_review_parser.add_argument(
        "--domain", default="airline", help="Domain (default airline)"
    )
    translation_review_parser.add_argument(
        "--from-existing",
        type=Path,
        default=None,
        help="Localized task-set JSON: export verdict-less rows for an "
        "already-translated set (retro-calibration)",
    )
    translation_review_parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output stem (default: the language's calibration dir)",
    )
    translation_review_parser.set_defaults(func=run_annotate_translation_review)

    # --- agreement -----------------------------------------------------------
    agreement_parser = annotate_sub.add_parser(
        "agreement",
        help="A language's persisted calibration agreement summaries "
        "(translation, communicate judge, parity probe)",
    )
    agreement_parser.add_argument("--lang", required=True, help="ISO 639-1 code")
    agreement_parser.set_defaults(func=run_annotate_agreement)

    # --- calibration-agreement ------------------------------------------------
    calibration_agreement_parser = annotate_sub.add_parser(
        "calibration-agreement",
        help="Per-judge/per-factor agreement of the calibration wave: blind "
        "rater CSVs + stored judge verdicts → precision/recall/F1/kappa next "
        "to the rater-rater ceiling; judge-visible (--show-judge) returns "
        "score verified precision plus the phase-2 (post-reveal) 2x2; owner "
        "decisions CSVs add adjudicated precision. Sidecar must be the "
        "blind build's",
    )
    calibration_agreement_parser.add_argument(
        "--sidecar",
        type=Path,
        required=True,
        help="The annotate build's coverage sidecar",
    )
    calibration_agreement_parser.add_argument(
        "--filled",
        type=Path,
        nargs="+",
        default=None,
        help="Returned rater CSVs (rubric-answer and/or nativeness-label "
        "exports); at least one of --filled / --decisions is required",
    )
    calibration_agreement_parser.add_argument(
        "--decisions",
        type=Path,
        nargs="+",
        default=None,
        help="Owner decisions CSVs from the adjudicate page (Confirmed/"
        "Rejected per judge positive), reported as adjudicated precision",
    )
    calibration_agreement_parser.add_argument(
        "--precision-bar",
        type=float,
        default=DEFAULT_ANNOTATION_PRECISION_BAR,
        help="LIVE/shadow gate bar (default %(default)s)",
    )
    calibration_agreement_parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Also persist the typed report JSON here",
    )
    calibration_agreement_parser.set_defaults(func=run_annotate_calibration_agreement)


# ---------------------------------------------------------------------------
# Verb bodies (imports deferred so `tau2 --help` stays fast)
# ---------------------------------------------------------------------------


def run_annotate_nativeness(args) -> None:
    from tau2.annotation.nativeness_sheets import export_nativeness_sheets

    manifest = export_nativeness_sheets(
        args.results,
        args.mode,
        args.out,
        # Reuse-by-default, mirroring the judges CLIs; --rejudge forces.
        reuse_existing=not args.rejudge,
        max_trans_chars=args.max_trans_chars,
        concurrency=args.max_concurrency,
    )
    print(f"Wrote nativeness {args.mode} family; manifest: {manifest}")


def run_annotate_workbook(args) -> None:
    from tau2.annotation.sheets.workbook import build_and_publish

    built = build_and_publish(args.csv, publish_to=args.publish_to)
    if not built:
        raise SystemExit("no workbooks built")
    print(f"Built {built} workbook(s)")


def run_annotate_localization_sheet(args) -> None:
    from tau2.annotation.localization_sheet import (
        LocalizationSheetOptions,
        build_localization_sheet,
    )

    xlsx, manifest = build_localization_sheet(
        LocalizationSheetOptions(
            domain=args.domain,
            languages=[c.strip() for c in args.langs.split(",") if c.strip()],
            out=args.out,
            only_subset=args.only_subset,
            publish_to=args.publish_to,
        )
    )
    print(f"Wrote {xlsx}\nManifest: {manifest}")


def run_annotate_audit(args) -> None:
    from tau2.annotation.sheets.audit import AUDIT_TABS, build_audit_bodies

    langs = sorted(AUDIT_TABS) if args.lang == "all" else [args.lang]
    if args.generate:
        from tau2.annotation.nuance_candidates import (
            generate_nuance_candidates,
            write_nuance_candidates,
        )
        from tau2.annotation.sheets.audit import nuance_audit_dir

        for iso in langs:
            kwargs = {"model": args.model} if args.model else {}
            candidates = generate_nuance_candidates(iso, **kwargs)
            write_nuance_candidates(iso, candidates, nuance_audit_dir(), **kwargs)
    written = build_audit_bodies(langs, blind=args.blind)
    print(f"Wrote {len(written)} audit bodies")


def run_annotate_source_audit(args) -> None:
    from tau2.annotation.artifacts import write_json_artifact
    from tau2.annotation.source_audit import (
        SourceStatus,
        audit_sources,
        format_report,
    )

    report = audit_sources(
        args.annotations_root, args.simulations_root, apply=args.apply
    )
    print(format_report(report))
    if args.out:
        write_json_artifact(args.out, report)
        print(f"Wrote {args.out}")
    if report.count(SourceStatus.RELOCATED) and not args.apply:
        print("\nRe-run with --apply to rewrite the relocated paths.")
    if report.unreadable:
        print(
            f"\n{len(report.unreadable)} manifest(s) could not be audited — "
            "rebuild those packets (`tau2 annotate packets`)."
        )


def run_annotate_packets(args) -> None:
    from tau2.annotation.packets.builder import PacketBuildOptions, build_packet
    from tau2.annotation.packets.forms import FormType

    manifest = build_packet(
        PacketBuildOptions(
            form=FormType(args.form),
            batch_name=args.batch_name,
            results=args.results or [],
            out_dir=args.out,
            append=args.append,
            shuffle=args.shuffle,
            seed=args.seed,
            max_items=args.max_items,
            filter_reward=args.filter_reward,
            filter_tasks=args.filter_tasks,
            emit_zip=not args.no_zip,
        )
    )
    print(f"Built packet; manifest: {manifest}")
    print(f"Open {manifest.parent / 'index.html'} in a browser to annotate.")


def run_annotate_prompt_bed_packet(args) -> None:
    from tau2.annotation.packets.prompt_bed import (
        PromptBedPacketOptions,
        build_prompt_bed_packet,
    )

    manifest = build_prompt_bed_packet(
        PromptBedPacketOptions(
            language=args.lang,
            domain=args.domain,
            task_index=args.task_index,
            task_set=args.task_set,
            out_dir=args.out,
            voice_samples=args.voice_samples,
        )
    )
    print(f"Built prompt+bed packet; manifest: {manifest}")
    print(f"Open {manifest.parent / 'index.html'} in a browser to review.")


def run_annotate_recall_corpus(args) -> None:
    from tau2.annotation.recall_corpus import (
        RecallCorpusOptions,
        build_recall_corpus,
    )

    option_data: dict = {
        "results": args.results,
        "language": args.lang,
        "seed": args.seed,
        "out_dir": args.out,
    }
    if args.n_calls is not None:
        option_data["n_calls"] = args.n_calls
    build = build_recall_corpus(RecallCorpusOptions(**option_data))
    print(f"Materialized recall corpus ({build.n_calls} calls): {build.corpus_dir}")
    for cell, count in sorted(build.cells.items()):
        print(f"  {cell}: {count}")
    print(f"Sampling frame (internal-only, never ships): {build.frame_path}")


def run_annotate_calibration_packets(args) -> None:
    if args.mode == "annotate":
        forbidden = [
            name
            for name, value in (
                ("--sidecar", args.sidecar),
                ("--filled", args.filled),
                ("--watermark", args.watermark),
                ("--per-factor-cap", args.per_factor_cap),
            )
            if value
        ]
        if forbidden:
            raise SystemExit(
                f"--mode annotate does not take {', '.join(forbidden)} "
                "(those belong to --mode adjudicate)"
            )
        if not args.results or not args.lang:
            raise SystemExit("--mode annotate requires --results and --lang")
        from tau2.annotation.packets.calibration import (
            CalibrationPacketOptions,
            build_calibration_packet,
        )

        build = build_calibration_packet(
            CalibrationPacketOptions(
                results=args.results,
                language=args.lang,
                conversation_artifact=args.conversation_artifact,
                n_calls=args.n_calls,
                control_fraction=args.control_fraction,
                rule=args.draw,
                seed=args.seed,
                batch_name=args.batch_name,
                out_dir=args.out,
                emit_zip=not args.no_zip,
                show_judge=args.show_judge,
            )
        )
        flavor = "TWO-PHASE (judge-visible)" if args.show_judge else "blind (recall)"
        print(
            f"Built {flavor} judge-calibration packet; manifest: {build.manifest_path}"
        )
        print(f"Sampling frame (do NOT distribute): {build.frame_path}")
        print(f"Coverage sidecar (do NOT distribute): {build.sidecar_path}")
        print(
            f"Open {build.manifest_path.parent / 'index.html'} in a browser "
            "to annotate."
        )
        return
    forbidden = [
        name
        for name, value in (
            ("--n-calls", args.n_calls),
            ("--control-fraction", args.control_fraction),
            ("--show-judge", args.show_judge),
            ("--draw random", args.draw == "random"),
        )
        if value
    ]
    if forbidden:
        raise SystemExit(
            f"--mode adjudicate does not take {', '.join(forbidden)} "
            "(those belong to --mode annotate)"
        )
    if bool(args.sidecar) == bool(args.results):
        raise SystemExit(
            "--mode adjudicate takes exactly one cohort source: --sidecar "
            "(the annotate build's pinned cohort) or --results (an "
            "independent per-factor draw over stored judge results)"
        )
    if args.sidecar:
        forbidden = [
            name
            for name, value in (
                ("--lang", args.lang),
                ("--conversation-artifact", args.conversation_artifact),
                ("--per-factor-cap", args.per_factor_cap),
            )
            if value
        ]
        if forbidden:
            raise SystemExit(
                f"--mode adjudicate --sidecar does not take "
                f"{', '.join(forbidden)} — the cohort is pinned by the "
                "coverage sidecar"
            )
        from tau2.annotation.packets.calibration import (
            CalibrationAdjudicationOptions,
            build_calibration_adjudication,
        )

        manifest = build_calibration_adjudication(
            CalibrationAdjudicationOptions(
                sidecar=args.sidecar,
                filled=args.filled or [],
                batch_name=args.batch_name,
                out_dir=args.out,
                emit_zip=not args.no_zip,
                watermark=args.watermark,
                **(
                    {"decision_row_cap": args.decision_row_cap}
                    if args.decision_row_cap is not None
                    else {}
                ),
            )
        )
        print(f"Built calibration adjudication packet; manifest: {manifest}")
        print(f"Open {manifest.parent / 'index.html'} in a browser to adjudicate.")
        return
    if not args.lang:
        raise SystemExit("--mode adjudicate --results requires --lang")
    if args.filled:
        raise SystemExit(
            "--mode adjudicate --results does not take --filled — rater "
            "returns are keyed to an annotate build's clip ids; adjudicate "
            "them through that build's --sidecar"
        )
    from tau2.annotation.packets.calibration import (
        CalibrationAdjudicationDrawOptions,
        build_calibration_adjudication_draw,
    )

    build = build_calibration_adjudication_draw(
        CalibrationAdjudicationDrawOptions(
            results=args.results,
            language=args.lang,
            conversation_artifact=args.conversation_artifact,
            per_factor_cap=args.per_factor_cap,
            seed=args.seed,
            batch_name=args.batch_name,
            out_dir=args.out,
            emit_zip=not args.no_zip,
            watermark=args.watermark,
            **(
                {"decision_row_cap": args.decision_row_cap}
                if args.decision_row_cap is not None
                else {}
            ),
        )
    )
    print(
        "Built per-factor calibration adjudication packet; manifest: "
        f"{build.manifest_path}"
    )
    print(f"Sampling frame: {build.frame_path}")
    print(f"Coverage sidecar: {build.sidecar_path}")
    print(
        f"Open {build.manifest_path.parent / 'index.html'} in a browser to adjudicate."
    )


def run_annotate_calibration_agreement(args) -> None:
    from tau2.annotation.artifacts import write_json_artifact
    from tau2.annotation.calibration_report import (
        build_calibration_agreement,
        render_calibration_agreement,
    )

    if not args.filled and not args.decisions:
        raise SystemExit(
            "calibration-agreement needs --filled rater CSVs and/or "
            "--decisions owner CSVs"
        )
    report = build_calibration_agreement(
        args.sidecar,
        args.filled or [],
        decisions=args.decisions or [],
        precision_bar=args.precision_bar,
    )
    print(render_calibration_agreement(report))
    if args.out is not None:
        path = write_json_artifact(args.out, report)
        print(f"\nWrote report JSON: {path}")


def run_annotate_feature_table(args) -> None:
    from tau2.annotation.features import (
        FeatureExtractionConfig,
        export_feature_table,
    )

    path = export_feature_table(
        args.results,
        args.out,
        FeatureExtractionConfig(
            long_silence_threshold_s=args.long_silence_threshold,
            langs=args.langs,
            domain=args.domain,
            max_sims=args.max_sims,
        ),
    )
    print(f"Wrote feature table: {path}")


def run_annotate_translation_review(args) -> None:
    from tau2.annotation.translation import export_translation_review

    manifest = export_translation_review(
        args.lang,
        args.domain,
        from_existing=args.from_existing,
        out_stem=args.out,
    )
    print(f"Wrote translation-review family; manifest: {manifest}")


def run_annotate_ingest(args) -> None:
    from tau2.annotation.artifacts import ArtifactManifest, read_artifact

    if args.manifest is not None:
        manifest_path = Path(args.manifest)
        if not manifest_path.exists():
            raise SystemExit(f"manifest not found: {manifest_path}")
        named_kind = ArtifactManifest.model_validate_json(
            manifest_path.read_text()
        ).kind
        if named_kind.startswith("packet_"):
            raise SystemExit(
                f"{manifest_path} describes an HTML packet ({named_kind}); "
                "browser-exported packet CSVs carry no sheet-family manifest — "
                "run `tau2 annotate ingest <filled.csv>` without --manifest "
                "(dispatch is by exact header match)."
            )
    try:
        artifact = read_artifact(args.filled, args.manifest)
    except FileNotFoundError:
        if args.manifest is not None:
            raise  # an explicit --manifest that is missing is a user error
        # No manifest in the CSV's directory lists this file. Browser-exported
        # packet CSVs carry no manifest (client JS cannot write one) — the ONE
        # sanctioned header-match dispatch path.
        from tau2.annotation.packets.forms import ingest_browser_csv

        browser = ingest_browser_csv(args.filled)
        raters = sorted({row.rater for row in browser.rows if row.rater})
        print(
            f"kind: {browser.kind} | rows: {len(browser.rows)} | "
            f"completed: {browser.completed} | raters: {', '.join(raters) or '-'}"
        )
        return
    kind, role = artifact.manifest.kind, artifact.role or "sheet"
    precision_bar = (
        args.precision_bar
        if args.precision_bar is not None
        else DEFAULT_ANNOTATION_PRECISION_BAR
    )
    print(f"kind: {kind} | role: {role} | batch: {artifact.manifest.batch_id}")

    if kind == "translation_review":
        from tau2.annotation.translation import ingest_translation_review

        path = ingest_translation_review(artifact, args.filled)
        print(f"Persisted translation agreement to {path}")
    elif kind == "communicate_judge":
        from tau2.annotation.communicate_judge import ingest_communicate_judge

        path = ingest_communicate_judge(artifact, args.filled)
        print(f"Persisted communicate-judge agreement to {path}")
    elif kind == "nativeness_precision" or (
        kind == "nativeness_cold" and role == "precision"
    ):
        from tau2.annotation.metrics import analyze_precision
        from tau2.annotation.nativeness_sheets import precision_rows_from_artifact
        from tau2.annotation.reports import format_precision_report, write_metrics_csv

        metrics = analyze_precision(precision_rows_from_artifact(artifact, role))
        print(format_precision_report(metrics, precision_bar))
        if args.out:
            write_metrics_csv(metrics, args.out)
    elif kind == "nativeness_cold":
        from tau2.annotation.metrics import analyze_cold, inter_annotator_kappa
        from tau2.annotation.nativeness_sheets import cold_sheet_from_artifact
        from tau2.annotation.reports import format_cold_report, write_metrics_csv

        sheet, sidecar = cold_sheet_from_artifact(artifact)
        human_kappa = None
        if args.second:
            # A second annotator's sheet is a renamed copy of the SAME family,
            # so it reads against the first artifact's manifest (exact-name
            # manifest discovery would rightly reject the renamed copy).
            second = read_artifact(args.second, args.manifest or artifact.manifest_path)
            human_kappa = inter_annotator_kappa(sheet, second.cold_sheet())
        metrics = analyze_cold(sheet, sidecar)
        print(format_cold_report(metrics, precision_bar, human_kappa))
        if args.out:
            write_metrics_csv(metrics, args.out)
    else:  # pragma: no cover — packet_* kinds never carry a sheet-family
        # manifest (their browser CSVs dispatch above by header match), so
        # reaching here means a new SHEET_FAMILIES kind lacks a dispatch arm.
        raise SystemExit(f"no ingest path for artifact kind '{kind}'")


def run_annotate_agreement(args) -> None:
    from tau2.annotation.reports import agreement_report

    print(json.dumps(agreement_report(args.lang), indent=2, ensure_ascii=False))
