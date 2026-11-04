"""Package-owned selection, execution and pinning of released benchmarks."""

import argparse
import copy
from datetime import datetime
from pathlib import Path
from typing import Callable

from loguru import logger

from tau2.config import (
    DEFAULT_BENCHMARK_WORKERS,
    DEFAULT_VOICE_BENCHMARK,
)
from tau2.data_model.simulation import RunConfig
from tau2.runner.presets import (
    BenchmarkRequest,
    ExecutionSettings,
    execute_plan,
    load_frame,
    load_manifest,
    pin_manifest,
    preset_names,
    resolve_benchmark,
)


def add_pass3_args(
    parser: argparse.ArgumentParser,
    *,
    add_run_args: Callable[[argparse.ArgumentParser], None],
    run_command: Callable[[argparse.Namespace], object],
) -> None:
    """Register the generic default and opt-in R/C/S voice experiment."""
    add_run_args(parser)
    parser.add_argument(
        "--pass3-mode",
        choices=("generic", "crossed"),
        default="generic",
        help="Generic repeated trials (default), or crossed R/C/S voice conditions.",
    )
    parser.add_argument(
        "--regular-results",
        type=Path,
        help="Existing ordinary seed-42 regular results to reuse read-only in crossed mode.",
    )

    def run_pass3(args: argparse.Namespace) -> object:
        from tau2.data_model.simulation import Results, VoiceRunConfig
        from tau2.runner.batch import _load_run_tasks, make_voice_run_settings
        from tau2.runner.helpers import get_info
        from tau2.runner.pass3 import (
            CROSSED_CONDITIONS,
            summarize_crossed_pass3,
            validate_regular_source,
        )

        if args.num_trials not in (1, 3):
            raise ValueError("run-pass3 uses exactly three trials")
        if args.pass3_mode == "generic":
            if args.regular_results is not None:
                raise ValueError("--regular-results requires --pass3-mode crossed")
            generic = copy.copy(args)
            generic.num_trials = 3
            return run_command(generic)

        if not args.audio_native:
            raise ValueError("Crossed Pass^3 requires --audio-native")
        if not args.domain or len(args.domain) != 1:
            raise ValueError("Crossed Pass^3 requires exactly one --domain")
        if args.num_trials != 1:
            raise ValueError("Crossed Pass^3 runs one trial per condition")
        if not args.save_to:
            raise ValueError("Crossed Pass^3 requires --save-to as a results prefix")
        if args.speech_complexity != "regular":
            raise ValueError("Crossed Pass^3 uses the regular complexity preset")
        if (
            args.channel_effects_mode != "regular"
            or args.speech_effects_mode != "regular"
        ):
            raise ValueError("Crossed Pass^3 owns both effects modes")
        results = {}
        source_paths = {}
        if args.regular_results is not None:
            regular = CROSSED_CONDITIONS[0]
            source = args.regular_results.resolve()
            if source.is_dir():
                source /= "results.json"
            if not source.is_file():
                raise FileNotFoundError(f"Regular results not found: {source}")
            result = Results.load(source)
            preflight = copy.copy(args)
            preflight.benchmark = "none"
            preflight.seed = regular.seed
            preflight.voice_sampling_seed = None
            preflight.pass3_condition = None
            preflight.dry_run = True
            preflight.save_to = f"{args.save_to}_{regular.label}"
            config = run_command(preflight)
            if not isinstance(config, VoiceRunConfig):
                raise ValueError("Crossed Pass^3 regular preflight needs a voice run")
            tasks = _load_run_tasks(config)
            voice_settings, persona_config = make_voice_run_settings(config)
            expected_info = get_info(
                config,
                tasks=tasks,
                user_voice_settings=voice_settings,
                user_persona_config=persona_config,
            )
            validate_regular_source(result, expected_info, tasks)
            results[regular.label] = result
            source_paths[regular.label] = str(source)
        for condition in CROSSED_CONDITIONS:
            if condition.label in results:
                continue
            selected = copy.copy(args)
            selected.benchmark = "none"
            selected.num_trials = 1
            selected.seed = condition.seed
            selected.voice_sampling_seed = None
            selected.channel_effects_mode = condition.channel_effects_mode
            selected.speech_effects_mode = condition.speech_effects_mode
            selected.pass3_condition = condition.label
            selected.save_to = f"{args.save_to}_{condition.label}"
            selected.auto_resume = True
            result = run_command(selected)
            if not args.dry_run:
                results[condition.label] = result
            from tau2.utils import DATA_DIR

            source_paths[condition.label] = str(
                DATA_DIR
                / "simulations"
                / f"{args.save_to}_{condition.label}"
                / "results.json"
            )
        if args.dry_run:
            return None

        from tau2.utils import DATA_DIR

        report = summarize_crossed_pass3(
            results,
            source_paths,
        )
        report_file = DATA_DIR / "simulations" / f"{args.save_to}_pass3.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(report.model_dump_json(indent=2) + "\n")
        print(
            f"Crossed Pass^3: {report.passed_all}/{report.scored_tasks} scored "
            f"({len(report.infrastructure_error_task_ids)} infrastructure errors) "
            f"({report.pass3:.1%}); {report_file}"
        )
        return report

    parser.set_defaults(func=run_pass3)


def add_benchmark_run_args(parser: argparse.ArgumentParser) -> None:
    """Extend the ordinary run command with explicit released/custom selection."""
    parser.add_argument(
        "--benchmark",
        default=None,
        help="Released benchmark manifest: voice runs default to 'voice' (τ-i18n, V2); "
        "use 'voice-v1' for the English current-leaderboard frame. Text runs "
        "are custom single-domain runs unless you select 'text'. Use 'none' for a "
        "custom single-domain run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and display the run without provider calls or output writes.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Released benchmarks: print the full resolved plan (with --dry-run) "
        "and the full score summary as JSON instead of compact tables.",
    )
    parser.add_argument(
        "--global-limit",
        type=int,
        help="Total concurrent calls across the benchmark worker fleet.",
    )
    parser.add_argument(
        "--voice",
        help="OpenAI Realtime agent voice override; the released default is marin.",
    )


def add_benchmarks_args(parser: argparse.ArgumentParser) -> None:
    """Register released-manifest inspection and pinning verbs."""
    sub = parser.add_subparsers(dest="benchmarks_command", required=True)
    listing = sub.add_parser("list", help="List installed benchmark manifests")
    listing.set_defaults(func=run_benchmarks_list)
    pin = sub.add_parser(
        "pin",
        help="Recompute a manifest's subset digests and binding revisions from the "
        "current checkout (or derive a new manifest with --from)",
    )
    pin.add_argument("name", help="Benchmark manifest name, e.g. voice or text")
    pin.add_argument(
        "--revision",
        help="New manifest revision label; required when any pin changes",
    )
    pin.add_argument(
        "--from", dest="source", help="Derive a new manifest from this one's cells"
    )
    pin.add_argument("--mode", choices=("text", "voice"), help="Call mode with --from")
    pin.add_argument("--benchmark-version", help="Benchmark generation with --from")
    pin.set_defaults(func=run_benchmarks_pin)


def run_benchmarks_list(args: argparse.Namespace) -> None:
    """Print each installed manifest's identity and matrix size."""
    import json

    rows = []
    for name in preset_names():
        manifest, _ = load_manifest(name)
        rows.append(
            {
                "name": name,
                "benchmark_version": manifest.benchmark_version,
                "revision": manifest.revision,
                "mode": manifest.mode,
                "cells": len(manifest.cells),
            }
        )
    print(json.dumps(rows, indent=2))


def run_benchmarks_pin(args: argparse.Namespace) -> None:
    """Rewrite one manifest's derived pins and print the result."""
    manifest = pin_manifest(
        args.name,
        revision=args.revision,
        source=args.source,
        mode=args.mode,
        benchmark_version=args.benchmark_version,
    )
    print(manifest.model_dump_json(indent=2))


def selected_benchmark(args: argparse.Namespace) -> str | None:
    """The released benchmark a run selects; None for a custom run."""
    if args.benchmark is None:
        return DEFAULT_VOICE_BENCHMARK if args.audio_native else None
    return None if args.benchmark == "none" else args.benchmark


def check_benchmark_selection(args: argparse.Namespace) -> None:
    """Reject selections outside the released frame before building any config."""
    frame = load_frame(args.benchmark)
    mode = "voice" if args.audio_native else "text"
    if frame.mode != mode:
        raise ValueError(
            f"The '{args.benchmark}' benchmark runs {frame.mode} calls; "
            + (
                "drop --audio-native or choose a voice benchmark"
                if mode == "voice"
                else "add --audio-native or choose a text benchmark"
            )
        )
    if frame.benchmark_version == "voice-v1":
        if args.task_ids is not None or args.num_tasks is not None:
            raise ValueError(
                "Voice V1 submissions require the complete task frame; "
                "use --benchmark none for task samples"
            )
    for flag, selected, allowed in (
        ("--domain", args.domain, {cell.domain for cell in frame.cells}),
        ("--language", args.language, {cell.language for cell in frame.cells}),
    ):
        outside = sorted(set(selected or []) - allowed)
        if outside:
            raise ValueError(
                f"{flag} {', '.join(outside)} is not part of the released "
                f"'{args.benchmark}' benchmark (available: {', '.join(sorted(allowed))}); "
                "use --benchmark none for a custom single-domain run"
            )


def benchmark_workers(requested: int | None) -> int:
    """Resolve fleet size; released benchmarks always run through the controller."""
    if requested is None:
        return DEFAULT_BENCHMARK_WORKERS
    if requested < 0:
        raise ValueError("workers must be nonnegative")
    if requested == 0:
        logger.warning(
            "Released benchmarks run through the worker controller; "
            f"--workers 0 is overridden to {DEFAULT_BENCHMARK_WORKERS}"
        )
        return DEFAULT_BENCHMARK_WORKERS
    return requested


def run_benchmark(args: argparse.Namespace, template: RunConfig) -> None:
    """Resolve one typed request and execute all cells through the shared controller."""
    from tau2.metrics.benchmark import export_submission
    from tau2.runner.report import render_plan, render_summary
    from tau2.utils import DATA_DIR

    output = (
        DATA_DIR
        / "simulations"
        / (args.save_to or f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{args.benchmark}")
    )
    plan = resolve_benchmark(
        BenchmarkRequest(
            preset=args.benchmark,
            template=template,
            output=output,
            languages=args.language,
            domains=args.domain,
        )
    )
    settings = ExecutionSettings(
        workers=benchmark_workers(args.workers),
        provider_limits=template.provider_limits,
        global_limit=args.global_limit,
    )
    calls = sum(len(config.task_ids) * config.num_trials for config in plan.configs)
    languages = list(dict.fromkeys(config.language for config in plan.configs))
    domains = list(dict.fromkeys(config.domain for config in plan.configs))
    print(
        f"{plan.frame.benchmark_version}: {', '.join(languages)} × {', '.join(domains)}; "
        f"{calls} calls, {len(plan.frame.trials)} trial(s), {settings.workers} worker(s)."
    )
    print(
        f"Scope: {'sample/slice' if plan.partial else 'complete released benchmark'}. Output: {output}"
    )
    if args.dry_run:
        print(plan.model_dump_json(indent=2) if args.json else render_plan(plan))
    execute_plan(plan, output, settings, dry_run=args.dry_run)
    if not args.dry_run:
        submission = export_submission(output, plan.frame)
        headline = (
            f"{submission.summary.score:.1%}"
            if submission.summary.complete
            else "incomplete"
        )
        print(f"Overall pass@1: {headline}")
        if args.json:
            print(submission.summary.model_dump_json(indent=2))
            if submission.requested is not None:
                print("Requested scope:")
                print(submission.requested.model_dump_json(indent=2))
        else:
            # A sample/slice reports its requested cells; the released frame's
            # other cells would only list missing trials.
            print(render_summary(submission.requested or submission.summary))
