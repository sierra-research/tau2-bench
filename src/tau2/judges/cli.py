"""Focused post-hoc speech-judge CLI for the tau-Elicitation release."""

import argparse
import json
from pathlib import Path


def add_judges_args(parser: argparse.ArgumentParser) -> None:
    """Register the standard delivery-only rejudge command."""
    commands = parser.add_subparsers(
        dest="judges_command", help="Judge commands", required=True
    )
    rejudge = commands.add_parser(
        "rejudge",
        help="Regenerate the v6 speech judgments from stored both.wav audio",
    )
    rejudge.add_argument("results", help="Results file, cell, or results-tree root")
    rejudge.add_argument(
        "--delivery", action="store_true", help="Run the multimodal speech judge"
    )
    rejudge.add_argument(
        "--delivery-only",
        action="store_true",
        help="Do not run any text judges (required by this compact release)",
    )
    rejudge.add_argument(
        "--rejudge",
        action="store_true",
        help="Replace existing speech judgments; otherwise fill only missing ones",
    )
    rejudge.add_argument(
        "--delivery-sample-rate",
        type=float,
        default=1.0,
        help="Deterministic fraction of calls to judge (paper setting: 1.0)",
    )
    rejudge.add_argument(
        "--delivery-model",
        default="gemini/gemini-3.1-pro-preview",
        help="Multimodal model id (paper model shown as default)",
    )
    rejudge.add_argument(
        "--judge-args",
        default='{"temperature":0.0,"max_tokens":8192,"timeout":120}',
        help="JSON object forwarded to every model call",
    )
    rejudge.add_argument(
        "--max-concurrency",
        type=int,
        default=8,
        help="Concurrent utterance-level API calls",
    )
    rejudge.add_argument(
        "--max-segments",
        type=int,
        default=100,
        help="Maximum agent utterances per call (paper setting: 100)",
    )
    rejudge.add_argument(
        "--limit-sims",
        type=int,
        default=None,
        help="Optional deterministic simulation cap for a smoke test",
    )
    rejudge.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write mirrored simulation JSON under this directory; default in-place",
    )
    rejudge.set_defaults(func=run_judges_rejudge)


def run_judges_rejudge(args: argparse.Namespace) -> None:
    """Validate CLI arguments and run the release speech judge."""
    from tau2.judges.speech import SpeechJudgeConfig, rejudge_results

    if not args.delivery or not args.delivery_only:
        raise SystemExit(
            "This compact release supports only --delivery --delivery-only"
        )
    try:
        model_args = json.loads(args.judge_args)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--judge-args must be valid JSON: {exc}") from exc
    if not isinstance(model_args, dict):
        raise SystemExit("--judge-args must be a JSON object")
    config = SpeechJudgeConfig(
        model=args.delivery_model,
        model_args=model_args,
        sample_rate=args.delivery_sample_rate,
        max_segments=args.max_segments,
        concurrency=args.max_concurrency,
    )
    report = rejudge_results(
        Path(args.results),
        output=args.output,
        replace_existing=args.rejudge,
        limit_sims=args.limit_sims,
        config=config,
    )
    print(
        f"speech rejudge: {report.judged} judged, {report.reused} reused, "
        f"{report.unsampled} unsampled, {report.unavailable} unavailable -> "
        f"{report.output_description}"
    )
