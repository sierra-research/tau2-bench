# Copyright Sierra
"""Package-owned CLI for the LLM conversation-judge suite."""

import argparse
import json
from pathlib import Path

from tau2.config import DEFAULT_CONVERSATION_JUDGE_CONCURRENCY
from tau2.judges.conversation.models import ConversationJudgeConfig
from tau2.judges.conversation.runner import run_conversation_judging


def add_conversation_args(subparsers: argparse._SubParsersAction) -> None:
    """Register ``tau2 judges legacy conversation``."""
    parser = subparsers.add_parser(
        "conversation",
        help="LLM conversation judges (EVA progression per dimension + "
        "per-turn conciseness failure modes) into a resumable sidecar; "
        "composites land in shadow_scores only",
    )
    parser.add_argument("results", nargs="+", help="Results dir(s) or JSON file(s)")
    parser.add_argument("--out", type=Path, required=True, help="Output JSON artifact")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Reusable judgment cache (default: a hidden sibling of --out)",
    )
    parser.add_argument("--eva-x-model", default=None, help="EVA-X model override")
    parser.add_argument(
        "--eva-x-model-args", default=None, help="EVA-X generate() JSON args"
    )
    parser.add_argument(
        "--ours",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="include_ours",
        help="Also compute the τ quality composite into shadow_scores "
        "(default: enabled)",
    )
    parser.add_argument("--ours-model", default=None, help="τ quality model override")
    parser.add_argument(
        "--ours-model-args", default=None, help="τ quality generate() JSON args"
    )
    parser.add_argument(
        "--ours-llm-judge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include τ semantic factors (default: enabled)",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=DEFAULT_CONVERSATION_JUDGE_CONCURRENCY,
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Smoke-run first N calls"
    )
    parser.add_argument(
        "--force", action="store_true", help="Ignore valid caches and re-judge"
    )
    parser.set_defaults(func=run_conversation_cli)


def _json_object(value: str | None, flag: str) -> dict:
    if value is None:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise SystemExit(f"{flag} must be a JSON object")
    return parsed


def run_conversation_cli(args) -> None:
    """Validate CLI settings and run the artifact producer."""
    config_data = {
        "include_ours": args.include_ours,
        "ours_llm_judge": args.ours_llm_judge,
        "ours_model": args.ours_model,
        "ours_model_args": _json_object(args.ours_model_args, "--ours-model-args"),
        "max_concurrency": args.max_concurrency,
        "limit": args.limit,
    }
    if args.eva_x_model is not None:
        config_data["eva_x_model"] = args.eva_x_model
    if args.eva_x_model_args is not None:
        config_data["eva_x_model_args"] = _json_object(
            args.eva_x_model_args, "--eva-x-model-args"
        )
    config = ConversationJudgeConfig(**config_data)
    cache_root = args.cache_dir or args.out.parent / f".{args.out.stem}.cache"
    artifact = run_conversation_judging(
        [Path(path) for path in args.results],
        out_path=args.out,
        cache_root=cache_root,
        config=config,
        force=args.force,
    )
    print(
        f"conversation -> {args.out} ({artifact.num_calls} calls, "
        f"{artifact.num_errors} suite errors; cache {cache_root})"
    )
    print("headline (per language x domain):")
    for cell in artifact.cells:
        majors = " ".join(
            f"{summary.name.value}={summary.major_rate:.2f}"
            for summary in cell.dimensions
            if summary.major_rate is not None
        )
        minors = " ".join(
            f"{summary.name.value}={summary.minor_rate:.2f}"
            for summary in cell.dimensions
            if summary.minor_rate is not None
        )
        modes = " ".join(
            f"{summary.mode.value}={summary.rate:.2f}"
            for summary in cell.conciseness_failure_modes
            if summary.rate is not None and summary.turn_count > 0
        )
        print(
            f"  {cell.language}/{cell.domain}: n={cell.n_calls} "
            f"judged={cell.n_judged} turns={cell.n_judged_turns}\n"
            f"    progression major-rates (rating 1): {majors or 'n/a'}\n"
            f"    progression minor-rates (rating 2): {minors or 'n/a'}\n"
            f"    conciseness failure-modes: {modes or 'none'}"
        )
    print("shadow scores (uncalibrated-shadow; comparability only, never a ranking):")
    for cell in artifact.cells:
        shadow = cell.shadow_scores

        def fmt(value):
            return "n/a" if value is None else f"{value:.3f}"

        print(
            f"  {cell.language}/{cell.domain}: "
            f"eva_x_pass_rate={fmt(shadow.eva_x_pass_rate)} "
            f"progression={fmt(shadow.progression_score_mean)} "
            f"conciseness={fmt(shadow.conciseness_score_mean)} "
            f"turn_taking={fmt(shadow.turn_taking_score_mean)} "
            f"ours={fmt(shadow.ours_score_mean)}"
        )
