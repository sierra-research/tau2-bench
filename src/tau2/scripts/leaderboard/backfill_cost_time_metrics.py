"""
Maintainer tool: backfill cost/time breakdown metrics for text submissions.

For each text submission in the leaderboard manifest, downloads the public
trajectories from S3 (cached locally), recomputes the per-domain cost and
time breakdown (see AgentMetrics / build_domain_results), and patches the
corresponding web/leaderboard/public/submissions/<dir>/submission.json.
Trajectories on S3 are never modified; the patched submission.json files are
committed via PR and synced to S3 by CI.

Only the cost/time fields are touched; pass^k values and everything else in
the submission are left as-is. Existing cost values that differ from the
recomputed ones are overwritten (the old value is printed for review).

Usage:
    python -m tau2.scripts.leaderboard.backfill_cost_time_metrics \\
        [--submissions DIR ...] [--cache-dir PATH] [--dry-run]
"""

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

from rich.console import Console

from tau2.data_model.simulation import Results as TrajectoryResults
from tau2.metrics.agent_metrics import compute_metrics
from tau2.scripts.leaderboard.prepare_submission import build_domain_results
from tau2.scripts.leaderboard.submission import (
    MANIFEST_FILE_NAME,
    SUBMISSION_FILE_NAME,
    TRAJECTORY_FILES_DIR_NAME,
)

S3_BUCKET = "sierra-tau-bench-public"
S3_PREFIX = "submissions"

REPO_ROOT = Path(__file__).resolve().parents[4]
SUBMISSIONS_DIR = REPO_ROOT / "web" / "leaderboard" / "public" / "submissions"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "tau2" / "backfill_trajectories"

# The cost/time breakdown fields owned by this script (DomainResults fields).
COST_TIME_FIELDS = [
    "cost",
    "user_cost",
    "total_cost",
    "duration_seconds",
    "agent_time_seconds",
    "user_time_seconds",
    "tool_time_seconds",
]

# Official first-party list prices (USD per 1M tokens) used to estimate agent
# cost for runs that did not record per-message costs (self-hosted models or
# internal gateways). Keyed by the exact agent_info.llm string found in the
# trajectories. Sources (checked 2026-07-29):
#   GLM-5: docs.z.ai/guides/overview/pricing
#   Qwen3.5-397B-A17B: Alibaba Cloud Model Studio, international/Singapore
#   Claude Opus 4.6/4.7: platform.claude.com/docs/en/about-claude/pricing
USAGE_PRICING: dict[str, tuple[str, float, float]] = {
    "openai/glm-5-fp8": ("GLM-5", 1.00, 3.20),
    "openai/qwen3.5-397b": ("Qwen3.5-397B-A17B", 0.60, 3.60),
    "claude-opus-4-6": ("Claude Opus 4.6", 5.00, 25.00),
    "claude-opus-4-7": ("Claude Opus 4.7", 5.00, 25.00),
}

# Marker used to keep the methodology note idempotent across reruns.
ESTIMATE_NOTE_MARKER = "estimated from trajectory token usage"

console = Console()


def sync_trajectories(submission_name: str, cache_dir: Path) -> Path:
    """Download a submission's trajectories from public S3 (idempotent sync)."""
    dest = cache_dir / submission_name / TRAJECTORY_FILES_DIR_NAME
    dest.mkdir(parents=True, exist_ok=True)
    s3_url = (
        f"s3://{S3_BUCKET}/{S3_PREFIX}/{submission_name}/{TRAJECTORY_FILES_DIR_NAME}/"
    )
    console.print(f"  Syncing {s3_url}")
    subprocess.run(
        [
            "aws",
            "s3",
            "sync",
            s3_url,
            str(dest),
            "--delete",
            "--no-sign-request",
            "--only-show-errors",
        ],
        check=True,
    )
    return dest


def _repair_trimmed_messages(data: dict) -> int:
    """Fix messages broken by older trim_trajectories passes.

    Some older S3 trajectory files were trimmed with the message ``id`` field
    stripped, but ``ToolMessage.id`` is required by the data model. Restore a
    placeholder so validation succeeds; the id is not used by any metric.
    Returns the number of repaired messages.
    """
    repaired = 0
    for sim in data.get("simulations", []):
        for msg in sim.get("messages") or []:
            if msg.get("role") == "tool" and "id" not in msg:
                msg["id"] = "trimmed"
                repaired += 1
    return repaired


def estimate_agent_costs_from_usage(results: TrajectoryResults) -> str | None:
    """Estimate per-simulation agent cost from token usage at list prices.

    Applies only when the run recorded no agent costs at all and the agent
    model has an entry in USAGE_PRICING. Sets sim.agent_cost in place so
    compute_metrics aggregates it exactly like a recorded cost. Returns a
    description of the pricing applied, or None if not applicable.
    """
    if any(sim.agent_cost for sim in results.simulations):
        return None
    agent_info = results.info.agent_info
    pricing = USAGE_PRICING.get(agent_info.llm) if agent_info else None
    if pricing is None:
        return None
    label, input_per_m, output_per_m = pricing
    priced_any = False
    for sim in results.simulations:
        cost = 0.0
        found = False
        for msg in sim.messages or []:
            usage = getattr(msg, "usage", None)
            if getattr(msg, "role", None) == "assistant" and usage:
                cost += (
                    usage.get("prompt_tokens", 0) * input_per_m
                    + usage.get("completion_tokens", 0) * output_per_m
                ) / 1e6
                found = True
        if found:
            sim.agent_cost = cost
            priced_any = True
    if not priced_any:
        return None
    return f"{label} list price ${input_per_m}/M input, ${output_per_m}/M output"


def compute_domain_cost_time(results_path: Path) -> tuple[str, dict, str | None]:
    """Compute the cost/time field values for one domain's trajectory file.

    Returns (domain_name, {field: value}, estimate_note). Values include None
    so a field that can't be derived is explicitly recorded as absent;
    estimate_note describes the pricing used when the agent cost had to be
    estimated from token usage.
    """
    with open(results_path) as f:
        raw = json.load(f)
    _repair_trimmed_messages(raw)
    results = TrajectoryResults.model_validate(raw)
    domain = results.info.environment_info.domain_name
    estimate_note = estimate_agent_costs_from_usage(results)
    metrics = compute_metrics(results)
    domain_results = build_domain_results(metrics, include_time=True)
    values = {field: getattr(domain_results, field) for field in COST_TIME_FIELDS}
    # An agent cost of exactly 0.0 means per-message costs were never recorded
    # (e.g. self-hosted models); report the cost as unknown, not free.
    if values["cost"] == 0.0:
        values["cost"] = None
        values["total_cost"] = None
    return domain, values, estimate_note


def backfill_submission(submission_name: str, cache_dir: Path, dry_run: bool) -> bool:
    """Compute and patch cost/time metrics for one submission."""
    submission_file = SUBMISSIONS_DIR / submission_name / SUBMISSION_FILE_NAME
    if not submission_file.exists():
        console.print(f"  [red]No submission.json at {submission_file}[/red]")
        return False

    with open(submission_file) as f:
        data = json.load(f)

    if data.get("modality", "text") != "text":
        console.print("  [yellow]Skipping: not a text submission[/yellow]")
        return True
    if not data.get("trajectories_available", False):
        console.print("  [yellow]Skipping: trajectories not available[/yellow]")
        return True

    trajectories_dir = sync_trajectories(submission_name, cache_dir)
    results_files = sorted(trajectories_dir.glob("*.json"))
    if not results_files:
        console.print(
            "  [yellow]Skipping: no trajectory files on S3 "
            "(submission never uploaded trajectories)[/yellow]"
        )
        return True

    changed = False
    estimated_domains: dict[str, str] = {}
    for results_path in results_files:
        try:
            domain, values, estimate_note = compute_domain_cost_time(results_path)
            if estimate_note:
                estimated_domains[domain] = estimate_note
        except Exception as e:
            # Truncate: pydantic errors on big files can be megabytes long,
            # and rich takes minutes to render them.
            console.print(f"  [red]{results_path.name}: {str(e)[:500]}[/red]")
            return False

        domain_block = data.get("results", {}).get(domain)
        if domain_block is None:
            console.print(
                f"  [yellow]{domain}: in trajectories but not in submission.json; "
                "skipping[/yellow]"
            )
            continue

        summary = []
        for field, value in values.items():
            old = domain_block.get(field)
            if value is None:
                # Never erase an existing manually-provided value — except a
                # bogus 0.0 written by an earlier run of this script.
                if old == 0:
                    domain_block[field] = None
                    summary.append(f"{field}: {old} -> null")
                    changed = True
                continue
            if old is not None and old != value:
                summary.append(f"{field}: {old} -> {value}")
            elif old is None:
                summary.append(f"{field}: {value}")
            domain_block[field] = value
            changed = changed or old != value

        console.print(
            f"  {domain}: " + ("; ".join(summary) if summary else "no change")
        )

    if estimated_domains:
        methodology = data.setdefault("methodology", {})
        notes = methodology.get("notes") or ""
        if ESTIMATE_NOTE_MARKER not in notes:
            today = datetime.date.today().isoformat()
            pricing_detail = "; ".join(sorted(set(estimated_domains.values())))
            note = (
                f"Agent cost ({', '.join(sorted(estimated_domains))}) backfilled "
                f"on {today}: {ESTIMATE_NOTE_MARKER} at official list prices "
                f"({pricing_detail}); the original run did not record "
                "per-message costs."
            )
            methodology["notes"] = f"{notes} {note}".strip()
            changed = True
            console.print(f"  [cyan]Added methodology note: {note}[/cyan]")

    if dry_run:
        console.print("  [yellow]Dry run: submission.json not modified[/yellow]")
        return True

    if changed:
        with open(submission_file, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        console.print(f"  [green]Patched {submission_file}[/green]")
    else:
        console.print("  No changes needed")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Backfill cost/time metrics for text submissions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--submissions",
        nargs="*",
        default=None,
        help="Submission directory names to backfill "
        "(default: all text submissions in the manifest)",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help=f"Local cache for downloaded trajectories (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print metrics without modifying submission.json files",
    )
    args = parser.parse_args()

    if args.submissions:
        submission_names = args.submissions
    else:
        with open(SUBMISSIONS_DIR / MANIFEST_FILE_NAME) as f:
            manifest = json.load(f)
        submission_names = manifest.get("submissions", [])

    console.print(
        f"[bold blue]Backfilling {len(submission_names)} text submission(s)[/bold blue]"
    )

    failures = []
    for name in submission_names:
        console.print(f"\n[bold]{name}[/bold]")
        try:
            if not backfill_submission(name, Path(args.cache_dir), args.dry_run):
                failures.append(name)
        except Exception as e:
            console.print(f"  [red]FAILED: {str(e)[:500]}[/red]")
            failures.append(name)

    if failures:
        console.print(f"\n[red bold]Failed: {failures}[/red bold]")
        sys.exit(1)
    console.print("\n[green bold]All submissions backfilled.[/green bold]")


if __name__ == "__main__":
    main()
