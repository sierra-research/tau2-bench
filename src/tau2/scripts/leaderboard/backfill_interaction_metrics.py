"""
Maintainer tool: backfill interaction metrics for existing voice submissions.

For each voice submission in the leaderboard manifest, downloads the public
trajectories from S3 (cached locally), computes the interaction_metrics block,
and patches the corresponding web/leaderboard/public/submissions/<dir>/
submission.json. Trajectories on S3 are never modified; the patched
submission.json files are committed via PR and synced to S3 by CI.

Usage:
    python -m tau2.scripts.leaderboard.backfill_interaction_metrics \\
        [--submissions DIR ...] [--cache-dir PATH] [--dry-run]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from rich.console import Console

from tau2.scripts.leaderboard.compute_interaction_metrics import (
    compute_interaction_metrics_block,
    discover_experiment_paths,
    print_interaction_metrics,
)
from tau2.scripts.leaderboard.submission import (
    MANIFEST_FILE_NAME,
    SUBMISSION_FILE_NAME,
    TRAJECTORY_FILES_DIR_NAME,
    InteractionMetrics,
)

S3_BUCKET = "sierra-tau-bench-public"
S3_PREFIX = "submissions"

REPO_ROOT = Path(__file__).resolve().parents[4]
SUBMISSIONS_DIR = REPO_ROOT / "web" / "leaderboard" / "public" / "submissions"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "tau2" / "backfill_trajectories"

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


def backfill_submission(submission_name: str, cache_dir: Path, dry_run: bool) -> bool:
    """Compute and patch interaction metrics for one submission."""
    submission_file = SUBMISSIONS_DIR / submission_name / SUBMISSION_FILE_NAME
    if not submission_file.exists():
        console.print(f"  [red]No submission.json at {submission_file}[/red]")
        return False

    trajectories_dir = sync_trajectories(submission_name, cache_dir)
    experiment_paths = discover_experiment_paths([str(trajectories_dir)])
    block = compute_interaction_metrics_block(experiment_paths)

    # Validate against the schema before writing
    InteractionMetrics.model_validate(block)
    print_interaction_metrics(block)

    if dry_run:
        console.print("  [yellow]Dry run: submission.json not modified[/yellow]")
        return True

    with open(submission_file) as f:
        data = json.load(f)
    data["interaction_metrics"] = block
    with open(submission_file, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    console.print(f"  [green]Patched {submission_file}[/green]")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Backfill interaction metrics for voice submissions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--submissions",
        nargs="*",
        default=None,
        help="Submission directory names to backfill "
        "(default: all voice submissions in the manifest)",
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
        submission_names = manifest.get("voice_submissions", [])

    console.print(
        f"[bold blue]Backfilling {len(submission_names)} voice submission(s)[/bold blue]"
    )

    failures = []
    for name in submission_names:
        console.print(f"\n[bold]{name}[/bold]")
        try:
            if not backfill_submission(name, Path(args.cache_dir), args.dry_run):
                failures.append(name)
        except Exception as e:
            console.print(f"  [red]FAILED: {e}[/red]")
            failures.append(name)

    if failures:
        console.print(f"\n[red bold]Failed: {failures}[/red bold]")
        sys.exit(1)
    console.print("\n[green bold]All submissions backfilled.[/green bold]")


if __name__ == "__main__":
    main()
