"""Maintainer tool for reviewing and uploading leaderboard submissions.

Usage:
    python -m tau2.scripts.leaderboard.review_submission \\
        <submission-dir> <trajectory-source> \\
        [--aws-profile PROFILE] [--upload] [--verify-only]

Example:
    # Validate only (dry run)
    python -m tau2.scripts.leaderboard.review_submission \\
        web/leaderboard/public/submissions/my-model_org_2026-01-01 \\
        /tmp/downloaded-trajectories/

    # Validate + upload to S3
    python -m tau2.scripts.leaderboard.review_submission \\
        web/leaderboard/public/submissions/my-model_org_2026-01-01 \\
        /tmp/downloaded-trajectories/ \\
        --upload --aws-profile tau-bench-ci

    # Verify an existing S3 upload
    python -m tau2.scripts.leaderboard.review_submission \\
        web/leaderboard/public/submissions/my-model_org_2026-01-01 \\
        /tmp/downloaded-trajectories/ \\
        --verify-only --aws-profile tau-bench-ci
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen

from rich.console import Console

from tau2.data_model.simulation import Results as TrajectoryResults
from tau2.scripts.leaderboard.prepare_submission import (
    validate_submission_metrics,
    validate_submission_traj_set,
)
from tau2.scripts.leaderboard.submission import (
    SUBMISSION_FILE_NAME,
    TRAJECTORY_FILES_DIR_NAME,
    Submission,
)
from tau2.scripts.leaderboard.verify_trajectories import (
    VerificationMode,
    verify_single_file,
)
from tau2.utils.io_utils import expand_paths

S3_BUCKET = "sierra-tau-bench-public"
S3_PREFIX = "submissions"
S3_PUBLIC_URL = f"https://{S3_BUCKET}.s3.us-west-2.amazonaws.com/{S3_PREFIX}"


def _load_submission(submission_dir: Path, console: Console) -> Submission:
    submission_file = submission_dir / SUBMISSION_FILE_NAME
    if not submission_file.exists():
        console.print(f"[red]submission.json not found in {submission_dir}[/red]")
        sys.exit(1)
    with open(submission_file) as f:
        return Submission.model_validate_json(f.read())


def _discover_trajectory_files(
    source: Path, is_voice: bool, console: Console
) -> list[str]:
    """Find trajectory results.json files in the source directory."""
    if is_voice:
        # Voice: look for results.json inside experiment directories
        files = sorted(source.glob("*/results.json"))
        if not files:
            files = sorted(source.glob("results.json"))
    else:
        files = expand_paths([source], extension=".json")
        files = [str(f) for f in files] if not isinstance(files[0], str) else files
        files = [Path(f) for f in files]

    if not files:
        console.print("[red]No trajectory files found in source directory[/red]")
        sys.exit(1)

    console.print(f"  Found {len(files)} trajectory file(s):")
    for f in files:
        size_mb = Path(f).stat().st_size / 1e6
        console.print(f"    {f} ({size_mb:.1f} MB)")
    return [str(f) for f in files]


def _validate(
    submission: Submission,
    trajectory_files: list[str],
    console: Console,
) -> list[TrajectoryResults]:
    """Run all validation checks. Returns loaded results."""
    results = []
    console.print("\n[bold]Step 2: Loading trajectory data...[/bold]")
    for path in trajectory_files:
        try:
            r = TrajectoryResults.load(path)
            domain = r.info.environment_info.domain_name
            n_tasks = len(r.tasks)
            n_sims = len(r.simulations)
            console.print(
                f"  [green]OK[/green] {domain}: {n_tasks} tasks, {n_sims} simulations"
            )
            results.append(r)
        except Exception as e:
            console.print(f"  [red]FAIL[/red] {path}: {e}")
            sys.exit(1)

    console.print("\n[bold]Step 3: Validating trajectory set...[/bold]")
    valid, error = validate_submission_traj_set(results)
    if not valid:
        console.print(f"  [red]FAIL: {error}[/red]")
        sys.exit(1)
    console.print("  [green]OK[/green] Trajectory set is consistent")

    console.print("\n[bold]Step 4: Verifying trajectory contents...[/bold]")
    all_passed = True
    for path in trajectory_files:
        passed = verify_single_file(path, VerificationMode.PUBLIC, console)
        if not passed:
            all_passed = False
    if not all_passed:
        console.print("  [red]Trajectory verification failed[/red]")
        sys.exit(1)
    console.print("  [green]OK[/green] All trajectory files verified")

    console.print("\n[bold]Step 5: Validating metrics...[/bold]")
    validate_submission_metrics(submission, results, console)

    return results


def _build_trajectory_map(
    results_list: list[TrajectoryResults],
    trajectory_files: list[str],
    is_voice: bool,
) -> dict[str, str]:
    """Build the domain -> filename/dirname mapping for submission.json."""
    mapping = {}
    for r, path in zip(results_list, trajectory_files):
        domain = r.info.environment_info.domain_name
        p = Path(path)
        if is_voice:
            mapping[domain] = p.parent.name
        else:
            if p.name == "results.json":
                mapping[domain] = f"{domain}_results.json"
            else:
                mapping[domain] = p.name
    return mapping


def _update_submission_json(
    submission_dir: Path,
    trajectory_map: dict[str, str],
    console: Console,
) -> None:
    """Update submission.json with trajectory references."""
    submission_file = submission_dir / SUBMISSION_FILE_NAME
    with open(submission_file) as f:
        data = json.load(f)

    data["trajectories_available"] = True
    data["trajectory_files"] = trajectory_map

    with open(submission_file, "w") as f:
        json.dump(data, f, indent=2, default=str)
        f.write("\n")

    console.print(f"  Updated {submission_file}")
    console.print(f"  trajectory_files: {json.dumps(trajectory_map, indent=4)}")


def _s3_key_prefix(submission_dir: Path) -> str:
    """Get the S3 key prefix for a submission directory."""
    return f"{S3_PREFIX}/{submission_dir.name}"


def _run_aws(cmd: list[str], profile: str, console: Console) -> bool:
    """Run an AWS CLI command. Returns True on success."""
    full_cmd = ["aws", *cmd, "--profile", profile]
    console.print(f"  [dim]$ {' '.join(full_cmd)}[/dim]")
    result = subprocess.run(full_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        console.print(f"  [red]AWS error: {result.stderr.strip()}[/red]")
        return False
    if result.stdout.strip():
        console.print(f"  {result.stdout.strip()}")
    return True


def _upload(
    submission_dir: Path,
    trajectory_files: list[str],
    trajectory_map: dict[str, str],
    is_voice: bool,
    profile: str,
    console: Console,
) -> None:
    """Upload trajectories and submission.json to S3."""
    s3_prefix = _s3_key_prefix(submission_dir)
    traj_prefix = f"{s3_prefix}/{TRAJECTORY_FILES_DIR_NAME}"

    console.print("\n[bold]Uploading trajectories to S3...[/bold]")

    if is_voice:
        for path_str in trajectory_files:
            p = Path(path_str)
            exp_dir = p.parent
            exp_name = exp_dir.name
            s3_dest = f"s3://{S3_BUCKET}/{traj_prefix}/{exp_name}/"
            console.print(f"  Syncing {exp_dir.name}/...")
            _run_aws(
                ["s3", "sync", str(exp_dir), s3_dest],
                profile,
                console,
            )
    else:
        for path_str, (domain, dest_name) in zip(
            trajectory_files, trajectory_map.items()
        ):
            s3_dest = f"s3://{S3_BUCKET}/{traj_prefix}/{dest_name}"
            console.print(f"  Uploading {dest_name}...")
            _run_aws(
                ["s3", "cp", path_str, s3_dest],
                profile,
                console,
            )

    console.print("\n[bold]Uploading submission.json to S3...[/bold]")
    local_sub = str(submission_dir / SUBMISSION_FILE_NAME)
    s3_sub = f"s3://{S3_BUCKET}/{s3_prefix}/{SUBMISSION_FILE_NAME}"
    _run_aws(["s3", "cp", local_sub, s3_sub], profile, console)


def _verify_upload(
    submission_dir: Path,
    trajectory_files: list[str],
    trajectory_map: dict[str, str],
    is_voice: bool,
    profile: str,
    console: Console,
) -> None:
    """Verify uploaded files are accessible and valid."""
    all_ok = True

    console.print("\n[bold]Verifying S3 upload...[/bold]")

    # Verify submission.json
    sub_url = f"{S3_PUBLIC_URL}/{submission_dir.name}/{SUBMISSION_FILE_NAME}"
    try:
        with urlopen(sub_url) as resp:
            data = json.loads(resp.read())
            if data.get("trajectories_available") and data.get("trajectory_files"):
                console.print(
                    f"  [green]OK[/green] submission.json (trajectories_available=true)"
                )
            else:
                console.print(
                    f"  [red]FAIL[/red] submission.json missing trajectory fields"
                )
                all_ok = False
    except Exception as e:
        console.print(f"  [red]FAIL[/red] submission.json: {e}")
        all_ok = False

    # Verify each trajectory file
    for domain, dest_name in trajectory_map.items():
        if is_voice:
            url = f"{S3_PUBLIC_URL}/{submission_dir.name}/{TRAJECTORY_FILES_DIR_NAME}/{dest_name}/results.json"
        else:
            url = f"{S3_PUBLIC_URL}/{submission_dir.name}/{TRAJECTORY_FILES_DIR_NAME}/{dest_name}"

        # Find the corresponding local file for size comparison
        local_path = None
        for path_str in trajectory_files:
            p = Path(path_str)
            r = TrajectoryResults.load(path_str)
            if r.info.environment_info.domain_name == domain:
                local_path = p
                break

        try:
            with urlopen(url) as resp:
                s3_size = int(resp.headers.get("Content-Length", 0))
                local_size = local_path.stat().st_size if local_path else 0

                size_match = abs(s3_size - local_size) < 1024 if local_path else True
                size_str = f"{s3_size / 1e6:.1f} MB"
                if local_path and not size_match:
                    console.print(
                        f"  [yellow]WARN[/yellow] {domain} ({dest_name}): "
                        f"S3 size {size_str} != local {local_size / 1e6:.1f} MB"
                    )
                    all_ok = False
                else:
                    console.print(
                        f"  [green]OK[/green] {domain} ({dest_name}): {size_str}"
                    )

                # Try parsing JSON for smaller files
                if s3_size < 100 * 1e6:
                    content = resp.read()
                    json.loads(content)
                    console.print(f"       JSON parse: OK")
        except Exception as e:
            console.print(f"  [red]FAIL[/red] {domain} ({dest_name}): {e}")
            all_ok = False

    if all_ok:
        console.print("\n[green bold]All uploads verified successfully![/green bold]")
    else:
        console.print(
            "\n[red bold]Some verifications failed. Check output above.[/red bold]"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Review and upload a leaderboard submission (maintainer tool).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "submission_dir",
        help="Path to the submission directory (contains submission.json)",
    )
    parser.add_argument(
        "trajectory_source",
        help="Path to the downloaded trajectory files",
    )
    parser.add_argument(
        "--aws-profile",
        default="tau-bench-ci",
        help="AWS CLI profile for S3 operations (default: tau-bench-ci)",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload trajectories and updated submission.json to S3",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify an existing S3 upload (skip validation and upload)",
    )
    args = parser.parse_args()

    console = Console()
    submission_dir = Path(args.submission_dir)
    source = Path(args.trajectory_source)

    console.print(
        f"\n[bold blue]Reviewing submission: {submission_dir.name}[/bold blue]"
    )
    console.print(f"  Submission dir: {submission_dir}")
    console.print(f"  Trajectory source: {source}")

    # Step 1: Load submission.json
    console.print("\n[bold]Step 1: Loading submission.json...[/bold]")
    submission = _load_submission(submission_dir, console)
    is_voice = submission.modality == "voice"
    console.print(f"  Model: {submission.model_name}")
    console.print(f"  Organization: {submission.model_organization}")
    console.print(f"  Modality: {submission.modality}")
    console.print(f"  Submission type: {submission.submission_type}")

    # Discover trajectory files
    console.print("\n[bold]Discovering trajectory files...[/bold]")
    trajectory_files = _discover_trajectory_files(source, is_voice, console)

    if args.verify_only:
        # Build map from existing submission or from files
        if submission.trajectory_files:
            trajectory_map = dict(submission.trajectory_files)
        else:
            results = [TrajectoryResults.load(f) for f in trajectory_files]
            trajectory_map = _build_trajectory_map(results, trajectory_files, is_voice)
        _verify_upload(
            submission_dir,
            trajectory_files,
            trajectory_map,
            is_voice,
            args.aws_profile,
            console,
        )
        return

    # Validate
    results = _validate(submission, trajectory_files, console)

    # Build trajectory map
    trajectory_map = _build_trajectory_map(results, trajectory_files, is_voice)
    console.print(f"\n[bold]Trajectory file mapping:[/bold]")
    for domain, name in trajectory_map.items():
        console.print(f"  {domain} -> {name}")

    # Update submission.json
    console.print("\n[bold]Step 6: Updating submission.json...[/bold]")
    _update_submission_json(submission_dir, trajectory_map, console)

    if args.upload:
        # Upload
        console.print("\n[bold]Step 7: Uploading to S3...[/bold]")
        _upload(
            submission_dir,
            trajectory_files,
            trajectory_map,
            is_voice,
            args.aws_profile,
            console,
        )

        # Verify
        console.print("\n[bold]Step 8: Verifying upload...[/bold]")
        _verify_upload(
            submission_dir,
            trajectory_files,
            trajectory_map,
            is_voice,
            args.aws_profile,
            console,
        )
    else:
        console.print(
            "\n[yellow]Dry run complete. Re-run with --upload to push to S3.[/yellow]"
        )

    console.print("\n[green bold]Done![/green bold]")


if __name__ == "__main__":
    main()
