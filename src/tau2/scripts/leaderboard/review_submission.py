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

    # 2.0 package (benchmark-submission.json): the source is the submitter's
    # saved matrix root; cells upload to trajectories/<language>/<domain>/.
    python -m tau2.scripts.leaderboard.review_submission \\
        web/leaderboard/public/submissions/my-voice-model_org_2026-10-20 \\
        /tmp/downloaded-matrix-root/ --upload

    # The frozen 2026-10-02 release export, with its saved verdicts
    python -m tau2.scripts.leaderboard.review_submission \\
        web/leaderboard/public/submissions/<name> /path/to/release-export \\
        --historical-voice-v2 --judges-root /path/to/verdicts/<provider> \\
        [--judges-root ...]
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.request import urlopen

from rich.console import Console

from tau2.data_model.simulation import Results as TrajectoryResults
from tau2.metrics.voice_interaction_metrics import NoVoiceTicksError
from tau2.scripts.leaderboard.benchmark_package import (
    HISTORICAL_PASS3_DIRS,
    PACKAGE_FILE_NAMES,
    BenchmarkPackage,
    evidence_cells,
    is_package,
    load_package,
)
from tau2.scripts.leaderboard.compute_interaction_metrics import (
    build_interaction_metrics_block,
    compute_metrics_for_loaded_results,
    config_with_resolved_ticks,
    resolve_experiment_config,
)
from tau2.scripts.leaderboard.prepare_submission import (
    validate_benchmark_submission,
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
from tau2.scripts.leaderboard.viewer_data import sync as sync_viewer_tree
from tau2.scripts.leaderboard.viewer_data import viewer_cell_dir, write_viewer_cell
from tau2.scripts.leaderboard.voice_v2_publication import (
    validate_historical_package,
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


def _compute_interaction_metrics(
    results_list: list[TrajectoryResults],
    console: Console,
) -> dict | None:
    """Recompute voice interaction metrics from the reviewed trajectories.

    Maintainer-side recomputation is the anti-gaming guarantee: the values on
    the leaderboard come from the submitted tick data, not from self-reported
    numbers.
    """
    domain_metrics: dict[str, dict] = {}
    resolved_configs = []
    for results in results_list:
        domain = results.info.environment_info.domain_name
        if domain in domain_metrics:
            raise ValueError(f"Domain {domain} appears in multiple trajectory files")
        try:
            domain_metrics[domain] = compute_metrics_for_loaded_results(results)
            resolved_configs.append(resolve_experiment_config(results))
            console.print(f"  [green]OK[/green] {domain}")
        except NoVoiceTicksError:
            console.print(
                f"  [yellow]WARN[/yellow] {domain}: no tick data, "
                "skipping interaction metrics for this domain"
            )
    if not domain_metrics:
        return None
    return build_interaction_metrics_block(
        domain_metrics, config_with_resolved_ticks(None, resolved_configs)
    )


def _update_submission_json(
    submission_dir: Path,
    trajectory_map: dict[str, str],
    console: Console,
    interaction_metrics: dict | None = None,
) -> None:
    """Update submission.json with trajectory references and recomputed metrics."""
    submission_file = submission_dir / SUBMISSION_FILE_NAME
    with open(submission_file) as f:
        data = json.load(f)

    data["trajectories_available"] = True
    data["trajectory_files"] = trajectory_map
    if interaction_metrics is not None:
        if data.get("interaction_metrics") not in (None, interaction_metrics):
            console.print(
                "  [yellow]Replacing submitter-provided interaction_metrics "
                "with recomputed values[/yellow]"
            )
        data["interaction_metrics"] = interaction_metrics

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


def _upload_viewer_cells(
    submission_dir: Path,
    trajectory_files: list[str],
    trajectory_map: dict[str, str],
    profile: str,
    console: Console,
) -> None:
    """Split a text submission's trajectory files into viewer cells and sync them."""
    with tempfile.TemporaryDirectory() as staging:
        for path_str, domain in zip(trajectory_files, trajectory_map):
            cell = (
                Path(staging)
                / submission_dir.name
                / TRAJECTORY_FILES_DIR_NAME
                / viewer_cell_dir(domain)
            )
            count = write_viewer_cell(json.loads(Path(path_str).read_bytes()), cell)
            console.print(f"  {domain}: {count} simulations")
        # The staging dir holds only this submission's text cells.
        sync_viewer_tree(Path(staging), profile, dry_run=False)


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

        # The visualizer reads text trajectories split per simulation, next
        # to the monolithic files (which stay the download). A failure here
        # leaves the visualizer on the monolithic files, so it must not stop
        # submission.json from being uploaded.
        console.print("\n[bold]Uploading viewer cells to S3...[/bold]")
        try:
            _upload_viewer_cells(
                submission_dir, trajectory_files, trajectory_map, profile, console
            )
        except Exception as error:
            console.print(
                f"[yellow]Viewer cell upload failed ({error}); rerun "
                "`viewer_data split` + `upload` for this submission.[/yellow]"
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


def _package_uploads(
    package: BenchmarkPackage,
    source: Path,
    judges_roots: list[Path],
) -> list[tuple[Path, str]]:
    """Local evidence → S3 key under ``submissions/<name>/``, language layer first.

    Cells go to ``trajectories/<language>/<domain>/`` for both evidence layouts.
    A matrix root's own files (frame, plan, execution record) go to
    ``trajectories/``. For the historical export, its ``correction.json`` goes
    to ``source_correction.json`` and saved verdicts to
    ``judges/<agent_provider>/<language>/``, the layout of the 2026-10-09 Gemini
    upload. An entity package's other two crossed Pass^3 conditions go to
    ``pass3/<language>/intake/<condition>/``.
    """
    cells = [(cell.language, cell.domain) for cell in package.submission.frame.cells]
    uploads = [
        (local, f"{TRAJECTORY_FILES_DIR_NAME}/{key}/")
        for key, local in evidence_cells(
            source, package.metadata.evidence, cells
        ).items()
    ]
    if package.metadata.evidence == "matrix":
        uploads.extend(
            (path, f"{TRAJECTORY_FILES_DIR_NAME}/{path.name}")
            for path in sorted(source.glob("*.json"))
        )
    else:
        # Crossed Pass^3 cites the other two intake conditions too.
        if package.pass3 is not None:
            uploads.extend(
                (
                    source / "trajectories" / language / directory,
                    f"pass3/{language}/{directory}/",
                )
                for language, _ in cells
                for directory in HISTORICAL_PASS3_DIRS.values()
            )
        if (source / "correction.json").is_file():
            uploads.append((source / "correction.json", "source_correction.json"))
        languages = dict.fromkeys(language for language, _ in cells)
        provider = package.metadata.agent_provider
        uploads.extend(
            (root / language, f"judges/{provider}/{language}/")
            for root in judges_roots
            for language in languages
            if (root / language).is_dir()
        )
    missing = [str(local) for local, _ in uploads if not local.exists()]
    if missing:
        raise ValueError("Missing evidence: " + ", ".join(missing))
    return uploads


def _review_package(
    package_dir: Path, source: Path, args: argparse.Namespace, console: Console
) -> None:
    """Validate a 2.0 package against its evidence, then (optionally) upload."""
    prefix = _s3_key_prefix(package_dir)
    if args.verify_only:
        package = load_package(package_dir)
    elif args.historical_voice_v2:
        console.print("\n[bold]Recomputing from the frozen release export...[/bold]")
        package = validate_historical_package(package_dir, source, args.judges_root)
    else:
        console.print("\n[bold]Recomputing from the saved matrix...[/bold]")
        package = validate_benchmark_submission(package_dir, source)
    summary = package.submission.summary
    console.print(
        f"  {package.metadata.model_name} ({package.metadata.model_organization}): "
        f"{summary.benchmark_version} pass@1 {100 * summary.score:.1f}% over "
        f"{summary.scored_trials} calls"
    )
    if package.name != package_dir.name:
        raise ValueError(f"Package directory must be named {package.name}")
    uploads = _package_uploads(package, source, args.judges_root)
    if args.verify_only:
        _verify_package_upload(package, prefix, console)
        return
    console.print(f"\n[bold]Upload plan (s3://{S3_BUCKET}/{prefix}/):[/bold]")
    for local, key in uploads:
        console.print(f"  {local} -> {key}")
    if not args.upload:
        console.print(
            "\n[yellow]Dry run complete. Re-run with --upload to push to S3.[/yellow]"
        )
        return
    failed = False
    for local, key in uploads:
        command = "sync" if local.is_dir() else "cp"
        failed |= not _run_aws(
            ["s3", command, str(local), f"s3://{S3_BUCKET}/{prefix}/{key}"],
            args.aws_profile,
            console,
        )
    if failed:
        raise SystemExit("Upload failed; benchmark-metadata.json was not updated")
    package.metadata.trajectories = f"s3://{S3_BUCKET}/{prefix}"
    package.write(package_dir)
    for name in PACKAGE_FILE_NAMES:
        if (package_dir / name).is_file():
            _run_aws(
                [
                    "s3",
                    "cp",
                    str(package_dir / name),
                    f"s3://{S3_BUCKET}/{prefix}/{name}",
                ],
                args.aws_profile,
                console,
            )
    console.print(f"  Set trajectories to {package.metadata.trajectories}")
    _verify_package_upload(package, prefix, console)


def _verify_package_upload(
    package: BenchmarkPackage, prefix: str, console: Console
) -> None:
    """Every cell's results.json is publicly readable at the language layer."""
    console.print("\n[bold]Verifying S3 upload...[/bold]")
    base = f"https://{S3_BUCKET}.s3.us-west-2.amazonaws.com/{prefix}"
    missing = []
    for cell in package.submission.frame.cells:
        url = (
            f"{base}/{TRAJECTORY_FILES_DIR_NAME}/{cell.language}/{cell.domain}/"
            "results.json"
        )
        try:
            with urlopen(url, timeout=30):
                pass
        except OSError:
            missing.append(f"{cell.language}/{cell.domain}")
    if missing:
        raise SystemExit("Not publicly readable: " + ", ".join(missing))
    console.print(f"  All {len(package.submission.frame.cells)} cells are readable")


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
    parser.add_argument(
        "--historical-voice-v2",
        action="store_true",
        help="2.0 package citing the frozen 2026-10-02 release export",
    )
    parser.add_argument(
        "--judges-root",
        type=Path,
        action="append",
        default=[],
        help="With --historical-voice-v2: saved verdicts (<language>/<id>.json)",
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

    if is_package(submission_dir):
        _review_package(submission_dir, source, args, console)
        return
    if args.historical_voice_v2 or args.judges_root:
        raise ValueError("--historical-voice-v2 applies only to 2.0 packages")

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

    # Recompute interaction metrics from the reviewed trajectories (voice only)
    interaction_metrics = None
    if is_voice:
        console.print(
            "\n[bold]Step 5b: Recomputing voice interaction metrics...[/bold]"
        )
        interaction_metrics = _compute_interaction_metrics(results, console)

    # Update submission.json
    console.print("\n[bold]Step 6: Updating submission.json...[/bold]")
    _update_submission_json(
        submission_dir, trajectory_map, console, interaction_metrics
    )

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
