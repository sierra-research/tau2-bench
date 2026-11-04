"""Maintainer tool: stage trajectories in the layout the web visualizer reads.

The visualizer lists a cell from a small ``results.json`` index (metadata,
tasks and a ``simulation_index``) and fetches ``simulations/<id>.json`` only
when a conversation is opened. Voice runs are saved that way already. Text
submissions upload one monolithic results file per domain (20–330 MB), which
the browser would otherwise download whole before showing anything.

``split`` writes each text submission's domain file as a viewer cell next to it
(``trajectories/<domain>/``), every JSON file gzipped. ``upload`` syncs the
staging tree to S3 with ``Content-Encoding: gzip`` and long-lived caching for
simulation files, which never change once published. The original monolithic
files stay where they are: they remain the download.

Only text cells are staged. Voice cells are already in this layout and stay
uncompressed on S3, because ``aws s3 sync`` and Python readers do not undo
``Content-Encoding: gzip``; ``upload`` refuses a staging tree holding anything
other than text submissions' viewer cells.

Usage:
    # Split every text submission with trajectories (downloads from S3)
    python -m tau2.scripts.leaderboard.viewer_data split --staging /tmp/viewer

    # Only some submissions, from local files instead of S3
    python -m tau2.scripts.leaderboard.viewer_data split --staging /tmp/viewer \\
        --source /path/to/trajectories gpt-5-5_sierra_2026-05-05

    # Upload the staging tree
    python -m tau2.scripts.leaderboard.viewer_data upload --staging /tmp/viewer \\
        --aws-profile tau-bench-ci
"""

import argparse
import gzip
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from tau2.data_model.simulation import SimulationIndexEntry
from tau2.scripts.leaderboard.submission import (
    MANIFEST_FILE_NAME,
    SUBMISSION_FILE_NAME,
    TRAJECTORY_FILES_DIR_NAME,
)

S3_BUCKET = "sierra-tau-bench-public"
S3_PREFIX = "submissions"
S3_ORIGIN = f"https://{S3_BUCKET}.s3.us-west-2.amazonaws.com"
S3_PUBLIC_URL = f"{S3_ORIGIN}/{S3_PREFIX}"

SUBMISSIONS_DIR = (
    Path(__file__).resolve().parents[4]
    / "web"
    / "leaderboard"
    / "public"
    / "submissions"
)

RESULTS_FILE_NAME = "results.json"
SIMULATIONS_DIR_NAME = "simulations"

# Simulation files are content-addressed by simulation id and never rewritten;
# an index can be corrected (e.g. a re-scored cell), so it revalidates hourly.
SIMULATION_CACHE_CONTROL = "public, max-age=31536000, immutable"
INDEX_CACHE_CONTROL = "public, max-age=3600"


def viewer_cell_dir(domain: str) -> str:
    """Path of a text submission's viewer cell under its ``trajectories/``."""
    return domain


def split_results(data: dict) -> tuple[dict, list[dict]]:
    """Split a monolithic results file into a viewer index and simulations.

    The index keeps every top-level field except ``simulations`` and adds a
    ``simulation_index`` with one :class:`SimulationIndexEntry` per simulation.
    Simulations are returned unchanged.
    """
    simulations = data.get("simulations") or []
    index = [
        SimulationIndexEntry(
            id=sim["id"],
            task_id=sim["task_id"],
            trial=sim.get("trial") or 0,
            reward=(sim.get("reward_info") or {}).get("reward"),
            termination_reason=sim.get("termination_reason"),
            agent_cost=sim.get("agent_cost"),
            duration=sim.get("duration"),
        ).model_dump(mode="json")
        for sim in simulations
    ]
    meta = {k: v for k, v in data.items() if k != "simulations"}
    meta["simulation_index"] = index
    return meta, simulations


def _write_gzip_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    path.write_bytes(gzip.compress(payload.encode("utf-8"), mtime=0))


def write_viewer_cell(data: dict, cell_dir: Path) -> int:
    """Write ``data`` as a gzipped viewer cell; returns the simulation count."""
    meta, simulations = split_results(data)
    if cell_dir.exists():
        shutil.rmtree(cell_dir)
    for sim in simulations:
        _write_gzip_json(cell_dir / SIMULATIONS_DIR_NAME / f"{sim['id']}.json", sim)
    # The index goes last: a cell with an index is a complete cell.
    _write_gzip_json(cell_dir / RESULTS_FILE_NAME, meta)
    return len(simulations)


def _fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, open(dest, "wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)


def _text_submissions(names: list[str]) -> list[str]:
    manifest = json.loads((SUBMISSIONS_DIR / MANIFEST_FILE_NAME).read_text())
    listed = manifest.get("submissions") or []
    if not names:
        return listed
    unknown = sorted(set(names) - set(listed))
    if unknown:
        raise SystemExit(f"Not text submissions in the manifest: {unknown}")
    return names


def split(staging: Path, names: list[str], source: str | None) -> None:
    """Stage a viewer cell for each domain of each text submission."""
    for name in _text_submissions(names):
        sub = json.loads((SUBMISSIONS_DIR / name / SUBMISSION_FILE_NAME).read_text())
        if not (sub.get("trajectories_available") and sub.get("trajectory_files")):
            continue
        for domain, file_name in sub["trajectory_files"].items():
            cell_dir = (
                staging / name / TRAJECTORY_FILES_DIR_NAME / viewer_cell_dir(domain)
            )
            if (cell_dir / RESULTS_FILE_NAME).exists():
                print(f"  skip {name} {domain} (staged)")
                continue
            with tempfile.TemporaryDirectory() as tmp:
                if source:
                    local = Path(source) / file_name
                else:
                    local = Path(tmp) / file_name
                    url = f"{S3_PUBLIC_URL}/{name}/{TRAJECTORY_FILES_DIR_NAME}/{file_name}"
                    print(f"  fetch {url}")
                    try:
                        _fetch(url, local)
                    except urllib.error.HTTPError as error:
                        print(f"  MISSING {name} {domain}: HTTP {error.code}")
                        continue
                data = json.loads(local.read_bytes())
            count = write_viewer_cell(data, cell_dir)
            print(f"  {name} {domain}: {count} simulations")


def check_staging(staging: Path, submissions_dir: Path = SUBMISSIONS_DIR) -> None:
    """Raise unless every staged file is in a text submission's viewer cell.

    Anything else would be uploaded gzip-encoded over the S3 object at the same
    key, which breaks readers that do not decode ``Content-Encoding``.
    """
    manifest = json.loads((submissions_dir / MANIFEST_FILE_NAME).read_text())
    cells = set()
    for name in manifest.get("submissions") or []:
        sub = json.loads((submissions_dir / name / SUBMISSION_FILE_NAME).read_text())
        for domain in sub.get("trajectory_files") or {}:
            cells.add((name, TRAJECTORY_FILES_DIR_NAME, viewer_cell_dir(domain)))
    stray = [
        path
        for path in staging.rglob("*")
        if path.is_file() and tuple(path.relative_to(staging).parts[:3]) not in cells
    ]
    if stray:
        listed = "\n".join(f"  {p.relative_to(staging)}" for p in stray[:10])
        raise SystemExit(
            f"{len(stray)} staged files are not in a text submission's viewer "
            f"cell; refusing to upload:\n{listed}"
        )


def upload(staging: Path, profile: str | None, dry_run: bool) -> None:
    """Check the staging tree, then sync it to S3 as gzip-encoded JSON."""
    check_staging(staging)
    sync(staging, profile, dry_run)


def sync(staging: Path, profile: str | None, dry_run: bool) -> None:
    """Sync a staging tree of viewer cells to S3 as gzip-encoded JSON."""
    base = [
        "aws",
        "s3",
        "sync",
        str(staging),
        f"s3://{S3_BUCKET}/{S3_PREFIX}/",
        "--content-encoding",
        "gzip",
        "--content-type",
        "application/json",
    ]
    if profile:
        base += ["--profile", profile]
    if dry_run:
        base.append("--dryrun")
    passes = [
        ["--exclude", "*", "--include", f"*/{SIMULATIONS_DIR_NAME}/*.json"],
        ["--exclude", "*", "--include", f"*/{RESULTS_FILE_NAME}"],
    ]
    for filters, cache in zip(passes, [SIMULATION_CACHE_CONTROL, INDEX_CACHE_CONTROL]):
        cmd = [*base, *filters, "--cache-control", cache]
        print("$ " + " ".join(cmd))
        subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_split = sub.add_parser("split", help="Split text submissions into cells")
    p_split.add_argument("names", nargs="*", help="Submission dirs (default: all)")
    p_split.add_argument("--staging", type=Path, required=True)
    p_split.add_argument(
        "--source", help="Local directory holding the trajectory files (default: S3)"
    )

    p_upload = sub.add_parser("upload", help="Upload the staging tree to S3")
    p_upload.add_argument("--staging", type=Path, required=True)
    p_upload.add_argument("--aws-profile")
    p_upload.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    if args.command == "split":
        split(args.staging, args.names, args.source)
    else:
        upload(args.staging, args.aws_profile, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
