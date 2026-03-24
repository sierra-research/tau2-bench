#!/usr/bin/env python3
"""Create a trimmed copy of an experiment directory.

Copies the directory structure but:
- Skips hallucination_discarded/ directories
- For each task, only keeps the sim directory referenced in results.json
- Within each sim, only keeps the audio/ directory (drops llm_debug/ and task.log)
- Text-only experiments (no tasks/ dir) get just their results.json copied

The original experiment directory is NEVER modified.
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_results_json(results_path: Path) -> dict[str, str]:
    """Return a mapping of task_id -> sim_id from a results.json file."""
    with open(results_path) as f:
        data = json.load(f)

    task_to_sim: dict[str, str] = {}
    for sim in data.get("simulations", []):
        task_id = str(sim["task_id"])
        sim_id = sim["id"]
        task_to_sim[task_id] = sim_id

    return task_to_sim


def process_experiment(
    exp_src: Path, exp_dst: Path, *, dry_run: bool
) -> dict[str, int]:
    """Process a single experiment directory. Returns stats dict."""
    stats = {"copied_audio": 0, "skipped_sims": 0, "missing_audio": 0}
    results_path = exp_src / "results.json"

    if dry_run:
        log.info("  [dry-run] copy %s", results_path.name)
    else:
        exp_dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(results_path, exp_dst / "results.json")

    tasks_dir = exp_src / "tasks"
    if not tasks_dir.is_dir():
        log.info("  No tasks/ directory (text-only experiment), copied results.json")
        return stats

    task_to_sim = parse_results_json(results_path)

    all_task_dirs = sorted(tasks_dir.iterdir())
    for task_dir in all_task_dirs:
        if not task_dir.is_dir() or not task_dir.name.startswith("task_"):
            continue

        task_id = task_dir.name.split("_", 1)[1]
        canonical_sim_id = task_to_sim.get(task_id)

        if canonical_sim_id is None:
            log.warning(
                "  Task %s has no simulation in results.json, skipping all sims",
                task_dir.name,
            )
            sim_dirs = [d for d in task_dir.iterdir() if d.is_dir()]
            stats["skipped_sims"] += len(sim_dirs)
            continue

        canonical_sim_name = f"sim_{canonical_sim_id}"

        for sim_dir in sorted(task_dir.iterdir()):
            if not sim_dir.is_dir() or not sim_dir.name.startswith("sim_"):
                continue

            if sim_dir.name != canonical_sim_name:
                stats["skipped_sims"] += 1
                continue

            audio_src = sim_dir / "audio"
            if not audio_src.is_dir():
                log.warning(
                    "  Missing audio/ in %s/%s/%s",
                    task_dir.name,
                    sim_dir.name,
                    "audio",
                )
                stats["missing_audio"] += 1
                continue

            audio_dst = exp_dst / "tasks" / task_dir.name / sim_dir.name / "audio"
            if dry_run:
                log.info(
                    "  [dry-run] copy %s/%s/audio/",
                    task_dir.name,
                    sim_dir.name,
                )
            else:
                shutil.copytree(audio_src, audio_dst)
            stats["copied_audio"] += 1

    return stats


def trim_experiment_data(input_dir: Path, output_dir: Path, *, dry_run: bool) -> None:
    """Walk the input directory, find all experiments, and create trimmed copies."""
    if not input_dir.is_dir():
        log.error("Input directory does not exist: %s", input_dir)
        raise SystemExit(1)

    if output_dir.exists() and not dry_run:
        log.error("Output directory already exists: %s", output_dir)
        log.error(
            "Remove it first or choose a different name to avoid accidental data mixing."
        )
        raise SystemExit(1)

    experiment_dirs = sorted(input_dir.rglob("results.json"))
    log.info("Found %d experiments in %s", len(experiment_dirs), input_dir)

    totals = {"copied_audio": 0, "skipped_sims": 0, "missing_audio": 0}

    for results_path in experiment_dirs:
        exp_src = results_path.parent
        rel_path = exp_src.relative_to(input_dir)
        exp_dst = output_dir / rel_path

        log.info("Processing: %s", rel_path)
        stats = process_experiment(exp_src, exp_dst, dry_run=dry_run)

        for k in totals:
            totals[k] += stats[k]

    log.info("--- Summary ---")
    log.info("Experiments processed: %d", len(experiment_dirs))
    log.info("Audio dirs copied:     %d", totals["copied_audio"])
    log.info("Sim dirs skipped:      %d", totals["skipped_sims"])
    if totals["missing_audio"] > 0:
        log.warning("Missing audio dirs:    %d", totals["missing_audio"])
    if dry_run:
        log.info("(dry-run mode -- nothing was actually copied)")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Source experiment directory (will NOT be modified)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory for trimmed copy",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be copied without actually copying",
    )
    args = parser.parse_args()

    trim_experiment_data(args.input_dir, args.output_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
