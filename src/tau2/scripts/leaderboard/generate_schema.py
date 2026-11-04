"""Generate the leaderboard JSON schemas from their Pydantic models.

Usage:
    python -m tau2.scripts.leaderboard.generate_schema [--check]

Writes, under web/leaderboard/public/submissions/:

- ``schema.json``: 1.x and voice-v1 ``submission.json`` (``Submission``).
- ``schema_benchmark_metadata.json``: the 2.0 package's
  ``benchmark-metadata.json`` (``PackageMetadata``).
- ``schema_benchmark_pass3.json``: the voice-v2-entity package's
  ``benchmark-pass3.json`` (``BenchmarkPass3``).
- ``schema_benchmark_quality.json``: the voice-v2 package's
  ``benchmark-quality.json`` (``BenchmarkQuality``). The package's other files
  are ``BenchmarkSubmission`` and ``BenchmarkDiagnostics`` exports.

With --check, verifies the files are up-to-date and exits non-zero if not.
"""

import argparse
import json
import sys
from pathlib import Path

from pydantic import BaseModel

from tau2.scripts.leaderboard.benchmark_package import PackageMetadata
from tau2.scripts.leaderboard.benchmark_pass3 import BenchmarkPass3
from tau2.scripts.leaderboard.benchmark_quality import BenchmarkQuality
from tau2.scripts.leaderboard.submission import Submission

SUBMISSIONS_DIR = (
    Path(__file__).parent.parent.parent.parent.parent
    / "web"
    / "leaderboard"
    / "public"
    / "submissions"
)
SCHEMA_PATH = SUBMISSIONS_DIR / "schema.json"
METADATA_SCHEMA_PATH = SUBMISSIONS_DIR / "schema_benchmark_metadata.json"
QUALITY_SCHEMA_PATH = SUBMISSIONS_DIR / "schema_benchmark_quality.json"
PASS3_SCHEMA_PATH = SUBMISSIONS_DIR / "schema_benchmark_pass3.json"


def _schema(model: type[BaseModel], title: str, description: str) -> str:
    raw = model.model_json_schema()
    raw["$schema"] = "http://json-schema.org/draft-07/schema#"
    raw["title"] = title
    raw["description"] = description
    return json.dumps(raw, indent=2) + "\n"


def generate_schema() -> str:
    """Generate JSON schema string from the Submission model."""
    return _schema(
        Submission,
        "Tau2-Bench Leaderboard Submission",
        "Schema for submitting model results to the tau2-bench leaderboard. "
        "Auto-generated from the Pydantic model in "
        "src/tau2/scripts/leaderboard/submission.py — do not edit by hand.",
    )


def generate_metadata_schema() -> str:
    """Generate the 2.0 package metadata schema."""
    return _schema(
        PackageMetadata,
        "Tau2-Bench 2.0 Package Metadata",
        "benchmark-metadata.json in a 2.0 leaderboard package. Auto-generated "
        "from src/tau2/scripts/leaderboard/benchmark_package.py — do not edit "
        "by hand.",
    )


def generate_quality_schema() -> str:
    """Generate the voice-v2 package call-quality schema."""
    return _schema(
        BenchmarkQuality,
        "Tau2-Bench 2.0 Package Call Quality",
        "benchmark-quality.json in a voice-v2 leaderboard package: call-level "
        "Interaction and Experience (Generation quality). Auto-generated from "
        "src/tau2/scripts/leaderboard/benchmark_quality.py — do not edit by hand.",
    )


def generate_pass3_schema() -> str:
    """Generate the voice-v2-entity package crossed Pass^3 schema."""
    return _schema(
        BenchmarkPass3,
        "Tau2-Bench 2.0 Package Crossed Pass^3",
        "benchmark-pass3.json in a voice-v2-entity leaderboard package: each "
        "task's outcome in the three crossed voice conditions and the Pass^3 "
        "score. Auto-generated from src/tau2/scripts/leaderboard/benchmark_pass3.py "
        "— do not edit by hand.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check that the schemas are up-to-date (exit 1 if stale)",
    )
    args = parser.parse_args()

    outputs = {
        SCHEMA_PATH: generate_schema(),
        METADATA_SCHEMA_PATH: generate_metadata_schema(),
        QUALITY_SCHEMA_PATH: generate_quality_schema(),
        PASS3_SCHEMA_PATH: generate_pass3_schema(),
    }
    if args.check:
        stale = [
            path
            for path, schema in outputs.items()
            if not path.exists() or path.read_text() != schema
        ]
        for path in stale:
            print(f"FAIL: {path} is out of date. Run 'make generate-schema'.")
        if stale:
            sys.exit(1)
        print("OK: leaderboard schemas are up-to-date")
        return
    for path, schema in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(schema)
        print(f"Wrote schema to {path}")


if __name__ == "__main__":
    main()
