"""Generate JSON schema from the Submission Pydantic model.

Usage:
    python -m tau2.scripts.leaderboard.generate_schema [--check]

Without --check, writes the schema to web/leaderboard/public/submissions/schema.json.
With --check, verifies the file is up-to-date and exits non-zero if it differs.
"""

import argparse
import json
import sys
from pathlib import Path

from tau2.scripts.leaderboard.submission import Submission

SCHEMA_PATH = (
    Path(__file__).parent.parent.parent.parent.parent
    / "web"
    / "leaderboard"
    / "public"
    / "submissions"
    / "schema.json"
)


def generate_schema() -> str:
    """Generate JSON schema string from the Submission model."""
    raw = Submission.model_json_schema()

    raw["$schema"] = "http://json-schema.org/draft-07/schema#"
    raw["title"] = "Tau2-Bench Leaderboard Submission"
    raw["description"] = (
        "Schema for submitting model results to the tau2-bench leaderboard. "
        "Auto-generated from the Pydantic model in "
        "src/tau2/scripts/leaderboard/submission.py — do not edit by hand."
    )

    return json.dumps(raw, indent=2) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check that schema.json is up-to-date (exit 1 if stale)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (default: web/leaderboard/public/submissions/schema.json)",
    )
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else SCHEMA_PATH
    new_schema = generate_schema()

    if args.check:
        if not output_path.exists():
            print(f"FAIL: {output_path} does not exist")
            sys.exit(1)
        existing = output_path.read_text()
        if existing != new_schema:
            print(
                f"FAIL: {output_path} is out of date. "
                f"Run 'make generate-schema' to update."
            )
            sys.exit(1)
        print(f"OK: {output_path} is up-to-date")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(new_schema)
    print(f"Wrote schema to {output_path}")


if __name__ == "__main__":
    main()
