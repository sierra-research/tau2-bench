"""Analyze one saved text Results experiment without running models."""

import argparse
import json
from pathlib import Path

from experiments.bayes_at_n import compute_bayes_at_n
from tau2.data_model.simulation import Results


def main(argv: list[str] | None = None) -> None:
    """Print a Bayes@N JSON summary and optionally save it to a new file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Saved Results JSON file or directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Save JSON to this new file; existing files are refused.",
    )
    args = parser.parse_args(argv)
    try:
        summary = compute_bayes_at_n(Results.load(args.results))
        payload = json.dumps(summary.model_dump(mode="json"), indent=2, allow_nan=False)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x") as output:
                output.write(payload + "\n")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(payload)


if __name__ == "__main__":
    main()
