"""Run ten retail conversations with Haiku customers and Cartesia speech."""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def main():
    """Load credentials, then run the benchmark with bounded example settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cartesia-env",
        help="Optional existing dotenv file containing CARTESIA_API_KEY",
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Keep saved calls and retry infrastructure failures",
    )
    args = parser.parse_args()
    load_dotenv()
    if args.cartesia_env:
        load_dotenv(args.cartesia_env, override=False)
    missing = [
        key for key in ("ANTHROPIC_API_KEY", "CARTESIA_API_KEY") if not os.getenv(key)
    ]
    if missing:
        parser.error("Missing credentials: " + ", ".join(missing))
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    sys.argv = [
        "tau2",
        "run",
        "--domain",
        "retail",
        "--audio-native",
        "--audio-native-provider",
        "livekit",
        "--audio-native-model",
        "claude-haiku-4-5-20251001",
        "--cascaded-config",
        "cartesia-haiku",
        "--user-llm",
        "anthropic/claude-haiku-4-5-20251001",
        "--user-llm-args",
        '{"temperature": 0}',
        "--user-tts-config",
        "examples/voice/cartesia-customer.json",
        "--speech-complexity",
        "control",
        "--task-ids",
        "0",
        "2",
        "5",
        "10",
        "12",
        "15",
        "16",
        "17",
        "18",
        "21",
        "--num-trials",
        "1",
        "--max-concurrency",
        "2",
        "--max-steps-seconds",
        "300",
        "--max-retries",
        "1",
        "--save-to",
        "cartesia-haiku-both-10",
        "--verbose-logs",
    ]
    if args.auto_resume:
        sys.argv.append("--auto-resume")
    from tau2.cli import main as run_cli

    run_cli()


if __name__ == "__main__":
    main()
