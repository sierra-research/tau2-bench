#!/usr/bin/env python3
"""Fetch and SHA-256 verify the public reference submission artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "reference.json"
DEFAULT_OUTPUT = HERE / "artifacts"
CHUNK_SIZE = 1024 * 1024
PROGRESS_INTERVAL = 64 * 1024 * 1024


class ArtifactError(RuntimeError):
    """Raised when a reference artifact cannot be safely verified."""


def sha256_file(path: Path) -> tuple[str, int]:
    """Return the SHA-256 digest and size of *path* without loading it in memory."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def load_config(path: Path) -> dict[str, Any]:
    """Load a reference configuration and validate its artifact section."""
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"Cannot read reference config {path}: {exc}") from exc
    if not isinstance(config.get("artifacts"), dict):
        raise ArtifactError(f"Reference config {path} has no artifact map")
    return config


def verify(path: Path, spec: dict[str, Any]) -> None:
    """Verify one file against the immutable size and digest in *spec*."""
    actual_digest, actual_size = sha256_file(path)
    expected_digest = str(spec["sha256"])
    expected_size = int(spec["bytes"])
    errors = []
    if actual_size != expected_size:
        errors.append(f"size {actual_size}, expected {expected_size}")
    if actual_digest != expected_digest:
        errors.append(f"sha256 {actual_digest}, expected {expected_digest}")
    if errors:
        raise ArtifactError(f"Verification failed for {path}: {'; '.join(errors)}")


def fetch(
    name: str,
    spec: dict[str, Any],
    output_dir: Path,
    *,
    force: bool,
    verify_only: bool,
) -> Path:
    """Fetch one artifact atomically and verify it before publication."""
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / str(spec["filename"])
    if destination.exists() and not force:
        verify(destination, spec)
        print(f"verified existing {name}: {destination}")
        return destination
    if verify_only:
        if not destination.exists():
            raise ArtifactError(f"Missing artifact for --verify-only: {destination}")
        verify(destination, spec)
        print(f"verified {name}: {destination}")
        return destination

    request = urllib.request.Request(
        str(spec["url"]),
        headers={
            "Accept-Encoding": "identity",
            "User-Agent": "tau3-banking-reproduction/1",
        },
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_dir,
            prefix=f".{destination.name}.part-",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            downloaded = 0
            next_progress = PROGRESS_INTERVAL
            with urllib.request.urlopen(request, timeout=60) as response:
                while chunk := response.read(CHUNK_SIZE):
                    temporary.write(chunk)
                    downloaded += len(chunk)
                    if downloaded >= next_progress:
                        print(f"fetching {name}: {downloaded // (1024 * 1024)} MiB")
                        next_progress += PROGRESS_INTERVAL
            temporary.flush()
            os.fsync(temporary.fileno())
        verify(temporary_path, spec)
        temporary_path.replace(destination)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise

    print(f"fetched and verified {name}: {destination}")
    return destination


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="Reference JSON path"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination directory (gitignored by default)",
    )
    parser.add_argument(
        "--artifact",
        choices=("submission", "trajectory", "all"),
        default="all",
        help="Artifact to fetch and verify",
    )
    parser.add_argument(
        "--force", action="store_true", help="Replace an existing destination file"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Do not use the network; verify files already in --output-dir",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the artifact fetcher."""
    args = parse_args(argv)
    try:
        config = load_config(args.config)
        names = (
            tuple(config["artifacts"]) if args.artifact == "all" else (args.artifact,)
        )
        for name in names:
            if name not in config["artifacts"]:
                raise ArtifactError(f"Unknown artifact in config: {name}")
            fetch(
                name,
                config["artifacts"][name],
                args.output_dir,
                force=args.force,
                verify_only=args.verify_only,
            )
    except (ArtifactError, OSError, urllib.error.URLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
