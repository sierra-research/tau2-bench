"""Trim trajectory JSON files to fit within GitHub's 100 MB file size limit.

The leaderboard visualizer only needs a subset of the data stored in raw
trajectory files.  This script applies a sequence of progressively more
aggressive trimming passes until the file is under the target size:

  1. Strip `raw_data` from every message (LLM provider response objects).
  2. Strip voice/audio-only fields that are null or unused in text mode.
  3. Strip simulation-level fields not used by the visualizer.
  4. Truncate the longest tool-result and assistant-content messages.

Usage:
    python -m tau2.scripts.leaderboard.trim_trajectories <path> [--target-mb 95] [--in-place]

The script will write a trimmed copy next to the original (with a
`_trimmed.json` suffix) unless ``--in-place`` is given.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ── Fields to strip ──────────────────────────────────────────────────────────

# Message-level fields that are never used by the leaderboard visualizer.
# The visualizer uses: role, content, tool_calls, turn_idx, timestamp, cost, usage
MESSAGE_STRIP_FIELDS = {
    # Raw LLM provider response – by far the largest contributor
    "raw_data",
    # Voice/audio-only fields (always null in text-mode trajectories)
    "audio_format",
    "audio_path",
    "audio_script_gold",
    "speech_effects",
    "source_effects",
    "channel_effects",
    "turn_taking_action",
    "utterance_ids",
    "chunk_id",
    "source",
    "is_final_chunk",
    "contains_speech",
    "is_audio",
    # Internal bookkeeping
    "id",  # per-message UUIDs, not needed for display
    "requestor",
    "error",
}

# Simulation-level fields not consumed by the visualizer.
SIMULATION_STRIP_FIELDS = {
    "speech_environment",
    "review",
    "user_only_review",
    "auth_classification",
    "hallucination_retries_used",
    "hallucination_check",
    "provider_session_id",
    "policy",
    "effect_timeline",
    "info",
    "ticks",
}

# ── Helpers ──────────────────────────────────────────────────────────────────


def _json_size(obj: object) -> int:
    """Return the serialised JSON byte length of *obj*."""
    return len(json.dumps(obj, ensure_ascii=False).encode())


def _strip_message_fields(data: dict) -> int:
    """Strip unnecessary fields from every message.  Returns bytes saved."""
    saved = 0
    for sim in data.get("simulations", []):
        for msg in sim.get("messages") or []:
            for field in MESSAGE_STRIP_FIELDS:
                if field in msg:
                    saved += _json_size(msg[field])
                    del msg[field]
    return saved


def _strip_simulation_fields(data: dict) -> int:
    """Strip unnecessary simulation-level fields.  Returns bytes saved."""
    saved = 0
    for sim in data.get("simulations", []):
        for field in SIMULATION_STRIP_FIELDS:
            if field in sim:
                saved += _json_size(sim[field])
                del sim[field]
    return saved


def _truncate_long_messages(data: dict, target_bytes: int, current_bytes: int) -> int:
    """Truncate the longest message contents until under *target_bytes*.

    Truncates tool results first (they contain DB dumps), then assistant
    content.  Each pass halves the largest message content that exceeds a
    dynamic threshold.  Returns total bytes saved.
    """
    saved = 0
    max_iterations = 50  # safety valve

    for _ in range(max_iterations):
        if current_bytes - saved <= target_bytes:
            break

        # Collect all (sim_idx, msg_idx, content_size, role) tuples
        candidates: list[tuple[int, int, int, str]] = []
        for si, sim in enumerate(data.get("simulations", [])):
            for mi, msg in enumerate((sim.get("messages") or [])):
                content = msg.get("content")
                if content and isinstance(content, str) and len(content) > 2000:
                    candidates.append((si, mi, len(content), msg.get("role", "")))

        if not candidates:
            break

        # Sort: tool messages first (safest to truncate), then by size desc
        candidates.sort(key=lambda c: (0 if c[3] == "tool" else 1, -c[2]))

        # Truncate the biggest candidate
        si, mi, size, role = candidates[0]
        msg = data["simulations"][si]["messages"][mi]
        content = msg["content"]
        # Keep first 500 + last 200 chars with a truncation marker
        keep = 500
        tail = 200
        if len(content) > keep + tail + 100:
            truncated_chars = len(content) - keep - tail
            msg["content"] = (
                content[:keep]
                + f"\n\n[... {truncated_chars:,} characters trimmed for size ...]\n\n"
                + content[-tail:]
            )
            saved += len(content) - len(msg["content"])

    return saved


# ── Main ─────────────────────────────────────────────────────────────────────


def trim_trajectory(
    path: Path,
    target_mb: float = 95.0,
    in_place: bool = False,
    truncate_content: bool = False,
) -> Path:
    """Trim a single trajectory file.  Returns the output path.

    Args:
        path: Path to the trajectory JSON file.
        target_mb: Target maximum file size in MB (used only when
            *truncate_content* is True).
        in_place: Overwrite the original file instead of creating a
            ``*_trimmed.json`` copy.
        truncate_content: If True, truncate long message contents (tool
            results, assistant messages) to fit under *target_mb*.  This is
            lossy and off by default — trajectories are hosted on S3 where
            there is no file-size constraint.
    """
    target_bytes = int(target_mb * 1024 * 1024)

    print(f"Loading {path} …")
    with open(path, "r") as f:
        data = json.load(f)

    original_bytes = path.stat().st_size
    print(f"  Original size: {original_bytes / 1e6:.1f} MB")

    # Pass 1: strip raw_data and voice fields from messages
    saved = _strip_message_fields(data)
    current = original_bytes - saved
    print(
        f"  After stripping message fields: ~{current / 1e6:.1f} MB (saved {saved / 1e6:.1f} MB)"
    )

    # Pass 2: strip simulation-level fields
    saved2 = _strip_simulation_fields(data)
    saved += saved2
    current -= saved2
    print(
        f"  After stripping simulation fields: ~{current / 1e6:.1f} MB (saved {saved2 / 1e6:.1f} MB)"
    )

    # Pass 3 (optional): truncate long messages to meet size target
    if truncate_content and current > target_bytes:
        saved3 = _truncate_long_messages(data, target_bytes, current)
        saved += saved3
        current -= saved3
        print(
            f"  After truncating long messages: ~{current / 1e6:.1f} MB (saved {saved3 / 1e6:.1f} MB)"
        )

    # Write output
    if in_place:
        out_path = path
    else:
        out_path = path.with_stem(path.stem + "_trimmed")

    print(f"  Writing {out_path} …")
    with open(out_path, "w") as f:
        json.dump(data, f, ensure_ascii=False)

    actual_bytes = out_path.stat().st_size
    print(f"  Final size: {actual_bytes / 1e6:.1f} MB")

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Trim trajectory files to fit within GitHub's 100 MB limit."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Trajectory JSON file(s) to trim.",
    )
    parser.add_argument(
        "--target-mb",
        type=float,
        default=95.0,
        help="Target maximum file size in MB (default: 95, leaves headroom below GitHub's 100 MB limit).",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the original file instead of creating a *_trimmed.json copy.",
    )
    parser.add_argument(
        "--truncate-content",
        action="store_true",
        help="Truncate long message contents (lossy) to fit under --target-mb. "
        "Off by default since trajectories are hosted on S3.",
    )
    args = parser.parse_args()

    for path in args.paths:
        if not path.exists():
            print(f"Error: {path} not found", file=sys.stderr)
            sys.exit(1)
        trim_trajectory(
            path,
            target_mb=args.target_mb,
            in_place=args.in_place,
            truncate_content=args.truncate_content,
        )
        print()


if __name__ == "__main__":
    main()
