#!/usr/bin/env python3
"""Replay every official BM25 call locally and compare exact tool output."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_CONFIG = HERE / "reference.json"
DEFAULT_REFERENCE = HERE / "artifacts" / "banking_knowledge_results.json"
TIMING_FOOTER = re.compile(
    r"\n\n\[Timing: retrieval=\d+ms(?:, reranking=\d+ms)?, total=\d+ms\]$"
)


class OracleError(RuntimeError):
    """Raised when the oracle artifact or call/result pairing is invalid."""


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OracleError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise OracleError(f"Expected a JSON object: {path}")
    return value


def normalize_output(value: Any) -> Any:
    return TIMING_FOOTER.sub("", value) if isinstance(value, str) else value


def extract_calls(reference: dict[str, Any]) -> list[dict[str, Any]]:
    calls = []
    for simulation in reference.get("simulations") or []:
        messages = simulation.get("messages") or []
        outputs = {
            message.get("id"): message.get("content")
            for message in messages
            if isinstance(message, dict) and message.get("role") == "tool"
        }
        for message in messages:
            if not isinstance(message, dict):
                continue
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict) or call.get("name") != "KB_search_bm25":
                    continue
                arguments = call.get("arguments")
                if not isinstance(arguments, dict) or not isinstance(
                    arguments.get("query"), str
                ):
                    raise OracleError("Official BM25 call has invalid arguments")
                call_id = call.get("id")
                if call_id not in outputs:
                    raise OracleError(
                        f"Official BM25 call has no ToolMessage: {call_id}"
                    )
                k = arguments.get("k", 10)
                if not isinstance(k, int) or isinstance(k, bool) or k < 1:
                    raise OracleError(f"Official BM25 call has invalid k: {k!r}")
                calls.append(
                    {
                        "task_id": simulation.get("task_id"),
                        "trial": simulation.get("trial"),
                        "query": arguments["query"],
                        "k": k,
                        "expected": normalize_output(outputs[call_id]),
                    }
                )
    if not calls:
        raise OracleError("Official artifact contains no BM25 calls")
    return calls


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--max-mismatch-details", type=int, default=20)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_json(args.config.resolve())
        reference_path = args.reference.resolve()
        expected_digest = config["artifacts"]["trajectory"]["sha256"]
        actual_digest = digest_file(reference_path)
        if actual_digest != expected_digest:
            raise OracleError(f"Official trajectory SHA-256 mismatch: {actual_digest}")
        calls = extract_calls(load_json(reference_path))

        expected_by_key: dict[tuple[str, int], Any] = {}
        for call in calls:
            key = (call["query"], call["k"])
            previous = expected_by_key.setdefault(key, call["expected"])
            if previous != call["expected"]:
                raise OracleError(
                    "Official artifact has conflicting outputs for one BM25 query/k"
                )

        # Import after suppressing project logs so stdout is one JSON report.
        from loguru import logger

        logger.remove()
        from tau2.domains.banking_knowledge.data_model import KnowledgeBase
        from tau2.domains.banking_knowledge.retrieval import (
            create_bm25_retrieval_pipeline,
        )
        from tau2.domains.banking_knowledge.retrieval_mixins import _run_kb_search
        from tau2.knowledge.embeddings_cache import clear_cached_docs

        clear_cached_docs()
        knowledge_base = KnowledgeBase.load(
            str(REPO_ROOT / "data/tau2/domains/banking_knowledge/documents")
        )
        pipeline = create_bm25_retrieval_pipeline(knowledge_base)
        actual_by_key = {
            key: normalize_output(_run_kb_search(pipeline, key[0], top_k=key[1]))
            for key in expected_by_key
        }

        mismatches = []
        exact_calls = 0
        exact_unique = 0
        for key, expected in expected_by_key.items():
            if actual_by_key[key] == expected:
                exact_unique += 1
            elif len(mismatches) < args.max_mismatch_details:
                mismatches.append(
                    {
                        "query": key[0],
                        "k": key[1],
                        "expected_sha256": hashlib.sha256(
                            expected.encode("utf-8")
                        ).hexdigest(),
                        "actual_sha256": hashlib.sha256(
                            actual_by_key[key].encode("utf-8")
                        ).hexdigest(),
                    }
                )
        for call in calls:
            if actual_by_key[(call["query"], call["k"])] == call["expected"]:
                exact_calls += 1

        report = {
            "schema_version": 1,
            "reference_sha256": actual_digest,
            "recorded_call_count": len(calls),
            "unique_query_k_count": len(expected_by_key),
            "exact_call_count": exact_calls,
            "exact_unique_query_k_count": exact_unique,
            "exact_percent": 100.0 * exact_calls / len(calls),
            "mismatch_count": len(expected_by_key) - exact_unique,
            "mismatch_details": mismatches,
            "mismatch_details_truncated": (
                len(expected_by_key) - exact_unique > len(mismatches)
            ),
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["mismatch_count"] == 0 else 1
    except (OracleError, KeyError, TypeError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
