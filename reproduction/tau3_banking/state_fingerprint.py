#!/usr/bin/env python3
"""Content-address the committed benchmark runtime and dense-embedding cache."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import stat
import subprocess
from pathlib import Path
from typing import Any

RUNTIME_GIT_PATHS = (
    "src/tau2",
    "pyproject.toml",
    "uv.lock",
    "reproduction/tau3_banking",
    "data/tau2/domains/banking_knowledge",
)
DEFAULT_CACHE_PATH = Path("data/.embeddings_cache")
DOCUMENTS_PATH = Path("data/tau2/domains/banking_knowledge/documents")
EMBEDDER_TYPE = "openai"
EFFECTIVE_EMBEDDER_CONFIG = {
    "model": "text-embedding-3-large",
    "_transport": "openrouter-openai-provider-v1",
}
EXPECTED_CACHE_SEMANTIC_SHA256 = (
    "7b1668a5b9afd48edba1ef195c10b534accafb91f1da8229c91ea0c0fabb562b"
)
EXPECTED_DOCUMENT_COUNT = 698
EXPECTED_EMBEDDING_SHAPE = (698, 3072)
EXPECTED_EMBEDDING_DTYPE = "<f8"


class StateFingerprintError(RuntimeError):
    """Raised when a reproducible execution state cannot be captured."""


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git(repo_root: Path, *args: str) -> str:
    process = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode:
        detail = process.stderr.strip() or process.stdout.strip()
        raise StateFingerprintError(detail or f"git {' '.join(args)} failed")
    return process.stdout.strip()


def capture_committed_runtime(
    repo_root: Path, *, require_clean: bool = True
) -> dict[str, Any]:
    """Fingerprint the exact committed code/config/data used by the harness."""
    repo_root = repo_root.resolve()
    top_level = Path(_git(repo_root, "rev-parse", "--show-toplevel")).resolve()
    if top_level != repo_root:
        raise StateFingerprintError(
            f"Expected repository root {repo_root}, git reports {top_level}"
        )

    status = _git(
        repo_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if require_clean and status:
        raise StateFingerprintError(
            "The repository has uncommitted or untracked files. Commit the exact "
            "runtime before creating or consuming a parity gate:\n"
            f"{status}"
        )

    objects: dict[str, str] = {}
    for relative_path in RUNTIME_GIT_PATHS:
        try:
            objects[relative_path] = _git(
                repo_root, "rev-parse", f"HEAD:{relative_path}"
            )
        except StateFingerprintError as exc:
            raise StateFingerprintError(
                f"Runtime path is not committed at HEAD: {relative_path}"
            ) from exc

    runtime = {
        "head": _git(repo_root, "rev-parse", "HEAD"),
        "root_tree": _git(repo_root, "rev-parse", "HEAD^{tree}"),
        "git_objects": objects,
        "worktree_clean": not bool(status),
    }
    runtime["digest"] = _canonical_digest(runtime)
    return runtime


def _digest_stable_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise StateFingerprintError(
                    f"Cache entry is not a regular file: {path}"
                )
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise StateFingerprintError(f"Cannot hash cache entry {path}: {exc}") from exc
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise StateFingerprintError(f"Cache entry changed while hashing: {path}")
    return before.st_size, digest.hexdigest()


def digest_checkpoint_artifact(results_path: Path) -> str:
    """Digest a monolithic checkpoint or its complete directory-format tree."""
    results_path = results_path.resolve()
    if not results_path.is_file() or results_path.is_symlink():
        raise StateFingerprintError(
            f"Checkpoint results file is missing or unsafe: {results_path}"
        )
    simulations_dir = results_path.parent / "simulations"
    if not simulations_dir.is_dir():
        return _digest_stable_file(results_path)[1]
    entries = []
    for path in [results_path, *sorted(simulations_dir.rglob("*"))]:
        if path.is_symlink():
            raise StateFingerprintError(f"Checkpoint symlinks are not allowed: {path}")
        if path.is_dir():
            continue
        size, digest = _digest_stable_file(path)
        entries.append(
            {
                "path": path.relative_to(results_path.parent).as_posix(),
                "size": size,
                "sha256": digest,
            }
        )
    return _canonical_digest(entries)


def _load_current_documents(repo_root: Path) -> list[dict[str, str]]:
    documents_dir = repo_root / DOCUMENTS_PATH
    documents = []
    try:
        paths = sorted(documents_dir.glob("*.json"))
        for path in paths:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise StateFingerprintError(f"Document is not a JSON object: {path}")
            doc_id = value.get("id")
            content = value.get("content")
            if not isinstance(doc_id, str) or not isinstance(content, str):
                raise StateFingerprintError(
                    f"Document lacks string id/content fields: {path}"
                )
            documents.append({"id": doc_id, "text": content})
    except (OSError, json.JSONDecodeError) as exc:
        raise StateFingerprintError(
            f"Cannot resolve current banking documents: {exc}"
        ) from exc
    if len({document["id"] for document in documents}) != len(documents):
        raise StateFingerprintError("Current banking documents contain duplicate IDs")
    if len(documents) != EXPECTED_DOCUMENT_COUNT:
        raise StateFingerprintError(
            f"Expected {EXPECTED_DOCUMENT_COUNT} banking documents, found "
            f"{len(documents)}"
        )
    return documents


def _document_hash(documents: list[dict[str, str]]) -> str:
    representation = []
    for document in sorted(documents, key=lambda item: item["id"]):
        content_hash = hashlib.md5(  # noqa: S324 - compatibility cache key
            document["text"].encode("utf-8")
        ).hexdigest()
        representation.append(f"{document['id']}:{content_hash}")
    return hashlib.sha256("|".join(representation).encode("utf-8")).hexdigest()


def _embedder_hash() -> str:
    payload = json.dumps(
        {"type": EMBEDDER_TYPE, "config": EFFECTIVE_EMBEDDER_CONFIG},
        sort_keys=True,
    )
    return hashlib.md5(payload.encode("utf-8")).hexdigest()  # noqa: S324


def _inspect_pickle_via_uv(
    cache_file: Path, expected_doc_ids: list[str]
) -> dict[str, Any]:
    """Use the locked project runtime when the invoking Python lacks NumPy."""
    script = (
        "import json,sys; from pathlib import Path; "
        f"sys.path.insert(0, {str(Path(__file__).resolve().parent)!r}); "
        "from state_fingerprint import _inspect_pickle; "
        "print(json.dumps(_inspect_pickle(Path(sys.argv[1]), json.load(sys.stdin))))"
    )
    process = subprocess.run(
        [
            "uv",
            "run",
            "--offline",
            "--frozen",
            "--extra",
            "knowledge",
            "python",
            "-c",
            script,
            str(cache_file),
        ],
        cwd=cache_file.parents[2],
        input=json.dumps(expected_doc_ids),
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode:
        raise StateFingerprintError(
            process.stderr.strip()
            or "Locked runtime failed to inspect the selected embedding cache"
        )
    try:
        value = json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        raise StateFingerprintError(
            "Locked runtime returned invalid cache-inspection output"
        ) from exc
    if not isinstance(value, dict):
        raise StateFingerprintError("Locked runtime returned invalid cache inspection")
    return value


def _inspect_pickle(cache_file: Path, expected_doc_ids: list[str]) -> dict[str, Any]:
    try:
        import numpy as np
    except ModuleNotFoundError:
        return _inspect_pickle_via_uv(cache_file, expected_doc_ids)

    try:
        with cache_file.open("rb") as handle:
            value = pickle.load(handle)  # noqa: S301 - runtime uses this trusted cache
    except (OSError, pickle.PickleError, EOFError) as exc:
        raise StateFingerprintError(
            f"Cannot load selected cache {cache_file}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise StateFingerprintError("Selected cache payload is not a dictionary")
    doc_ids = value.get("doc_ids")
    embeddings = value.get("embeddings")
    if (
        not isinstance(doc_ids, list)
        or not all(isinstance(doc_id, str) for doc_id in doc_ids)
        or len(doc_ids) != len(set(doc_ids))
        or set(doc_ids) != set(expected_doc_ids)
    ):
        raise StateFingerprintError(
            "Selected cache document IDs do not exactly match the current documents"
        )
    if not isinstance(embeddings, np.ndarray):
        raise StateFingerprintError("Selected cache embeddings are not a NumPy array")
    if embeddings.ndim < 1 or int(embeddings.shape[0]) != len(doc_ids):
        raise StateFingerprintError(
            "Selected cache row count does not match its document ID list"
        )
    source_order_matches_runtime = doc_ids == expected_doc_ids
    if not source_order_matches_runtime:
        source_index = {doc_id: index for index, doc_id in enumerate(doc_ids)}
        embeddings = embeddings[[source_index[doc_id] for doc_id in expected_doc_ids]]
    shape = tuple(int(item) for item in embeddings.shape)
    dtype = embeddings.dtype.str
    if shape != EXPECTED_EMBEDDING_SHAPE:
        raise StateFingerprintError(
            f"Selected cache shape mismatch: {shape} != {EXPECTED_EMBEDDING_SHAPE}"
        )
    if dtype != EXPECTED_EMBEDDING_DTYPE:
        raise StateFingerprintError(
            f"Selected cache dtype mismatch: {dtype} != {EXPECTED_EMBEDDING_DTYPE}"
        )
    if not bool(np.isfinite(embeddings).all()):
        raise StateFingerprintError("Selected cache contains non-finite embeddings")
    descriptor = {
        "doc_ids": expected_doc_ids,
        "shape": list(shape),
        "dtype": dtype,
    }
    semantic = hashlib.sha256()
    semantic.update(
        json.dumps(
            descriptor,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    )
    semantic.update(b"\0")
    semantic.update(embeddings.tobytes(order="C"))
    return {
        "doc_ids": expected_doc_ids,
        "source_order_matches_runtime": source_order_matches_runtime,
        "shape": list(shape),
        "dtype": dtype,
        "all_finite": True,
        "semantic_sha256": semantic.hexdigest(),
    }


def capture_embedding_cache(
    repo_root: Path, *, require_nonempty: bool = True
) -> dict[str, Any]:
    """Resolve and validate only the selected OpenRouter document cache."""
    cache_root = repo_root.resolve() / DEFAULT_CACHE_PATH
    empty_cache = {
        "path": DEFAULT_CACHE_PATH.as_posix(),
        "selected": None,
        "validation_scope": (
            "No document cache was required. Live query embeddings are not "
            "cached or fingerprinted."
        ),
    }
    empty_cache["digest"] = _canonical_digest({"selected": None})
    if not cache_root.exists():
        if require_nonempty:
            raise StateFingerprintError(f"Embedding cache does not exist: {cache_root}")
        return empty_cache
    if not cache_root.is_dir() or cache_root.is_symlink():
        raise StateFingerprintError(
            f"Embedding cache must be a real directory: {cache_root}"
        )

    documents = _load_current_documents(repo_root.resolve())
    doc_hash = _document_hash(documents)
    embedder_hash = _embedder_hash()
    cache_key = f"{doc_hash}_{embedder_hash}"
    cache_file = cache_root / f"{cache_key}.pkl"
    if not cache_file.is_file() or cache_file.is_symlink():
        if not require_nonempty and not cache_file.exists():
            return empty_cache
        raise StateFingerprintError(
            f"Selected OpenRouter document cache is missing or unsafe: {cache_file}"
        )

    metadata_path = cache_root / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StateFingerprintError(f"Cannot read cache metadata: {exc}") from exc
    entry = metadata.get(cache_key) if isinstance(metadata, dict) else None
    expected_metadata = {
        "embedder_type": EMBEDDER_TYPE,
        "embedder_config": EFFECTIVE_EMBEDDER_CONFIG,
        "num_documents": len(documents),
        "doc_hash": doc_hash,
        "embedder_hash": embedder_hash,
    }
    if not isinstance(entry, dict) or any(
        entry.get(key) != value for key, value in expected_metadata.items()
    ):
        raise StateFingerprintError(
            "Selected OpenRouter cache metadata does not match current documents "
            "and effective embedder configuration"
        )

    size, file_digest = _digest_stable_file(cache_file)
    semantic = _inspect_pickle(cache_file, [item["id"] for item in documents])
    if semantic["semantic_sha256"] != EXPECTED_CACHE_SEMANTIC_SHA256:
        raise StateFingerprintError(
            "Selected cache semantic digest mismatch: "
            f"{semantic['semantic_sha256']} != {EXPECTED_CACHE_SEMANTIC_SHA256}"
        )

    selected = {
        "cache_key": cache_key,
        "path": cache_file.relative_to(repo_root.resolve()).as_posix(),
        "size": size,
        "sha256": file_digest,
        "effective_embedder_config": EFFECTIVE_EMBEDDER_CONFIG,
        "document_count": len(documents),
        "ordered_doc_ids_sha256": _canonical_digest(semantic["doc_ids"]),
        "shape": semantic["shape"],
        "dtype": semantic["dtype"],
        "all_finite": semantic["all_finite"],
        "semantic_sha256": semantic["semantic_sha256"],
    }
    cache = {
        "path": DEFAULT_CACHE_PATH.as_posix(),
        "selected": selected,
        "validation_scope": (
            "Only the selected OpenRouter document-embedding cache is bound. "
            "Stale cache entries and metadata timestamps are ignored. Live query "
            "embeddings have no recoverable oracle; subset ToolMessage output "
            "comparison detects their retrieval effects."
        ),
    }
    cache["digest"] = _canonical_digest(
        {
            key: selected[key]
            for key in (
                "cache_key",
                "effective_embedder_config",
                "document_count",
                "ordered_doc_ids_sha256",
                "shape",
                "dtype",
                "all_finite",
                "semantic_sha256",
            )
        }
    )
    return cache


def capture_reproduction_state(
    repo_root: Path,
    *,
    require_clean: bool = True,
    require_cache: bool = True,
) -> dict[str, Any]:
    """Return the stable gate payload for the current runtime and cache."""
    state = {
        "schema_version": 1,
        "runtime": capture_committed_runtime(repo_root, require_clean=require_clean),
        "embedding_cache": capture_embedding_cache(
            repo_root, require_nonempty=require_cache
        ),
    }
    state["digest"] = _canonical_digest(
        {
            "schema_version": state["schema_version"],
            "runtime_digest": state["runtime"]["digest"],
            "embedding_cache_digest": state["embedding_cache"]["digest"],
        }
    )
    return state
