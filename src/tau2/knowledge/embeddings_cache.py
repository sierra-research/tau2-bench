import hashlib
import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

_docs_cache: Optional[List[Dict[str, Any]]] = None
_query_embeddings_cache: Dict[str, np.ndarray] = {}


def get_cached_docs() -> Optional[List[Dict[str, Any]]]:
    global _docs_cache
    return _docs_cache


def set_cached_docs(docs: List[Dict[str, Any]]) -> None:
    global _docs_cache
    _docs_cache = docs


def clear_cached_docs() -> None:
    global _docs_cache
    _docs_cache = None


def _compute_query_cache_key(
    query: str,
    embedder_type: str,
    embedder_config: Optional[Dict[str, Any]] = None,
) -> str:
    """Compute a cache key for a query embedding."""
    config_str = json.dumps(embedder_config or {}, sort_keys=True)
    combined = f"{embedder_type}|{config_str}|{query}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def get_cached_query_embedding(
    query: str,
    embedder_type: str,
    embedder_config: Optional[Dict[str, Any]] = None,
) -> Optional[np.ndarray]:
    cache_key = _compute_query_cache_key(query, embedder_type, embedder_config)
    return _query_embeddings_cache.get(cache_key)


def cache_query_embedding(
    query: str,
    embedding: np.ndarray,
    embedder_type: str,
    embedder_config: Optional[Dict[str, Any]] = None,
) -> None:
    cache_key = _compute_query_cache_key(query, embedder_type, embedder_config)
    _query_embeddings_cache[cache_key] = embedding


class EmbeddingsCache:
    """
    Cache for storing document embeddings to avoid recomputation.

    The cache uses a hash of the document set (document IDs + content hashes)
    and the embedder configuration to uniquely identify cached embeddings.
    """

    def __init__(self, cache_dir: str = None):
        """
        Initialize the embeddings cache.

        Args:
            cache_dir: Directory to store cache files (default: data/.embeddings_cache)
        """
        if cache_dir is None:
            cache_dir = os.path.join("data", ".embeddings_cache")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load cache metadata from disk."""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        return {}

    def _save_metadata(self):
        """Save cache metadata to disk."""
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def _compute_document_hash(self, documents: List[Dict[str, str]]) -> str:
        """
        Compute a hash representing the document set.

        Args:
            documents: List of documents with 'id' and 'text' keys

        Returns:
            Hash string representing the document set
        """
        # Sort documents by ID for consistent hashing
        sorted_docs = sorted(documents, key=lambda x: x["id"])

        # Create a string representation of document IDs and content hashes
        doc_representation = []
        for doc in sorted_docs:
            content_hash = hashlib.md5(doc["text"].encode("utf-8")).hexdigest()
            doc_representation.append(f"{doc['id']}:{content_hash}")

        # Hash the entire representation
        combined = "|".join(doc_representation)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def _compute_embedder_hash(
        self, embedder_type: str, embedder_config: Dict = None
    ) -> str:
        """
        Compute a hash representing the embedder configuration.

        Args:
            embedder_type: Type of embedder (e.g., 'bm25', 'openai', 'together:model-name')
            embedder_config: Additional embedder configuration

        Returns:
            Hash string representing the embedder configuration
        """
        config_str = json.dumps(
            {"type": embedder_type, "config": embedder_config or {}}, sort_keys=True
        )
        return hashlib.md5(config_str.encode("utf-8")).hexdigest()

    def _get_cache_key(self, doc_hash: str, embedder_hash: str) -> str:
        """Generate a cache key from document and embedder hashes."""
        return f"{doc_hash}_{embedder_hash}"

    def _get_cache_file(self, cache_key: str) -> Path:
        """Get the file path for a cache key."""
        return self.cache_dir / f"{cache_key}.pkl"

    def get(
        self,
        documents: List[Dict[str, str]],
        embedder_type: str,
        embedder_config: Dict = None,
    ) -> Optional[Tuple[np.ndarray, List[str]]]:
        """
        Retrieve cached embeddings if available.

        Args:
            documents: List of documents with 'id' and 'text' keys
            embedder_type: Type of embedder (e.g., 'bm25', 'openai', 'together:model-name')
            embedder_config: Additional embedder configuration

        Returns:
            Tuple of (embeddings array, document IDs) if cached, None otherwise
        """
        doc_hash = self._compute_document_hash(documents)
        embedder_hash = self._compute_embedder_hash(embedder_type, embedder_config)
        cache_key = self._get_cache_key(doc_hash, embedder_hash)
        cache_file = self._get_cache_file(cache_key)

        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    cached_data = pickle.load(f)

                if cached_data["doc_ids"] == [doc["id"] for doc in documents]:
                    return cached_data["embeddings"], cached_data["doc_ids"]
                else:
                    print(
                        f"⚠️  Cache entry exists but document IDs don't match, invalidating (key: {cache_key[:8]}...)"
                    )
                    cache_file.unlink()
            except Exception as e:
                print(f"⚠️  Error reading cache file, will recompute: {e}")
                try:
                    cache_file.unlink()
                except Exception:
                    pass

        print(
            f"⚠️  Document preprocessing cache miss (key: {cache_key[:8]}...). "
            f"Computing embeddings for {len(documents)} documents."
        )
        return None

    def put(
        self,
        documents: List[Dict[str, str]],
        embedder_type: str,
        embeddings: np.ndarray,
        doc_ids: List[str],
        embedder_config: Dict = None,
    ):
        """
        Store embeddings in cache.

        Args:
            documents: List of documents with 'id' and 'text' keys
            embedder_type: Type of embedder (e.g., 'bm25', 'openai', 'together:model-name')
            embeddings: Embeddings array
            doc_ids: List of document IDs corresponding to embeddings
            embedder_config: Additional embedder configuration
        """
        doc_hash = self._compute_document_hash(documents)
        embedder_hash = self._compute_embedder_hash(embedder_type, embedder_config)
        cache_key = self._get_cache_key(doc_hash, embedder_hash)
        cache_file = self._get_cache_file(cache_key)

        try:
            # Save embeddings to pickle file
            cached_data = {
                "embeddings": embeddings,
                "doc_ids": doc_ids,
                "embedder_type": embedder_type,
                "embedder_config": embedder_config,
                "num_documents": len(documents),
                "created_at": datetime.utcnow().isoformat(),
            }

            with open(cache_file, "wb") as f:
                pickle.dump(cached_data, f)

            # Update metadata
            self.metadata[cache_key] = {
                "embedder_type": embedder_type,
                "embedder_config": embedder_config,
                "num_documents": len(documents),
                "doc_hash": doc_hash,
                "embedder_hash": embedder_hash,
                "created_at": cached_data["created_at"],
                "file_size_mb": cache_file.stat().st_size / (1024 * 1024),
            }
            self._save_metadata()

            print(
                f"💾 Cached embeddings (key: {cache_key[:8]}..., size: {self.metadata[cache_key]['file_size_mb']:.2f} MB)"
            )

        except Exception as e:
            print(f"⚠️  Error saving to cache: {e}")

    def _compute_single_doc_hash(self, doc_id: str, text: str) -> str:
        """
        Compute hash for a single document.

        Args:
            doc_id: Document ID
            text: Document text content

        Returns:
            Hash string for this document
        """
        content_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        return f"{doc_id}:{content_hash}"

    def _get_per_doc_cache_dir(
        self, embedder_type: str, embedder_config: Dict = None
    ) -> Path:
        """Get directory for per-document cache for a specific embedder."""
        embedder_hash = self._compute_embedder_hash(embedder_type, embedder_config)
        cache_dir = self.cache_dir / "per_doc" / embedder_hash
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _get_per_doc_cache_file(
        self,
        doc_id: str,
        doc_hash: str,
        embedder_type: str,
        embedder_config: Dict = None,
    ) -> Path:
        """Get cache file path for a single document."""
        cache_dir = self._get_per_doc_cache_dir(embedder_type, embedder_config)
        # Use doc_hash in filename to ensure content changes invalidate cache
        safe_doc_id = doc_id.replace("/", "_").replace("\\", "_")
        return cache_dir / f"{safe_doc_id}_{doc_hash[:16]}.pkl"

    def get_incremental(
        self,
        documents: List[Dict[str, str]],
        embedder_type: str,
        embedder_config: Dict = None,
    ) -> Tuple[Dict[str, np.ndarray], Set[str]]:
        """
        Retrieve cached embeddings incrementally, returning both cached and missing docs.

        Args:
            documents: List of documents with 'id' and 'text' keys
            embedder_type: Type of embedder (e.g., 'bm25', 'openai', 'together:model-name')
            embedder_config: Additional embedder configuration

        Returns:
            Tuple of (dict mapping doc_id to embedding, set of doc_ids that need embedding)
        """
        cached_embeddings = {}
        docs_to_embed = set()

        for doc in documents:
            doc_id = doc["id"]
            text = doc["text"]
            doc_hash = hashlib.md5(text.encode("utf-8")).hexdigest()

            cache_file = self._get_per_doc_cache_file(
                doc_id, doc_hash, embedder_type, embedder_config
            )

            if cache_file.exists():
                try:
                    with open(cache_file, "rb") as f:
                        cached_data = pickle.load(f)

                    # Verify the cached data matches
                    if (
                        cached_data["doc_id"] == doc_id
                        and cached_data["doc_hash"] == doc_hash
                    ):
                        cached_embeddings[doc_id] = cached_data["embedding"]
                    else:
                        # Cache invalid, need to re-embed
                        docs_to_embed.add(doc_id)
                        try:
                            cache_file.unlink()
                        except Exception:
                            pass
                except Exception as e:
                    print(f"⚠️  Error reading cache for {doc_id}: {e}")
                    docs_to_embed.add(doc_id)
                    try:
                        cache_file.unlink()
                    except Exception:
                        pass
            else:
                docs_to_embed.add(doc_id)

        if cached_embeddings:
            print(
                f"✅ Incremental cache: {len(cached_embeddings)} docs cached, {len(docs_to_embed)} need embedding"
            )
        else:
            print(
                f"❌ Incremental cache miss: all {len(docs_to_embed)} docs need embedding"
            )

        return cached_embeddings, docs_to_embed

    def put_incremental(
        self,
        doc_id: str,
        text: str,
        embedding: np.ndarray,
        embedder_type: str,
        embedder_config: Dict = None,
    ):
        """
        Store a single document's embedding in cache.

        Args:
            doc_id: Document ID
            text: Document text content
            embedding: Document embedding
            embedder_type: Type of embedder
            embedder_config: Additional embedder configuration
        """
        doc_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        cache_file = self._get_per_doc_cache_file(
            doc_id, doc_hash, embedder_type, embedder_config
        )

        # Remove old cache files for this doc_id (with different hashes)
        cache_dir = self._get_per_doc_cache_dir(embedder_type, embedder_config)
        safe_doc_id = doc_id.replace("/", "_").replace("\\", "_")
        for old_file in cache_dir.glob(f"{safe_doc_id}_*.pkl"):
            if old_file != cache_file:
                try:
                    old_file.unlink()
                except Exception:
                    pass

        try:
            cached_data = {
                "doc_id": doc_id,
                "doc_hash": doc_hash,
                "embedding": embedding,
                "embedder_type": embedder_type,
                "embedder_config": embedder_config,
                "created_at": datetime.utcnow().isoformat(),
            }

            with open(cache_file, "wb") as f:
                pickle.dump(cached_data, f)

        except Exception as e:
            print(f"⚠️  Error caching embedding for {doc_id}: {e}")

    def clear(self, embedder_type: str = None):
        """
        Clear cache entries (both bulk and incremental caches).

        Args:
            embedder_type: If specified, only clear entries for this embedder type
        """
        import shutil

        # Clear bulk cache entries
        if embedder_type:
            # Clear specific embedder type
            keys_to_remove = [
                key
                for key, meta in self.metadata.items()
                if meta.get("embedder_type") == embedder_type
            ]
        else:
            # Clear all
            keys_to_remove = list(self.metadata.keys())

        for key in keys_to_remove:
            cache_file = self._get_cache_file(key)
            if cache_file.exists():
                cache_file.unlink()
            del self.metadata[key]

        self._save_metadata()

        # Clear per-document cache
        per_doc_cache_dir = self.cache_dir / "per_doc"
        if per_doc_cache_dir.exists():
            if embedder_type:
                # Clear specific embedder's per-doc cache
                embedder_hash = self._compute_embedder_hash(embedder_type, None)
                embedder_cache_dir = per_doc_cache_dir / embedder_hash
                if embedder_cache_dir.exists():
                    shutil.rmtree(embedder_cache_dir)
                    print(
                        f"🗑️  Cleared {len(keys_to_remove)} bulk cache entries + per-doc cache for {embedder_type}"
                    )
            else:
                # Clear all per-doc cache
                shutil.rmtree(per_doc_cache_dir)
                print(
                    f"🗑️  Cleared {len(keys_to_remove)} bulk cache entries + all per-doc cache"
                )
        else:
            print(f"🗑️  Cleared {len(keys_to_remove)} cache entries")

    def get_stats(self) -> Dict:
        """
        Get cache statistics (both bulk and incremental caches).

        Returns:
            Dictionary with cache statistics
        """
        total_size_mb = sum(
            meta.get("file_size_mb", 0) for meta in self.metadata.values()
        )

        embedder_counts = {}
        for meta in self.metadata.values():
            embedder_type = meta.get("embedder_type", "unknown")
            embedder_counts[embedder_type] = embedder_counts.get(embedder_type, 0) + 1

        # Count per-document cache files
        per_doc_cache_dir = self.cache_dir / "per_doc"
        per_doc_stats = {}
        per_doc_total_size = 0
        per_doc_total_files = 0

        if per_doc_cache_dir.exists():
            for embedder_dir in per_doc_cache_dir.iterdir():
                if embedder_dir.is_dir():
                    files = list(embedder_dir.glob("*.pkl"))
                    count = len(files)
                    size_mb = sum(f.stat().st_size for f in files) / (1024 * 1024)
                    per_doc_stats[embedder_dir.name] = {
                        "count": count,
                        "size_mb": size_mb,
                    }
                    per_doc_total_files += count
                    per_doc_total_size += size_mb

        return {
            "bulk_cache": {
                "total_entries": len(self.metadata),
                "total_size_mb": total_size_mb,
                "embedder_counts": embedder_counts,
            },
            "incremental_cache": {
                "total_files": per_doc_total_files,
                "total_size_mb": per_doc_total_size,
                "by_embedder": per_doc_stats,
            },
            "total_size_mb": total_size_mb + per_doc_total_size,
            "cache_dir": str(self.cache_dir),
        }


# Global cache instance
_global_cache = None


def get_embeddings_cache() -> EmbeddingsCache:
    """Get the global embeddings cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = EmbeddingsCache()
    return _global_cache


def warm_kb_cache(
    embedder_configs: Optional[List[Tuple[str, Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    """Pre-warm the knowledge base cache with documents and embeddings.

    This function should be called once before running tasks to:
    1. Load and cache documents (shared across all retrieval configs)
    2. Compute and cache embeddings for each unique embedder config

    Args:
        embedder_configs: List of (embedder_type, embedder_params) tuples to pre-compute.
                         If None, only loads documents without computing embeddings.

    Returns:
        List of documents (for use in pipelines)
    """
    from tau2.domains.banking_knowledge.environment import get_knowledge_base

    cache = get_embeddings_cache()

    cached_docs = get_cached_docs()
    if cached_docs is not None:
        print(f"✅ Using in-memory cached documents ({len(cached_docs)} docs)")
        docs = cached_docs
    else:
        print("🔄 Loading documents...")
        knowledge_base = get_knowledge_base()
        docs = [
            {"id": doc.id, "text": doc.content, "title": doc.title}
            for doc in knowledge_base.documents.values()
        ]
        set_cached_docs(docs)
        print(f"✅ Loaded {len(docs)} documents")

    if embedder_configs:
        for embedder_type, embedder_params in embedder_configs:
            cached = cache.get(docs, embedder_type, embedder_params)
            if cached is not None:
                print(
                    f"✅ Embeddings already cached for {embedder_type}:{embedder_params.get('model', 'default')}"
                )
            else:
                print(
                    f"🔄 Computing embeddings for {embedder_type}:{embedder_params.get('model', 'default')}..."
                )
                _compute_and_cache_embeddings(
                    docs, embedder_type, embedder_params, cache
                )

    return docs


def _compute_and_cache_embeddings(
    docs: List[Dict[str, Any]],
    embedder_type: str,
    embedder_params: Dict[str, Any],
    cache: EmbeddingsCache,
) -> np.ndarray:
    """Compute embeddings and cache them."""
    from tau2.knowledge.embedders import (
        OpenAIEmbedder,
        OpenRouterEmbedder,
    )

    EMBEDDER_REGISTRY = {
        "openai": OpenAIEmbedder,
        "openrouter": OpenRouterEmbedder,
    }

    if embedder_type not in EMBEDDER_REGISTRY:
        raise ValueError(f"Unknown embedder_type: {embedder_type}")

    embedder = EMBEDDER_REGISTRY[embedder_type](**embedder_params)
    texts = [doc["text"] for doc in docs]
    embeddings = embedder.embed(texts)
    doc_ids = [doc["id"] for doc in docs]

    cache.put(docs, embedder_type, embeddings, doc_ids, embedder_params)

    return embeddings


def get_unique_embedder_configs_for_retrieval_configs(
    retrieval_config_names: List[str],
) -> List[Tuple[str, Dict[str, Any]]]:
    """Get unique embedder configurations for a list of retrieval config names.

    Args:
        retrieval_config_names: List of retrieval config names (e.g., ["classic_rag_qwen", "classic_rag_openai"])

    Returns:
        List of unique (embedder_type, embedder_params) tuples
    """
    CONFIG_EMBEDDERS = {
        "qwen_embeddings_grep": ("openrouter", {"model": "qwen3-embedding-8b"}),
        "qwen_embeddings_reranker_grep": (
            "openrouter",
            {"model": "qwen3-embedding-8b"},
        ),
        "qwen_embeddings": ("openrouter", {"model": "qwen3-embedding-8b"}),
        "qwen_embeddings_reranker": ("openrouter", {"model": "qwen3-embedding-8b"}),
        "openai_embeddings_grep": ("openai", {"model": "text-embedding-3-large"}),
        "openai_embeddings_reranker_grep": (
            "openai",
            {"model": "text-embedding-3-large"},
        ),
        "openai_embeddings": ("openai", {"model": "text-embedding-3-large"}),
        "openai_embeddings_reranker": ("openai", {"model": "text-embedding-3-large"}),
    }

    seen = set()
    unique_configs = []

    for config_name in retrieval_config_names:
        if config_name in CONFIG_EMBEDDERS:
            config = CONFIG_EMBEDDERS[config_name]
            key = (config[0], json.dumps(config[1], sort_keys=True))
            if key not in seen:
                seen.add(key)
                unique_configs.append(config)

    return unique_configs
