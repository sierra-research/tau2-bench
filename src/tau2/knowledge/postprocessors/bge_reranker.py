"""BGE Reranker M3 postprocessor using Baseten-hosted API."""

import os
from typing import Any, Dict, List, Optional, Tuple

import requests

from tau2.knowledge.postprocessors.base import BasePostprocessor
from tau2.knowledge.registry import register_postprocessor

DEFAULT_ENDPOINT = (
    "https://model-5qerx1p3.api.sierra.baseten.io/environments/production/predict"
)


@register_postprocessor("bge_reranker")
class BGEReranker(BasePostprocessor):
    """BGE Reranker M3 using Baseten-hosted API.

    Uses the BGE-Reranker-v2-m3 model to score query-document relevance.
    Higher scores indicate stronger relevance. Scores can be negative.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        top_k: int = 5,
        query_key: str = "query",
        batch_size: int = 32,
        min_score: Optional[float] = None,
        **kwargs,
    ):
        """Initialize the BGE reranker.

        Args:
            endpoint: Baseten API endpoint URL. Defaults to Sierra's BGE M3 endpoint.
            api_key: API key. Falls back to BGE_RERANKER_API_KEY or BASETEN_API_KEY env vars.
            top_k: Number of top results to return after reranking.
            query_key: Key to look up query in input_data dict.
            batch_size: Number of texts to process per API call.
            min_score: Minimum score threshold for filtering. None disables filtering.
        """
        super().__init__(**kwargs)
        self.endpoint = endpoint or DEFAULT_ENDPOINT
        self.api_key = (
            api_key
            or os.environ.get("BGE_RERANKER_API_KEY")
            or os.environ.get("BASETEN_API_KEY")
        )
        self.top_k = top_k
        self.query_key = query_key
        self.batch_size = batch_size
        self.min_score = min_score
        self._session: Optional[requests.Session] = None

    def _get_session(self) -> requests.Session:
        """Get or create a requests session for connection reuse."""
        if self._session is None:
            self._session = requests.Session()
        return self._session

    def _call_api(self, query: str, texts: List[str]) -> List[float]:
        """Call the BGE reranker API.

        Args:
            query: The search query.
            texts: List of document texts to score.

        Returns:
            Relevance scores in the same order as input texts.

        Raises:
            ValueError: If API key is not configured or API returns an error.
        """
        if not self.api_key:
            raise ValueError(
                "API key not configured. Set BGE_RERANKER_API_KEY or BASETEN_API_KEY."
            )

        if not texts:
            return []

        response = self._get_session().post(
            self.endpoint,
            headers={
                "Authorization": f"Api-Key {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "query": query,
                "texts": texts,
                "truncate": True,
                "raw_scores": True,
                "return_text": False,
                "truncation_direction": "Right",
            },
            timeout=120,
        )

        if not response.ok:
            raise ValueError(f"API error {response.status_code}: {response.text}")

        # Response format: [{"index": 1, "score": 5.54}, {"index": 0, "score": -11.02}, ...]
        # Results are sorted by score descending; we reorder by original index
        result = response.json()
        if not isinstance(result, list):
            raise ValueError(f"Unexpected response format: {result}")

        scores = [0.0] * len(texts)
        for item in result:
            idx = item.get("index")
            if idx is not None and 0 <= idx < len(texts):
                scores[idx] = float(item.get("score", 0.0))

        return scores

    def process(
        self,
        results: List[Tuple[str, float]],
        input_data: Dict[str, Any],
        state: Dict[str, Any],
    ) -> List[Tuple[str, float]]:
        """Rerank results using the BGE model.

        Args:
            results: List of (doc_id, score) tuples from initial retrieval.
            input_data: Dict containing query and other input info.
            state: Pipeline state containing doc_content_map.

        Returns:
            Reranked list of (doc_id, score) tuples, limited to top_k.
        """
        if not results:
            return results

        query = input_data.get(self.query_key, "")
        if not query:
            return results[: self.top_k]

        doc_content_map = state.get("doc_content_map", {})

        # Collect documents that have content
        doc_ids, texts = [], []
        for doc_id, _ in results:
            content = doc_content_map.get(doc_id, "")
            if content:
                doc_ids.append(doc_id)
                texts.append(content)

        if not texts:
            return results[: self.top_k]

        # Score in batches
        all_scores = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            all_scores.extend(self._call_api(query, batch))

        # Sort by score descending
        reranked = sorted(zip(doc_ids, all_scores), key=lambda x: x[1], reverse=True)

        # Apply min_score filter if configured
        if self.min_score is not None:
            reranked = [(d, s) for d, s in reranked if s >= self.min_score]

        return reranked[: self.top_k]

    def rerank_standalone(
        self,
        query: str,
        documents: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Standalone reranking for debugging/testing.

        Args:
            query: The search query.
            documents: List of dicts with 'id', 'content', and optionally 'score', 'title'.

        Returns:
            Documents with 'rerank_score' field added, sorted by score descending.
        """
        if not documents:
            return documents

        # Map document index to content for docs that have content
        index_to_content = {
            i: doc.get("content", "")
            for i, doc in enumerate(documents)
            if doc.get("content")
        }

        if not index_to_content:
            return documents

        # Prepare texts in index order
        sorted_indices = sorted(index_to_content.keys())
        texts = [index_to_content[i] for i in sorted_indices]

        # Score in batches
        all_scores = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            all_scores.extend(self._call_api(query, batch))

        # Map scores back to original indices
        index_to_score = dict(zip(sorted_indices, all_scores))

        # Build result with scores
        result_docs = []
        for i, doc in enumerate(documents):
            doc_copy = dict(doc)
            doc_copy["rerank_score"] = index_to_score.get(i, 0.0)
            result_docs.append(doc_copy)

        # Sort by rerank score descending
        result_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
        return result_docs
