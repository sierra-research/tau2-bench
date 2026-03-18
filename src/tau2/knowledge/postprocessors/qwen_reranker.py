"""Qwen reranker postprocessor using Qwen reranker model."""

import json
import os
from typing import Any, Dict, List, Tuple

import requests

from tau2.knowledge.postprocessors.base import BasePostprocessor
from tau2.knowledge.registry import register_postprocessor

PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
    'Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
    "<Instruct>: {instruction}\n"
    "<Query>: {query}\n"
    "<Document>: {document}<|im_end|>\n"
    "<|im_start|>assistant\n"
    "<think>\n\n</think>\n\n"
)

DEFAULT_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)

# Model endpoints by size
MODEL_ENDPOINTS = {
    "8b": "https://model-232n5713.api.sierra.baseten.io/environments/production/predict",
    "0.6b": "https://model-zq8vxggw.api.sierra.baseten.io/environments/production/predict",
}
DEFAULT_MODEL_SIZE = "8b"


@register_postprocessor("qwen_reranker")
class QwenReranker(BasePostprocessor):
    """Qwen reranker using Baseten-hosted API.

    This reranker uses a Qwen judge-style model to score query-document relevance.
    The model outputs "yes" or "no" with associated log-likelihood scores.
    Higher "yes" scores indicate stronger relevance.
    """

    def __init__(
        self,
        endpoint: str = None,
        api_key: str = None,
        instruction: str = None,
        top_k: int = 5,
        query_key: str = "query",
        batch_size: int = 32,
        min_score: float = None,
        model_size: str = None,
        **kwargs,
    ):
        """Initialize the Qwen reranker.

        Args:
            endpoint: Baseten API endpoint URL. If provided, overrides model_size selection.
            api_key: API key. Reads from QWEN_RERANKER_API_KEY or BASETEN_API_KEY env var if not provided.
            instruction: Custom instruction for the reranker.
            top_k: Number of top results to return after reranking.
            query_key: Key to look up query in input_data dict.
            batch_size: Number of query-doc pairs to process per API call.
            min_score: Minimum score threshold. Documents below this are filtered out.
                       If None, no filtering is applied.
            model_size: Model size to use ("8b" or "0.6b"). Defaults to "8b".
        """
        super().__init__(**kwargs)
        self.model_size = model_size or DEFAULT_MODEL_SIZE
        if endpoint:
            self.endpoint = endpoint
        elif self.model_size in MODEL_ENDPOINTS:
            self.endpoint = MODEL_ENDPOINTS[self.model_size]
        else:
            raise ValueError(
                f"Unknown model_size '{self.model_size}'. Choose from: {list(MODEL_ENDPOINTS.keys())}"
            )
        self.api_key = (
            api_key
            or os.environ.get("QWEN_RERANKER_API_KEY")
            or os.environ.get("BASETEN_API_KEY")
        )
        self.instruction = instruction or DEFAULT_INSTRUCTION
        self.top_k = top_k
        self.query_key = query_key
        self.batch_size = batch_size
        self.min_score = min_score
        self._session = None

    def _get_session(self) -> requests.Session:
        """Get or create a requests session for connection reuse."""
        if self._session is None:
            self._session = requests.Session()
        return self._session

    def _format_prompt(self, query: str, document: str) -> str:
        """Format the prompt for the Qwen reranker model."""
        return PROMPT_TEMPLATE.format(
            instruction=self.instruction,
            query=query,
            document=document,
        )

    def _call_api(self, prompts: List[str]) -> List[float]:
        """Call the Qwen reranker API.

        Args:
            prompts: List of formatted prompt strings.

        Returns:
            List of relevance scores (higher = more relevant).
        """
        if not self.api_key:
            raise ValueError(
                "API key not configured. Set QWEN_RERANKER_API_KEY or BASETEN_API_KEY environment variable."
            )

        if not prompts:
            return []

        # API expects batch format: [[prompt1], [prompt2], ...] for multiple inputs
        batch_inputs = [[p] for p in prompts]

        payload = {
            "inputs": batch_inputs,
            "truncate": True,
            "raw_scores": True,
            "truncation_direction": "Right",
        }

        response = self._get_session().post(
            self.endpoint,
            headers={
                "Authorization": f"Api-Key {self.api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            timeout=120,
        )

        if not response.ok:
            raise ValueError(f"API error {response.status_code}: {response.text}")

        # Parse response: [[{"score": float, "label": "yes"|"no"}, ...], ...]
        # Extract the "yes" score from each result
        scores = []
        for item in response.json():
            if isinstance(item, list):
                yes_score = next(
                    (
                        float(p.get("score", 0.0))
                        for p in item
                        if p.get("label") == "yes"
                    ),
                    0.0,
                )
                scores.append(yes_score)
            else:
                scores.append(0.0)

        return scores

    def process(
        self,
        results: List[Tuple[str, float]],
        input_data: Dict[str, Any],
        state: Dict[str, Any],
    ) -> List[Tuple[str, float]]:
        """Rerank results using the Qwen model.

        Args:
            results: List of (doc_id, score) tuples from initial retrieval.
            input_data: Dict containing query and other input info.
            state: Pipeline state containing doc_content_map.

        Returns:
            Reranked list of (doc_id, score) tuples.
        """
        if not results:
            return results

        query = input_data.get(self.query_key, "")
        if not query:
            return results[: self.top_k]

        doc_content_map = state.get("doc_content_map", {})

        # Build prompts for documents with content
        prompts = []
        valid_doc_ids = []
        for doc_id, _ in results:
            content = doc_content_map.get(doc_id, "")
            if content:
                prompts.append(self._format_prompt(query, content))
                valid_doc_ids.append(doc_id)

        if not prompts:
            return results[: self.top_k]

        # Get scores in batches
        all_scores = []
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i : i + self.batch_size]
            all_scores.extend(self._call_api(batch))

        # Combine and sort by score
        reranked = sorted(
            zip(valid_doc_ids, all_scores),
            key=lambda x: x[1],
            reverse=True,
        )

        # Apply min_score filter if configured
        if self.min_score is not None:
            reranked = [
                (doc_id, score) for doc_id, score in reranked if score >= self.min_score
            ]

        return reranked[: self.top_k]

    def rerank_standalone(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        instruction: str = None,
    ) -> List[Dict[str, Any]]:
        """Standalone reranking for debugging/testing.

        Args:
            query: The search query.
            documents: List of dicts with 'id', 'content', and optionally 'score', 'title'.
            instruction: Optional custom instruction (overrides instance instruction).

        Returns:
            List of documents with 'rerank_score' field, sorted by score descending.
        """
        if not documents:
            return documents

        # Temporarily override instruction if provided
        original_instruction = self.instruction
        if instruction:
            self.instruction = instruction

        try:
            # Build prompts for documents with content
            prompts = []
            valid_indices = []
            for i, doc in enumerate(documents):
                content = doc.get("content", "")
                if content:
                    prompts.append(self._format_prompt(query, content))
                    valid_indices.append(i)

            if not prompts:
                return documents

            # Get scores in batches
            all_scores = []
            for i in range(0, len(prompts), self.batch_size):
                batch = prompts[i : i + self.batch_size]
                all_scores.extend(self._call_api(batch))

            # Add scores to documents
            result_docs = []
            score_idx = 0
            for i, doc in enumerate(documents):
                doc_copy = dict(doc)
                if i in valid_indices:
                    doc_copy["rerank_score"] = all_scores[score_idx]
                    score_idx += 1
                else:
                    doc_copy["rerank_score"] = 0.0
                result_docs.append(doc_copy)

            # Sort by rerank score descending
            result_docs.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
            return result_docs

        finally:
            self.instruction = original_instruction
