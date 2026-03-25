from typing import Any, Dict, List, Tuple

import numpy as np

from tau2.knowledge.registry import register_retriever
from tau2.knowledge.retrievers.base import BaseRetriever


@register_retriever("cosine")
class CosineRetriever(BaseRetriever):
    def __init__(
        self,
        embedding_key: str = "query_embedding",
        index_key: str = "doc_embeddings",
        top_k: int = 10,
        **kwargs,
    ):
        super().__init__(
            embedding_key=embedding_key,
            index_key=index_key,
            top_k=top_k,
            **kwargs,
        )
        self.embedding_key = embedding_key
        self.index_key = index_key
        self.top_k = top_k

    def retrieve(
        self, input_data: Dict[str, Any], state: Dict[str, Any]
    ) -> List[Tuple[str, float]]:
        query_embedding = input_data.get(self.embedding_key)
        if query_embedding is None:
            return []

        doc_embeddings = state[self.index_key]
        doc_ids = state[f"{self.index_key}_doc_ids"]

        scores = self._cosine_similarity_single(query_embedding, doc_embeddings)

        top_k = min(self.top_k, len(doc_ids))
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = [(doc_ids[idx], float(scores[idx])) for idx in top_indices]

        return results

    def retrieve_batch(
        self, input_data_list: List[Dict[str, Any]], state: Dict[str, Any]
    ) -> List[List[Tuple[str, float]]]:
        doc_embeddings = state[self.index_key]
        doc_ids = state[f"{self.index_key}_doc_ids"]

        query_embeddings = np.array(
            [input_data[self.embedding_key] for input_data in input_data_list]
        )

        all_scores = self._cosine_similarity_batch(query_embeddings, doc_embeddings)

        top_k = min(self.top_k, len(doc_ids))
        all_results = []
        for scores in all_scores:
            top_indices = np.argsort(scores)[::-1][:top_k]
            results = [(doc_ids[idx], float(scores[idx])) for idx in top_indices]
            all_results.append(results)

        return all_results

    def _cosine_similarity_single(
        self, query: np.ndarray, docs: np.ndarray
    ) -> np.ndarray:
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return np.zeros(len(docs))
        normalized_query = query / query_norm

        doc_norms = np.linalg.norm(docs, axis=1, keepdims=True)
        doc_norms = np.where(doc_norms == 0, 1, doc_norms)
        normalized_docs = docs / doc_norms

        similarities = np.dot(normalized_docs, normalized_query)
        return similarities

    def _cosine_similarity_batch(
        self, queries: np.ndarray, docs: np.ndarray
    ) -> np.ndarray:
        query_norms = np.linalg.norm(queries, axis=1, keepdims=True)
        query_norms = np.where(query_norms == 0, 1, query_norms)
        normalized_queries = queries / query_norms

        doc_norms = np.linalg.norm(docs, axis=1, keepdims=True)
        doc_norms = np.where(doc_norms == 0, 1, doc_norms)
        normalized_docs = docs / doc_norms

        similarities = np.dot(normalized_queries, normalized_docs.T)
        return similarities
