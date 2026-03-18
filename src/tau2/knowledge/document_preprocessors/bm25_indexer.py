from typing import Any, Dict, List

from rank_bm25 import BM25Okapi

from tau2.knowledge.document_preprocessors.base import (
    BaseDocumentPreprocessor,
)
from tau2.knowledge.registry import register_document_preprocessor


@register_document_preprocessor("bm25_indexer")
class BM25Indexer(BaseDocumentPreprocessor):
    def __init__(
        self,
        state_key: str = "bm25",
        content_field: str = "text",
        **kwargs,
    ):
        super().__init__(
            state_key=state_key,
            content_field=content_field,
            **kwargs,
        )
        self.state_key = state_key
        self.content_field = content_field

    def process(
        self, documents: List[Dict[str, Any]], state: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        texts = []
        for doc in documents:
            text = doc.get(self.content_field) or doc.get("content") or doc.get("text")
            if text is None:
                raise ValueError(
                    f"Document {doc.get('id', 'unknown')} missing content field"
                )
            texts.append(text)

        tokenized_corpus = [text.lower().split() for text in texts]
        bm25 = BM25Okapi(tokenized_corpus)

        state[self.state_key] = bm25
        state[f"{self.state_key}_doc_ids"] = [doc["id"] for doc in documents]

        return documents
