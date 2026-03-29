import re
from typing import Any, Dict, List, Tuple

from tau2.knowledge.registry import register_retriever
from tau2.knowledge.retrievers.base import BaseRetriever


@register_retriever("grep")
class GrepRetriever(BaseRetriever):
    def __init__(
        self,
        query_key: str = "query",
        content_state_key: str = "doc_content_map",
        top_k: int = 10,
        case_sensitive: bool = False,
        **kwargs,
    ):
        super().__init__(
            query_key=query_key,
            content_state_key=content_state_key,
            top_k=top_k,
            case_sensitive=case_sensitive,
            **kwargs,
        )
        self.query_key = query_key
        self.content_state_key = content_state_key
        self.top_k = top_k
        self.case_sensitive = case_sensitive

    def retrieve(
        self, input_data: Dict[str, Any], state: Dict[str, Any]
    ) -> List[Tuple[str, float]]:
        query = input_data[self.query_key]
        doc_content_map = state[self.content_state_key]

        if not query.strip():
            return []

        flags = 0 if self.case_sensitive else re.IGNORECASE

        try:
            pattern = re.compile(query, flags)
        except re.error:
            pattern = re.compile(re.escape(query), flags)

        results = []
        for doc_id, content in doc_content_map.items():
            matches = pattern.findall(content)
            if matches:
                score = float(len(matches))
                results.append((doc_id, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[: self.top_k]

    def retrieve_batch(
        self, input_data_list: List[Dict[str, Any]], state: Dict[str, Any]
    ) -> List[List[Tuple[str, float]]]:
        return [self.retrieve(input_data, state) for input_data in input_data_list]
