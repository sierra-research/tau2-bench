from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


class BaseRetriever(ABC):
    """Abstract base class for retrievers that fetch relevant documents from an index."""

    def __init__(self, **params):
        """Initialize the retriever with arbitrary keyword parameters.

        Args:
            **params: Configuration parameters for the retriever.
        """
        self.params = params

    @abstractmethod
    def retrieve(
        self, input_data: Dict[str, Any], state: Dict[str, Any]
    ) -> List[Tuple[str, float]]:
        """Retrieve documents relevant to a single input query.

        Args:
            input_data: A dict containing the query and any associated metadata.
            state: Shared pipeline state holding indexes and other precomputed data.

        Returns:
            A list of (document_id, score) tuples, typically sorted by relevance.
        """
        pass

    def retrieve_batch(
        self, input_data_list: List[Dict[str, Any]], state: Dict[str, Any]
    ) -> List[List[Tuple[str, float]]]:
        """Retrieve documents for a batch of input queries.

        Default implementation applies :meth:`retrieve` to each input independently.

        Args:
            input_data_list: A list of input data dicts, one per query.
            state: Shared pipeline state holding indexes and other precomputed data.

        Returns:
            A list of result lists, one per input query.
        """
        return [self.retrieve(input_data, state) for input_data in input_data_list]
