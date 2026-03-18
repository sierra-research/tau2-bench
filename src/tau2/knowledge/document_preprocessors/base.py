from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseDocumentPreprocessor(ABC):
    """Base class for document preprocessors that run during index_documents().

    Document preprocessors prepare documents for retrieval by building indexes
    or other data structures (e.g., BM25 indexes, embedding matrices) and
    storing them in the shared pipeline state dict. They may also transform
    the documents themselves (e.g., chunking, filtering, augmenting metadata).

    Preprocessors are chained: each one's returned documents are passed as
    input to the next, so process() must always return the documents list.
    """

    def __init__(self, **params):
        self.params = params

    @abstractmethod
    def process(
        self, documents: List[Dict[str, Any]], state: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Process documents and/or update pipeline state.

        Args:
            documents: List of document dicts (must contain at least "id").
            state: Shared pipeline state dict. Preprocessors typically store
                indexes or precomputed data here for use by retrievers.

        Returns:
            The (possibly transformed) list of documents. Even if documents are
            not modified, they must be returned for downstream chaining.
        """
        pass
