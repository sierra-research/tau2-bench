"""Base embedder interface."""

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class BaseEmbedder(ABC):
    """Abstract base class for all embedders."""

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            Array of embeddings with shape (len(texts), embedding_dim)
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the name of the embedder."""
        pass
