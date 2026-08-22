"""OpenAI embedder using text-embedding models."""

import os
from typing import List

import numpy as np
from openai import OpenAI

from tau2.knowledge.embedders.base import BaseEmbedder

OPENROUTER_EMBEDDING_TRANSPORT = "openrouter-openai-provider-v1"


def get_openai_cache_config(params: dict) -> dict:
    """Include transport details that can change otherwise-identical embeddings."""
    config = dict(params)
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url and "openrouter.ai" in base_url.lower():
        config["_transport"] = OPENROUTER_EMBEDDING_TRANSPORT
    return config


class OpenAIEmbedder(BaseEmbedder):
    """Embedder using OpenAI's embedding models."""

    def __init__(self, model: str = "text-embedding-ada-002", api_key: str = None):
        """
        Initialize OpenAI embedder.

        Args:
            model: OpenAI model name. Supported models include:
                   - text-embedding-ada-002 (default, 1536 dimensions)
                   - text-embedding-3-small (1536 dimensions)
                   - text-embedding-3-large (3072 dimensions)
            api_key: API key override. When ``OPENAI_BASE_URL`` points to
                OpenRouter, ``OPENROUTER_API_KEY`` is used by default;
                otherwise ``OPENAI_API_KEY`` is used.
        """
        self.model = model
        base_url = os.getenv("OPENAI_BASE_URL")
        self._using_openrouter = bool(base_url and "openrouter.ai" in base_url.lower())
        self._api_model = (
            f"openai/{model}" if self._using_openrouter and "/" not in model else model
        )

        default_api_key = (
            os.getenv("OPENROUTER_API_KEY")
            if self._using_openrouter
            else os.getenv("OPENAI_API_KEY")
        )
        client_kwargs = {"api_key": api_key or default_api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed texts using OpenAI API.

        Args:
            texts: List of text strings to embed

        Returns:
            Array of embeddings with shape (len(texts), embedding_dim)
        """
        if not texts:
            raise ValueError("No text to embed.")

        request_kwargs = {"input": texts, "model": self._api_model}
        if self._using_openrouter:
            # Match direct OpenAI embedding behavior while honoring an
            # OpenRouter-only credential setup. Disabling provider fallback
            # prevents Azure routing from changing dense-retrieval rankings.
            request_kwargs["extra_body"] = {
                "provider": {
                    "order": ["OpenAI"],
                    "allow_fallbacks": False,
                }
            }
        response = self.client.embeddings.create(**request_kwargs)
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings)

    def get_name(self) -> str:
        """Return the name of the embedder."""
        return f"openai_{self.model}"
