"""Tests for OpenAI-compatible embedding routing."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from tau2.knowledge.document_preprocessors.embedding_indexer import EmbeddingIndexer
from tau2.knowledge.embedders.openai_embedder import (
    OPENROUTER_EMBEDDING_TRANSPORT,
    OpenAIEmbedder,
)
from tau2.knowledge.input_preprocessors.embedding_encoder import EmbeddingEncoder


def test_openai_embedder_uses_direct_openai_defaults(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")

    with patch("tau2.knowledge.embedders.openai_embedder.OpenAI") as client_cls:
        embeddings = MagicMock()
        embeddings.create.return_value = SimpleNamespace(
            data=[SimpleNamespace(embedding=[1.0, 2.0])]
        )
        client_cls.return_value.embeddings = embeddings

        result = OpenAIEmbedder(model="text-embedding-3-large").embed(["hello"])

    client_cls.assert_called_once_with(api_key="openai-test-key")
    embeddings.create.assert_called_once_with(
        input=["hello"], model="text-embedding-3-large"
    )
    np.testing.assert_array_equal(result, np.array([[1.0, 2.0]]))


def test_openai_embedder_routes_through_openrouter_without_provider_drift(
    monkeypatch,
):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-test-key")

    with patch("tau2.knowledge.embedders.openai_embedder.OpenAI") as client_cls:
        embeddings = MagicMock()
        embeddings.create.return_value = SimpleNamespace(
            data=[SimpleNamespace(embedding=[3.0, 4.0])]
        )
        client_cls.return_value.embeddings = embeddings

        result = OpenAIEmbedder(model="text-embedding-3-large").embed(["hello"])

    client_cls.assert_called_once_with(
        api_key="openrouter-test-key",
        base_url="https://openrouter.ai/api/v1",
    )
    embeddings.create.assert_called_once_with(
        input=["hello"],
        model="openai/text-embedding-3-large",
        extra_body={
            "provider": {
                "order": ["OpenAI"],
                "allow_fallbacks": False,
            }
        },
    )
    np.testing.assert_array_equal(result, np.array([[3.0, 4.0]]))


def test_openrouter_transport_uses_distinct_document_cache(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

    indexer = EmbeddingIndexer(
        embedder_type="openai",
        embedder_params={"model": "text-embedding-3-large"},
    )

    assert indexer._get_cache_config() == {
        "model": "text-embedding-3-large",
        "_transport": OPENROUTER_EMBEDDING_TRANSPORT,
    }


def test_openrouter_transport_uses_distinct_query_cache(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-test-key")

    encoder = EmbeddingEncoder(
        embedder_type="openai",
        embedder_params={"model": "text-embedding-3-large"},
    )

    assert encoder._get_cache_config() == {
        "model": "text-embedding-3-large",
        "_transport": OPENROUTER_EMBEDDING_TRANSPORT,
    }
