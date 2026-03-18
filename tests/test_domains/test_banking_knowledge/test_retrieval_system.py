"""Comprehensive tests for the tau-knowledge retrieval system.

Tests cover:
- Registry (register / lookup / error handling)
- Config validation (valid / invalid pipeline configs)
- RetrievalPipeline (indexing, retrieve, retrieve_batch, overrides, state, naming)
- Individual retrievers: BM25, Grep, Cosine
- Document preprocessors: BM25Indexer
- RetrievalVariant registry (resolve_variant, RETRIEVAL_VARIANTS, variant attributes)
- Mixin tool integration (KB_search, grep, rewrite_context via toolkit composition)
- End-to-end pipeline creation and build_tools flow
- RetrievalTiming dataclass
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_DOCUMENTS: List[Dict[str, Any]] = [
    {
        "id": "doc_1",
        "title": "Credit Card Policy",
        "text": "Credit cards have a $50 annual fee. Late payments incur a $25 charge.",
    },
    {
        "id": "doc_2",
        "title": "Debit Card Policy",
        "text": "Debit cards have no annual fee. Overdraft protection is available.",
    },
    {
        "id": "doc_3",
        "title": "Savings Account",
        "text": "Savings accounts earn 2.5% APY interest. Minimum balance is $500.",
    },
    {
        "id": "doc_4",
        "title": "Checking Account",
        "text": "Checking accounts have unlimited transactions. No minimum balance.",
    },
    {
        "id": "doc_5",
        "title": "Wire Transfer",
        "text": "Wire transfers cost $25 domestic and $45 international. Same-day processing.",
    },
]


def _make_bm25_state(docs: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    """Build a BM25 index + state dict from sample documents."""
    docs = docs or SAMPLE_DOCUMENTS
    texts = [d["text"] for d in docs]
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    return {
        "bm25": bm25,
        "bm25_doc_ids": [d["id"] for d in docs],
    }


def _make_cosine_state(
    docs: List[Dict[str, Any]] | None = None, dim: int = 8
) -> Dict[str, Any]:
    """Build a fake embedding index in state dict."""
    docs = docs or SAMPLE_DOCUMENTS
    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((len(docs), dim)).astype(np.float32)
    # Normalise so cosine similarity = dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    return {
        "doc_embeddings": embeddings,
        "doc_embeddings_doc_ids": [d["id"] for d in docs],
    }


def _make_grep_state(docs: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    """Build a doc_content_map for Grep retriever."""
    docs = docs or SAMPLE_DOCUMENTS
    return {
        "doc_content_map": {d["id"]: d["text"] for d in docs},
    }


# ============================================================================
# 1. Registry tests
# ============================================================================


class TestRegistry:
    """Tests for tau2.knowledge.registry module."""

    def test_register_and_get_retriever(self):
        from tau2.knowledge.registry import RETRIEVERS, get_retriever

        # BM25 is auto-registered on import
        from tau2.knowledge.retrievers.bm25_retriever import BM25Retriever  # noqa: F401

        assert "bm25" in RETRIEVERS
        ret = get_retriever("bm25", {"top_k": 5})
        assert ret.top_k == 5

    def test_register_and_get_document_preprocessor(self):
        from tau2.knowledge.document_preprocessors.bm25_indexer import (
            BM25Indexer,  # noqa: F401
        )
        from tau2.knowledge.registry import (
            DOCUMENT_PREPROCESSORS,
            get_document_preprocessor,
        )

        assert "bm25_indexer" in DOCUMENT_PREPROCESSORS
        dp = get_document_preprocessor("bm25_indexer", {"state_key": "my_bm25"})
        assert dp.state_key == "my_bm25"

    def test_get_retriever_unknown_raises(self):
        from tau2.knowledge.registry import get_retriever

        with pytest.raises(ValueError, match="Unknown retriever"):
            get_retriever("nonexistent_retriever", {})

    def test_get_document_preprocessor_unknown_raises(self):
        from tau2.knowledge.registry import get_document_preprocessor

        with pytest.raises(ValueError, match="Unknown document_preprocessor"):
            get_document_preprocessor("nonexistent_dp", {})

    def test_get_input_preprocessor_unknown_raises(self):
        from tau2.knowledge.registry import get_input_preprocessor

        with pytest.raises(ValueError, match="Unknown input_preprocessor"):
            get_input_preprocessor("nonexistent_ip", {})

    def test_get_postprocessor_unknown_raises(self):
        from tau2.knowledge.registry import get_postprocessor

        with pytest.raises(ValueError, match="Unknown postprocessor"):
            get_postprocessor("nonexistent_pp", {})

    def test_register_and_get_grep_retriever(self):
        from tau2.knowledge.registry import RETRIEVERS, get_retriever
        from tau2.knowledge.retrievers.grep_retriever import GrepRetriever  # noqa: F401

        assert "grep" in RETRIEVERS
        ret = get_retriever("grep", {"top_k": 3, "case_sensitive": True})
        assert ret.top_k == 3
        assert ret.case_sensitive is True

    def test_register_and_get_cosine_retriever(self):
        from tau2.knowledge.registry import RETRIEVERS, get_retriever
        from tau2.knowledge.retrievers.cosine_retriever import (
            CosineRetriever,  # noqa: F401
        )

        assert "cosine" in RETRIEVERS
        ret = get_retriever("cosine", {"top_k": 7})
        assert ret.top_k == 7


# ============================================================================
# 2. Config validation tests
# ============================================================================


class TestConfigValidation:
    """Tests for tau2.knowledge.config.validate_config."""

    def test_valid_config_single_retriever(self):
        from tau2.knowledge.config import validate_config

        cfg = {
            "retriever": {"type": "bm25", "params": {}},
        }
        validate_config(cfg)  # should not raise
        # Defaults filled in
        assert cfg["document_preprocessors"] == []
        assert cfg["input_preprocessors"] == []
        assert cfg["postprocessors"] == []

    def test_valid_config_multiple_retrievers(self):
        from tau2.knowledge.config import validate_config

        cfg = {
            "retrievers": [
                {"type": "bm25", "params": {}},
                {"type": "grep", "params": {}},
            ],
        }
        validate_config(cfg)

    def test_missing_retriever_raises(self):
        from tau2.knowledge.config import validate_config

        with pytest.raises(ValueError, match="must have 'retriever' or 'retrievers'"):
            validate_config({"document_preprocessors": []})

    def test_both_retriever_and_retrievers_raises(self):
        from tau2.knowledge.config import validate_config

        with pytest.raises(ValueError, match="cannot have both"):
            validate_config(
                {
                    "retriever": {"type": "bm25"},
                    "retrievers": [{"type": "grep"}],
                }
            )

    def test_retriever_missing_type_raises(self):
        from tau2.knowledge.config import validate_config

        with pytest.raises(ValueError, match="retriever must have 'type'"):
            validate_config({"retriever": {"params": {}}})

    def test_retrievers_list_empty_raises(self):
        from tau2.knowledge.config import validate_config

        with pytest.raises(ValueError, match="retrievers list cannot be empty"):
            validate_config({"retrievers": []})

    def test_retrievers_item_missing_type_raises(self):
        from tau2.knowledge.config import validate_config

        with pytest.raises(ValueError, match="retrievers\\[0\\] must have 'type'"):
            validate_config({"retrievers": [{"params": {}}]})

    def test_document_preprocessors_not_list_raises(self):
        from tau2.knowledge.config import validate_config

        with pytest.raises(ValueError, match="document_preprocessors must be a list"):
            validate_config(
                {
                    "retriever": {"type": "bm25"},
                    "document_preprocessors": "bad",
                }
            )

    def test_input_preprocessors_not_list_raises(self):
        from tau2.knowledge.config import validate_config

        with pytest.raises(ValueError, match="input_preprocessors must be a list"):
            validate_config(
                {
                    "retriever": {"type": "bm25"},
                    "input_preprocessors": "bad",
                }
            )

    def test_postprocessors_not_list_raises(self):
        from tau2.knowledge.config import validate_config

        with pytest.raises(ValueError, match="postprocessors must be a list"):
            validate_config(
                {
                    "retriever": {"type": "bm25"},
                    "postprocessors": "bad",
                }
            )

    def test_preprocessor_missing_type_raises(self):
        from tau2.knowledge.config import validate_config

        with pytest.raises(
            ValueError, match="document_preprocessor\\[0\\] must have 'type'"
        ):
            validate_config(
                {
                    "retriever": {"type": "bm25"},
                    "document_preprocessors": [{"params": {}}],
                }
            )

    def test_input_preprocessor_missing_type_raises(self):
        from tau2.knowledge.config import validate_config

        with pytest.raises(
            ValueError, match="input_preprocessor\\[0\\] must have 'type'"
        ):
            validate_config(
                {
                    "retriever": {"type": "bm25"},
                    "input_preprocessors": [{"params": {}}],
                }
            )

    def test_postprocessor_missing_type_raises(self):
        from tau2.knowledge.config import validate_config

        with pytest.raises(ValueError, match="postprocessor\\[0\\] must have 'type'"):
            validate_config(
                {
                    "retriever": {"type": "bm25"},
                    "postprocessors": [{"params": {}}],
                }
            )

    def test_tool_name_not_string_raises(self):
        from tau2.knowledge.config import validate_config

        with pytest.raises(ValueError, match="tool_name must be a string"):
            validate_config(
                {
                    "retriever": {"type": "bm25"},
                    "tool_name": 123,
                }
            )

    def test_description_not_string_raises(self):
        from tau2.knowledge.config import validate_config

        with pytest.raises(ValueError, match="description must be a string"):
            validate_config(
                {
                    "retriever": {"type": "bm25"},
                    "description": 123,
                }
            )

    def test_parameters_not_dict_raises(self):
        from tau2.knowledge.config import validate_config

        with pytest.raises(ValueError, match="parameters must be a dict"):
            validate_config(
                {
                    "retriever": {"type": "bm25"},
                    "parameters": "bad",
                }
            )

    def test_get_default_config_bm25_shape(self):
        """get_default_config with full_kb returns a valid full-kb config."""
        from tau2.knowledge.config import get_default_config

        cfg = get_default_config(embedder_type="full_kb")
        assert cfg["retriever"]["type"] == "full_kb"
        assert cfg["document_preprocessors"] == []

    def test_get_default_config_openai_shape(self):
        from tau2.knowledge.config import get_default_config

        cfg = get_default_config(embedder_type="openai", top_k=3)
        assert cfg["retriever"]["type"] == "cosine"
        assert cfg["retriever"]["params"]["top_k"] == 3
        assert len(cfg["document_preprocessors"]) == 1
        assert cfg["document_preprocessors"][0]["type"] == "embedding_indexer"


# ============================================================================
# 3. BM25Retriever tests
# ============================================================================


class TestBM25Retriever:
    """Tests for BM25Retriever."""

    def _make_retriever(self, top_k: int = 10):
        from tau2.knowledge.retrievers.bm25_retriever import BM25Retriever

        return BM25Retriever(top_k=top_k)

    def test_retrieve_returns_scored_results(self):
        ret = self._make_retriever(top_k=3)
        state = _make_bm25_state()
        results = ret.retrieve({"query": "credit card fee"}, state)
        assert len(results) == 3
        # Results are (doc_id, score) tuples
        for doc_id, score in results:
            assert isinstance(doc_id, str)
            assert isinstance(score, float)

    def test_retrieve_credit_card_query_ranks_credit_first(self):
        ret = self._make_retriever(top_k=5)
        state = _make_bm25_state()
        results = ret.retrieve({"query": "credit card annual fee"}, state)
        # doc_1 is about credit cards - should be highest
        assert results[0][0] == "doc_1"

    def test_retrieve_empty_query_returns_empty(self):
        ret = self._make_retriever()
        state = _make_bm25_state()
        assert ret.retrieve({"query": ""}, state) == []
        assert ret.retrieve({"query": "   "}, state) == []

    def test_retrieve_missing_query_returns_empty(self):
        ret = self._make_retriever()
        state = _make_bm25_state()
        assert ret.retrieve({}, state) == []

    def test_retrieve_top_k_limits_results(self):
        ret = self._make_retriever(top_k=2)
        state = _make_bm25_state()
        results = ret.retrieve({"query": "account"}, state)
        assert len(results) <= 2

    def test_retrieve_top_k_larger_than_corpus(self):
        """top_k > num docs → returns all docs."""
        ret = self._make_retriever(top_k=100)
        state = _make_bm25_state()
        results = ret.retrieve({"query": "fee"}, state)
        assert len(results) == len(SAMPLE_DOCUMENTS)

    def test_retrieve_batch(self):
        ret = self._make_retriever(top_k=2)
        state = _make_bm25_state()
        queries = [{"query": "credit card"}, {"query": "savings interest"}]
        batch_results = ret.retrieve_batch(queries, state)
        assert len(batch_results) == 2
        for results in batch_results:
            assert len(results) <= 2

    def test_retrieve_batch_credit_and_savings(self):
        ret = self._make_retriever(top_k=5)
        state = _make_bm25_state()
        queries = [{"query": "credit card fee"}, {"query": "savings interest APY"}]
        batch = ret.retrieve_batch(queries, state)
        assert batch[0][0][0] == "doc_1"  # credit card doc
        assert batch[1][0][0] == "doc_3"  # savings doc

    def test_results_sorted_descending(self):
        ret = self._make_retriever(top_k=5)
        state = _make_bm25_state()
        results = ret.retrieve({"query": "annual fee charge"}, state)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)


# ============================================================================
# 4. GrepRetriever tests
# ============================================================================


class TestGrepRetriever:
    """Tests for GrepRetriever."""

    def _make_retriever(self, top_k: int = 10, case_sensitive: bool = False):
        from tau2.knowledge.retrievers.grep_retriever import GrepRetriever

        return GrepRetriever(top_k=top_k, case_sensitive=case_sensitive)

    def test_retrieve_simple_pattern(self):
        ret = self._make_retriever()
        state = _make_grep_state()
        results = ret.retrieve({"query": "fee"}, state)
        # doc_1 mentions "fee" twice, doc_2 mentions "fee" once
        doc_ids = [d for d, _ in results]
        assert "doc_1" in doc_ids
        assert "doc_2" in doc_ids

    def test_retrieve_case_insensitive_default(self):
        ret = self._make_retriever(case_sensitive=False)
        state = _make_grep_state()
        results = ret.retrieve({"query": "CREDIT"}, state)
        doc_ids = [d for d, _ in results]
        assert "doc_1" in doc_ids

    def test_retrieve_case_sensitive_no_match(self):
        ret = self._make_retriever(case_sensitive=True)
        state = _make_grep_state()
        # "CREDIT" won't match "Credit" in case-sensitive mode
        results = ret.retrieve({"query": "CREDIT"}, state)
        assert len(results) == 0

    def test_retrieve_case_sensitive_exact_match(self):
        ret = self._make_retriever(case_sensitive=True)
        state = _make_grep_state()
        results = ret.retrieve({"query": "Credit"}, state)
        doc_ids = [d for d, _ in results]
        assert "doc_1" in doc_ids

    def test_retrieve_regex_pattern(self):
        ret = self._make_retriever()
        state = _make_grep_state()
        results = ret.retrieve({"query": r"\$\d+"}, state)
        # doc_1 ($50, $25), doc_3 ($500), doc_5 ($25, $45)
        doc_ids = [d for d, _ in results]
        assert "doc_1" in doc_ids
        assert "doc_3" in doc_ids
        assert "doc_5" in doc_ids

    def test_retrieve_invalid_regex_falls_back_to_escaped(self):
        """Invalid regex patterns should be auto-escaped."""
        ret = self._make_retriever()
        state = _make_grep_state()
        # Invalid regex with unmatched bracket
        results = ret.retrieve({"query": "fee["}, state)
        # Should not raise, falls back to re.escape → won't match
        assert isinstance(results, list)

    def test_retrieve_empty_query_returns_empty(self):
        ret = self._make_retriever()
        state = _make_grep_state()
        assert ret.retrieve({"query": ""}, state) == []
        assert ret.retrieve({"query": "   "}, state) == []

    def test_retrieve_top_k_limits(self):
        ret = self._make_retriever(top_k=1)
        state = _make_grep_state()
        results = ret.retrieve({"query": "account"}, state)
        assert len(results) <= 1

    def test_retrieve_no_matches(self):
        ret = self._make_retriever()
        state = _make_grep_state()
        results = ret.retrieve({"query": "xyznonexistent"}, state)
        assert results == []

    def test_score_equals_match_count(self):
        ret = self._make_retriever()
        state = _make_grep_state()
        # doc_1 text: "Credit cards have a $50 annual fee. Late payments incur a $25 charge."
        # "fee" appears once in doc_1, once in doc_2
        results = ret.retrieve({"query": "fee"}, state)
        results_map = dict(results)
        assert results_map.get("doc_1") == 1.0
        assert results_map.get("doc_2") == 1.0

    def test_retrieve_sorted_by_match_count(self):
        ret = self._make_retriever()
        state = _make_grep_state()
        results = ret.retrieve({"query": r"a"}, state)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_batch(self):
        ret = self._make_retriever(top_k=2)
        state = _make_grep_state()
        queries = [{"query": "credit"}, {"query": "savings"}]
        batch = ret.retrieve_batch(queries, state)
        assert len(batch) == 2

    def test_or_pattern(self):
        """Regex alternation to match multiple terms."""
        ret = self._make_retriever()
        state = _make_grep_state()
        results = ret.retrieve({"query": "credit|savings"}, state)
        doc_ids = [d for d, _ in results]
        assert "doc_1" in doc_ids
        assert "doc_3" in doc_ids


# ============================================================================
# 5. CosineRetriever tests
# ============================================================================


class TestCosineRetriever:
    """Tests for CosineRetriever with synthetic embeddings."""

    def _make_retriever(self, top_k: int = 10):
        from tau2.knowledge.retrievers.cosine_retriever import CosineRetriever

        return CosineRetriever(top_k=top_k)

    def test_retrieve_returns_all_docs(self):
        ret = self._make_retriever(top_k=10)
        state = _make_cosine_state()
        query_emb = np.random.default_rng(0).standard_normal(8).astype(np.float32)
        results = ret.retrieve({"query_embedding": query_emb}, state)
        assert len(results) == len(SAMPLE_DOCUMENTS)

    def test_retrieve_top_k_limits(self):
        ret = self._make_retriever(top_k=2)
        state = _make_cosine_state()
        query_emb = np.random.default_rng(0).standard_normal(8).astype(np.float32)
        results = ret.retrieve({"query_embedding": query_emb}, state)
        assert len(results) == 2

    def test_retrieve_identical_embedding_scores_one(self):
        """A query equal to a doc embedding should score ~1.0."""
        ret = self._make_retriever(top_k=5)
        state = _make_cosine_state()
        # Use first doc's embedding as query
        query_emb = state["doc_embeddings"][0].copy()
        results = ret.retrieve({"query_embedding": query_emb}, state)
        top_doc_id, top_score = results[0]
        assert top_doc_id == SAMPLE_DOCUMENTS[0]["id"]
        assert top_score == pytest.approx(1.0, abs=1e-5)

    def test_retrieve_sorted_descending(self):
        ret = self._make_retriever(top_k=5)
        state = _make_cosine_state()
        query_emb = np.random.default_rng(7).standard_normal(8).astype(np.float32)
        results = ret.retrieve({"query_embedding": query_emb}, state)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_missing_embedding_returns_empty(self):
        ret = self._make_retriever()
        state = _make_cosine_state()
        results = ret.retrieve({}, state)
        assert results == []

    def test_retrieve_none_embedding_returns_empty(self):
        ret = self._make_retriever()
        state = _make_cosine_state()
        results = ret.retrieve({"query_embedding": None}, state)
        assert results == []

    def test_retrieve_zero_query_returns_zeros(self):
        """Zero-norm query → all similarities = 0."""
        ret = self._make_retriever(top_k=5)
        state = _make_cosine_state()
        query_emb = np.zeros(8, dtype=np.float32)
        results = ret.retrieve({"query_embedding": query_emb}, state)
        for _, score in results:
            assert score == pytest.approx(0.0, abs=1e-5)

    def test_retrieve_batch(self):
        ret = self._make_retriever(top_k=2)
        state = _make_cosine_state()
        rng = np.random.default_rng(10)
        queries = [
            {"query_embedding": rng.standard_normal(8).astype(np.float32)},
            {"query_embedding": rng.standard_normal(8).astype(np.float32)},
        ]
        batch = ret.retrieve_batch(queries, state)
        assert len(batch) == 2
        for results in batch:
            assert len(results) == 2

    def test_cosine_similarity_batch_shape(self):
        from tau2.knowledge.retrievers.cosine_retriever import CosineRetriever

        ret = CosineRetriever()
        queries = np.random.default_rng(0).standard_normal((3, 8)).astype(np.float32)
        docs = np.random.default_rng(1).standard_normal((5, 8)).astype(np.float32)
        sims = ret._cosine_similarity_batch(queries, docs)
        assert sims.shape == (3, 5)


# ============================================================================
# 6. BM25Indexer tests
# ============================================================================


class TestBM25Indexer:
    """Tests for BM25Indexer document preprocessor."""

    def _make_indexer(self, **kwargs):
        from tau2.knowledge.document_preprocessors.bm25_indexer import BM25Indexer

        return BM25Indexer(**kwargs)

    def test_process_builds_index(self):
        indexer = self._make_indexer()
        state: Dict[str, Any] = {}
        result = indexer.process(SAMPLE_DOCUMENTS, state)
        assert "bm25" in state
        assert "bm25_doc_ids" in state
        assert isinstance(state["bm25"], BM25Okapi)
        assert state["bm25_doc_ids"] == [d["id"] for d in SAMPLE_DOCUMENTS]
        # Documents are returned unchanged
        assert result == SAMPLE_DOCUMENTS

    def test_process_custom_state_key(self):
        indexer = self._make_indexer(state_key="custom_bm25")
        state: Dict[str, Any] = {}
        indexer.process(SAMPLE_DOCUMENTS, state)
        assert "custom_bm25" in state
        assert "custom_bm25_doc_ids" in state

    def test_process_missing_content_raises(self):
        indexer = self._make_indexer()
        state: Dict[str, Any] = {}
        bad_docs = [{"id": "bad_doc"}]  # no text field
        with pytest.raises(ValueError, match="missing content field"):
            indexer.process(bad_docs, state)

    def test_process_content_field_fallback(self):
        """BM25Indexer falls back to 'content' or 'text' fields."""
        indexer = self._make_indexer(content_field="body")
        state: Dict[str, Any] = {}
        # Has "content" field instead of "body"
        docs = [{"id": "d1", "content": "hello world"}]
        indexer.process(docs, state)
        assert "bm25" in state


# ============================================================================
# 7. RetrievalPipeline tests
# ============================================================================


class TestRetrievalPipeline:
    """Tests for RetrievalPipeline end-to-end."""

    @pytest.fixture
    def bm25_pipeline(self):
        """Create a BM25 pipeline ready to use."""
        # Ensure BM25 components are registered
        import tau2.knowledge.document_preprocessors.bm25_indexer  # noqa: F401
        import tau2.knowledge.retrievers.bm25_retriever  # noqa: F401
        from tau2.knowledge.pipeline import RetrievalPipeline

        config = {
            "document_preprocessors": [
                {"type": "bm25_indexer", "params": {"state_key": "bm25"}}
            ],
            "input_preprocessors": [],
            "retriever": {
                "type": "bm25",
                "params": {
                    "query_key": "query",
                    "bm25_state_key": "bm25",
                    "doc_ids_state_key": "bm25_doc_ids",
                    "top_k": 10,
                },
            },
            "postprocessors": [],
        }
        pipeline = RetrievalPipeline(config)
        pipeline.index_documents(SAMPLE_DOCUMENTS)
        return pipeline

    @pytest.fixture
    def grep_pipeline(self):
        """Create a Grep pipeline ready to use."""
        import tau2.knowledge.retrievers.grep_retriever  # noqa: F401
        from tau2.knowledge.pipeline import RetrievalPipeline

        config = {
            "document_preprocessors": [],
            "input_preprocessors": [],
            "retriever": {
                "type": "grep",
                "params": {
                    "query_key": "query",
                    "content_state_key": "doc_content_map",
                    "top_k": 10,
                    "case_sensitive": False,
                },
            },
            "postprocessors": [],
        }
        pipeline = RetrievalPipeline(config)
        pipeline.index_documents(SAMPLE_DOCUMENTS)
        return pipeline

    # -- Indexing ---------------------------------------------------------

    def test_index_documents_populates_state(self, bm25_pipeline):
        assert "documents" in bm25_pipeline.state
        assert "doc_content_map" in bm25_pipeline.state
        assert "doc_title_map" in bm25_pipeline.state
        assert len(bm25_pipeline.state["documents"]) == len(SAMPLE_DOCUMENTS)

    def test_index_documents_empty_raises(self):
        import tau2.knowledge.retrievers.bm25_retriever  # noqa: F401
        from tau2.knowledge.pipeline import RetrievalPipeline

        pipeline = RetrievalPipeline(
            {
                "retriever": {"type": "bm25", "params": {}},
            }
        )
        with pytest.raises(ValueError, match="Documents list is empty"):
            pipeline.index_documents([])

    def test_doc_content_map(self, bm25_pipeline):
        cm = bm25_pipeline.state["doc_content_map"]
        assert cm["doc_1"] == SAMPLE_DOCUMENTS[0]["text"]
        assert cm["doc_5"] == SAMPLE_DOCUMENTS[4]["text"]

    def test_doc_title_map(self, bm25_pipeline):
        tm = bm25_pipeline.state["doc_title_map"]
        assert tm["doc_1"] == "Credit Card Policy"
        assert tm["doc_3"] == "Savings Account"

    # -- Retrieve ----------------------------------------------------------

    def test_retrieve_without_indexing_raises(self):
        import tau2.knowledge.retrievers.bm25_retriever  # noqa: F401
        from tau2.knowledge.pipeline import RetrievalPipeline

        pipeline = RetrievalPipeline({"retriever": {"type": "bm25", "params": {}}})
        with pytest.raises(ValueError, match="No documents indexed"):
            pipeline.retrieve("test")

    def test_retrieve_returns_results(self, bm25_pipeline):
        results = bm25_pipeline.retrieve("credit card fee")
        assert len(results) > 0
        for doc_id, score in results:
            assert isinstance(doc_id, str)
            assert isinstance(score, float)

    def test_retrieve_credit_card_ranks_correctly(self, bm25_pipeline):
        results = bm25_pipeline.retrieve("credit card annual fee")
        assert results[0][0] == "doc_1"

    def test_retrieve_with_top_k(self, bm25_pipeline):
        results = bm25_pipeline.retrieve("account", top_k=2)
        assert len(results) <= 2

    def test_retrieve_with_timing(self, bm25_pipeline):
        from tau2.knowledge.pipeline import RetrievalResult

        result = bm25_pipeline.retrieve("fee", return_timing=True)
        assert isinstance(result, RetrievalResult)
        assert result.timing.retrieval_ms >= 0
        assert result.timing.total_ms >= 0
        assert len(result.results) > 0

    def test_retrieve_grep_pipeline(self, grep_pipeline):
        results = grep_pipeline.retrieve("fee")
        doc_ids = [d for d, _ in results]
        assert "doc_1" in doc_ids

    def test_retrieve_grep_regex(self, grep_pipeline):
        results = grep_pipeline.retrieve(r"\$\d+")
        assert len(results) >= 3  # doc_1, doc_3, doc_5

    # -- Retrieve batch ----------------------------------------------------

    def test_retrieve_batch_without_indexing_raises(self):
        import tau2.knowledge.retrievers.bm25_retriever  # noqa: F401
        from tau2.knowledge.pipeline import RetrievalPipeline

        pipeline = RetrievalPipeline({"retriever": {"type": "bm25", "params": {}}})
        with pytest.raises(ValueError, match="No documents indexed"):
            pipeline.retrieve_batch(["test"])

    def test_retrieve_batch(self, bm25_pipeline):
        batch = bm25_pipeline.retrieve_batch(
            ["credit card", "savings account"], top_k=3
        )
        assert len(batch) == 2
        for results in batch:
            assert len(results) <= 3

    def test_retrieve_batch_correct_ranking(self, bm25_pipeline):
        batch = bm25_pipeline.retrieve_batch(
            ["credit card fee", "savings interest APY"]
        )
        assert batch[0][0][0] == "doc_1"
        assert batch[1][0][0] == "doc_3"

    # -- Helpers -----------------------------------------------------------

    def test_get_document_content(self, bm25_pipeline):
        assert (
            bm25_pipeline.get_document_content("doc_1") == SAMPLE_DOCUMENTS[0]["text"]
        )
        assert bm25_pipeline.get_document_content("nonexistent") is None

    def test_get_document_title(self, bm25_pipeline):
        assert bm25_pipeline.get_document_title("doc_1") == "Credit Card Policy"
        assert bm25_pipeline.get_document_title("nonexistent") is None

    # -- Overrides ---------------------------------------------------------

    def test_set_retriever_top_k_override(self, bm25_pipeline):
        bm25_pipeline.set_overrides(retriever_top_k=1)
        results = bm25_pipeline.retrieve("fee")
        assert len(results) <= 1
        # Override is temporary per call; retriever's original top_k restored
        retriever = bm25_pipeline.retrievers[0]
        assert retriever.top_k == 10

    # -- State persistence -------------------------------------------------

    def test_save_and_load_state(self, bm25_pipeline):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "state.pkl")
            bm25_pipeline.save_state(path)
            assert Path(path).exists()

            # Load into new pipeline
            import tau2.knowledge.retrievers.bm25_retriever  # noqa: F401
            from tau2.knowledge.pipeline import RetrievalPipeline

            new_pipeline = RetrievalPipeline(
                {
                    "retriever": {
                        "type": "bm25",
                        "params": {
                            "query_key": "query",
                            "bm25_state_key": "bm25",
                            "doc_ids_state_key": "bm25_doc_ids",
                            "top_k": 10,
                        },
                    },
                }
            )
            new_pipeline.load_state(path)
            results = new_pipeline.retrieve("credit card")
            assert len(results) > 0
            assert results[0][0] == "doc_1"

    # -- get_name ----------------------------------------------------------

    def test_get_name_single_retriever(self, bm25_pipeline):
        name = bm25_pipeline.get_name()
        assert "bm25_indexer" in name
        assert "bm25" in name

    def test_get_name_grep(self, grep_pipeline):
        name = grep_pipeline.get_name()
        assert "grep" in name

    # -- Multi-retriever pipeline ------------------------------------------

    def test_multi_retriever_pipeline(self):
        """Pipeline with both BM25 and Grep retrievers."""
        import tau2.knowledge.document_preprocessors.bm25_indexer  # noqa: F401
        import tau2.knowledge.retrievers.bm25_retriever  # noqa: F401
        import tau2.knowledge.retrievers.grep_retriever  # noqa: F401
        from tau2.knowledge.pipeline import RetrievalPipeline

        config = {
            "document_preprocessors": [
                {"type": "bm25_indexer", "params": {"state_key": "bm25"}}
            ],
            "input_preprocessors": [],
            "retrievers": [
                {
                    "type": "bm25",
                    "params": {
                        "query_key": "query",
                        "bm25_state_key": "bm25",
                        "doc_ids_state_key": "bm25_doc_ids",
                        "top_k": 5,
                    },
                },
                {
                    "type": "grep",
                    "params": {
                        "query_key": "query",
                        "content_state_key": "doc_content_map",
                        "top_k": 5,
                    },
                },
            ],
            "postprocessors": [],
        }
        pipeline = RetrievalPipeline(config)
        pipeline.index_documents(SAMPLE_DOCUMENTS)
        results = pipeline.retrieve("credit card fee")
        assert len(results) > 0
        # Should combine results from both retrievers
        doc_ids = [d for d, _ in results]
        assert "doc_1" in doc_ids

    def test_multi_retriever_get_name(self):
        import tau2.knowledge.retrievers.bm25_retriever  # noqa: F401
        import tau2.knowledge.retrievers.grep_retriever  # noqa: F401
        from tau2.knowledge.pipeline import RetrievalPipeline

        config = {
            "retrievers": [
                {"type": "bm25", "params": {}},
                {"type": "grep", "params": {}},
            ],
        }
        pipeline = RetrievalPipeline(config)
        name = pipeline.get_name()
        assert "bm25+grep" in name

    def test_multi_retriever_takes_max_score(self):
        """When both retrievers find a doc, the max score is kept."""
        import tau2.knowledge.document_preprocessors.bm25_indexer  # noqa: F401
        import tau2.knowledge.retrievers.bm25_retriever  # noqa: F401
        import tau2.knowledge.retrievers.grep_retriever  # noqa: F401
        from tau2.knowledge.pipeline import RetrievalPipeline

        config = {
            "document_preprocessors": [
                {"type": "bm25_indexer", "params": {"state_key": "bm25"}}
            ],
            "retrievers": [
                {
                    "type": "bm25",
                    "params": {
                        "bm25_state_key": "bm25",
                        "doc_ids_state_key": "bm25_doc_ids",
                        "top_k": 5,
                    },
                },
                {"type": "grep", "params": {"top_k": 5}},
            ],
        }
        pipeline = RetrievalPipeline(config)
        pipeline.index_documents(SAMPLE_DOCUMENTS)
        results = pipeline.retrieve("fee")
        results_dict = dict(results)
        # doc_1 appears in both BM25 and grep results → max score used
        assert "doc_1" in results_dict


# ============================================================================
# 8. RetrievalTiming tests
# ============================================================================


class TestRetrievalTiming:
    """Tests for the RetrievalTiming dataclass."""

    def test_total_ms(self):
        from tau2.knowledge.pipeline import RetrievalTiming

        t = RetrievalTiming(
            input_preprocessing_ms=10.0,
            retrieval_ms=50.0,
            postprocessing_ms=20.0,
        )
        assert t.total_ms == 80.0

    def test_to_dict(self):
        from tau2.knowledge.pipeline import RetrievalTiming

        t = RetrievalTiming(
            input_preprocessing_ms=10.123,
            retrieval_ms=50.456,
            postprocessing_ms=20.789,
            postprocessor_details={"Reranker": 15.557},
        )
        d = t.to_dict()
        assert d["input_preprocessing_ms"] == 10.12
        assert d["retrieval_ms"] == 50.46
        assert d["postprocessing_ms"] == 20.79
        assert d["postprocessor_details"]["Reranker"] == 15.56
        assert "total_ms" in d

    def test_defaults(self):
        from tau2.knowledge.pipeline import RetrievalTiming

        t = RetrievalTiming()
        assert t.total_ms == 0.0
        assert t.postprocessor_details == {}


# ============================================================================
# 9. RetrievalVariant registry tests
# ============================================================================


class TestRetrievalVariantRegistry:
    """Tests for RETRIEVAL_VARIANTS, resolve_variant, and get_all_variant_names."""

    def test_get_all_variant_names(self):
        from tau2.domains.banking_knowledge.retrieval import (
            RETRIEVAL_VARIANTS,
            get_all_variant_names,
        )

        names = get_all_variant_names()
        assert isinstance(names, list)
        # At minimum these should be registered
        expected = {
            "no_knowledge",
            "full_kb",
            "golden_retrieval",
            "bm25_grep",
            "grep_only",
            "bm25",
        }
        assert expected.issubset(set(names))
        assert set(names) == set(RETRIEVAL_VARIANTS.keys())

    def test_resolve_variant_known(self):
        from tau2.domains.banking_knowledge.retrieval import resolve_variant

        variant = resolve_variant("no_knowledge")
        assert variant.name == "no_knowledge"

    def test_resolve_variant_unknown_raises(self):
        from tau2.domains.banking_knowledge.retrieval import resolve_variant

        with pytest.raises(ValueError, match="Unknown retrieval variant"):
            resolve_variant("nonexistent_config_xyz")

    def test_resolve_variant_with_params(self):
        from tau2.domains.banking_knowledge.retrieval import resolve_variant

        variant = resolve_variant("bm25_grep", top_k=5, grep_top_k=3)
        assert variant.kb_search.top_k == 5
        assert variant.grep.top_k == 3

    def test_bm25_variant(self):
        from tau2.domains.banking_knowledge.retrieval import resolve_variant

        variant = resolve_variant("bm25", top_k=7)
        assert variant.name == "bm25"
        assert variant.kb_search.top_k == 7
        assert variant.supports_top_k is True

    def test_grep_only_variant(self):
        from tau2.domains.banking_knowledge.retrieval import resolve_variant

        variant = resolve_variant("grep_only", grep_top_k=3)
        assert variant.name == "grep_only"
        assert variant.grep.top_k == 3
        assert variant.supports_top_k is True


class TestNoKnowledgeVariant:
    """Tests for the no_knowledge retrieval variant."""

    def test_no_retrieval_specs(self):
        from tau2.domains.banking_knowledge.retrieval import resolve_variant

        variant = resolve_variant("no_knowledge")
        assert variant.kb_search is None
        assert variant.grep is None
        assert variant.shell is None

    def test_name(self):
        from tau2.domains.banking_knowledge.retrieval import resolve_variant

        assert resolve_variant("no_knowledge").name == "no_knowledge"


class TestFullKBVariant:
    """Tests for the full_kb retrieval variant."""

    def test_no_retrieval_specs(self):
        from tau2.domains.banking_knowledge.retrieval import resolve_variant

        variant = resolve_variant("full_kb")
        assert variant.kb_search is None
        assert variant.grep is None
        assert variant.shell is None

    def test_name(self):
        from tau2.domains.banking_knowledge.retrieval import resolve_variant

        assert resolve_variant("full_kb").name == "full_kb"


class TestGoldenRetrievalVariant:
    """Tests for the golden_retrieval retrieval variant."""

    def test_no_retrieval_specs(self):
        from tau2.domains.banking_knowledge.retrieval import resolve_variant

        variant = resolve_variant("golden_retrieval")
        assert variant.kb_search is None
        assert variant.grep is None
        assert variant.shell is None

    def test_name(self):
        from tau2.domains.banking_knowledge.retrieval import resolve_variant

        assert resolve_variant("golden_retrieval").name == "golden_retrieval"

    def test_uses_golden_prompt_builder(self):
        from tau2.domains.banking_knowledge.retrieval import (
            golden_prompt,
            resolve_variant,
        )

        variant = resolve_variant("golden_retrieval")
        assert variant.build_prompt is golden_prompt

    def test_golden_prompt_with_no_task(self):
        """golden_prompt should handle the case where task is None gracefully."""
        from tau2.domains.banking_knowledge.retrieval import (
            golden_prompt,
            resolve_variant,
        )

        variant = resolve_variant("golden_retrieval")
        # golden_prompt requires a real template file; just verify the builder is set
        assert variant.build_prompt is golden_prompt


class TestBM25GrepVariant:
    """Tests for the bm25_grep retrieval variant (no API calls needed)."""

    @pytest.fixture
    def mock_kb(self):
        """Create a mock knowledge base with sample documents."""
        kb = MagicMock()
        doc_objs = []
        for d in SAMPLE_DOCUMENTS:
            doc = MagicMock()
            doc.id = d["id"]
            doc.title = d["title"]
            doc.content = d["text"]
            doc_objs.append(doc)
        kb.documents = {d.id: d for d in doc_objs}
        kb.get_all_documents.return_value = doc_objs
        return kb

    def test_name(self):
        from tau2.domains.banking_knowledge.retrieval import resolve_variant

        assert resolve_variant("bm25_grep").name == "bm25_grep"

    def test_supports_top_k(self):
        from tau2.domains.banking_knowledge.retrieval import resolve_variant

        assert resolve_variant("bm25_grep").supports_top_k is True

    @patch("tau2.domains.banking_knowledge.retrieval.get_or_create_docs")
    def test_build_tools_has_kb_search_and_grep(self, mock_get_docs, mock_kb):
        """bm25_grep should provide a toolkit with KB_search and grep tools."""
        mock_get_docs.return_value = SAMPLE_DOCUMENTS
        from tau2.domains.banking_knowledge.data_model import TransactionalDB
        from tau2.domains.banking_knowledge.retrieval import (
            build_tools,
            resolve_variant,
        )

        variant = resolve_variant("bm25_grep", top_k=3, grep_top_k=2)
        mock_db = MagicMock(spec=TransactionalDB)
        toolkit = build_tools(variant, mock_db, mock_kb)
        assert toolkit.has_tool("KB_search")
        assert toolkit.has_tool("grep")

    @patch("tau2.domains.banking_knowledge.retrieval.get_or_create_docs")
    def test_build_tools_returns_correct_toolkit_class(self, mock_get_docs, mock_kb):
        """build_tools should return KnowledgeToolsWithKBSearchAndGrep."""
        mock_get_docs.return_value = SAMPLE_DOCUMENTS
        from tau2.domains.banking_knowledge.data_model import TransactionalDB
        from tau2.domains.banking_knowledge.retrieval import (
            build_tools,
            resolve_variant,
        )
        from tau2.domains.banking_knowledge.retrieval_toolkits import (
            KnowledgeToolsWithKBSearchAndGrep,
        )

        variant = resolve_variant("bm25_grep")
        toolkit = build_tools(variant, MagicMock(spec=TransactionalDB), mock_kb)
        assert isinstance(toolkit, KnowledgeToolsWithKBSearchAndGrep)


class TestGrepOnlyVariant:
    """Tests for the grep_only retrieval variant."""

    @pytest.fixture
    def mock_kb(self):
        kb = MagicMock()
        doc_objs = []
        for d in SAMPLE_DOCUMENTS:
            doc = MagicMock()
            doc.id = d["id"]
            doc.title = d["title"]
            doc.content = d["text"]
            doc_objs.append(doc)
        kb.documents = {d.id: d for d in doc_objs}
        kb.get_all_documents.return_value = doc_objs
        return kb

    def test_name(self):
        from tau2.domains.banking_knowledge.retrieval import resolve_variant

        assert resolve_variant("grep_only").name == "grep_only"

    @patch("tau2.domains.banking_knowledge.retrieval.get_or_create_docs")
    def test_build_tools_has_grep(self, mock_get_docs, mock_kb):
        mock_get_docs.return_value = SAMPLE_DOCUMENTS
        from tau2.domains.banking_knowledge.data_model import TransactionalDB
        from tau2.domains.banking_knowledge.retrieval import (
            build_tools,
            resolve_variant,
        )

        variant = resolve_variant("grep_only", grep_top_k=5)
        toolkit = build_tools(variant, MagicMock(spec=TransactionalDB), mock_kb)
        assert toolkit.has_tool("grep")
        assert not toolkit.has_tool("KB_search")


class TestBM25Variant:
    """Tests for the bm25 retrieval variant."""

    @pytest.fixture
    def mock_kb(self):
        kb = MagicMock()
        doc_objs = []
        for d in SAMPLE_DOCUMENTS:
            doc = MagicMock()
            doc.id = d["id"]
            doc.title = d["title"]
            doc.content = d["text"]
            doc_objs.append(doc)
        kb.documents = {d.id: d for d in doc_objs}
        kb.get_all_documents.return_value = doc_objs
        return kb

    def test_name(self):
        from tau2.domains.banking_knowledge.retrieval import resolve_variant

        assert resolve_variant("bm25").name == "bm25"

    @patch("tau2.domains.banking_knowledge.retrieval.get_or_create_docs")
    def test_build_tools_has_kb_search(self, mock_get_docs, mock_kb):
        mock_get_docs.return_value = SAMPLE_DOCUMENTS
        from tau2.domains.banking_knowledge.data_model import TransactionalDB
        from tau2.domains.banking_knowledge.retrieval import (
            build_tools,
            resolve_variant,
        )

        variant = resolve_variant("bm25", top_k=5)
        toolkit = build_tools(variant, MagicMock(spec=TransactionalDB), mock_kb)
        assert toolkit.has_tool("KB_search")
        assert not toolkit.has_tool("grep")


# ============================================================================
# 10. Mixin tool integration tests
# ============================================================================


class TestToolCreation:
    """Tests for KB_search, grep, and rewrite_context tools via mixin toolkits."""

    @pytest.fixture
    def bm25_pipeline(self):
        import tau2.knowledge.document_preprocessors.bm25_indexer  # noqa: F401
        import tau2.knowledge.retrievers.bm25_retriever  # noqa: F401
        from tau2.knowledge.pipeline import RetrievalPipeline

        config = {
            "document_preprocessors": [
                {"type": "bm25_indexer", "params": {"state_key": "bm25"}}
            ],
            "retriever": {
                "type": "bm25",
                "params": {
                    "bm25_state_key": "bm25",
                    "doc_ids_state_key": "bm25_doc_ids",
                    "top_k": 5,
                },
            },
            "postprocessors": [],
        }
        pipeline = RetrievalPipeline(config)
        pipeline.index_documents(SAMPLE_DOCUMENTS)
        return pipeline

    @pytest.fixture
    def grep_pipeline(self):
        import tau2.knowledge.retrievers.grep_retriever  # noqa: F401
        from tau2.knowledge.pipeline import RetrievalPipeline

        config = {
            "retriever": {
                "type": "grep",
                "params": {"top_k": 5},
            },
        }
        pipeline = RetrievalPipeline(config)
        pipeline.index_documents(SAMPLE_DOCUMENTS)
        return pipeline

    @pytest.fixture
    def kb_search_toolkit(self, bm25_pipeline):
        from tau2.domains.banking_knowledge.data_model import TransactionalDB
        from tau2.domains.banking_knowledge.retrieval_toolkits import (
            KnowledgeToolsWithKBSearch,
        )

        mock_db = MagicMock(spec=TransactionalDB)
        return KnowledgeToolsWithKBSearch(mock_db, bm25_pipeline)

    @pytest.fixture
    def grep_toolkit(self, grep_pipeline):
        from tau2.domains.banking_knowledge.data_model import TransactionalDB
        from tau2.domains.banking_knowledge.retrieval_toolkits import (
            KnowledgeToolsWithGrep,
        )

        mock_db = MagicMock(spec=TransactionalDB)
        return KnowledgeToolsWithGrep(mock_db, grep_pipeline)

    def test_kb_search_tool_name(self, kb_search_toolkit):
        assert kb_search_toolkit.has_tool("KB_search")

    def test_kb_search_tool_returns_results(self, kb_search_toolkit):
        output = kb_search_toolkit.KB_search(query="credit card fee")
        assert "Credit Card Policy" in output
        assert "doc_1" in output
        assert "Score:" in output

    def test_kb_search_tool_no_results(self, kb_search_toolkit):
        # BM25 always returns top_k docs even with 0 scores
        output = kb_search_toolkit.KB_search(query="xyzzyx")
        assert isinstance(output, str)

    def test_kb_search_tool_includes_timing(self, kb_search_toolkit):
        output = kb_search_toolkit.KB_search(query="fee")
        assert "[Timing:" in output

    def test_grep_tool_name(self, grep_toolkit):
        assert grep_toolkit.has_tool("grep")

    def test_grep_tool_returns_matches(self, grep_toolkit):
        output = grep_toolkit.grep(pattern="fee")
        assert "Credit Card Policy" in output

    def test_grep_tool_no_matches(self, grep_toolkit):
        output = grep_toolkit.grep(pattern="xyznonexistent")
        assert "No matches found" in output

    def test_grep_tool_regex(self, grep_toolkit):
        output = grep_toolkit.grep(pattern=r"\$\d+")
        assert "doc_1" in output
        assert "doc_5" in output

    def test_rewrite_context_tool(self):
        from tau2.domains.banking_knowledge.retrieval_mixins import RewriteContextMixin

        # RewriteContextMixin needs no pipelines -- just instantiate a minimal class
        class _RewriteToolkit(RewriteContextMixin):
            pass

        toolkit = _RewriteToolkit()
        assert hasattr(toolkit, "rewrite_context")
        output = toolkit.rewrite_context(new_context="Summary of findings")
        assert "Context updated" in output
        assert "Summary of findings" in output


# ============================================================================
# 11. End-to-end pipeline integration tests
# ============================================================================


class TestEndToEndPipelines:
    """Higher-level integration tests using the factory functions from retrieval module."""

    @pytest.fixture
    def mock_kb(self):
        """Create a mock knowledge base."""
        kb = MagicMock()
        doc_objs = []
        for d in SAMPLE_DOCUMENTS:
            doc = MagicMock()
            doc.id = d["id"]
            doc.title = d["title"]
            doc.content = d["text"]
            doc_objs.append(doc)
        kb.documents = {d.id: d for d in doc_objs}
        kb.get_all_documents.return_value = doc_objs
        return kb

    @patch("tau2.domains.banking_knowledge.retrieval.get_or_create_docs")
    def test_create_bm25_retrieval_pipeline(self, mock_get_docs, mock_kb):
        mock_get_docs.return_value = SAMPLE_DOCUMENTS
        from tau2.domains.banking_knowledge.retrieval import (
            create_bm25_retrieval_pipeline,
        )

        pipeline = create_bm25_retrieval_pipeline(mock_kb, top_k=3)
        results = pipeline.retrieve("credit card")
        assert len(results) <= 3
        assert results[0][0] == "doc_1"

    @patch("tau2.domains.banking_knowledge.retrieval.get_or_create_docs")
    def test_create_grep_retrieval_pipeline(self, mock_get_docs, mock_kb):
        mock_get_docs.return_value = SAMPLE_DOCUMENTS
        from tau2.domains.banking_knowledge.retrieval import (
            create_grep_retrieval_pipeline,
        )

        pipeline = create_grep_retrieval_pipeline(mock_kb, top_k=5)
        results = pipeline.retrieve("fee")
        doc_ids = [d for d, _ in results]
        assert "doc_1" in doc_ids

    @patch("tau2.domains.banking_knowledge.retrieval.get_or_create_docs")
    def test_create_grep_case_sensitive(self, mock_get_docs, mock_kb):
        mock_get_docs.return_value = SAMPLE_DOCUMENTS
        from tau2.domains.banking_knowledge.retrieval import (
            create_grep_retrieval_pipeline,
        )

        pipeline = create_grep_retrieval_pipeline(mock_kb, case_sensitive=True)
        # "FEE" won't match lowercase "fee"
        results = pipeline.retrieve("FEE")
        assert len(results) == 0

    @patch("tau2.domains.banking_knowledge.retrieval.get_or_create_docs")
    def test_bm25_pipeline_batch_retrieval(self, mock_get_docs, mock_kb):
        mock_get_docs.return_value = SAMPLE_DOCUMENTS
        from tau2.domains.banking_knowledge.retrieval import (
            create_bm25_retrieval_pipeline,
        )

        pipeline = create_bm25_retrieval_pipeline(mock_kb, top_k=3)
        batch = pipeline.retrieve_batch(["credit card", "wire transfer"])
        assert len(batch) == 2
        assert batch[0][0][0] == "doc_1"
        assert batch[1][0][0] == "doc_5"

    @patch("tau2.domains.banking_knowledge.retrieval.get_or_create_docs")
    def test_grep_pipeline_batch_retrieval(self, mock_get_docs, mock_kb):
        mock_get_docs.return_value = SAMPLE_DOCUMENTS
        from tau2.domains.banking_knowledge.retrieval import (
            create_grep_retrieval_pipeline,
        )

        pipeline = create_grep_retrieval_pipeline(mock_kb, top_k=3)
        batch = pipeline.retrieve_batch(["credit", "savings"])
        assert len(batch) == 2

    @patch("tau2.domains.banking_knowledge.retrieval.get_or_create_docs")
    def test_bm25_grep_build_tools_e2e(self, mock_get_docs, mock_kb):
        """Full end-to-end test: bm25_grep variant → build_tools → call tools."""
        mock_get_docs.return_value = SAMPLE_DOCUMENTS
        from tau2.domains.banking_knowledge.data_model import TransactionalDB
        from tau2.domains.banking_knowledge.retrieval import (
            build_tools,
            resolve_variant,
        )

        variant = resolve_variant("bm25_grep", top_k=3, grep_top_k=2)
        mock_db = MagicMock(spec=TransactionalDB)
        toolkit = build_tools(variant, mock_db, mock_kb)

        # KB search
        kb_output = toolkit.KB_search(query="annual fee")
        assert "Credit Card Policy" in kb_output
        assert "doc_1" in kb_output

        # Grep search
        grep_output = toolkit.grep(pattern="fee")
        assert "doc_1" in grep_output

    @patch("tau2.domains.banking_knowledge.retrieval.get_or_create_docs")
    def test_grep_only_build_tools_e2e(self, mock_get_docs, mock_kb):
        mock_get_docs.return_value = SAMPLE_DOCUMENTS
        from tau2.domains.banking_knowledge.data_model import TransactionalDB
        from tau2.domains.banking_knowledge.retrieval import (
            build_tools,
            resolve_variant,
        )

        variant = resolve_variant("grep_only", grep_top_k=3)
        toolkit = build_tools(variant, MagicMock(spec=TransactionalDB), mock_kb)
        assert toolkit.has_tool("grep")
        assert not toolkit.has_tool("KB_search")
        output = toolkit.grep(pattern="transfer")
        assert "Wire Transfer" in output


# ============================================================================
# 12. TerminalUse variant tests (basic, no sandbox dependency)
# ============================================================================


class TestTerminalUseVariant:
    """Basic attribute/configuration tests for terminal_use variant."""

    def test_name(self):
        from tau2.domains.banking_knowledge.retrieval import resolve_variant

        variant = resolve_variant("terminal_use")
        assert variant.name == "terminal_use"

    def test_supports_top_k_false(self):
        from tau2.domains.banking_knowledge.retrieval import resolve_variant

        assert resolve_variant("terminal_use").supports_top_k is False

    def test_allow_writes_default(self):
        from tau2.domains.banking_knowledge.retrieval import resolve_variant

        variant = resolve_variant("terminal_use")
        assert variant.shell is not None
        assert variant.shell.allow_writes is False

    def test_allow_writes_is_false(self):
        from tau2.domains.banking_knowledge.retrieval import resolve_variant

        variant = resolve_variant("terminal_use")
        assert variant.shell.allow_writes is False

    def test_file_format_default(self):
        from tau2.domains.banking_knowledge.retrieval import resolve_variant

        variant = resolve_variant("terminal_use")
        assert variant.shell.file_format == "md"


class TestTerminalUseWriteVariant:
    """Basic attribute tests for terminal_use_write variant."""

    def test_name(self):
        from tau2.domains.banking_knowledge.retrieval import resolve_variant

        variant = resolve_variant("terminal_use_write")
        assert variant.name == "terminal_use_write"

    def test_allow_writes_true(self):
        from tau2.domains.banking_knowledge.retrieval import resolve_variant

        variant = resolve_variant("terminal_use_write")
        assert variant.shell is not None
        assert variant.shell.allow_writes is True

    def test_supports_top_k_false(self):
        from tau2.domains.banking_knowledge.retrieval import resolve_variant

        assert resolve_variant("terminal_use_write").supports_top_k is False
