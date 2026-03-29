"""End-to-end retrieval config tests.

Exercises the full path: variant name -> resolve_variant -> build_tools ->
tool invocation -> output, with a hard-coded 5-doc corpus.

BM25/grep variants run offline.  Embedding variants hit real APIs and are
gated by OPENAI_API_KEY / OPENROUTER_API_KEY env vars.
"""

from __future__ import annotations

import os
import shutil
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from loguru import logger

logger.disable("tau2")

requires_openai = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
requires_openrouter = pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set",
)
requires_sandbox_runtime = pytest.mark.skipif(
    shutil.which("srt") is None,
    reason="sandbox-runtime (srt) is not installed",
)

DOCUMENTS: List[Dict[str, Any]] = [
    {
        "id": "doc_mortgage",
        "title": "Mortgage Lending Policy",
        "text": (
            "Fixed-rate mortgages are available at 6.5% APR for 30-year terms. "
            "Applicants must have a credit score above 680. "
            "Down payment minimum is 10% of property value. "
            "Mortgage insurance is required for down payments below 20%."
        ),
    },
    {
        "id": "doc_fraud",
        "title": "Fraud Prevention Guidelines",
        "text": (
            "Suspicious transactions exceeding $5,000 trigger automatic review. "
            "Two-factor authentication is mandatory for international wire transfers. "
            "Compromised cards are frozen within 60 seconds of detection. "
            "Fraud claims must be filed within 30 days of the statement date."
        ),
    },
    {
        "id": "doc_savings",
        "title": "Savings Account Terms",
        "text": (
            "High-yield savings accounts earn 4.25% APY compounded daily. "
            "Minimum opening deposit is $100. "
            "Six free withdrawals per month; $10 fee per additional withdrawal. "
            "Interest is credited on the first business day of each month."
        ),
    },
    {
        "id": "doc_business",
        "title": "Business Banking Services",
        "text": (
            "Business checking accounts have no monthly maintenance fee for balances above $25,000. "
            "Payroll processing is available for businesses with 5 or more employees. "
            "Merchant services include point-of-sale terminals and online payment gateways. "
            "Business loans up to $500,000 with competitive interest rates."
        ),
    },
    {
        "id": "doc_credit_card",
        "title": "Credit Card Rewards Program",
        "text": (
            "Platinum cardholders earn 3x points on travel and dining purchases. "
            "Points never expire for active accounts. "
            "Redeem 25,000 points for a $250 statement credit. "
            "Annual fee of $95 is waived for the first year."
        ),
    },
]


def _make_mock_kb():
    kb = MagicMock()
    doc_objs = []
    for d in DOCUMENTS:
        doc = MagicMock()
        doc.id = d["id"]
        doc.title = d["title"]
        doc.content = d["text"]
        doc_objs.append(doc)
    kb.documents = {obj.id: obj for obj in doc_objs}
    kb.get_all_documents.return_value = doc_objs
    return kb


# (variant_name, expected_tools, api_gate)
_ALL_VARIANTS = [
    ("no_knowledge", set(), None),
    ("full_kb", set(), None),
    ("golden_retrieval", set(), None),
    ("bm25", {"KB_search"}, None),
    ("bm25_reranker", {"KB_search"}, None),
    ("bm25_grep", {"KB_search", "grep"}, None),
    ("bm25_reranker_grep", {"KB_search", "grep"}, None),
    ("grep_only", {"grep"}, None),
    ("terminal_use", {"shell"}, "sandbox_runtime"),
    ("terminal_use_write", {"shell"}, "sandbox_runtime"),
    ("qwen_embeddings", {"KB_search"}, "openrouter"),
    ("qwen_embeddings_reranker", {"KB_search"}, "openrouter"),
    ("qwen_embeddings_grep", {"KB_search", "grep"}, "openrouter"),
    ("qwen_embeddings_reranker_grep", {"KB_search", "grep"}, "openrouter"),
    ("openai_embeddings", {"KB_search"}, "openai"),
    ("openai_embeddings_reranker", {"KB_search"}, "openai"),
    ("openai_embeddings_grep", {"KB_search", "grep"}, "openai"),
    ("openai_embeddings_reranker_grep", {"KB_search", "grep"}, "openai"),
]


def _api_mark(gate):
    if gate == "openrouter":
        return requires_openrouter
    if gate == "openai":
        return requires_openai
    if gate == "sandbox_runtime":
        return requires_sandbox_runtime
    return pytest.mark.skipif(False, reason="")


def _build_toolkit(variant_name: str, **kwargs):
    """Build a toolkit from DOCUMENTS with rerankers disabled."""
    with patch(
        "tau2.domains.banking_knowledge.retrieval.get_or_create_docs",
        return_value=DOCUMENTS,
    ):
        from tau2.domains.banking_knowledge.data_model import TransactionalDB
        from tau2.domains.banking_knowledge.retrieval import (
            build_tools,
            resolve_variant,
        )

        variant = resolve_variant(variant_name, **kwargs)
        if variant.kb_search and variant.kb_search.reranker:
            variant.kb_search.reranker = False
        return build_tools(variant, MagicMock(spec=TransactionalDB), _make_mock_kb())


def _clear_query_embedding_cache() -> None:
    from tau2.knowledge import embeddings_cache as _mod

    _mod._query_embeddings_cache.clear()


def _recall(pipeline, knowledge_base, top_k, use_content=False, max_tasks=10):
    """Shared recall computation for embedding pipeline tests."""
    from tau2.domains.banking_knowledge.environment import get_tasks

    tasks = get_tasks()
    tasks_with_docs = [t for t in tasks if t.required_documents][:max_tasks]
    found = total = 0
    for task in tasks_with_docs:
        for doc_ref in task.required_documents[:2]:
            doc = knowledge_base.get_document(doc_ref)
            if doc is None:
                continue
            total += 1
            if use_content:
                query = " ".join(doc.content.split()[:20])
            else:
                query = doc.title
            results = pipeline.retrieve(query, top_k=top_k)
            if doc_ref in {r[0] for r in results}:
                found += 1
    return found / total if total else 0


class TestAllVariantsToolPresence:
    """Every registered variant must expose exactly the right tools."""

    def test_all_variants_table_matches_registry(self):
        from tau2.domains.banking_knowledge.retrieval import RETRIEVAL_VARIANTS

        tested = {v for v, _, _ in _ALL_VARIANTS}
        registered = set(RETRIEVAL_VARIANTS.keys())
        assert tested == registered, (
            f"_ALL_VARIANTS out of sync with RETRIEVAL_VARIANTS. "
            f"Missing: {registered - tested}, Extra: {tested - registered}"
        )

    @pytest.mark.parametrize(
        "variant_name, expected_tools, gate",
        [pytest.param(v, t, g, marks=_api_mark(g)) for v, t, g in _ALL_VARIANTS],
        ids=[v for v, _, _ in _ALL_VARIANTS],
    )
    def test_has_exactly_expected_tools(self, variant_name, expected_tools, gate):
        toolkit = _build_toolkit(variant_name, top_k=5, grep_top_k=5)
        retrieval_tools = {"KB_search", "grep", "shell"}
        for tool in expected_tools:
            assert toolkit.has_tool(tool), f"{variant_name}: missing {tool}"
        for tool in retrieval_tools - expected_tools:
            assert not toolkit.has_tool(tool), f"{variant_name}: unexpected {tool}"


class TestAllVariantsToolInvocation:
    """Invoke each tool on the 5-doc corpus and verify structured output."""

    _KB_SEARCH_VARIANTS = [
        pytest.param(v, marks=_api_mark(g))
        for v, t, g in _ALL_VARIANTS
        if "KB_search" in t
    ]
    _GREP_VARIANTS = [
        pytest.param(v, marks=_api_mark(g)) for v, t, g in _ALL_VARIANTS if "grep" in t
    ]
    _SHELL_VARIANTS = [
        pytest.param(v, marks=_api_mark(g)) for v, t, g in _ALL_VARIANTS if "shell" in t
    ]

    @pytest.mark.parametrize("variant_name", _KB_SEARCH_VARIANTS)
    def test_kb_search_returns_correct_top_doc(self, variant_name):
        toolkit = _build_toolkit(variant_name, top_k=5)
        output = toolkit.KB_search(query="mortgage lending credit score down payment")
        assert "Mortgage Lending Policy" in output.split("\n")[0]
        assert "Score:" in output

    @pytest.mark.parametrize("variant_name", _KB_SEARCH_VARIANTS)
    def test_kb_search_returns_all_docs_at_top5(self, variant_name):
        toolkit = _build_toolkit(variant_name, top_k=5)
        output = toolkit.KB_search(query="account fee interest")
        for doc in DOCUMENTS:
            assert doc["id"] in output, f"{variant_name}: missing {doc['id']}"

    @pytest.mark.parametrize("variant_name", _GREP_VARIANTS)
    def test_grep_exact_match(self, variant_name):
        toolkit = _build_toolkit(variant_name, grep_top_k=5)
        output = toolkit.grep(pattern="two-factor authentication")
        assert "Fraud Prevention Guidelines" in output
        assert "doc_fraud" in output

    @pytest.mark.parametrize("variant_name", _GREP_VARIANTS)
    def test_grep_regex(self, variant_name):
        toolkit = _build_toolkit(variant_name, grep_top_k=5)
        output = toolkit.grep(pattern=r"\d+\.\d+%")
        assert "doc_mortgage" in output
        assert "doc_savings" in output

    @pytest.mark.parametrize("variant_name", _GREP_VARIANTS)
    def test_grep_no_match(self, variant_name):
        toolkit = _build_toolkit(variant_name, grep_top_k=5)
        output = toolkit.grep(pattern="xyznonexistent999")
        assert "No matches found" in output

    @pytest.mark.parametrize("variant_name", _SHELL_VARIANTS)
    def test_shell_ls(self, variant_name):
        toolkit = _build_toolkit(variant_name)
        assert "INDEX" in toolkit.shell(command="ls")

    @pytest.mark.parametrize("variant_name", _SHELL_VARIANTS)
    def test_shell_grep(self, variant_name):
        toolkit = _build_toolkit(variant_name)
        assert "doc_mortgage" in toolkit.shell(command="grep -l mortgage *")

    @pytest.mark.parametrize("variant_name", _SHELL_VARIANTS)
    def test_shell_cat_reads_doc(self, variant_name):
        toolkit = _build_toolkit(variant_name)
        output = toolkit.shell(command="cat doc_mortgage.md")
        assert "Mortgage" in output or "mortgage" in output


class TestKBSearchAndGrepAgreement:
    """For dual-tool variants, both tools should surface the same doc."""

    _DUAL_TOOL_VARIANTS = [
        pytest.param(v, marks=_api_mark(g))
        for v, t, g in _ALL_VARIANTS
        if "KB_search" in t and "grep" in t
    ]

    @pytest.mark.parametrize("variant_name", _DUAL_TOOL_VARIANTS)
    def test_both_tools_find_fraud_doc(self, variant_name):
        toolkit = _build_toolkit(variant_name, top_k=5, grep_top_k=5)
        kb_out = toolkit.KB_search(query="fraud suspicious transactions frozen")
        grep_out = toolkit.grep(pattern="fraud")
        assert "Fraud Prevention Guidelines" in kb_out
        assert "Fraud Prevention Guidelines" in grep_out


class TestLiveEmbeddingModelUsage:
    """Verify embedding variants actually call the live API (not just cache)."""

    @requires_openrouter
    def test_qwen_variant_calls_openrouter_for_docs_and_query(self, tmp_path):
        from tau2.knowledge.document_preprocessors import embedding_indexer
        from tau2.knowledge.embedders.openrouter_embedder import OpenRouterEmbedder
        from tau2.knowledge.embeddings_cache import EmbeddingsCache

        _clear_query_embedding_cache()
        isolated_cache = EmbeddingsCache(cache_dir=str(tmp_path / "qwen_cache"))
        calls: list[tuple[str, int]] = []
        original_embed = OpenRouterEmbedder.embed

        def _spy(self, texts, max_retries=3):
            calls.append((self.model_api_string, len(texts)))
            return original_embed(self, texts, max_retries=max_retries)

        with (
            patch.object(OpenRouterEmbedder, "embed", new=_spy),
            patch.object(
                embedding_indexer, "get_embeddings_cache", return_value=isolated_cache
            ),
        ):
            toolkit = _build_toolkit("qwen_embeddings", top_k=5)
            toolkit.KB_search(query=f"mortgage lending {os.urandom(8).hex()}")

        assert any(
            m == "qwen/qwen3-embedding-8b" and n == len(DOCUMENTS) for m, n in calls
        )
        assert any(m == "qwen/qwen3-embedding-8b" and n == 1 for m, n in calls)

    @requires_openai
    def test_openai_variant_calls_openai_for_docs_and_query(self, tmp_path):
        from tau2.knowledge.document_preprocessors import embedding_indexer
        from tau2.knowledge.embedders.openai_embedder import OpenAIEmbedder
        from tau2.knowledge.embeddings_cache import EmbeddingsCache

        _clear_query_embedding_cache()
        isolated_cache = EmbeddingsCache(cache_dir=str(tmp_path / "openai_cache"))
        calls: list[tuple[str, int]] = []
        original_embed = OpenAIEmbedder.embed

        def _spy(self, texts):
            calls.append((self.model, len(texts)))
            return original_embed(self, texts)

        with (
            patch.object(OpenAIEmbedder, "embed", new=_spy),
            patch.object(
                embedding_indexer, "get_embeddings_cache", return_value=isolated_cache
            ),
        ):
            toolkit = _build_toolkit("openai_embeddings", top_k=5)
            toolkit.KB_search(query=f"credit card rewards {os.urandom(8).hex()}")

        assert any(
            m == "text-embedding-3-large" and n == len(DOCUMENTS) for m, n in calls
        )
        assert any(m == "text-embedding-3-large" and n == 1 for m, n in calls)


# ============================================================================
# Production KB tests
# ============================================================================


class TestProductionKBIntegrity:
    """Verify the real knowledge base loads correctly."""

    @pytest.fixture(scope="class")
    def knowledge_base(self):
        from tau2.domains.banking_knowledge.environment import get_knowledge_base

        return get_knowledge_base()

    @pytest.fixture(scope="class")
    def tasks(self):
        from tau2.domains.banking_knowledge.environment import get_tasks

        return get_tasks()

    def test_kb_loads_nonempty(self, knowledge_base):
        assert len(knowledge_base.documents) > 0

    def test_every_document_has_id_title_content(self, knowledge_base):
        for doc_id, doc in knowledge_base.documents.items():
            assert doc.id == doc_id
            assert doc.title and doc.title.strip()
            assert doc.content and doc.content.strip()

    def test_document_ids_are_unique(self, knowledge_base):
        ids = knowledge_base.get_document_ids()
        assert len(ids) == len(set(ids))

    def test_every_task_required_doc_exists_in_kb(self, knowledge_base, tasks):
        missing = [
            (t.id, ref)
            for t in tasks
            for ref in (t.required_documents or [])
            if knowledge_base.get_document(ref) is None
        ]
        assert missing == [], f"Tasks reference docs not in KB: {missing}"


class TestPipelineDocumentFidelity:
    """Documents must roundtrip through the pipeline without corruption."""

    @pytest.fixture(scope="class")
    def knowledge_base(self):
        from tau2.domains.banking_knowledge.environment import get_knowledge_base

        return get_knowledge_base()

    @pytest.fixture(scope="class")
    def bm25_pipeline(self, knowledge_base):
        from tau2.domains.banking_knowledge.retrieval import (
            create_bm25_retrieval_pipeline,
        )

        return create_bm25_retrieval_pipeline(knowledge_base, top_k=10)

    def test_pipeline_indexes_all_documents(self, knowledge_base, bm25_pipeline):
        indexed = set(bm25_pipeline.state["doc_content_map"].keys())
        expected = set(knowledge_base.get_document_ids())
        assert indexed == expected

    def test_pipeline_content_matches_original(self, knowledge_base, bm25_pipeline):
        mismatches = [
            doc_id
            for doc_id, doc in knowledge_base.documents.items()
            if bm25_pipeline.get_document_content(doc_id) != doc.content
        ]
        assert mismatches == [], f"Content mismatch for: {mismatches}"

    def test_pipeline_titles_match_original(self, knowledge_base, bm25_pipeline):
        mismatches = [
            doc_id
            for doc_id, doc in knowledge_base.documents.items()
            if bm25_pipeline.get_document_title(doc_id) != doc.title
        ]
        assert mismatches == [], f"Title mismatch for: {mismatches}"


class TestResolveVariantIsolation:
    """resolve_variant must deep-copy so overrides never mutate the registry."""

    def test_top_k_override_does_not_mutate_registry(self):
        from tau2.domains.banking_knowledge.retrieval import (
            RETRIEVAL_VARIANTS,
            resolve_variant,
        )

        original = RETRIEVAL_VARIANTS["bm25"].kb_search.top_k
        resolve_variant("bm25", top_k=999)
        assert RETRIEVAL_VARIANTS["bm25"].kb_search.top_k == original

    def test_grep_top_k_override_does_not_mutate_registry(self):
        from tau2.domains.banking_knowledge.retrieval import (
            RETRIEVAL_VARIANTS,
            resolve_variant,
        )

        original = RETRIEVAL_VARIANTS["grep_only"].grep.top_k
        resolve_variant("grep_only", grep_top_k=999)
        assert RETRIEVAL_VARIANTS["grep_only"].grep.top_k == original


class TestPolicyTemplateIntegrity:
    """Every variant's prompt template must exist and render non-empty."""

    @pytest.fixture(scope="class")
    def knowledge_base(self):
        from tau2.domains.banking_knowledge.environment import get_knowledge_base

        return get_knowledge_base()

    @pytest.mark.parametrize(
        "variant_name",
        [
            "no_knowledge",
            "full_kb",
            "golden_retrieval",
            "bm25",
            "bm25_grep",
            "grep_only",
            "terminal_use",
            "terminal_use_write",
            "qwen_embeddings",
            "qwen_embeddings_grep",
            "openai_embeddings",
            "openai_embeddings_grep",
        ],
    )
    def test_policy_renders_nonempty(self, variant_name, knowledge_base):
        from tau2.domains.banking_knowledge.retrieval import (
            build_policy,
            resolve_variant,
        )

        policy = build_policy(resolve_variant(variant_name), knowledge_base)
        assert len(policy) > 100

    def test_full_kb_policy_contains_all_documents(self, knowledge_base):
        from tau2.domains.banking_knowledge.retrieval import (
            build_policy,
            resolve_variant,
        )

        policy = build_policy(resolve_variant("full_kb"), knowledge_base)
        missing = [
            doc.id
            for doc in knowledge_base.get_all_documents()
            if doc.title not in policy
        ]
        assert missing == [], f"full_kb policy missing docs: {missing[:5]}"

    def test_golden_retrieval_policy_inlines_required_docs(self, knowledge_base):
        from tau2.domains.banking_knowledge.environment import get_tasks
        from tau2.domains.banking_knowledge.retrieval import (
            build_policy,
            resolve_variant,
        )

        tasks_with_docs = [t for t in get_tasks() if t.required_documents]
        assert len(tasks_with_docs) > 0
        task = tasks_with_docs[0]

        policy = build_policy(resolve_variant("golden_retrieval"), knowledge_base, task)
        for doc_ref in task.required_documents:
            doc = knowledge_base.get_document(doc_ref)
            if doc:
                assert doc.title in policy


class TestQueryStateIsolation:
    """Successive queries must not corrupt pipeline state."""

    @pytest.fixture(scope="class")
    def bm25_pipeline(self):
        from tau2.domains.banking_knowledge.environment import get_knowledge_base
        from tau2.domains.banking_knowledge.retrieval import (
            create_bm25_retrieval_pipeline,
        )

        return create_bm25_retrieval_pipeline(get_knowledge_base(), top_k=5)

    def test_repeated_identical_queries_return_same_results(self, bm25_pipeline):
        r1 = bm25_pipeline.retrieve("credit card rewards")
        r2 = bm25_pipeline.retrieve("credit card rewards")
        assert r1 == r2

    def test_different_queries_do_not_interfere(self, bm25_pipeline):
        r1 = bm25_pipeline.retrieve("credit card rewards")
        bm25_pipeline.retrieve("savings account interest")
        r2 = bm25_pipeline.retrieve("credit card rewards")
        assert r1 == r2

    def test_batch_and_single_agree(self, bm25_pipeline):
        queries = ["credit card", "savings account"]
        batch = bm25_pipeline.retrieve_batch(queries, top_k=5)
        assert batch[0] == bm25_pipeline.retrieve(queries[0], top_k=5)
        assert batch[1] == bm25_pipeline.retrieve(queries[1], top_k=5)


class TestRequiredDocumentRetrievability:
    """Can the retrieval pipeline find the documents each task needs?"""

    @pytest.fixture(scope="class")
    def knowledge_base(self):
        from tau2.domains.banking_knowledge.environment import get_knowledge_base

        return get_knowledge_base()

    @pytest.fixture(scope="class")
    def tasks(self):
        from tau2.domains.banking_knowledge.environment import get_tasks

        return get_tasks()

    @pytest.fixture(scope="class")
    def grep_pipeline(self, knowledge_base):
        from tau2.domains.banking_knowledge.retrieval import (
            create_grep_retrieval_pipeline,
        )

        return create_grep_retrieval_pipeline(knowledge_base, top_k=5)

    @pytest.fixture(scope="class")
    def bm25_pipeline(self, knowledge_base):
        from tau2.domains.banking_knowledge.retrieval import (
            create_bm25_retrieval_pipeline,
        )

        return create_bm25_retrieval_pipeline(knowledge_base, top_k=50)

    def test_every_required_doc_content_is_in_pipeline(
        self, tasks, knowledge_base, grep_pipeline
    ):
        all_required = {ref for t in tasks for ref in (t.required_documents or [])}
        content_map = grep_pipeline.state.get("doc_content_map", {})
        missing, corrupted = [], []
        for doc_ref in sorted(all_required):
            doc = knowledge_base.get_document(doc_ref)
            if doc is None:
                continue
            if doc_ref not in content_map:
                missing.append(doc_ref)
            elif content_map[doc_ref] != doc.content:
                corrupted.append(doc_ref)

        assert missing == [], f"Missing from pipeline: {missing[:10]}"
        assert corrupted == [], f"Corrupted content: {corrupted[:10]}"

    def test_bm25_retrieves_most_required_docs_by_title(
        self, tasks, knowledge_base, bm25_pipeline
    ):
        """BM25 title recall >= 80% (many titles are shared across docs)."""
        all_required = {ref for t in tasks for ref in (t.required_documents or [])}
        found = sum(
            1
            for ref in all_required
            if (doc := knowledge_base.get_document(ref)) is not None
            and ref in {r[0] for r in bm25_pipeline.retrieve(doc.title, top_k=50)}
        )
        recall = found / len(all_required) if all_required else 0
        assert recall >= 0.80, f"BM25 title recall: {recall:.0%} (threshold: 80%)"


class TestGetEnvironmentLivePath:
    """Test the exact function the live path calls: get_environment()."""

    def test_default_variant_produces_valid_environment(self):
        import warnings

        from tau2.domains.banking_knowledge.environment import get_environment

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            env = get_environment()
            assert any("--retrieval-config" in str(m.message) for m in w)
        assert env.policy and len(env.policy) > 100
        assert env.tools is not None
        assert env.user_tools is not None

    @pytest.mark.parametrize(
        "variant_name",
        ["no_knowledge", "full_kb", "bm25", "bm25_grep", "grep_only"],
    )
    def test_get_environment_succeeds_for_offline_variants(self, variant_name):
        from tau2.domains.banking_knowledge.environment import get_environment

        env = get_environment(retrieval_variant=variant_name)
        assert env.policy and len(env.policy) > 100

    @pytest.mark.parametrize(
        "variant_name",
        [
            pytest.param("qwen_embeddings", marks=requires_openrouter),
            pytest.param("openai_embeddings", marks=requires_openai),
            pytest.param("qwen_embeddings_grep", marks=requires_openrouter),
            pytest.param("openai_embeddings_grep", marks=requires_openai),
        ],
    )
    def test_get_environment_succeeds_for_embedding_variants(self, variant_name):
        from tau2.domains.banking_knowledge.environment import get_environment

        env = get_environment(retrieval_variant=variant_name)
        assert env.policy and len(env.policy) > 100

    def test_solo_mode_raises(self):
        from tau2.domains.banking_knowledge.environment import get_environment

        with pytest.raises(ValueError, match="solo mode"):
            get_environment(solo_mode=True)


# ============================================================================
# Rigorous embedding pipeline tests — real KB, real APIs
# ============================================================================


def _build_real_kb_embedding_pipeline(embedder_type: str, embedder_params: dict):
    from tau2.domains.banking_knowledge.environment import get_knowledge_base
    from tau2.domains.banking_knowledge.retrieval import (
        create_embedding_retrieval_pipeline,
    )

    kb = get_knowledge_base()
    pipeline = create_embedding_retrieval_pipeline(
        kb, embedder_type=embedder_type, embedder_params=embedder_params, top_k=10
    )
    return pipeline, kb


@requires_openrouter
class TestQwenEmbeddingPipelineRigorous:
    """Qwen/OpenRouter embedding pipeline against the real production KB."""

    @pytest.fixture(scope="class")
    def pipeline_and_kb(self):
        return _build_real_kb_embedding_pipeline(
            "openrouter", {"model": "qwen3-embedding-8b"}
        )

    @pytest.fixture(scope="class")
    def pipeline(self, pipeline_and_kb):
        return pipeline_and_kb[0]

    @pytest.fixture(scope="class")
    def knowledge_base(self, pipeline_and_kb):
        return pipeline_and_kb[1]

    def test_doc_embeddings_have_expected_dimension(self, pipeline):
        assert pipeline.state["doc_embeddings"].shape[1] == 4096

    def test_query_embedding_matches_doc_dimension(self, pipeline):
        from tau2.knowledge.pipeline import RetrievalResult

        result = pipeline.retrieve("test query", return_timing=True)
        assert isinstance(result, RetrievalResult)
        assert len(result.results) > 0

    def test_scores_in_valid_range(self, pipeline):
        for _, score in pipeline.retrieve("savings account interest rate"):
            assert -1.0 <= score <= 1.0

    def test_top_score_is_meaningful(self, pipeline):
        results = pipeline.retrieve("savings account interest rate")
        assert results[0][1] > 0.3

    def test_self_retrieval_by_content(self, pipeline, knowledge_base):
        for doc in list(knowledge_base.documents.values())[:5]:
            snippet = " ".join(doc.content.split()[:30])
            top_ids = [r[0] for r in pipeline.retrieve(snippet, top_k=3)]
            assert doc.id in top_ids, f"{doc.id!r} not in top-3: {top_ids}"

    def test_document_embedder_has_no_instruction_prefix(self, pipeline):
        embedder = pipeline.doc_preprocessors[0]._get_embedder()
        assert embedder.query_instruction == ""

    _EXPECTED_QWEN_INSTRUCTION = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )

    def test_query_encoder_has_correct_instruction_prefix(self, pipeline):
        """Hardcoded (not imported) so changes to the source constant are caught."""
        embedder = pipeline.input_preprocessors[0]._get_embedder()
        assert embedder.query_instruction == self._EXPECTED_QWEN_INSTRUCTION

    def test_required_docs_retrievable_by_title(self, pipeline, knowledge_base):
        assert _recall(pipeline, knowledge_base, top_k=20) >= 0.40

    def test_required_docs_retrievable_by_content(self, pipeline, knowledge_base):
        assert _recall(pipeline, knowledge_base, top_k=10, use_content=True) >= 0.60


@requires_openrouter
class TestQwenInstructionPrefixImpact:
    """Prove the instruction prefix materially affects embedding quality.

    Both instruction strings are hardcoded (not imported from the source module)
    so changes to DEFAULT_QWEN_QUERY_INSTRUCTION are caught.
    """

    CORRECT_INSTRUCTION = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )
    WRONG_INSTRUCTION = "Translate the following text into formal academic French prose"

    @pytest.fixture(scope="class")
    def doc_embedding(self):
        from tau2.knowledge.embedders.openrouter_embedder import OpenRouterEmbedder

        embedder = OpenRouterEmbedder(model="qwen3-embedding-8b", query_instruction="")
        return embedder.embed([DOCUMENTS[0]["text"]])[0]

    @pytest.fixture(scope="class")
    def correct_query_embedding(self):
        from tau2.knowledge.embedders.openrouter_embedder import OpenRouterEmbedder

        return OpenRouterEmbedder(
            model="qwen3-embedding-8b",
            query_instruction=TestQwenInstructionPrefixImpact.CORRECT_INSTRUCTION,
        ).embed(["mortgage lending credit score down payment"])[0]

    @pytest.fixture(scope="class")
    def wrong_query_embedding(self):
        from tau2.knowledge.embedders.openrouter_embedder import OpenRouterEmbedder

        return OpenRouterEmbedder(
            model="qwen3-embedding-8b",
            query_instruction=TestQwenInstructionPrefixImpact.WRONG_INSTRUCTION,
        ).embed(["mortgage lending credit score down payment"])[0]

    @staticmethod
    def _cosine_sim(a, b):
        import numpy as np

        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def test_correct_instruction_gives_higher_similarity_with_nontrivial_gap(
        self, doc_embedding, correct_query_embedding, wrong_query_embedding
    ):
        correct_sim = self._cosine_sim(doc_embedding, correct_query_embedding)
        wrong_sim = self._cosine_sim(doc_embedding, wrong_query_embedding)
        gap = correct_sim - wrong_sim
        assert gap > 0.01, (
            f"Correct: {correct_sim:.6f}, Wrong: {wrong_sim:.6f}, Gap: {gap:.6f}"
        )

    def test_correct_and_wrong_embeddings_differ(
        self, correct_query_embedding, wrong_query_embedding
    ):
        sim = self._cosine_sim(correct_query_embedding, wrong_query_embedding)
        assert sim < 0.98, f"cosine={sim:.6f} — model may be ignoring instruction"

    def test_source_constant_matches_expected(self):
        from tau2.knowledge.embedders.openrouter_embedder import (
            DEFAULT_QWEN_QUERY_INSTRUCTION,
        )

        assert DEFAULT_QWEN_QUERY_INSTRUCTION == self.CORRECT_INSTRUCTION, (
            f"DEFAULT_QWEN_QUERY_INSTRUCTION changed to "
            f"{DEFAULT_QWEN_QUERY_INSTRUCTION!r} — update this test if intentional"
        )


@requires_openai
class TestOpenAIEmbeddingPipelineRigorous:
    """OpenAI embedding pipeline against the real production KB."""

    @pytest.fixture(scope="class")
    def pipeline_and_kb(self):
        return _build_real_kb_embedding_pipeline(
            "openai", {"model": "text-embedding-3-large"}
        )

    @pytest.fixture(scope="class")
    def pipeline(self, pipeline_and_kb):
        return pipeline_and_kb[0]

    @pytest.fixture(scope="class")
    def knowledge_base(self, pipeline_and_kb):
        return pipeline_and_kb[1]

    def test_doc_embeddings_have_expected_dimension(self, pipeline):
        assert pipeline.state["doc_embeddings"].shape[1] == 3072

    def test_query_embedding_matches_doc_dimension(self, pipeline):
        from tau2.knowledge.pipeline import RetrievalResult

        result = pipeline.retrieve("test query", return_timing=True)
        assert isinstance(result, RetrievalResult)
        assert len(result.results) > 0

    def test_scores_in_valid_range(self, pipeline):
        for _, score in pipeline.retrieve("credit card rewards program"):
            assert -1.0 <= score <= 1.0

    def test_top_score_is_meaningful(self, pipeline):
        results = pipeline.retrieve("credit card rewards program")
        assert results[0][1] > 0.3

    def test_self_retrieval_by_content(self, pipeline, knowledge_base):
        for doc in list(knowledge_base.documents.values())[:5]:
            snippet = " ".join(doc.content.split()[:30])
            top_ids = [r[0] for r in pipeline.retrieve(snippet, top_k=3)]
            assert doc.id in top_ids, f"{doc.id!r} not in top-3: {top_ids}"

    def test_required_docs_retrievable_by_title(self, pipeline, knowledge_base):
        assert _recall(pipeline, knowledge_base, top_k=20) >= 0.40

    def test_required_docs_retrievable_by_content(self, pipeline, knowledge_base):
        assert _recall(pipeline, knowledge_base, top_k=10, use_content=True) >= 0.60


# ============================================================================
# Embedding cache correctness
# ============================================================================


class TestInMemoryDocsCacheCorrectness:
    def setup_method(self):
        from tau2.knowledge.embeddings_cache import clear_cached_docs

        clear_cached_docs()

    def teardown_method(self):
        from tau2.knowledge.embeddings_cache import clear_cached_docs

        clear_cached_docs()

    def test_initially_empty(self):
        from tau2.knowledge.embeddings_cache import get_cached_docs

        assert get_cached_docs() is None

    def test_set_then_get_returns_same(self):
        from tau2.knowledge.embeddings_cache import get_cached_docs, set_cached_docs

        docs = [{"id": "a", "text": "hello"}]
        set_cached_docs(docs)
        assert get_cached_docs() is docs

    def test_clear_resets_to_none(self):
        from tau2.knowledge.embeddings_cache import (
            clear_cached_docs,
            get_cached_docs,
            set_cached_docs,
        )

        set_cached_docs([{"id": "a", "text": "hello"}])
        clear_cached_docs()
        assert get_cached_docs() is None

    def test_overwrite_replaces_old(self):
        from tau2.knowledge.embeddings_cache import get_cached_docs, set_cached_docs

        set_cached_docs([{"id": "a", "text": "v1"}])
        docs_v2 = [{"id": "a", "text": "v2"}, {"id": "b", "text": "new"}]
        set_cached_docs(docs_v2)
        assert get_cached_docs() is docs_v2
        assert len(get_cached_docs()) == 2


class TestQueryEmbeddingCacheCorrectness:
    def setup_method(self):
        _clear_query_embedding_cache()

    def teardown_method(self):
        _clear_query_embedding_cache()

    def test_same_query_same_config_returns_cached(self):
        import numpy as np

        from tau2.knowledge.embeddings_cache import (
            cache_query_embedding,
            get_cached_query_embedding,
        )

        emb = np.array([1.0, 2.0, 3.0])
        cache_query_embedding("hello", emb, "openai", {"model": "test"})
        cached = get_cached_query_embedding("hello", "openai", {"model": "test"})
        assert cached is not None
        assert np.array_equal(cached, emb)

    def test_different_query_returns_none(self):
        import numpy as np

        from tau2.knowledge.embeddings_cache import (
            cache_query_embedding,
            get_cached_query_embedding,
        )

        cache_query_embedding("hello", np.array([1.0]), "openai", {"model": "test"})
        assert (
            get_cached_query_embedding("goodbye", "openai", {"model": "test"}) is None
        )

    def test_different_embedder_type_returns_none(self):
        import numpy as np

        from tau2.knowledge.embeddings_cache import (
            cache_query_embedding,
            get_cached_query_embedding,
        )

        cache_query_embedding("hello", np.array([1.0]), "openai", {"model": "test"})
        assert (
            get_cached_query_embedding("hello", "openrouter", {"model": "test"}) is None
        )

    def test_different_model_returns_none(self):
        import numpy as np

        from tau2.knowledge.embeddings_cache import (
            cache_query_embedding,
            get_cached_query_embedding,
        )

        cache_query_embedding("hello", np.array([1.0]), "openai", {"model": "ada"})
        assert get_cached_query_embedding("hello", "openai", {"model": "large"}) is None

    def test_config_key_order_independent(self):
        import numpy as np

        from tau2.knowledge.embeddings_cache import (
            cache_query_embedding,
            get_cached_query_embedding,
        )

        emb = np.array([1.0, 2.0, 3.0])
        cache_query_embedding("hello", emb, "openai", {"model": "m", "dim": 3})
        cached = get_cached_query_embedding("hello", "openai", {"dim": 3, "model": "m"})
        assert cached is not None
        assert np.array_equal(cached, emb)

    def test_different_instruction_prefix_is_separate_cache(self):
        import numpy as np

        from tau2.knowledge.embeddings_cache import (
            cache_query_embedding,
            get_cached_query_embedding,
        )

        config_a = {"model": "qwen3-embedding-8b", "_query_instruction": "retrieve"}
        config_b = {"model": "qwen3-embedding-8b", "_query_instruction": "translate"}
        cache_query_embedding("hello", np.array([1.0]), "openrouter", config_a)

        assert get_cached_query_embedding("hello", "openrouter", config_b) is None
        assert get_cached_query_embedding("hello", "openrouter", config_a) is not None


class TestEncoderCacheConfigIncludesInstruction:
    """EmbeddingEncoder must include instruction prefix in the cache config."""

    def test_openrouter_encoder_includes_instruction(self):
        from tau2.knowledge.input_preprocessors.embedding_encoder import (
            EmbeddingEncoder,
        )

        config = EmbeddingEncoder(
            embedder_type="openrouter",
            embedder_params={"model": "qwen3-embedding-8b"},
        )._get_cache_config()
        assert "_query_instruction" in config
        assert "retrieve" in config["_query_instruction"].lower()

    def test_openai_encoder_has_no_instruction(self):
        from tau2.knowledge.input_preprocessors.embedding_encoder import (
            EmbeddingEncoder,
        )

        config = EmbeddingEncoder(
            embedder_type="openai",
            embedder_params={"model": "text-embedding-3-large"},
        )._get_cache_config()
        assert "_query_instruction" not in config

    def test_different_instructions_produce_different_configs(self):
        from tau2.knowledge.input_preprocessors.embedding_encoder import (
            EmbeddingEncoder,
        )

        config_a = EmbeddingEncoder(
            embedder_type="openrouter",
            embedder_params={
                "model": "qwen3-embedding-8b",
                "query_instruction": "A",
            },
        )._get_cache_config()
        config_b = EmbeddingEncoder(
            embedder_type="openrouter",
            embedder_params={
                "model": "qwen3-embedding-8b",
                "query_instruction": "B",
            },
        )._get_cache_config()
        assert config_a != config_b


class TestDiskEmbeddingsCacheCorrectness:
    @pytest.fixture
    def cache(self, tmp_path):
        from tau2.knowledge.embeddings_cache import EmbeddingsCache

        return EmbeddingsCache(cache_dir=str(tmp_path / "test_cache"))

    @pytest.fixture
    def sample_docs(self):
        return [
            {"id": "d1", "text": "first document content"},
            {"id": "d2", "text": "second document content"},
        ]

    def test_put_then_get_roundtrip(self, cache, sample_docs):
        import numpy as np

        embeddings = np.array([[1.0, 2.0], [3.0, 4.0]])
        cache.put(sample_docs, "openai", embeddings, ["d1", "d2"], {"model": "test"})

        result = cache.get(sample_docs, "openai", {"model": "test"})
        assert result is not None
        assert np.array_equal(result[0], embeddings)
        assert result[1] == ["d1", "d2"]

    def test_cache_miss_returns_none(self, cache, sample_docs):
        assert cache.get(sample_docs, "openai", {"model": "test"}) is None

    def test_different_embedder_type_is_separate_cache(self, cache, sample_docs):
        import numpy as np

        emb_a = np.array([[1.0, 2.0], [3.0, 4.0]])
        emb_b = np.array([[5.0, 6.0], [7.0, 8.0]])
        cache.put(sample_docs, "openai", emb_a, ["d1", "d2"], {"model": "test"})
        cache.put(sample_docs, "openrouter", emb_b, ["d1", "d2"], {"model": "test"})

        assert np.array_equal(
            cache.get(sample_docs, "openai", {"model": "test"})[0], emb_a
        )
        assert np.array_equal(
            cache.get(sample_docs, "openrouter", {"model": "test"})[0], emb_b
        )

    def test_different_model_is_separate_cache(self, cache, sample_docs):
        import numpy as np

        emb_a = np.array([[1.0, 2.0], [3.0, 4.0]])
        emb_b = np.array([[5.0, 6.0], [7.0, 8.0]])
        cache.put(sample_docs, "openai", emb_a, ["d1", "d2"], {"model": "model-a"})
        cache.put(sample_docs, "openai", emb_b, ["d1", "d2"], {"model": "model-b"})

        assert np.array_equal(
            cache.get(sample_docs, "openai", {"model": "model-a"})[0], emb_a
        )
        assert np.array_equal(
            cache.get(sample_docs, "openai", {"model": "model-b"})[0], emb_b
        )

    def test_content_change_invalidates_cache(self, cache):
        import numpy as np

        docs_v1 = [{"id": "d1", "text": "original"}, {"id": "d2", "text": "other"}]
        cache.put(
            docs_v1,
            "openai",
            np.array([[1, 2], [3, 4]]),
            ["d1", "d2"],
            {"model": "test"},
        )
        docs_v2 = [{"id": "d1", "text": "MODIFIED"}, {"id": "d2", "text": "other"}]
        assert cache.get(docs_v2, "openai", {"model": "test"}) is None

    def test_doc_added_invalidates_cache(self, cache):
        import numpy as np

        docs = [{"id": "d1", "text": "one"}, {"id": "d2", "text": "two"}]
        cache.put(docs, "openai", np.array([[1, 2], [3, 4]]), ["d1", "d2"])
        docs_3 = docs + [{"id": "d3", "text": "three"}]
        assert cache.get(docs_3, "openai") is None

    def test_doc_removed_invalidates_cache(self, cache):
        import numpy as np

        docs = [{"id": "d1", "text": "one"}, {"id": "d2", "text": "two"}]
        cache.put(docs, "openai", np.array([[1, 2], [3, 4]]), ["d1", "d2"])
        assert cache.get([docs[0]], "openai") is None

    def test_clear_removes_cache(self, cache, sample_docs):
        import numpy as np

        cache.put(sample_docs, "openai", np.array([[1, 2], [3, 4]]), ["d1", "d2"])
        cache.clear()
        assert cache.get(sample_docs, "openai") is None

    def test_document_hash_order_independent(self, cache):
        docs_a = [{"id": "d2", "text": "two"}, {"id": "d1", "text": "one"}]
        docs_b = [{"id": "d1", "text": "one"}, {"id": "d2", "text": "two"}]
        assert cache._compute_document_hash(docs_a) == cache._compute_document_hash(
            docs_b
        )

    def test_embedder_hash_config_order_independent(self, cache):
        h1 = cache._compute_embedder_hash("openai", {"model": "m", "dim": 3})
        h2 = cache._compute_embedder_hash("openai", {"dim": 3, "model": "m"})
        assert h1 == h2


class TestCachePipelineIntegration:
    """Cached pipeline must produce identical results to a fresh one."""

    @pytest.mark.parametrize(
        "embedder_type, embedder_params, query",
        [
            pytest.param(
                "openrouter",
                {"model": "qwen3-embedding-8b"},
                "savings account interest",
                marks=requires_openrouter,
            ),
            pytest.param(
                "openai",
                {"model": "text-embedding-3-large"},
                "credit card rewards",
                marks=requires_openai,
            ),
        ],
    )
    def test_cached_pipeline_matches_fresh(self, embedder_type, embedder_params, query):
        from tau2.domains.banking_knowledge.retrieval import (
            create_embedding_retrieval_pipeline,
        )

        kb = _make_mock_kb()
        pipelines = []
        for _ in range(2):
            with patch(
                "tau2.domains.banking_knowledge.retrieval.get_or_create_docs",
                return_value=DOCUMENTS,
            ):
                p = create_embedding_retrieval_pipeline(
                    kb, embedder_type, embedder_params, top_k=5
                )
                pipelines.append([doc_id for doc_id, _ in p.retrieve(query)])

        assert pipelines[0] == pipelines[1]
