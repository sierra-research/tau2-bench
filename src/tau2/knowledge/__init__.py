"""Knowledge retrieval pipeline module."""

_registered = False


def _ensure_registered():
    """Import all knowledge components to trigger @register_* decorators.

    Called lazily before pipeline construction so that `import tau2` doesn't
    pull in optional dependencies like rank_bm25.
    """
    global _registered
    if _registered:
        return
    import tau2.knowledge.document_preprocessors.bm25_indexer  # noqa: F401
    import tau2.knowledge.document_preprocessors.embedding_indexer  # noqa: F401
    import tau2.knowledge.input_preprocessors.embedding_encoder  # noqa: F401
    import tau2.knowledge.postprocessors.bge_reranker  # noqa: F401
    import tau2.knowledge.postprocessors.pointwise_llm_reranker  # noqa: F401
    import tau2.knowledge.postprocessors.qwen_reranker  # noqa: F401
    import tau2.knowledge.retrievers.bm25_retriever  # noqa: F401
    import tau2.knowledge.retrievers.cosine_retriever  # noqa: F401
    import tau2.knowledge.retrievers.grep_retriever  # noqa: F401

    _registered = True
