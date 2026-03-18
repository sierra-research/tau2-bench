"""Concrete composed toolkit classes for the banking_knowledge domain.

Each class combines the base ``KnowledgeTools`` with the appropriate
retrieval MixIns.  The ``ToolKitType`` metaclass automatically collects
``@is_tool`` methods from all parents via MRO — no manual delegation
is needed.

MRO example for ``KnowledgeToolsWithKBSearchAndGrep``::

    KnowledgeToolsWithKBSearchAndGrep
      └─ KnowledgeTools          → get_user_information_by_id, transfer_to_human_agents, …
          └─ ToolKitBase         → (base class with db)
      └─ KBSearchMixin           → KB_search
      └─ GrepMixin               → grep
"""

from typing import TYPE_CHECKING

from tau2.domains.banking_knowledge.retrieval_mixins import (
    GrepMixin,
    KBSearchMixin,
    ShellMixin,
)
from tau2.domains.banking_knowledge.tools import KnowledgeTools

if TYPE_CHECKING:
    from tau2.domains.banking_knowledge.data_model import TransactionalDB
    from tau2.knowledge.pipeline import RetrievalPipeline
    from tau2.knowledge.sandbox_manager import SandboxManager


class KnowledgeToolsPlain(KnowledgeTools):
    """Base banking tools with no retrieval capabilities.

    Used by: no_knowledge, full_kb, golden_retrieval.
    """

    pass  # Inherits all KnowledgeTools @is_tool methods, adds nothing.


class KnowledgeToolsWithKBSearch(KBSearchMixin, KnowledgeTools):
    """Base banking tools + KB_search.

    Used by: bm25, qwen_embeddings, openai_embeddings, and reranker variants.
    """

    def __init__(self, db: "TransactionalDB", kb_pipeline: "RetrievalPipeline"):
        super().__init__(db)
        self._kb_pipeline = kb_pipeline


class KnowledgeToolsWithGrep(GrepMixin, KnowledgeTools):
    """Base banking tools + grep.

    Used by: grep_only.
    """

    def __init__(self, db: "TransactionalDB", grep_pipeline: "RetrievalPipeline"):
        super().__init__(db)
        self._grep_pipeline = grep_pipeline


class KnowledgeToolsWithKBSearchAndGrep(KBSearchMixin, GrepMixin, KnowledgeTools):
    """Base banking tools + KB_search + grep.

    Used by: qwen_embeddings_grep, openai_embeddings_grep, bm25_grep,
    and their reranker variants.
    """

    def __init__(
        self,
        db: "TransactionalDB",
        kb_pipeline: "RetrievalPipeline",
        grep_pipeline: "RetrievalPipeline",
    ):
        super().__init__(db)
        self._kb_pipeline = kb_pipeline
        self._grep_pipeline = grep_pipeline


class KnowledgeToolsWithShell(ShellMixin, KnowledgeTools):
    """Base banking tools + shell.

    Used by: terminal_use, terminal_use_write.
    """

    def __init__(self, db: "TransactionalDB", sandbox: "SandboxManager"):
        super().__init__(db)
        self._sandbox = sandbox
