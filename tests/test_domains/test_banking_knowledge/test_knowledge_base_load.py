import json
from pathlib import Path

from tau2.domains.banking_knowledge.data_model import KnowledgeBase


def test_load_orders_documents_by_filename(tmp_path, monkeypatch):
    """The loaded order must not depend on how the filesystem lists files."""
    for doc_id in ("doc_c", "doc_a", "doc_b"):
        (tmp_path / f"{doc_id}.json").write_text(
            json.dumps({"id": doc_id, "title": doc_id, "content": doc_id})
        )
    listed = sorted(tmp_path.glob("*.json"), reverse=True)
    monkeypatch.setattr(Path, "glob", lambda self, pattern: iter(listed))

    knowledge_base = KnowledgeBase.load(str(tmp_path))

    assert list(knowledge_base.documents) == ["doc_a", "doc_b", "doc_c"]
