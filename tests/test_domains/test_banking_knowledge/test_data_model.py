import json

from tau2.domains.banking_knowledge.data_model import KnowledgeBase


def test_knowledge_base_loads_documents_in_filename_order(tmp_path):
    documents = {
        "z.json": {"id": "doc_z", "title": "Z", "content": "last"},
        "a.json": {"id": "doc_a", "title": "A", "content": "first"},
    }
    for filename, document in documents.items():
        (tmp_path / filename).write_text(json.dumps(document), encoding="utf-8")

    knowledge_base = KnowledgeBase.load(str(tmp_path))

    assert knowledge_base.get_document_ids() == ["doc_a", "doc_z"]
