"""Tests for selecting the banking shell sandbox backend."""

from unittest.mock import MagicMock

import pytest

from tau2.domains.banking_knowledge.retrieval import ShellSpec, _create_sandbox


@pytest.fixture
def knowledge_base():
    document = MagicMock(id="doc_1", title="One", content="Body")
    knowledge_base = MagicMock()
    knowledge_base.get_all_documents.return_value = [document]
    return knowledge_base


def test_modal_backend_is_selected_and_populated(monkeypatch, knowledge_base):
    manager = MagicMock()
    manager_class = MagicMock(return_value=manager)
    monkeypatch.setenv("TAU2_SANDBOX_BACKEND", "modal")
    monkeypatch.setattr(
        "tau2.knowledge.modal_sandbox_manager.ModalSandboxManager", manager_class
    )

    result = _create_sandbox(knowledge_base, ShellSpec(file_format="md"))

    assert result is manager
    manager_class.assert_called_once_with(allow_writes=False)
    manager.export_documents.assert_called_once_with(
        [{"id": "doc_1", "title": "One", "content": "Body"}], file_format="md"
    )


def test_unknown_backend_fails_before_creating_sandbox(monkeypatch, knowledge_base):
    monkeypatch.setenv("TAU2_SANDBOX_BACKEND", "docker")

    with pytest.raises(ValueError, match="either 'local' or 'modal'"):
        _create_sandbox(knowledge_base, ShellSpec())
