"""Unit tests for the Modal-backed banking knowledge sandbox."""

from __future__ import annotations

import json
import tarfile
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tau2.knowledge import modal_sandbox_manager
from tau2.knowledge.modal_sandbox_manager import (
    BASH_EXECUTABLE,
    DEFAULT_MODAL_APP,
    REMOTE_KB_DIR,
    REMOTE_KB_TMPFS_DIR,
    ModalSandboxManager,
    ModalSandboxRuntimeError,
    _order_digest,
    _ordered_file_digest,
)


class FakeExecTimeoutError(Exception):
    """Test double for ``modal.exception.ExecTimeoutError``."""


def _fake_modal(
    *,
    stdout: str | bytes = "ok\n",
    stderr: str | bytes = "",
    return_code: int = 0,
):
    process = MagicMock()
    process.stdout.read.return_value = stdout
    process.stderr.read.return_value = stderr
    process.wait.return_value = return_code

    init_process = MagicMock()
    init_process.stdout.read.return_value = ""
    init_process.stderr.read.return_value = ""
    init_process.wait.return_value = 0

    sandbox = MagicMock()

    def exec_process(*args, **kwargs):
        is_initialization = (
            args[:2] == ("bash", "-c")
            and "knowledge_base.tar" in args[2]
            and REMOTE_KB_TMPFS_DIR in args[2]
        )
        return init_process if is_initialization else process

    sandbox.exec.side_effect = exec_process

    image = MagicMock()
    image.run_commands.return_value = image
    image.add_local_file.return_value = image

    modal = SimpleNamespace(
        App=SimpleNamespace(lookup=MagicMock(return_value=object())),
        Image=SimpleNamespace(debian_slim=MagicMock(return_value=image)),
        Sandbox=SimpleNamespace(create=MagicMock(return_value=sandbox)),
        exception=SimpleNamespace(ExecTimeoutError=FakeExecTimeoutError),
    )
    return modal, image, sandbox, process


@pytest.fixture
def manager_factory(tmp_path, monkeypatch):
    """Create managers and guarantee cleanup without importing real Modal."""
    managers: list[ModalSandboxManager] = []

    def create(**kwargs) -> ModalSandboxManager:
        manager = ModalSandboxManager(base_temp_dir=str(tmp_path), **kwargs)
        managers.append(manager)
        return manager

    yield create

    for manager in managers:
        manager.cleanup()


def test_export_is_local_and_remote_creation_is_lazy(manager_factory, monkeypatch):
    modal, image, sandbox, process = _fake_modal()
    load_modal = MagicMock(return_value=modal)
    monkeypatch.setattr(modal_sandbox_manager, "_load_modal", load_modal)

    manager = manager_factory(sandbox_id="banking-task")
    exported = manager.export_documents(
        [{"id": "cards/limits", "title": "Card Limits", "content": "Details"}],
        file_format="md",
    )

    assert load_modal.call_count == 0
    assert exported["cards/limits"].read_text() == "# Card Limits\n\nDetails"
    assert manager.list_files() == ["INDEX.md", "cards_limits.md"]
    assert manager.get_kb_path() == REMOTE_KB_DIR

    assert manager.run_command("ls -la") == (0, "ok\n", "")

    modal.App.lookup.assert_called_once_with(DEFAULT_MODAL_APP, create_if_missing=True)
    modal.Image.debian_slim.assert_called_once_with()
    image.run_commands.assert_called_once_with(f"mkdir -p {REMOTE_KB_DIR}")
    image.add_local_file.assert_called_once_with(
        local_path=str(manager.archive_path),
        remote_path="/tmp/knowledge_base.tar",
        copy=True,
    )
    create_kwargs = modal.Sandbox.create.call_args.kwargs
    assert create_kwargs == {
        "app": modal.App.lookup.return_value,
        "image": image,
        "workdir": REMOTE_KB_DIR,
        "block_network": True,
        "timeout": 3600,
    }
    assert "secrets" not in create_kwargs
    assert "env" not in create_kwargs
    assert process.wait.call_count == 1

    # Subsequent commands reuse the same remote sandbox.
    manager.run_command("cat INDEX.md")
    assert modal.Sandbox.create.call_count == 1
    assert sandbox.exec.call_count == 3


def test_app_and_remote_lifetime_are_configurable(manager_factory, monkeypatch):
    monkeypatch.setenv("TAU2_MODAL_APP", "custom-banking-app")
    monkeypatch.setenv("TAU2_MODAL_SANDBOX_TIMEOUT", "7200")
    modal, _, _, _ = _fake_modal()
    monkeypatch.setattr(modal_sandbox_manager, "_load_modal", lambda: modal)

    manager = manager_factory()
    manager.run_command("pwd")

    modal.App.lookup.assert_called_once_with(
        "custom-banking-app", create_if_missing=True
    )
    assert modal.Sandbox.create.call_args.kwargs["timeout"] == 7200


@pytest.mark.parametrize("value", ["0", "-1", "not-a-number"])
def test_invalid_remote_lifetime_fails_early(tmp_path, monkeypatch, value):
    monkeypatch.setenv("TAU2_MODAL_SANDBOX_TIMEOUT", value)

    with pytest.raises(ValueError, match="must be a positive integer"):
        ModalSandboxManager(base_temp_dir=str(tmp_path))

    assert list(tmp_path.iterdir()) == []


def test_escape_patterns_are_blocked_before_remote_creation(
    manager_factory, monkeypatch
):
    load_modal = MagicMock()
    monkeypatch.setattr(modal_sandbox_manager, "_load_modal", load_modal)
    manager = manager_factory()

    return_code, stdout, stderr = manager.run_command("cat ../private.txt")

    assert return_code == 1
    assert stdout == ""
    assert "contains '..'" in stderr
    load_modal.assert_not_called()


def test_read_only_commands_drop_privileges(manager_factory, monkeypatch):
    modal, _, sandbox, _ = _fake_modal()
    monkeypatch.setattr(modal_sandbox_manager, "_load_modal", lambda: modal)
    manager = manager_factory(allow_writes=False)
    manager.export_documents(
        [{"id": "doc", "title": "Doc", "content": "content"}],
    )

    manager.run_command("cat doc.txt")

    exec_args = sandbox.exec.call_args.args
    assert exec_args[0:2] == ("python3", "-c")
    assert exec_args[-3:] == (BASH_EXECUTABLE, "-c", "cat doc.txt")
    assert sandbox.exec.call_args.kwargs == {
        "timeout": 30,
        "workdir": REMOTE_KB_DIR,
        "env": {"PWD": REMOTE_KB_DIR},
        "text": True,
    }
    assert manager.kb_dir.stat().st_mode & 0o222 == 0
    assert all(path.stat().st_mode & 0o222 == 0 for path in manager.kb_dir.iterdir())


def test_writable_commands_run_directly(manager_factory, monkeypatch):
    modal, _, sandbox, _ = _fake_modal()
    monkeypatch.setattr(modal_sandbox_manager, "_load_modal", lambda: modal)
    manager = manager_factory(allow_writes=True)

    manager.run_command("touch notes.txt")

    assert sandbox.exec.call_args.args == (
        BASH_EXECUTABLE,
        "-c",
        "touch notes.txt",
    )


def test_command_output_is_sanitized(manager_factory, monkeypatch):
    raw_ls = "-rw-r--r-- 1 root root 12 Aug 22 14:30 doc.md\n"
    modal, _, _, _ = _fake_modal(stdout=raw_ls)
    monkeypatch.setattr(modal_sandbox_manager, "_load_modal", lambda: modal)
    manager = manager_factory()

    return_code, stdout, stderr = manager.run_command("ls -l")

    assert return_code == 0
    assert stderr == ""
    assert "root" not in stdout
    assert "kb_user" in stdout
    assert "kb_group" in stdout
    assert "Jan  1 00:00" in stdout


def test_exec_timeout_has_subprocess_compatible_result(manager_factory, monkeypatch):
    modal, _, _, process = _fake_modal()
    process.stdout.read.side_effect = FakeExecTimeoutError("remote details")
    monkeypatch.setattr(modal_sandbox_manager, "_load_modal", lambda: modal)
    manager = manager_factory()

    assert manager.run_command("sleep 60", timeout=2) == (
        124,
        "",
        "Command timed out after 2 seconds",
    )


def test_export_after_remote_start_is_rejected(manager_factory, monkeypatch):
    modal, _, _, _ = _fake_modal()
    monkeypatch.setattr(modal_sandbox_manager, "_load_modal", lambda: modal)
    manager = manager_factory()
    manager.run_command("ls")

    with pytest.raises(ModalSandboxRuntimeError, match="after.*started"):
        manager.export_documents([])


def test_cleanup_detaches_even_if_termination_fails(manager_factory, monkeypatch):
    modal, _, sandbox, _ = _fake_modal()
    sandbox.terminate.side_effect = RuntimeError("do not disclose this")
    monkeypatch.setattr(modal_sandbox_manager, "_load_modal", lambda: modal)
    manager = manager_factory()
    manager.run_command("ls")
    sandbox_dir = manager.sandbox_dir

    manager.cleanup()

    sandbox.terminate.assert_called_once_with(wait=True)
    sandbox.detach.assert_called_once_with()
    assert not sandbox_dir.exists()
    manager.cleanup()  # Idempotent.


def test_initialization_failure_terminates_and_detaches(manager_factory, monkeypatch):
    modal, _, sandbox, _ = _fake_modal()
    initialization = sandbox.exec.side_effect

    def fail_initialization(*args, **kwargs):
        if (
            args[:2] == ("bash", "-c")
            and "knowledge_base.tar" in args[2]
            and REMOTE_KB_TMPFS_DIR in args[2]
        ):
            raise RuntimeError("initialization failed")
        return initialization(*args, **kwargs)

    sandbox.exec.side_effect = fail_initialization
    sandbox.terminate.side_effect = RuntimeError("termination failed")
    monkeypatch.setattr(modal_sandbox_manager, "_load_modal", lambda: modal)
    manager = manager_factory()

    with pytest.raises(ModalSandboxRuntimeError, match="Failed to create"):
        manager.run_command("ls")

    sandbox.terminate.assert_called_once_with(wait=True)
    sandbox.detach.assert_called_once_with()


def test_context_manager_cleans_up_local_and_remote(tmp_path, monkeypatch):
    modal, _, sandbox, _ = _fake_modal()
    monkeypatch.setattr(modal_sandbox_manager, "_load_modal", lambda: modal)

    with ModalSandboxManager(base_temp_dir=str(tmp_path)) as manager:
        manager.run_command("ls")
        sandbox_dir = manager.sandbox_dir

    assert not sandbox_dir.exists()
    sandbox.terminate.assert_called_once_with(wait=True)
    sandbox.detach.assert_called_once_with()


def test_sandbox_info_exposes_only_non_secret_metadata(manager_factory):
    manager = manager_factory(sandbox_id="task-123")
    manager.export_documents([])

    info = manager.get_sandbox_info()

    assert info["sandbox_id"] == "task-123"
    assert info["kb_dir"] == REMOTE_KB_DIR
    assert info["num_files"] == 1
    assert info["remote_created"] is False
    assert not any("token" in key or "secret" in key for key in info)


def test_staged_archive_is_deterministic(manager_factory):
    documents = [{"id": "doc", "title": "Doc", "content": "content"}]
    first = manager_factory(sandbox_id="first")
    second = manager_factory(sandbox_id="second")
    first.export_documents(documents)
    second.export_documents(documents)

    first._build_archive()
    second._build_archive()

    assert first.archive_path.read_bytes() == second.archive_path.read_bytes()

    with tarfile.open(first.archive_path) as archive:
        assert {member.mode for member in archive.getmembers()} == {0o664}


def test_remote_permissions_match_official_but_remain_uid_protected(
    manager_factory, monkeypatch
):
    modal, _, sandbox, _ = _fake_modal()
    monkeypatch.setattr(modal_sandbox_manager, "_load_modal", lambda: modal)
    manager = manager_factory(allow_writes=False)
    manager.export_documents([{"id": "doc", "title": "Doc", "content": "text"}])

    manager.run_command("ls -la")

    initialization = sandbox.exec.call_args_list[0].args[2]
    assert f"tar -xf /tmp/knowledge_base.tar -C {REMOTE_KB_TMPFS_DIR}" in initialization
    assert f"chown -R 0:0 {REMOTE_KB_TMPFS_DIR}" in initialization
    assert f"find {REMOTE_KB_TMPFS_DIR} -type d -exec chmod 775" in initialization
    assert f"find {REMOTE_KB_TMPFS_DIR} -type f -exec chmod 664" in initialization
    assert f"rmdir {REMOTE_KB_DIR}" in initialization
    assert f"ln -s {REMOTE_KB_TMPFS_DIR} {REMOTE_KB_DIR}" in initialization


def test_custom_order_manifest_is_applied_only_to_exact_corpus(
    manager_factory, monkeypatch, tmp_path
):
    documents = [
        {"id": "a", "title": "A", "content": "alpha"},
        {"id": "b", "title": "B", "content": "beta"},
    ]
    baseline = manager_factory()
    baseline.export_documents(documents, file_format="md")
    paths_by_name = {path.name: path for path in baseline._archive_members}
    order = ["b.md", "a.md", "INDEX.md"]
    manifest = {
        "schema_version": 1,
        "entry_count": len(order),
        "corpus_export_sha256": _ordered_file_digest(paths_by_name),
        "order_sha256": _order_digest(order),
        "filenames": order,
    }
    manifest_path = tmp_path / "order.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setenv("TAU2_MODAL_ORDER_MANIFEST", str(manifest_path))

    manager = manager_factory()
    manager.export_documents(documents, file_format="md")

    assert [path.name for path in manager._archive_members] == order
    assert manager.get_sandbox_info()["order_manifest_applied"] is True
    assert manager.get_sandbox_info()["order_manifest_sha256"] == _order_digest(order)


def test_bare_root_ls_la_metadata_is_narrowly_normalized(manager_factory, monkeypatch):
    raw_ls = (
        "total 1440\n"
        "drwxrwxr-x 2 root root 14240 Aug 22 14:30 .\n"
        "drwxrwxrwt 3 root root 80 Aug 22 14:30 ..\n"
        "-rw-rw-r-- 1 root root 12 Aug 22 14:30 doc.md\n"
    )
    modal, _, _, _ = _fake_modal(stdout=raw_ls)
    monkeypatch.setattr(modal_sandbox_manager, "_load_modal", lambda: modal)
    manager = manager_factory()

    _, output, _ = manager.run_command("ls -la && cat INDEX.md")

    assert output.splitlines()[:3] == [
        "total 2992",
        "drwxrwxr-x 2 kb_user  kb_group  69632 Jan  1 00:00 .",
        "drwxrwxr-x 3 kb_user  kb_group  4096 Jan  1 00:00 ..",
    ]
    assert output.splitlines()[3] == (
        "-rw-rw-r-- 1 kb_user  kb_group  12 Jan  1 00:00 doc.md"
    )


def test_ls_la_with_explicit_path_keeps_backend_metadata(manager_factory, monkeypatch):
    raw_ls = (
        "total 1440\n"
        "drwxrwxr-x 2 root root 14240 Aug 22 14:30 .\n"
        "drwxrwxrwt 3 root root 80 Aug 22 14:30 ..\n"
    )
    modal, _, _, _ = _fake_modal(stdout=raw_ls)
    monkeypatch.setattr(modal_sandbox_manager, "_load_modal", lambda: modal)
    manager = manager_factory()

    _, output, _ = manager.run_command("ls -la .")

    assert output.startswith("total 1440\n")
    assert "14240" in output
    assert "69632" not in output
