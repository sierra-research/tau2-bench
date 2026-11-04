"""Tests for SandboxManager startup dep verification.

These tests cover the eager dependency check added to ``SandboxManager.__init__``.
The motivating bug: when ``srt`` was installed via npm but the system tools it
shells out to (``rg``, ``bwrap``, ``socat``) were missing, every shell invocation
silently returned ``"Sandbox dependencies are not available on this system"`` as
a tool result. The agent treated that as a normal tool output and learned to
abandon the shell tool, while every simulation looked superficially "successful".

The fix raises ``SandboxRuntimeError`` at construction time so misconfigured
environments are loud at the start of a run.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tau2.knowledge import sandbox_manager
from tau2.knowledge.sandbox_manager import (
    _SECRET_ENV_NAME_RE,
    SandboxManager,
    SandboxRuntimeError,
    _check_sandbox_dependencies,
)

_FAKE_SECRET = "fake-secret-value-0123456789"


@pytest.fixture(autouse=True)
def _reset_dep_cache():
    """Make every test independent of the global cache."""
    sandbox_manager._DEPS_VERIFIED = False
    yield
    sandbox_manager._DEPS_VERIFIED = False


def _which_returning(present: set[str]):
    """Build a fake ``shutil.which`` that returns a path only for ``present``."""

    def fake_which(name: str) -> str | None:
        return f"/usr/bin/{name}" if name in present else None

    return fake_which


class TestRequiredBinaries:
    def test_linux_requires_srt_rg_bwrap_socat(self):
        with patch.object(sandbox_manager.sys, "platform", "linux"):
            assert sandbox_manager._required_binaries() == (
                "srt",
                "rg",
                "bwrap",
                "socat",
            )

    def test_macos_requires_srt_rg(self):
        with patch.object(sandbox_manager.sys, "platform", "darwin"):
            assert sandbox_manager._required_binaries() == ("srt", "rg")

    def test_unsupported_platform_raises(self):
        with patch.object(sandbox_manager.sys, "platform", "win32"):
            with pytest.raises(SandboxRuntimeError, match="not supported"):
                sandbox_manager._required_binaries()


class TestCheckSandboxDependencies:
    def test_all_present_passes(self):
        with (
            patch.object(sandbox_manager.sys, "platform", "linux"),
            patch.object(
                sandbox_manager.shutil,
                "which",
                _which_returning({"srt", "rg", "bwrap", "socat"}),
            ),
        ):
            _check_sandbox_dependencies()

    def test_missing_rg_raises_with_install_hint(self):
        with (
            patch.object(sandbox_manager.sys, "platform", "linux"),
            patch.object(
                sandbox_manager.shutil,
                "which",
                _which_returning({"srt", "bwrap", "socat"}),
            ),
        ):
            with pytest.raises(SandboxRuntimeError) as exc_info:
                _check_sandbox_dependencies()
        msg = str(exc_info.value)
        assert "rg" in msg
        assert "ripgrep" in msg
        assert "apt install" in msg

    def test_missing_bwrap_raises_on_linux(self):
        with (
            patch.object(sandbox_manager.sys, "platform", "linux"),
            patch.object(
                sandbox_manager.shutil,
                "which",
                _which_returning({"srt", "rg", "socat"}),
            ),
        ):
            with pytest.raises(SandboxRuntimeError, match="bwrap"):
                _check_sandbox_dependencies()

    def test_missing_socat_raises_on_linux(self):
        with (
            patch.object(sandbox_manager.sys, "platform", "linux"),
            patch.object(
                sandbox_manager.shutil,
                "which",
                _which_returning({"srt", "rg", "bwrap"}),
            ),
        ):
            with pytest.raises(SandboxRuntimeError, match="socat"):
                _check_sandbox_dependencies()

    def test_missing_srt_raises(self):
        with (
            patch.object(sandbox_manager.sys, "platform", "linux"),
            patch.object(
                sandbox_manager.shutil,
                "which",
                _which_returning({"rg", "bwrap", "socat"}),
            ),
        ):
            with pytest.raises(SandboxRuntimeError) as exc_info:
                _check_sandbox_dependencies()
        msg = str(exc_info.value)
        assert "srt" in msg
        assert "@anthropic-ai/sandbox-runtime" in msg

    def test_macos_does_not_require_bwrap_or_socat(self):
        with (
            patch.object(sandbox_manager.sys, "platform", "darwin"),
            patch.object(
                sandbox_manager.shutil, "which", _which_returning({"srt", "rg"})
            ),
        ):
            _check_sandbox_dependencies()

    def test_macos_install_hint_uses_brew(self):
        with (
            patch.object(sandbox_manager.sys, "platform", "darwin"),
            patch.object(sandbox_manager.shutil, "which", _which_returning({"srt"})),
        ):
            with pytest.raises(SandboxRuntimeError) as exc_info:
                _check_sandbox_dependencies()
        msg = str(exc_info.value)
        assert "brew install" in msg
        assert "rg" in msg

    def test_check_is_cached(self):
        """First call hits ``shutil.which``; subsequent calls return immediately."""
        mock_which = MagicMock(
            side_effect=_which_returning({"srt", "rg", "bwrap", "socat"})
        )
        with (
            patch.object(sandbox_manager.sys, "platform", "linux"),
            patch.object(sandbox_manager.shutil, "which", mock_which),
        ):
            _check_sandbox_dependencies()
            calls_after_first = mock_which.call_count
            _check_sandbox_dependencies()
            _check_sandbox_dependencies()
            assert mock_which.call_count == calls_after_first

    def test_force_bypasses_cache(self):
        """``force=True`` re-runs the check even after a successful one."""
        mock_which = MagicMock(
            side_effect=_which_returning({"srt", "rg", "bwrap", "socat"})
        )
        with (
            patch.object(sandbox_manager.sys, "platform", "linux"),
            patch.object(sandbox_manager.shutil, "which", mock_which),
        ):
            _check_sandbox_dependencies()
            calls_after_first = mock_which.call_count
            _check_sandbox_dependencies(force=True)
            assert mock_which.call_count > calls_after_first


class TestSandboxManagerInit:
    def test_init_raises_when_deps_missing(self, tmp_path):
        with (
            patch.object(sandbox_manager.sys, "platform", "linux"),
            patch.object(sandbox_manager.shutil, "which", _which_returning({"srt"})),
        ):
            with pytest.raises(SandboxRuntimeError):
                SandboxManager(base_temp_dir=str(tmp_path))

    def test_init_succeeds_when_deps_present(self, tmp_path):
        # Skip if we can't actually create a kb dir on this filesystem
        with (
            patch.object(sandbox_manager.sys, "platform", "linux"),
            patch.object(
                sandbox_manager.shutil,
                "which",
                _which_returning({"srt", "rg", "bwrap", "socat"}),
            ),
        ):
            sm = SandboxManager(base_temp_dir=str(tmp_path))
            try:
                assert sm.kb_dir.exists()
            finally:
                sm.cleanup()

    def test_init_does_not_run_subprocess(self, tmp_path):
        """The dep check is purely PATH-based; it must not shell out to ``srt``.

        Running an actual srt invocation from __init__ would slow simulation
        startup and could fail for unrelated reasons (e.g., bwrap mount races).
        """
        with (
            patch.object(sandbox_manager.sys, "platform", "linux"),
            patch.object(
                sandbox_manager.shutil,
                "which",
                _which_returning({"srt", "rg", "bwrap", "socat"}),
            ),
            patch.object(sandbox_manager.subprocess, "run") as mock_run,
        ):
            SandboxManager(base_temp_dir=str(tmp_path))
            mock_run.assert_not_called()


@pytest.mark.skipif(
    shutil.which("srt") is None
    or (
        sys.platform.startswith("linux")
        and (shutil.which("bwrap") is None or shutil.which("socat") is None)
    )
    or shutil.which("rg") is None,
    reason="sandbox-runtime + system deps not all installed locally",
)
class TestSandboxManagerLive:
    """End-to-end check that real construction works on a properly-configured machine."""

    def test_can_construct_real_sandbox(self, tmp_path):
        sm = SandboxManager(base_temp_dir=str(tmp_path))
        try:
            assert sm.kb_dir.exists()
        finally:
            sm.cleanup()

    def test_live_env_does_not_leak_secrets(self, monkeypatch):
        """``env`` inside the real sandbox must not show parent API keys."""
        monkeypatch.setenv("FAKE_TEST_API_KEY", _FAKE_SECRET)
        # Default sandbox root: a pytest tmp_path sits in the shared temp dir,
        # whose every entry would need its own deny rule.
        sm = SandboxManager()
        try:
            rc, stdout, stderr = sm.run_command("env")
            assert rc == 0, stderr
            assert "FAKE_TEST_API_KEY" not in stdout
            assert _FAKE_SECRET not in stdout
            names = {line.split("=", 1)[0] for line in stdout.splitlines()}
            assert not any(_SECRET_ENV_NAME_RE.search(n) for n in names)
        finally:
            sm.cleanup()


# --- Minimal sandbox environment and output redaction -------------------------
#
# Motivating bug: ``run_command`` called ``subprocess.run`` without ``env=``, so
# an agent running ``env`` through the shell tool received every API key loaded
# from ``.env`` in its tool result.


@pytest.fixture
def mocked_sandbox(tmp_path):
    """A SandboxManager constructed with dependency checks mocked out."""
    with (
        patch.object(sandbox_manager.sys, "platform", "linux"),
        patch.object(
            sandbox_manager.shutil,
            "which",
            _which_returning({"srt", "rg", "bwrap", "socat"}),
        ),
    ):
        sm = SandboxManager(base_temp_dir=str(tmp_path))
    try:
        yield sm
    finally:
        sm.cleanup()


def _completed(stdout: str = "", stderr: str = "") -> MagicMock:
    return MagicMock(returncode=0, stdout=stdout, stderr=stderr)


class TestSandboxEnvironment:
    def test_run_command_passes_minimal_env(self, mocked_sandbox, monkeypatch):
        monkeypatch.setenv("FAKE_TEST_API_KEY", _FAKE_SECRET)
        monkeypatch.setenv("FAKE_SERVICE_TOKEN", _FAKE_SECRET)
        monkeypatch.setenv("SOME_UNRELATED_VAR", "launcher-value")
        monkeypatch.setenv("PATH", "/usr/bin:/bin")
        monkeypatch.setenv("LC_ALL", "en_US.UTF-8")

        with patch.object(
            sandbox_manager.subprocess, "run", return_value=_completed()
        ) as mock_run:
            mocked_sandbox.run_command("env")

        env = mock_run.call_args.kwargs["env"]
        assert env is not None
        assert "FAKE_TEST_API_KEY" not in env
        assert "FAKE_SERVICE_TOKEN" not in env
        assert "SOME_UNRELATED_VAR" not in env
        assert _FAKE_SECRET not in env.values()
        assert not any(_SECRET_ENV_NAME_RE.search(name) for name in env)
        # PATH is the policy-filtered PATH chosen at construction (parent
        # entries that stay readable, plus srt's own dependency dirs).
        assert env["PATH"] == mocked_sandbox._sandbox_path
        assert env["LC_ALL"] == "en_US.UTF-8"
        # HOME points at the sandbox, not the real user's home directory.
        assert env["HOME"] == str(mocked_sandbox.sandbox_dir)
        # Working directory behavior is unchanged.
        assert mock_run.call_args.kwargs["cwd"] == str(mocked_sandbox.kb_dir)

    def test_env_keys_are_allowlisted(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", _FAKE_SECRET)
        monkeypatch.setenv("USER", "someone")
        env = sandbox_manager._build_sandbox_env(home="/sandbox")
        for name in env:
            assert (
                name == "HOME"
                or name in sandbox_manager._SANDBOX_ENV_ALLOWLIST
                or name.startswith(sandbox_manager._SANDBOX_ENV_ALLOWED_PREFIXES)
            )

    def test_secret_like_name_dropped_even_with_allowed_prefix(self, monkeypatch):
        monkeypatch.setenv("LC_FAKE_TOKEN", _FAKE_SECRET)
        env = sandbox_manager._build_sandbox_env(home="/sandbox")
        assert "LC_FAKE_TOKEN" not in env

    def test_deny_read_paths_do_not_depend_on_sandbox_home(self, mocked_sandbox):
        """srt expands ``~`` via HOME, which now points at the sandbox dir."""
        settings = json.loads(mocked_sandbox.settings_path.read_text())
        deny_read = settings["filesystem"]["denyRead"]
        assert not any(p.startswith("~") for p in deny_read)
        # The real home directory (or ~/.ssh within it) is covered.
        assert _denied(Path.home() / ".ssh", deny_read)


class TestSecretRedaction:
    def test_sanitize_output_redacts_secret_env_values(
        self, mocked_sandbox, monkeypatch
    ):
        monkeypatch.setenv("FAKE_TEST_API_KEY", _FAKE_SECRET)
        out = mocked_sandbox._sanitize_output(
            f"FAKE_TEST_API_KEY={_FAKE_SECRET}\nother line\n", "env"
        )
        assert _FAKE_SECRET not in out
        assert "FAKE_TEST_API_KEY=[REDACTED]" in out
        assert "other line" in out

    def test_short_and_non_secret_values_are_not_redacted(
        self, mocked_sandbox, monkeypatch
    ):
        monkeypatch.setenv("FAKE_DEBUG_KEY", "1")
        monkeypatch.setenv("FAKE_PLAIN_VAR", "plain-but-long-value")
        out = mocked_sandbox._sanitize_output("1 plain-but-long-value\n", "cat")
        assert out == "1 plain-but-long-value\n"

    def test_run_command_redacts_stdout_and_stderr(self, mocked_sandbox, monkeypatch):
        monkeypatch.setenv("FAKE_TEST_API_KEY", _FAKE_SECRET)
        with patch.object(
            sandbox_manager.subprocess,
            "run",
            return_value=_completed(
                stdout=f"leaked {_FAKE_SECRET}", stderr=f"err {_FAKE_SECRET}"
            ),
        ):
            rc, stdout, stderr = mocked_sandbox.run_command("cat something.txt")
        assert rc == 0
        assert stdout == "leaked [REDACTED]"
        assert stderr == "err [REDACTED]"


# --- Read isolation -----------------------------------------------------------
#
# Motivating bug: srt's denyRead only covered ~/.ssh, ~/.aws, ~/.config and
# ~/.gnupg. A voice agent used ``python3 -c "print(os.listdir('/tmp'))"`` (the
# command filter ignores quoted text) to walk /tmp, the per-user temp dir and
# the home directory, and read other checkouts' banking_knowledge tasks.json /
# db.json and other runs' logs, leaking task answers into the run.


def _denied(path: Path | str, deny_read: list[str]) -> bool:
    real = Path(os.path.realpath(path))
    return any(real == Path(d) or Path(d) in real.parents for d in deny_read)


def _deny_read(sm: SandboxManager) -> list[str]:
    return json.loads(sm.settings_path.read_text())["filesystem"]["denyRead"]


@pytest.fixture
def default_root_sandbox():
    """A mocked-deps SandboxManager in the default (production) sandbox root."""
    with patch.object(
        sandbox_manager.shutil,
        "which",
        _which_returning({"srt", "rg", "bwrap", "socat", "node"}),
    ):
        sm = SandboxManager()
    try:
        yield sm
    finally:
        sm.cleanup()


class TestReadPolicy:
    def test_walk_denies_everything_but_allowed_paths(self, tmp_path):
        root = tmp_path / "root"
        kb = root / "a" / "sandbox" / "kb"
        kb.mkdir(parents=True)
        (kb / "doc.txt").write_text("doc")
        (root / "a" / "sandbox" / "settings.json").write_text("{}")
        (root / "a" / "other_checkout" / "data").mkdir(parents=True)
        (root / "b").mkdir()
        (root / "secret.txt").write_text("secret")
        (root / "link_to_b").symlink_to(root / "b")

        denies = sandbox_manager._compute_read_denies([kb], root=root)

        assert sorted(denies) == sorted(
            str(p)
            for p in (
                root / "a" / "sandbox" / "settings.json",
                root / "a" / "other_checkout",
                root / "b",
                root / "secret.txt",
            )
        )
        assert not _denied(kb / "doc.txt", denies)

    def test_kb_dir_is_not_under_any_denied_path(self, default_root_sandbox):
        sm = default_root_sandbox
        deny_read = _deny_read(sm)
        assert deny_read
        assert all(Path(d).is_absolute() for d in deny_read)
        assert not _denied(sm.kb_dir, deny_read)
        assert not _denied(sm.kb_dir / "doc.txt", deny_read)
        # The settings file sits next to kb_dir and is denied.
        assert _denied(sm.settings_path, deny_read)

    def test_broad_roots_are_denied(self, default_root_sandbox):
        from tau2.utils.utils import DATA_DIR

        sm = default_root_sandbox
        deny_read = _deny_read(sm)
        probe = "tau2_probe_does_not_need_to_exist"
        assert _denied(Path.home() / probe, deny_read)
        assert _denied(Path.home() / ".ssh", deny_read)
        assert _denied(Path(tempfile.gettempdir()) / probe, deny_read)
        assert _denied(Path("/tmp") / probe, deny_read)
        assert _denied(Path.cwd() / probe, deny_read)
        assert _denied(Path(DATA_DIR) / "tau2" / "domains", deny_read)
        repo = Path(sandbox_manager.__file__).resolve().parents[3]
        assert _denied(repo / "README.md", deny_read)
        assert _denied("/etc/passwd", deny_read)
        if sys.platform == "darwin":
            assert _denied("/Volumes", deny_read)
            assert _denied("/private/var/folders", deny_read)
            assert _denied("/System/Volumes/Data", deny_read)

    def test_sibling_sandbox_created_later_is_denied(self, default_root_sandbox):
        sm = default_root_sandbox
        sibling = sm.sandbox_dir.parent / f"agentic_search_probe_{uuid.uuid4().hex}"
        sibling.mkdir()
        try:
            # run_command refreshes the settings before every command.
            with patch.object(
                sandbox_manager.subprocess, "run", return_value=_completed()
            ):
                sm.run_command("ls")
            assert _denied(sibling, _deny_read(sm))
        finally:
            sibling.rmdir()

    def test_system_tool_dirs_stay_readable(self, default_root_sandbox):
        deny_read = _deny_read(default_root_sandbox)
        for d in ("/usr/bin", "/bin"):
            assert not _denied(d, deny_read)

    def test_extra_deny_never_covers_allowed_path(self, tmp_path):
        kb = Path(os.path.realpath(tmp_path)) / "kb"
        kb.mkdir()
        with patch.object(
            sandbox_manager, "_extra_read_denies", return_value=[kb.parent]
        ):
            denies = sandbox_manager._build_read_denies([kb])
        assert not _denied(kb, denies)

    def test_sandbox_root_is_private(self, default_root_sandbox):
        st = default_root_sandbox.sandbox_dir.parent.stat()
        assert st.st_uid == os.getuid()
        assert st.st_mode & 0o077 == 0

    def test_sandbox_dir_removed_when_manager_is_collected(self):
        import gc

        with patch.object(
            sandbox_manager.shutil,
            "which",
            _which_returning({"srt", "rg", "bwrap", "socat"}),
        ):
            sm = SandboxManager()
        sandbox_dir = sm.sandbox_dir
        assert sandbox_dir.exists()
        del sm
        gc.collect()
        assert not sandbox_dir.exists()

    def test_python_falls_back_to_running_interpreter(self, monkeypatch):
        """With no python3 in an allowed dir, link the running interpreter."""
        real_access = os.access

        def no_system_python(path, mode):
            if str(path).endswith("/python3"):
                return False
            return real_access(path, mode)

        monkeypatch.setattr(sandbox_manager.os, "access", no_system_python)
        with patch.object(
            sandbox_manager.shutil,
            "which",
            _which_returning({"srt", "rg", "bwrap", "socat"}),
        ):
            sm = SandboxManager()
        try:
            link = sm.bin_dir / "python3"
            assert link.is_symlink()
            target = Path(os.path.realpath(link))
            assert target.exists()
            assert sm._sandbox_path.split(os.pathsep)[0] == str(sm.bin_dir)
            deny_read = _deny_read(sm)
            for path in (target, sys.base_prefix, link, sm.kb_dir):
                assert not _denied(path, deny_read), path
        finally:
            sm.cleanup()


_LIVE_DEPS_MISSING = (
    shutil.which("srt") is None
    or (
        sys.platform.startswith("linux")
        and (shutil.which("bwrap") is None or shutil.which("socat") is None)
    )
    or shutil.which("rg") is None
)


@pytest.fixture
def secrets_outside_kb():
    """Secret files outside the KB: in the temp dir, /tmp and the home dir."""
    marker = f"tau2_sandbox_probe_{uuid.uuid4().hex[:8]}"
    content = f"SECRET-{uuid.uuid4().hex}"
    paths = []
    for base in {tempfile.gettempdir(), "/tmp", str(Path.home())}:
        p = Path(base) / f"{marker}.txt"
        p.write_text(content)
        paths.append(p)
    try:
        yield marker, content, paths
    finally:
        for p in paths:
            p.unlink(missing_ok=True)


@pytest.fixture
def live_sandbox():
    sm = SandboxManager()
    sm.export_documents(
        [
            {
                "id": "doc_travel",
                "title": "Travel",
                "content": "Fees when you travel abroad.\nCafé déjà vu — 日本語",
            },
            {"id": "doc_cards", "title": "Cards", "content": "Credit card limits."},
        ]
    )
    try:
        yield sm
    finally:
        sm.cleanup()


@pytest.mark.skipif(
    _LIVE_DEPS_MISSING, reason="sandbox-runtime + system deps not all installed"
)
class TestReadIsolationLive:
    def test_cannot_read_secrets_outside_kb(self, live_sandbox, secrets_outside_kb):
        marker, content, paths = secrets_outside_kb
        for p in paths:
            for cmd in (
                # Quoted paths pass the command filter, so srt must block them.
                f'cat "{p}"',
                f'ls "{p.parent}/"',
                f'ls -la "{p.parent}/"',
                # Exactly the quoted python3 form used in the escape.
                f"python3 -c \"print(open('{p}').read())\"",
                f"python3 -c \"import os; print(os.listdir('{p.parent}'))\"",
            ):
                rc, stdout, stderr = live_sandbox.run_command(cmd)
                assert rc != 0, (cmd, stdout)
                assert content not in stdout + stderr, cmd
                assert marker not in stdout, cmd
        # Listing the home directory itself reveals nothing either.
        rc, stdout, _ = live_sandbox.run_command(f'ls "{Path.home()}"')
        assert marker not in stdout

    def test_listing_tmp_reveals_nothing(self, live_sandbox, secrets_outside_kb):
        marker, content, _ = secrets_outside_kb
        for cmd in (
            'ls "/tmp/"',
            "python3 -c \"import os; print(os.listdir('/tmp'))\"",
            'python3 -c "import tempfile, os; '
            'print(os.listdir(tempfile.gettempdir()))"',
            'find "/tmp/" -maxdepth 1',
        ):
            rc, stdout, stderr = live_sandbox.run_command(cmd)
            assert marker not in stdout, cmd
            assert content not in stdout, cmd

    def test_cannot_read_repository_or_sibling_sandboxes(self, live_sandbox):
        repo = Path(sandbox_manager.__file__).resolve().parents[3]
        other = SandboxManager()
        try:
            other.export_documents(
                [{"id": "other_doc", "title": "Other", "content": "OTHER-RUN-NOTE"}]
            )
            for cmd in (
                f'ls "{repo}"',
                f"python3 -c \"print(open('{repo / 'AGENTS.md'}').read())\"",
                f'cat "{other.kb_dir}/other_doc.txt"',
                f'cat "{live_sandbox.settings_path}"',
            ):
                rc, stdout, stderr = live_sandbox.run_command(cmd)
                assert rc != 0, (cmd, stdout)
                assert "OTHER-RUN-NOTE" not in stdout
                assert "denyRead" not in stdout
        finally:
            other.cleanup()

    @pytest.mark.parametrize(
        "cmd,expected",
        [
            ("ls", "doc_travel.txt"),
            ("cat doc_travel.txt", "Café déjà vu — 日本語"),
            ('grep -riE "travel|abroad" .', "travel abroad"),
            ("rg -i 'café' .", "日本語"),
            ("find . -name '*.txt'", "./doc_cards.txt"),
            ("head -n 1 doc_cards.txt", "Title: Cards"),
            ("wc -l doc_travel.txt", "doc_travel.txt"),
            ("awk 'NR==1' doc_cards.txt", "Title: Cards"),
            ("sed -n 1p doc_travel.txt", "Title: Travel"),
            (
                "python3 -c \"print(open('doc_travel.txt', encoding='utf-8').read())\"",
                "Café déjà vu — 日本語",
            ),
            (
                "python3 -c \"import os, json, re; print(sorted(os.listdir('.')))\"",
                "doc_cards.txt",
            ),
        ],
    )
    def test_normal_commands_work_on_kb(self, live_sandbox, cmd, expected):
        rc, stdout, stderr = live_sandbox.run_command(cmd)
        assert rc == 0, stderr
        assert expected in stdout
        assert stderr.strip() == "", stderr

    def test_quoted_grep_pattern_is_not_blocked(self, live_sandbox):
        # Basic-regex grep treats "|" literally: no match, but no error.
        rc, stdout, stderr = live_sandbox.run_command('grep -ri "travel|abroad" .')
        assert rc == 1, stderr
        assert stderr == ""
