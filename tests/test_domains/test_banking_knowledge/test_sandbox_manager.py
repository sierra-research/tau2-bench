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

import os
import shutil
import subprocess
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

from tau2.knowledge import sandbox_manager
from tau2.knowledge.sandbox_manager import (
    SandboxManager,
    SandboxRuntimeError,
    _check_sandbox_dependencies,
)


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

    @pytest.mark.skipif(
        not sys.platform.startswith("linux"), reason="srt starts socat on Linux only"
    )
    def test_timed_out_command_leaves_no_proxies(self, tmp_path):
        sm = SandboxManager(base_temp_dir=str(tmp_path))
        try:
            assert sm.run_command("sleep 30", timeout=3) == (
                124,
                "",
                "Command timed out after 3 seconds",
            )
        finally:
            sm.cleanup()
        deadline = time.monotonic() + 10
        while _socat_in_own_group() and time.monotonic() < deadline:
            time.sleep(0.2)
        assert _socat_in_own_group() == []


def _socat_in_own_group() -> list[str]:
    """IDs of the socat processes in this process's group, as ``ps`` lists them."""
    listed = subprocess.run(
        ["ps", "-A", "-o", "pid=", "-o", "pgid=", "-o", "comm="],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    group = str(os.getpgid(0))
    return [
        fields[0]
        for fields in map(str.split, listed.splitlines())
        if fields[1:] == [group, "socat"]
    ]


def _child(code: str) -> list[str]:
    return [sys.executable, "-c", code]


class TestRunSrt:
    """``_run_srt`` on a stand-in process, since only its process handling differs."""

    def test_completed_command(self, tmp_path):
        result = sandbox_manager._run_srt(
            _child("print('ok')"), timeout=30, cwd=str(tmp_path)
        )
        assert (result.returncode, result.stdout) == (0, "ok\n")

    def test_timeout_sends_sigterm_first(self, tmp_path):
        marker = tmp_path / "terminated"
        code = (
            "import signal, sys, time\n"
            "def stop(*_):\n"
            f"    open({str(marker)!r}, 'w').close()\n"
            "    sys.exit(0)\n"
            "signal.signal(signal.SIGTERM, stop)\n"
            "time.sleep(60)\n"
        )
        with pytest.raises(subprocess.TimeoutExpired):
            sandbox_manager._run_srt(_child(code), timeout=3, cwd=str(tmp_path))
        assert marker.exists()

    def test_sigkill_after_the_grace_period(self, tmp_path):
        pid_file = tmp_path / "pid"
        code = (
            "import os, signal, time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            f"open({str(pid_file)!r}, 'w').write(str(os.getpid()))\n"
            "time.sleep(60)\n"
        )
        start = time.monotonic()
        with patch.object(sandbox_manager, "_SRT_TERMINATE_GRACE_SECONDS", 0.5):
            with pytest.raises(subprocess.TimeoutExpired):
                sandbox_manager._run_srt(_child(code), timeout=3, cwd=str(tmp_path))
        assert time.monotonic() - start < 30
        with pytest.raises(ProcessLookupError):
            os.kill(int(pid_file.read_text()), 0)

    def test_run_command_reports_a_timeout(self, tmp_path):
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
            with patch.object(
                sandbox_manager,
                "_run_srt",
                side_effect=subprocess.TimeoutExpired(["srt"], 7),
            ):
                assert sm.run_command("sleep 9", timeout=7) == (
                    124,
                    "",
                    "Command timed out after 7 seconds",
                )
        finally:
            sm.cleanup()
