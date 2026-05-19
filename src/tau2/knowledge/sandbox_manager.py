"""Sandbox manager for agentic search.

This module provides utilities for creating sandboxed environments where agents
can interact with knowledge base documents through shell commands (cat, grep, etc.).

SECURITY WARNING: This is BEST-EFFORT sandboxing only. The agent MAY be able to
escape the knowledge base directory and read files from elsewhere on the filesystem.
Use with caution and do not rely on this for security-critical applications.

Limitations:
- Read restrictions are "deny-only" in srt, so we cannot strictly limit reads to kb_dir
- We block obvious escape patterns (.., ~, absolute paths) but sophisticated attacks may bypass
- Write access is properly restricted via srt's "allow-only" pattern

Requires Anthropic's sandbox-runtime (srt) for filesystem isolation:
https://github.com/anthropic-experimental/sandbox-runtime

Install with:
    npm install -g @anthropic-ai/sandbox-runtime@0.0.23

NOTE: Version 0.0.23 is required. Versions 0.0.24+ have a regression on Linux where
bwrap leaves behind file stubs after sandbox exit, causing subsequent commands to fail
with "Can't mkdir parents for .../knowledge_base/.claude/commands: Not a directory".
TODO: Upgrade to latest version once https://github.com/anthropic-experimental/sandbox-runtime/pull/86 lands.

Additional dependencies:
- macOS: brew install ripgrep
- Linux: apt install ripgrep bubblewrap socat (or equivalent for your distro)

Constructing a SandboxManager calls ``_check_sandbox_dependencies`` exactly once
per process. If the required binaries are missing the constructor raises
``SandboxRuntimeError`` so a misconfigured environment fails loudly at run start
rather than silently degrading every shell tool call (the agent would otherwise
just see "Sandbox dependencies are not available on this system" repeatedly and
"abandon" the tool, which looks like an Anthropic policy change in the metrics
but is actually an install-time failure).
"""

import getpass
import json
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class SandboxRuntimeError(RuntimeError):
    """Raised when sandbox-runtime or its required system binaries are missing.

    The sandbox-runtime CLI ``srt`` shells out to a few system tools to do its
    job. On Linux it needs ``rg`` (ripgrep), ``bwrap`` (bubblewrap), and
    ``socat``. On macOS it needs ``rg``. If any are missing, every shell
    invocation fails with a generic "Sandbox dependencies are not available"
    message in stdout, and (without this exception) tau2 silently passes that
    error string back to the agent as if it were a normal tool result.

    This error is raised eagerly at ``SandboxManager`` construction time so the
    failure is loud and obvious, not buried in 388 trajectories.
    """


# Required binaries by platform. ``srt`` itself is always required.
_REQUIRED_BINARIES_LINUX: tuple[str, ...] = ("srt", "rg", "bwrap", "socat")
_REQUIRED_BINARIES_DARWIN: tuple[str, ...] = ("srt", "rg")

# Cached after first successful check so we only pay the cost once per process.
_DEPS_VERIFIED = False


def _required_binaries() -> tuple[str, ...]:
    if sys.platform.startswith("linux"):
        return _REQUIRED_BINARIES_LINUX
    if sys.platform == "darwin":
        return _REQUIRED_BINARIES_DARWIN
    raise SandboxRuntimeError(
        f"sandbox-runtime is not supported on platform {sys.platform!r}; "
        "only Linux and macOS are supported."
    )


def _install_hint() -> str:
    if sys.platform.startswith("linux"):
        return (
            "Install with:\n"
            "  npm install -g @anthropic-ai/sandbox-runtime@0.0.23\n"
            "  sudo apt install ripgrep bubblewrap socat"
        )
    if sys.platform == "darwin":
        return (
            "Install with:\n"
            "  npm install -g @anthropic-ai/sandbox-runtime@0.0.23\n"
            "  brew install ripgrep"
        )
    return "(unsupported platform)"


def _check_sandbox_dependencies(force: bool = False) -> None:
    """Verify ``srt`` and its required system tools are installed.

    Idempotent: succeeds without re-checking after the first successful call
    (process-wide cache). Pass ``force=True`` to re-run the check (used in tests).

    Raises:
        SandboxRuntimeError: if any required binary is missing on PATH, or if
            the platform is unsupported. The error message includes the install
            command for the current OS.
    """
    global _DEPS_VERIFIED
    if _DEPS_VERIFIED and not force:
        return

    required = _required_binaries()
    missing = [b for b in required if shutil.which(b) is None]
    if missing:
        raise SandboxRuntimeError(
            "Cannot use the agentic-shell sandbox: required binaries are not "
            f"installed: {', '.join(missing)}.\n\n"
            f"{_install_hint()}\n\n"
            "See src/tau2/knowledge/README.md for full setup. "
            "If you don't need the shell tool, switch to a retrieval config "
            "that doesn't require it (e.g., --retrieval-config bm25 or "
            "openai_embeddings)."
        )

    _DEPS_VERIFIED = True


# Static metadata for sanitized output
_SANDBOX_USER = "kb_user"
_SANDBOX_GROUP = "kb_group"
_SANDBOX_DATE = "Jan  1 00:00"


class SandboxManager:
    """Manages sandboxed file system environments for agentic search.

    WARNING: Best-effort sandboxing only. The agent may escape read restrictions.
    Write access is properly sandboxed; read access is NOT strictly enforced.

    Each sandbox:
    - Gets its own isolated temp directory with KB documents exported as files
    - Blocks obvious escape patterns (../, ~, absolute paths, $HOME)
    - Uses srt for OS-level protection of some sensitive paths
    - Properly restricts write access (if allow_writes=False)

    What IS enforced:
    - Write access is blocked unless allow_writes=True (srt allow-only pattern)
    - Network access is blocked
    - Some sensitive paths are denied (~/.ssh, /etc/passwd, etc.)

    What is NOT strictly enforced:
    - Read access to arbitrary filesystem paths (srt uses deny-only for reads)
    - Sophisticated escape attempts may bypass pattern checks

    Requires: sandbox-runtime (srt) must be installed globally via npm.
    """

    def __init__(
        self,
        allow_writes: bool = False,
        sandbox_id: Optional[str] = None,
        base_temp_dir: Optional[str] = None,
    ):
        """Initialize a sandbox manager.

        Args:
            allow_writes: If True, the agent can modify files in the sandbox.
                         If False, all write operations are blocked.
            sandbox_id: Optional unique identifier for this sandbox.
                       If not provided, a UUID will be generated.
            base_temp_dir: Optional base directory for creating sandboxes.
                          Defaults to system temp directory.

        Raises:
            SandboxRuntimeError: If sandbox-runtime (``srt``) or any of its
                required system dependencies (``rg``, ``bwrap``, ``socat`` on
                Linux; ``rg`` on macOS) is not installed. We fail loudly here
                rather than letting every shell invocation silently return
                "Sandbox dependencies are not available".
        """
        _check_sandbox_dependencies()
        self.sandbox_id = sandbox_id or str(uuid.uuid4())[:8]
        self.allow_writes = allow_writes
        self.base_temp_dir = base_temp_dir or tempfile.gettempdir()

        # Create the sandbox directory
        self.sandbox_dir = (
            Path(self.base_temp_dir) / f"agentic_search_{self.sandbox_id}"
        )
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectory for knowledge base files
        self.kb_dir = self.sandbox_dir / "knowledge_base"
        self.kb_dir.mkdir(exist_ok=True)

        # Create settings file for srt
        self.settings_path = self.sandbox_dir / "srt-settings.json"
        self._create_srt_settings()

    def _create_srt_settings(self) -> None:
        """Create the srt-settings.json configuration file.

        NOTE: srt's read restrictions use a "deny-only" pattern, meaning we can only
        block specific paths, not restrict reads to only kb_dir. Write restrictions
        use "allow-only" and ARE properly enforced.
        """
        settings = {
            "network": {
                "allowedDomains": [],  # No network access
                "deniedDomains": [],
            },
            "filesystem": {
                # Best-effort: block some sensitive paths (but agent may still read elsewhere)
                "denyRead": [
                    "~/.ssh",
                    "~/.aws",
                    "~/.config",
                    "~/.gnupg",
                    "/etc/passwd",
                    "/etc/shadow",
                ],
                "allowWrite": [str(self.kb_dir)] if self.allow_writes else [],
                "denyWrite": [str(self.settings_path)],
            },
        }

        with open(self.settings_path, "w") as f:
            json.dump(settings, f, indent=2)

    def _has_escape_pattern(self, command: str) -> Optional[str]:
        """Check if command contains obvious patterns that could escape the sandbox.

        NOTE: This is best-effort only. Sophisticated escape attempts may bypass these checks.

        Returns the problematic pattern if found, None if no obvious escape pattern detected.
        """
        import re

        stripped = re.sub(r"'[^']*'", '""', command)
        stripped = re.sub(r'"[^"]*"', '""', stripped)

        # Patterns that allow escaping the working directory
        if ".." in stripped:
            return ".."

        abs_path_match = re.search(r"(?:^|[\s;|&])(/[a-zA-Z])", stripped)
        if abs_path_match:
            full_match = re.search(r"(?:^|[\s;|&])(/[a-zA-Z][^\s;|&]*)", stripped)
            if full_match:
                return f"absolute path: {full_match.group(1)}"

        if stripped.strip().startswith("/"):
            return "absolute path at start"

        if "~" in stripped:
            return "~"
        if "$HOME" in stripped or "${HOME}" in stripped:
            return "$HOME"

        return None

    def _sanitize_output(self, output: str, command: str) -> str:
        """Sanitize command output to remove real user metadata.

        Replaces:
        - Real username/group with generic values in ls output
        - Real timestamps with static date in ls output
        - Real username in paths (e.g., /Users/username/...)

        Args:
            output: The raw command output
            command: The command that was run (to determine sanitization strategy)

        Returns:
            Sanitized output string
        """
        if not output:
            return output

        result = output

        # Get the real username to replace
        try:
            real_user = getpass.getuser()
        except Exception:
            real_user = None

        # Sanitize ls -l style output (matches: "drwxr-xr-x  5 username  group  160 Jan 11 21:44")
        # Pattern matches: permissions, links, user, group, size, date, name
        ls_pattern = re.compile(
            r"^([d\-lrwxst@]+\s+\d+\s+)"  # permissions and link count
            r"(\S+)\s+"  # username (capture group 2)
            r"(\S+)\s+"  # group (capture group 3)
            r"(\d+)\s+"  # size
            r"(\w{3}\s+\d+\s+[\d:]+)\s+"  # date/time (capture group 5)
            r"(.*)$",  # filename
            re.MULTILINE,
        )

        def replace_ls_line(match):
            perms_links = match.group(1)
            size = match.group(4)
            filename = match.group(6)
            return f"{perms_links}{_SANDBOX_USER}  {_SANDBOX_GROUP}  {size} {_SANDBOX_DATE} {filename}"

        result = ls_pattern.sub(replace_ls_line, result)

        # Also sanitize any paths that contain the real username
        if real_user:
            # Replace /Users/username or /home/username patterns
            result = re.sub(
                rf"/(?:Users|home)/{re.escape(real_user)}(?=/|$)",
                "/home/kb_user",
                result,
            )

        return result

    def export_documents(
        self,
        documents: List[Dict[str, Any]],
        file_format: str = "txt",
    ) -> Dict[str, Path]:
        """Export knowledge base documents to the sandbox directory.

        Args:
            documents: List of documents, each with 'id', 'title', and 'content' keys
            file_format: File format to use: 'txt', 'md', or 'json'

        Returns:
            Dict mapping document IDs to their file paths
        """
        exported_files = {}

        for doc in documents:
            doc_id = doc.get("id", "unknown")
            title = doc.get("title", doc_id)
            content = doc.get("content", "")

            # Create safe filename from doc_id
            safe_filename = self._sanitize_filename(doc_id)

            if file_format == "json":
                file_path = self.kb_dir / f"{safe_filename}.json"
                with open(file_path, "w") as f:
                    json.dump(
                        {"id": doc_id, "title": title, "content": content}, f, indent=2
                    )
            elif file_format == "md":
                file_path = self.kb_dir / f"{safe_filename}.md"
                with open(file_path, "w") as f:
                    f.write(f"# {title}\n\n{content}")
            else:  # txt
                file_path = self.kb_dir / f"{safe_filename}.txt"
                with open(file_path, "w") as f:
                    f.write(f"Title: {title}\n{'=' * 50}\n\n{content}")

            exported_files[doc_id] = file_path

        # Create an index file listing all documents
        self._create_index_file(documents, file_format)

        return exported_files

    def _create_index_file(
        self, documents: List[Dict[str, Any]], file_format: str = "md"
    ) -> None:
        """Create an index file listing all documents in the knowledge base."""
        # Use same extension as documents
        ext = file_format if file_format != "json" else "md"
        index_path = self.kb_dir / f"INDEX.{ext}"

        with open(index_path, "w") as f:
            if ext == "md":
                f.write("# Knowledge Base Index\n\n")
                f.write(f"**Total documents:** {len(documents)}\n\n")
                f.write("## Documents\n\n")

                for doc in sorted(documents, key=lambda d: d.get("id", "")):
                    doc_id = doc.get("id", "unknown")
                    title = doc.get("title", doc_id)
                    safe_filename = self._sanitize_filename(doc_id)
                    f.write(f"- **{safe_filename}.{ext}** - {title}\n")
            else:  # txt
                f.write("KNOWLEDGE BASE INDEX\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Total documents: {len(documents)}\n\n")
                f.write("Documents:\n")
                f.write("-" * 50 + "\n")

                for doc in sorted(documents, key=lambda d: d.get("id", "")):
                    doc_id = doc.get("id", "unknown")
                    title = doc.get("title", doc_id)
                    safe_filename = self._sanitize_filename(doc_id)
                    f.write(f"\n{safe_filename}.{ext}\n")
                    f.write(f"  Title: {title}\n")

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize a string to be used as a filename."""
        safe = filename.replace("/", "_").replace("\\", "_").replace("..", "_")
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in safe)
        return safe[:255]

    def run_command(self, command: str, timeout: int = 30) -> Tuple[int, str, str]:
        """Run a shell command inside the sandbox.

        Args:
            command: The shell command to execute
            timeout: Maximum execution time in seconds

        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        # Check for obvious escape patterns
        escape_pattern = self._has_escape_pattern(command)
        if escape_pattern:
            return (
                1,
                "",
                f"Error: Command blocked - contains '{escape_pattern}' which could escape the sandbox",
            )

        # Execute via srt with working directory set to kb_dir
        srt_command = ["srt", "--settings", str(self.settings_path), command]

        try:
            result = subprocess.run(
                srt_command,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.kb_dir),
            )
            # Sanitize output to remove real user metadata
            sanitized_stdout = self._sanitize_output(result.stdout, command)
            sanitized_stderr = self._sanitize_output(result.stderr, command)

            # If srt itself complains about missing system deps, that's an
            # install-time failure of the host machine -- not something the
            # agent should learn to "work around". Raise loudly.
            combined = (sanitized_stdout + sanitized_stderr).lower()
            if "sandbox dependencies are not available" in combined:
                raise SandboxRuntimeError(
                    "sandbox-runtime reported missing system dependencies "
                    "while executing a command. This is a host-level install "
                    "failure -- the agent never had a working shell tool. "
                    "Re-check the dep verification on this machine.\n\n"
                    f"srt stdout: {sanitized_stdout.strip()[:400]}\n"
                    f"srt stderr: {sanitized_stderr.strip()[:400]}\n\n"
                    f"{_install_hint()}"
                )

            return (result.returncode, sanitized_stdout, sanitized_stderr)
        except subprocess.TimeoutExpired:
            return (124, "", f"Command timed out after {timeout} seconds")
        except FileNotFoundError:
            # ``srt`` was on PATH at construction but vanished mid-run, or its
            # interpreter died. Surface as a structured error.
            raise SandboxRuntimeError(
                "sandbox-runtime (srt) binary disappeared between sandbox "
                "construction and command execution. " + _install_hint()
            )
        except SandboxRuntimeError:
            raise
        except Exception as e:
            return (1, "", f"Command failed: {str(e)}")

    def list_files(self) -> List[str]:
        """List all files in the knowledge base directory."""
        return sorted([f.name for f in self.kb_dir.iterdir() if f.is_file()])

    def get_kb_path(self) -> str:
        """Get the path to the knowledge base directory."""
        return str(self.kb_dir)

    def get_sandbox_info(self) -> Dict[str, Any]:
        """Get information about the sandbox configuration."""
        return {
            "sandbox_id": self.sandbox_id,
            "sandbox_dir": str(self.sandbox_dir),
            "kb_dir": str(self.kb_dir),
            "allow_writes": self.allow_writes,
            "num_files": len(self.list_files()),
        }

    def cleanup(self) -> None:
        """Remove the sandbox directory and all its contents."""
        if self.sandbox_dir.exists():
            shutil.rmtree(self.sandbox_dir)

    def __enter__(self) -> "SandboxManager":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - cleanup the sandbox."""
        self.cleanup()


class SandboxPool:
    """Pool of sandboxes for parallel agent execution."""

    def __init__(
        self,
        max_sandboxes: int = 10,
        allow_writes: bool = False,
        base_temp_dir: Optional[str] = None,
    ):
        self.max_sandboxes = max_sandboxes
        self.allow_writes = allow_writes
        self.base_temp_dir = base_temp_dir
        self._sandboxes: Dict[str, SandboxManager] = {}

    def acquire(self, sandbox_id: Optional[str] = None) -> SandboxManager:
        """Acquire a sandbox from the pool."""
        if len(self._sandboxes) >= self.max_sandboxes:
            raise RuntimeError(f"Sandbox pool is full (max: {self.max_sandboxes})")

        sandbox = SandboxManager(
            allow_writes=self.allow_writes,
            sandbox_id=sandbox_id,
            base_temp_dir=self.base_temp_dir,
        )
        self._sandboxes[sandbox.sandbox_id] = sandbox
        return sandbox

    def release(self, sandbox_id: str) -> None:
        """Release a sandbox back to the pool."""
        if sandbox_id in self._sandboxes:
            self._sandboxes[sandbox_id].cleanup()
            del self._sandboxes[sandbox_id]

    def cleanup_all(self) -> None:
        """Clean up all sandboxes in the pool."""
        for sandbox in list(self._sandboxes.values()):
            sandbox.cleanup()
        self._sandboxes.clear()

    def __enter__(self) -> "SandboxPool":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup_all()
