"""Sandbox manager for agentic search.

This module provides utilities for creating sandboxed environments where agents
can interact with knowledge base documents through shell commands (cat, grep, etc.).

Read isolation:
srt's read restrictions are "deny-only" (``filesystem.denyRead``; there is no
allow-list for reads). We emulate an allow-list: starting from ``/``, every
directory entry that is neither inside an allowed path nor an ancestor of one is
denied, and ancestors are walked recursively (see ``_compute_read_denies``).
Allowed paths are the knowledge base copy, a small set of system directories
needed to run standard tools (cat, grep, rg, find, python3, ...), and, only when
no python3 is available from those directories, the install prefix of the
running interpreter. The deny list is recomputed before every command so files
created after the sandbox started are covered too. Sandboxes live under a
dedicated per-user root (``/var/tmp/tau2-kb-sandboxes-<uid>`` by default) so the
walk does not have to enumerate a large shared temp directory.

The command pattern filter (``_has_escape_pattern``) is a convenience, NOT a
security boundary: it ignores quoted text, so e.g.
``python3 -c "print(open('/abs/path').read())"`` passes it. The srt read policy
is what keeps the agent inside the knowledge base.

Limitations:
- Directory *names* along the path to an allowed directory (e.g. the entries of
  ``/`` or of the sandbox root) can be listed; their contents cannot be read.
- Allowed system directories are readable in full.
- Write access is restricted via srt's "allow-only" pattern.

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
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
import weakref
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


# Environment variables passed through to ``srt`` and the sandboxed command.
#
# The sandboxed command must NOT inherit the parent environment: tau2 loads
# provider API keys from ``.env`` into ``os.environ``, and an agent calling the
# shell tool with ``env`` would otherwise receive every key in its tool result
# (which is then saved to trajectories and logs). Only what ``srt`` (a node
# script) and the read-only search tools (cat, ls, grep, rg, find, python3, ...)
# need is kept:
# - PATH: to resolve ``node`` (srt's shebang), the shell, and the search tools.
# - LANG / LC_*: so tools handle non-ASCII document text consistently.
# - TERM, TZ: harmless formatting hints for tools.
# - TMPDIR: srt on macOS derives its allowed per-user temp dirs from it.
# HOME is not inherited; it is set to the sandbox directory (see
# ``SandboxManager.run_command``). Anything whose name looks secret-like is
# dropped even if it appears here.
_SANDBOX_ENV_ALLOWLIST: tuple[str, ...] = ("PATH", "LANG", "TERM", "TZ", "TMPDIR")
_SANDBOX_ENV_ALLOWED_PREFIXES: tuple[str, ...] = ("LC_",)

# Env var names that are treated as secrets: never passed into the sandbox, and
# their values are redacted from command output as defense in depth.
_SECRET_ENV_NAME_RE = re.compile(
    r"KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|PRIVATE|AUTH", re.IGNORECASE
)
# Shorter values are too likely to collide with ordinary output text.
_MIN_REDACT_LEN = 8
_REDACTED = "[REDACTED]"


def _build_sandbox_env(home: str) -> Dict[str, str]:
    """Return the minimal allow-listed environment for a sandboxed command."""
    env = {
        name: value
        for name, value in os.environ.items()
        if (
            name in _SANDBOX_ENV_ALLOWLIST
            or name.startswith(_SANDBOX_ENV_ALLOWED_PREFIXES)
        )
        and not _SECRET_ENV_NAME_RE.search(name)
    }
    env["HOME"] = home
    return env


def _redact_secret_env_values(text: str) -> str:
    """Replace values of secret-like parent env vars in ``text`` with a marker."""
    if not text:
        return text
    secrets = {
        value
        for name, value in os.environ.items()
        if _SECRET_ENV_NAME_RE.search(name) and len(value) >= _MIN_REDACT_LEN
    }
    # Longest first so a secret containing another is replaced whole.
    for value in sorted(secrets, key=len, reverse=True):
        text = text.replace(value, _REDACTED)
    return text


# --- Read policy ---------------------------------------------------------------
#
# System directories the sandboxed shell may read. Everything else on the
# machine is denied except the knowledge base itself (see
# ``_compute_read_denies``). Paths that do not exist are ignored.
_SYSTEM_READ_ALLOW_DARWIN: tuple[str, ...] = (
    "/bin",
    "/sbin",
    "/usr",
    "/dev",
    "/Library",
    "/System/Library",
    "/System/Cryptexes",
    "/System/DriverKit",
    "/System/iOSSupport",
    # dyld shared cache / cryptex OS content.
    "/System/Volumes/Preboot",
    "/private/etc",
    "/private/var/select",
    "/private/var/db/timezone",
    "/private/var/db/dyld",
    "/opt/homebrew",
)
_SYSTEM_READ_ALLOW_LINUX: tuple[str, ...] = (
    "/bin",
    "/sbin",
    "/usr",
    "/lib",
    "/lib32",
    "/lib64",
    "/libx32",
    "/etc",
    "/dev",
    "/proc",
    "/sys",
    "/nix/store",
)

# python3 launchers that do not work under the read policy. macOS's
# /usr/bin/python3 is an xcrun shim that writes a cache in the (denied) per-user
# temp dir and prints an error on every call.
_UNUSABLE_PYTHONS: frozenset[Path] = frozenset(
    {Path("/usr/bin/python3")} if sys.platform == "darwin" else set()
)

# Always denied, even inside an allowed system directory.
_SENSITIVE_READ_DENY: tuple[str, ...] = ("/etc/passwd", "/etc/shadow")


def _system_read_allow() -> tuple[str, ...]:
    if sys.platform == "darwin":
        return _SYSTEM_READ_ALLOW_DARWIN
    return _SYSTEM_READ_ALLOW_LINUX


def _real(path: os.PathLike | str) -> Path:
    return Path(os.path.realpath(path))


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _compute_read_denies(allowed: List[Path], root: Path = Path("/")) -> List[str]:
    """Emulate a read allow-list with srt's deny-only ``denyRead``.

    Walks the filesystem from ``/`` and denies every entry that is neither
    inside an ``allowed`` path nor an ancestor of one; ancestors are walked
    recursively. Symlinks are skipped: srt (and macOS seatbelt) match the
    resolved path, which the walk reaches through its real location.

    Args:
        allowed: Real (symlink-resolved) paths that must stay readable.
        root: Where the walk starts (``/``; overridable for tests).

    Returns:
        Sorted absolute paths to pass as ``filesystem.denyRead``.
    """
    allowed = [a for a in allowed if a.exists()]
    denies: List[str] = []

    def visit(directory: Path) -> None:
        # Child names on the way to an allowed path, and whether that child is
        # itself allowed (keep whole) or only an ancestor (walk into it).
        on_path: Dict[str, bool] = {}
        for a in allowed:
            if directory in a.parents:
                rel = a.relative_to(directory)
                name = rel.parts[0]
                on_path[name] = on_path.get(name, False) or len(rel.parts) == 1
        try:
            entries = list(os.scandir(directory))
        except OSError:
            # Unlistable by us; its children are unreadable without srt too.
            return
        for entry in entries:
            if entry.name in on_path:
                if not on_path[entry.name] and not entry.is_symlink():
                    visit(directory / entry.name)
            elif not entry.is_symlink():
                denies.append(entry.path)

    if not any(_is_within(root, a) for a in allowed):
        visit(root)
    return sorted(denies)


def _default_sandbox_root() -> Path:
    """Return a private, per-user directory to hold sandboxes.

    A dedicated directory keeps the deny-list walk small: the shared temp dir
    can hold tens of thousands of entries (other runs' logs, other tools'
    scratch files), and every one of them would otherwise need a deny rule.
    """
    name = f"tau2-kb-sandboxes-{os.getuid()}"
    for base in ("/var/tmp", tempfile.gettempdir()):
        root = Path(base) / name
        try:
            root.mkdir(mode=0o700, exist_ok=True)
            st = root.lstat()
        except OSError:
            continue
        if (
            not root.is_symlink()
            and root.is_dir()
            and st.st_uid == os.getuid()
            and os.access(root, os.W_OK)
        ):
            return root
    raise SandboxRuntimeError(
        "Could not create a private sandbox root under /var/tmp or "
        f"{tempfile.gettempdir()}."
    )


def _extra_read_denies() -> List[Path]:
    """Locations holding task data that must be denied even if a system
    directory that contains them is allowed (e.g. a container that installs the
    repository under /usr/src)."""
    paths = [Path.cwd()]
    if sys.prefix != sys.base_prefix:  # a virtualenv, not the base interpreter
        paths.append(Path(sys.prefix))
    try:
        from tau2.utils.utils import DATA_DIR

        paths.append(Path(DATA_DIR))
    except Exception:
        pass
    package_dir = Path(__file__).resolve().parents[1]  # .../tau2
    paths.append(package_dir)
    if package_dir.parent.name == "src":  # source checkout: deny the whole repo
        paths.append(package_dir.parent.parent)
    return [_real(p) for p in paths]


def _build_read_denies(allowed: List[Path]) -> List[str]:
    """Full ``denyRead`` list: the allow-list walk plus targeted denies."""
    denies = _compute_read_denies(allowed)
    covered = [Path(d) for d in denies]
    for extra in [*_extra_read_denies(), *map(_real, _SENSITIVE_READ_DENY)]:
        if not extra.exists():
            continue
        # Never deny an allowed path or one of its ancestors (on Linux this
        # would mount an empty tmpfs over the knowledge base).
        if any(_is_within(a, extra) for a in allowed):
            continue
        # Already denied by the walk; a nested deny is redundant.
        if any(_is_within(extra, c) for c in covered):
            continue
        denies.append(str(extra))
        covered.append(extra)
    return sorted(set(denies))


# Static metadata for sanitized output
_SANDBOX_USER = "kb_user"
_SANDBOX_GROUP = "kb_group"
_SANDBOX_DATE = "Jan  1 00:00"


class SandboxManager:
    """Manages sandboxed file system environments for agentic search.

    Each sandbox:
    - Gets its own directory under a private per-user sandbox root, with KB
      documents exported as files
    - Blocks obvious escape patterns (../, ~, absolute paths, $HOME) as a
      convenience; this filter is not a security boundary
    - Uses srt to deny reads of everything except the knowledge base and the
      system directories standard tools need (see module docstring)
    - Restricts write access (if allow_writes=False, nothing is writable)

    What IS enforced (by srt, at the OS level):
    - Reads outside the knowledge base and allowed system directories fail
    - Write access is blocked unless allow_writes=True (srt allow-only pattern)
    - Network access is blocked

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
                          Defaults to a private per-user root
                          (``/var/tmp/tau2-kb-sandboxes-<uid>``, falling back
                          to the system temp dir). Every sibling entry along
                          the path to the sandbox gets its own deny rule, so
                          prefer a directory with few entries.

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
        self.base_temp_dir = str(
            Path(base_temp_dir) if base_temp_dir else _default_sandbox_root()
        )

        # Create the sandbox directory. Resolve symlinks (e.g. macOS
        # /var -> /private/var) so the read policy sees real paths.
        sandbox_dir = Path(self.base_temp_dir) / f"agentic_search_{self.sandbox_id}"
        sandbox_dir.mkdir(parents=True, exist_ok=True)
        self.sandbox_dir = _real(sandbox_dir)
        # Sandboxes created by the retrieval toolkits are never cleaned up
        # explicitly; remove the directory once the manager is garbage
        # collected or at interpreter exit. Leftover sandboxes would otherwise
        # pile up in the shared root (each needs its own deny rule).
        self._finalizer = weakref.finalize(
            self, shutil.rmtree, str(self.sandbox_dir), ignore_errors=True
        )

        # Create subdirectory for knowledge base files
        self.kb_dir = self.sandbox_dir / "knowledge_base"
        self.kb_dir.mkdir(exist_ok=True)

        # Holds a python3 link when no python3 is available from the allowed
        # system directories. Readable from inside the sandbox; not writable.
        self.bin_dir = self.sandbox_dir / "bin"

        # Settings for srt. They live next to (not inside) kb_dir, so the read
        # policy denies them to the sandboxed command.
        self.settings_path = self.sandbox_dir / "srt-settings.json"

        self._read_allow, self._sandbox_path = self._plan_read_access()
        self._create_srt_settings()

    def _plan_read_access(self) -> Tuple[List[Path], str]:
        """Choose the readable paths and the PATH for the sandboxed command.

        Returns:
            (allowed real paths, PATH string). PATH keeps the parent PATH
            entries that remain readable under the policy, plus the
            directories of srt's own dependencies (srt runs unsandboxed with
            this same environment and must find node, rg, bwrap, socat).
        """
        allowed = [self.kb_dir]
        allowed += [_real(p) for p in _system_read_allow() if os.path.exists(p)]

        parent_path = os.environ.get("PATH", os.defpath).split(os.pathsep)
        readable = [
            d
            for d in parent_path
            if d and any(_is_within(_real(d), a) for a in allowed)
        ]

        pythons = [
            os.path.join(d, "python3")
            for d in readable
            if os.access(os.path.join(d, "python3"), os.X_OK)
        ]
        usable = [p for p in pythons if _real(p) not in _UNUSABLE_PYTHONS]
        if not pythons or pythons[0] not in usable:
            if usable:
                target = _real(usable[0])
            else:
                # No usable python3 in an allowed system directory (e.g. only
                # a pyenv shim or a uv-managed interpreter under the home
                # directory). Allow just the running interpreter's install
                # prefix.
                target = _real(getattr(sys, "_base_executable", None) or sys.executable)
                prefix = _real(sys.base_prefix)
                allowed.append(prefix)
                if not _is_within(target, prefix):
                    allowed.append(target)
            # Expose the chosen interpreter as the first ``python3`` on PATH.
            self.bin_dir.mkdir(exist_ok=True)
            link = self.bin_dir / "python3"
            if not link.is_symlink():
                link.symlink_to(target)
            allowed.append(self.bin_dir)
            readable.insert(0, str(self.bin_dir))

        for tool in ("node", *_required_binaries()):
            found = shutil.which(tool)
            if found and os.path.dirname(found) not in readable:
                readable.append(os.path.dirname(found))

        return allowed, os.pathsep.join(readable)

    def _create_srt_settings(self) -> None:
        """Write the srt-settings.json configuration file.

        Called again before every command so the deny list covers files that
        appeared on the machine after the sandbox was created.
        """
        # Must exist before the walk so the settings file itself is denied.
        self.settings_path.touch(exist_ok=True)
        settings = {
            "network": {
                "allowedDomains": [],  # No network access
                "deniedDomains": [],
            },
            "filesystem": {
                # Everything outside the knowledge base and the allowed system
                # directories (see ``_compute_read_denies``). Paths are
                # absolute: srt would expand "~" via HOME, which the sandboxed
                # process sets to the sandbox dir.
                "denyRead": _build_read_denies(self._read_allow),
                "allowWrite": [str(self.kb_dir)] if self.allow_writes else [],
                "denyWrite": [str(self.settings_path)],
            },
        }

        tmp_path = self.settings_path.with_suffix(".json.tmp")
        with open(tmp_path, "w") as f:
            json.dump(settings, f, indent=2)
        os.replace(tmp_path, self.settings_path)

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
        - Values of secret-like parent env vars (API keys, tokens) with
          "[REDACTED]" (defense in depth; the sandbox env excludes them)

        Args:
            output: The raw command output
            command: The command that was run (to determine sanitization strategy)

        Returns:
            Sanitized output string
        """
        if not output:
            return output

        result = _redact_secret_env_values(output)

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

        # Never inherit os.environ: it holds provider API keys.
        env = _build_sandbox_env(home=str(self.sandbox_dir))
        # Only PATH entries that stay readable under the read policy (plus the
        # directories srt itself needs).
        env["PATH"] = self._sandbox_path

        try:
            # Refresh the deny list so it covers entries created since the
            # last command (e.g. another sandbox or log next to an allowed
            # directory).
            self._create_srt_settings()
            result = subprocess.run(
                srt_command,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.kb_dir),
                env=env,
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
        self._finalizer()

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
