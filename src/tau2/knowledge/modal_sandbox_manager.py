"""Modal-backed sandbox manager for the banking knowledge shell.

Documents are staged locally and copied into a Modal Image only when the first
shell command is executed.  This keeps retrieval construction local and avoids
creating a billable remote sandbox for tasks that never use the shell tool.

The sandbox receives no application secrets or local environment variables.
Its network is disabled and commands run from ``/knowledge_base``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tarfile
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tau2.knowledge.sandbox_manager import SandboxManager

logger = logging.getLogger(__name__)

DEFAULT_MODAL_APP = "tau3-banking-sandboxes"
DEFAULT_MODAL_SANDBOX_TIMEOUT = 3600
MODAL_APP_ENV = "TAU2_MODAL_APP"
MODAL_SANDBOX_TIMEOUT_ENV = "TAU2_MODAL_SANDBOX_TIMEOUT"
MODAL_ORDER_MANIFEST_ENV = "TAU2_MODAL_ORDER_MANIFEST"
MODAL_EXPECTED_IMAGE_ID_ENV = "TAU2_MODAL_EXPECTED_IMAGE_ID"
REMOTE_KB_DIR = "/knowledge_base"
REMOTE_KB_ARCHIVE = "/tmp/knowledge_base.tar"
REMOTE_KB_TMPFS_DIR = "/dev/shm/tau3_knowledge_base"
BASH_EXECUTABLE = "/usr/bin/bash"
MODAL_PIP_PACKAGES = ("scipy==1.16.3",)
MODAL_TERMINATION_ATTEMPTS = 3
DEFAULT_ORDER_MANIFEST = (
    Path(__file__).resolve().parents[3]
    / "reproduction"
    / "tau3_banking"
    / "full_shell_order_manifest.json"
)

ORDER_MANIFEST_SCHEMA_VERSION = 1
MODAL_IMAGE_RECIPE = {
    "base": "modal.Image.debian_slim",
    "pip_install": list(MODAL_PIP_PACKAGES),
    "run_commands": [f"mkdir -p {REMOTE_KB_DIR}"],
    "workdir": REMOTE_KB_DIR,
    "block_network": True,
    "runtime_uid": 65534,
    "runtime_gid": 65534,
}
MODAL_IMAGE_RECIPE_SHA256 = hashlib.sha256(
    json.dumps(
        MODAL_IMAGE_RECIPE,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
).hexdigest()

# Modal containers run as root.  In read-only mode, launch the agent command
# after permanently dropping to Debian's conventional ``nobody`` user/group.
# The exported KB is world-readable but root-owned in the image, so this keeps
# it readable without allowing the command to modify it.
_DROP_PRIVILEGES = (
    "import os,sys;"
    "os.setgroups([]);"
    "os.setgid(65534);"
    "os.setuid(65534);"
    "os.execvp(sys.argv[1],sys.argv[1:])"
)


class ModalSandboxRuntimeError(RuntimeError):
    """Raised when the Modal backend cannot create or use its remote sandbox."""


def _load_modal() -> Any:
    """Import Modal on first remote use, not while constructing retrieval."""
    try:
        import modal
    except ImportError as exc:
        raise ModalSandboxRuntimeError(
            "The Modal sandbox backend requires the 'modal' Python package."
        ) from exc
    return modal


def _sandbox_timeout_from_env() -> int:
    """Return the configured remote sandbox lifetime in seconds."""
    raw_timeout = os.environ.get(
        MODAL_SANDBOX_TIMEOUT_ENV, str(DEFAULT_MODAL_SANDBOX_TIMEOUT)
    )
    try:
        timeout = int(raw_timeout)
    except ValueError as exc:
        raise ValueError(
            f"{MODAL_SANDBOX_TIMEOUT_ENV} must be a positive integer"
        ) from exc
    if timeout <= 0:
        raise ValueError(f"{MODAL_SANDBOX_TIMEOUT_ENV} must be a positive integer")
    return timeout


def _safe_temp_component(value: str) -> str:
    """Make an identifier safe to use in a temporary-directory prefix."""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in value)
    return safe[:64] or "sandbox"


def _ordered_file_digest(paths_by_name: dict[str, Path]) -> str:
    """Digest staged names and bytes independently of filesystem iteration order."""
    entries = []
    for name, path in sorted(paths_by_name.items()):
        payload = path.read_bytes()
        entries.append(
            {
                "name": name,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    canonical = json.dumps(entries, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _order_digest(filenames: list[str]) -> str:
    """Digest the newline-delimited order used by tar and tmpfs materialization."""
    return hashlib.sha256(("\n".join(filenames) + "\n").encode("utf-8")).hexdigest()


def _load_order_manifest(path: Path) -> dict[str, Any]:
    """Load and validate a disclosed shell-order compatibility fixture."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModalSandboxRuntimeError(
            f"Cannot read Modal order manifest {path}: {type(exc).__name__}"
        ) from exc
    if not isinstance(value, dict):
        raise ModalSandboxRuntimeError("Modal order manifest must be a JSON object")
    if value.get("schema_version") != ORDER_MANIFEST_SCHEMA_VERSION:
        raise ModalSandboxRuntimeError("Unsupported Modal order manifest schema")
    filenames = value.get("filenames")
    if (
        not isinstance(filenames, list)
        or not filenames
        or not all(isinstance(name, str) and name for name in filenames)
        or len(filenames) != len(set(filenames))
    ):
        raise ModalSandboxRuntimeError(
            "Modal order manifest filenames must be non-empty and unique"
        )
    if value.get("entry_count") != len(filenames):
        raise ModalSandboxRuntimeError("Modal order manifest entry count is invalid")
    if value.get("order_sha256") != _order_digest(filenames):
        raise ModalSandboxRuntimeError("Modal order manifest checksum mismatch")
    corpus_sha256 = value.get("corpus_export_sha256")
    if not isinstance(corpus_sha256, str) or len(corpus_sha256) != 64:
        raise ModalSandboxRuntimeError(
            "Modal order manifest corpus checksum is invalid"
        )
    return value


class ModalSandboxManager(SandboxManager):
    """Manage an isolated Modal Sandbox containing knowledge-base documents.

    The public interface mirrors :class:`SandboxManager`, but no local
    ``sandbox-runtime`` binaries are required.  Modal itself is imported and a
    remote sandbox is created only on the first valid call to
    :meth:`run_command`.
    """

    def __init__(
        self,
        allow_writes: bool = False,
        sandbox_id: Optional[str] = None,
        base_temp_dir: Optional[str] = None,
    ):
        """Initialize local staging without making any Modal API calls.

        Args:
            allow_writes: Allow commands to modify the remote knowledge base.
            sandbox_id: Optional logical identifier for this sandbox.
            base_temp_dir: Optional parent directory for local document staging.
        """
        self.sandbox_id = sandbox_id or str(uuid.uuid4())[:8]
        self.allow_writes = allow_writes
        self.base_temp_dir = base_temp_dir or tempfile.gettempdir()
        self.modal_app_name = os.environ.get(MODAL_APP_ENV) or DEFAULT_MODAL_APP
        self.modal_timeout = _sandbox_timeout_from_env()
        expected_image_id = os.environ.get(MODAL_EXPECTED_IMAGE_ID_ENV)
        if expected_image_id is not None and not expected_image_id.strip():
            raise ValueError(f"{MODAL_EXPECTED_IMAGE_ID_ENV} must not be blank")
        self.expected_modal_image_id = expected_image_id
        configured_manifest = os.environ.get(MODAL_ORDER_MANIFEST_ENV)
        self.order_manifest_path = (
            Path(configured_manifest).expanduser()
            if configured_manifest
            else DEFAULT_ORDER_MANIFEST
        )
        self._order_manifest_required = configured_manifest is not None
        self._order_manifest_applied = False
        self._order_manifest_sha256: Optional[str] = None

        base_path = Path(self.base_temp_dir)
        base_path.mkdir(parents=True, exist_ok=True)
        temp_prefix = f"modal_agentic_search_{_safe_temp_component(self.sandbox_id)}_"
        self.sandbox_dir = Path(
            tempfile.mkdtemp(prefix=temp_prefix, dir=str(base_path))
        )
        self.kb_dir = self.sandbox_dir / "knowledge_base"
        self.kb_dir.mkdir()
        self.archive_path = self.sandbox_dir / "knowledge_base.tar"
        self._archive_members: list[Path] = []

        self._modal_sandbox: Any = None
        self._modal_image_id: Optional[str] = None
        self._remote_ready = False
        self._cleanup_pending = False
        self._closed = False
        self._lifecycle_lock = threading.RLock()

    def export_documents(
        self,
        documents: List[Dict[str, Any]],
        file_format: str = "txt",
    ) -> Dict[str, Path]:
        """Stage documents locally before the remote image is created."""
        with self._lifecycle_lock:
            self._ensure_open()
            if self._modal_sandbox is not None:
                raise ModalSandboxRuntimeError(
                    "Documents cannot be exported after the Modal sandbox has started"
                )

            # A previous export may have sealed these files for read-only use.
            if not self.allow_writes:
                self.kb_dir.chmod(0o755)
                for file_path in self.kb_dir.iterdir():
                    if file_path.is_file():
                        file_path.chmod(0o644)

            try:
                exported = super().export_documents(documents, file_format=file_format)
            finally:
                if not self.allow_writes:
                    for file_path in self.kb_dir.iterdir():
                        if file_path.is_file():
                            file_path.chmod(0o444)
                    self.kb_dir.chmod(0o555)

            index_ext = file_format if file_format != "json" else "md"
            self._archive_members = [
                *exported.values(),
                self.kb_dir / f"INDEX.{index_ext}",
            ]
            self._apply_order_manifest()

            return exported

    def _apply_order_manifest(self) -> None:
        """Apply the fixture only when the complete banking corpus matches."""
        path = self.order_manifest_path
        if not path.is_file():
            if self._order_manifest_required:
                raise ModalSandboxRuntimeError(
                    f"Configured Modal order manifest does not exist: {path}"
                )
            return

        manifest = _load_order_manifest(path)
        paths_by_name = {member.name: member for member in self._archive_members}
        filenames = manifest["filenames"]
        if set(paths_by_name) != set(filenames):
            # This manager is also used by focused tests and may be used by
            # future knowledge domains.  The trace-derived fixture is only
            # valid for the complete banking corpus plus INDEX.md.
            if self._order_manifest_required:
                raise ModalSandboxRuntimeError(
                    "The staged corpus filenames do not exactly match the "
                    "configured Modal order manifest"
                )
            return
        if _ordered_file_digest(paths_by_name) != manifest["corpus_export_sha256"]:
            raise ModalSandboxRuntimeError(
                "The banking corpus does not match the Modal order manifest"
            )
        self._archive_members = [paths_by_name[name] for name in filenames]
        self._order_manifest_applied = True
        self._order_manifest_sha256 = manifest["order_sha256"]

    def _ensure_open(self) -> None:
        if self._closed:
            raise ModalSandboxRuntimeError("The Modal sandbox manager is closed")
        if self._cleanup_pending:
            raise ModalSandboxRuntimeError(
                "The Modal sandbox is retained only for termination retry"
            )

    def _ensure_remote_sandbox(self) -> Any:
        """Create the remote sandbox exactly once, on first command use."""
        self._ensure_open()
        if self._modal_sandbox is not None:
            return self._modal_sandbox

        self._build_archive()
        modal = _load_modal()
        try:
            app = modal.App.lookup(self.modal_app_name, create_if_missing=True)
            image = (
                modal.Image.debian_slim()
                .pip_install(*MODAL_PIP_PACKAGES)
                .run_commands(f"mkdir -p {REMOTE_KB_DIR}")
                .add_local_file(
                    local_path=str(self.archive_path),
                    remote_path=REMOTE_KB_ARCHIVE,
                    copy=True,
                )
            )
            sandbox = modal.Sandbox.create(
                app=app,
                image=image,
                workdir=REMOTE_KB_DIR,
                block_network=True,
                timeout=self.modal_timeout,
            )
            image_id = image.object_id
            if not isinstance(image_id, str) or not image_id.strip():
                raise ModalSandboxRuntimeError(
                    "Modal did not expose a nonblank hydrated image object ID"
                )
            if (
                self.expected_modal_image_id is not None
                and image_id != self.expected_modal_image_id
            ):
                raise ModalSandboxRuntimeError(
                    "Hydrated Modal image object ID does not match the guarded run"
                )
            self._initialize_remote_files(sandbox)
            self._modal_image_id = image_id
            self._modal_sandbox = sandbox
            self._remote_ready = True
        except Exception as exc:
            sandbox = locals().get("sandbox")
            if sandbox is not None:
                try:
                    self._terminate_and_detach(sandbox)
                except ModalSandboxRuntimeError as termination_exc:
                    self._modal_sandbox = sandbox
                    self._remote_ready = False
                    self._cleanup_pending = True
                    raise ModalSandboxRuntimeError(
                        "Failed to create the Modal sandbox and could not prove "
                        "that the remote sandbox terminated"
                    ) from termination_exc
            raise ModalSandboxRuntimeError(
                "Failed to create the Modal sandbox "
                f"({type(exc).__name__}); verify Modal authentication and configuration"
            ) from exc

        return self._modal_sandbox

    @staticmethod
    def _terminate_and_detach(sandbox: Any) -> None:
        """Retry termination and detach only after remote shutdown is confirmed."""
        last_error: Exception | None = None
        for _ in range(MODAL_TERMINATION_ATTEMPTS):
            try:
                sandbox.terminate(wait=True)
            except Exception as exc:
                last_error = exc
            else:
                last_error = None
                break
        if last_error is not None:
            raise ModalSandboxRuntimeError(
                "Modal sandbox termination failed after "
                f"{MODAL_TERMINATION_ATTEMPTS} attempts"
            ) from last_error
        try:
            sandbox.detach()
        except Exception as exc:
            logger.warning("Modal sandbox detach failed (%s)", type(exc).__name__)

    def _build_archive(self) -> None:
        """Pack files in export order so recursive shell output stays stable."""
        members = self._archive_members or list(self.kb_dir.iterdir())
        with tarfile.open(self.archive_path, "w") as archive:
            for path in members:
                archive.add(
                    path,
                    arcname=path.name,
                    recursive=False,
                    filter=self._normalize_tar_info,
                )

    @staticmethod
    def _normalize_tar_info(info: tarfile.TarInfo) -> tarfile.TarInfo:
        """Remove host/time metadata so Modal can reuse one image layer."""
        info.uid = 0
        info.gid = 0
        info.uname = ""
        info.gname = ""
        info.mtime = 0
        # Match the Linux umask used by the official local sandbox. Commands
        # still cannot write in read-only mode: files are owned by root:root,
        # while agent commands permanently drop to uid/gid 65534.
        info.mode = 0o664
        return info

    def _initialize_remote_files(self, sandbox: Any) -> None:
        """Materialize the ordered archive on tmpfs before exposing the shell."""
        process = sandbox.exec(
            "bash",
            "-c",
            (
                f"mkdir -p {REMOTE_KB_TMPFS_DIR}"
                f" && tar -xf {REMOTE_KB_ARCHIVE} -C {REMOTE_KB_TMPFS_DIR}"
                f" && chown -R 0:0 {REMOTE_KB_TMPFS_DIR}"
                f" && find {REMOTE_KB_TMPFS_DIR} -type d -exec chmod 775 {{}} +"
                f" && find {REMOTE_KB_TMPFS_DIR} -type f -exec chmod 664 {{}} +"
                f" && rmdir {REMOTE_KB_DIR}"
                f" && ln -s {REMOTE_KB_TMPFS_DIR} {REMOTE_KB_DIR}"
            ),
            timeout=30,
            workdir="/",
            text=True,
        )
        stdout = process.stdout.read()
        stderr = process.stderr.read()
        return_code = process.wait()
        if return_code != 0:
            raise ModalSandboxRuntimeError(
                "Failed to initialize Modal knowledge-base files "
                f"(exit code {return_code}): {(stderr or stdout)[:200]}"
            )

    def _exec_args(self, command: str) -> Tuple[str, ...]:
        if self.allow_writes:
            return (BASH_EXECUTABLE, "-c", command)
        return (
            "python3",
            "-c",
            _DROP_PRIVILEGES,
            BASH_EXECUTABLE,
            "-c",
            command,
        )

    @staticmethod
    def _prepare_command_for_srt_compatibility(command: str) -> str:
        """Preserve the pinned official ``srt`` command-argv bang escaping.

        The official v1.0.1 trajectories used sandbox-runtime 0.0.23, whose
        command argv layer inserted a backslash before every exclamation mark,
        including marks inside quoted Python and awk programs.  The resulting
        behavior is observable in shell tool outputs, so the Modal adapter
        mirrors it before invoking Bash.
        """
        return command.replace("!", r"\!")

    @staticmethod
    def _is_timeout_error(exc: Exception, modal: Any) -> bool:
        exception_module = getattr(modal, "exception", None)
        timeout_type = getattr(exception_module, "ExecTimeoutError", None)
        return (
            isinstance(timeout_type, type) and isinstance(exc, timeout_type)
        ) or type(exc).__name__ == "ExecTimeoutError"

    def run_command(self, command: str, timeout: int = 30) -> Tuple[int, str, str]:
        """Run a shell command in the network-isolated Modal sandbox."""
        escape_pattern = self._has_escape_pattern(command)
        if escape_pattern:
            return (
                1,
                "",
                "Error: Command blocked - contains "
                f"'{escape_pattern}' which could escape the sandbox",
            )

        with self._lifecycle_lock:
            sandbox = self._ensure_remote_sandbox()
            modal = _load_modal()
            compatible_command = self._prepare_command_for_srt_compatibility(command)
            try:
                process = sandbox.exec(
                    *self._exec_args(compatible_command),
                    timeout=timeout,
                    workdir=REMOTE_KB_DIR,
                    env={"PWD": REMOTE_KB_DIR},
                    text=True,
                )
                stdout = process.stdout.read()
                stderr = process.stderr.read()
                return_code = process.wait()
            except Exception as exc:
                if self._is_timeout_error(exc, modal):
                    return (124, "", f"Command timed out after {timeout} seconds")
                raise ModalSandboxRuntimeError(
                    f"Modal sandbox command failed ({type(exc).__name__})"
                ) from exc

        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")

        return (
            int(return_code),
            self._sanitize_modal_output(stdout, command),
            self._sanitize_modal_output(stderr, command),
        )

    @staticmethod
    def _starts_with_bare_ls_la(command: str) -> bool:
        """Return whether the first simple command is bare ``ls -la``."""
        return bool(re.match(r"^\s*ls\s+-la(?=\s*(?:&&|\|\||;|\||$))", command))

    def _sanitize_modal_output(self, output: str, command: str) -> str:
        """Hide the tmpfs implementation and normalize one known root listing."""
        result = self._sanitize_output(output, command).replace(
            REMOTE_KB_TMPFS_DIR, REMOTE_KB_DIR
        )
        if not self._starts_with_bare_ls_la(command):
            return result

        lines = result.splitlines(keepends=True)
        if len(lines) < 3 or not lines[0].startswith("total "):
            return result

        def ending(line: str) -> str:
            return "\n" if line.endswith("\n") else ""

        if lines[1].rstrip().endswith(" .") and lines[2].rstrip().endswith(" .."):
            lines[0] = f"total 2992{ending(lines[0])}"
            lines[1] = (
                "drwxrwxr-x 2 kb_user  kb_group  69632 Jan  1 00:00 ."
                f"{ending(lines[1])}"
            )
            lines[2] = (
                "drwxrwxr-x 3 kb_user  kb_group  4096 Jan  1 00:00 .."
                f"{ending(lines[2])}"
            )
        return "".join(lines)

    def get_kb_path(self) -> str:
        """Return the knowledge-base path visible to remote shell commands."""
        return REMOTE_KB_DIR

    def get_sandbox_info(self) -> Dict[str, Any]:
        """Return non-sensitive local and remote sandbox metadata."""
        return {
            "sandbox_id": self.sandbox_id,
            "sandbox_dir": str(self.sandbox_dir),
            "kb_dir": REMOTE_KB_DIR,
            "local_kb_dir": str(self.kb_dir),
            "allow_writes": self.allow_writes,
            "num_files": len(self.list_files()),
            "modal_app": self.modal_app_name,
            "modal_timeout": self.modal_timeout,
            "remote_created": self._modal_sandbox is not None,
            "order_manifest_applied": self._order_manifest_applied,
            "order_manifest_sha256": self._order_manifest_sha256,
            "modal_image_recipe_sha256": MODAL_IMAGE_RECIPE_SHA256,
            "modal_image_object_id": self._modal_image_id,
        }

    def cleanup(self) -> None:
        """Terminate and detach Modal resources, then remove local staging."""
        with self._lifecycle_lock:
            if self._closed:
                return
            sandbox = self._modal_sandbox
            termination_error: ModalSandboxRuntimeError | None = None
            if sandbox is not None:
                try:
                    self._terminate_and_detach(sandbox)
                except ModalSandboxRuntimeError as exc:
                    termination_error = exc
                else:
                    self._modal_sandbox = None
                    self._remote_ready = False
                    self._cleanup_pending = False

            self._closed = termination_error is None
            if termination_error is not None:
                self._cleanup_pending = True

            if self.sandbox_dir.exists():
                # Restore owner permissions in case the KB was sealed read-only.
                if self.kb_dir.exists():
                    self.kb_dir.chmod(0o755)
                shutil.rmtree(self.sandbox_dir)
            if termination_error is not None:
                raise termination_error
