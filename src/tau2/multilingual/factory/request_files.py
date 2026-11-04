"""Stage request files that reference prior artifacts by path and digest.

Stage artifacts (drafts, inventories, maps, voice assets, approvals) are large and
immutable. A request file names them instead of inlining their content::

    {"draft": {"$artifact": "draft.json", "sha256": "<digest printed by draft>"}}

Paths are relative to the request file unless absolute. The digest is the same
``content_digest`` every stage prints for its output, so a reference fails loudly
when the referenced file was edited or replaced after it was recorded.
"""

import hashlib
import json
from pathlib import Path
from typing import Annotated, Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field

ARTIFACT_KEY = "$artifact"

ModelT = TypeVar("ModelT", bound=BaseModel)


class ArtifactRef(BaseModel):
    """One request-file reference to an immutable stage artifact."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    path: Annotated[
        Path,
        Field(
            alias=ARTIFACT_KEY,
            description="Artifact JSON path, relative to the request file.",
        ),
    ]
    sha256: Annotated[
        str,
        Field(
            pattern="^[0-9a-f]{64}$",
            description="content_digest of the artifact, as printed by its stage.",
        ),
    ]


def json_digest(value: Any) -> str:
    """Digest parsed JSON exactly as ``content_digest`` digests a typed artifact."""
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _resolve(value: Any, base: Path) -> Any:
    if isinstance(value, list):
        return [_resolve(item, base) for item in value]
    if not isinstance(value, dict):
        return value
    if ARTIFACT_KEY not in value:
        return {key: _resolve(item, base) for key, item in value.items()}
    ref = ArtifactRef.model_validate(value)
    path = ref.path if ref.path.is_absolute() else base / ref.path
    content = json.loads(path.read_text())
    if json_digest(content) != ref.sha256:
        raise ValueError(f"{path} does not match its referenced sha256")
    return content


def load_request(model: type[ModelT], request: Path) -> ModelT:
    """Validate a stage request after resolving its digest-checked references."""
    data = json.loads(request.read_text())
    return model.model_validate(_resolve(data, request.parent))


def artifact_ref(path: Path, value: BaseModel, *, relative_to: Path) -> dict:
    """Serialize a reference to a written artifact for a request file."""
    from tau2.utils.artifacts import content_digest

    target = Path(path)
    try:
        target = target.resolve().relative_to(relative_to.resolve())
    except ValueError:
        target = target.resolve()
    return ArtifactRef(path=target, sha256=content_digest(value)).model_dump(
        mode="json", by_alias=True
    )
