"""Review digests and immutable publication for typed authoring artifacts."""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


def content_digest(value: BaseModel) -> str:
    """Hash typed artifact content consistently, independent of JSON formatting."""
    return hashlib.sha256(
        json.dumps(
            value.model_dump(mode="json"),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


class ReviewApproval(BaseModel):
    """Review attribution bound to the exact typed artifact under review."""

    model_config = ConfigDict(extra="forbid")

    content_sha256: Annotated[
        str,
        Field(pattern="^[0-9a-f]{64}$", description="Digest of the reviewed artifact."),
    ]
    reviewed_by: Annotated[
        str, Field(min_length=1, description="Reviewer of this exact artifact.")
    ]

    def require(self, artifact: BaseModel) -> None:
        """Reject review of different or subsequently edited content."""
        if self.content_sha256 != content_digest(artifact):
            raise ValueError("review does not match current artifact")


def write_artifact(artifact: BaseModel, output: Path) -> bool:
    """Write new staging evidence; identical reruns preserve it byte-for-byte."""
    artifact = type(artifact).model_validate(artifact.model_dump())
    content = artifact.model_dump_json(indent=2) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    # Publish complete bytes without replacing an existing artifact, including
    # when two writers race. A killed write leaves only a temporary file.
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
            stream.flush()
            os.link(temporary, output)
        except FileExistsError:
            if output.read_text(encoding="utf-8") == content:
                return False
            raise FileExistsError(
                f"Artifact already exists; use a new output path: {output}"
            ) from None
        finally:
            temporary.unlink(missing_ok=True)
    return True
