from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


class ManifestAgentEntry(BaseModel):
    """Manifest record for a staged or promoted external Tau2 agent."""

    name: str
    factory: str
    base_agent: str
    domain: str
    source_commit: str
    contract_version: str = "v1"
    source_repo: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def load_manifest_entries(manifest_path: Path) -> list[ManifestAgentEntry]:
    """Load manifest entries from a JSON file.

    The manifest format is either a bare JSON list or a JSON object with an
    `agents` list.
    """

    if not manifest_path.exists():
        return []

    raw = json.loads(manifest_path.read_text())
    if isinstance(raw, dict):
        raw = raw.get("agents", [])
    if not isinstance(raw, list):
        raise ValueError(f"Manifest must contain a list of agents: {manifest_path}")
    return [ManifestAgentEntry.model_validate(item) for item in raw]


def load_factory(factory_path: str):
    """Import a factory callable from `module.submodule:attribute`."""

    if ":" not in factory_path:
        raise ValueError(
            f"Factory path must use 'module.path:attribute' syntax, got: {factory_path}"
        )
    module_name, attribute_name = factory_path.split(":", 1)
    module = import_module(module_name)
    return getattr(module, attribute_name)


def register_manifest_agents(
    registry_obj,
    manifest_path: Path,
    *,
    only_names: Optional[set[str]] = None,
    skip_existing: bool = False,
) -> list[ManifestAgentEntry]:
    """Register agents described by a manifest into a Tau2 registry."""

    registered: list[ManifestAgentEntry] = []
    for entry in load_manifest_entries(manifest_path):
        if only_names is not None and entry.name not in only_names:
            continue

        if registry_obj.get_agent_factory(entry.name) is not None:
            if skip_existing:
                continue
            raise ValueError(f"Agent factory {entry.name} already registered")

        factory = load_factory(entry.factory)
        metadata = {
            "managed_by_manifest": True,
            "base_agent": entry.base_agent,
            "domain": entry.domain,
            "source_commit": entry.source_commit,
            "contract_version": entry.contract_version,
            "source_repo": entry.source_repo,
        }
        metadata.update(entry.metadata)
        registry_obj.register_agent_factory(factory, entry.name, metadata=metadata)
        registered.append(entry)

    return registered
