from __future__ import annotations

from pathlib import Path
from typing import Optional

from tau2.agent.manifest_bootstrap import ManifestAgentEntry, register_manifest_agents

MANIFEST_PATH = Path(__file__).with_name("manifest.json")


def register_promoted_agents(
    registry_obj,
    *,
    only_names: Optional[set[str]] = None,
    skip_existing: bool = False,
) -> list[ManifestAgentEntry]:
    return register_manifest_agents(
        registry_obj,
        MANIFEST_PATH,
        only_names=only_names,
        skip_existing=skip_existing,
    )
