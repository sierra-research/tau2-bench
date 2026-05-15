from __future__ import annotations

import json
from pathlib import Path

import pytest

from tau2.agent.manifest_bootstrap import (
    load_manifest_entries,
    register_manifest_agents,
)
from tau2.registry import Registry


def _write_factory_module(package_dir: Path) -> None:
    (package_dir / "__init__.py").write_text("")
    (package_dir / "factory_module.py").write_text(
        """
def create_agent(**kwargs):
    return {"kind": "managed", "kwargs": kwargs}
""".strip()
    )


def test_load_manifest_entries_supports_list_and_object_formats(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "agents": [
                    {
                        "name": "demo_agent",
                        "factory": "demo.factory_module:create_agent",
                        "base_agent": "llm_agent",
                        "domain": "retail",
                        "source_commit": "abc1234",
                    }
                ]
            }
        )
    )

    entries = load_manifest_entries(manifest)
    assert [entry.name for entry in entries] == ["demo_agent"]


def test_register_manifest_agents_registers_factories_and_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    package_dir = tmp_path / "demo"
    package_dir.mkdir()
    _write_factory_module(package_dir)
    monkeypatch.syspath_prepend(str(tmp_path))

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "name": "llm_agent__retail__coral__abc1234",
                    "factory": "demo.factory_module:create_agent",
                    "base_agent": "llm_agent",
                    "domain": "retail",
                    "source_commit": "abc1234",
                    "metadata": {"validation_split": "train"},
                }
            ]
        )
    )

    reg = Registry()
    registered = register_manifest_agents(reg, manifest)

    assert [entry.name for entry in registered] == ["llm_agent__retail__coral__abc1234"]
    factory = reg.get_agent_factory("llm_agent__retail__coral__abc1234")
    assert callable(factory)
    assert reg.get_agent_metadata("llm_agent__retail__coral__abc1234", "domain") == "retail"
    assert (
        reg.get_agent_metadata(
            "llm_agent__retail__coral__abc1234", "validation_split"
        )
        == "train"
    )


def test_register_manifest_agents_rejects_duplicate_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    package_dir = tmp_path / "demo"
    package_dir.mkdir()
    _write_factory_module(package_dir)
    monkeypatch.syspath_prepend(str(tmp_path))

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "name": "llm_agent",
                    "factory": "demo.factory_module:create_agent",
                    "base_agent": "llm_agent",
                    "domain": "retail",
                    "source_commit": "abc1234",
                }
            ]
        )
    )

    reg = Registry()
    reg.register_agent_factory(lambda **kwargs: kwargs, "llm_agent")

    with pytest.raises(ValueError, match="already registered"):
        register_manifest_agents(reg, manifest)
