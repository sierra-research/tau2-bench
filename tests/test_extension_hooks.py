"""Extension points for domains that live outside tau2 (installed as plugins)."""

import sys
from types import SimpleNamespace

import pytest

from tau2.data_model.message import ToolMessage
from tau2.data_model.simulation import TextRunConfig
from tau2.domains.mock.environment import get_environment, get_tasks
from tau2.registry import DomainOptions, Registry, load_plugins, registry
from tau2.runner.batch import run_single_task


def test_domains_default_to_no_options():
    assert registry.get_domain_options("mock") == DomainOptions()
    assert registry.get_domain_options("not_a_domain") == DomainOptions()


def test_register_domain_keeps_options():
    target = Registry()
    options = DomainOptions(env_kwargs=lambda task: {"task": task})
    target.register_domain(get_environment, "plugin_domain", options)
    assert target.get_domain_options("plugin_domain") is options


def test_plugins_register_through_entry_points(monkeypatch):
    target = Registry()

    def register(r):
        r.register_domain(get_environment, "plugin_domain")

    def broken(r):
        raise RuntimeError("bad plugin")

    found = [
        SimpleNamespace(name="good", load=lambda: register),
        SimpleNamespace(name="bad", load=lambda: broken),
    ]
    monkeypatch.setattr(
        sys.modules["tau2.registry"],
        "entry_points",
        lambda group: found if group == "tau2.plugins" else [],
    )

    load_plugins(target)

    assert "plugin_domain" in target.get_domains()


def test_batch_runner_uses_domain_orchestrator_builder(monkeypatch):
    task = get_tasks()[0]
    built = []

    class Built(Exception):
        pass

    def domain_builder(config, t, **kwargs):
        built.append((config.domain, t.id, kwargs["seed"]))
        raise Built

    monkeypatch.setitem(
        registry._domain_options,
        "mock",
        DomainOptions(build_orchestrator=domain_builder),
    )
    with pytest.raises(Built):
        run_single_task(TextRunConfig(domain="mock"), task, seed=7)

    assert built == [("mock", task.id, 7)]


def test_environment_hooks_default_to_none():
    env = get_environment()
    assert env.get_evaluation_info() is None
    assert env.get_run_record() is None


def test_tool_message_raw_data_is_omitted_when_unset():
    plain = ToolMessage(id="1", role="tool", content="ok")
    assert "raw_data" not in plain.model_dump()
    rich = ToolMessage(id="1", role="tool", content="ok", raw_data={"a": 1})
    assert rich.model_dump()["raw_data"] == {"a": 1}
    assert ToolMessage.model_validate(rich.model_dump()).raw_data == {"a": 1}
