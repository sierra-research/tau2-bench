"""Leaderboard task checks compare against the recorded language binding."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from tau2.data_model.elicitation import ElicitationConfig, ElicitationStrategy
from tau2.environment.environment import EnvironmentInfo
from tau2.multilingual.runtime import TaskSelection, load_language
from tau2.registry import registry
from tau2.scripts.leaderboard.verify_trajectories_public import check_tasks


def _results(domain, tasks, *, language, elicitation=None):
    info = SimpleNamespace(
        environment_info=EnvironmentInfo(domain_name=domain, policy=""),
        language=language,
        localize_db=None,
        pack_extension=None,
        benchmark=None,
        audio_native_config=None,
        elicitation=elicitation,
    )
    return SimpleNamespace(info=info, tasks=tasks)


@pytest.mark.parametrize("domain", ["retail", "telecom"])
def test_english_pack_replacements_pass(domain):
    pack_tasks = load_language(
        "en", domain, mode="text", selection=TaskSelection(task_split_name="base")
    ).tasks
    registry_tasks = registry.get_tasks_loader(domain)("base")
    assert [t.id for t in pack_tasks] == [t.id for t in registry_tasks]
    assert pack_tasks != registry_tasks

    assert check_tasks(_results(domain, pack_tasks, language="en")) == (True, "")
    ok, error = check_tasks(_results(domain, registry_tasks, language="en"))
    assert not ok and error.startswith("Task mismatches found in 50 tasks")


def test_results_without_language_use_registry_tasks():
    registry_tasks = registry.get_tasks_loader("telecom")("base")
    assert check_tasks(_results("telecom", registry_tasks, language=None)) == (
        True,
        "",
    )


@pytest.mark.parametrize("strategy", list(ElicitationStrategy))
def test_intake_results_use_recorded_strategy_variant(strategy):
    def tasks_for(chosen):
        return load_language(
            "en",
            "intake",
            mode="text",
            selection=TaskSelection(
                task_split_name="base", task_kwargs={"strategy": chosen.value}
            ),
        ).tasks

    config = ElicitationConfig(strategy=strategy)
    recorded = _results(
        "intake", tasks_for(strategy), language="en", elicitation=config
    )
    assert check_tasks(recorded) == (True, "")

    other = next(s for s in ElicitationStrategy if s != strategy)
    swapped = _results("intake", tasks_for(other), language="en", elicitation=config)
    assert not check_tasks(swapped)[0]


# 1.x submissions (no recorded language) are checked against the registry base
# split. Those tasks must stay byte-identical to the v1.0.1 release so results
# produced with v1.0.1 keep verifying; changing them is a breaking change for
# every 1.x leaderboard entry and needs a deliberate update here.
V1_0_1_BASE_TASKS = {
    "airline": (50, "0673fb5ce6150c759d97ac6e7202e8ad9f36a4cfdc444be393cea87945276411"),
    "retail": (114, "736b6e436c8278b2268631a6b7d6447fd97955784fc1c2ea5f08bc46ef137768"),
    "telecom": (
        114,
        "fed0f81f4c5a57255906cf2673a4c56a05a540e2b4802aab1f5ffe9ad61b25a7",
    ),
}


@pytest.mark.parametrize("domain", sorted(V1_0_1_BASE_TASKS))
def test_registry_base_tasks_match_v1_0_1(domain):
    tasks = registry.get_tasks_loader(domain)("base")
    digest = hashlib.sha256(
        json.dumps([t.model_dump(mode="json") for t in tasks], sort_keys=True).encode()
    ).hexdigest()
    assert (len(tasks), digest) == V1_0_1_BASE_TASKS[domain]
