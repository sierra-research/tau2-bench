# Copyright Sierra
"""Task-localization invariants, parametrized over ALL localized task sets.

Discovers every ``data/tau2/multilingual/<lang>/<domain>_tasks_<suffix>.json``
and holds it to the shared invariants (``tau2.multilingual.invariants`` — the
same implementation the translator tooling enforces):

- ids are ``<source_id>_<suffix>`` over the domain's task set;
- evaluation_criteria byte-identical to the English source, or — for
  identity-variant sets that rename the caller via ``initialization_data`` —
  identical modulo exactly that rename (reconstructed from the domain DB, so
  data and generator cannot drift apart silently);
- every concrete value survives verbatim; prose is in the pack's script (for
  domains whose sets are translated — deferred-translation domains keep the
  English source prose and skip the script checks); no localized digit forms;
- tasks validate against the Task model and load via the registry.

A new language gets this entire suite for free — no per-language test file
required.
"""

import json

import pytest

from tau2.data_model.tasks import Task
from tau2.multilingual.domain_profiles import DOMAIN_PROFILES
from tau2.multilingual.invariants import check_task_set_localization
from tau2.multilingual.loader import load_language_packs
from tau2.multilingual.localize_lib import load_domain_db, load_domain_tasks
from tau2.multilingual.registry import get_language_pack
from tau2.multilingual.task_sets import (
    TASK_FILE_SEPARATOR,
    discover_localized_task_files,
)
from tau2.registry import registry


def localized_task_sets() -> list[str]:
    return sorted(discover_localized_task_files())


def _load(path):
    with open(path) as fp:
        return json.load(fp)


@pytest.mark.parametrize("task_set_name", localized_task_sets())
class TestLocalizedTaskSet:
    @pytest.fixture
    def context(self, task_set_name):
        path = discover_localized_task_files()[task_set_name]
        domain, _, suffix = path.stem.partition(TASK_FILE_SEPARATOR)
        language = path.parent.name
        load_language_packs()
        pack = get_language_pack(language)
        assert pack is not None, (
            f"task set '{task_set_name}' sits in folder '{language}' but no "
            "language pack with that code is registered"
        )
        scripts = sorted({p.script for p in pack.personas.values() if p.script})
        assert scripts, f"pack '{language}' declares no script codes"
        profile = DOMAIN_PROFILES.get(domain)
        # Deferred-translation domains (profile.tasks_translated False) keep
        # the English source prose — script_code=None skips the script checks.
        script_code = (
            scripts[0] if profile is None or profile.tasks_translated else None
        )
        return {
            "tasks": _load(path),
            "source_tasks": load_domain_tasks(domain),
            "db": load_domain_db(domain),
            "suffix": suffix,
            "script_code": script_code,
            "domain": domain,
        }

    def test_registered_and_loads(self, task_set_name):
        assert task_set_name in registry.get_task_sets()
        loader = registry.get_tasks_loader(task_set_name)
        tasks = loader(task_split_name="base")
        assert tasks
        assert all(isinstance(task, Task) for task in tasks)
        assert len(loader(task_split_name=None)) == len(tasks)
        with pytest.raises(ValueError):
            loader(task_split_name="full")

    def test_tasks_validate_against_task_model(self, task_set_name, context):
        for task in context["tasks"]:
            Task.model_validate(task)

    def test_localization_invariants(self, task_set_name, context):
        problems = check_task_set_localization(
            context["tasks"],
            context["source_tasks"],
            suffix=context["suffix"],
            script_code=context["script_code"],
            domain_db=context["db"],
            domain=context["domain"],
        )
        assert not problems, "\n".join(problems)

    def test_full_source_coverage_or_explicit_subset(self, task_set_name, context):
        """Localized sets currently cover the full domain set; a deliberate
        subset should shrink this assertion consciously."""
        assert len(context["tasks"]) == len(context["source_tasks"])


class TestIdentityRenameCoverage:
    """An identity-variant task whose SOURCE caller is absent from the domain
    DB must be reported as a problem, not crash the checker with a KeyError."""

    @staticmethod
    def _source(caller_id: str) -> dict:
        return {
            "id": "1",
            "user_scenario": {
                "instructions": {"known_info": f"Your user id is {caller_id}."}
            },
        }

    @staticmethod
    def _identity_task(new_id: str) -> dict:
        # Full-identity patch: keyed by the NEW id (a fresh DB key), full record.
        return {
            "id": "1_xx_identity",
            "user_scenario": {
                "instructions": {"known_info": f"Your user id is {new_id}."}
            },
            "initial_state": {
                "initialization_data": {
                    "agent_data": {
                        "users": {
                            new_id: {
                                "name": {"first_name": "Carlos", "last_name": "Gomez"}
                            }
                        }
                    }
                }
            },
        }

    def test_unknown_source_caller_is_reported_not_raised(self):
        from tau2.multilingual.invariants import check_identity_rename_coverage

        db = {
            "users": {"known_user_1": {"name": {"first_name": "A", "last_name": "B"}}}
        }
        problems = check_identity_rename_coverage(
            self._identity_task("carlos_gomez_1"),
            db,
            self._source("ghost_user_9"),
        )
        assert len(problems) == 1
        assert "ghost_user_9" in problems[0]

    def test_known_source_caller_passes(self):
        from tau2.multilingual.invariants import check_identity_rename_coverage

        db = {
            "users": {
                "emma_kim_9957": {"name": {"first_name": "Emma", "last_name": "Kim"}}
            }
        }
        assert (
            check_identity_rename_coverage(
                self._identity_task("carlos_gomez_1"), db, self._source("emma_kim_9957")
            )
            == []
        )
