"""Offline contract checks for the first Voice V2 data extraction slice."""

import json
from pathlib import Path

import pytest

from tau2.data_model.language import NameOrder
from tau2.data_model.tasks import Task

DATA = Path(__file__).resolve().parents[2] / "data" / "tau2"
LANGUAGES = ("en", "es", "pt", "hi", "zh", "ko")
DOMAINS = ("airline", "retail")


def _bundle(language: str, domain: str) -> dict:
    return json.loads(
        (DATA / "multilingual" / language / "regional" / f"{domain}.json").read_text()
    )


def _published_tasks(language: str, domain: str) -> list[dict]:
    """English binds registry tasks (retail with reviewed replacements); other
    languages ship tasks in regional bundles."""
    if language == "en":
        path = DATA / "multilingual" / "en" / f"{domain}_tasks_en.json"
        if path.exists():
            return json.loads(path.read_text())
        metadata = json.loads(
            (DATA / "multilingual" / "en" / f"{domain}_metadata_en.json").read_text()
        )
        registry = json.loads((DATA / "domains" / domain / "tasks.json").read_text())
        return [row for row in registry if row["id"] in metadata]
    return _bundle(language, domain)["tasks"]


@pytest.mark.parametrize("domain", DOMAINS)
@pytest.mark.parametrize("language", LANGUAGES)
def test_frozen_cells_have_valid_tasks_and_complete_gender(domain, language):
    """Every shipped row resolves to the same frozen frame with one gender."""
    root = DATA / "multilingual" / language
    suffix = language if language == "en" else f"{language}_identity"
    subset = json.loads((DATA / "task_subsets" / f"{domain}_50.json").read_text())
    raw = _published_tasks(language, domain)
    genders = {
        task_id: metadata["caller_gender"]
        for task_id, metadata in json.loads(
            (root / f"{domain}_metadata_{language}.json").read_text()
        ).items()
    }
    tasks = [Task.model_validate(row) for row in raw]
    # English reviewed replacements keep canonical registry IDs.
    expected = [
        task_id if language == "en" else f"{task_id}_{suffix}"
        for task_id in subset["task_ids"]
    ]
    assert len(tasks) == len(set(expected)) == 50
    assert [task.id for task in tasks] == expected
    assert set(genders) == set(expected)
    assert set(genders.values()) <= {"male", "female"}


def test_english_airline_binds_registry_tasks():
    """English airline publishes metadata only; its tasks are the registry's."""
    from tau2.multilingual.runtime import load_language
    from tau2.registry import registry

    assert not (DATA / "multilingual/en/airline_tasks_en.json").exists()
    loaded = load_language("en", "airline", mode="voice")
    assert loaded.binding.tasks_file is None
    assert loaded.tasks == registry.get_tasks_loader("airline")(task_split_name=None)
    subset = json.loads((DATA / "task_subsets/airline_50.json").read_text())
    assert [task.id for task in loaded.tasks] == subset["task_ids"]


@pytest.mark.parametrize("domain", ["retail"])
def test_english_preserves_main_task_semantics(domain):
    """Only database-derived retail name facts may differ from main."""
    original = {
        row["id"]: Task.model_validate(row)
        for row in json.loads((DATA / "domains" / domain / "tasks.json").read_text())
    }
    rows = json.loads(
        (DATA / "multilingual" / "en" / f"{domain}_tasks_en.json").read_text()
    )
    if domain == "retail":
        from tau2.domains.retail.data_model import RetailDB
        from tau2.multilingual.factory.localization_compile import (
            _retail_tools_for_task,
        )
        from tau2.multilingual.names import (
            clarify_retail_name,
            resolve_retail_caller,
        )

        database = RetailDB.model_validate_json(
            (DATA / "domains/retail/db.json").read_text()
        )
    for row in rows:
        task = Task.model_validate(row)
        baseline = original[task.id.removesuffix("_en")]
        if domain == "retail":
            tools = _retail_tools_for_task(database, baseline)
            caller = resolve_retail_caller(baseline, tools, baseline, tools, {})
            baseline = clarify_retail_name(
                baseline,
                caller_id=caller.user_id,
                caller=caller,
                name_order=NameOrder.GIVEN_FIRST,
            )
        assert task.model_dump(exclude={"id"}) == baseline.model_dump(exclude={"id"})


@pytest.mark.parametrize("domain", DOMAINS)
@pytest.mark.parametrize("language", LANGUAGES[1:])
def test_golden_callers_resolve_in_regional_database(domain, language):
    """Golden user references resolve to regional DB users named in the scenario."""
    bundle = _bundle(language, domain)
    users = bundle["database"]["users"]
    defects = []
    for task in bundle["tasks"]:
        known_info = task["user_scenario"]["instructions"]["known_info"] or ""
        for action in task["evaluation_criteria"]["actions"] or []:
            user_id = action["arguments"].get("user_id")
            if user_id is None:
                continue
            user = users.get(user_id)
            if user is None or (
                user_id not in known_info
                and not all(name in known_info for name in user["name"].values())
            ):
                defects.append((task["id"], user_id))
    assert not defects


@pytest.mark.parametrize("language", LANGUAGES[1:])
def test_retail_wrong_zip_golden_matches_localized_caller_instructions(language):
    """The first failed lookup uses the scenario's incorrect ZIP; the last, the DB's."""
    bundle = _bundle(language, "retail")
    task = next(t for t in bundle["tasks"] if t["id"] == f"68_{language}_identity")
    scenario = " ".join(
        text for text in task["user_scenario"]["instructions"].values() if text
    )
    lookups = [
        action["arguments"]
        for action in task["evaluation_criteria"]["actions"]
        if action["name"] == "find_user_id_by_name_zip"
    ]
    assert len(lookups) == 2
    assert all(lookup["zip"] in scenario for lookup in lookups)
    zips = {
        user["address"]["zip"]
        for user in bundle["database"]["users"].values()
        if user["name"]
        == {
            "first_name": lookups[0]["first_name"],
            "last_name": lookups[0]["last_name"],
        }
    }
    assert lookups[0]["zip"] not in zips
    assert lookups[-1]["zip"] in zips
