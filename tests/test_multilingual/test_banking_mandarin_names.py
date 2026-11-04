"""The opt-in Mandarin banking bundle uses distinct, naturally written names."""

import json
from collections import defaultdict
from pathlib import Path

from tau2.multilingual.factory.localization_artifact import RegionalMapArtifact
from tau2.multilingual.runtime import load_language
from tests.test_multilingual.reviewed_corrections import (
    assert_review_reproduces_published_names,
    load_review,
)

DATA = Path(__file__).resolve().parents[2] / "data"
OLD_MAP = DATA / "tau2/multilingual/zh/localization/banking_knowledge.json"
NEW_MAP = (
    DATA / "tau2/multilingual/zh/localization/"
    "banking_knowledge.name-orthography-v1.json"
)
REVIEW = (
    Path(__file__).resolve().parents[2]
    / "tests/test_multilingual/fixtures/banking_name_review/zh.json"
)


def test_reviewed_mandarin_name_edits_are_guarded_and_source_bound():
    review = load_review(REVIEW)
    assert len(review["corrections"]) == 84
    assert_review_reproduces_published_names(
        RegionalMapArtifact.model_validate_json(OLD_MAP.read_bytes()),
        RegionalMapArtifact.model_validate_json(NEW_MAP.read_bytes()),
        review,
    )


def test_corrected_mandarin_names_preserve_aliases_and_separate_people():
    original = RegionalMapArtifact.model_validate_json(OLD_MAP.read_bytes())
    corrected = RegionalMapArtifact.model_validate_json(NEW_MAP.read_bytes())
    before = {entry.id: entry for entry in original.entries}
    names = [entry for entry in corrected.entries if entry.category == "identity_name"]
    assert len(names) == 90
    assert (
        sum(
            " " in entry.forms.native
            for entry in original.entries
            if entry.category == "identity_name"
        )
        == 84
    )
    assert all(" " not in entry.forms.native for entry in names)
    assert corrected.revision.endswith("name-orthography-dedup-v1")

    owners_by_name = defaultdict(set)
    for entry in names:
        owners_by_name[entry.forms.native].add(entry.source.casefold())
        assert entry.forms.romanized == before[entry.id].forms.romanized or (
            entry.source in {"Sarah Mitchell", "Bob Nakamoto"}
        )
    assert all(len(owners) == 1 for owners in owners_by_name.values())
    assert {
        entry.forms.native for entry in names if entry.source.casefold() == "sarah chen"
    } == {"杨子涵"}
    by_source = defaultdict(set)
    for entry in names:
        by_source[entry.source.casefold()].add(entry.forms.native)
    for source in ("Ayan Wang", "Yuki Nakamura", "Camila Reyes"):
        assert len(by_source[source.casefold()]) == 1
    assert by_source["sarah bosch"] == {"赵雨萱"}
    assert by_source["sarah mitchell"] == {"赵欣怡"}
    assert by_source["marcus chen"] == {"赵宇轩", "蔡明轩"}
    assert by_source["bob nakamoto"] == {"赵浩然"}


def test_opt_in_mandarin_banking_names_match_db_tasks_and_gold():
    old = load_language(
        "zh", "banking_knowledge", mode="text", pack_extension="banking"
    )
    new = load_language(
        "zh", "banking_knowledge", mode="text", pack_extension="banking-dates-v1"
    )
    assert old.scenario.database["users"]["data"]["123"]["name"] == "何 子涵"
    users = new.scenario.database["users"]["data"]
    assert len(users) == len({user["name"] for user in users.values()}) == 39
    assert all(" " not in user["name"] for user in users.values())
    assert users["123"]["name"] == "何子涵"
    assert len(new.scenario.tasks) == 97
    assert {task.id for task in new.scenario.tasks} == {
        task.id for task in old.scenario.tasks
    }

    old_names = {
        entry.forms.native
        for entry in RegionalMapArtifact.model_validate_json(
            OLD_MAP.read_bytes()
        ).entries
        if entry.category == "identity_name" and " " in entry.forms.native
    }
    serialized = json.dumps(new.scenario.model_dump(mode="json"), ensure_ascii=False)
    assert not any(name in serialized for name in old_names)
    for name in ("赵欣怡", "赵浩然", "赵雨萱", "赵宇轩"):
        assert name in serialized
