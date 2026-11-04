"""The opt-in Korean banking bundle uses distinct, naturally written names."""

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
OLD_MAP = DATA / "tau2/multilingual/ko/localization/banking_knowledge.json"
NEW_MAP = (
    DATA / "tau2/multilingual/ko/localization/"
    "banking_knowledge.name-orthography-v1.json"
)


def test_korean_corrections_reproduce_published_name_forms():
    review = load_review(Path(__file__).parent / "fixtures/banking_name_review/ko.json")
    assert len(review["corrections"]) == 89
    assert_review_reproduces_published_names(
        RegionalMapArtifact.model_validate_json(OLD_MAP.read_bytes()),
        RegionalMapArtifact.model_validate_json(NEW_MAP.read_bytes()),
        review,
    )


def test_corrected_korean_name_map_preserves_aliases_and_separates_people():
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
        == 89
    )
    assert all(" " not in entry.forms.native for entry in names)

    owners_by_name = defaultdict(set)
    for entry in names:
        owners_by_name[entry.forms.native].add(entry.source.casefold())
        assert entry.forms.romanized == before[entry.id].forms.romanized or (
            entry.source
            in {
                "Anastasia Volkov",
                "Amara Okonkwo",
                "Sarah Bosch",
                "Marcus Chen",
                "Sarah Mitchell",
            }
        )
    assert all(len(owners) == 1 for owners in owners_by_name.values())
    aliases = [entry for entry in names if entry.source.casefold() == "sarah chen"]
    assert len(aliases) == 2
    assert {entry.forms.native for entry in aliases} == {"서아린"}
    assert corrected.revision.endswith("name-orthography-dedup-v1")


def test_opt_in_korean_banking_names_match_db_tasks_and_gold():
    old = load_language(
        "ko", "banking_knowledge", mode="text", pack_extension="banking"
    )
    new = load_language(
        "ko", "banking_knowledge", mode="text", pack_extension="banking-dates-v1"
    )
    assert old.scenario.database["users"]["data"]["890389b165"]["name"] == "김 지아"
    users = new.scenario.database["users"]["data"]
    assert len(users) == len({user["name"] for user in users.values()}) == 39
    assert all(" " not in user["name"] for user in users.values())
    assert users["890389b165"]["name"] == "김지아"
    assert users["av96d4e7f1"]["name"] == "김서연"
    assert users["av96d4e7f1"]["email"] == "kim.seoyeon9858@example.com"
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
    for name in ("김서연", "김수아", "도서연", "안준호", "서수아"):
        assert name in serialized
