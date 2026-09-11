# Copyright Sierra
"""Tests for the locale identity swap (``tau2 factory localize-entities``).

Covers the engine (:func:`derive_identity_tasks`), the deterministic
corpus-driven generator, gender derivation from the English source caller's
name (closed catalog, balanced 25/25 on airline), the extended id/email
rename invariants, and voice gender pinning — against the toy airline
fixture, a synthetic corpus, and the committed identity artifacts of every
language that ships a locale corpus (es, de).
"""

import copy
import hashlib
import json
import re
from pathlib import Path

import pytest

from tau2.multilingual.domain_profiles import CallerIdentityKind
from tau2.multilingual.factory import entity_localization as ent
from tau2.multilingual.invariants import (
    AddressProseFormat,
    IdentityRename,
    caller_address_variant,
    caller_email_variant_pairs,
    caller_zip_variant_pairs,
    check_task_set_localization,
    expected_eval_criteria,
    identity_rename,
    localized_address_variant,
    variant_email,
)
from tau2.multilingual.localize_lib import derive_identity_tasks
from tau2.multilingual.names import compose_full_name, name_order_for_language

FIXTURES = Path(__file__).parent / "factory_testing" / "fixtures" / "toyair"


def _toy():
    db = json.loads((FIXTURES / "db.json").read_text())
    src = json.loads((FIXTURES / "tasks.json").read_text())
    # A pretend 'es' localized set: English prose passes the latn script check.
    localized = [{**copy.deepcopy(t), "id": f"{t['id']}_es"} for t in src]
    return db, src, localized


TOY_MAP = {
    "pat_lee_1234": {
        "user_id": "carlos_gomez_1234",
        "first_name": "Carlos",
        "last_name": "Gomez",
        "email": "carlos.gomez1234@example.com",
        "gender": "male",
        "address": {
            "address1": "Calle Mayor 5",
            "address2": None,
            "city": "Madrid",
            "state": "MD",
            "country": "ESP",
            "zip": "28013",
        },
    },
    "sam_chen_5678": {
        "user_id": "lucia_ruiz_5678",
        "first_name": "Lucia",
        "last_name": "Ruiz",
        "email": "lucia.ruiz5678@example.com",
        "gender": "female",
    },
}


class TestDeriveIdentityTasks:
    def test_toy_set_is_invariant_clean(self):
        db, src, localized = _toy()
        out, problems = derive_identity_tasks(
            localized, src, db, TOY_MAP, "es", "latn", domain="airline"
        )
        assert problems == []
        assert len(out) == len(src)
        assert all(t["id"].endswith("_es_identity") for t in out)

    def test_full_identity_swapped_in_prose_and_patch(self):
        db, src, localized = _toy()
        out, _ = derive_identity_tasks(
            localized, src, db, TOY_MAP, "es", "latn", domain="airline"
        )
        task = next(t for t in out if t["id"] == "1_es_identity")
        ins = task["user_scenario"]["instructions"]
        # Old identity gone, new identity present, across prose fields.
        blob = "\n".join(v or "" for v in ins.values() if isinstance(v, str))
        assert "Pat Lee" not in blob and "pat_lee_1234" not in blob
        assert "pat.lee1234@example.com" not in blob
        assert "Carlos Gomez" in blob and "carlos_gomez_1234" in blob
        assert "carlos.gomez1234@example.com" in blob
        # The patch carries the FULL record under the NEW id + address.
        patch = task["initial_state"]["initialization_data"]["agent_data"]
        assert set(patch["users"]) == {"carlos_gomez_1234"}
        new_user = patch["users"]["carlos_gomez_1234"]
        assert new_user["email"] == "carlos.gomez1234@example.com"
        assert new_user["address"]["city"] == "Madrid"
        # Reservation back-ref repointed + caller's passenger renamed.
        res = patch["reservations"]["AB12CD"]
        assert res["user_id"] == "carlos_gomez_1234"
        assert res["passengers"][0]["first_name"] == "Carlos"

    def test_eval_criteria_user_id_swapped(self):
        db, src, localized = _toy()
        out, _ = derive_identity_tasks(
            localized, src, db, TOY_MAP, "es", "latn", domain="airline"
        )
        task = next(t for t in out if t["id"] == "1_es_identity")
        action = task["evaluation_criteria"]["actions"][0]
        assert action["arguments"]["user_id"] == "carlos_gomez_1234"

    def test_unknown_caller_reported_not_raised(self):
        db, src, localized = _toy()
        out, problems = derive_identity_tasks(
            localized, src, db, {}, "es", "latn", domain="airline"
        )
        assert out == []
        assert any("not in identity map" in p for p in problems)


def _toy_telecom_db() -> dict:
    return {
        "customers": [
            {
                "customer_id": "C1001",
                "full_name": "John Smith",
                "date_of_birth": "1985-06-15",
                "email": "john.smith@example.com",
                "phone_number": "555-123-2002",
            },
            {
                "customer_id": "C1002",
                "full_name": "Sarah Johnson",
                "date_of_birth": "1990-11-22",
                "email": "sarah.j@example.com",
                "phone_number": "555-123-1002",
            },
        ],
        "lines": [
            {
                "line_id": "L1002",
                "phone_number": "555-123-2002",
            },
            {
                "line_id": "L1004",
                "phone_number": "555-123-2004",
            },
        ],
        "bills": [
            {
                "bill_id": "B1001",
                "line_items": [
                    {"description": "Premium Plan - Line 555-123-2002"},
                    {"description": "Basic Plan - Line 555-123-2004"},
                ],
            }
        ],
    }


def _toy_telecom_task(task_id: str) -> dict:
    """A DIVERSIFIED seed task — the new normal: `tau2 factory seed-tasks`
    patches a pool caller (here Marcus Delgado) over the canonical C1001
    record via initialization_data, so every telecom source task the identity
    swap sees already carries a customers patch."""
    return {
        "id": task_id,
        "description": {"purpose": "Fix mobile data."},
        "user_scenario": {
            "instructions": {
                "domain": "telecom",
                "reason_for_call": "Your mobile data is not working.",
                "known_info": (
                    "You are Marcus Delgado with phone number 555-123-2002."
                ),
                "task_instructions": "You are willing to refuel 2.0 GB of data.",
            }
        },
        "ticket": "Customer name: Marcus Delgado, phone number: 555-123-2002.",
        "initial_state": {
            "initialization_data": {
                "agent_data": {
                    "customers": [
                        {
                            "customer_id": "C1001",
                            "full_name": "Marcus Delgado",
                            "date_of_birth": "1980-07-09",
                            "email": "marcus.delgado@example.com",
                            "phone_number": "555-123-2002",
                        },
                        _toy_telecom_db()["customers"][1],
                    ]
                },
                "user_data": None,
            },
            "initialization_actions": [
                {
                    "env_type": "user",
                    "func_name": "set_user_info",
                    "arguments": {
                        "name": "Marcus Delgado",
                        "phone_number": "555-123-2002",
                    },
                },
                {
                    "env_type": "assistant",
                    "func_name": "enable_roaming",
                    "arguments": {"customer_id": "C1001", "line_id": "L1002"},
                },
            ],
            "message_history": None,
        },
        "evaluation_criteria": {
            "actions": [
                {
                    "action_id": "toggle_roaming_0",
                    "requestor": "user",
                    "name": "toggle_roaming",
                    "arguments": {},
                }
            ],
            "nl_assertions": ["Agent confirms Marcus Delgado's identity."],
            "reward_basis": ["ENV_ASSERTION"],
        },
    }


# Keyed by the ENGLISH caller full name (the diversified seed callers all
# share C1001's phone/customer record, so the customer id identifies nobody).
TELECOM_MAP = {
    "Marcus Delgado": {
        "first_name": "Jorge",
        "last_name": "Gil",
        "full_name": "Jorge Gil",
        "email": "jorge.gil@example.com",
        "phone_number": "612 34 56 78",
        "gender": "male",
    }
}


class TestDeriveTelecomIdentityTasks:
    """The STRUCTURED_NAME_PHONE branch localizes one complete caller identity."""

    def _toy(self):
        db = _toy_telecom_db()
        src = [_toy_telecom_task("[toy]a"), _toy_telecom_task("[toy]b")]
        localized = [{**copy.deepcopy(t), "id": f"{t['id']}_hi"} for t in src]
        return db, src, localized

    def test_toy_set_is_invariant_clean_despite_english_prose(self):
        db, src, localized = self._toy()
        # 'deva' would fail the script-presence checks on English prose —
        # the telecom profile (tasks_translated=False) must disable them.
        out, problems = derive_identity_tasks(
            localized, src, db, TELECOM_MAP, "hi", "deva", domain="telecom"
        )
        assert problems == []
        assert [t["id"] for t in out] == ["[toy]a_hi_identity", "[toy]b_hi_identity"]

    def test_rename_lands_in_prose_ticket_action_and_db_patch(self):
        db, src, localized = self._toy()
        out, _ = derive_identity_tasks(
            localized, src, db, TELECOM_MAP, "hi", None, domain="telecom"
        )
        task = out[0]
        ins = task["user_scenario"]["instructions"]
        assert "Jorge Gil" in ins["known_info"]
        assert "Marcus Delgado" not in json.dumps(ins) + task["ticket"]
        assert "Jorge Gil" in task["ticket"]
        # The phone number is a canonical environment key: it stays put in
        # prose and never takes the identity map's locale number.
        assert "555-123-2002" in ins["known_info"]
        assert "612 34 56 78" not in json.dumps(ins) + task["ticket"]

        actions = task["initial_state"]["initialization_actions"]
        set_info = next(a for a in actions if a["func_name"] == "set_user_info")
        assert set_info["arguments"] == {
            "name": "Jorge Gil",
            "phone_number": "555-123-2002",
        }
        # enable_roaming untouched (canonical ids).
        roaming = next(a for a in actions if a["func_name"] == "enable_roaming")
        assert roaming["arguments"] == {"customer_id": "C1001", "line_id": "L1002"}

        # Customers patch composed over the source's EFFECTIVE list: caller
        # renamed, phone canonical, the diversified DOB carried through (NOT
        # reverted to the DB's canonical 1985-06-15), bystander byte-intact.
        patch = task["initial_state"]["initialization_data"]["agent_data"]
        customers = {c["customer_id"]: c for c in patch["customers"]}
        assert set(customers) == {"C1001", "C1002"}
        assert customers["C1001"]["full_name"] == "Jorge Gil"
        assert customers["C1001"]["email"] == "jorge.gil@example.com"
        assert customers["C1001"]["phone_number"] == "555-123-2002"
        assert customers["C1001"]["date_of_birth"] == "1980-07-09"
        assert customers["C1002"] == _toy_telecom_db()["customers"][1]

        # No phone swap -> no alias map.
        assert "phone_number_aliases" not in patch

    def test_eval_criteria_rebuilt_with_rename(self):
        db, src, localized = self._toy()
        out, _ = derive_identity_tasks(
            localized, src, db, TELECOM_MAP, "hi", None, domain="telecom"
        )
        criteria = out[0]["evaluation_criteria"]
        assert criteria == expected_eval_criteria(src[0], out[0], db)
        assert criteria["nl_assertions"] == ["Agent confirms Jorge Gil's identity."]

    def test_unknown_caller_reported_not_raised(self):
        db, src, localized = self._toy()
        out, problems = derive_identity_tasks(
            localized, src, db, {}, "hi", None, domain="telecom"
        )
        assert out == []
        assert any("not in identity map" in p for p in problems)

    def test_identity_rename_resolves_customers_patch_shape(self):
        db, src, localized = self._toy()
        out, _ = derive_identity_tasks(
            localized, src, db, TELECOM_MAP, "hi", None, domain="telecom"
        )
        rename = identity_rename(src[0], out[0], db)
        assert rename is not None
        assert rename.old_user_id == rename.new_user_id == "C1001"
        # The OLD identity is the source's EFFECTIVE (diversified) caller,
        # not the DB's canonical John Smith record.
        assert ("Marcus Delgado", "Jorge Gil") in rename.string_pairs
        assert (
            "marcus.delgado@example.com",
            "jorge.gil@example.com",
        ) in rename.string_pairs
        # The phone number stays canonical, so no phone pair is claimed.
        assert not any("555-123-2002" in pair for pair in rename.string_pairs)
        # The customer id never changes, so no id exemption is claimed.
        assert rename.renamed_source_values == {
            "marcus.delgado@example.com",
        }


class TestStructuredIdentityMap:
    def test_map_keyed_by_english_name_and_deterministic(self, monkeypatch):
        corpus = ent.load_corpus("es")
        db = _toy_telecom_db()
        tasks = [_toy_telecom_task("[toy]a")]
        monkeypatch.setattr(ent, "load_domain_db", lambda d: db)
        monkeypatch.setattr(ent, "load_domain_tasks", lambda d: tasks)

        a = ent.build_identity_map("telecom", corpus)
        b = ent.build_identity_map("telecom", corpus)
        assert a == b
        # Keyed by the ENGLISH caller name the task announces (the
        # diversified callers all share C1001's record), not the customer id.
        assert set(a) == {"Marcus Delgado"}
        identity = a["Marcus Delgado"]
        assert identity["gender"] == "male"  # inherited from 'Marcus'
        assert identity["full_name"] == (
            f"{identity['first_name']} {identity['last_name']}"
        )
        assert re.fullmatch(r"6\d{2} \d{2} \d{2} \d{2}", identity["phone_number"])
        # Committed names are the ASCII fold of the drawn corpus spelling.
        assert identity["first_name"] in {
            ent.fold_to_ascii(n) for n in corpus.male_first_names
        }
        assert "user_id" not in identity  # telecom callers have no user_id

    def test_distinct_callers_get_distinct_locale_phone_numbers(self, monkeypatch):
        corpus = ent.load_corpus("es")
        db = _toy_telecom_db()
        tasks = [_toy_telecom_task("[toy]a"), _toy_telecom_task("[toy]b")]
        tasks[1]["initial_state"]["initialization_actions"][0]["arguments"]["name"] = (
            "Trevor Nash"
        )
        monkeypatch.setattr(ent, "load_domain_db", lambda d: db)
        monkeypatch.setattr(ent, "load_domain_tasks", lambda d: tasks)

        idmap = ent.build_identity_map("telecom", corpus)

        phones = [identity["phone_number"] for identity in idmap.values()]
        assert len(set(phones)) == len(phones)
        assert all(re.fullmatch(r"6\d{2} \d{2} \d{2} \d{2}", p) for p in phones)

    def test_distinct_callers_on_one_phone_get_distinct_identities(self, monkeypatch):
        """The diversified seed set: two pool callers, same canonical phone.
        A customer-id-keyed map would collapse them into ONE identity."""
        corpus = ent.load_corpus("es")
        db = _toy_telecom_db()
        tasks = [_toy_telecom_task("[toy]a"), _toy_telecom_task("[toy]b")]
        tasks[1]["initial_state"]["initialization_actions"][0]["arguments"]["name"] = (
            "Olivia Bennett"
        )
        monkeypatch.setattr(ent, "load_domain_db", lambda d: db)
        monkeypatch.setattr(ent, "load_domain_tasks", lambda d: tasks)

        idmap = ent.build_identity_map("telecom", corpus)
        assert set(idmap) == {"Marcus Delgado", "Olivia Bennett"}
        assert idmap["Marcus Delgado"]["gender"] == "male"
        assert idmap["Olivia Bennett"]["gender"] == "female"
        assert (
            idmap["Marcus Delgado"]["full_name"] != idmap["Olivia Bennett"]["full_name"]
        )

    def test_email_collision_bumps(self, monkeypatch):
        corpus = ent.LocaleCorpus.model_validate(
            {
                "female_first_names": ["Ana"],
                "male_first_names": ["Jorge"],
                "last_names": ["Gil"],
                "email_domain": "example.com",
                "phone_number_format": "6## ## ## ##",
                "cities": [{"city": "Madrid", "region": "MD"}],
                "country": "ESP",
            }
        )
        db = _toy_telecom_db()
        tasks = [_toy_telecom_task("[toy]a"), _toy_telecom_task("[toy]b")]
        tasks[1]["initial_state"]["initialization_actions"][0]["arguments"]["name"] = (
            "Trevor Nash"  # second MALE caller, same canonical phone
        )
        monkeypatch.setattr(ent, "load_domain_db", lambda d: db)
        monkeypatch.setattr(ent, "load_domain_tasks", lambda d: tasks)

        idmap = ent.build_identity_map("telecom", corpus)
        emails = [v["email"] for v in idmap.values()]
        assert len(set(emails)) == 2  # single-name corpus forced a bump


class TestExpectedEvalCriteriaIdRename:
    """The invariant layer must swap user_id (not just names) for _identity tasks."""

    def _pair(self):
        source = {
            "id": "1",
            "user_scenario": {
                "instructions": {"known_info": "user id is pat_lee_1234"}
            },
            "evaluation_criteria": {
                "actions": [{"name": "x", "arguments": {"user_id": "pat_lee_1234"}}],
                "nl_assertions": ["Confirms user pat_lee_1234."],
            },
        }
        localized = {
            "id": "1_es_identity",
            "user_scenario": {
                "instructions": {"known_info": "user id is carlos_gomez_1234"}
            },
            "initial_state": {
                "initialization_data": {
                    "agent_data": {
                        "users": {
                            "carlos_gomez_1234": {
                                "name": {"first_name": "Carlos", "last_name": "Gomez"}
                            }
                        }
                    }
                }
            },
        }
        db = {
            "users": {
                "pat_lee_1234": {"name": {"first_name": "Pat", "last_name": "Lee"}}
            }
        }
        return source, localized, db

    def test_id_swapped_in_actions_and_strings(self):
        source, localized, db = self._pair()
        expected = expected_eval_criteria(source, localized, db)
        assert expected["actions"][0]["arguments"]["user_id"] == "carlos_gomez_1234"
        assert "carlos_gomez_1234" in expected["nl_assertions"][0]

    def test_identity_rename_pairs_old_and_new(self):
        source, localized, db = self._pair()
        rename = identity_rename(source, localized, db)
        assert rename.old_user_id == "pat_lee_1234"
        assert rename.new_user_id == "carlos_gomez_1234"
        assert "pat_lee_1234" in rename.renamed_source_values


class TestApplyToProse:
    """``IdentityRename.apply_to_prose`` — the ONE swap both the identity-set
    emitter and english-prompt mode run, and the shape gate that decides how
    much of the caller it rewrites."""

    RENAME = IdentityRename(
        old_user_id="yusuf_rossi_9620",
        new_user_id="ivan_blanco_9620",
        old_name=("Yusuf", "Rossi"),
        new_name=("Iván", "Blanco"),
        old_full_name="Yusuf Rossi",
        new_full_name="Iván Blanco",
        old_email="yusuf.rossi8836@example.com",
        new_email="ivan.blanco8836@example.com",
        old_address={
            "address1": "763 Broadway",
            "address2": "Suite 135",
            "city": "Philadelphia",
            "state": "PA",
            "country": "USA",
            "zip": "19122",
        },
        new_address={
            "address1": "Calle de Alcalá 79",
            "address2": "Piso 2",
            "city": "Murcia",
            "state": "MC",
            "country": "ESP",
            "zip": "11532",
        },
    )

    def _prose(self, kind):
        return self.RENAME.apply_to_prose(
            "You are Yusuf Rossi in zip code 19122. Yusuf wants the order "
            "shipped to 763 Broadway, Suite 135, Philadelphia, PA 19122 — "
            "you used to live in Philadelphia, Pennsylvania.",
            kind,
        )

    def test_prose_name_zip_swaps_the_whole_caller(self):
        text = self._prose(CallerIdentityKind.PROSE_NAME_ZIP)
        assert "Iván Blanco" in text
        assert "Yusuf" not in text and "Rossi" not in text
        assert "19122" not in text and "11532" in text
        assert "Calle de Alcalá 79, Piso 2" in text
        # The city carries its region with it — never "Murcia, PA".
        assert "Murcia, MC" in text
        assert "Philadelphia" not in text
        # A spelled-out state is swapped as part of the compound, so the
        # sentence never ends up half-localized ("Murcia, Pennsylvania").
        assert "Pennsylvania" not in text

    def test_handle_shape_swaps_only_the_literal_pairs(self):
        """Airline's caller is not anchored by address, and its prose names
        places the caller does not own (a Houston-based caller flies to
        Houston). Only full name / id / email may move."""
        text = self._prose(CallerIdentityKind.USER_ID_HANDLE)
        assert "Iván Blanco" in text
        assert "Philadelphia" in text and "19122" in text
        assert "763 Broadway" in text
        # The bare given name is left alone: it may belong to a co-passenger.
        assert "Yusuf wants" in text

    def test_bare_region_code_is_never_swapped_alone(self):
        """'PA'/'MC' are two letters that occur inside ordinary words; only
        the "City, REGION" compound is safe to rewrite."""
        text = self.RENAME.apply_to_prose(
            "The PA system announced it.", CallerIdentityKind.PROSE_NAME_ZIP
        )
        assert text == "The PA system announced it."

    def test_the_state_spelled_out_resolves_to_the_locale_city(self):
        """4 retail tasks name the caller's state with no city beside it
        ("you live in Texas in zipcode 76171"). Leaving it stranded next to a
        swapped zip is the worst outcome — a Spanish zip in Florida."""
        text = self.RENAME.apply_to_prose(
            "You live in Pennsylvania in zipcode 19122.",
            CallerIdentityKind.PROSE_NAME_ZIP,
        )
        assert text == "You live in Murcia in zipcode 11532."

    def test_the_near_miss_address_keeps_its_gap(self):
        """Task 41/42's caller mistyped 443 Maple Drive as 445 and wants it
        corrected — the one-digit gap IS the task, so it has to survive the
        move to the locale rather than leaving a US street in Spanish prose."""
        rename = self.RENAME.model_copy(
            update={
                "old_address_variant": {
                    **self.RENAME.old_address,
                    "address1": "765 Broadway",
                },
                "new_address_variant": localized_address_variant(
                    self.RENAME.old_address,
                    self.RENAME.new_address,
                    {**self.RENAME.old_address, "address1": "765 Broadway"},
                ),
            }
        )
        # +2 on the source street number carries over to the locale's.
        assert rename.new_address_variant["address1"] == "Calle de Alcalá 81"
        text = rename.apply_to_prose(
            "You live at 763 Broadway but typed 765 Broadway, Suite 135.",
            CallerIdentityKind.PROSE_NAME_ZIP,
        )
        assert "Calle de Alcalá 79" in text  # the address on file
        assert "Calle de Alcalá 81, Piso 2" in text  # the near miss, gap intact
        assert "Broadway" not in text

    def test_a_dictated_destination_is_not_a_near_miss(self):
        """An address the caller dictates as a NEW destination differs in its
        place fields; it is task material, not identity material."""
        source = {
            "evaluation_criteria": {
                "actions": [
                    {
                        "arguments": {
                            "address1": "1234 Elm St",
                            "address2": "",
                            "city": "Springfield",
                            "state": "IL",
                            "country": "USA",
                            "zip": "62701",
                        }
                    }
                ]
            }
        }
        assert caller_address_variant(source, self.RENAME.old_address) is None

    def test_unit_only_destination_stays_canonical(self):
        """Task 17 asks for Suite 641 as a destination, not as a correction
        to caller identity data; the review packet requires it to remain the
        English-source address value."""
        source = {
            "evaluation_criteria": {
                "actions": [
                    {
                        "arguments": {
                            **self.RENAME.old_address,
                            "address2": "Suite 136",
                        }
                    }
                ]
            }
        }
        assert caller_address_variant(source, self.RENAME.old_address) is None

    def test_no_comma_city_state_moves_as_a_unit(self):
        """8 retail stems write the caller's origin without the comma
        ("Houston TX, 77004", "Seattle WA 98187") — the compound must still
        move whole, never leaving "Murcia TX"."""
        text = self.RENAME.apply_to_prose(
            "You are Yusuf Rossi from Philadelphia PA, 19122.",
            CallerIdentityKind.PROSE_NAME_ZIP,
        )
        assert text == "You are Iván Blanco from Murcia MC, 11532."

    def test_city_country_moves_as_a_unit(self):
        """4 retail stems anchor the caller as "living in Denver, USA, 80273";
        the country token must follow the city or the swapped prose claims a
        Spanish caller lives in the USA."""
        text = self.RENAME.apply_to_prose(
            "You are living in Philadelphia, USA, 19122.",
            CallerIdentityKind.PROSE_NAME_ZIP,
        )
        assert text == "You are living in Murcia, ESP, 11532."

    def test_locale_controls_place_order(self):
        """The structured DB fields stay fixed while prose uses locale order."""
        text = self.RENAME.apply_to_prose(
            "You are Yusuf Rossi from Philadelphia, PA, 19122.",
            CallerIdentityKind.PROSE_NAME_ZIP,
            AddressProseFormat(
                city_region="{region}, {city}",
                city_country="{country}, {region}, {city}",
            ),
        )
        assert text == "You are Iván Blanco from MC, Murcia, 11532."

    def test_locale_controls_full_address_order(self):
        rename = self.RENAME.model_copy(
            update={
                "old_address_variant": {
                    **self.RENAME.old_address,
                    "address1": "765 Broadway",
                },
                "new_address_variant": {
                    **self.RENAME.new_address,
                    "address1": "Calle de Alcalá 81",
                },
            }
        )
        text = rename.apply_to_prose(
            "Use 765 Broadway, Suite 135, Philadelphia, PA, 19122.",
            CallerIdentityKind.PROSE_NAME_ZIP,
            AddressProseFormat(
                city_region="{region} {city}",
                city_country="{country} {region} {city}",
                full_address=("{region} {city}, {address1}, {address2}, {postcode}"),
            ),
        )
        assert text == "Use MC Murcia, Calle de Alcalá 81, Piso 2, 11532."

    def test_caller_city_alias_follows_localized_identity(self):
        rename = self.RENAME.model_copy(
            update={"old_address": {**self.RENAME.old_address, "city": "New York"}}
        )
        text = rename.apply_to_prose(
            "You live in the Big Apple and call it your NYC place.",
            CallerIdentityKind.PROSE_NAME_ZIP,
        )
        assert text == "You live in Murcia and call it your Murcia place."

    def test_given_name_with_handle_expands_to_locale_full_name(self):
        text = self.RENAME.apply_to_prose(
            "You are Yusuf (yusuf_rossi_9620).",
            CallerIdentityKind.PROSE_NAME_ZIP,
        )
        assert text == "You are Iván Blanco (ivan_blanco_9620)."

    def test_postcode_decoys_follow_the_localized_postcode(self):
        rename = self.RENAME.model_copy(
            update={
                "zip_variant_pairs": (
                    ("19113", "11523"),
                    ("19121", "11531"),
                )
            }
        )
        text = rename.apply_to_prose(
            "Try 19113, then 19121, then 19122.",
            CallerIdentityKind.PROSE_NAME_ZIP,
        )
        assert text == "Try 11523, then 11531, then 11532."

    def test_bare_street_name_follows_the_caller(self):
        """Tasks 109/110 name the caller's street with no house number ("You
        live on Elm Avenue") — it must move with the address on file. A
        number-bearing mention is a dictated destination and stays put."""
        text = self.RENAME.apply_to_prose(
            "You live on Broadway in Philadelphia. Ship it to 915 Broadway.",
            CallerIdentityKind.PROSE_NAME_ZIP,
        )
        assert "You live on Calle de Alcalá in Murcia" in text
        assert "915 Broadway" in text

    def test_mid_line_house_number_takes_its_counter_word(self):
        """zh streets write the house number mid-line ('Jianshe Lu 280 Hao');
        the bare-street mention must come out as 'Jianshe Lu', never the
        malformed 'Jianshe Lu  Hao' that stranding the counter word leaves."""
        rename = self.RENAME.model_copy(
            update={
                "new_address": {
                    **self.RENAME.new_address,
                    "address1": "Jianshe Lu 280 Hao",
                }
            }
        )
        text = rename.apply_to_prose(
            "You live on Broadway in Philadelphia.",
            CallerIdentityKind.PROSE_NAME_ZIP,
        )
        assert "You live on Jianshe Lu in" in text
        assert "Hao" not in text

    def test_decoy_email_carries_its_gap(self):
        """A near-miss email the task turns on moves with the caller exactly
        as the near-miss address does, and the REAL email is replaced first —
        a given-name-dropped decoy is a substring of the real address, so the
        opposite order would corrupt both."""
        rename = self.RENAME.model_copy(
            update={
                "email_variant_pairs": (
                    ("rossi8836@example.com", "blanco8836@example.com"),
                )
            }
        )
        text = rename.apply_to_prose(
            "You have two emails: rossi8836@example.com and "
            "yusuf.rossi8836@example.com.",
            CallerIdentityKind.PROSE_NAME_ZIP,
        )
        assert "blanco8836@example.com and ivan.blanco8836@example.com" in text
        assert "rossi8836" not in text
        assert "rossi8836@example.com" in rename.renamed_source_values


class TestVariantEmail:
    """``variant_email`` / ``caller_email_variant_pairs`` — the deterministic
    transforms behind decoy-email swaps (the email analogue of
    :func:`localized_address_variant`)."""

    def test_dots_dropped(self):
        assert (
            variant_email(
                "daiki.sanchez1479@example.com",
                "daikisanchez1479@example.com",
                "raul.serrano3253@example.com",
            )
            == "raulserrano3253@example.com"
        )

    def test_given_name_dropped(self):
        assert (
            variant_email(
                "amelia.silva7872@example.com",
                "silva7872@example.com",
                "rocio.ruiz7726@example.com",
            )
            == "ruiz7726@example.com"
        )

    def test_digit_run_shifted(self):
        assert (
            variant_email(
                "sofia.thomas3069@example.com",
                "sofia.thomas3019@example.com",
                "irene.diaz1518@example.com",
            )
            == "irene.diaz1468@example.com"
        )

    def test_unrelated_email_is_not_a_variant(self):
        assert (
            variant_email("a.b1@example.com", "c.d2@example.com", "x.y3@example.com")
            is None
        )

    def test_pairs_skip_another_users_real_email(self):
        """A prose email that happens to be a near-miss of the caller's but is
        some OTHER user's actual address is a bystander, never rewritten."""
        source = {
            "user_scenario": {
                "instructions": {
                    "known_info": "You are X (a.b1@example.com); your friend "
                    "is a.b2@example.com."
                }
            }
        }
        with_guard = caller_email_variant_pairs(
            source,
            "a.b1@example.com",
            "c.d9@example.com",
            known_emails={"a.b2@example.com"},
        )
        assert with_guard == ()
        without_guard = caller_email_variant_pairs(
            source, "a.b1@example.com", "c.d9@example.com"
        )
        assert without_guard == (("a.b2@example.com", "c.d10@example.com"),)


class TestVariantPostcode:
    def test_decoys_keep_their_numeric_gap_and_new_format(self):
        source = {
            "user_scenario": {
                "instructions": {
                    "task_instructions": "Try 98178, then 98186, then 98187."
                }
            }
        }
        assert caller_zip_variant_pairs(source, "98187", "51626-691") == (
            ("98178", "51626-682"),
            ("98186", "51626-690"),
        )

    def test_unrelated_destination_postcode_is_not_a_variant(self):
        source = {
            "user_scenario": {
                "instructions": {
                    "reason_for_call": "Move from 95154 to Springfield 62701."
                }
            }
        }
        assert caller_zip_variant_pairs(source, "95154", "03001") == ()

    def test_eval_criteria_email_arguments_follow_the_swap(self):
        """A golden find_user_id_by_email must ask for the swapped address —
        the old one is prose the renamed caller never speaks."""
        source = {
            "id": "10",
            "user_scenario": {
                "instructions": {
                    "known_info": "You are pat_lee_1234 (pat.lee9@example.com)."
                }
            },
            "evaluation_criteria": {
                "actions": [
                    {"arguments": {"email": "pat.lee9@example.com"}},
                    {"arguments": {"email": "someone.else@example.com"}},
                ]
            },
        }
        localized = {
            "id": "10_es_identity",
            "user_scenario": {"instructions": {"known_info": ""}},
            "evaluation_criteria": {},
            "initial_state": {
                "initialization_data": {
                    "agent_data": {
                        "users": {
                            "carlos_gomez_1234": {
                                "name": {
                                    "first_name": "Carlos",
                                    "last_name": "Gomez",
                                },
                                "email": "carlos.gomez9@example.com",
                            }
                        }
                    }
                }
            },
        }
        db = {
            "users": {
                "pat_lee_1234": {
                    "name": {"first_name": "Pat", "last_name": "Lee"},
                    "email": "pat.lee9@example.com",
                }
            }
        }
        expected = expected_eval_criteria(source, localized, db)
        arguments = [a["arguments"]["email"] for a in expected["actions"]]
        assert arguments == [
            "carlos.gomez9@example.com",
            "someone.else@example.com",
        ]


class TestGenerator:
    def test_romanize_yields_ascii_user_id_tokens(self):
        from tau2.multilingual.invariants import CALLER_USER_ID_RE

        # Diacritics stripped; stroke letters transliterated (not dropped).
        cases = {
            ("José", "García"): ("jose", "garcia"),
            ("Ayşe", "Şahin"): ("ayse", "sahin"),
            ("Nguyễn", "Đức"): ("nguyen", "duc"),
            ("Paweł", "Łukasz"): ("pawel", "lukasz"),
        }
        for (first, last), (rf, rl) in cases.items():
            assert ent._romanize(first) == rf
            assert ent._romanize(last) == rl
            assert CALLER_USER_ID_RE.fullmatch(f"{rf}_{rl}_9957")

    def test_fold_to_ascii_preserves_case_and_folds_diacritics(self):
        # Display values entering an identity fold to plain ASCII: reward
        # compares tool-call args byte-exactly, and whether a voice agent's
        # transcription carries the diacritics is an orthography coin-flip.
        cases = {
            "Iván Sánchez": "Ivan Sanchez",
            "Łukasz Woźniak": "Lukasz Wozniak",
            "Đức Nguyễn": "Duc Nguyen",
            "Hauptstraße 12": "Hauptstrasse 12",
            "İstanbul": "Istanbul",
            "Kadıköy": "Kadikoy",
            "Şule Yıldız": "Sule Yildiz",
            "João Simões": "Joao Simoes",
            "Müller-Lüdenscheid": "Muller-Ludenscheid",
        }
        for raw, folded in cases.items():
            assert ent.fold_to_ascii(raw) == folded

    def test_fold_to_ascii_rejects_non_latin_residue(self):
        # A value that cannot fold to ASCII (non-Latin script) must fail
        # loudly, never ship a byte the reward comparison can trip on.
        with pytest.raises(ValueError, match="does not fold to ASCII"):
            ent.fold_to_ascii("Иван")

    def test_build_identity_is_deterministic(self):
        corpus = ent.load_corpus("es")
        user = {"address": {"country": "USA"}}
        a = ent.build_identity("emma_kim_9957", user, corpus)
        b = ent.build_identity("emma_kim_9957", user, corpus)
        assert a == b
        # user_id preserves the numeric suffix and is Latin/lowercase.
        assert a["user_id"].endswith("_9957")
        assert a["user_id"].islower()

    def test_built_identity_values_are_pure_ascii(self):
        # An accented corpus (es) must still yield an all-ASCII identity,
        # addresses included — the fold happens at build, corpora stay
        # authored in natural orthography.
        corpus = ent.load_corpus("es")
        user = {"address": {"country": "ESP"}}
        identity = ent.build_identity("emma_kim_9957", user, corpus)

        def walk(value):
            if isinstance(value, str):
                assert value.isascii(), value
            elif isinstance(value, dict):
                for v in value.values():
                    walk(v)

        walk(identity)
        structured = ent.build_structured_identity("Emma Kim", corpus)
        walk(structured)

    def test_collision_bumps_id_and_email_in_lockstep(self, monkeypatch):
        # A corpus with one name per pool forces two same-suffix callers to the
        # same base id — the dedup must bump id AND email together.
        corpus = ent.LocaleCorpus.model_validate(
            {
                "female_first_names": ["Ana"],
                "male_first_names": ["Ana"],
                "last_names": ["Gomez"],
                "email_domain": "example.com",
                "phone_number_format": "6## ## ## ##",
                "cities": [{"city": "Madrid", "region": "MD"}],
                "country": "ESP",
            }
        )
        db = {
            "users": {
                "noah_xa_1": {
                    "name": {"first_name": "Noah", "last_name": "Xa"},
                    "reservations": [],
                },
                "emma_yb_1": {
                    "name": {"first_name": "Emma", "last_name": "Yb"},
                    "reservations": [],
                },
            }
        }
        tasks = [
            {
                "id": "0",
                "user_scenario": {"instructions": {"known_info": "id noah_xa_1"}},
            },
            {
                "id": "1",
                "user_scenario": {"instructions": {"known_info": "id emma_yb_1"}},
            },
        ]
        monkeypatch.setattr(ent, "load_domain_db", lambda d: db)
        monkeypatch.setattr(ent, "load_domain_tasks", lambda d: tasks)

        idmap = ent.build_identity_map("airline", corpus)
        ids = [v["user_id"] for v in idmap.values()]
        emails = [v["email"] for v in idmap.values()]
        assert len(set(ids)) == 2  # bumped, so distinct
        assert len(set(emails)) == 2  # email bumped in lockstep, not left dangling
        # each email's local-part numeric tail matches its id's numeric tail
        for v in idmap.values():
            assert v["email"].split("@")[0].endswith(v["user_id"].rsplit("_", 1)[-1])

    @pytest.mark.parametrize("bad_name", ["Иван", "فاطمة"], ids=["cyrillic", "arabic"])
    def test_non_latin_corpus_names_fail_loudly_at_load(self, bad_name):
        """M4: a name that romanizes to nothing (non-Latin script) must fail
        corpus validation with an actionable error — not silently yield empty
        identities like ``__123`` / ``.123@example.com``."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="romanize"):
            ent.LocaleCorpus.model_validate(
                {
                    "female_first_names": [bad_name],
                    "male_first_names": ["Ivan"],
                    "last_names": ["Petrov"],
                    "email_domain": "example.com",
                    "phone_number_format": "09## ### ## ##",
                    "cities": [{"city": "Moscow", "region": "MOW"}],
                    "country": "RUS",
                }
            )

    def test_diacritic_latin_names_still_validate(self):
        """Diacritics are fine — romanization strips them; only scriptural
        non-Latin names are rejected."""
        corpus = ent.LocaleCorpus.model_validate(
            {
                "female_first_names": ["José", "Ayşe"],
                "male_first_names": ["Paweł"],
                "last_names": ["Nguyễn"],
                "email_domain": "example.com",
                "phone_number_format": "6## ## ## ##",
                "cities": [{"city": "Madrid", "region": "MD"}],
                "country": "ESP",
            }
        )
        assert isinstance(corpus, ent.LocaleCorpus)

    def test_identity_map_has_unique_ids_and_both_genders(self):
        corpus = ent.load_corpus("es")
        idmap = ent.build_identity_map("airline", corpus)
        assert idmap
        ids = [v["user_id"] for v in idmap.values()]
        assert len(set(ids)) == len(ids)
        genders = {v["gender"] for v in idmap.values()}
        assert genders == {"male", "female"}


class TestGenderDerivation:
    """Identity gender is inherited from the English source caller's name.

    Not sampled: ``noah_*`` callers are always male, ``emma_*`` always female,
    in every language — so a task keeps one caller gender across its plain and
    ``_identity`` variants, and the airline task set stays balanced 25/25.
    """

    def test_gender_follows_english_source_name(self):
        corpus = ent.load_corpus("es")
        user = {}
        assert ent.build_identity("noah_muller_9847", user, corpus)["gender"] == "male"
        assert ent.build_identity("emma_kim_9957", user, corpus)["gender"] == "female"
        # Locale first name drawn from the matching gender pool.
        ident = ent.build_identity("noah_muller_9847", user, corpus)
        assert ident["first_name"] in corpus.male_first_names

    def test_unisex_source_name_uses_reviewed_assignment(self):
        from tau2.multilingual.factory.name_genders import source_caller_gender

        # 'Chen' is unisex; the reviewed assignment (female) balances the
        # airline set to exactly 25/25 at task level.
        assert source_caller_gender("Chen") == "female"

    def test_uncovered_source_name_fails_loudly(self):
        from tau2.multilingual.factory.name_genders import source_caller_gender

        with pytest.raises(ValueError, match="name_genders"):
            source_caller_gender("Zzyzx")

    def test_catalog_covers_every_airline_caller(self):
        """Coverage guard for the closed catalog: every caller in the airline
        task set must resolve — an uncovered name would abort identity-map
        builds for every language."""
        from tau2.multilingual.factory.name_genders import source_caller_gender
        from tau2.multilingual.localize_lib import (
            find_caller_user_id,
            load_domain_tasks,
        )

        callers = {
            cid for t in load_domain_tasks("airline") if (cid := find_caller_user_id(t))
        }
        assert callers
        for caller in sorted(callers):
            assert source_caller_gender(caller.split("_")[0]) in ("male", "female")

    def test_catalog_covers_every_telecom_caller(self):
        """Coverage guard for STRUCTURED_NAME_PHONE callers: every distinct
        ``set_user_info`` given name across the telecom POOL (not just the
        seed split) must resolve, so identity-map builds never abort."""
        from tau2.multilingual.factory.name_genders import source_caller_gender
        from tau2.multilingual.invariants import caller_set_user_info
        from tau2.multilingual.localize_lib import data_dir

        pool_path = data_dir() / "tau2" / "domains" / "telecom" / "tasks.json"
        pool = json.loads(pool_path.read_text())
        names = {
            info["name"].split()[0] for t in pool if (info := caller_set_user_info(t))
        }
        assert names
        for name in sorted(names):
            assert source_caller_gender(name) in ("male", "female")

    def test_catalog_covers_every_retail_caller(self):
        """Coverage guard for PROSE_NAME_ZIP callers: every retail task must
        resolve to a DB user whose given name has a gender, so identity-map
        builds never abort.

        Stronger than the telecom guard because retail's caller is not a
        structured field: the task must first RESOLVE (by handle, email, or
        name+zip) and only then have its gender looked up. A task whose prose
        the resolver cannot pin to exactly one user fails here.
        """
        from tau2.multilingual.factory.name_genders import source_caller_gender
        from tau2.multilingual.invariants import resolve_caller_user_id
        from tau2.multilingual.localize_lib import data_dir

        retail_dir = data_dir() / "tau2" / "domains" / "retail"
        tasks = json.loads((retail_dir / "tasks.json").read_text())
        users = json.loads((retail_dir / "db.json").read_text())["users"]
        assert tasks
        unresolved = [
            task["id"] for task in tasks if resolve_caller_user_id(task, users) is None
        ]
        assert not unresolved, f"retail tasks with no resolvable caller: {unresolved}"
        for task in tasks:
            caller = users[resolve_caller_user_id(task, users)]
            assert source_caller_gender(caller["name"]["first_name"]) in (
                "male",
                "female",
            )

    def test_retail_callers_own_the_orders_their_tasks_name(self):
        """The retail resolver picks the RIGHT user, not merely a unique one.

        Uniqueness alone would be satisfied by a confident wrong answer, so
        this cross-checks against independent evidence: every ``#W...`` order
        id a task mentions must belong to the caller the resolver chose.
        """
        import re

        from tau2.multilingual.invariants import resolve_caller_user_id
        from tau2.multilingual.localize_lib import data_dir

        retail_dir = data_dir() / "tau2" / "domains" / "retail"
        tasks = json.loads((retail_dir / "tasks.json").read_text())
        db = json.loads((retail_dir / "db.json").read_text())
        checked = 0
        for task in tasks:
            mentioned = set(re.findall(r"#W\d+", json.dumps(task)))
            if not mentioned:
                continue
            caller = resolve_caller_user_id(task, db["users"])
            owned = set(db["users"][caller]["orders"])
            assert mentioned <= owned, (
                f"task {task['id']}: resolved caller {caller} does not own "
                f"{sorted(mentioned - owned)}"
            )
            checked += 1
        assert checked >= 100, f"only {checked} retail tasks named an order"

    def test_airline_task_level_gender_split_is_balanced(self):
        """The 50 airline tasks split exactly 25 male / 25 female."""
        from collections import Counter

        from tau2.multilingual.factory.name_genders import source_caller_gender
        from tau2.multilingual.localize_lib import (
            find_caller_user_id,
            load_domain_tasks,
        )

        counts = Counter(
            source_caller_gender(find_caller_user_id(t).split("_")[0])
            for t in load_domain_tasks("airline")
        )
        assert counts == Counter({"male": 25, "female": 25})


def _shipped_corpus_paths() -> list[Path]:
    from tau2.multilingual.localize_lib import multilingual_data_dir

    return sorted(multilingual_data_dir().glob(f"*/{ent.CORPUS_FILENAME}"))


class TestShippedCorpora:
    """Guard: every SHIPPED locale_corpus.yaml validates against LocaleCorpus.

    Parametrized over whatever corpora exist under data/tau2/multilingual/
    (only ``es`` today; the rest arrive with pack regeneration), so a stray or
    typo'd key in a committed corpus fails loudly in-PR via
    ``LocaleCorpus``'s ``extra="forbid"``.
    """

    @pytest.mark.parametrize(
        "path", _shipped_corpus_paths(), ids=lambda p: p.parent.name
    )
    def test_shipped_corpus_validates(self, path):
        corpus = ent.load_corpus(path.parent.name)
        assert isinstance(corpus, ent.LocaleCorpus)

    def test_at_least_one_shipped_corpus_exists(self):
        # If this fires, the parametrized guard above ran zero cases — the
        # corpora moved or the glob broke, not "all corpora are valid".
        assert _shipped_corpus_paths(), "no shipped locale_corpus.yaml found"


_CORPUS_LANGS = sorted(p.parent.name for p in _shipped_corpus_paths())
_SHIPPED_DOMAINS = ("airline", "telecom")
_REVIEWED_RETAIL_LANGS = ("es", "pt", "ko", "zh", "hi")


class TestReviewedRetailAddresses:
    @pytest.mark.parametrize("lang", _REVIEWED_RETAIL_LANGS)
    def test_generated_addresses_are_reviewed_city_bundles(self, lang):
        corpus = ent.load_corpus(lang)
        assert corpus.has_reviewed_city_addresses
        identity_map = ent.build_identity_map(
            "retail", corpus, name_order_for_language(lang)
        )
        assert ent.check_identity_map_addresses(identity_map, corpus) == []

    @pytest.mark.parametrize("lang", _REVIEWED_RETAIL_LANGS)
    def test_committed_retail_map_is_fresh(self, lang):
        corpus = ent.load_corpus(lang)
        committed = json.loads(ent.identity_map_path(lang, "retail").read_text())
        assert committed == ent.build_identity_map(
            "retail", corpus, name_order_for_language(lang)
        )

    @pytest.mark.parametrize("lang", _REVIEWED_RETAIL_LANGS)
    def test_manifest_records_review_and_fresh_hashes(self, lang):
        from tau2.multilingual.domain_profiles import get_domain_profile
        from tau2.multilingual.localize_lib import data_dir

        path = ent.identity_manifest_path(lang, "retail")
        manifest = ent.EntityLocalizationManifest.model_validate_json(path.read_text())
        assert manifest.generator_version == ent.IDENTITY_LOCALIZATION_VERSION
        assert manifest.address_review == ent.load_corpus(lang).address_review
        assert manifest.identity_count == 53
        assert manifest.task_count == 114

        profile = get_domain_profile("retail")
        source_path = (
            data_dir() / "tau2" / "domains" / "retail" / profile.source_tasks_filename
        )
        assert (
            manifest.corpus_sha256
            == hashlib.sha256(ent.corpus_path(lang).read_bytes()).hexdigest()
        )
        assert (
            manifest.source_tasks_sha256
            == hashlib.sha256(source_path.read_bytes()).hexdigest()
        )
        assert (
            manifest.localized_tasks_sha256
            == hashlib.sha256(
                ent.localized_task_set_path(lang, "retail").read_bytes()
            ).hexdigest()
        )
        for filename, digest in manifest.output_sha256.items():
            output = path.parent / filename
            assert digest == hashlib.sha256(output.read_bytes()).hexdigest()

    def test_mandarin_secondary_unit_stays_within_reviewed_range(self):
        identity_map = ent.build_identity_map(
            "retail", ent.load_corpus("zh"), name_order_for_language("zh")
        )
        for identity in identity_map.values():
            unit = identity["address"]["address2"]
            assert 1 <= int(re.search(r"\d+", unit).group()) <= 19

    @pytest.mark.parametrize("lang", _REVIEWED_RETAIL_LANGS)
    def test_task_17_keeps_source_destination_unit(self, lang):
        tasks = json.loads(ent.identity_task_set_path(lang, "retail").read_text())
        task = next(task for task in tasks if task["id"] == f"17_{lang}_identity")
        assert "Suite 641" in task["user_scenario"]["instructions"]["reason_for_call"]
        action = next(
            action
            for action in task["evaluation_criteria"]["actions"]
            if action["name"] == "modify_pending_order_address"
        )
        assert action["arguments"]["address2"] == "Suite 641"

    def test_mandarin_full_address_uses_large_to_small_order(self):
        tasks = json.loads(ent.identity_task_set_path("zh", "retail").read_text())
        task = next(task for task in tasks if task["id"] == "41_zh_identity")
        known_info = task["user_scenario"]["instructions"]["known_info"]
        users = task["initial_state"]["initialization_data"]["agent_data"]["users"]
        address = next(iter(users.values()))["address"]
        street = re.split(r"\d+", address["address1"], maxsplit=1)[0].strip()
        assert known_info.index(address["state"]) < known_info.index(street)

    @pytest.mark.parametrize("lang", _REVIEWED_RETAIL_LANGS)
    def test_new_york_aliases_follow_the_caller_identity(self, lang):
        tasks = json.loads(ent.identity_task_set_path(lang, "retail").read_text())
        for task_id in (f"33_{lang}_identity", f"34_{lang}_identity"):
            task = next(task for task in tasks if task["id"] == task_id)
            prose = json.dumps(task["user_scenario"], ensure_ascii=False)
            assert "Big Apple" not in prose
            assert "NYC" not in prose

    def test_korean_given_name_with_handle_uses_full_name(self):
        tasks = json.loads(ent.identity_task_set_path("ko", "retail").read_text())
        task = next(task for task in tasks if task["id"] == "43_ko_identity")
        known_info = task["user_scenario"]["instructions"]["known_info"]
        users = task["initial_state"]["initialization_data"]["agent_data"]["users"]
        user_id, user = next(iter(users.items()))
        full_name = compose_full_name(
            user["name"]["first_name"],
            user["name"]["last_name"],
            name_order_for_language("ko"),
        )
        assert f"{full_name} ({user_id})" in known_info

    def test_noncaller_destinations_remain_backed_by_task_data(self):
        """Seattle and Phoenix are destinations, not caller identity fields.
        Keeping both prose and golden action canonical avoids inventing a
        localized destination that the retail DB does not contain."""
        cases = (("pt", "33", "Seattle"), ("zh", "39", "Phoenix"))
        for lang, source_id, destination in cases:
            tasks = json.loads(ent.identity_task_set_path(lang, "retail").read_text())
            task = next(
                task for task in tasks if task["id"] == f"{source_id}_{lang}_identity"
            )
            prose = json.dumps(task["user_scenario"], ensure_ascii=False)
            actions = task["evaluation_criteria"]["actions"]
            assert destination in prose
            assert any(
                action["arguments"].get("city") == destination for action in actions
            )


class TestSecondaryAddressLine:
    """Retail records always carry a unit within the street address, and some
    tasks turn on the unit NEXT DOOR — so the locale line needs a number of
    its own. Airline's is Optional and its committed maps carry None."""

    @pytest.mark.parametrize("lang", ["es", "pt", "ko", "zh", "hi"])
    def test_every_retail_identity_has_a_movable_unit(self, lang):
        path = ent.identity_map_path(lang, "retail")
        if not path.exists():
            pytest.skip(f"{lang}/retail identity map not generated")
        for caller, entry in json.loads(path.read_text()).items():
            line = (entry.get("address") or {}).get("address2") or ""
            assert line, f"{lang}/{caller}: no secondary address line"
            assert len(re.findall(r"\d+", line)) == 1, (
                f"{lang}/{caller}: secondary line {line!r} has no single number to move"
            )

    @pytest.mark.parametrize("lang", ["es", "de"])
    def test_airline_identities_keep_an_empty_unit(self, lang):
        """The airline swap predates secondary lines and its committed maps
        carry None; drawing one for a corpus shared with retail would rewrite
        them."""
        path = ent.identity_map_path(lang, "airline")
        if not path.exists():
            pytest.skip(f"{lang}/airline identity map not generated")
        for caller, entry in json.loads(path.read_text()).items():
            address = entry.get("address")
            if address:
                assert address.get("address2") is None, caller


class TestShippedIdentityArtifacts:
    """The committed identity-variant artifacts must stay clean, per
    (language, domain) — telecom cases skip until the language's telecom arm
    file is seeded."""

    @pytest.mark.parametrize("domain", _SHIPPED_DOMAINS)
    @pytest.mark.parametrize("lang", _CORPUS_LANGS)
    def test_identity_set_regenerates_clean(self, lang, domain):
        if not ent.localized_task_set_path(lang, domain).exists():
            pytest.skip(f"{lang}/{domain} localized set not generated")
        names, problems, _ = ent.localize_entities(lang, domain, write=False)
        assert problems == []
        assert names

    @pytest.mark.parametrize("domain", _SHIPPED_DOMAINS)
    @pytest.mark.parametrize("lang", _CORPUS_LANGS)
    def test_committed_identity_map_is_pure_ascii(self, lang, domain):
        # Every committed identity value is plain ASCII (fold_to_ascii at
        # build time): these values become DB records and golden-action
        # arguments, which reward evaluation compares byte-exactly. The
        path = ent.identity_map_path(lang, domain)
        if not path.exists():
            pytest.skip(f"{lang}/{domain} identity map not generated")
        text = path.read_text()
        assert text.isascii(), (
            f"{path} carries non-ASCII identity values; regenerate with "
            f"`tau2 factory localize-entities --lang {lang} --domain {domain}`"
        )

    @pytest.mark.parametrize("domain", _SHIPPED_DOMAINS)
    @pytest.mark.parametrize("lang", _CORPUS_LANGS)
    def test_committed_identity_set_is_invariant_clean(self, lang, domain):
        from tau2.multilingual.domain_profiles import get_domain_profile
        from tau2.multilingual.localize_lib import (
            load_domain_db,
            load_domain_tasks,
            resolve_script_code,
        )

        path = ent.identity_task_set_path(lang, domain)
        if not path.exists():
            pytest.skip(f"{lang}/{domain} identity-variant set not generated")
        names = json.loads(path.read_text())
        script_code = (
            resolve_script_code(lang)
            if get_domain_profile(domain).tasks_translated
            else None
        )
        problems = check_task_set_localization(
            names,
            load_domain_tasks(domain),
            suffix=f"{lang}_identity",
            script_code=script_code,
            domain_db=load_domain_db(domain),
            domain=domain,
        )
        assert problems == []

    @pytest.mark.parametrize("domain", _SHIPPED_DOMAINS)
    @pytest.mark.parametrize("lang", _CORPUS_LANGS)
    def test_sidecar_covers_every_task_and_matches_generated_names(self, lang, domain):
        """The caller-gender sidecar covers EVERY localized task id (plain and
        ``_identity``), the pair shares one gender, and that gender matches
        the generated locale name's corpus pool AND the English source
        caller's name."""
        from tau2.multilingual.domain_profiles import get_domain_profile
        from tau2.multilingual.factory.name_genders import source_caller_gender
        from tau2.multilingual.localize_lib import load_domain_db

        sidecar_path = ent.gender_sidecar_path(lang, domain)
        if not sidecar_path.exists():
            pytest.skip(f"{lang}/{domain} gender sidecar not generated")
        sidecar = ent.load_gender_sidecar(lang, domain)
        corpus = ent.load_corpus(lang)
        profile = get_domain_profile(domain)
        db = load_domain_db(domain)
        identity_map = json.loads(ent.identity_map_path(lang, domain).read_text())
        localized = json.loads(ent.localized_task_set_path(lang, domain).read_text())

        assert set(sidecar.values()) <= {"male", "female"}
        for task in localized:
            plain_id = task["id"]
            identity_id = f"{plain_id}_identity"
            gender = sidecar.get(plain_id)
            assert gender is not None, f"no sidecar entry for {plain_id}"
            assert sidecar.get(identity_id) == gender, (
                f"{plain_id} / {identity_id} disagree on gender"
            )
            caller = ent._caller_key(task, db, profile)
            identity = identity_map[caller]
            # Generated locale name drawn from the pool of the sidecar gender
            # (committed names are the ASCII fold of the corpus spelling).
            pool = (
                corpus.female_first_names
                if gender == "female"
                else corpus.male_first_names
            )
            assert identity["gender"] == gender
            assert identity["first_name"] in {ent.fold_to_ascii(n) for n in pool}
            # ... and the gender is the English source caller's name gender.
            # (The caller key IS the English identity in every kind: the
            # user-id handle for airline, and the full name for telecom.)
            source_given = (
                caller.split("_")[0]
                if profile.caller_identity.value == "user_id_handle"
                else caller.split()[0]
            )
            assert source_caller_gender(source_given) == gender


class TestCommittedMapMatchesCorpus:
    """Guard: a committed identity map is exactly what its corpus produces.

    The corpus is the reviewed raw material; the identity map is a pure,
    deterministic function of it (:func:`build_identity_map`). Editing the
    corpus without re-running ``tau2 factory localize-entities`` leaves the
    shipped map — and the identity task set derived from it — carrying stale
    names. That is precisely how the es pack shipped with the corpus's
    unaccented Spanish spellings baked into 59 caller identities.
    """

    @pytest.mark.parametrize("domain", _SHIPPED_DOMAINS)
    @pytest.mark.parametrize("lang", _CORPUS_LANGS)
    def test_identity_map_is_a_fresh_build_of_the_corpus(self, lang, domain):
        path = ent.identity_map_path(lang, domain)
        if not path.exists():
            pytest.skip(f"{lang}/{domain} identity map not generated")
        committed = json.loads(path.read_text())
        stale = (
            f"{path.name} is stale — re-run "
            f"`tau2 factory localize-entities --lang {lang} --domain {domain}`"
        )
        # The map is a function of the corpus AND the pack's declared name
        # order — a family-first language's display names come out reordered.
        rebuilt = ent.build_identity_map(
            domain, ent.load_corpus(lang), name_order_for_language(lang)
        )
        assert committed == rebuilt, stale

    @pytest.mark.parametrize("domain", _SHIPPED_DOMAINS)
    @pytest.mark.parametrize("lang", _CORPUS_LANGS)
    def test_every_localized_name_comes_from_the_corpus_pools(self, lang, domain):
        """Both name halves — not just the given name — are corpus values, so a
        corpus respelling that never reached the map fails here too. Committed
        names are the ASCII fold of the corpus spelling (``fold_to_ascii`` at
        build), so membership is checked on the folded pools."""
        path = ent.identity_map_path(lang, domain)
        if not path.exists():
            pytest.skip(f"{lang}/{domain} identity map not generated")
        corpus = ent.load_corpus(lang)
        given = {
            ent.fold_to_ascii(n)
            for n in (*corpus.female_first_names, *corpus.male_first_names)
        }
        last = {ent.fold_to_ascii(n) for n in corpus.last_names}
        payload = json.loads(path.read_text())
        for key, identity in payload.items():
            assert identity["first_name"] in given, key
            assert identity["last_name"] in last, key

    @pytest.mark.parametrize("domain", _SHIPPED_DOMAINS)
    @pytest.mark.parametrize("lang", _CORPUS_LANGS)
    def test_derived_ids_and_emails_stay_ascii(self, lang, domain):
        """Accented corpus names are correct locale orthography, but the
        derived handles are romanized: a non-ASCII user_id or email local part
        means ``_romanize`` was bypassed somewhere."""
        path = ent.identity_map_path(lang, domain)
        if not path.exists():
            pytest.skip(f"{lang}/{domain} identity map not generated")
        for key, identity in json.loads(path.read_text()).items():
            for field in ("user_id", "email"):
                value = identity.get(field)
                if value is not None:
                    assert value.isascii(), f"{key}: non-ASCII {field} {value!r}"


class TestShippedTelecomIdentityMap:
    """The committed telecom identity maps are keyed by the English pool
    caller names (the breaking artifact change of the caller-diversity PR),
    with locale identities unique across the map."""

    @pytest.mark.parametrize("lang", _CORPUS_LANGS)
    def test_keyed_by_pool_names_with_unique_identities(self, lang):
        from tau2.multilingual.factory.caller_diversity import load_caller_pool

        path = ent.identity_map_path(lang, "telecom")
        if not path.exists():
            pytest.skip(f"{lang}/telecom identity map not generated")
        idmap = json.loads(path.read_text())
        pool = load_caller_pool("telecom")
        assert set(idmap) == {c.full_name for c in pool.callers}
        emails = [v["email"] for v in idmap.values()]
        assert len(set(emails)) == len(emails)
        for english_name, identity in idmap.items():
            from tau2.multilingual.factory.name_genders import (
                source_caller_gender,
            )

            assert identity["gender"] == source_caller_gender(english_name.split()[0])


class TestVoiceGenderPinning:
    def test_gender_sidecar_lookup_covers_plain_and_identity_ids(self):
        sidecar = ent.gender_sidecar_path("es", "airline")
        if not sidecar.exists():
            pytest.skip("es gender sidecar not generated")
        gender = ent.load_gender_sidecar("es", "airline")
        assert gender
        plain = {k for k in gender if not k.endswith("_identity")}
        identity = {k for k in gender if k.endswith("_identity")}
        assert plain and identity and len(plain) == len(identity)
        assert set(gender.values()) <= {"male", "female"}
        # Plain localized tasks resolve too (voice pinning covers ALL tasks).
        assert ent.caller_gender_for_task("0_es", "es", "airline") == gender["0_es"]
        assert (
            ent.caller_gender_for_task("0_es_identity", "es", "airline")
            == gender["0_es_identity"]
        )
        # Unknown ids / languages without a sidecar fall back to None.
        assert ent.caller_gender_for_task("999_zz", "zz") is None

    def test_sidecar_read_is_cached_until_the_file_changes(self, tmp_path, monkeypatch):
        """The per-task gender lookup must not re-read the JSON per call: the
        second load with an unchanged mtime is a cache hit, and rewriting the
        file (new mtime) invalidates it."""
        import json as json_mod
        import os

        sidecar_dir = tmp_path / "xx"
        sidecar_dir.mkdir()
        monkeypatch.setattr(ent, "multilingual_data_dir", lambda: tmp_path)
        path = ent.gender_sidecar_path("xx", "airline")
        path.write_text(json_mod.dumps({"0_xx": "female"}))

        reads = []
        real_loads = ent.json.loads
        monkeypatch.setattr(
            ent.json, "loads", lambda s: reads.append(1) or real_loads(s)
        )
        assert ent.load_gender_sidecar("xx", "airline") == {"0_xx": "female"}
        assert ent.load_gender_sidecar("xx", "airline") == {"0_xx": "female"}
        assert len(reads) == 1  # second call served from the cache

        path.write_text(json_mod.dumps({"0_xx": "male"}))
        os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 1))
        assert ent.load_gender_sidecar("xx", "airline") == {"0_xx": "male"}
        assert len(reads) == 2  # rewrite (new mtime) invalidated the cache

    def test_persona_pinned_to_caller_gender(self):
        from tau2.data_model.voice import SynthesisConfig
        from tau2.multilingual.loader import load_language_packs
        from tau2.multilingual.registry import get_language_pack
        from tau2.user_simulation_voice_presets import sample_voice_config

        load_language_packs()
        pack = get_language_pack("es")
        genders = {
            pid: (p.tags or {}).get("gender") for pid, p in pack.personas.items()
        }

        for want in ("male", "female"):
            cfg = sample_voice_config(
                seed=7,
                synthesis_config=SynthesisConfig(),
                complexity="regular",
                persona_name="es",
                task_id="0_es_identity",
                run_seed=42,
                caller_gender=want,
            )
            assert genders[cfg.persona_name] == want

    def test_live_path_pins_plain_and_identity_tasks(self):
        """`get_or_load_task_voice_config` consults the sidecar for EVERY
        localized task id — the plain set included, not just `_identity`."""
        from tau2.data_model.voice import SynthesisConfig
        from tau2.multilingual.loader import load_language_packs
        from tau2.multilingual.registry import get_language_pack
        from tau2.user_simulation_voice_presets import get_or_load_task_voice_config

        if not ent.gender_sidecar_path("es", "airline").exists():
            pytest.skip("es gender sidecar not generated")
        load_language_packs()
        pack = get_language_pack("es")
        genders = {
            pid: (p.tags or {}).get("gender") for pid, p in pack.personas.items()
        }

        for task_id in ("0_es", "0_es_identity", "7_es", "7_es_identity"):
            want = ent.caller_gender_for_task(task_id, "es", "airline")
            assert want in ("male", "female")
            for run_seed in (0, 7, 42):
                cfg = get_or_load_task_voice_config(
                    "airline",
                    task_id,
                    task_seed=run_seed + 3,
                    complexity="regular",
                    synthesis_config=SynthesisConfig(),
                    persona_name="es",
                    run_seed=run_seed,
                )
                assert genders[cfg.persona_name] == want

    def test_english_persona_pools_are_catalogued_and_mixed_gender(self):
        """Every stock English persona resolves in the closed name catalog
        AND both preset pools carry both genders — the caller-gender filter
        in `sample_voice_config` must never guess or empty a pool."""
        from tau2.data_model.voice_personas import (
            CONTROL_PERSONA_NAMES,
            REGULAR_PERSONA_NAMES,
        )
        from tau2.user_simulation_voice_presets import _english_persona_gender

        for pool in (CONTROL_PERSONA_NAMES, REGULAR_PERSONA_NAMES):
            genders = {name: _english_persona_gender(name) for name in pool}
            assert set(genders.values()) == {"male", "female"}, genders

    def test_english_persona_pinned_to_caller_gender(self):
        from tau2.data_model.voice import SynthesisConfig
        from tau2.user_simulation_voice_presets import (
            _english_persona_gender,
            sample_voice_config,
        )

        for want in ("male", "female"):
            for seed in (0, 7, 42):
                for complexity in ("control", "regular"):
                    cfg = sample_voice_config(
                        seed=seed,
                        synthesis_config=SynthesisConfig(),
                        complexity=complexity,
                        caller_gender=want,
                    )
                    assert _english_persona_gender(cfg.persona_name) == want

    def test_english_live_path_pins_en_arm_tasks(self):
        """`get_or_load_task_voice_config` with NO persona override consults
        the ``en`` sidecar: seed-diversified ``_en`` task ids get a
        gender-matched stock English persona (bypassing pre-sampled configs,
        which bake in an arbitrary-gender persona)."""
        from tau2.data_model.voice import SynthesisConfig
        from tau2.user_simulation_voice_presets import (
            _english_persona_gender,
            get_or_load_task_voice_config,
        )

        if not ent.gender_sidecar_path("en", "telecom").exists():
            pytest.skip("en telecom gender sidecar not generated")
        sidecar = ent.load_gender_sidecar("en", "telecom")
        by_gender = {
            gender: next(tid for tid, g in sidecar.items() if g == gender)
            for gender in ("male", "female")
        }
        for want, task_id in by_gender.items():
            cfg = get_or_load_task_voice_config(
                "telecom",
                task_id,
                task_seed=11,
                complexity="regular",
                synthesis_config=SynthesisConfig(),
            )
            assert _english_persona_gender(cfg.persona_name) == want


class TestCustomerPatchIntegrityGate:
    """Mutation tests for the set-level gate on STRUCTURED_NAME_PHONE
    identity sets: generation refuses to WRITE bad artifacts, but only the
    gate protects a hand-edited/regressed COMMITTED file — each corruption
    here used to pass the gate silently."""

    def _derived(self):
        db = _toy_telecom_db()
        src = [_toy_telecom_task("[toy]a")]
        localized = [{**copy.deepcopy(t), "id": f"{t['id']}_hi"} for t in src]
        out, problems = derive_identity_tasks(
            localized, src, db, TELECOM_MAP, "hi", None, domain="telecom"
        )
        assert problems == []
        return db, src, out

    def _gate(self, out, src, db):
        return check_task_set_localization(
            out, src, suffix="hi_identity", script_code=None, domain_db=db
        )

    def _patch_customers(self, task):
        return task["initial_state"]["initialization_data"]["agent_data"]["customers"]

    def test_unmutated_set_passes(self):
        db, src, out = self._derived()
        assert self._gate(out, src, db) == []

    def test_set_user_info_reverted_to_source_name_is_caught(self):
        db, src, out = self._derived()
        actions = out[0]["initial_state"]["initialization_actions"]
        set_info = next(a for a in actions if a["func_name"] == "set_user_info")
        set_info["arguments"]["name"] = "Marcus Delgado"
        problems = self._gate(out, src, db)
        assert any("does not match the customers-patch rename" in p for p in problems)

    def test_bystander_rename_is_caught(self):
        db, src, out = self._derived()
        customers = self._patch_customers(out[0])
        bystander = next(c for c in customers if c["customer_id"] == "C1002")
        bystander["full_name"] = "Sara Johnson"
        problems = self._gate(out, src, db)
        assert any("bystander customer 'C1002'" in p for p in problems)

    def test_caller_phone_mutation_is_caught(self):
        db, src, out = self._derived()
        customers = self._patch_customers(out[0])
        caller = next(c for c in customers if c["customer_id"] == "C1001")
        caller["phone_number"] = "555-999-0000"
        problems = self._gate(out, src, db)
        assert any("set_user_info phone" in p for p in problems)

    def test_caller_dob_mutation_is_caught(self):
        db, src, out = self._derived()
        customers = self._patch_customers(out[0])
        caller = next(c for c in customers if c["customer_id"] == "C1001")
        # Reverting to the DB's canonical DOB must ALSO be caught — the
        # baseline is the source's effective (diversified) record.
        caller["date_of_birth"] = "1985-06-15"
        problems = self._gate(out, src, db)
        assert any("only full_name/email/phone_number" in p for p in problems)

    def test_dropped_bystander_is_caught(self):
        db, src, out = self._derived()
        customers = self._patch_customers(out[0])
        customers[:] = [c for c in customers if c["customer_id"] != "C1002"]
        problems = self._gate(out, src, db)
        assert any("effective customer ids in order" in p for p in problems)

    def test_leftover_source_caller_mention_is_caught(self):
        db, src, out = self._derived()
        ins = out[0]["user_scenario"]["instructions"]
        ins["task_instructions"] += " Your name is Marcus Delgado."
        problems = self._gate(out, src, db)
        assert any("still names the source caller" in p for p in problems)

    def test_compose_merges_over_preexisting_initialization_data(self):
        """Diversified seed tasks ALL carry initialization_data — the swap
        must merge (replace only the customers list) and preserve sibling
        agent_data keys and user_data, never wholesale-assign or refuse."""
        db = _toy_telecom_db()
        src = [_toy_telecom_task("[toy]a")]
        init = src[0]["initial_state"]["initialization_data"]
        init["agent_data"]["phone_number_aliases"] = {"+44 7700 900123": "555-123-2001"}
        init["user_data"] = {"device": "toy-phone"}
        localized = [{**copy.deepcopy(t), "id": f"{t['id']}_hi"} for t in src]
        out, problems = derive_identity_tasks(
            localized, src, db, TELECOM_MAP, "hi", None, domain="telecom"
        )
        assert problems == []
        out_init = out[0]["initial_state"]["initialization_data"]
        # Pre-existing aliases ride through untouched; the swap adds none
        # (the caller's phone number stays canonical).
        assert out_init["agent_data"]["phone_number_aliases"] == {
            "+44 7700 900123": "555-123-2001",
        }
        assert out_init["user_data"] == {"device": "toy-phone"}
        customers = {c["customer_id"]: c for c in out_init["agent_data"]["customers"]}
        assert customers["C1001"]["full_name"] == "Jorge Gil"
        assert customers["C1001"]["date_of_birth"] == "1980-07-09"


def _english_arm_domains() -> list[str]:
    """Profiled domains that ship an English arm task set."""
    from tau2.multilingual.domain_profiles import DOMAIN_PROFILES
    from tau2.multilingual.localize_lib import multilingual_data_dir

    return sorted(
        domain
        for domain in DOMAIN_PROFILES
        if (multilingual_data_dir() / "en" / f"{domain}_tasks_en.json").exists()
    )


class TestEnglishGenderSidecar:
    """The English arm's caller-gender sidecar (``english-gender-sidecar``).

    The coverage guard here is the point: without a committed sidecar an
    English arm silently falls back to the domain's pre-sampled
    ``tasks_voice.json`` personas, whose gender is uncorrelated with the
    caller's name — airline shipped that way and was 25/50 contradictory.
    A missing sidecar is a failure, not a skip.
    """

    def test_at_least_one_english_arm_exists(self):
        # If this fires the parametrized guards below ran zero cases.
        assert _english_arm_domains(), "no <domain>_tasks_en.json found"

    @pytest.mark.parametrize("domain", _english_arm_domains())
    def test_committed_sidecar_covers_every_english_task(self, domain):
        sidecar = ent.load_gender_sidecar("en", domain)
        assert sidecar, (
            f"{domain} ships an English arm but no caller-gender sidecar — "
            f"run `tau2 factory english-gender-sidecar --domain {domain}`"
        )
        arm = json.loads(
            ent.multilingual_data_dir()
            .joinpath("en", f"{domain}_tasks_en.json")
            .read_text()
        )
        arm_ids = {task["id"] for task in arm}
        assert arm_ids == set(sidecar)
        assert set(sidecar.values()) <= {"male", "female"}

    @pytest.mark.parametrize("domain", _english_arm_domains())
    def test_committed_sidecar_regenerates_identically(self, domain):
        """Deterministic and idempotent: the committed file is what the verb
        emits today, so a stale sidecar cannot outlive its arm file."""
        committed = ent.load_gender_sidecar("en", domain)
        assert ent.build_english_gender_sidecar(domain) == committed

    @pytest.mark.parametrize("domain", _english_arm_domains())
    @pytest.mark.parametrize("lang", _CORPUS_LANGS)
    def test_english_sidecar_agrees_with_localized_sidecars(self, lang, domain):
        """Cross-source check: the localized sidecars are built from the
        identity map, the English one straight off the arm file. One caller,
        one gender — the two derivations must not drift."""
        if not ent.gender_sidecar_path(lang, domain).exists():
            pytest.skip(f"{lang}/{domain} gender sidecar not generated")
        english = ent.load_gender_sidecar("en", domain)
        localized = ent.load_gender_sidecar(lang, domain)
        shared = 0
        for task_id, gender in localized.items():
            source_id = task_id.removesuffix("_identity").removesuffix(f"_{lang}")
            if (expected := english.get(f"{source_id}_en")) is not None:
                shared += 1
                assert gender == expected, f"{task_id}: {gender} != en {expected}"
        assert shared, f"no task ids shared between en and {lang} for {domain}"


class TestEnglishCallerGender:
    """``english_caller_gender`` dispatches on the profile's identity kind."""

    def test_user_id_handle_reads_the_handle(self):
        from tau2.multilingual.domain_profiles import get_domain_profile

        profile = get_domain_profile("airline")
        task = {
            "user_scenario": {"instructions": {"known_info": "user id emma_kim_9957"}}
        }
        assert ent.english_caller_gender(task, profile) == "female"

    def test_no_caller_is_none_not_a_guess(self):
        from tau2.multilingual.domain_profiles import get_domain_profile

        profile = get_domain_profile("airline")
        task = {"user_scenario": {"instructions": {"known_info": "no handle here"}}}
        assert ent.english_caller_gender(task, profile) is None
