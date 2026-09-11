# Copyright Sierra
"""Locale name ORDER — the caller says the name the agent's record holds.

The defect these guard: the identity generator composed every localized
caller as ``f"{given} {family}"``, so a Mandarin caller introduced itself
"Hu Wei" (the only order a native says) while the agent's customer record
held "Wei Hu", and ``get_customer_by_name`` — an exact match modulo
case/accents — missed. 57 of 114 zh telecom tasks are the name+DOB auth arm,
so the mismatch was not an occasional challenge but a floor under the whole
arm.
"""

import json

import pytest

from tau2.multilingual.factory import entity_localization as ent
from tau2.multilingual.invariants import (
    caller_set_user_info,
    identity_rename,
)
from tau2.multilingual.localize_lib import load_domain_db
from tau2.multilingual.names import (
    NameOrder,
    compose_full_name,
    name_order_for_language,
    split_full_name,
)
from tau2.utils.text_match import fold_for_match

# Declared family-first in their packs; the rest of the benchmark is
# given-first and must stay byte-identical.
FAMILY_FIRST_LANGS = ("zh", "ko")
GIVEN_FIRST_LANGS = ("en", "es", "hi", "pt")


class TestComposeAndSplit:
    def test_given_first_is_the_english_order(self):
        assert compose_full_name("Wei", "Hu") == "Wei Hu"

    def test_family_first_reorders(self):
        assert compose_full_name("Wei", "Hu", NameOrder.FAMILY_FIRST) == "Hu Wei"

    @pytest.mark.parametrize("order", list(NameOrder))
    @pytest.mark.parametrize(
        "given,family",
        [("Wei", "Hu"), ("Ana", "de Souza"), ("Minseo", "Park")],
    )
    def test_round_trip(self, given, family, order):
        """split(compose(x)) == x, so roles survive a write/read cycle."""
        assert split_full_name(compose_full_name(given, family, order), order) == (
            given,
            family,
        )


class TestDeclaredOrder:
    @pytest.mark.parametrize("lang", FAMILY_FIRST_LANGS)
    def test_family_first_packs(self, lang):
        assert name_order_for_language(lang) is NameOrder.FAMILY_FIRST

    @pytest.mark.parametrize("lang", GIVEN_FIRST_LANGS)
    def test_given_first_packs(self, lang):
        assert name_order_for_language(lang) is NameOrder.GIVEN_FIRST

    @pytest.mark.parametrize("lang", [None, "", "xx"])
    def test_unknown_language_keeps_the_english_order(self, lang):
        """An unregistered or absent language must not change any behavior."""
        assert name_order_for_language(lang) is NameOrder.GIVEN_FIRST


class TestGeneratedIdentities:
    """Roles stay roles; only the display string reorders."""

    def test_structured_identity_display_name_follows_the_order(self):
        corpus = ent.load_corpus("zh")
        identity = ent.build_structured_identity(
            "Allison Reeves", corpus, NameOrder.FAMILY_FIRST
        )
        given, family = identity["first_name"], identity["last_name"]
        assert identity["full_name"] == f"{family} {given}"
        # first_name is still the GIVEN name — the roles never swap, or the
        # gender routing and email derivation would follow the wrong half.
        assert identity["email"].startswith(f"{given.lower()}.{family.lower()}")

    def test_same_corpus_draw_regardless_of_order(self):
        """Order changes presentation only: the drawn name pair is identical,
        so flipping a language does not reshuffle who its callers are."""
        corpus = ent.load_corpus("zh")
        given_first = ent.build_structured_identity("Allison Reeves", corpus)
        family_first = ent.build_structured_identity(
            "Allison Reeves", corpus, NameOrder.FAMILY_FIRST
        )
        assert (given_first["first_name"], given_first["last_name"]) == (
            family_first["first_name"],
            family_first["last_name"],
        )
        assert given_first["email"] == family_first["email"]


class TestShippedTelecomIdentitiesAreLookupConsistent:
    """The regression that started this: what the caller says must be exactly
    what ``get_customer_by_name`` matches against."""

    @pytest.mark.parametrize("lang", FAMILY_FIRST_LANGS + ("es",))
    def test_caller_name_matches_the_agent_record(self, lang):
        path = ent.identity_task_set_path(lang, "telecom")
        if not path.exists():
            pytest.skip(f"{lang}/telecom identity set not generated")
        tasks = json.loads(path.read_text())
        assert tasks
        for task in tasks:
            caller = caller_set_user_info(task)
            assert caller is not None, task["id"]
            records = task["initial_state"]["initialization_data"]["agent_data"][
                "customers"
            ]
            # fold_for_match is exactly what the telecom tool applies.
            wanted = fold_for_match(caller["name"])
            assert any(fold_for_match(r["full_name"]) == wanted for r in records), (
                f"{task['id']}: caller says '{caller['name']}' but no customer "
                "record carries that name — the name lookup cannot succeed"
            )

    @pytest.mark.parametrize("lang", FAMILY_FIRST_LANGS)
    def test_family_first_language_says_the_family_name_first(self, lang):
        path = ent.identity_map_path(lang, "telecom")
        if not path.exists():
            pytest.skip(f"{lang}/telecom identity map not generated")
        identities = json.loads(path.read_text()).values()
        assert identities
        for identity in identities:
            given, family = identity["first_name"], identity["last_name"]
            assert identity["full_name"] == f"{family} {given}"


class TestEnglishPromptModeKeepsTheLocaleOrder:
    """English-prompt mode is the default: it swaps the user simulator's
    instructions to English but keeps the LOCALE caller identity. Composing
    that name given-first here would hand the simulator a different name from
    the one in the agent's record — the original bug, reintroduced one layer
    up."""

    @pytest.mark.parametrize("lang", FAMILY_FIRST_LANGS)
    def test_english_instructions_carry_the_family_first_name(self, lang):
        """Airline: the USER_ID_HANDLE shape composes the display name from
        name ROLES, so this is where a given-first default would silently
        rewrite 'Song Xue' as 'Xue Song' in the English instructions."""
        from tau2.data_model.tasks import Task
        from tau2.multilingual.english_prompts import english_user_task_variant

        path = ent.identity_task_set_path(lang, "airline")
        if not path.exists():
            pytest.skip(f"{lang}/airline identity set not generated")
        identities = json.loads(ent.identity_map_path(lang, "airline").read_text())
        by_user_id = {e["user_id"]: e for e in identities.values()}

        for raw in json.loads(path.read_text())[:5]:
            task = Task.model_validate(raw)
            variant = english_user_task_variant(
                task, domain="airline", task_set_name=f"airline_{lang}_identity"
            )
            known_info = variant.user_scenario.instructions.known_info or ""
            user_id = next(uid for uid in by_user_id if uid in known_info)
            expected = compose_full_name(
                by_user_id[user_id]["first_name"],
                by_user_id[user_id]["last_name"],
                NameOrder.FAMILY_FIRST,
            )
            assert expected in known_info, (
                f"{task.id}: English-mode known_info should name the caller "
                f"'{expected}' (the order the agent's record holds) — got "
                f"{known_info!r}"
            )


class TestTelecomRenameRolesFollowTheLocaleOrder:
    """The display strings a telecom patch carries are verbatim, but the ROLE
    split behind structured {first_name, last_name} swaps still has to read
    them in the order they were written. Splitting "Park Minseo" given-first
    would call the family name a given name."""

    @pytest.mark.parametrize("lang", FAMILY_FIRST_LANGS)
    def test_roles_recompose_into_the_display_name(self, lang):
        path = ent.identity_task_set_path(lang, "telecom")
        if not path.exists():
            pytest.skip(f"{lang}/telecom identity set not generated")
        db = load_domain_db("telecom")
        source_by_id = {t["id"]: t for t in ent.load_domain_tasks("telecom")}
        seen = 0
        for task in json.loads(path.read_text()):
            source = source_by_id.get(task["id"].removesuffix(f"_{lang}_identity"))
            if source is None:
                continue
            rename = identity_rename(source, task, db, NameOrder.FAMILY_FIRST)
            if rename is None or not rename.renames_caller:
                continue
            seen += 1
            # Source side is English, so its roles read given-first.
            assert compose_full_name(*rename.old_name) == rename.old_full_name
            assert (
                compose_full_name(*rename.new_name, NameOrder.FAMILY_FIRST)
                == rename.new_full_name
            ), f"{task['id']}: roles {rename.new_name} do not spell the caller"
        assert seen, f"{lang}: no telecom caller rename found to check"


class TestPlainSetsClaimNoRename:
    """A plain (non-identity) localized set keeps the English caller. Reading
    its customers patch as if it were a locale identity would invent a rename
    and rewrite prose that must stay verbatim."""

    @pytest.mark.parametrize("lang", FAMILY_FIRST_LANGS)
    def test_no_rename_for_the_plain_set(self, lang):
        path = ent.localized_task_set_path(lang, "telecom")
        if not path.exists():
            pytest.skip(f"{lang}/telecom localized set not generated")
        db = load_domain_db("telecom")
        source_by_id = {t["id"]: t for t in ent.load_domain_tasks("telecom")}
        for task in json.loads(path.read_text()):
            source = source_by_id.get(task["id"].removesuffix(f"_{lang}"))
            if source is None:
                continue
            rename = identity_rename(source, task, db, name_order_for_language(lang))
            assert rename is None or not rename.renames_caller, (
                f"{task['id']}: plain localized set reports a caller rename "
                f"({rename.old_full_name!r} -> {rename.new_full_name!r})"
            )
