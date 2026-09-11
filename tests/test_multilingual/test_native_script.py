# Copyright Sierra
"""Tests for the native-script DB ablation (`--native-script` identity builds).

Three layers:

- the pure spell-out machinery (`devanagari_spellout` traps, `han_spellout`
  gloss discipline, the script-dispatching `native_name_spellout`);
- the truthful-prompt plumbing (the pack `agent_native_script_db_clause`
  flip, which must raise for a pack without one);
- the COMMITTED artifacts of both native builds (hi/zh retail): ids, script
  invariants (names native, user_id/email still ASCII — no fold, and nothing
  but the name flips), zh gloss coverage of every character, payload ==
  recomputation, gender-sidecar fallback, and the frozen retail_30 prefix
  the ablation pools run on.
"""

import json

import pytest

from tau2.multilingual.factory.entity_localization import (
    caller_gender_for_task,
    load_corpus,
    load_gender_sidecar,
    native_identity_map_path,
    native_identity_task_set_path,
)
from tau2.multilingual.names import compose_full_name, name_order_for_language
from tau2.multilingual.native_script import (
    devanagari_spellout,
    han_spellout,
    is_devanagari,
    is_han,
    is_identity_task_id,
    is_native_identity_task_id,
    native_identity_task_set_name,
    native_name_spellout,
    native_spellout_block_for_task,
    render_native_name_spellout_block,
)

NATIVE_LANGS = ("hi", "zh")
DOMAIN = "retail"


# ---------------------------------------------------------------------------
# Suffix contract
# ---------------------------------------------------------------------------


class TestTaskIdContract:
    def test_native_ids_are_identity_ids_but_not_vice_versa(self):
        assert is_native_identity_task_id("0_zh_identity_native")
        assert is_identity_task_id("0_zh_identity_native")
        assert not is_native_identity_task_id("0_zh_identity")
        assert is_identity_task_id("0_zh_identity")
        assert not is_identity_task_id("0_zh")
        assert not is_native_identity_task_id("0")

    def test_task_set_name(self):
        assert native_identity_task_set_name("retail", "hi") == (
            "retail_hi_identity_native"
        )


# ---------------------------------------------------------------------------
# Devanagari spell-out (the algorithmic payload)
# ---------------------------------------------------------------------------


class TestDevanagariSpellout:
    def test_short_vs_long_i_matra(self):
        # मि carries the SHORT i matra, सी the LONG one — conflating them is
        # exactly the dictation ambiguity the payload exists to remove.
        amit = " ".join(devanagari_spellout("अमित"))
        assert "छोटी इ" in amit and "बड़ी ई" not in amit
        sita = " ".join(devanagari_spellout("सीता"))
        assert "बड़ी ई" in sita and "छोटी इ" not in sita

    def test_short_vs_long_u_matra(self):
        sunita = " ".join(devanagari_spellout("सुनीता"))
        assert "छोटा उ" in sunita
        pooja = " ".join(devanagari_spellout("पूजा"))
        assert "बड़ा ऊ" in pooja and "छोटा उ" not in pooja

    def test_conjuncts_name_the_half_letter(self):
        # क्स in सक्सेना and क्ष in साक्षी both segment as conjunct aksharas
        # whose rendition names the half consonant.
        saxena = devanagari_spellout("सक्सेना")
        assert any("आधा क" in r for r in saxena)
        sakshi = devanagari_spellout("साक्षी")
        assert any("आधा" in r for r in sakshi)

    def test_nukta_is_called_out(self):
        chopra = " ".join(devanagari_spellout("चोपड़ा"))
        assert "नुक़्ता" in chopra

    def test_deterministic_and_total(self):
        assert devanagari_spellout("श्रीवास्तव") == devanagari_spellout("श्रीवास्तव")
        with pytest.raises(ValueError, match="not entirely Devanagari"):
            devanagari_spellout("Amit")


# ---------------------------------------------------------------------------
# Han glosses (the curated payload)
# ---------------------------------------------------------------------------


class TestHanSpellout:
    CATALOG = {"郑": "郑州的郑", "志": "志气的志", "强": "强大的强"}

    def test_family_first_order_and_gloss_shape(self):
        lines = native_name_spellout(
            "志强", "郑", character_clarifications=self.CATALOG
        )
        assert lines == ["郑 — 郑州的郑", "志 — 志气的志", "强 — 强大的强"]

    def test_missing_gloss_raises_instead_of_improvising(self):
        with pytest.raises(ValueError, match="character_clarifications"):
            han_spellout("郑伟", self.CATALOG)

    def test_mixed_script_names_are_refused(self):
        with pytest.raises(ValueError, match="neither all-Han nor all-Devanagari"):
            native_name_spellout("志强", "Zheng", character_clarifications=self.CATALOG)


class TestComposeFullName:
    def test_han_names_join_without_a_space(self):
        assert compose_full_name("志强", "郑", name_order_for_language("zh")) == (
            "郑志强"
        )

    def test_devanagari_names_keep_the_space(self):
        assert compose_full_name("अमित", "सक्सेना", name_order_for_language("hi")) == (
            "अमित सक्सेना"
        )


# ---------------------------------------------------------------------------
# Agent clause flip (the prompt must not lie about the DB)
# ---------------------------------------------------------------------------


class TestAgentClauseFlip:
    # The DB clause's own language, absent from the response-language clause.
    _DB_MARKER = "the customer database stores"

    @pytest.mark.parametrize("lang,example", [("zh", '"志强"'), ("hi", '"अमित"')])
    def test_native_flag_appends_the_pack_clause(self, lang, example):
        from tau2.multilingual.registry import get_agent_language_clause

        plain = get_agent_language_clause(lang)
        flipped = get_agent_language_clause(lang, native_script_db=True)
        assert self._DB_MARKER not in plain
        assert self._DB_MARKER in flipped
        assert example in flipped
        assert flipped.startswith(plain)

    def test_pack_without_clause_raises(self):
        # es ships no native-script identity build; running one against it
        # would tell the agent the DB is something it is not.
        from tau2.multilingual.registry import get_agent_language_clause

        with pytest.raises(ValueError, match="agent_native_script_db_clause"):
            get_agent_language_clause("es", native_script_db=True)


# ---------------------------------------------------------------------------
# Committed artifacts (both native builds)
# ---------------------------------------------------------------------------


def _load_native_tasks(lang: str) -> list[dict]:
    path = native_identity_task_set_path(lang, DOMAIN)
    assert path.is_file(), f"missing committed native task set: {path}"
    return json.loads(path.read_text())


def _load_native_map(lang: str) -> dict[str, dict]:
    path = native_identity_map_path(lang, DOMAIN)
    assert path.is_file(), f"missing committed native identity map: {path}"
    return json.loads(path.read_text())


def _caller_record(task: dict) -> dict:
    users = task["initial_state"]["initialization_data"]["agent_data"]["users"]
    assert len(users) == 1, task["id"]
    return next(iter(users.values()))


_IS_NATIVE = {"zh": is_han, "hi": is_devanagari}


class TestCommittedNativeBuilds:
    @pytest.mark.parametrize("lang", NATIVE_LANGS)
    def test_every_task_id_carries_the_native_suffix(self, lang):
        tasks = _load_native_tasks(lang)
        assert tasks, lang
        for task in tasks:
            assert task["id"].endswith(f"_{lang}_identity_native"), task["id"]

    @pytest.mark.parametrize("lang", NATIVE_LANGS)
    def test_names_are_native_script_and_everything_else_stays_ascii(self, lang):
        # The point of the variant: the DB stores the NAME in native script,
        # unfolded — while user_id and email stay the folded Latin values, so
        # the name-script axis is the only thing this arm moves.
        is_native = _IS_NATIVE[lang]
        for identity in _load_native_map(lang).values():
            assert is_native(identity["first_name"]), identity
            assert is_native(identity["last_name"]), identity
            assert identity["user_id"].isascii(), identity
            assert identity["email"].isascii(), identity
            assert identity["romanized_first_name"].isascii(), identity
            assert identity["romanized_last_name"].isascii(), identity

    @pytest.mark.parametrize("lang", NATIVE_LANGS)
    def test_task_patches_and_golden_lookups_use_the_native_name(self, lang):
        # The DB record the environment loads AND the golden
        # find_user_id_by_name_zip arguments both carry the native name —
        # agent-side lookup must match native script exactly (no fold).
        is_native = _IS_NATIVE[lang]
        for task in _load_native_tasks(lang):
            record = _caller_record(task)
            name = record["name"]
            assert is_native(name["first_name"]), task["id"]
            assert is_native(name["last_name"]), task["id"]
            for action in (task.get("evaluation_criteria") or {}).get("actions") or []:
                if action["name"] != "find_user_id_by_name_zip":
                    continue
                args = action["arguments"]
                assert args["first_name"] == name["first_name"], task["id"]
                assert args["last_name"] == name["last_name"], task["id"]

    def test_zh_gloss_catalog_covers_every_character_of_every_identity(self):
        catalog = load_corpus("zh").character_clarifications
        assert catalog, "zh corpus lost its character_clarifications catalog"
        for identity in _load_native_map("zh").values():
            for ch in identity["last_name"] + identity["first_name"]:
                assert ch in catalog, f"no reviewed gloss for {ch!r}"

    @pytest.mark.parametrize("lang", NATIVE_LANGS)
    def test_committed_payload_equals_the_recomputation(self, lang):
        # The map payload is what the sim dictates; it must never drift from
        # the catalog/algorithm that claims to produce it.
        catalog = load_corpus(lang).character_clarifications
        for identity in _load_native_map(lang).values():
            expected = native_name_spellout(
                identity["first_name"],
                identity["last_name"],
                character_clarifications=catalog,
            )
            assert identity["character_clarifications"] == expected, identity

    @pytest.mark.parametrize("lang", NATIVE_LANGS)
    def test_native_ids_resolve_gender_through_the_identity_sibling(self, lang):
        sidecar = load_gender_sidecar(lang, DOMAIN)
        assert sidecar, lang
        for task in _load_native_tasks(lang)[:5]:
            native_id = task["id"]
            sibling = native_id.removesuffix("_native")
            gender = caller_gender_for_task(native_id, lang, DOMAIN)
            assert gender == sidecar[sibling]
            assert gender in ("male", "female")

    @pytest.mark.parametrize("lang", NATIVE_LANGS)
    def test_spellout_block_renders_from_the_committed_artifacts(self, lang):
        task = _load_native_tasks(lang)[0]
        block = native_spellout_block_for_task(task, lang, DOMAIN)
        assert "## SPELLING YOUR NAME (NATIVE SCRIPT)" in block
        record = _caller_record(task)
        full = compose_full_name(
            record["name"]["first_name"],
            record["name"]["last_name"],
            name_order_for_language(lang),
        )
        assert full in block
        # Every payload line lands verbatim in the directive.
        identity_map = _load_native_map(lang)
        by_user_id = {i["user_id"]: i for i in identity_map.values()}
        for line in by_user_id[record["user_id"]]["character_clarifications"]:
            assert f"- {line}" in block

    def test_render_block_refuses_unsupported_scripts(self):
        with pytest.raises(ValueError, match="unsupported name script"):
            render_native_name_spellout_block("John", "Smith", "John Smith", ["x"])


# ---------------------------------------------------------------------------
# The frozen retail_30 prefix the ablation pools run on
# ---------------------------------------------------------------------------


class TestRetail30Prefix:
    def test_retail_30_is_the_first_30_of_retail_50_in_draw_order(self):
        from tau2.task_subsets.store import load_subset

        parent = load_subset("retail_50")
        prefix = load_subset("retail_30")
        assert prefix.task_ids == parent.task_ids[:30]
        assert prefix.strategy.value == "prefix_of"
        assert prefix.derived_from == "retail_50"
        assert prefix.frame == parent.frame
        assert prefix.seed == parent.seed
        assert prefix.stratifier == parent.stratifier

    def test_retail_30_resolves_against_both_native_sets(self):
        from tau2.registry import registry
        from tau2.task_subsets.store import load_subset

        prefix = load_subset("retail_30")
        for lang in NATIVE_LANGS:
            tasks = registry.get_tasks_loader(f"retail_{lang}_identity_native")()
            selected = prefix.select(tasks)
            assert len(selected) == 30
            assert all(t.id.endswith("_identity_native") for t in selected)
