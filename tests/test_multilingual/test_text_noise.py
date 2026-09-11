# Copyright Sierra
"""Noisy-text probes: catalog closure, deterministic plans, injection, bank.

Guards the Item-2 contract of docs/designs/text-mode-multilingual.md:

- the operator catalog and per-language applicability are CLOSED (unknown
  operators/languages fail loudly);
- corruption plans derive only from (catalog version, seed, task id, entity):
  same seed => byte-identical bank (paired design), and the clean ground
  truth is recorded alongside every corruption;
- injection substitutes each entity's FIRST conveyance only, so later
  conveyances stay clean and the agent's re-ask can repair;
- the run wiring arms the user simulator, renders the fixed conveyance
  clause, and records settings + plan in provenance;
- the review bank is built through the exact runtime plan builder.
"""

import argparse

import pytest

from tau2.data_model.simulation import (
    TextNoiseInfo,
    TextNoiseSettings,
    TextRunConfig,
)
from tau2.multilingual.text_noise import (
    NOISE_OPERATORS_BY_LANGUAGE,
    TEXT_NOISE_CATALOG_VERSION,
    TEXT_NOISE_OPERATORS,
    build_text_noise_bank,
    get_noise_operator,
    inject_first_conveyance_noise,
    noise_operators_for,
    pinned_entities,
    plan_task_noise,
)
from tau2.registry import registry
from tau2.runner.build import build_text_orchestrator
from tau2.runner.helpers import get_info


def _task(task_set: str, task_id: str):
    return next(t for t in registry.get_tasks_loader(task_set)() if t.id == task_id)


def _telecom_es_task():
    tasks = registry.get_tasks_loader("telecom_es")()
    return next(
        t
        for t in tasks
        if "You are " in str(t.user_scenario) and "phone number" in str(t.user_scenario)
    )


class TestClosedCatalog:
    def test_unknown_operator_raises(self):
        with pytest.raises(KeyError, match="Unknown text-noise operator"):
            get_noise_operator("emoji_swap")

    def test_language_maps_reference_only_catalog_operators(self):
        for language, operators in NOISE_OPERATORS_BY_LANGUAGE.items():
            for operator_id in operators:
                assert operator_id in TEXT_NOISE_OPERATORS, (language, operator_id)

    def test_unmapped_language_raises(self):
        with pytest.raises(KeyError, match="no text-noise operator mapping"):
            noise_operators_for("ja")

    def test_settings_reject_unknown_operator(self):
        with pytest.raises(ValueError):
            TextNoiseSettings(operators=["emoji_swap"])

    def test_paper_languages_mapped(self):
        for language in ("en", "es", "pt", "hi", "ko", "zh"):
            assert noise_operators_for(language)


class TestEntityPinning:
    def test_airline_pins_user_id(self):
        entities = pinned_entities(_task("airline_hi", "7_hi"), "airline")
        assert ("user_id", "daiki_muller_1116") in [(e.kind, e.value) for e in entities]
        # Airline anchors the caller by handle: no name pinning.
        assert not [e for e in entities if e.kind == "name"]

    def test_telecom_pins_name_and_phone(self):
        task = _telecom_es_task()
        kinds = {e.kind for e in pinned_entities(task, "telecom")}
        assert "name" in kinds
        assert "phone" in kinds


class TestDeterministicPlans:
    def test_same_seed_identical_plan(self):
        task = _task("airline_es", "7_es")
        settings = TextNoiseSettings(seed=42)
        first = plan_task_noise(task, "es", settings, "airline")
        second = plan_task_noise(task, "es", settings, "airline")
        assert first == second
        assert first.info.catalog_version == TEXT_NOISE_CATALOG_VERSION
        assert first.info.seed == 42

    def test_different_seed_changes_corruptions(self):
        task = _task("airline_es", "7_es")
        plans = {
            plan_task_noise(
                task, "es", TextNoiseSettings(seed=seed), "airline"
            ).info.model_dump_json()
            for seed in range(6)
        }
        assert len(plans) > 1

    def test_clean_truth_recorded_and_differs(self):
        task = _task("airline_hi", "7_hi")
        plan = plan_task_noise(task, "hi", TextNoiseSettings(seed=7), "airline")
        assert plan.info.entities, "the airline caller user id must be corrupted"
        for entity in plan.info.entities:
            assert entity.corrupted != entity.clean
            assert entity.operator in noise_operators_for("hi")

    def test_operator_restriction_is_a_subset(self):
        task = _task("airline_es", "7_es")
        plan = plan_task_noise(
            task,
            "es",
            TextNoiseSettings(seed=42, operators=["char_double"]),
            "airline",
        )
        assert {e.operator for e in plan.info.entities} <= {"char_double"}


class TestOperators:
    def test_diacritic_strip_only_fires_with_diacritics(self):
        from tau2.multilingual.text_noise import _can_diacritic_strip

        assert _can_diacritic_strip("José García", "es")
        assert not _can_diacritic_strip("Jose Garcia", "es")

    def test_ko_spacing_needs_hangul(self):
        from random import Random

        from tau2.multilingual.text_noise import _apply_ko_spacing, _can_ko_spacing

        assert not _can_ko_spacing("Jihun Kim", "ko")
        assert _can_ko_spacing("김지훈", "ko")
        spaced = _apply_ko_spacing("김지훈", Random(0), "ko")
        assert spaced != "김지훈" and spaced.replace(" ", "") == "김지훈"
        collapsed = _apply_ko_spacing("김지훈 입니다", Random(0), "ko")
        assert " " not in collapsed or collapsed != "김지훈 입니다"

    def test_ko_spacing_can_apply_agree(self):
        # Regression: two Hangul syllables that are neither adjacent nor
        # separated by a single flanked space give _apply_ko_spacing nothing
        # to drop or split — _can_ko_spacing must not fire on them (it used
        # to count Hangul chars anywhere and _apply crashed on rng.choice([])).
        from random import Random

        from tau2.multilingual.text_noise import _apply_ko_spacing, _can_ko_spacing

        for value in ("김a철", "김-철", "김  철", "가 b 나"):
            assert not _can_ko_spacing(value, "ko"), value
        # Adjacent cases stay eligible, and apply never raises on them.
        for value in ("김철", "김지훈 입니다", "a김철b", "김 이-박"):
            assert _can_ko_spacing(value, "ko"), value
            out = _apply_ko_spacing(value, Random(0), "ko")
            assert out != value
            assert out.replace(" ", "") == value.replace(" ", "")

    def test_autocorrect_uses_language_table(self):
        from random import Random

        from tau2.multilingual.text_noise import _apply_autocorrect, _can_autocorrect

        assert _can_autocorrect("Jose Perez", "es")
        assert _apply_autocorrect("Jose Perez", Random(0), "es") == "José Perez"
        assert not _can_autocorrect("Jose Perez", "en")

    def test_digit_transpose_preserves_length(self):
        task = _telecom_es_task()
        plan = plan_task_noise(
            task,
            "es",
            TextNoiseSettings(seed=1, operators=["digit_transpose"]),
            "telecom",
        )
        for entity in plan.info.entities:
            assert len(entity.corrupted) == len(entity.clean)
            assert sorted(entity.corrupted) == sorted(entity.clean)


class TestInjection:
    INFO = TextNoiseInfo(
        catalog_version=TEXT_NOISE_CATALOG_VERSION,
        seed=42,
        entities=[
            {
                "kind": "user_id",
                "clean": "daiki_muller_1116",
                "corrupted": "daiki_muller_1161",
                "operator": "digit_transpose",
            }
        ],
    )

    def test_first_conveyance_corrupted(self):
        out = inject_first_conveyance_noise(
            "My user id is daiki_muller_1116.", self.INFO, prior_user_texts=[]
        )
        assert out == "My user id is daiki_muller_1161."

    def test_later_conveyances_stay_clean(self):
        out = inject_first_conveyance_noise(
            "Sorry, it is daiki_muller_1116.",
            self.INFO,
            prior_user_texts=["My user id is daiki_muller_1161."],
        )
        assert out == "Sorry, it is daiki_muller_1116."

    def test_message_without_entity_untouched(self):
        out = inject_first_conveyance_noise(
            "Hello, I need help.", self.INFO, prior_user_texts=[]
        )
        assert out == "Hello, I need help."

    SHORT_INFO = TextNoiseInfo(
        catalog_version=TEXT_NOISE_CATALOG_VERSION,
        seed=42,
        entities=[
            {
                "kind": "name",
                "clean": "mei",
                "corrupted": "me",
                "operator": "autocorrect_substitution",
            }
        ],
    )

    def test_prior_check_ignores_incidental_substrings(self):
        # Regression: the corrupted form "me" appearing INSIDE an unrelated
        # word ("some") must not count as a prior conveyance — the first real
        # conveyance still gets corrupted.
        out = inject_first_conveyance_noise(
            "My name is mei.",
            self.SHORT_INFO,
            prior_user_texts=["I have some questions."],
        )
        assert out == "My name is me."

    def test_prior_whole_token_conveyance_still_counts(self):
        out = inject_first_conveyance_noise(
            "It is mei.",
            self.SHORT_INFO,
            prior_user_texts=["My name is me."],
        )
        assert out == "It is mei."

    def test_clean_value_inside_larger_token_not_rewritten(self):
        out = inject_first_conveyance_noise(
            "Ask for meister mei.", self.SHORT_INFO, prior_user_texts=[]
        )
        assert out == "Ask for meister me."

    def test_hangul_neighbor_does_not_block_conveyance(self):
        info = TextNoiseInfo(
            catalog_version=TEXT_NOISE_CATALOG_VERSION,
            seed=42,
            entities=[
                {
                    "kind": "name",
                    "clean": "김지훈",
                    "corrupted": "김지 훈",
                    "operator": "ko_spacing",
                }
            ],
        )
        out = inject_first_conveyance_noise(
            "저는 김지훈입니다.", info, prior_user_texts=[]
        )
        assert out == "저는 김지 훈입니다."
        repaired = inject_first_conveyance_noise(
            "다시요, 김지훈입니다.", info, prior_user_texts=["저는 김지 훈입니다."]
        )
        assert repaired == "다시요, 김지훈입니다."


class TestRunWiring:
    def test_orchestrator_arms_user_and_prompt(self):
        config = TextRunConfig(
            domain="airline",
            agent="llm_agent",
            user_persona_id="hi",
            text_noise=TextNoiseSettings(seed=42),
        )
        orch = build_text_orchestrator(config, _task("airline_hi", "7_hi"), seed=42)
        assert orch.user.entity_noise is not None
        assert orch.user.entity_noise.catalog_version == TEXT_NOISE_CATALOG_VERSION
        assert any(
            e.clean == "daiki_muller_1116" for e in orch.user.entity_noise.entities
        )
        assert "## TYPED VALUES" in orch.user.system_prompt

    def test_unarmed_run_untouched(self):
        config = TextRunConfig(
            domain="airline", agent="llm_agent", user_persona_id="hi"
        )
        orch = build_text_orchestrator(config, _task("airline_hi", "7_hi"), seed=42)
        assert orch.user.entity_noise is None
        assert "## TYPED VALUES" not in orch.user.system_prompt

    def test_info_records_settings(self):
        config = TextRunConfig(
            domain="airline",
            agent="llm_agent",
            text_noise=TextNoiseSettings(seed=9),
        )
        info = get_info(config)
        assert info.text_noise is not None and info.text_noise.seed == 9

    def test_plain_english_run_supported(self):
        config = TextRunConfig(
            domain="airline",
            agent="llm_agent",
            text_noise=TextNoiseSettings(seed=42),
        )
        orch = build_text_orchestrator(config, _task("airline", "7"), seed=42)
        assert orch.user.entity_noise is not None


class TestBank:
    def test_bank_matches_runtime_plans(self):
        settings = TextNoiseSettings(seed=42)
        bank = build_text_noise_bank("es", "airline", "airline_es", settings)
        assert bank.manifest.catalog_version == TEXT_NOISE_CATALOG_VERSION
        assert bank.manifest.num_tasks == len(bank.plans) > 0
        assert bank.manifest.num_corrupted == sum(
            len(p.info.entities) for p in bank.plans
        )
        by_id = {p.task_id: p for p in bank.plans}
        task = _task("airline_es", "7_es")
        assert (
            by_id["7_es"].info == plan_task_noise(task, "es", settings, "airline").info
        )

    def test_bank_verb_writes_artifact(self, tmp_path):
        from tau2.multilingual.factory.cli import run_factory_text_noise_bank

        out = tmp_path / "bank.json"
        run_factory_text_noise_bank(
            argparse.Namespace(
                lang="es",
                domain="airline",
                task_set=None,
                seed=42,
                operators=None,
                out=str(out),
            )
        )
        from tau2.multilingual.text_noise import TextNoiseBank

        bank = TextNoiseBank.model_validate_json(out.read_text())
        assert bank.manifest.task_set_name == "airline_es"
        assert bank.manifest.seed == 42
