# Copyright Sierra
"""Offline end-to-end localization pipeline over the toy language.

Drives the library localization path with zero LLM calls, inside an isolated
DATA_DIR (the ``isolated_pack_env`` fixture):

    toy 'toyair' tasks
      -> ``localize_lib.inject_translations`` (the gate every writer shares)
      -> ``localize_lib.write_task_set`` (<multilingual>/tl/toyair_tasks_tl.json)
      -> shared invariants pass over the written file
      -> filename-convention registration (``tau2.multilingual.task_sets``).

Plus the CHAINED factory pipeline (draft -> finalize -> localize-entities) over
the same toy language, every LLM call scripted through ``FakeFactoryLLM``.

The translator CSV round-trip, the translate/verify/fix loop and the parity
probe used to sit in the middle of both flows; all three are retired and
deleted, so the localized task set is installed from the toy fixture instead
of being produced by them.
"""

import json

import pytest

from tau2.data_model.tasks import Task
from tau2.multilingual.invariants import check_task_set_localization
from tau2.multilingual.localize_lib import (
    inject_translations,
    iter_rows,
    write_task_set,
)
from tau2.multilingual.task_sets import (
    discover_localized_task_files,
    register_localized_task_sets,
)
from test_multilingual.factory_testing.toy_language import (
    TOY_DOMAIN,
    TOY_LANGUAGE,
    TOY_LOCALIZED_TASKS_FILENAME,
    toy_db,
    toy_tasks,
    toy_tasks_localized,
)


@pytest.fixture
def tasks_csv_env(isolated_pack_env):
    """The isolated environment, under its e2e-suite name.

    ``isolated_pack_env`` patches ``tau2.utils.DATA_DIR``, and every
    ``localize_lib`` path helper resolves through it at call time — no
    per-module path globals remain to patch.
    """
    return isolated_pack_env


class FakeTaskRegistry:
    def __init__(self):
        self.registered = {}

    def register_tasks(self, loader, name):
        self.registered[name] = loader


def _translations_from_fixture() -> dict[tuple[str, str], str]:
    """The (task_id, field) -> translation map, from the pre-localized set.

    Stands in for a filled translator sheet: the same mapping the retired CSV
    round-trip used to build, taken straight off the valid Toylang fixture.
    """
    localized_by_source_id = {
        task["id"].removesuffix(f"_{TOY_LANGUAGE}"): task
        for task in toy_tasks_localized()
    }
    translations: dict[tuple[str, str], str] = {}
    for task in toy_tasks():
        localized = localized_by_source_id[task["id"]]
        for field, _english in iter_rows(task):
            if field.endswith("(metadata)"):
                value = localized["description"][field.removesuffix(" (metadata)")]
            else:
                value = localized["user_scenario"]["instructions"][field]
            assert value, f"fixture is missing a translation for {task['id']}/{field}"
            translations[(task["id"], field)] = value
    assert translations, "fixture produced no translation rows"
    return translations


def test_inject_write_register_round_trip(tasks_csv_env):
    isolated_pack_env = tasks_csv_env
    # 1. Inject: apply the fixture translations through the shared gate.
    localized, problems = inject_translations(
        toy_tasks(),
        _translations_from_fixture(),
        TOY_LANGUAGE,
        script_code="latn",
    )
    assert problems == [], "\n".join(problems)
    assert localized == toy_tasks_localized()

    # 2. Write the conventional file into the (isolated) multilingual dir.
    out_path = isolated_pack_env.pack_dir / TOY_LOCALIZED_TASKS_FILENAME
    write_task_set(localized, out_path)
    assert out_path.is_file()
    with open(out_path) as fp:
        written = json.load(fp)
    assert written == toy_tasks_localized()

    # 3. The shared invariants pass over the written task set.
    invariant_problems = check_task_set_localization(
        written,
        toy_tasks(),
        suffix=TOY_LANGUAGE,
        script_code="latn",
        domain_db=toy_db(),
    )
    assert not invariant_problems, "\n".join(invariant_problems)

    # 4. Filename-convention registration picks the file up with no code.
    discovered = discover_localized_task_files()
    assert discovered.get(f"{TOY_DOMAIN}_{TOY_LANGUAGE}") == out_path
    registry = FakeTaskRegistry()
    register_localized_task_sets(registry)
    tasks = registry.registered[f"{TOY_DOMAIN}_{TOY_LANGUAGE}"]()
    assert [task.id for task in tasks] == ["1_tl", "2_tl", "3_tl", "4_tl"]
    assert all(isinstance(task, Task) for task in tasks)


def test_inject_refuses_a_value_dropping_translation(tasks_csv_env):
    """The inject gate is real: dropping one concrete value blocks the write."""
    translations = _translations_from_fixture()
    # Strip the reservation code out of one field.
    victim = next(key for key, text in translations.items() if "AB12CD" in text)
    translations[victim] = translations[victim].replace("AB12CD", "")

    localized, problems = inject_translations(
        toy_tasks(),
        translations,
        TOY_LANGUAGE,
        script_code="latn",
    )
    assert problems, "a dropped concrete value must be refused"
    assert any("AB12CD" in problem for problem in problems)
    assert localized is not None  # the gate reports; it does not write


TL_CORPUS = """\
female_first_names: [Mira, Sanne]
male_first_names: [Joren, Piet]
last_names: [Vandermolen, Dekker]
email_domain: toymail.tl
phone_number_format: "06## ### ###"
cities:
  - {city: Noordstad, region: NRD}
  - {city: Zuidhaven, region: ZUD}
street_patterns: ["{n} Hoofdstraat"]
country: TLD
zip_format: "####"
"""


def test_full_factory_pipeline_offline(isolated_pack_env, monkeypatch):
    """draft -> finalize -> localize-entities, every LLM call scripted."""
    import tau2.multilingual.factory.entity_localization as ent
    from tau2.multilingual.factory.drafting import run_draft
    from tau2.multilingual.factory.finalize import finalize
    from test_multilingual.factory_testing.draft_scripts import (
        TOY_FORM,
        draft_scripts,
    )
    from test_multilingual.factory_testing.fake_llm import FakeFactoryLLM

    env = isolated_pack_env

    # --- 1. DRAFT: three scripted creative calls + deterministic assembly.
    form_path = env.data_dir / "toylang_form.md"
    form_path.write_text(TOY_FORM)
    draft_outcome = run_draft(TOY_LANGUAGE, form_path, FakeFactoryLLM(draft_scripts()))
    assert draft_outcome.report.ok, "\n".join(draft_outcome.report.problems)

    # --- 2. FINALIZE: promote the draft over the pre-installed toy pack.
    finalize_outcome = finalize(TOY_LANGUAGE, force=True)
    assert finalize_outcome.report.ok, "\n".join(finalize_outcome.report.problems)

    # --- 3. Install the localized arm set. The retired translate loop used to
    # produce this; today a curated domain's arm files come from `arm-tasks`
    # and a generated-pool domain's from `seed-tasks`. Either way
    # localize-entities reads the file off disk, so the fixture stands in.
    localized, problems = inject_translations(
        toy_tasks(),
        _translations_from_fixture(),
        TOY_LANGUAGE,
        script_code="latn",
    )
    assert problems == [], "\n".join(problems)
    write_task_set(localized, env.pack_dir / TOY_LOCALIZED_TASKS_FILENAME)

    # --- 4. LOCALIZE-ENTITIES: corpus-driven identity swap over the localized
    # set; the _identity variant registers by filename convention.
    (env.pack_dir / ent.CORPUS_FILENAME).write_text(TL_CORPUS)
    identity_tasks, ent_problems, identity_map = ent.localize_entities(
        TOY_LANGUAGE, TOY_DOMAIN
    )
    assert ent_problems == [], "\n".join(ent_problems)
    assert identity_map and identity_tasks
    identity_path = env.pack_dir / f"{TOY_DOMAIN}_tasks_{TOY_LANGUAGE}_identity.json"
    assert identity_path.is_file()
    discovered = discover_localized_task_files()
    assert f"{TOY_DOMAIN}_{TOY_LANGUAGE}_identity" in discovered
