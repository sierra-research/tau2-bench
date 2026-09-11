# Copyright Sierra
"""Acceptance test for the no-code Language Factory.

Proves the no-code contract end to end: **a new language is one folder under
``data/tau2/multilingual/<lang>/`` — pack.yaml, a guidelines markdown, and
task JSON — with zero edits to any Python file.** The test materializes a
synthetic Cyrillic-script language ('xq') as pure data and drives every
Factory layer over it:

1. pack.yaml discovery -> validated, registered LanguagePack (personas
   become VoicePersonas; the guidelines file resolves off the yaml);
2. task-file discovery by naming convention -> registered task set;
3. preset generation from the experiment block -> a runnable RunPreset with
   the matched English-baseline arm and judge env;
4. the shared localization invariants pass over its tasks.

Everything is created in a temp folder inside the real data dir and cleaned
up; registries are snapshot/restored.
"""

import json
import shutil

import pytest

import tau2.data_model.voice_personas as voice_personas
import tau2.multilingual.loader as ml_loader
import tau2.multilingual.registry as ml_registry
from tau2.config import FORCE_LLM_COMMUNICATE_JUDGE_ENV
from tau2.multilingual.invariants import check_task_set_localization
from tau2.multilingual.localize_lib import multilingual_data_dir
from tau2.multilingual.run_presets import _build_preset
from tau2.multilingual.task_sets import (
    discover_localized_task_files,
    register_localized_task_sets,
)
from tau2.utils import DATA_DIR

LANG = "xq"  # synthetic test language, Cyrillic script
FOLDER = multilingual_data_dir() / LANG

PACK_YAML = """\
language: xq
display_name: Testlang
guidelines_voice_path: simulation_guidelines_voice_xq.md
personas:
  xara_xq_v1:
    persona_id: xara_xq_v1
    display_name: Xara
    short_description: Synthetic Cyrillic-script test persona
    language: xq
    locale: XQ-TS
    script: cyrl
    verbosity: standard
    interrupt_tendency: waits
    tts_voice_prompt: A synthetic test voice, calm and clear.
    pragmatics_clauses:
      - "You greet with 'Привет' and stay polite."
    backchannel_phrases: ["да", "ага"]
    non_directed_phrases: ["минутку, я говорю по телефону"]
backchannel_level: medium
agent_language_clause: >-
  Respond in Testlang using Cyrillic script; keep all tool calls, arguments,
  and structured outputs in English exactly as specified.
default_out_of_turn_events_per_minute: 0.8
experiments:
- preset_name: multilingual_v1_testlang
  description: Testlang synthetic experiment.
  domain: airline
  main_arm_name: testlang
  task_set_suffix: xq
  smoke_task_stem: "0"
"""

GUIDELINES_MD = """\
# Voice Call Simulation Guidelines (Testlang)

Говорите естественно, как настоящий звонящий.

Use '###STOP###', '###TRANSFER###', '###OUT-OF-SCOPE###' in ASCII.
<PERSONA_GUIDELINES>
"""


@pytest.fixture
def xq_language_folder():
    """Materialize the xq folder as pure data; restore all global state."""
    saved_packs = dict(ml_registry._LANGUAGE_PACKS)
    saved_personas = dict(voice_personas.ALL_PERSONAS)
    saved_names = list(voice_personas.ALL_PERSONA_NAMES)
    saved_experiments = dict(ml_loader._EXPERIMENTS)
    assert not FOLDER.exists(), f"stale test folder {FOLDER}"

    FOLDER.mkdir(parents=True)
    (FOLDER / "pack.yaml").write_text(PACK_YAML)
    (FOLDER / "simulation_guidelines_voice_xq.md").write_text(GUIDELINES_MD)

    # Two "translated" airline tasks: Cyrillic-tagged prose with every
    # concrete value intact; evaluation_criteria untouched.
    with open(DATA_DIR / "tau2" / "domains" / "airline" / "tasks.json") as fp:
        source_tasks = json.load(fp)
    localized = []
    for task in source_tasks[:2]:
        out = json.loads(json.dumps(task))
        out["id"] = f"{task['id']}_{LANG}"
        out["description"]["purpose"] = f"Тест: {task['description']['purpose']}"
        instructions = out["user_scenario"]["instructions"]
        for field in ("task_instructions", "reason_for_call", "known_info"):
            if instructions.get(field):
                instructions[field] = f"Тест: {instructions[field]}"
        localized.append(out)
    with open(FOLDER / f"airline_tasks_{LANG}.json", "w") as fp:
        json.dump(localized, fp, ensure_ascii=False, indent=2)

    try:
        yield {"source_tasks": source_tasks, "localized": localized}
    finally:
        shutil.rmtree(FOLDER)
        ml_registry._LANGUAGE_PACKS.clear()
        ml_registry._LANGUAGE_PACKS.update(saved_packs)
        voice_personas.ALL_PERSONAS.clear()
        voice_personas.ALL_PERSONAS.update(saved_personas)
        voice_personas.ALL_PERSONA_NAMES[:] = saved_names
        ml_loader._EXPERIMENTS.clear()
        ml_loader._EXPERIMENTS.update(saved_experiments)


def test_no_touch_language_addition(xq_language_folder):
    # 1. Pack discovery: the yaml alone yields a validated, registered pack.
    ml_loader._load_yaml_pack(FOLDER / "pack.yaml")
    pack = ml_registry.get_language_pack(LANG)
    assert pack is not None
    assert pack.display_name == "Testlang"
    assert set(pack.personas) == {"xara_xq_v1"}
    # Persona became a VoicePersona; the guidelines file resolved off the yaml.
    assert "xara_xq_v1" in voice_personas.ALL_PERSONAS
    guidelines = pack.guidelines_voice_path.read_text()
    assert "<PERSONA_GUIDELINES>" in guidelines
    assert "Привет" not in guidelines  # persona content stays in the slot
    assert "###STOP###" in guidelines
    assert "говорите" in guidelines.lower()

    # 2. Task-set discovery by filename convention.
    discovered = discover_localized_task_files()
    assert f"airline_{LANG}" in discovered

    class FakeRegistry:
        def __init__(self):
            self.registered = {}

        def register_tasks(self, loader, name):
            self.registered[name] = loader

    fake = FakeRegistry()
    register_localized_task_sets(fake)
    tasks = fake.registered[f"airline_{LANG}"](task_split_name="base")
    assert [t.id for t in tasks] == [f"0_{LANG}", f"1_{LANG}"]

    # 3. Preset generation from the experiment block.
    experiment = ml_loader.get_language_experiment(LANG, "airline")
    assert experiment is not None
    preset = _build_preset(LANG, experiment)
    assert preset.name == "multilingual_v1_testlang"
    # Single localized arm — the shared English baseline is its own preset.
    assert [arm.name for arm in preset.arms] == ["testlang"]
    (main,) = preset.arms
    assert main.task_ids == [f"0_{LANG}", f"1_{LANG}"]
    assert main.smoke_task_ids == [f"0_{LANG}"]
    assert main.user_persona_id == LANG
    # Non-English localized arm: the judge auto-activates from the run
    # language, so the force env is NOT set here.
    assert FORCE_LLM_COMMUNICATE_JUDGE_ENV not in main.env
    command = preset.cli_command(main, stage="smoke")
    assert command[:2] == ["tau2", "run"]
    assert "--user-persona-id" in command

    # 4. The shared localization invariants hold for the fixture tasks.
    problems = check_task_set_localization(
        xq_language_folder["localized"],
        xq_language_folder["source_tasks"],
        suffix=LANG,
        script_code="cyrl",
    )
    assert not problems, "\n".join(problems)


def test_broken_pack_yaml_fails_loudly(xq_language_folder):
    """A pack.yaml referencing a missing guidelines file is rejected at
    discovery with the schema's validation error, not a silent skip."""
    bad = (
        (FOLDER / "pack.yaml")
        .read_text()
        .replace("simulation_guidelines_voice_xq.md", "missing.md")
    )
    (FOLDER / "pack.yaml").write_text(bad)
    with pytest.raises(Exception, match="missing.md"):
        ml_loader._load_yaml_pack(FOLDER / "pack.yaml")
    assert ml_registry.get_language_pack(LANG) is None
