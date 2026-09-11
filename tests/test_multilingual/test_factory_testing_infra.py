# Copyright Sierra
"""Self-tests for the shared factory test infrastructure.

The ``factory_testing`` package (fake LLM, toy language pack, results
factory, isolated environment fixture) is what every factory test suite
builds on, so its own contracts are pinned here:

- the toy Toylang pack is complete and VALID (and its noise files exist);
- ``results_factory`` run dirs round-trip through ``Results.load``;
- ``isolated_pack_env`` swaps in the toy pack and restores the real packs.
"""

import tau2.data_model.voice_personas as voice_personas
import tau2.multilingual.loader as ml_loader
import tau2.multilingual.registry as ml_registry
from tau2.data_model.simulation import Results, TerminationReason
from tau2.multilingual.invariants import (
    check_task_set_localization,
    extract_values,
    instruction_text,
    script_regex,
)
from tau2.multilingual.schema import LanguagePack
from tau2.multilingual.task_sets import discover_localized_task_files
from tau2.voice_config import BACKGROUND_NOISE_CONTINUOUS_DIR, BURST_NOISE_DIR
from test_multilingual.factory_testing.results_factory import (
    make_run_dir,
    paired_run_dirs,
)
from test_multilingual.factory_testing.toy_language import (
    TOY_GUIDELINES_FILENAME,
    TOY_PACK_DIR,
    load_toy_pack,
    normalize_pack_dict,
    toy_db,
    toy_tasks,
    toy_tasks_localized,
    write_toy_pack_dir,
)

# ---------------------------------------------------------------------------
# Toy language pack
# ---------------------------------------------------------------------------


def test_toy_pack_is_valid():
    pack = load_toy_pack()
    assert pack.language == "tl"
    assert sorted(pack.personas) == ["orin_tl_v1", "tessa_tl_v1"]
    assert all(p.script == "latn" for p in pack.personas.values())
    script_regex("latn")  # the script code is registered


def test_toy_pack_noise_files_exist():
    pack = load_toy_pack()
    assert pack.acoustic_presets, "toy pack must carry an acoustic preset"
    for preset in pack.acoustic_presets.values():
        for filename in preset.background_noise_files:
            assert (BACKGROUND_NOISE_CONTINUOUS_DIR / filename).is_file(), filename
        for filename in preset.burst_noise_files:
            assert (BURST_NOISE_DIR / filename).is_file(), filename


def test_toy_guidelines_have_slot_and_stop_token():
    text = (TOY_PACK_DIR / TOY_GUIDELINES_FILENAME).read_text()
    assert "<PERSONA_GUIDELINES>" in text
    assert "###STOP###" in text


def test_toy_pack_declares_backchannel_level():
    pack = load_toy_pack()
    assert pack.backchannel_level is not None
    assert pack.backchannel_level.value in {"low", "medium", "high"}


def test_toy_tasks_contain_generated_value_kinds():
    texts = [instruction_text(task) for task in toy_tasks()]
    values = extract_values("\n".join(texts))
    # The email pattern keeps trailing sentence punctuation ('.' is a valid
    # address char), so match on the prefix.
    assert any(v.startswith("pat.lee1234@example.com") for v in values)  # email
    assert "pat_lee_1234" in values  # user id
    assert "AB12CD" in values  # reservation code
    assert "250" in values  # dollar amount
    assert any("$250" in text for text in texts)


def test_toy_localized_variant_passes_invariants():
    problems = check_task_set_localization(
        toy_tasks_localized(),
        toy_tasks(),
        suffix="tl",
        script_code="latn",
        domain_db=toy_db(),
    )
    assert not problems, "\n".join(problems)


def test_write_toy_pack_dir_round_trips(tmp_path):
    lang_dir = write_toy_pack_dir(tmp_path)
    assert (lang_dir / "pack.yaml").is_file()
    assert (lang_dir / TOY_GUIDELINES_FILENAME).is_file()
    import yaml

    written = yaml.safe_load((lang_dir / "pack.yaml").read_text())
    pack = LanguagePack.model_validate(normalize_pack_dict(written, lang_dir))
    assert pack.language == "tl"


# ---------------------------------------------------------------------------
# Results factory
# ---------------------------------------------------------------------------


def test_make_run_dir_round_trips_through_results_load(tmp_path):
    rewards = {"1_tl": [1.0, 0.0], "2_tl": [0.5]}
    run_dir = make_run_dir(
        tmp_path,
        "toylang",
        rewards,
        persona_by_task={"2_tl": "orin_tl_v1"},
        termination_by_task={"2_tl": TerminationReason.TOO_MANY_ERRORS},
    )
    assert run_dir == tmp_path / "toylang"
    assert (run_dir / "results.json").is_file()
    assert (run_dir / "simulations").is_dir()

    results = Results.load(run_dir)
    assert {t.id for t in results.tasks} == set(rewards)
    by_task = {}
    for sim in results.simulations:
        by_task.setdefault(sim.task_id, []).append(sim)
    assert sorted(s.reward_info.reward for s in by_task["1_tl"]) == [0.0, 1.0]
    assert [s.trial for s in sorted(by_task["1_tl"], key=lambda s: s.trial)] == [0, 1]
    (sim2,) = by_task["2_tl"]
    assert sim2.reward_info.reward == 0.5
    assert sim2.termination_reason == TerminationReason.TOO_MANY_ERRORS
    assert sim2.speech_environment.persona_id == "orin_tl_v1"
    assert sim2.speech_environment.language == "tl"
    default = by_task["1_tl"][0].speech_environment
    assert default.persona_id == "tessa_tl_v1"


def test_paired_run_dirs_apply_clamped_deltas(tmp_path):
    baseline = {"1_tl": [1.0, 1.0], "2_tl": [0.5], "3_tl": [0.0]}
    main_dir, baseline_dir = paired_run_dirs(
        tmp_path,
        baseline_rewards=baseline,
        deltas={"1_tl": -0.5, "3_tl": -1.0},
    )
    main = Results.load(main_dir)
    base = Results.load(baseline_dir)

    def rewards_of(results, task_id):
        return sorted(
            s.reward_info.reward for s in results.simulations if s.task_id == task_id
        )

    assert rewards_of(main, "1_tl") == [0.5, 0.5]  # delta applied
    assert rewards_of(main, "2_tl") == [0.5]  # parity by default
    assert rewards_of(main, "3_tl") == [0.0]  # clamped at 0
    assert rewards_of(base, "1_tl") == [1.0, 1.0]
    # Baseline arm is stamped English.
    assert all(s.speech_environment.language is None for s in base.simulations)
    assert all(s.speech_environment.language == "tl" for s in main.simulations)


# ---------------------------------------------------------------------------
# isolated_pack_env
# ---------------------------------------------------------------------------


def test_isolated_pack_env_swaps_in_only_the_toy_pack(isolated_pack_env):
    pack = ml_registry.get_language_pack("tl")
    assert pack is not None and pack.display_name == "Toylang"
    # The real packs are invisible inside the isolated environment.
    assert ml_registry.get_language_pack("hi") is None
    assert "tessa_tl_v1" in voice_personas.ALL_PERSONAS

    experiment = ml_loader.get_language_experiment("tl", "toyair")
    assert experiment is not None
    assert experiment.preset_name == "multilingual_v1_toylang"

    # The localized task set is NOT installed until asked for...
    assert "toyair_tl" not in discover_localized_task_files()
    isolated_pack_env.install_localized_tasks()
    discovered = discover_localized_task_files()
    assert "toyair_tl" in discovered
    assert discovered["toyair_tl"].parent == isolated_pack_env.pack_dir


def test_real_packs_restored_after_isolated_env():
    """Runs in file order after the test above: state must be restored."""
    assert ml_registry.get_language_pack("tl") is None
    assert "tessa_tl_v1" not in voice_personas.ALL_PERSONAS
    assert ml_registry.get_language_pack("hi") is not None
