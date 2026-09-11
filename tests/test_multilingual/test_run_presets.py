# Copyright Sierra
"""Tests for the multilingual run presets.

The preset is the experiment's reproducibility contract, so these tests pin
the matrix down:
- Each language preset has a SINGLE localized arm; the shared English baseline
  is its own ``multilingual_v1_english`` preset, compared against post-hoc.
- Hindi runs the full ``airline_hi_identity`` task set (identity-variant sets
  replace the plain localized set as the run arm once the language's
  ``localize-entities`` artifacts ship).
- Provider/complexity/seed are the matched conditions the plan locked.
- The English baseline preset — and only it — forces the LLM communicate judge
  (English would otherwise fall back to substring matching); localized arms
  activate the judge automatically from their non-English run language.
- The smoke stage is 1 task per arm, and every smoke task is in the full list.
- CLI command construction produces a valid ``tau2 run`` invocation.
"""

import pytest

from tau2.config import (
    DEFAULT_MATRIX_MAX_STEPS_SECONDS,
    DEFAULT_MULTILINGUAL_VOICE_TIMEOUT_SECONDS,
    FORCE_LLM_COMMUNICATE_JUDGE_ENV,
)
from tau2.multilingual.registry import get_multilingual_persona
from tau2.multilingual.run_presets import (
    PresetArm,
    get_run_preset,
)
from tau2.registry import registry

# The hi main arm's full task-id list, straight from the generated preset.
HINDI_AIRLINE_TASK_IDS = (
    get_run_preset("multilingual_v1_hindi").get_arm("hindi").task_ids
)


@pytest.fixture
def preset():
    return get_run_preset("multilingual_v1_hindi")


def test_preset_lookup(preset):
    # Equality, not identity: the preset table is lazily built and cache-reset
    # by isolated-env tests, so lookups may rebuild an equal object.
    assert get_run_preset("multilingual_v1_hindi") == preset
    with pytest.raises(ValueError, match="Unknown run preset"):
        get_run_preset("nope")


def test_language_preset_is_single_arm(preset):
    """A language preset has exactly its localized arm — no baseline arm."""
    assert [arm.name for arm in preset.arms] == ["hindi"]


def test_hindi_arm_covers_the_airline_hi_identity_task_set(preset):
    """The preset's Hindi task list is exactly the registered
    airline_hi_identity set — the identity arm replaces the plain set."""
    assert len(HINDI_AIRLINE_TASK_IDS) == 50
    tasks = registry.get_tasks_loader("airline_hi_identity")()
    assert sorted(t.id for t in tasks) == sorted(HINDI_AIRLINE_TASK_IDS)
    assert preset.get_arm("hindi").task_ids == HINDI_AIRLINE_TASK_IDS


def test_personas_resolve(preset):
    """The Hindi arm samples the pack per task via the language-code override."""
    from tau2.multilingual.registry import resolve_run_language

    assert preset.get_arm("hindi").user_persona_id == "hi"
    assert resolve_run_language("hi") == "hi"
    # The language-code form must not be a registered persona id itself.
    assert get_multilingual_persona("hi") is None


def test_judge_forced_on_english_preset_only(preset):
    """Identical-metric invariant. The English baseline preset forces the LLM
    communicate judge (English would otherwise default to substring matching);
    localized arms keep env empty and activate the judge from their non-English
    run language, so forcing it there would mask a regression in that path."""
    for arm in preset.arms:
        assert FORCE_LLM_COMMUNICATE_JUDGE_ENV not in arm.env
    english = get_run_preset("multilingual_v1_english")
    assert [arm.name for arm in english.arms] == ["english"]
    assert english.get_arm("english").env == {FORCE_LLM_COMMUNICATE_JUDGE_ENV: "1"}


def test_english_preset_runs_localized_en_task_set():
    """The English baseline preset runs the airline_en task set (suffixed ids)."""
    english = get_run_preset("multilingual_v1_english")
    arm = english.get_arm("english")
    assert arm.task_set_name == "airline_en"
    assert arm.user_persona_id == "en"
    assert arm.task_ids and all(tid.endswith("_en") for tid in arm.task_ids)


def test_matched_conditions(preset):
    """Same provider, complexity, and seed (single values on the preset, by
    construction) and the values the plan locked."""
    assert preset.audio_native_provider == "openai"
    assert preset.speech_complexity == "regular"
    assert preset.max_steps_seconds == DEFAULT_MATRIX_MAX_STEPS_SECONDS
    assert len(preset.arms) == 1


def test_smoke_stage_is_one_task_per_arm(preset):
    for arm in preset.arms:
        assert len(arm.smoke_task_ids) == 1
        assert set(arm.smoke_task_ids) <= set(arm.task_ids)


def test_smoke_tasks_must_be_subset_of_full():
    with pytest.raises(ValueError, match="smoke tasks"):
        PresetArm(
            name="bad",
            domain="airline",
            task_set_name="airline",
            task_ids=["3"],
            smoke_task_ids=["4"],
            user_persona_id="priya_patil",
        )


def test_cli_command_construction(preset):
    arm = preset.get_arm("hindi")
    command = preset.cli_command(arm, stage="smoke")
    assert command[:2] == ["tau2", "run"]
    assert "--audio-native" in command
    text = " ".join(command)
    assert "--domain airline" in text
    assert "--task-set-name airline_hi" in text
    assert "--task-ids 3_hi" in text
    assert "--audio-native-provider openai" in text
    assert "--speech-complexity regular" in text
    assert "--user-persona-id hi" in text
    assert "--save-to multilingual/multilingual_v1_hindi/smoke_hindi" in text
    # 20-minute conversation cap (doubled 2026-07-22 so the hard end of the
    # telecom difficulty dial finishes instead of truncating).
    assert f"--max-steps-seconds {DEFAULT_MATRIX_MAX_STEPS_SECONDS}" in text
    # Per-simulation wall-clock timeout, shared with matrix mode: a hung
    # provider session must not block the arm forever.
    assert f"--timeout {DEFAULT_MULTILINGUAL_VOICE_TIMEOUT_SECONDS}" in text

    full = " ".join(preset.cli_command(arm, stage="full"))
    assert f"--task-ids {' '.join(HINDI_AIRLINE_TASK_IDS)}" in full
    assert "--save-to multilingual/multilingual_v1_hindi/full_hindi" in full


def test_cli_command_full_matrix_is_unique(preset):
    """No two (arm, stage) runs collide on a save path."""
    save_paths = [
        preset.save_to(arm, stage) for arm in preset.arms for stage in ("smoke", "full")
    ]
    assert len(save_paths) == len(set(save_paths))


def test_every_preset_run_lands_under_the_multilingual_subtree():
    """The experiment tree is ONE subtree of data/simulations/: a preset run
    must never write a multilingual_v1_* dir at the top level again."""
    from tau2.config import MULTILINGUAL_SIMULATIONS_SUBDIR
    from tau2.multilingual.run_presets import get_run_presets

    for preset in get_run_presets().values():
        for arm in preset.arms:
            for stage in ("smoke", "full"):
                save_to = preset.save_to(arm, stage)
                assert save_to.split("/")[0] == MULTILINGUAL_SIMULATIONS_SUBDIR
                assert save_to == (
                    f"{MULTILINGUAL_SIMULATIONS_SUBDIR}/{preset.name}/"
                    f"{stage}_{arm.name}"
                )


# ---------------------------------------------------------------------------
# Shared task-set selection (preset generation + matrix mode)
# ---------------------------------------------------------------------------


def test_resolve_main_task_suffix_identity_selection():
    """The identity variant replaces the plain set when present, honoring the
    include_identity_arm opt-out — the ONE selection rule presets and matrix
    mode share."""
    from tau2.multilingual.run_presets import resolve_main_task_suffix

    files = {"airline_es": object(), "airline_es_identity": object()}
    assert resolve_main_task_suffix("airline", "es", task_files=files) == "es_identity"
    assert (
        resolve_main_task_suffix(
            "airline", "es", include_identity_arm=False, task_files=files
        )
        == "es"
    )
    assert (
        resolve_main_task_suffix("airline", "hi", task_files={"airline_hi": object()})
        == "hi"
    )


# ---------------------------------------------------------------------------
# Lazy preset table: per-pack failure isolation
# ---------------------------------------------------------------------------


def _write_broken_sibling_pack(env) -> None:
    """A second, loadable pack ('xx') whose experiment block references a task
    set that does not exist — preset generation for it must fail."""
    import shutil

    import yaml

    from test_multilingual.factory_testing.toy_language import (
        TOY_GUIDELINES_FILENAME,
        TOY_PACK_DIR,
        toy_pack_yaml_dict,
    )

    data = toy_pack_yaml_dict()
    data["language"] = "xx"
    data["display_name"] = "Xxlang"
    personas = {}
    for pid, persona in data["personas"].items():
        new_pid = pid.replace("_tl_", "_xx_")
        persona["persona_id"] = new_pid
        persona["language"] = "xx"
        personas[new_pid] = persona
    data["personas"] = personas
    data["experiments"] = [
        {
            **data["experiments"][0],
            "preset_name": "multilingual_v1_xxlang",
            "main_arm_name": "xxlang",
            "task_set_suffix": "xx",  # no toyair_tasks_xx.json exists -> broken
        }
    ]
    pack_dir = env.multilingual_dir / "xx"
    pack_dir.mkdir(parents=True)
    (pack_dir / "pack.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    shutil.copy(
        TOY_PACK_DIR / TOY_GUIDELINES_FILENAME, pack_dir / TOY_GUIDELINES_FILENAME
    )


def test_one_broken_pack_does_not_take_down_the_preset_table(isolated_pack_env):
    """H3 isolation: a pack whose experiment block cannot generate a preset is
    skipped (with a warning); every other language's preset still generates."""
    from tau2.multilingual.run_presets import get_run_presets

    isolated_pack_env.install_localized_tasks()
    _write_broken_sibling_pack(isolated_pack_env)

    presets = get_run_presets()
    assert "multilingual_v1_toylang" in presets  # the healthy pack survived
    assert "multilingual_v1_xxlang" not in presets  # the broken one was skipped

    with pytest.raises(ValueError, match="Unknown run preset"):
        get_run_preset("multilingual_v1_xxlang")
