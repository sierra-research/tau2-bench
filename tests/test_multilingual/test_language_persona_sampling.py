# Copyright Sierra
"""Tests for per-task persona assignment via a bare language code.

``--user-persona-id`` accepts either a pack persona id (pins one speaker for
the whole run) or a bare language code like 'hi'. With a task id available
(the live-run path), the persona is assigned by a balanced, seed-rotated
round-robin: the pack's persona ids are shuffled once with a seed-keyed RNG
(string-seeded, so stable across processes), then indexed by the task's
number. For a fixed rotation seed the counts per persona differ by at most 1
across a contiguous task set (exactly 25/25 on the 50 airline_hi tasks), the
assignment is deterministic per (seed, task_id), and different seeds rotate
which persona gets which task. The batch runner passes the run-level seed as
``run_seed`` so the rotation stays constant across a run even though each
task's sampling seed differs.
"""

from collections import Counter

import pytest

from tau2.data_model.voice import SynthesisConfig
from tau2.multilingual.registry import resolve_run_language
from tau2.multilingual.run_presets import get_run_preset
from tau2.multilingual.schema import MultilingualPersonaConfig
from tau2.user_simulation_voice_presets import sample_voice_config

# The hi main arm's full task-id list, straight from the generated preset.
HINDI_AIRLINE_TASK_IDS = (
    get_run_preset("multilingual_v1_hindi").get_arm("hindi").task_ids
)

HINDI_PERSONA_IDS = {"rishika_hindi_v1", "imran_hindi_v1"}

# Seeds chosen (and pinned) to produce different persona rotations for the
# two-persona hi pack.
ROTATING_SEEDS = (0, 7)


def sample(persona_name, seed=7, task_id=None, run_seed=None):
    # Exercises the rotation layer directly (caller_gender=None). The live
    # voice path enters via get_or_load_task_voice_config, which looks up the
    # caller-gender sidecar and pins the pool first — covered separately in
    # TestTextVoiceKeyingParity and TestTextPathCallerGenderPinning.
    return sample_voice_config(
        seed=seed,
        synthesis_config=SynthesisConfig(),
        complexity="regular",
        persona_name=persona_name,
        task_id=task_id,
        run_seed=run_seed,
    )


class TestLanguageCodeAssignment:
    def test_language_code_resolves_to_pack_persona(self):
        cfg = sample("hi", task_id="3_hi")
        assert cfg.persona_name in HINDI_PERSONA_IDS
        assert isinstance(cfg.persona_config, MultilingualPersonaConfig)
        assert cfg.persona_config.persona_id == cfg.persona_name

    def test_assignment_is_deterministic_per_seed_and_task(self):
        """Same (seed, task_id) -> same persona on every call."""
        for task_id in HINDI_AIRLINE_TASK_IDS[:8]:
            for seed in ROTATING_SEEDS:
                names = {
                    sample("hi", seed=seed, task_id=task_id).persona_name
                    for _ in range(3)
                }
                assert len(names) == 1

    def test_assignment_rotates_with_seed(self):
        """Different seeds rotate the (task, persona) pairing (pinned seeds)."""
        per_seed = [
            sample("hi", seed=seed, task_id="3_hi").persona_name
            for seed in ROTATING_SEEDS
        ]
        assert per_seed[0] != per_seed[1]

    def test_run_seed_keys_the_rotation(self):
        """With run_seed given, the per-task sampling seed is irrelevant to
        the assignment — mirroring the live path, where each task's sampling
        seed differs but the run seed is constant."""
        for task_id in HINDI_AIRLINE_TASK_IDS[:8]:
            names = {
                sample("hi", seed=s, task_id=task_id, run_seed=42).persona_name
                for s in (0, 7, 123)
            }
            assert len(names) == 1

    def test_airline_hi_split_is_exactly_even(self):
        """The preset's 50 tasks split exactly 25/25 for any fixed seed."""
        for seed in ROTATING_SEEDS:
            counts = Counter(
                sample("hi", seed=seed, task_id=tid).persona_name
                for tid in HINDI_AIRLINE_TASK_IDS
            )
            assert counts == Counter({"rishika_hindi_v1": 25, "imran_hindi_v1": 25})

    def test_no_task_id_falls_back_to_seeded_rng(self):
        seen = {sample("hi", seed=s).persona_name for s in range(40)}
        assert seen == HINDI_PERSONA_IDS

    def test_explicit_persona_id_still_pins(self):
        for tid in HINDI_AIRLINE_TASK_IDS[:4]:
            cfg = sample("imran_hindi_v1", task_id=tid)
            assert cfg.persona_name == "imran_hindi_v1"

    def test_assigned_persona_uses_stock_environments(self):
        # Packs carry no acoustic presets (shared benchmark environments):
        # language personas flow through the same seed-keyed indoor/outdoor
        # selection as stock English personas, on shared top-level beds.
        for seed in range(4):
            cfg = sample("hi", task_id="3_hi", seed=seed)
            assert cfg.environment in ("indoor", "outdoor")
            assert cfg.background_noise_file is not None
            assert "/" not in cfg.background_noise_file

    def test_english_persona_names_unaffected(self):
        cfg = sample("priya_patil", task_id="3")
        assert cfg.persona_name == "priya_patil"
        assert not isinstance(cfg.persona_config, MultilingualPersonaConfig)


class TestTextVoiceKeyingParity:
    """Text (``resolve_task_persona``) and voice (``sample_voice_config``)
    share ONE rotation-keying function (``persona_rotation_key``) and ONE
    caller-gender sidecar, so the same (task_id, run_seed) resolves to the
    same persona on both sides — including digit-less ids (both fall back to
    the first element of the seeded order, never a salted rng), ids with
    multiple digit runs (keyed on the FIRST run, so a numeric suffix does not
    perturb the rotation), and sidecar-pinned live task ids (both restrict to
    the caller's gender first). The voice side goes through the live entry
    point ``get_or_load_task_voice_config`` so the sidecar lookup it performs
    is part of what's compared."""

    @pytest.mark.parametrize(
        "task_id",
        [
            "no_digits_task",  # digit-less: shared order[0] fallback
            "task_3_hi_v2",  # two digit runs: keys on '3', not '32'
            "3_hi",  # live-path shape; sidecar-pinned to the caller's gender
        ],
    )
    def test_text_and_voice_assign_the_same_persona(self, task_id):
        from tau2.multilingual.registry import resolve_task_persona
        from tau2.user_simulation_voice_presets import get_or_load_task_voice_config

        for run_seed in (0, 7, 42):
            text_choice = resolve_task_persona(
                "hi", task_id=task_id, run_seed=run_seed, domain="airline"
            )
            voice_choice = get_or_load_task_voice_config(
                domain="airline",
                task_id=task_id,
                task_seed=123,
                complexity="regular",
                synthesis_config=SynthesisConfig(),
                persona_name="hi",
                run_seed=run_seed,
            ).persona_name
            assert text_choice == voice_choice

    def test_rotation_key_is_the_first_digit_run(self):
        from tau2.multilingual.registry import persona_rotation_key

        assert persona_rotation_key("3_hi") == 3
        assert persona_rotation_key("task_12_v3") == 12
        assert persona_rotation_key("no_digits") is None
        assert persona_rotation_key(None) is None
        assert persona_rotation_key("") is None


class TestTextPathCallerGenderPinning:
    """`resolve_task_persona` (the text-run path) applies the SAME caller-gender
    pinning as the voice sampler: for a localized task with a caller-gender
    sidecar entry (plain or _identity id), the resolved persona's tags.gender
    matches the task's caller name — so voice and text assign the same persona
    and the prompt's first-person gender agreement matches the spoken name."""

    def test_pins_to_sidecar_gender_for_plain_and_identity_ids(self):
        import pytest

        from tau2.multilingual.factory.entity_localization import (
            caller_gender_for_task,
            gender_sidecar_path,
        )
        from tau2.multilingual.registry import (
            get_multilingual_persona,
            resolve_task_persona,
        )

        if not gender_sidecar_path("es", "airline").exists():
            pytest.skip("es gender sidecar not generated")
        for task_id in ("0_es", "0_es_identity", "7_es", "7_es_identity"):
            want = caller_gender_for_task(task_id, "es", "airline")
            assert want in ("male", "female")
            for run_seed in (0, 7, 42):
                persona_id = resolve_task_persona(
                    "es", task_id=task_id, run_seed=run_seed, domain="airline"
                )
                _pack, persona = get_multilingual_persona(persona_id)
                assert persona.tags.get("gender") == want

    def test_language_without_sidecar_keeps_round_robin(self):
        from collections import Counter

        from tau2.multilingual.registry import resolve_task_persona

        counts = Counter(
            resolve_task_persona("hi", task_id=tid, run_seed=42, domain="airline")
            for tid in HINDI_AIRLINE_TASK_IDS
        )
        assert counts == Counter({"rishika_hindi_v1": 25, "imran_hindi_v1": 25})


class TestResolveRunLanguage:
    def test_persona_id_form(self):
        assert resolve_run_language("rishika_hindi_v1") == "hi"
        assert resolve_run_language("imran_hindi_v1") == "hi"

    def test_language_code_form(self):
        assert resolve_run_language("hi") == "hi"

    def test_english_and_unknown_resolve_to_none(self):
        assert resolve_run_language("priya_patil") is None
        assert resolve_run_language("wei_lin") is None
        assert resolve_run_language("nope_xx") is None


class TestPersonaStableAcrossRetries:
    """Two hallucination retries for the same (task, trial) with bumped seeds
    must yield the same pack persona as the original attempt."""

    def test_same_persona_across_bumped_seeds(self):
        """build_voice_user with persona_seed=constant gives same persona
        regardless of how much the per-retry seed changes."""
        from tau2.user_simulation_voice_presets import (
            derive_task_seed,
            get_or_load_task_voice_config,
        )

        # Use 'hi' (bare language code) with a known Hindi task id.
        TASK_ID = "3_hi"
        ORIGINAL_SEED = 42
        PERSONA_SEED = ORIGINAL_SEED  # invariant across retries

        retry_seeds = [
            ORIGINAL_SEED,
            ORIGINAL_SEED + 1000,  # first hallucination retry
            ORIGINAL_SEED + 2000,  # second retry
        ]

        persona_names = []
        for retry_seed in retry_seeds:
            task_seed = derive_task_seed(retry_seed, TASK_ID)
            cfg = get_or_load_task_voice_config(
                domain="airline",
                task_id=TASK_ID,
                task_seed=task_seed,
                complexity="regular",
                synthesis_config=SynthesisConfig(),
                persona_name="hi",
                run_seed=PERSONA_SEED,  # the fix: constant persona_seed
            )
            persona_names.append(cfg.persona_name)

        # All three (original + 2 retries) must resolve to the same persona.
        assert len(set(persona_names)) == 1, (
            f"Persona changed across retries: {persona_names} — "
            "hallucination retry must not swap the persona"
        )

    def test_different_persona_seed_changes_persona(self):
        """Sanity check: a genuinely different persona_seed CAN produce a
        different persona (for the seeds chosen to rotate differently)."""
        from tau2.user_simulation_voice_presets import get_or_load_task_voice_config

        # Not a real airline_hi task id: real ids are in the caller-gender
        # sidecar, and pinning collapses the two-persona pool to one, so
        # rotation could not vary the persona there.
        TASK_ID = "999_hi"
        cfg_a = get_or_load_task_voice_config(
            domain="airline",
            task_id=TASK_ID,
            task_seed=42,
            complexity="regular",
            synthesis_config=SynthesisConfig(),
            persona_name="hi",
            run_seed=0,
        )
        cfg_b = get_or_load_task_voice_config(
            domain="airline",
            task_id=TASK_ID,
            task_seed=42,
            complexity="regular",
            synthesis_config=SynthesisConfig(),
            persona_name="hi",
            run_seed=7,
        )
        # With seeds 0 and 7 the persona rotates (pinned in the existing
        # test_assignment_rotates_with_seed test).
        assert cfg_a.persona_name != cfg_b.persona_name
