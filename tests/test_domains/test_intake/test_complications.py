"""Tests for scripted caller complications: catalog gates, the closed
profile catalog and its literature-anchored rates, sampler determinism and
per-kind invariants, the mass-to-none feasibility rule, the channel gate
(voice-only ``mispronounced_term`` and ``spelling_style``), profile-imposed
task selection, prompt
injection (and NON-injection) through the ``user_prompt_task`` seam,
provenance round-trips (including refill), and the review packet."""

import json

import pytest

from tau2.data_model.simulation import (
    AudioNativeConfig,
    ComplicationProfile,
    SampledComplication,
    SimulationRun,
    TextRunConfig,
    VoiceRunConfig,
)
from tau2.data_model.tasks import Task
from tau2.domains.intake.complications import (
    BANK_COMPLICATIONS,
    BANK_SPELLING_STYLES,
    COMPLICATION_CATALOG_VERSION,
    COMPLICATION_LINES,
    LAZY_OMISSION_COMPONENTS,
    LAZY_OMISSION_RATE,
    MISPRONOUNCED_TERM_BANK_RATES,
    MISPRONOUNCED_TERM_BANKS,
    MISPRONOUNCED_TERM_MEDICATIONS_RATE,
    MISPRONOUNCED_TERM_PERSON_NAMES_RATE,
    PROFILE_KIND_RATES,
    SELF_CORRECTION_RATE,
    SPELL_CORRECTION_MARKERS,
    SPELL_CORRECTION_RATE,
    SPELLING_STYLE_RATE,
    WRONG_SLOT_RATE,
    ComplicationKind,
    SpellingStyle,
    _kind_feasible,
    _parse_task_id,
    _pool_values,
    build_complications_packet,
    effective_kind_rates,
    is_hard_tier,
    profile_kind_rates,
    sample_complication,
    validate_complication_catalog,
    validate_profile_rates,
)
from tau2.domains.intake.folds import FoldKind, fold_value
from tau2.domains.intake.tasks.banks import BANK_NAMES, Difficulty, load_banks
from tau2.domains.intake.tasks.generator import CANONICAL_BANKS
from tau2.domains.intake.utils import INTAKE_DATA_DIR

DEFAULT = ComplicationProfile.DEFAULT
HARD = ComplicationProfile.HARD


@pytest.fixture(scope="module")
def frozen_tasks() -> list[Task]:
    payload = json.loads((INTAKE_DATA_DIR / "tasks.json").read_text())
    return [Task.model_validate(raw) for raw in payload["tasks"]]


def _init_args(task: Task, func_name: str) -> dict:
    for action in task.initial_state.initialization_actions:
        if action.func_name == func_name:
            return action.arguments
    raise AssertionError(f"{task.id} has no {func_name}")


def _gold(task: Task) -> tuple[str, str]:
    ((field, value),) = _init_args(task, "set_entities")["entities"].items()
    return field, value


# ---------------------------------------------------------------------------
# Catalog gates
# ---------------------------------------------------------------------------


def test_catalog_validates():
    validate_complication_catalog()
    validate_profile_rates()


def test_catalog_version_is_bumped_for_profiles():
    # v2.0.0: profiles reshape the draw (per-kind rates instead of
    # rate-then-uniform). v2.1.0: mispronounced_term rate split per bank —
    # trigger outcomes change, the version marks the comparability break.
    # v2.2.0: properties + vehicles gain pronunciation columns (design doc
    # sec. 8b), so mispronounced_term joins their kind sets.
    # v2.3.0: spelling_style is voice-gated — text draws change (the kind's
    # mass goes to none) and the bump re-keys every draw.
    # v2.4.0: spell_correction added (mid-spell-out falter-and-restart,
    # owner directive 2026-09-01) — the bump re-keys every draw.
    assert COMPLICATION_CATALOG_VERSION == "2.4.0"


def test_every_bank_has_at_least_one_kind():
    assert set(BANK_COMPLICATIONS) == set(BANK_NAMES)
    for bank, kinds in BANK_COMPLICATIONS.items():
        assert kinds, f"bank {bank} has no complication kinds"
        assert ComplicationKind.SELF_CORRECTION in kinds


def test_every_kind_and_style_has_a_line_template():
    assert "self_correction" in COMPLICATION_LINES
    assert "wrong_slot" in COMPLICATION_LINES
    assert "mispronounced_term" in COMPLICATION_LINES
    assert (
        "NOT injected; applied at the voice-synthesis seam"
        in (COMPLICATION_LINES["mispronounced_term"])
    )
    for style in SpellingStyle:
        assert f"spelling_style.{style.value}" in COMPLICATION_LINES
    for component in LAZY_OMISSION_COMPONENTS.values():
        assert f"lazy_omission.{component}" in COMPLICATION_LINES


def test_line_templates_are_ascii():
    for key, line in COMPLICATION_LINES.items():
        assert line.isascii(), f"line {key} is not ASCII"


def test_spelling_banks_all_carry_styles():
    for bank, kinds in BANK_COMPLICATIONS.items():
        if ComplicationKind.SPELLING_STYLE in kinds:
            assert BANK_SPELLING_STYLES[bank]


# ---------------------------------------------------------------------------
# Profile catalog: closed, documented rates
# ---------------------------------------------------------------------------


def test_profile_catalog_is_closed():
    assert [profile.value for profile in ComplicationProfile] == ["default", "hard"]
    assert set(PROFILE_KIND_RATES) == set(ComplicationProfile)


def test_default_rates_match_the_documented_constants():
    # DEFAULT carries exactly the bank-invariant kinds; mispronounced_term's
    # rate is per bank (owner directive 2026-08-26).
    assert PROFILE_KIND_RATES[DEFAULT] == {
        ComplicationKind.SELF_CORRECTION: SELF_CORRECTION_RATE,
        ComplicationKind.SPELLING_STYLE: SPELLING_STYLE_RATE,
        ComplicationKind.LAZY_OMISSION: LAZY_OMISSION_RATE,
        ComplicationKind.WRONG_SLOT: WRONG_SLOT_RATE,
        ComplicationKind.SPELL_CORRECTION: SPELL_CORRECTION_RATE,
    }
    assert (SELF_CORRECTION_RATE, SPELLING_STYLE_RATE) == (0.05, 0.15)
    assert (LAZY_OMISSION_RATE, WRONG_SLOT_RATE) == (0.10, 0.03)
    # spell_correction: JUDGMENT at the conservative end of the adjacent
    # restart-disfluency range (owner directive 2026-09-01); conditional on
    # a spell-out actually happening.
    assert SPELL_CORRECTION_RATE == 0.10
    assert MISPRONOUNCED_TERM_BANK_RATES == {
        "medications": MISPRONOUNCED_TERM_MEDICATIONS_RATE,
        "person_names": MISPRONOUNCED_TERM_PERSON_NAMES_RATE,
        # Foreign tokens in hotel/car names are the same construct as person
        # names (design doc sec. 8b) — they share the rate. JUDGMENT.
        "properties": MISPRONOUNCED_TERM_PERSON_NAMES_RATE,
        "vehicles": MISPRONOUNCED_TERM_PERSON_NAMES_RATE,
    }
    assert MISPRONOUNCED_TERM_MEDICATIONS_RATE == 0.50
    assert MISPRONOUNCED_TERM_PERSON_NAMES_RATE == 0.15
    # The remainder is the "none" mass on a fully-feasible task, per bank.
    for bank in MISPRONOUNCED_TERM_BANK_RATES:
        assert sum(profile_kind_rates(DEFAULT, bank).values()) < 1.0


def test_hard_profile_is_rate_one_everywhere():
    assert PROFILE_KIND_RATES[HARD] == {kind: 1.0 for kind in ComplicationKind}


def test_rate_override_scales_uniformly():
    for bank in ("medications", "person_names", "phones"):
        base = profile_kind_rates(DEFAULT, bank)
        total = sum(base.values())
        scaled = effective_kind_rates(DEFAULT, total / 2, bank)
        for kind, rate in base.items():
            assert scaled[kind] == pytest.approx(rate / 2)
        # The override names the fully-feasible trigger probability.
        assert sum(effective_kind_rates(DEFAULT, 0.5, bank).values()) == (
            pytest.approx(0.5)
        )
        assert sum(effective_kind_rates(HARD, 0.5, bank).values()) == (
            pytest.approx(0.5)
        )
        # No override: the bank's profile vector verbatim.
        assert effective_kind_rates(DEFAULT, None, bank) == base


def test_bank_rate_resolution():
    # Mispronounceable banks carry their own rate; every other bank gets an
    # honest 0.0 for the kind (structurally infeasible there anyway).
    meds = profile_kind_rates(DEFAULT, "medications")
    names = profile_kind_rates(DEFAULT, "person_names")
    other = profile_kind_rates(DEFAULT, "phones")
    assert meds[ComplicationKind.MISPRONOUNCED_TERM] == 0.50
    assert names[ComplicationKind.MISPRONOUNCED_TERM] == 0.15
    assert other[ComplicationKind.MISPRONOUNCED_TERM] == 0.0
    # Bank-invariant kinds are identical across banks.
    for kind in ComplicationKind:
        if kind is not ComplicationKind.MISPRONOUNCED_TERM:
            assert meds[kind] == names[kind] == other[kind]
    # HARD is 1.0 everywhere regardless of bank.
    assert profile_kind_rates(HARD, "phones") == {k: 1.0 for k in ComplicationKind}
    with pytest.raises(ValueError, match="Unknown bank"):
        profile_kind_rates(DEFAULT, "not_a_bank")


def test_rate_override_outside_unit_interval_fails_loud():
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match="within \\[0, 1\\]"):
            effective_kind_rates(DEFAULT, bad, "medications")


# ---------------------------------------------------------------------------
# Sampler determinism and the categorical draw
# ---------------------------------------------------------------------------


def test_same_seed_and_task_is_identical(frozen_tasks):
    for task in frozen_tasks[:40]:
        first = sample_complication(11, task, profile=HARD, channel="voice")
        second = sample_complication(11, task, profile=HARD, channel="voice")
        assert first is not None
        assert first.model_dump() == second.model_dump()


def test_different_seeds_vary(frozen_tasks):
    draws_a = [
        sample_complication(1, task, profile=HARD, channel="voice").model_dump()
        for task in frozen_tasks
    ]
    draws_b = [
        sample_complication(2, task, profile=HARD, channel="voice").model_dump()
        for task in frozen_tasks
    ]
    assert draws_a != draws_b


def test_override_zero_never_triggers(frozen_tasks):
    for profile in ComplicationProfile:
        assert all(
            sample_complication(
                5, task, profile=profile, rate_override=0.0, channel="voice"
            )
            is None
            for task in frozen_tasks
        )


def test_unknown_channel_fails_loud(frozen_tasks):
    with pytest.raises(ValueError, match="Unknown run channel"):
        sample_complication(
            5, frozen_tasks[0], profile=DEFAULT, channel="carrier-pigeon"
        )


def test_hard_profile_always_triggers(frozen_tasks):
    for channel in ("text", "voice"):
        for task in frozen_tasks:
            sampled = sample_complication(5, task, profile=HARD, channel=channel)
            assert sampled is not None
            assert sampled.catalog_version == COMPLICATION_CATALOG_VERSION
            assert sampled.line.startswith("Scripted complication")
            assert sampled.line.isascii()
            # The mispronunciation slot is filled for exactly the
            # mispronounced_term kind, and only that kind is prompt-silent.
            is_mispronounced = sampled.kind == ComplicationKind.MISPRONOUNCED_TERM.value
            assert (sampled.mispronunciation is not None) == is_mispronounced
            assert sampled.injected == (not is_mispronounced)
            bank, _tier = _parse_task_id(task.id)
            assert ComplicationKind(sampled.kind) in BANK_COMPLICATIONS[bank]


def test_default_profile_trigger_fraction_matches_the_rates(frozen_tasks):
    """The frozen-set trigger fraction sits near the feasibility-weighted sum
    of the per-kind rates (empirically ~0.2 on voice through catalog v2.3.0;
    ~0.39 from v2.4.0, spell_correction adding 0.10 on nearly every task)."""
    triggered = sum(
        1
        for task in frozen_tasks
        if sample_complication(500, task, profile=DEFAULT, channel="voice") is not None
    )
    fraction = triggered / len(frozen_tasks)
    assert 0.25 <= fraction <= 0.50, fraction


def test_mass_to_none_makes_text_triggers_a_subset_of_voice(frozen_tasks):
    """The channel gates feasibility (mispronounced_term and spelling_style
    are voice-only) and infeasible mass goes to none, so at the same seed a
    task that triggers on text always triggers on voice — never the other
    way around."""
    for seed in (500, 501):
        for task in frozen_tasks:
            text = sample_complication(seed, task, profile=DEFAULT, channel="text")
            voice = sample_complication(seed, task, profile=DEFAULT, channel="voice")
            if text is not None:
                assert voice is not None


def test_text_channel_never_draws_voice_only_kinds(frozen_tasks):
    """spelling_style scripts how a value is VOICED and mispronounced_term
    lives at the voice-synthesis seam (catalog v2.3.0) — a text draw never
    yields either, at any profile."""
    voice_only = {
        ComplicationKind.SPELLING_STYLE.value,
        ComplicationKind.MISPRONOUNCED_TERM.value,
    }
    for profile in (DEFAULT, HARD):
        for seed in (5, 500):
            for task in frozen_tasks:
                sampled = sample_complication(
                    seed, task, profile=profile, channel="text"
                )
                if sampled is not None:
                    assert sampled.kind not in voice_only, task.id


def test_mass_to_none_property_per_kind_rates_unchanged(frozen_tasks):
    """The feasibility-mass-to-none rule, empirically on one task where the
    channel changes the feasible set: fewer feasible kinds mean a LOWER total
    trigger probability, while every shared kind keeps its conditional rate.

    Uses a hard medications task (mispronounced_term feasible on voice,
    infeasible on text) and sweeps seeds; 1500 draws put the 3-sigma band of
    a 0.5 rate at ~0.039, so the 0.05 tolerance separates the hypotheses.
    """
    banks = load_banks()
    task = next(
        t
        for t in frozen_tasks
        if _parse_task_id(t.id) == ("medications", Difficulty.HARD)
        and any(
            e.difficulty is Difficulty.HARD
            and e.value == _gold(t)[1]
            and any(p.mispronounced is not None for p in e.pronunciations)
            for e in banks.medications
        )
    )
    _field, gold_value = _gold(task)
    feasible_voice = [
        kind
        for kind in BANK_COMPLICATIONS["medications"]
        if _kind_feasible(
            kind, "medications", Difficulty.HARD, gold_value, "voice", gold_value
        )
    ]
    feasible_text = [
        kind
        for kind in BANK_COMPLICATIONS["medications"]
        if _kind_feasible(
            kind, "medications", Difficulty.HARD, gold_value, "text", gold_value
        )
    ]
    assert ComplicationKind.MISPRONOUNCED_TERM in feasible_voice
    assert ComplicationKind.MISPRONOUNCED_TERM not in feasible_text
    rates = profile_kind_rates(DEFAULT, "medications")
    n = 1500
    counts = {"text": {}, "voice": {}}
    for channel in ("text", "voice"):
        for seed in range(n):
            sampled = sample_complication(seed, task, profile=DEFAULT, channel=channel)
            if sampled is not None:
                counts[channel][sampled.kind] = counts[channel].get(sampled.kind, 0) + 1
    total_voice = sum(counts["voice"].values()) / n
    total_text = sum(counts["text"].values()) / n
    # Total complication probability equals the feasible kinds' rate sum —
    # lower on the channel with fewer feasible kinds.
    assert total_voice == pytest.approx(
        sum(rates[kind] for kind in feasible_voice), abs=0.05
    )
    assert total_text == pytest.approx(
        sum(rates[kind] for kind in feasible_text), abs=0.05
    )
    assert total_text < total_voice
    # Every shared feasible kind keeps its per-opportunity conditional rate.
    for kind in feasible_text:
        assert counts["text"].get(kind.value, 0) / n == pytest.approx(
            rates[kind], abs=0.05
        )
        assert counts["voice"].get(kind.value, 0) / n == pytest.approx(
            rates[kind], abs=0.05
        )


# ---------------------------------------------------------------------------
# Per-kind invariants (swept over several seeds on the HARD profile so every
# kind is exercised)
# ---------------------------------------------------------------------------


def _draws(frozen_tasks, kind: ComplicationKind, channel: str = "voice"):
    for seed in range(6):
        for task in frozen_tasks:
            sampled = sample_complication(seed, task, profile=HARD, channel=channel)
            if sampled.kind == kind.value:
                yield task, sampled


def test_self_correction_decoy_is_fold_distinct_and_from_the_pool(frozen_tasks):
    seen = 0
    for task, sampled in _draws(frozen_tasks, ComplicationKind.SELF_CORRECTION):
        seen += 1
        bank, tier = _parse_task_id(task.id)
        record = _init_args(task, "seed_record")["record"]
        gold_field, gold_value = _gold(task)
        fold_kind = FoldKind(
            next(
                field["fold"]
                for field in record["fields"]
                if field["name"] == gold_field
            )
        )
        decoy = sampled.params["decoy_value"]
        assert decoy in _pool_values(bank, tier, record["callee_full_name"])
        assert fold_value(fold_kind, decoy) != fold_value(fold_kind, gold_value)
        assert decoy in sampled.line
    assert seen > 0


def test_wrong_slot_answers_with_a_real_context_field(frozen_tasks):
    seen = 0
    for task, sampled in _draws(frozen_tasks, ComplicationKind.WRONG_SLOT):
        seen += 1
        record = _init_args(task, "seed_record")["record"]
        blanked = set(_init_args(task, "blank_fields")["field_names"])
        gold_field, _gold_value = _gold(task)
        off_field = sampled.params["off_slot_field"]
        off_value = sampled.params["off_slot_value"]
        assert off_field != gold_field
        assert off_field not in blanked
        assert any(
            field["name"] == off_field and field["value"] == off_value
            for field in record["fields"]
        )
        assert off_value in sampled.line
    assert seen > 0


def test_lazy_omission_only_fires_on_omittable_banks(frozen_tasks):
    seen_components = set()
    for task, sampled in _draws(frozen_tasks, ComplicationKind.LAZY_OMISSION):
        bank, _tier = _parse_task_id(task.id)
        assert bank in ("times", "dates", "person_names")
        component = sampled.params["omitted_component"]
        assert component == LAZY_OMISSION_COMPONENTS[bank]
        seen_components.add(component)
        if component == "year":
            # A spoken relative date has no year to omit; the feasibility
            # gate must have kept those out.
            _field, gold_value = _gold(task)
            assert any(
                part.isdigit() and len(part) == 4 for part in gold_value.split("-")
            )
        if component == "last_name":
            # A single-token name has no last name to omit; the feasibility
            # gate must have kept those out.
            _field, gold_value = _gold(task)
            assert len(gold_value.split()) >= 2
    assert "am_pm" in seen_components


def test_spell_correction_partial_is_one_edit_from_a_gold_prefix(frozen_tasks):
    """Every drawn faltered partial is the generator's construct exactly:
    one substitute/drop edit on a prefix of the gold spell sequence, with a
    marker from the closed set — and the line carries both verbatim."""
    from tau2.domains.intake.complications import _spell_units
    from tau2.domains.intake.spell_events import _is_faltered_prefix, value_skeleton

    seen = 0
    for task, sampled in _draws(frozen_tasks, ComplicationKind.SPELL_CORRECTION):
        seen += 1
        params = sampled.params
        assert params["operator"] in ("substitute", "drop")
        assert params["marker"] in SPELL_CORRECTION_MARKERS
        _field, gold_raw = _gold(task)
        gold_skeleton = "".join(unit.lower() for unit in _spell_units(gold_raw))
        assert params["sequence_length"] == len(gold_skeleton)
        partial_skeleton = value_skeleton(params["faltered_partial"])
        # One edit from a prefix — or, for a drop of one of a doubled
        # character, an exact prefix (dropping either of "LL" leaves "L").
        assert _is_faltered_prefix(partial_skeleton, gold_skeleton) or (
            params["operator"] == "drop" and gold_skeleton.startswith(partial_skeleton)
        ), (task.id, params)
        assert sampled.injected
        assert params["faltered_partial"] in sampled.line
        assert params["marker"] in sampled.line
    assert seen > 0


def test_spell_correction_never_fires_on_text_channel(frozen_tasks):
    for seed in range(4):
        for task in frozen_tasks:
            sampled = sample_complication(seed, task, profile=HARD, channel="text")
            if sampled is not None:
                assert sampled.kind != ComplicationKind.SPELL_CORRECTION.value


def test_spell_correction_deterministic_across_re_renders(frozen_tasks):
    """The seeded edit is a pure function of (seed, task, catalog): the same
    draw re-renders byte-identically (the packet-vs-run contract)."""
    hits = 0
    for seed in range(6):
        for task in frozen_tasks:
            first = sample_complication(seed, task, profile=HARD, channel="voice")
            if first.kind != ComplicationKind.SPELL_CORRECTION.value:
                continue
            second = sample_complication(seed, task, profile=HARD, channel="voice")
            assert first == second
            hits += 1
            if hits >= 25:
                return
    assert hits > 0


def test_spelling_style_is_feasible_for_the_gold_value(frozen_tasks):
    import re

    seen = 0
    checks = {
        "grouped_numbers": re.compile(r"\d\d"),
        "oh_for_zero": re.compile(r"0"),
        "uk_letters": re.compile(r"[zZ]"),
        "doubled_letters": re.compile(r"([0-9A-Za-z])\1"),
    }
    for task, sampled in _draws(frozen_tasks, ComplicationKind.SPELLING_STYLE):
        seen += 1
        bank, _tier = _parse_task_id(task.id)
        style = SpellingStyle(sampled.params["style"])
        assert style in BANK_SPELLING_STYLES[bank]
        _field, gold_value = _gold(task)
        assert checks[style.value].search(gold_value), (task.id, style, gold_value)
    assert seen > 0


def test_mispronounced_term_never_fires_on_text_channel(frozen_tasks):
    draws = list(
        _draws(frozen_tasks, ComplicationKind.MISPRONOUNCED_TERM, channel="text")
    )
    assert draws == []


def test_mispronounced_term_values_come_from_the_bank(frozen_tasks):
    """Every drawn (term, respelling, operator, mispronunciation) is the
    bank's curated pronunciation column verbatim — never invented — and only
    entries that carry a mispronounced variant ever draw the kind."""
    banks = load_banks()
    seen_banks = set()
    for task, sampled in _draws(frozen_tasks, ComplicationKind.MISPRONOUNCED_TERM):
        bank, tier = _parse_task_id(task.id)
        assert bank in MISPRONOUNCED_TERM_BANKS
        seen_banks.add(bank)
        if bank == "person_names":
            # Only hard-tier person names carry mispronounced variants.
            assert tier is Difficulty.HARD
        _field, gold_value = _gold(task)
        entry = next(
            e
            for e in getattr(banks, bank)
            if e.difficulty is tier and e.value == gold_value
        )
        assert set(sampled.params) == {"term", "respelling", "operator"}
        token = next(
            p for p in entry.pronunciations if p.token == sampled.params["term"]
        )
        assert token.mispronounced is not None
        assert sampled.mispronunciation == token.mispronounced
        assert sampled.params["respelling"] == token.respelling
        assert sampled.params["operator"] == token.operator.value
        # Provenance shape: prompt-silent, packet-display line marked as such.
        assert sampled.injected is False
        assert "NOT injected; applied at the voice-synthesis seam" in sampled.line
        assert sampled.params["term"] in sampled.line
        assert sampled.mispronunciation in sampled.line
    # The catalog still covers vehicles, but the canonical freeze only draws
    # from the final-10 banks (generator CANONICAL_BANKS, 2026-08-26 cut).
    assert seen_banks == set(MISPRONOUNCED_TERM_BANKS) & set(CANONICAL_BANKS)


# ---------------------------------------------------------------------------
# Profile-imposed task selection
# ---------------------------------------------------------------------------


def test_frozen_set_is_a_50_50_tier_mix(frozen_tasks):
    """The DEFAULT profile runs the full frozen set BECAUSE it is already the
    designed 50/50 easy/hard mix — this is the assertion that keeps that
    true (a re-freeze that breaks the mix must also revisit the profile)."""
    hard = sum(1 for task in frozen_tasks if is_hard_tier(task.id))
    assert len(frozen_tasks) == 200
    assert hard == 100


def test_hard_profile_selects_the_hard_tier_only(frozen_tasks):
    from tau2.runner.helpers import resolve_tasks

    resolved = resolve_tasks(TextRunConfig(domain="intake", complication_profile=HARD))
    assert len(resolved.tasks) == 100
    assert all(is_hard_tier(task.id) for task in resolved.tasks)
    assert any("hard tier only" in notice for notice in resolved.notices)


def test_default_profile_runs_the_full_frozen_set(frozen_tasks):
    from tau2.runner.helpers import resolve_tasks

    resolved = resolve_tasks(TextRunConfig(domain="intake"))
    assert len(resolved.tasks) == len(frozen_tasks)
    hard = sum(1 for task in resolved.tasks if is_hard_tier(task.id))
    assert hard == 100


def test_hard_profile_on_a_domain_without_tiers_fails_loud():
    from tau2.runner.complications import profile_task_filter

    with pytest.raises(ValueError, match="no hard-tier predicate"):
        profile_task_filter(TextRunConfig(domain="airline", complication_profile=HARD))
    # The DEFAULT profile imposes nothing anywhere.
    assert profile_task_filter(TextRunConfig(domain="airline")) is None


# ---------------------------------------------------------------------------
# Injection through the user_prompt_task seam
# ---------------------------------------------------------------------------


def test_injection_appends_exactly_the_sampled_line(frozen_tasks):
    from tau2.runner.build import user_prompt_task

    config = TextRunConfig(domain="intake", complication_profile=HARD, seed=123)
    task = frozen_tasks[0]
    before = task.model_dump()
    prompt_task = user_prompt_task(config, task, None)
    sampled = sample_complication(123, task, profile=HARD, channel="text")
    assert prompt_task is not task
    assert prompt_task.user_scenario.instructions.task_instructions.endswith(
        "\n\n" + sampled.line
    )
    assert str(prompt_task.user_scenario).rstrip().endswith(sampled.line)
    # The original task object — what the evaluator/environment sees — is
    # untouched.
    assert task.model_dump() == before


def _voice_config(seed: int, **kwargs) -> VoiceRunConfig:
    return VoiceRunConfig(
        domain="intake",
        complication_profile=HARD,
        seed=seed,
        audio_native_config=AudioNativeConfig(),
        **kwargs,
    )


def test_mispronounced_term_is_never_injected(frozen_tasks):
    """A mispronounced_term draw leaves the user-sim prompt byte-identical to
    an uncomplicated run: user_prompt_task appends nothing."""
    from tau2.runner.build import user_prompt_task

    seen = 0
    for seed in range(6):
        for task in frozen_tasks:
            sampled = sample_complication(seed, task, profile=HARD, channel="voice")
            if sampled.kind != ComplicationKind.MISPRONOUNCED_TERM.value:
                continue
            seen += 1
            armed = user_prompt_task(_voice_config(seed), task, None)
            clean = user_prompt_task(
                _voice_config(seed, complication_rate=0.0), task, None
            )
            assert armed is task  # no copy, nothing appended
            assert armed.model_dump() == clean.model_dump()
            assert str(armed.user_scenario) == str(clean.user_scenario)
    assert seen > 0


def test_voice_and_text_channels_thread_through_the_dispatcher(frozen_tasks):
    """sample_run_complication derives the channel from the config type, so a
    voice run can draw mispronounced_term and a text run never does."""
    from tau2.runner.complications import sample_run_complication

    voice_kinds = set()
    text_kinds = set()
    for seed in range(6):
        for task in frozen_tasks:
            voice = sample_run_complication(_voice_config(seed), task)
            text = sample_run_complication(
                TextRunConfig(domain="intake", complication_profile=HARD, seed=seed),
                task,
            )
            voice_kinds.add(voice.kind)
            text_kinds.add(text.kind)
    assert ComplicationKind.MISPRONOUNCED_TERM.value in voice_kinds
    assert ComplicationKind.MISPRONOUNCED_TERM.value not in text_kinds


def test_run_channel_fails_loud_on_unknown_config_types():
    from tau2.runner.complications import run_channel

    assert run_channel(TextRunConfig(domain="intake")) == "text"
    assert (
        run_channel(
            VoiceRunConfig(domain="intake", audio_native_config=AudioNativeConfig())
        )
        == "voice"
    )

    class NotARunConfig:
        domain = "intake"
        seed = 1
        complication_profile = HARD
        complication_rate = None

    with pytest.raises(ValueError, match="Cannot determine the run channel"):
        run_channel(NotARunConfig())


def test_override_zero_leaves_the_prompt_byte_identical(frozen_tasks):
    from tau2.runner.build import user_prompt_task

    task = frozen_tasks[0]
    for config in (
        TextRunConfig(domain="intake", complication_rate=0.0, seed=123),
        TextRunConfig(
            domain="intake",
            complication_profile=HARD,
            complication_rate=0.0,
            seed=123,
        ),
    ):
        prompt_task = user_prompt_task(config, task, None)
        assert prompt_task is task
        assert str(prompt_task.user_scenario) == str(task.user_scenario)


def test_domains_without_a_sampler_run_clean_unless_overridden(frozen_tasks):
    from tau2.runner.complications import sample_run_complication

    # The profile field's default arms nothing on a sampler-less domain...
    clean = TextRunConfig(domain="airline", seed=1)
    assert sample_run_complication(clean, frozen_tasks[0]) is None
    # ...but an explicit rate override there fails loud.
    armed = TextRunConfig(domain="airline", complication_rate=0.5, seed=1)
    with pytest.raises(ValueError, match="no complication sampler"):
        sample_run_complication(armed, frozen_tasks[0])


# ---------------------------------------------------------------------------
# Recording / round-trips
# ---------------------------------------------------------------------------


def test_sampled_complication_round_trips(frozen_tasks):
    sampled = sample_complication(9, frozen_tasks[0], profile=HARD, channel="voice")
    restored = SampledComplication.model_validate_json(sampled.model_dump_json())
    assert restored == sampled


def test_mispronounced_term_round_trips_with_provenance(frozen_tasks):
    sampled = next(
        s for _t, s in _draws(frozen_tasks, ComplicationKind.MISPRONOUNCED_TERM)
    )
    restored = SampledComplication.model_validate_json(sampled.model_dump_json())
    assert restored == sampled
    assert restored.injected is False
    assert restored.mispronunciation == sampled.mispronunciation


def test_simulation_run_round_trips_with_complication(frozen_tasks):
    sampled = sample_complication(9, frozen_tasks[0], profile=HARD, channel="voice")
    run = SimulationRun(
        id="sim-1",
        task_id=frozen_tasks[0].id,
        start_time="2026-08-25T00:00:00",
        end_time="2026-08-25T00:01:00",
        duration=60.0,
        termination_reason="user_stop",
        complication=sampled,
    )
    restored = SimulationRun.model_validate_json(run.model_dump_json())
    assert restored.complication == sampled
    # And a run without one stays None (results predating the field).
    bare = SimulationRun.model_validate_json(
        run.model_copy(update={"complication": None}).model_dump_json()
    )
    assert bare.complication is None


def test_info_records_profile_and_override_and_refill_restores_them(frozen_tasks):
    from tau2.runner.helpers import get_info
    from tau2.runner.refill import reconstruct_config

    config = TextRunConfig(
        domain="intake",
        complication_profile=HARD,
        complication_rate=0.5,
        seed=3,
    )
    info = get_info(config)
    assert info.complication_profile == "hard"
    assert info.complication_rate == 0.5
    # Round-trips through Info serialization.
    restored_info = type(info).model_validate_json(info.model_dump_json())
    assert restored_info.complication_profile == "hard"
    rebuilt = reconstruct_config(
        restored_info, run_dir="/tmp/run", task_ids=[frozen_tasks[0].id]
    )
    assert rebuilt.config.complication_profile is HARD
    assert rebuilt.config.complication_rate == 0.5
    assert "complication_profile" not in rebuilt.unrecorded
    assert "complication_rate" not in rebuilt.unrecorded


def test_refill_treats_no_override_as_a_real_record(frozen_tasks):
    from tau2.runner.helpers import get_info
    from tau2.runner.refill import reconstruct_config

    info = get_info(TextRunConfig(domain="intake", seed=3))
    assert info.complication_profile == "default"
    assert info.complication_rate is None  # profile rates ran verbatim
    rebuilt = reconstruct_config(
        info, run_dir="/tmp/run", task_ids=[frozen_tasks[0].id]
    )
    assert rebuilt.config.complication_profile is DEFAULT
    assert rebuilt.config.complication_rate is None
    assert "complication_rate" not in rebuilt.unrecorded


# ---------------------------------------------------------------------------
# Review packet
# ---------------------------------------------------------------------------


def test_packet_is_deterministic_and_counts_add_up(frozen_tasks):
    first = build_complications_packet(seed=500, profile=DEFAULT, channel="voice")
    second = build_complications_packet(seed=500, profile=DEFAULT, channel="voice")
    assert first.markdown == second.markdown
    assert first.channel == "voice"
    assert first.total_tasks == len(frozen_tasks)
    assert first.tier_mix == {"easy": 100, "hard": 100}
    assert first.triggered == sum(first.per_kind.values())
    assert 0 < first.triggered < first.total_tasks
    # Every triggered task's exact line is in the packet body.
    for task in frozen_tasks:
        sampled = sample_complication(500, task, profile=DEFAULT, channel="voice")
        if sampled is not None:
            assert f"## {task.id}" in first.markdown
            assert sampled.line in first.markdown


def test_packet_records_profile_and_rates_in_provenance_header():
    packet = build_complications_packet(seed=500, profile=DEFAULT, channel="voice")
    header = packet.markdown.splitlines()[2]
    assert "profile default" in header
    assert "rate override none" in header
    assert "channel voice" in header
    for kind, rate in PROFILE_KIND_RATES[DEFAULT].items():
        assert f"- {kind.value}: {rate:.4f}" in packet.markdown
    for bank, rate in MISPRONOUNCED_TERM_BANK_RATES.items():
        assert f"- mispronounced_term ({bank}): {rate:.4f}" in packet.markdown
    expected_rows = {
        kind.value: rate for kind, rate in PROFILE_KIND_RATES[DEFAULT].items()
    } | {
        f"mispronounced_term ({bank})": rate
        for bank, rate in MISPRONOUNCED_TERM_BANK_RATES.items()
    }
    assert packet.per_kind_rates == expected_rows
    # An override shows in the header; the recorded rates stay the BASE
    # vector (the override rescales each task's bank vector at draw time).
    overridden = build_complications_packet(
        seed=500, profile=DEFAULT, channel="voice", rate_override=0.315
    )
    assert "rate override 0.315" in overridden.markdown.splitlines()[2]
    assert overridden.rate_override == 0.315
    assert overridden.per_kind_rates == expected_rows


def test_hard_packet_renders_hard_tier_tasks_only():
    packet = build_complications_packet(seed=500, profile=HARD, channel="voice")
    assert packet.total_tasks == 100
    assert packet.tier_mix == {"easy": 0, "hard": 100}
    assert packet.triggered == 100  # every call draws under HARD
    assert "profile hard" in packet.markdown.splitlines()[2]
    assert "_easy_" not in packet.markdown.split("## Summary")[1]


def test_packet_channel_gates_the_voice_only_kind(frozen_tasks):
    voice = build_complications_packet(seed=500, profile=DEFAULT, channel="voice")
    text = build_complications_packet(seed=500, profile=DEFAULT, channel="text")
    assert "channel voice" in voice.markdown.splitlines()[2]
    assert "channel text" in text.markdown.splitlines()[2]
    # A text draw never contains the voice-only kind, and (mass-to-none) the
    # text draw triggers no MORE tasks than the voice draw.
    assert text.per_kind[ComplicationKind.MISPRONOUNCED_TERM.value] == 0
    assert text.triggered <= voice.triggered


def test_packet_marks_prompt_silent_draws():
    # The HARD profile guarantees mispronounced_term draws appear.
    packet = build_complications_packet(seed=500, profile=HARD, channel="voice")
    assert packet.per_kind[ComplicationKind.MISPRONOUNCED_TERM.value] > 0
    assert (
        "Packet-display line (NOT injected; applied at the voice-synthesis seam):"
        in (packet.markdown)
    )


# ---------------------------------------------------------------------------
# Catalog validation for kind/bank drift
# ---------------------------------------------------------------------------


def test_catalog_rejects_mispronounced_term_on_a_bank_without_columns(monkeypatch):
    monkeypatch.setitem(
        BANK_COMPLICATIONS,
        "codes",
        BANK_COMPLICATIONS["codes"] + [ComplicationKind.MISPRONOUNCED_TERM],
    )
    with pytest.raises(ValueError, match="mispronounced_term applies to exactly"):
        validate_complication_catalog()


def test_catalog_rejects_mispronounced_term_missing_from_a_column_bank(monkeypatch):
    monkeypatch.setitem(
        BANK_COMPLICATIONS,
        "medications",
        [
            kind
            for kind in BANK_COMPLICATIONS["medications"]
            if kind is not ComplicationKind.MISPRONOUNCED_TERM
        ],
    )
    with pytest.raises(ValueError, match="mispronounced_term applies to exactly"):
        validate_complication_catalog()


def test_profile_rates_reject_a_missing_kind(monkeypatch):
    import tau2.domains.intake.complications as complications

    broken = {profile: dict(rates) for profile, rates in PROFILE_KIND_RATES.items()}
    del broken[DEFAULT][ComplicationKind.WRONG_SLOT]
    monkeypatch.setattr(complications, "PROFILE_KIND_RATES", broken)
    with pytest.raises(ValueError, match="bank-invariant kinds"):
        validate_profile_rates()


def test_profile_rates_reject_a_hard_rate_below_one(monkeypatch):
    import tau2.domains.intake.complications as complications

    broken = {profile: dict(rates) for profile, rates in PROFILE_KIND_RATES.items()}
    broken[HARD][ComplicationKind.WRONG_SLOT] = 0.5
    monkeypatch.setattr(complications, "PROFILE_KIND_RATES", broken)
    with pytest.raises(ValueError, match="HARD profile puts every kind at 1.0"):
        validate_profile_rates()
