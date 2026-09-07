"""Tests for the v4 task generator (design doc §6): determinism, the frozen
canonical artifact, cell coverage, contract enforcement, and manifest
provenance."""

import json
import shutil

import pytest
import yaml

from tau2.domains.intake.call_frame import CALL_FRAME_VERSION
from tau2.domains.intake.tasks.banks import (
    INTAKE_BANKS_DIR,
    load_banks,
)
from tau2.domains.intake.tasks.generator import (
    CANONICAL_BANKS,
    DEFAULT_FREEZE_SEED,
    DRAW_PER_CELL,
    GENERATOR_VERSION,
    MANIFEST_FILENAME,
    SPLITS_FILENAME,
    TASKS_FILENAME,
    FreezeManifest,
    GeneratorError,
    _manifest_payload,
    _splits_payload,
    _task_payload,
    enumerate_frame,
    freeze_tasks,
    generate_tasks,
)
from tau2.domains.intake.utils import INTAKE_DATA_DIR


@pytest.fixture(scope="module")
def canonical():
    """One canonical generation, shared across the module's tests."""
    return generate_tasks(seed=DEFAULT_FREEZE_SEED, n_entities=1)


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------


def test_enumerate_frame_reports_the_design_numbers():
    frame = enumerate_frame()
    assert frame.banks == 10
    assert frame.tiers == 2
    assert frame.values_per_cell == 40
    assert frame.frame_size == 800
    assert frame.canonical_size == 200


# ---------------------------------------------------------------------------
# Determinism + the checked-in freeze
# ---------------------------------------------------------------------------


def test_same_seed_regenerates_byte_identically(canonical):
    tasks_a, manifest_a = canonical
    tasks_b, manifest_b = generate_tasks(seed=DEFAULT_FREEZE_SEED, n_entities=1)
    assert _task_payload(tasks_a) == _task_payload(tasks_b)
    assert _splits_payload(tasks_a) == _splits_payload(tasks_b)
    assert _manifest_payload(manifest_a) == _manifest_payload(manifest_b)


def test_a_different_seed_draws_a_different_set(canonical):
    tasks_default, _ = canonical
    tasks_other, _ = generate_tasks(seed=DEFAULT_FREEZE_SEED + 1, n_entities=1)
    values = lambda tasks: [  # noqa: E731
        t.evaluation_criteria.actions[0].arguments["fields"] for t in tasks
    ]
    assert values(tasks_default) != values(tasks_other)


def test_checked_in_freeze_is_the_default_seed_regeneration(canonical):
    """tasks.json / split_tasks.json / tasks.manifest.json are exactly what
    `tau2 intake-tasks freeze` writes — a code change that would alter them
    must bump GENERATOR_VERSION and re-freeze, visibly."""
    tasks, manifest = canonical
    assert (INTAKE_DATA_DIR / TASKS_FILENAME).read_text() == _task_payload(tasks)
    assert (INTAKE_DATA_DIR / SPLITS_FILENAME).read_text() == _splits_payload(tasks)
    assert (INTAKE_DATA_DIR / MANIFEST_FILENAME).read_text() == _manifest_payload(
        manifest
    )


# ---------------------------------------------------------------------------
# Cell coverage
# ---------------------------------------------------------------------------


def test_canonical_draw_covers_every_cell_without_reuse(canonical):
    tasks, manifest = canonical
    assert len(tasks) == 200
    assert len(manifest.cells) == 20
    seen_cells = set()
    for cell in manifest.cells:
        seen_cells.add((cell.bank, cell.tier))
        assert len(cell.drawn) == DRAW_PER_CELL, (cell.bank, cell.tier)
        assert len(set(cell.drawn)) == DRAW_PER_CELL, (cell.bank, cell.tier)
    assert seen_cells == {
        (bank, tier) for bank in CANONICAL_BANKS for tier in ("easy", "hard")
    }
    by_cell = {}
    for draw in manifest.tasks:
        by_cell.setdefault((draw.bank, draw.tier), []).append(draw)
    for cell_key, draws in by_cell.items():
        assert len(draws) == DRAW_PER_CELL, cell_key
        captured = [d.value for d in draws]
        assert len(set(captured)) == DRAW_PER_CELL, cell_key


def test_cell_draws_come_from_the_bank_contract_pool(canonical):
    """Every drawn key is a real entry of its bank x tier (emails by pattern
    key), so the manifest is auditable against the checked-in banks."""
    _, manifest = canonical
    banks = load_banks()
    for cell in manifest.cells:
        entries = [
            e for e in getattr(banks, cell.bank) if e.difficulty.value == cell.tier
        ]
        if cell.bank == "emails":
            pool = {f"{e.pattern}@{e.domain}" for e in entries}
        else:
            pool = {e.value for e in entries}
        assert set(cell.drawn) <= pool, (cell.bank, cell.tier)


# ---------------------------------------------------------------------------
# Contract enforcement: the generator refuses a violating bank
# ---------------------------------------------------------------------------


def test_generator_refuses_a_short_bank(tmp_path, monkeypatch):
    banks_dir = tmp_path / "banks"
    shutil.copytree(INTAKE_BANKS_DIR, banks_dir)
    path = banks_dir / "shops.yaml"
    raw = yaml.safe_load(path.read_text())
    easy = next(e for e in raw["entries"] if e["difficulty"] == "easy")
    raw["entries"].remove(easy)
    path.write_text(yaml.safe_dump(raw))
    violating = load_banks(banks_dir)
    with pytest.raises(ValueError, match=r"bank shops: 39 easy values \(short by 1"):
        generate_tasks(seed=DEFAULT_FREEZE_SEED, n_entities=1, banks=violating)


def test_multi_entity_draw_never_touches_the_canonical_files():
    with pytest.raises(GeneratorError, match="explicit out_dir"):
        freeze_tasks(seed=DEFAULT_FREEZE_SEED, n_entities=3)


def test_invalid_n_entities_fails_loud():
    with pytest.raises(GeneratorError, match="n_entities"):
        generate_tasks(seed=DEFAULT_FREEZE_SEED, n_entities=4)


# ---------------------------------------------------------------------------
# Manifest provenance
# ---------------------------------------------------------------------------


def test_manifest_round_trips_and_stamps_provenance(canonical):
    _, manifest = canonical
    on_disk = FreezeManifest.model_validate(
        json.loads((INTAKE_DATA_DIR / MANIFEST_FILENAME).read_text())
    )
    assert on_disk == manifest
    assert on_disk.generator_version == GENERATOR_VERSION
    assert on_disk.call_frame_version == CALL_FRAME_VERSION
    assert on_disk.seed == DEFAULT_FREEZE_SEED
    assert on_disk.n_entities == 1
    assert on_disk.bank_contract.easy_per_bank == 40
    assert on_disk.bank_contract.hard_per_bank == 40
    assert on_disk.bank_contract.draw_per_cell == 10
    assert on_disk.frame_size == 800
    assert on_disk.task_count == 200
    assert on_disk.bank_shas == load_banks().shas
    assert len(on_disk.tasks) == 200


def test_manifest_task_draws_match_the_task_file(canonical):
    tasks, manifest = canonical
    by_id = {t.id: t for t in tasks}
    for draw in manifest.tasks:
        task = by_id[draw.task_id]
        submit = task.evaluation_criteria.actions[0]
        assert submit.arguments["record_id"] == draw.record_id
        assert submit.arguments["fields"] == {draw.field_name: draw.value}
        record = next(
            a
            for a in task.initial_state.initialization_actions
            if a.func_name == "seed_record"
        ).arguments["record"]
        assert record["callee_full_name"] == draw.callee_full_name
        assert record["submitted_on"] == draw.submitted_on
        assert task.agent_opener == f"Hello, this is {draw.org_name} calling."


# ---------------------------------------------------------------------------
# Multi-entity bands (the kept n_entities knob)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_entities", [2, 3])
def test_multi_entity_bands_generate_and_bundle_same_vertical(n_entities):
    tasks, manifest = generate_tasks(seed=DEFAULT_FREEZE_SEED, n_entities=n_entities)
    assert manifest.n_entities == n_entities
    assert len(tasks) == 60  # 3 verticals x 2 tiers x 10 calls
    for task in tasks:
        submit = task.evaluation_criteria.actions[0]
        assert len(submit.arguments["fields"]) == n_entities, task.id


# ---------------------------------------------------------------------------
# Dual-form entity payloads (relative dates, frame 4.5.0)
# ---------------------------------------------------------------------------


def test_relative_date_entities_carry_both_forms(canonical):
    from tau2.domains.intake.utils import spoken_form

    tasks, _ = canonical
    hard_dates = [t for t in tasks if t.id.startswith("intake_dates_hard_")]
    assert len(hard_dates) == DRAW_PER_CELL
    for task in hard_dates:
        entities = next(
            a
            for a in task.initial_state.initialization_actions
            if a.func_name == "set_entities"
        ).arguments["entities"]
        ((field, payload),) = entities.items()
        assert field == "drop_off_date"
        gold = task.evaluation_criteria.actions[0].arguments["fields"][field]
        # The payload names the real written date and round-trips: stripping
        # the annotation leaves exactly the spoken phrase, which never
        # contains the ISO form itself.
        assert payload.endswith(f", from your records: {gold})")
        spoken = spoken_form(payload)
        assert spoken != payload
        assert gold not in spoken


def test_spoken_form_is_identity_for_plain_values():
    from tau2.domains.intake.utils import render_dual_form, spoken_form

    assert spoken_form("Michael Santiago") == "Michael Santiago"
    assert spoken_form("2:45 PM") == "2:45 PM"
    payload = render_dual_form("the fourth Monday of next month", "2025-07-28")
    assert spoken_form(payload) == "the fourth Monday of next month"
