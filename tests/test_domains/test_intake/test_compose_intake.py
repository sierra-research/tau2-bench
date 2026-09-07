"""Composition bands: paired flat bundles + staged chain twins.

Covers the entity-composition experiment machinery
(docs/designs/intake-entity-composition.md, ``tau2 intake-tasks compose``):

1. **Determinism + checked-in bands** — same seed regenerates every band and
   the manifest byte-identically, and the checked-in ``bands/`` files are
   exactly the default-seed regeneration (the freeze-regression discipline).
2. **Shape** — 60 n=2 + 30 n=3 calls per protocol, tier-pure cells, distinct
   banks and fields within a call, position balance, bounded value reuse.
3. **Parent pairing** — every slot pairs to a canonical atomic task with the
   same bank/tier/entry; non-email slots carry the parent's exact value,
   email slots re-instantiate their parent's pattern against this callee.
4. **Twin alignment** — a chain task and its flat twin share callee, org,
   record, values, and slot order; only protocol differs.
5. **Golden replay reward** — a flat bundle and a staged chain both score
   1.0 through the real evaluator on their own domains; dropping a stage
   scores 0.0.
"""

import json
from collections import defaultdict

import pytest

from tau2.data_model.message import AssistantMessage, ToolCall
from tau2.data_model.tasks import Task
from tau2.domains.intake.environment import get_environment as get_flat_environment
from tau2.domains.intake.environment import get_tasks as get_intake_tasks
from tau2.domains.intake.staged import (
    get_environment as get_staged_environment,
)
from tau2.domains.intake.staged import get_tasks as get_staged_tasks
from tau2.domains.intake.tasks.compose import (
    BANDS_DIR,
    CALLS_PER_CELL,
    CHAIN_BAND_NAMES,
    COMPOSE_MANIFEST_FILENAME,
    FLAT_BAND_NAMES,
    MAX_PARENT_USES_PER_BAND,
    _splits_payload,
    _task_payload,
    compose_tasks,
)
from tau2.domains.intake.tasks.generator import MANIFEST_FILENAME
from tau2.domains.intake.utils import INTAKE_DATA_DIR
from tau2.evaluator.evaluator_env import EnvironmentEvaluator


@pytest.fixture(scope="module")
def composed():
    """One default-seed composition, shared across the module."""
    return compose_tasks()


@pytest.fixture(scope="module")
def canonical_manifest() -> dict:
    return json.loads((INTAKE_DATA_DIR / MANIFEST_FILENAME).read_text())


# ---------------------------------------------------------------------------
# Determinism + the checked-in bands
# ---------------------------------------------------------------------------


def test_same_seed_regenerates_byte_identically(composed):
    bands_a, manifest_a = composed
    bands_b, manifest_b = compose_tasks()
    for name in (*FLAT_BAND_NAMES, *CHAIN_BAND_NAMES):
        assert _task_payload(bands_a[name]) == _task_payload(bands_b[name]), name
    assert manifest_a.model_dump() == manifest_b.model_dump()


def test_checked_in_bands_are_the_default_seed_regeneration(composed):
    """bands/*/tasks.json and the compose manifest are exactly what
    `tau2 intake-tasks compose` writes — a code change that would alter them
    must bump COMPOSE_VERSION and re-compose, visibly."""
    bands, manifest = composed
    for name, tasks in bands.items():
        band_dir = BANDS_DIR / name
        assert (band_dir / "tasks.json").read_text() == _task_payload(tasks), name
        assert (band_dir / "split_tasks.json").read_text() == _splits_payload(tasks), (
            name
        )
    checked_in = json.loads((BANDS_DIR / COMPOSE_MANIFEST_FILENAME).read_text())
    assert checked_in == manifest.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


def test_band_shapes(composed):
    bands, manifest = composed
    assert {name: len(tasks) for name, tasks in bands.items()} == {
        "compose_n2": 6 * CALLS_PER_CELL[2],
        "compose_n3": 6 * CALLS_PER_CELL[3],
        "chain_n2": 6 * CALLS_PER_CELL[2],
        "chain_n3": 6 * CALLS_PER_CELL[3],
    }
    assert len(manifest.tasks) == 90
    for draw in manifest.tasks:
        assert len(draw.slots) == draw.n_entities
        banks = [slot.bank for slot in draw.slots]
        assert len(set(banks)) == draw.n_entities, draw.task_id
        fields = [slot.field_name for slot in draw.slots]
        assert len(set(fields)) == draw.n_entities, draw.task_id
        assert [slot.slot_index for slot in draw.slots] == list(
            range(1, draw.n_entities + 1)
        )
        assert all(slot.tier == draw.tier for slot in draw.slots), draw.task_id


def test_value_reuse_is_bounded_and_flagged(composed):
    _, manifest = composed
    uses: dict[tuple[int, str], int] = defaultdict(int)
    for draw in manifest.tasks:
        parents_in_call = [slot.parent_task_id for slot in draw.slots]
        assert len(set(parents_in_call)) == len(parents_in_call), draw.task_id
        for slot in draw.slots:
            uses[(draw.n_entities, slot.parent_task_id)] += 1
    for (n, parent_id), count in uses.items():
        assert count <= MAX_PARENT_USES_PER_BAND, (n, parent_id)
    flagged = {
        (draw.n_entities, slot.parent_task_id)
        for draw in manifest.tasks
        for slot in draw.slots
        if slot.reused
    }
    multi_use = {key for key, count in uses.items() if count > 1}
    assert flagged == multi_use


def test_position_balance_within_each_cell(composed):
    _, manifest = composed
    cells: dict[tuple, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0, 0])
    )
    for draw in manifest.tasks:
        for slot in draw.slots:
            cells[(draw.n_entities, draw.vertical, draw.tier)][slot.bank][
                slot.slot_index - 1
            ] += 1
    for (n, vertical, tier), banks in cells.items():
        for bank, counts in banks.items():
            live = counts[:n]
            assert max(live) - min(live) <= 1, (n, vertical, tier, bank, live)


# ---------------------------------------------------------------------------
# Parent pairing
# ---------------------------------------------------------------------------


def test_every_slot_pairs_to_a_canonical_task(composed, canonical_manifest):
    _, manifest = composed
    parents = {draw["task_id"]: draw for draw in canonical_manifest["tasks"]}
    assert manifest.parent_seed == canonical_manifest["seed"]
    assert manifest.parent_generator_version == canonical_manifest["generator_version"]
    for draw in manifest.tasks:
        for slot in draw.slots:
            parent = parents[slot.parent_task_id]
            assert slot.bank == parent["bank"], slot
            assert slot.tier.value == parent["tier"], slot
            assert slot.field_name == parent["field_name"], slot
            assert slot.parent_value == parent["value"], slot
            expected_key = parent["email_pattern"] or parent["value"]
            assert slot.entry_key == expected_key, slot
            if slot.bank == "emails":
                # Same pattern, re-instantiated against this call's callee.
                assert slot.value.count("@") == 1
            else:
                assert slot.value == parent["value"], slot


def test_clean_parents_are_preferred(composed, canonical_manifest):
    """Triggered-reference parents may appear (small cells force them) but
    never while an unused clean parent of the same cell bank remains."""
    _, manifest = composed
    triggered = sum(
        1
        for draw in manifest.tasks
        for slot in draw.slots
        if slot.parent_triggered_reference
    )
    total = sum(len(draw.slots) for draw in manifest.tasks)
    # ~25% of canonical parents trigger under the reference draw; clean-first
    # ordering must keep the drawn share well under that.
    assert triggered / total < 0.20, (triggered, total)


# ---------------------------------------------------------------------------
# Twin alignment
# ---------------------------------------------------------------------------


def _submit_actions(task: Task) -> list:
    return [a for a in task.evaluation_criteria.actions if a.name == "submit_fields"]


def test_chain_twins_match_their_flat_bundles(composed):
    bands, manifest = composed
    flat_by_id = {t.id: t for name in FLAT_BAND_NAMES for t in bands[name]}
    chain_by_id = {t.id: t for name in CHAIN_BAND_NAMES for t in bands[name]}
    assert len(chain_by_id) == len(flat_by_id) == 90
    for draw in manifest.tasks:
        flat = flat_by_id[draw.task_id]
        chain = chain_by_id[draw.chain_task_id]
        assert chain.user_scenario == flat.user_scenario
        assert chain.agent_opener == flat.agent_opener

        flat_init = {c.func_name: c for c in flat.initial_state.initialization_actions}
        chain_init = {
            c.func_name: c for c in chain.initial_state.initialization_actions
        }
        assert set(chain_init) == set(flat_init) | {"stage_fields"}
        assert chain_init["seed_record"] == flat_init["seed_record"]
        assert chain_init["blank_fields"] == flat_init["blank_fields"]
        assert chain_init["set_entities"] == flat_init["set_entities"]
        assert (
            chain_init["stage_fields"].arguments["field_names"]
            == flat_init["blank_fields"].arguments["field_names"]
        )

        (flat_submit,) = _submit_actions(flat)
        chain_submits = _submit_actions(chain)
        staged_fields = {}
        for action in chain_submits:
            assert len(action.arguments["fields"]) == 1
            staged_fields.update(action.arguments["fields"])
        assert staged_fields == flat_submit.arguments["fields"]
        assert [next(iter(a.arguments["fields"])) for a in chain_submits] == list(
            flat_submit.arguments["fields"]
        )


# ---------------------------------------------------------------------------
# Golden replay reward (through the real evaluator, on the real domains)
# ---------------------------------------------------------------------------


def _replay_reward(task: Task, env_constructor, actions) -> float:
    """Score a trajectory of golden-style actions through the real evaluator
    (tolerates refused tool calls — a rejected submission is a real
    trajectory whose reward must be computable)."""
    env = env_constructor()
    env.set_state(
        initialization_data=None,
        initialization_actions=task.initial_state.initialization_actions,
        message_history=[],
    )
    messages = []
    for index, action in enumerate(actions):
        tool_call = ToolCall(
            id=f"sim_{index}",
            name=action.name,
            arguments=action.arguments,
            requestor=action.requestor,
        )
        messages.append(
            AssistantMessage(role="assistant", content=None, tool_calls=[tool_call])
        )
        messages.append(env.get_response(tool_call))
    return EnvironmentEvaluator.calculate_reward(
        environment_constructor=env_constructor,
        task=task,
        full_trajectory=messages,
    ).reward


def test_flat_golden_replay_scores_one(composed):
    bands, _ = composed
    for task in (bands["compose_n2"][0], bands["compose_n3"][-1]):
        reward = _replay_reward(
            task, get_flat_environment, task.evaluation_criteria.actions
        )
        assert reward == 1.0, task.id


def test_chain_golden_replay_scores_one(composed):
    bands, _ = composed
    for task in (bands["chain_n2"][0], bands["chain_n3"][-1]):
        reward = _replay_reward(
            task, get_staged_environment, task.evaluation_criteria.actions
        )
        assert reward == 1.0, task.id


def test_chain_missing_stage_scores_zero(composed):
    bands, _ = composed
    task = bands["chain_n2"][0]
    reward = _replay_reward(task, get_staged_environment, _submit_actions(task)[:1])
    assert reward == 0.0


def test_chain_out_of_order_stage_scores_zero(composed):
    """Submitting only the SECOND stage is refused by the gate, leaving the
    record incomplete — reward 0, no partial credit."""
    bands, _ = composed
    task = bands["chain_n2"][0]
    reward = _replay_reward(task, get_staged_environment, _submit_actions(task)[1:])
    assert reward == 0.0


# ---------------------------------------------------------------------------
# Band loading (the run seam)
# ---------------------------------------------------------------------------


def test_flat_bands_load_as_intake_splits(composed):
    bands, _ = composed
    for name in FLAT_BAND_NAMES:
        loaded = get_intake_tasks(name)
        assert [t.id for t in loaded] == [t.id for t in bands[name]]


def test_chain_bands_load_as_staged_splits(composed):
    bands, _ = composed
    for name in CHAIN_BAND_NAMES:
        loaded = get_staged_tasks(name)
        assert [t.id for t in loaded] == [t.id for t in bands[name]]
    base = get_staged_tasks("base")
    assert len(base) == 90
    with pytest.raises(ValueError, match="Invalid task split"):
        get_staged_tasks("compose_n2")


def test_canonical_intake_loading_is_untouched():
    tasks = get_intake_tasks()
    assert len(tasks) == 200
    band_prefixes = ("intake_c2_", "intake_c3_", "intake_h2_", "intake_h3_")
    assert not any(t.id.startswith(band_prefixes) for t in tasks)
