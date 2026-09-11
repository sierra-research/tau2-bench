# Copyright Sierra
"""Fixed task subsets: the draw, the resolution rules, and the shipped files.

The shipped-artifact tests are the load-bearing ones: they are what catches a
subset silently drifting away from the frame it claims, or a language pack
whose task set can no longer supply every pinned task.
"""

import json

import pytest

from tau2.config import CANONICAL_TASK_SUBSETS
from tau2.data_model.tasks import Task, UserInstructions, UserScenario
from tau2.multilingual.task_sets import discover_localized_task_files
from tau2.runner.helpers import get_tasks
from tau2.task_subsets import (
    allocate,
    build_subset,
    canonical_subset_name,
    check_frame,
    list_subsets,
    load_subset,
    resolve_subset,
)
from tau2.task_subsets.schema import SubsetStrategy


def _task(task_id: str) -> Task:
    return Task(
        id=task_id,
        user_scenario=UserScenario(
            instructions=UserInstructions(task_instructions="t", domain="mock")
        ),
    )


# ---------------------------------------------------------------------------
# Allocation
# ---------------------------------------------------------------------------


def test_allocate_is_proportional():
    alloc = allocate({"a": 50, "b": 30, "c": 20}, 50)
    assert alloc == {"a": 25, "b": 15, "c": 10}
    assert sum(alloc.values()) == 50


def test_allocate_covers_every_nonempty_stratum():
    """The whole reason to stratify: a tiny stratum still gets a slot."""
    alloc = allocate({"big": 995, "tiny": 5}, 50)
    assert alloc["tiny"] >= 1
    assert sum(alloc.values()) == 50


@pytest.mark.parametrize(
    "counts",
    [
        {"a": 2, "b": 100},
        {"a": 1, "b": 1, "c": 1, "d": 200},
        {f"s{i}": i + 1 for i in range(20)},
        {"only": 60},
    ],
)
def test_allocate_stays_inside_every_stratum(counts):
    alloc = allocate(counts, 50)
    assert sum(alloc.values()) == 50
    assert all(alloc[k] <= counts[k] for k in counts)
    assert all(alloc[k] >= 1 for k in counts)


def test_allocate_census_when_budget_exceeds_frame():
    assert allocate({"a": 3, "b": 4}, 50) == {"a": 3, "b": 4}


# ---------------------------------------------------------------------------
# Draw
# ---------------------------------------------------------------------------


def test_build_subset_is_deterministic():
    tasks = get_tasks("telecom", task_split_name="base")
    first = build_subset(
        name="telecom_50", domain="telecom", task_set="telecom", tasks=tasks
    )
    second = build_subset(
        name="telecom_50", domain="telecom", task_set="telecom", tasks=tasks
    )
    assert first.task_ids == second.task_ids


def test_build_subset_seed_changes_the_draw():
    tasks = get_tasks("telecom", task_split_name="base")
    a = build_subset(
        name="t", domain="telecom", task_set="telecom", tasks=tasks, seed=1
    )
    b = build_subset(
        name="t", domain="telecom", task_set="telecom", tasks=tasks, seed=2
    )
    assert a.task_ids != b.task_ids
    # ... but the design is identical: same strata, same allocation.
    assert [(s.key, s.drawn) for s in a.strata] == [(s.key, s.drawn) for s in b.strata]


def test_build_subset_covers_every_telecom_family():
    """The defect that motivated this module: a prefix drops whole families."""
    tasks = get_tasks("telecom", task_split_name="base")
    subset = build_subset(
        name="telecom_50", domain="telecom", task_set="telecom", tasks=tasks
    )
    families = {tid[1 : tid.index("]")] for tid in subset.task_ids}
    assert families == {"mms_issue", "mobile_data_issue", "service_issue"}
    prefix_families = {t.id[1 : t.id.index("]")] for t in tasks[:50]}
    assert "mms_issue" not in prefix_families


def test_build_subset_is_a_census_when_the_frame_is_small():
    tasks = get_tasks("airline", task_split_name="base")
    subset = build_subset(
        name="airline_50", domain="airline", task_set="airline", tasks=tasks
    )
    assert subset.strategy is SubsetStrategy.CENSUS
    assert subset.task_ids == [t.id for t in tasks]


def test_build_subset_keeps_frame_order():
    tasks = get_tasks("retail", task_split_name="base")
    subset = build_subset(
        name="retail_50", domain="retail", task_set="retail", tasks=tasks
    )
    order = {t.id: i for i, t in enumerate(tasks)}
    positions = [order[tid] for tid in subset.task_ids]
    assert positions == sorted(positions)


# ---------------------------------------------------------------------------
# Localized-id resolution
# ---------------------------------------------------------------------------


def test_select_matches_localized_and_identity_variants():
    subset = load_subset("airline_50")
    for task_set in ("airline_es", "airline_es_identity"):
        selected = subset.select(get_tasks(task_set, task_split_name="base"))
        assert len(selected) == subset.size


def test_select_does_not_match_a_different_numeric_id():
    """'1' must not swallow '10_es' — the separator is required."""
    subset = load_subset("airline_50")
    assert subset.matches("1_es") == "1"
    assert subset.matches("10_es") == "10"


def test_select_refuses_a_task_set_missing_pinned_tasks():
    subset = load_subset("airline_50")
    short = get_tasks("airline", task_split_name="base")[:10]
    with pytest.raises(ValueError, match="does not fit this task set"):
        subset.select(short)


# ---------------------------------------------------------------------------
# Resolution rules
# ---------------------------------------------------------------------------


def test_auto_resolves_the_domains_canonical_subset():
    subset = resolve_subset(domain="telecom", requested="auto")
    assert subset is not None
    assert subset.name == canonical_subset_name("telecom")


def test_auto_stands_down_for_an_explicit_task_selection():
    assert (
        resolve_subset(domain="telecom", requested="auto", explicit_task_selection=True)
        is None
    )


def test_all_runs_the_whole_task_set():
    assert resolve_subset(domain="telecom", requested="all") is None
    every = get_tasks("telecom", task_split_name="base", task_subset=None)
    assert len(every) == 114


def test_auto_is_a_no_op_for_a_domain_without_a_subset():
    assert resolve_subset(domain="banking_knowledge", requested="auto") is None


def test_a_named_subset_refuses_an_explicit_task_selection():
    with pytest.raises(ValueError, match="cannot be combined"):
        resolve_subset(
            domain="telecom", requested="telecom_50", explicit_task_selection=True
        )


def test_subsets_are_never_shared_across_domains():
    with pytest.raises(ValueError, match="belongs to domain"):
        resolve_subset(domain="airline", requested="telecom_50")


@pytest.mark.parametrize("requested", ["auto", "telecom_50"])
def test_an_alias_domain_shares_the_pools_subset(requested):
    """telecom-workflow is telecom's task pool under a different policy.

    It registers the same task loader, so it must resolve the same 50 — not
    crash on the cross-domain guard, which is what a plain `subset.domain !=
    domain` check does to every default-flag telecom-workflow run.
    """
    subset = resolve_subset(domain="telecom-workflow", requested=requested)
    assert subset is not None
    assert subset.name == "telecom_50"
    tasks = get_tasks("telecom-workflow", task_split_name="base", task_subset=subset)
    assert [t.id for t in tasks] == subset.task_ids


def test_every_domain_claiming_a_subset_can_resolve_it():
    """No domain may name a subset in the map that it is then refused."""
    for domain in CANONICAL_TASK_SUBSETS:
        assert resolve_subset(domain=domain, requested="auto") is not None


def test_get_tasks_applies_the_subset():
    subset = resolve_subset(domain="telecom", requested="auto")
    tasks = get_tasks("telecom_es", task_split_name="base", task_subset=subset)
    assert [t.id for t in tasks] == [f"{tid}_es" for tid in subset.task_ids]


# ---------------------------------------------------------------------------
# The shipped artifacts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("domain,name", sorted(set(CANONICAL_TASK_SUBSETS.items())))
def test_every_canonical_subset_ships(domain, name):
    subset = load_subset(name)
    assert subset.domain in CANONICAL_TASK_SUBSETS
    assert CANONICAL_TASK_SUBSETS[subset.domain] == name
    assert subset.size == 50


@pytest.mark.parametrize("name", sorted(list_subsets()))
def test_shipped_subset_still_matches_its_frame(name):
    subset = load_subset(name)
    frame = get_tasks(subset.frame.task_set, task_split_name=subset.frame.task_split)
    assert check_frame(subset, [t.id for t in frame]) is None, (
        "The frame changed under a frozen subset. Draw a NEW subset name "
        "rather than editing this one — runs have scored on it."
    )


@pytest.mark.parametrize("name", sorted(list_subsets()))
def test_shipped_subset_resolves_against_every_localized_set(name):
    subset = load_subset(name)
    localized = [
        ts
        for ts in sorted(discover_localized_task_files())
        if ts.startswith(f"{subset.domain}_")
    ]
    for task_set in localized:
        selected = subset.select(get_tasks(task_set, task_split_name="base"))
        assert len(selected) == subset.size, task_set


@pytest.mark.parametrize("name", sorted(list_subsets()))
def test_shipped_subset_is_reproducible_from_its_recorded_design(name):
    """The artifact is not hand-editable: re-running the derivation reproduces it.

    A PREFIX_OF subset is derived, not drawn: reproducing it means re-cutting
    the recorded prefix from its parent, and its ids must be exactly the
    parent's leading ids (that pairing is the reason it exists)."""
    subset = load_subset(name)
    frame = get_tasks(subset.frame.task_set, task_split_name=subset.frame.task_split)
    if subset.strategy is SubsetStrategy.PREFIX_OF:
        from tau2.task_subsets.sampler import build_prefix_subset

        parent = load_subset(subset.derived_from)
        assert subset.task_ids == parent.task_ids[: subset.size]
        recut = build_prefix_subset(
            name=subset.name, parent=parent, tasks=frame, size=subset.size
        )
        assert recut.task_ids == subset.task_ids
        assert recut.strata == subset.strata
        return
    redrawn = build_subset(
        name=subset.name,
        domain=subset.domain,
        task_set=subset.frame.task_set,
        task_split=subset.frame.task_split,
        tasks=frame,
        size=subset.size,
        seed=subset.seed,
        stratifier=None if subset.stratifier == "none" else subset.stratifier,
    )
    assert redrawn.task_ids == subset.task_ids


def test_auto_degrades_when_the_artifact_is_absent(tmp_path, monkeypatch, caplog):
    """An isolated data dir must still run, loudly, not crash."""
    import tau2.utils

    monkeypatch.setattr(tau2.utils, "DATA_DIR", tmp_path)
    assert resolve_subset(domain="telecom", requested="auto") is None


def test_a_named_missing_subset_still_raises(tmp_path, monkeypatch):
    import tau2.utils

    monkeypatch.setattr(tau2.utils, "DATA_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        resolve_subset(domain="telecom", requested="telecom_50")


def test_intersect_ids_is_lenient_where_select_ids_is_strict():
    subset = load_subset("airline_50")
    partial = [f"{tid}_xq" for tid in subset.task_ids[:2]]
    assert subset.intersect_ids(partial) == partial
    with pytest.raises(ValueError, match="does not fit this task set"):
        subset.select_ids(partial)


# ---------------------------------------------------------------------------
# Preset integration
# ---------------------------------------------------------------------------


def test_run_presets_score_on_the_fixed_subset():
    """Presets pass --task-ids, so they must apply the subset at build time."""
    from tau2.multilingual.run_presets import get_run_presets

    presets = get_run_presets()
    assert presets, "no run presets registered"
    for name, preset in presets.items():
        for arm in preset.arms:
            expected = canonical_subset_name(arm.domain)
            if expected is None:
                continue
            assert len(arm.task_ids) == load_subset(expected).size, (
                f"{name}/{arm.name} runs {len(arm.task_ids)} tasks, not the "
                f"{expected} subset"
            )
            assert set(arm.smoke_task_ids) <= set(arm.task_ids)


def test_info_records_what_ran_not_just_what_the_subset_pins():
    """An agent task filter can score fewer tasks than the subset pins.

    `llm_agent_gt`/`llm_agent_solo` need ground-truth actions and drop tasks
    without them, so `size` alone would claim a population the run never
    scored.
    """
    from tau2.data_model.simulation import TextRunConfig
    from tau2.runner.helpers import get_info

    subset = resolve_subset(domain="telecom", requested="auto")
    config = TextRunConfig(domain="telecom", agent="llm_agent", user="user_simulator")
    info = get_info(config, task_subset=subset, tasks_scored=37)
    assert info.task_subset is not None
    assert info.task_subset.size == subset.size
    assert info.task_subset.tasks_scored == 37


def test_explicit_ids_drawn_from_the_subset_still_record_it():
    """`tau2 run-preset` passes --task-ids, and its numbers go in the paper."""
    from tau2.task_subsets import subset_covering_ids

    subset = load_subset("telecom_50")
    localized = [f"{tid}_es" for tid in subset.task_ids]
    covering = subset_covering_ids(domain="telecom", task_ids=localized)
    assert covering is not None
    assert covering.name == subset.name
    # A smoke arm runs one of them: still the same population, one task of it.
    assert (
        subset_covering_ids(domain="telecom", task_ids=localized[:1]).name
        == subset.name
    )


def test_ids_outside_the_subset_are_not_labelled_with_it():
    from tau2.task_subsets import subset_covering_ids

    frame = [t.id for t in get_tasks("telecom", task_split_name="base")]
    subset = load_subset("telecom_50")
    outside = next(tid for tid in frame if subset.matches(tid) is None)
    assert subset_covering_ids(domain="telecom", task_ids=[outside]) is None
    assert (
        subset_covering_ids(domain="telecom", task_ids=[subset.task_ids[0], outside])
        is None
    )
    assert subset_covering_ids(domain="telecom", task_ids=[]) is None
    assert subset_covering_ids(domain="banking_knowledge", task_ids=["0"]) is None


def test_artifacts_are_valid_json_with_the_declared_name():
    from tau2.task_subsets import subset_path

    for name in list_subsets():
        payload = json.loads(subset_path(name).read_text())
        assert payload["name"] == name
