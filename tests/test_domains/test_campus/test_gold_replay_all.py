"""Full gold-replay CI for all 50 campus tasks.

Aligns with upstream #499: the local evaluator (evaluator_env.py) swallows
exceptions raised while replaying golden actions, so a half-executed gold DB
silently becomes the grading target for every agent. This test replays each
task's `evaluation_criteria.actions` on a fresh environment **without any
try/except** — any exception fails the test — and then asserts terminal-state
invariants plus replay determinism:

  ① no orphan SIG/TODO rows: every (ref_type, ref_id) resolves to a row in
     the task's own main DB table;
  ② per-offering counts consistent and non-negative: `enrolled_count` equals
     the number of enrollment rows with status=已选 — caliber verified
     empirically on all 50 task seeds (data_model.py contract: special-
     channel review rows are NOT counted); `waitlist_count`
     equals the number of rows with status=候补中;
  ③ unique active waitlist positions per offering;
  ④ determinism: a second fresh environment replaying the same task yields
     identical get_db_hash()/get_user_db_hash().

Baseline (2026-10-05): gold-action exceptions 0/50; invariants ①③④ pass
50/50; invariant ② was red on 4 task/offering pairs (count bookkeeping drift
in task data / settle special-channel branches), temporarily covered by an
explicit KNOWN_COUNT_GAPS exemption table (exactly four pairs).

All four gaps have since been fixed (authoritative `_recount_enrolled`
in the special-channel settle branches + M03 initial-state count override),
so the exemption table is gone and invariant ② now applies to every
task/offering without exception.
"""

import pytest

from tau2.domains.campus.environment import get_environment, get_tasks

TASK_IDS = [t.id for t in get_tasks(task_split_name=None)]

MAIN_TABLES = (
    "enrollments",
    "deferral_requests",
    "scholarship_apps",
    "certificates",
    "tickets",
)

def _replay(task):
    """Fresh env → task initial state → gold actions (exceptions propagate)."""
    init = task.initial_state
    env = get_environment()
    env.set_state(
        initialization_data=init.initialization_data if init else None,
        initialization_actions=init.initialization_actions if init else None,
        message_history=(init.message_history or []) if init else [],
        strict=True,
    )
    for action in (task.evaluation_criteria.actions if task.evaluation_criteria else []) or []:
        # No try/except on purpose (upstream #499 countermeasure).
        env.make_tool_call(tool_name=action.name, requestor=action.requestor, **action.arguments)
    return env


def _invariant_violations(env):
    db, user_db = env.tools.db, env.tools.user_db
    tables = {name: getattr(db, name) for name in MAIN_TABLES}
    problems = []

    # ① orphan SIG/TODO
    for sig in user_db.pending_signatures.values():
        if sig.ref_type not in tables or sig.ref_id not in tables[sig.ref_type]:
            problems.append(f"orphan SIG {sig.sig_id} -> {sig.ref_type}/{sig.ref_id}")
    for todo in user_db.app_todos.values():
        if todo.ref_type not in tables or todo.ref_id not in tables[todo.ref_type]:
            problems.append(f"orphan TODO {todo.todo_id} -> {todo.ref_type}/{todo.ref_id}")

    for off in db.course_offerings.values():
        rows = [e for e in db.enrollments.values() if e.offering_id == off.offering_id]
        enrolled = sum(1 for e in rows if e.status == "已选")
        waitlisted = sum(1 for e in rows if e.status == "候补中")
        # ② counts consistent and non-negative (caliber: status=已选; special-
        #    channel rows not counted — seed-verified, see module docstring)
        if off.enrolled_count < 0 or off.waitlist_count < 0:
            problems.append(
                f"negative count {off.offering_id}: "
                f"enrolled={off.enrolled_count} waitlist={off.waitlist_count}"
            )
        if off.enrolled_count != enrolled:
            problems.append(
                f"enrolled_count {off.offering_id}: count={off.enrolled_count} rows(已选)={enrolled}"
            )
        if off.waitlist_count != waitlisted:
            problems.append(
                f"waitlist_count {off.offering_id}: count={off.waitlist_count} rows(候补中)={waitlisted}"
            )
        # ③ active waitlist positions unique
        positions = [e.waitlist_position for e in rows if e.status == "候补中"]
        if len(positions) != len(set(positions)):
            problems.append(f"duplicate waitlist positions {off.offering_id}: {positions}")
    return problems


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_gold_replay_all(task_id):
    """Replay every task's gold actions exception-free; assert terminal
    invariants and per-task double-hash determinism."""
    tasks = {t.id: t for t in get_tasks(task_split_name=None)}
    task = tasks[task_id]

    env = _replay(task)

    # ④ determinism: second fresh env, full replay, both hashes equal
    env2 = _replay(task)
    assert env.get_db_hash() == env2.get_db_hash(), f"{task_id}: agent DB hash not deterministic"
    assert env.get_user_db_hash() == env2.get_user_db_hash(), f"{task_id}: user DB hash not deterministic"

    problems = _invariant_violations(env)
    assert not problems, f"{task_id}: terminal-state invariants violated: {problems}"
