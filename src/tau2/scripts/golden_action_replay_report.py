"""Offline report on replaying the reference trajectories of the shipped tasks.

Issue #499 is about the *gold* side of environment evaluation: the reference
trajectory in ``task.evaluation_criteria.actions`` is replayed on a fresh
environment, and until now an action that raised was swallowed with a
``logger.warning`` while the DB verdict was computed anyway.

Every number quoted about that issue in the changelog, the PR description or the
tests should be reproducible by one command. This script is that command. It is
offline (no LLM, no API key, no network), reads nothing but the registered task
sets and environments, and writes nothing unless ``--json`` is given.

It reports, per domain:
  * how many tasks and reference actions it walked, and how many actions raised;
  * each failing action with its id, tool, requestor, exception type and message;
  * whether that failure is *state-mutating* according to
    ``Environment.is_mutating_tool``, the same predicate ``set_state`` uses --
    this is the classification the evaluator now applies;
  * a two-mode gold-hash invariance check: replaying every golden action (what
    the evaluator does) versus skipping the non-mutating ones (the alternative
    fix that was measured and rejected). Only tasks where *both* modes produced a
    hash are compared, and a run with no comparable task reports ``INVALID``
    rather than a clean result -- an empty comparison is not evidence;
  * a cross-check of each affected task against the real evaluator, so the
    ``RewardInfo.info`` the evaluator writes is compared with this independent
    replay rather than taken on trust.

Exit codes: ``0`` clean, ``1`` findings (some reference action raised, or the two
replay modes disagreed on a gold hash), ``2`` nothing measurable (no tasks
loaded, or every task uncomparable).

Usage:
    python -m tau2.scripts.golden_action_replay_report --domains retail
    python -m tau2.scripts.golden_action_replay_report --limit 20
    python -m tau2.scripts.golden_action_replay_report --json report.json

The report goes to stdout; tau2's own log lines go to stderr, so redirecting
stderr leaves the report on its own.
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from loguru import logger

from tau2.data_model.tasks import Action, Task
from tau2.environment.environment import Environment
from tau2.evaluator.evaluator_env import EnvironmentEvaluator
from tau2.registry import registry

#: The four shipped task corpora. ``mock`` is excluded: it has 8 reference
#: actions and is covered by the test suite instead.
DEFAULT_DOMAINS = ("retail", "airline", "telecom", "banking_knowledge")

#: Environments that cannot be constructed without arguments.
DEFAULT_ENV_KWARGS: dict[str, dict[str, Any]] = {
    "banking_knowledge": {"retrieval_variant": "bm25"}
}

MODE_SHIPPED = "replay-all-classify-failures"
MODE_SKIP = "skip-non-mutating"


@dataclass
class TaskReplay:
    """What one task contributed to the report."""

    task_id: str
    n_actions: int
    gold_hashes: dict[str, Optional[str]] = field(default_factory=dict)
    failures: list[dict] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def mutating_failures(self) -> list[dict]:
        return [f for f in self.failures if f["mutating"]]


def _quiet_logs() -> None:
    """Keep the report readable: the replay exercises known-raising tools.

    The evaluator logs each swallowed failure at WARNING/ERROR, which is correct
    for a grading run and pure noise for a corpus walk over a corpus that has
    known-bad references. Loguru is silenced here, so what this script *prints*
    about failures comes from its own replay, not from parsed log lines.
    """
    logger.remove()


def replay_reference(
    environment: Environment,
    golden_actions: list[Action],
    task_id: str,
    *,
    skip_non_mutating: bool = False,
) -> list[dict]:
    """Execute a reference trajectory, returning one record per action that raised.

    Mirrors ``tau2.evaluator.evaluator_env.replay_golden_actions``: same call
    form, same order, nothing filtered out of the executed sequence. Each record
    carries whether ``is_mutating_tool`` classifies that tool as writing to the
    DB, which is the distinction the evaluator now uses to decide whether the
    gold state is certifiable.

    ``skip_non_mutating`` selects the *rejected* alternative (drop read-only
    actions before calling them). It exists only so the two modes can be compared
    against each other in :func:`invariance_verdict`.
    """
    failures = []
    for action in golden_actions:
        mutating = environment.is_mutating_tool(action.name)
        if skip_non_mutating and not mutating:
            continue
        try:
            environment.make_tool_call(
                tool_name=action.name,
                requestor=action.requestor,
                **action.arguments,
            )
        except Exception as e:
            failures.append(
                {
                    "task_id": task_id,
                    "action_id": action.action_id,
                    "tool_name": action.name,
                    "requestor": action.requestor,
                    "mutating": mutating,
                    "error_type": type(e).__name__,
                    "error": str(e),
                }
            )
    return failures


def build_gold_environment(
    constructor: Callable[[], Environment],
    task: Task,
    env_kwargs: Optional[dict],
) -> Environment:
    """The gold environment of ``task``, built the way the evaluator builds it."""
    environment = constructor(**(env_kwargs or {}))
    initial_state = task.initial_state
    environment.set_state(
        initialization_data=(
            initial_state.initialization_data if initial_state is not None else None
        ),
        initialization_actions=(
            initial_state.initialization_actions if initial_state is not None else None
        ),
        message_history=(
            list(initial_state.message_history)
            if initial_state is not None and initial_state.message_history
            else []
        ),
    )
    return environment


def replay_task(
    constructor: Callable[[], Environment],
    task: Task,
    env_kwargs: Optional[dict],
    modes: tuple[str, ...] = (MODE_SHIPPED, MODE_SKIP),
) -> TaskReplay:
    """Replay ``task``'s reference in each mode and record the gold hashes.

    A task that cannot even be set up (bad initialization, unreadable data) is
    recorded as an error and dropped from every comparison; it is never counted as
    agreement.
    """
    golden_actions = (
        list(task.evaluation_criteria.actions or [])
        if task.evaluation_criteria is not None
        else []
    )
    replay = TaskReplay(task_id=str(task.id), n_actions=len(golden_actions))
    for mode in modes:
        try:
            environment = build_gold_environment(constructor, task, env_kwargs)
            failures = replay_reference(
                environment,
                golden_actions,
                str(task.id),
                skip_non_mutating=(mode == MODE_SKIP),
            )
            hashes = environment.get_db_hash(), environment.get_user_db_hash()
        except Exception as e:
            if mode == MODE_SHIPPED:
                replay.error = f"{type(e).__name__}: {e}"
            replay.gold_hashes[mode] = None
            continue
        if mode == MODE_SHIPPED:
            replay.failures = failures
        replay.gold_hashes[mode] = f"{hashes[0]}|{hashes[1]}"
    return replay


def invariance_verdict(replays: list[TaskReplay]) -> dict:
    """Compare the two replay modes over the tasks where *both* produced a hash.

    The rule that matters: a pair is comparable only if neither side is ``None``.
    A domain in which every task failed to load would otherwise have both sides
    equal to the same error sentinel and read as a clean 0-difference result. Zero
    comparable pairs is ``INVALID``, never ``EQUAL``.
    """
    comparable = [
        r
        for r in replays
        if r.gold_hashes.get(MODE_SHIPPED) is not None
        and r.gold_hashes.get(MODE_SKIP) is not None
    ]
    diffs = [
        r.task_id
        for r in comparable
        if r.gold_hashes[MODE_SHIPPED] != r.gold_hashes[MODE_SKIP]
    ]
    return {
        "mode_a": MODE_SHIPPED,
        "mode_b": MODE_SKIP,
        "comparable": len(comparable),
        "dropped": len(replays) - len(comparable),
        "diff_count": len(diffs),
        "diff_sample": diffs[:10],
        "verdict": "INVALID" if not comparable else ("EQUAL" if not diffs else "DIFF"),
    }


def cross_check_evaluator(
    constructor: Callable[[], Environment],
    task: Task,
    replay: TaskReplay,
    env_kwargs: Optional[dict],
) -> dict:
    """Ask the real evaluator about one affected task, with an empty trajectory.

    The point is independence: the replay above decided "this task's gold state
    cannot be built" without calling the evaluator, so this compares that claim
    against ``RewardInfo.info`` as produced.
    """
    reward_info = EnvironmentEvaluator.calculate_reward(
        environment_constructor=constructor,
        task=task,
        full_trajectory=[],
        env_kwargs=env_kwargs,
    )
    info = reward_info.info or {}
    recorded = [f["action_id"] for f in info.get("failed_mutating_actions", [])]
    expected = [f["action_id"] for f in replay.mutating_failures]
    return {
        "task_id": replay.task_id,
        "db_check": (
            None if reward_info.db_check is None else reward_info.db_check.db_match
        ),
        "reward": reward_info.reward,
        "reward_breakdown": {
            str(k): v for k, v in (reward_info.reward_breakdown or {}).items()
        },
        "gold_replay_incomplete": info.get("gold_replay_incomplete"),
        "evaluator_failed_mutating_actions": recorded,
        "replay_agrees": recorded == expected,
    }


def run_domain(
    domain: str,
    *,
    split: Optional[str],
    env_kwargs: Optional[dict],
    limit: Optional[int],
    verify: bool,
) -> dict:
    """Walk one corpus and return its section of the report."""
    constructor = registry.get_env_constructor(domain)
    tasks = registry.get_tasks_loader(domain)(split)
    if limit is not None:
        tasks = tasks[:limit]

    section: dict[str, Any] = {
        "domain": domain,
        "tasks_loaded": len(tasks),
        "tasks_without_reference": 0,
        "actions": 0,
        "failing_actions": 0,
        "failing_read_actions": 0,
        "failing_mutating_actions": 0,
        "affected_tasks": 0,
        "setup_errors": 0,
        "seconds": 0.0,
        "failures": [],
        "invariance": {"verdict": "INVALID", "comparable": 0, "dropped": 0},
        "evaluator_cross_checks": [],
    }
    started = time.time()
    replays: list[TaskReplay] = []
    for task in tasks:
        if task.evaluation_criteria is None or not task.evaluation_criteria.actions:
            section["tasks_without_reference"] += 1
        replay = replay_task(constructor, task, env_kwargs)
        replays.append(replay)
        section["actions"] += replay.n_actions
        if replay.error is not None:
            section["setup_errors"] += 1
            continue
        section["failing_actions"] += len(replay.failures)
        for failure in replay.failures:
            key = (
                "failing_mutating_actions"
                if failure["mutating"]
                else "failing_read_actions"
            )
            section[key] += 1
            section["failures"].append(failure)
        if replay.mutating_failures:
            section["affected_tasks"] += 1
            if verify:
                section["evaluator_cross_checks"].append(
                    cross_check_evaluator(constructor, task, replay, env_kwargs)
                )
    section["seconds"] = round(time.time() - started, 1)
    section["invariance"] = invariance_verdict(replays)
    return section


def _print_section(section: dict) -> None:
    inv = section["invariance"]
    print(f"\n=== domain={section['domain']} ===")
    print(
        f"tasks={section['tasks_loaded']} "
        f"tasks_without_reference={section['tasks_without_reference']} "
        f"actions={section['actions']} "
        f"setup_errors={section['setup_errors']} "
        f"{section['seconds']}s"
    )
    print(
        f"failing_reference_actions={section['failing_actions']} "
        f"(read_only={section['failing_read_actions']} "
        f"mutating={section['failing_mutating_actions']} "
        f"affected_tasks={section['affected_tasks']})"
    )
    print(
        f"gold-hash invariance [{MODE_SHIPPED}] vs [{MODE_SKIP}]: "
        f"comparable={inv['comparable']} dropped={inv['dropped']} "
        f"diffs={inv['diff_count']} verdict={inv['verdict']}"
    )
    for failure in section["failures"]:
        kind = "MUTATING" if failure["mutating"] else "read"
        print(
            f"  task={failure['task_id']} action={failure['action_id']} "
            f"tool={failure['tool_name']} requestor={failure['requestor']} "
            f"{kind} {failure['error_type']}: {failure['error']}"
        )
    for check in section["evaluator_cross_checks"]:
        print(
            f"  evaluator cross-check task={check['task_id']}: "
            f"db_check={check['db_check']} reward={check['reward']} "
            f"gold_replay_incomplete={check['gold_replay_incomplete']} "
            f"replay_agrees={check['replay_agrees']}"
        )


def summarize(sections: list[dict]) -> dict:
    totals = {
        "domains": len(sections),
        "tasks": sum(s["tasks_loaded"] for s in sections),
        "actions": sum(s["actions"] for s in sections),
        "failing_actions": sum(s["failing_actions"] for s in sections),
        "failing_read_actions": sum(s["failing_read_actions"] for s in sections),
        "failing_mutating_actions": sum(
            s["failing_mutating_actions"] for s in sections
        ),
        "affected_tasks": sum(s["affected_tasks"] for s in sections),
        "affected_task_ids": sorted(
            {f["task_id"] for s in sections for f in s["failures"] if f["mutating"]},
            key=lambda x: (len(x), x),
        ),
        "comparable": sum(s["invariance"]["comparable"] for s in sections),
        "dropped": sum(s["invariance"]["dropped"] for s in sections),
        "gold_hash_diffs": sum(s["invariance"]["diff_count"] for s in sections),
        "verdicts": {s["domain"]: s["invariance"]["verdict"] for s in sections},
        "cross_checks_disagreed": [
            c["task_id"]
            for s in sections
            for c in s["evaluator_cross_checks"]
            if not c["replay_agrees"]
        ],
    }
    if totals["tasks"] == 0 or totals["comparable"] == 0:
        totals["status"] = "INVALID"
    elif (
        totals["failing_actions"]
        or totals["gold_hash_diffs"]
        or totals["cross_checks_disagreed"]
        or any(v == "DIFF" for v in totals["verdicts"].values())
    ):
        totals["status"] = "FINDINGS"
    else:
        totals["status"] = "CLEAN"
    return totals


def exit_code(status: str) -> int:
    return {"CLEAN": 0, "FINDINGS": 1, "INVALID": 2}[status]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tau2.scripts.golden_action_replay_report",
        description=(
            "Offline walk of every reference trajectory: which golden actions "
            "raise, which of those are state-mutating, and whether the two "
            "replay modes agree on the resulting gold DB hashes."
        ),
    )
    parser.add_argument(
        "--domains",
        "--domain",
        dest="domains",
        default=",".join(DEFAULT_DOMAINS),
        help=(
            "comma-separated registered task-set names "
            f"(default: {','.join(DEFAULT_DOMAINS)})"
        ),
    )
    parser.add_argument(
        "--split",
        default=None,
        help="task split name, e.g. 'base'. Default: every task in the corpus.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only the first N tasks of each domain (smoke runs)",
    )
    parser.add_argument(
        "--env-kwargs",
        default=json.dumps(DEFAULT_ENV_KWARGS),
        help=(
            "JSON object mapping domain -> constructor arguments. Defaults to "
            f"{json.dumps(DEFAULT_ENV_KWARGS)}, which banking_knowledge needs."
        ),
    )
    parser.add_argument(
        "--no-cross-check",
        action="store_true",
        help="do not re-ask the evaluator about the affected tasks",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="also write the machine-readable report to this path",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="keep tau2's own WARNING/ERROR output (silenced by default here)",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Run the corpus walk. Returns the process exit code."""
    args = build_parser().parse_args(argv)
    if not args.log:
        _quiet_logs()
    env_kwargs_by_domain = json.loads(args.env_kwargs)
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]

    sections = []
    for domain in domains:
        try:
            sections.append(
                run_domain(
                    domain,
                    split=args.split,
                    env_kwargs=env_kwargs_by_domain.get(domain),
                    limit=args.limit,
                    verify=not args.no_cross_check,
                )
            )
        except Exception as e:  # a domain that will not load is a finding
            print(f"domain={domain} FAILED TO RUN: {type(e).__name__}: {e}")
            sections.append(
                {
                    "domain": domain,
                    "tasks_loaded": 0,
                    "tasks_without_reference": 0,
                    "actions": 0,
                    "failing_actions": 0,
                    "failing_read_actions": 0,
                    "failing_mutating_actions": 0,
                    "affected_tasks": 0,
                    "setup_errors": 0,
                    "seconds": 0.0,
                    "failures": [],
                    "invariance": {
                        "verdict": "INVALID",
                        "comparable": 0,
                        "dropped": 0,
                        "diff_count": 0,
                    },
                    "evaluator_cross_checks": [],
                    "error": f"{type(e).__name__}: {e}",
                }
            )

    print("golden action replay report (offline; no LLM, no network)")
    for section in sections:
        _print_section(section)
    totals = summarize(sections)
    print("\n=== totals ===")
    print(json.dumps(totals, indent=2, sort_keys=True))
    if args.json is not None:
        payload = {"totals": totals, "domains": sections}
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"json: {args.json}")
    print(f"status: {totals['status']}")
    return exit_code(totals["status"])


# Entry point. This module deliberately has no ``if __name__ == "__main__"``
# block: script modules under ``tau2.scripts`` are imported by name elsewhere in
# this repo (``src/tau2/cli.py`` does ``from tau2.scripts.<name> import main``),
# so importing this file must stay free of side effects while
# ``python -m tau2.scripts.golden_action_replay_report`` still runs it. Under
# ``-m`` the interpreter puts this module's own path in ``sys.argv[0]``.
if sys.argv and Path(sys.argv[0]).stem == Path(__file__).stem:  # pragma: no cover
    raise SystemExit(main())
