"""Regression tests for replaying the reference trajectory in the evaluators.

Issue #499: both environment evaluators replayed *every* golden action on the
gold environment and swallowed failures with a bare ``logger.warning``, so a
reference trajectory that could not be applied still produced a gold DB hash --
and an agent that did nothing could match it and be certified as a DB pass.

The fix classifies those failures instead of filtering the replay: the calls the
gold environment receives are unchanged (so the gold hash is unchanged, by
construction rather than by measurement), a failure of a *state-mutating* action
is reported and withdraws a DB *match* (a real mismatch keeps its ``0.0``, never
downgraded), and a failure of a read-only action keeps its warning because those
16 corpus-wide failures are themselves evidence about the task data.

Offline: only the ``mock`` and ``retail`` domains, no LLM, no API key. Tasks
that would need the NL judge are called through ``EnvironmentEvaluator``
directly, which never calls an LLM.

Contract tests build their own tasks on the ``mock`` domain, so the behavior stays
pinned even if ``data/`` is fixed underneath it -- the defect this patch repairs
may disappear from the shipped corpus, and a test that dies when that happens
proves nothing about the code. The two retail-corpus cases are written as an
agreement check ("the DB verdict must match whether the reference applied"), which
is green on the current data and green on corrected data, and red on unpatched
code for behavioral reasons only: no name introduced by this patch is used there.
"""

from typing import Callable

import pytest
from loguru import logger

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import (
    Info,
    Results,
    RewardInfo,
    SimulationRun,
    TerminationReason,
    UserInfo,
)
from tau2.data_model.tasks import Action, EvaluationCriteria, RewardType, Task
from tau2.domains.retail.environment import get_environment as get_retail_environment
from tau2.domains.retail.environment import get_tasks as get_retail_tasks
from tau2.environment.environment import Environment, EnvironmentInfo
from tau2.evaluator.evaluator_env import (
    EnvironmentEvaluator,
    FullDuplexEnvironmentEvaluator,
)
from tau2.metrics.agent_metrics import compute_metrics
from tau2.metrics.break_down_metrics import analyze_reward

# Markers of the library's own gold-replay log lines, on either side of the fix.
REPLAY_MARKERS = ("Gold replay", "golden action")


@pytest.fixture
def replay_logs() -> list[dict]:
    """Capture WARNING-and-above records emitted by the evaluator."""
    records: list[dict] = []
    sink_id = logger.add(
        lambda message: records.append(
            {"level": message.record["level"].name, "text": message.record["message"]}
        ),
        level="WARNING",
    )
    yield records
    logger.remove(sink_id)


def replay_log_records(records: list[dict]) -> list[dict]:
    return [r for r in records if any(m in r["text"] for m in REPLAY_MARKERS)]


def _task_with_golden_actions(base_task: Task, actions: list[Action]) -> Task:
    """Same task, but with the given reference trajectory and a DB-only basis."""
    return base_task.model_copy(
        update={
            "evaluation_criteria": EvaluationCriteria(
                actions=actions,
                reward_basis=[RewardType.DB],
            )
        }
    )


def _failing_gold_write(action_id: str = "gold_0") -> Action:
    """A golden WRITE action that cannot be applied.

    ``create_task`` is decorated ``@is_tool(ToolType.WRITE)`` and raises before
    touching the DB when the user is unknown, so the gold environment stays at
    its initial state -- the shape of retail task 105's ``exchange_delivered_order_items``.
    """
    return Action(
        action_id=action_id,
        name="create_task",
        requestor="assistant",
        arguments={"user_id": "no_such_user", "title": "Impossible"},
    )


def _applied_gold_write(action_id: str = "gold_0") -> Action:
    """A golden WRITE action that applies, so the gold DB leaves its initial state.

    Paired with :func:`_failing_gold_write` this reproduces the shape of retail
    task 64 -- one reference write lands, a later one raises -- without depending
    on any shipped task data.
    """
    return Action(
        action_id=action_id,
        name="create_task",
        requestor="assistant",
        arguments={"user_id": "user_1", "title": "Important Meeting"},
    )


def _synthetic_task(base_task: Task, *actions: Action) -> Task:
    return _task_with_golden_actions(base_task, list(actions))


def _recovered_write_trajectory() -> list[Message]:
    """A predicted trajectory that does write to the DB (valid ``create_task``)."""
    return [
        UserMessage(id="1", role="user", content="Create a task for user_1."),
        AssistantMessage(
            id="2",
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(
                    id="write",
                    name="create_task",
                    arguments={"user_id": "user_1", "title": "Important Meeting"},
                )
            ],
        ),
        ToolMessage(
            id="write",
            role="tool",
            content='{"task_id": "task_2", "title": "Important Meeting", "description": null, "status": "pending"}',
        ),
    ]


def _retail_task(task_id: str) -> Task:
    for task in get_retail_tasks("base"):
        if task.id == task_id:
            return task
    raise AssertionError(f"retail task {task_id} not found")


def _gold_hash_of(constructor: Callable[[], Environment], task: Task) -> str:
    """The gold DB hash the evaluator computes for ``task`` with an empty trajectory.

    ``calculate_reward`` builds the predicted environment with ``solo_mode=...``
    and the gold environment without it, so shadowing ``get_db_hash`` on that
    second instance records the hash the DB check is computed from.
    """
    recorded = []

    def construct(**kwargs):
        env = constructor(**kwargs)
        if "solo_mode" in kwargs:
            return env
        original = env.get_db_hash
        env.get_db_hash = lambda: recorded.append(original()) or recorded[-1]
        return env

    EnvironmentEvaluator.calculate_reward(
        environment_constructor=construct,
        task=task,
        full_trajectory=[],
    )
    assert recorded, "gold environment DB hash was never read"
    return recorded[0]


def _initial_hash(constructor: Callable[[], Environment], task: Task) -> str:
    """DB hash of a fresh environment with the task's setup but no reference replay."""
    environment = constructor()
    initial_state = task.initial_state
    environment.set_state(
        initialization_data=initial_state.initialization_data
        if initial_state
        else None,
        initialization_actions=initial_state.initialization_actions
        if initial_state
        else None,
        message_history=list(initial_state.message_history)
        if initial_state and initial_state.message_history
        else [],
    )
    return environment.get_db_hash()


# --------------------------------------------------------------------------
# Layer 1: the replay is not filtered; a read-only failure is classified, not
# swallowed and not allowed to invalidate anything
# --------------------------------------------------------------------------


def test_read_only_reference_keeps_the_verdict_it_already_had():
    """Retail task 67's reference is five read-only calls, two of which raise.

    Reads cannot move the DB hash (``docs/evaluation.md``), so a failure among
    them must not invalidate the DB check. This case deliberately asserts only
    what is true *both* before and after this patch -- the gold hash, the DB
    check and the total reward -- so it is a guard rather than a repro: any
    variant that over-retracts (for instance one that treats *any* golden-action
    failure as "gold unusable", which is what issue #499 tempts a reader into)
    turns it red, on either tree. The premise that 67 has no state-mutating
    reference action is observed through the gold hash equaling the fresh
    database's hash, not through any tool-metadata accessor.
    """
    task = _retail_task("67")

    assert _gold_hash_of(get_retail_environment, task) == _initial_hash(
        get_retail_environment, task
    ), "a reference of pure reads must leave the gold DB at its initial state"

    reward_info = EnvironmentEvaluator.calculate_reward(
        environment_constructor=get_retail_environment,
        task=task,
        full_trajectory=[],
    )
    assert reward_info.db_check is not None
    assert reward_info.db_check.db_match
    assert reward_info.db_check.db_reward == 1.0
    assert reward_info.reward == 1.0
    assert reward_info.info is None, "a read failure must not invalidate the DB check"


def test_read_only_reference_failures_stay_visible_and_attributable(
    replay_logs: list[dict],
):
    """The same two read failures must still be reported, and name the action.

    Patch-visible half of the retail-67 pair: before this patch the library's
    line was ``Error in golden actions {name}({arguments}): {e}`` -- no task id,
    no action id -- which is the "silently" issue #499 is titled around.
    """
    task = _retail_task("67")

    assert _gold_hash_of(get_retail_environment, task) == _initial_hash(
        get_retail_environment, task
    )

    records = replay_log_records(replay_logs)
    # The one replay this case performs raises twice (67_0 and 67_1).
    assert len(records) == 2
    assert [r["level"] for r in records] == ["WARNING"] * len(records)
    for action_id in ("67_0", "67_1"):
        assert any(action_id in r["text"] for r in records), (
            f"failing read {action_id} must still be reported, and be attributable"
        )


def test_failing_read_golden_action_is_executed_and_leaves_the_verdict_intact(
    replay_logs: list[dict],
):
    """Classifying a failure is not the same as dropping the call that made it.

    ``make_tool_call`` is recorded on the gold environment, so the executed
    sequence is checked rather than assumed. The rejected alternative -- skipping
    non-mutating golden actions the way ``Environment.set_state`` skips them on
    the predicted side -- would remove ``67_0``/``67_1`` from that sequence; it
    changes no gold hash anywhere in the shipped corpora, so it buys no
    correctness, and it silences 16 read-only failures that are themselves
    evidence that these references name users, orders and products the DB lacks.
    """
    task = _retail_task("67")
    golden_actions = task.evaluation_criteria.actions
    gold_calls: list[str] = []

    def construct(**kwargs):
        # The evaluator builds the predicted environment with ``solo_mode=`` and
        # the gold one without it (as in ``_gold_hash_of``); only the gold side
        # is instrumented.
        environment = get_retail_environment(**kwargs)
        if "solo_mode" in kwargs:
            return environment
        original = environment.make_tool_call

        def record(*args, **named):
            gold_calls.append(
                named.get("tool_name") if "tool_name" in named else args[0]
            )
            return original(*args, **named)

        environment.make_tool_call = record
        return environment

    reward_info = EnvironmentEvaluator.calculate_reward(
        environment_constructor=construct,
        task=task,
        full_trajectory=[],
    )

    assert gold_calls[-len(golden_actions) :] == [
        action.name for action in golden_actions
    ], "every golden action must reach make_tool_call, in reference order"
    assert reward_info.db_check is not None
    assert reward_info.db_check.db_match
    assert reward_info.reward == 1.0
    assert reward_info.info is None
    assert [r["level"] for r in replay_log_records(replay_logs)] == [
        "WARNING",
        "WARNING",
    ]


def test_reference_with_failed_write_keeps_its_mismatch_and_says_why(
    get_environment: Callable[[], Environment],
    base_task: Task,
    replay_logs: list[dict],
):
    """A golden write raises but an earlier one lands, so the gold state is not stuck.

    Synthetic (mock domain, no shipped task data): ``gold_0`` applies and moves the
    gold DB, ``gold_1`` raises. The task therefore keeps the verdict it already
    reported -- a mismatch for the empty trajectory -- instead of being re-labelled.
    But the mismatch is scored against a gold state whose reference never fully
    applied, so it carries the same provenance as a withdrawn match: a ``0.0`` that
    cannot be traced is how a data defect gets read as a model failure. This is the
    shape retail task 64 had; see
    :func:`test_shipped_retail_verdict_agrees_with_whether_the_reference_applied`
    for the corpus-side check, which stays green whichever way the data goes.
    """
    task = _synthetic_task(
        base_task, _applied_gold_write(), _failing_gold_write("gold_1")
    )
    assert _gold_hash_of(get_environment, task) != _initial_hash(get_environment, task)

    reward_info = EnvironmentEvaluator.calculate_reward(
        environment_constructor=get_environment,
        task=task,
        full_trajectory=[],
    )
    assert reward_info.reward == 0.0
    assert reward_info.db_check is not None
    assert not reward_info.db_check.db_match
    assert reward_info.db_check.db_reward == 0.0

    info = reward_info.info
    assert info is not None, (
        "a mismatch against an incomplete gold state must be traceable"
    )
    assert info["gold_replay_incomplete"] is True
    assert info["db_evaluated"] is True
    failed = info["failed_mutating_actions"]
    assert [f["action_id"] for f in failed] == ["gold_1"]
    assert failed[0]["tool_name"] == "create_task"
    assert failed[0]["error_type"] == "ValueError"
    assert [r["level"] for r in replay_log_records(replay_logs)] == [
        "ERROR",
        "WARNING",
    ] * 2


# --------------------------------------------------------------------------
# Layer 2: an incomplete gold state cannot be certified as a match
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "evaluator", [EnvironmentEvaluator, FullDuplexEnvironmentEvaluator]
)
def test_unappliable_gold_write_is_not_reported_as_db_pass(
    evaluator,
    get_environment: Callable[[], Environment],
    base_task: Task,
):
    """#499's core claim, on a synthetic task: the only golden action is a write
    that raises, so the gold DB is the untouched initial state and every
    non-writing trajectory matches it.

    The DB check must then report "not evaluated" (``db_check=None``, counted in
    ``db_not_checked`` by ``metrics/agent_metrics.py``) rather than a pass. Built
    on the mock domain so the contract does not travel with the corpus: when the
    retail data defect is fixed upstream this test still proves the same thing.
    """
    task = _synthetic_task(base_task, _failing_gold_write())

    reward_info = evaluator.calculate_reward(
        environment_constructor=get_environment,
        task=task,
        full_trajectory=[],
    )

    assert reward_info.db_check is None
    assert RewardType.DB not in (reward_info.reward_breakdown or {})
    info = reward_info.info or {}
    assert info["gold_replay_incomplete"] is True
    failed = info["failed_mutating_actions"]
    assert [f["action_id"] for f in failed] == ["gold_0"]
    assert failed[0]["tool_name"] == "create_task"
    assert failed[0]["error_type"] == "ValueError"


@pytest.mark.parametrize(
    "evaluator", [EnvironmentEvaluator, FullDuplexEnvironmentEvaluator]
)
def test_gold_replay_failure_is_logged_with_task_and_action(
    evaluator,
    get_environment: Callable[[], Environment],
    base_task: Task,
    replay_logs: list[dict],
):
    """The swallowed exception must be attributable without re-running the eval."""
    task = _synthetic_task(base_task, _failing_gold_write())
    evaluator.calculate_reward(
        environment_constructor=get_environment,
        task=task,
        full_trajectory=[],
    )

    records = replay_log_records(replay_logs)
    assert [r["level"] for r in records] == ["ERROR", "ERROR"]
    action_line, verdict_line = (r["text"] for r in records)
    for expected in (task.id, "gold_0", "create_task", "ValueError"):
        assert expected in action_line
    assert "db_check=None" in verdict_line
    assert "db_reward=1.0" in verdict_line


def test_incomplete_gold_state_still_scores_write_mismatch_as_zero(
    get_environment: Callable[[], Environment], base_task: Task
):
    """Reporting "not evaluated" must never hand out a better verdict.

    Gold: one write that raises, so gold stays at the initial DB. Predicted: a
    trajectory that does write. The hashes differ, so the DB check still reports
    the mismatch with its 0.0 -- it is not downgraded to ``db_check=None``.
    """
    task = _task_with_golden_actions(base_task, [_failing_gold_write()])

    reward_info = EnvironmentEvaluator.calculate_reward(
        environment_constructor=get_environment,
        task=task,
        full_trajectory=_recovered_write_trajectory(),
    )

    assert reward_info.reward == 0.0
    assert reward_info.db_check is not None
    assert not reward_info.db_check.db_match
    assert reward_info.db_check.db_reward == 0.0


def test_mock_gold_write_failure_removes_the_db_pass(
    get_environment: Callable[[], Environment], base_task: Task
):
    """Minimal repro of #499 on the mock domain, without shipped task data.

    The empty trajectory writes nothing, so it matches the stuck gold state and
    used to be certified as a DB pass with a full reward.
    """
    task = _task_with_golden_actions(base_task, [_failing_gold_write()])

    reward_info = EnvironmentEvaluator.calculate_reward(
        environment_constructor=get_environment,
        task=task,
        full_trajectory=[],
    )

    assert reward_info.db_check is None
    assert RewardType.DB not in (reward_info.reward_breakdown or {})
    assert reward_info.info["failed_mutating_actions"][0]["action_id"] == "gold_0"


def test_unknown_golden_tool_name_is_not_certified_as_db_pass(
    get_environment: Callable[[], Environment], base_task: Task
):
    """A reference naming a tool the domain does not have cannot be applied.

    ``Environment.is_mutating_tool`` falls back to ``True`` for unknown names, so
    this is treated like any other unappliable write rather than the no-op that
    ``set_state`` applies to a *hallucinated* agent call.
    """
    task = _task_with_golden_actions(
        base_task,
        [
            Action(
                action_id="gold_missing",
                name="this_tool_does_not_exist",
                requestor="assistant",
                arguments={},
            )
        ],
    )

    reward_info = EnvironmentEvaluator.calculate_reward(
        environment_constructor=get_environment,
        task=task,
        full_trajectory=[],
    )

    assert reward_info.db_check is None
    assert reward_info.info["failed_mutating_actions"][0]["action_id"] == (
        "gold_missing"
    )


def test_applyable_reference_still_reports_db_check(
    get_environment: Callable[[], Environment], base_task: Task
):
    """Guard: the common case is untouched -- a reference that replays cleanly
    keeps reporting a real DB check for a matching trajectory."""
    reward_info = EnvironmentEvaluator.calculate_reward(
        environment_constructor=get_environment,
        task=base_task,
        full_trajectory=_recovered_write_trajectory(),
    )

    assert reward_info.db_check is not None
    assert reward_info.db_check.db_match
    assert reward_info.reward == 1.0
    assert reward_info.reward_breakdown[RewardType.DB] == 1.0
    assert reward_info.info is None


# --------------------------------------------------------------------------
# Corpus-side supplement: what the SHIPPED retail data does today. Written as an
# agreement check, not as "105 must be withdrawn", so it stays green whichever way
# the data goes and only ever reports the pairing between data and verdict.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("task_id", ["105", "64"])
def test_shipped_retail_verdict_agrees_with_whether_the_reference_applied(task_id: str):
    """The withdrawal must track the reference, not a task number -- and only ever
    withdraw a match, never a mismatch.

    Classifies each golden action of the shipped task independently of the
    evaluator (replay, catch, ask the toolkits' own ``tool_mutates_state``
    metadata, the same source ``set_state`` uses), then checks the evaluator's DB
    verdict against it. Today's corpus has 105 with an unappliable write whose
    gold state never left the initial hash, and 64 whose failing write is followed
    by a successful one; once either is fixed in ``data/`` this test moves with it
    instead of failing, while the contract itself lives in the synthetic cases
    above. No name from this PR is used here, so the case is red on unpatched code
    because of behavior, not because of a missing attribute.
    """
    task = _retail_task(task_id)
    environment = get_retail_environment()
    initial_state = task.initial_state
    environment.set_state(
        initialization_data=initial_state.initialization_data
        if initial_state
        else None,
        initialization_actions=initial_state.initialization_actions
        if initial_state
        else None,
        message_history=list(initial_state.message_history)
        if initial_state and initial_state.message_history
        else [],
    )
    initial_hash = environment.get_db_hash()

    def mutates(name: str) -> bool:
        for toolkit in (environment.tools, environment.user_tools):
            if toolkit is not None and toolkit.has_tool(name):
                return toolkit.tool_mutates_state(name)
        return True

    raised = []
    for action in task.evaluation_criteria.actions or []:
        try:
            environment.make_tool_call(
                tool_name=action.name, requestor=action.requestor, **action.arguments
            )
        except Exception:  # noqa: BLE001 - classify here, let the patch decide
            if mutates(action.name):
                raised.append(action.action_id)
    incomplete = bool(raised)
    would_match = environment.get_db_hash() == initial_hash

    reward_info = EnvironmentEvaluator.calculate_reward(
        environment_constructor=get_retail_environment,
        task=task,
        full_trajectory=[],
    )
    if incomplete and would_match:
        assert reward_info.db_check is None, (
            f"task {task_id}: mutating golden actions {raised} raised, the gold state "
            "never left its initial hash, and an empty trajectory is still certified a "
            "DB match"
        )
        assert RewardType.DB not in (reward_info.reward_breakdown or {})
        assert [
            f["action_id"] for f in (reward_info.info or {})["failed_mutating_actions"]
        ] == raised
    elif incomplete:
        assert reward_info.db_check is not None, (
            f"task {task_id}: a real mismatch must not be downgraded to not-evaluated"
        )
        assert not reward_info.db_check.db_match
        assert reward_info.db_check.db_reward == 0.0
        assert reward_info.reward == 0.0
        assert (reward_info.info or {})["gold_replay_incomplete"] is True
    else:
        assert reward_info.db_check is not None, (
            f"task {task_id}: its reference replays cleanly, so there is nothing to withdraw"
        )
        assert reward_info.info is None or not (reward_info.info or {}).get(
            "gold_replay_incomplete"
        )


# --------------------------------------------------------------------------
# Downstream consumers: a withheld DB component must not break the metrics
# --------------------------------------------------------------------------


def _withheld_db_reward_info(
    get_environment: Callable[[], Environment], base_task: Task
) -> RewardInfo:
    """Synthetic failing write + an empty trajectory: DB in the basis, none in the breakdown.

    This is the state layer 2 introduces, and the state ``analyze_reward`` used
    to raise ``KeyError`` on. Built through the public evaluator API, not by
    hand-assembling a ``RewardInfo``, so the test tracks the real producer -- and
    not the shipped retail data, which is a moving target (PR-DATA fixes task 105).
    """
    task = _synthetic_task(base_task, _failing_gold_write())
    reward_info = EnvironmentEvaluator.calculate_reward(
        environment_constructor=get_environment,
        task=task,
        full_trajectory=[],
    )
    assert reward_info.db_check is None
    assert RewardType.DB in reward_info.reward_basis
    assert RewardType.DB not in reward_info.reward_breakdown
    return task, reward_info


def test_analyze_reward_survives_a_withheld_db_component(
    get_environment: Callable[[], Environment], base_task: Task
):
    """``break_down_metrics.analyze_reward`` subscripted the breakdown directly.

    ``{}`` is not ``None``, so the existing guard fell through to
    ``reward_breakdown[RewardType.DB]``; the surrounding ``except`` prints and
    re-raises, so rescoring any result set containing this task aborted with
    ``KeyError: <RewardType.DB: 'DB'>``. The crash guard makes a missing
    component fall back to the value the ``None`` branch already produced (0) --
    no third state enters the report.
    """
    _, reward_info = _withheld_db_reward_info(get_environment, base_task)

    # No exception may escape: pytest fails on a raise, so this call is the guard.
    assert analyze_reward(reward_info, set(), set()) == {
        "success": True,
        "communication": None,  # COMMUNICATE is not in this task's reward_basis
        "environment": None,  # ENV_ASSERTION is not in this task's reward_basis
        "database": 0,
        "num_correct_write_action": 0,
        "num_write_action": 0,
    }


def test_analyze_reward_still_reports_db_success_when_the_component_is_present():
    """Anti-vacuous control: the guard must not change any observable value.

    Retail 67's reference replays cleanly, so its breakdown still carries the DB
    component and ``analyze_reward`` reports exactly what it reported before.
    """
    reward_info = EnvironmentEvaluator.calculate_reward(
        environment_constructor=get_retail_environment,
        task=_retail_task("67"),
        full_trajectory=[],
    )
    assert reward_info.reward_breakdown[RewardType.DB] == 1.0

    analysis = analyze_reward(reward_info, set(), set())
    assert analysis["database"] is True
    assert analysis["success"] is True


def test_agent_metrics_counts_the_same_reward_info_as_db_not_checked(
    get_environment: Callable[[], Environment], base_task: Task
):
    """The two consumers of ``db_check=None`` disagree, and neither moves reward.

    ``agent_metrics`` reads ``reward_info.db_check`` (never the breakdown), so
    the simulation lands in ``db_not_checked``; ``analyze_reward`` has no
    "not evaluated" state and reports ``database=0``. Recorded here so the
    disagreement stated in the PR body is the tested behavior, not a hope.
    """
    task, reward_info = _withheld_db_reward_info(get_environment, base_task)
    results = Results(
        info=Info(
            git_commit="abc123",
            num_trials=1,
            max_steps=100,
            max_errors=10,
            user_info=UserInfo(implementation="dummy_user"),
            agent_info={"implementation": "llm_agent"},
            environment_info=EnvironmentInfo(domain_name="mock", policy="test policy"),
        ),
        tasks=[task],
        simulations=[
            SimulationRun(
                id="sim-synthetic-db-not-checked",
                task_id=task.id,
                start_time="2026-01-01T00:00:00",
                end_time="2026-01-01T00:01:00",
                duration=60.0,
                termination_reason=TerminationReason.AGENT_STOP,
                messages=[],
                reward_info=reward_info,
            )
        ],
    )

    metrics = compute_metrics(results)
    assert metrics.db_not_checked == 1
    assert metrics.db_match_count == 0
    assert metrics.db_mismatch_count == 0
    assert metrics.avg_reward == reward_info.reward == 1.0
    assert analyze_reward(reward_info, set(), set())["database"] == 0
