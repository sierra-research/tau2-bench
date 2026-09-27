"""Regression tests for ``metrics.break_down_metrics.analyze_reward``.

``analyze_reward()`` builds one report row per simulation and reads each reward
component that the simulation's ``reward_basis`` names straight out of
``reward_breakdown`` by subscript, guarded only by ``if reward_breakdown is not
None``. A ``RewardInfo`` whose basis names a component the breakdown does not
carry is therefore not a value it can report -- it is a ``KeyError`` re-raised
out of the surrounding ``except Exception``, which aborts the whole analysis run.

That shape is not hypothetical and does not require hand-assembling anything: two
public entry points produce it on shipped task data today (see the two natural
cases below). A missing component now reports the same ``0`` that the
``reward_breakdown is None`` branch already produced, so no new state enters the
report and no column is renamed or redefined.

Offline: ``airline`` and ``retail`` task data plus hand-built ``RewardInfo``
values. No LLM, no API key, no network.
"""

import pytest

from tau2.data_model.simulation import (
    RewardInfo,
    SimulationRun,
    TerminationReason,
)
from tau2.data_model.tasks import RewardType
from tau2.domains.airline.environment import get_environment as get_airline_environment
from tau2.domains.airline.environment import get_tasks as get_airline_tasks
from tau2.domains.retail.environment import get_tasks as get_retail_tasks
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.evaluator.evaluator_env import EnvironmentEvaluator
from tau2.metrics.break_down_metrics import analyze_reward
from tau2.orchestrator.modes import CommunicationMode


def _analyze(reward_info: RewardInfo) -> dict:
    """Call the consumer under test the way its own caller does."""
    return analyze_reward(reward_info, set(), set())


# --------------------------------------------------------------------------
# Natural producers: no hand-assembled RewardInfo involved
# --------------------------------------------------------------------------


def _airline_env_only_reward_info() -> RewardInfo:
    """The environment evaluator echoes the task's full basis but scores two components.

    ``airline`` tasks default to ``[DB, COMMUNICATE]`` (``docs/evaluation.md``),
    while ``EnvironmentEvaluator`` only ever fills in DB / ENV_ASSERTION and then
    returns ``reward_basis=task.evaluation_criteria.reward_basis``. So its own
    result already names a component its breakdown does not carry -- no patch
    needed to reach that state, and no test fixture inventing it.
    """
    task = next(
        t
        for t in get_airline_tasks("base")
        if RewardType.COMMUNICATE in (t.evaluation_criteria.reward_basis or [])
    )
    reward_info = EnvironmentEvaluator.calculate_reward(
        environment_constructor=get_airline_environment,
        task=task,
        full_trajectory=[],
    )
    assert RewardType.COMMUNICATE in reward_info.reward_basis
    assert RewardType.COMMUNICATE not in reward_info.reward_breakdown
    return reward_info


def test_analyze_reward_survives_a_communicate_component_nobody_scored():
    """``KeyError: <RewardType.COMMUNICATE>`` used to abort the row."""
    reward_info = _airline_env_only_reward_info()

    analysis = _analyze(reward_info)
    assert analysis["communication"] == 0


def _all_ignore_basis_reward_info() -> RewardInfo:
    """``ALL_IGNORE_BASIS`` hard-codes a basis and fills the breakdown conditionally.

    ``evaluate_simulation``'s ``ALL_IGNORE_BASIS`` branch sets
    ``reward_basis=[DB, ENV_ASSERTION, ACTION, COMMUNICATE]`` for every
    simulation, while ``reward_breakdown`` receives ENV_ASSERTION only if the
    *task's own* basis had it -- which no airline, retail or telecom task does.
    So the analyser is handed a basis naming a component its breakdown cannot
    show, through the public entry point and shipped data.
    """
    task = next(t for t in get_retail_tasks("base") if t.id == "67")
    sim = SimulationRun(
        id="sim-67-all-ignore-basis",
        task_id=task.id,
        start_time="2026-01-01T00:00:00",
        end_time="2026-01-01T00:01:00",
        duration=60.0,
        termination_reason=TerminationReason.USER_STOP,
        messages=[],
    )
    reward_info = evaluate_simulation(
        simulation=sim,
        task=task,
        evaluation_type=EvaluationType.ALL_IGNORE_BASIS,
        solo_mode=False,
        domain="retail",
        mode=CommunicationMode.HALF_DUPLEX,
    )
    assert RewardType.ENV_ASSERTION in reward_info.reward_basis
    assert RewardType.ENV_ASSERTION not in reward_info.reward_breakdown
    assert reward_info.info
    return reward_info


def test_analyze_reward_survives_the_all_ignore_basis_breakdown_gap():
    reward_info = _all_ignore_basis_reward_info()

    analysis = _analyze(reward_info)
    assert analysis["environment"] == 0


# --------------------------------------------------------------------------
# Hand-built shapes, and the control that keeps the guards honest
# --------------------------------------------------------------------------


def test_analyze_reward_survives_two_missing_components_at_once():
    """A reader-visible ``info`` plus two named-but-absent components.

    Hand-built on purpose: it pins that the two guards are independent of each
    other (the first one raised, so the second was never reached) and that
    having something in ``info`` -- the field a caller would use to explain why
    a component is missing -- does not change how the columns are filled.
    """
    reward_info = RewardInfo(
        reward=1.0,
        reward_basis=[
            RewardType.COMMUNICATE,
            RewardType.ENV_ASSERTION,
            RewardType.DB,
        ],
        reward_breakdown={RewardType.DB: 1.0},
        info={"component_note": "communicate and env assertion were not scored"},
    )
    assert reward_info.info, "premise: the record is reader-visible"

    analysis = _analyze(reward_info)
    assert analysis["communication"] == 0
    assert analysis["environment"] == 0
    assert analysis["database"] is True


@pytest.mark.parametrize(
    "values",
    [
        # every named component present: the dict must be exactly what the
        # subscripts produced before any guard existed
        {
            RewardType.COMMUNICATE: 1.0,
            RewardType.ENV_ASSERTION: 1.0,
            RewardType.DB: 1.0,
        },
        {
            RewardType.COMMUNICATE: 0.0,
            RewardType.ENV_ASSERTION: 1.0,
            RewardType.DB: 0.0,
        },
    ],
)
def test_analyze_reward_reports_present_components_exactly_as_before(values):
    """Anti-vacuous control: the guards may only cover the missing-key path.

    This case passes with and without the patch -- that is the point. Without it
    a "guard" that silently turned a present 1.0 into 0 would still make the two
    missing-component cases pass.
    """
    reward_info = RewardInfo(
        reward=1.0,
        reward_basis=[
            RewardType.COMMUNICATE,
            RewardType.ENV_ASSERTION,
            RewardType.DB,
        ],
        reward_breakdown=values,
    )

    analysis = _analyze(reward_info)
    assert analysis["communication"] is (values[RewardType.COMMUNICATE] == 1.0)
    assert analysis["environment"] is (values[RewardType.ENV_ASSERTION] == 1.0)
    assert analysis["database"] is (values[RewardType.DB] == 1.0)
    # "success" is derived from reward_info.reward, never from the components.
    assert analysis["success"] is True
