"""The retail_faults task set, run on tau2's own fault schedule and graded by tau2's evaluator.

retail_faults comes from the `faults` extra (probity-ratchet-cases), loaded through the
tau2.plugins entry point: one task per retail write tool and timeout kind. A
scripted agent replays each task's gold actions; no model, no user simulator,
no key. Run with `make test-faults`.
"""

import pytest

from tau2.data_model.message import AssistantMessage, Message
from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment
from tau2.evaluator.evaluator_env import EnvironmentEvaluator
from tau2.registry import registry

faults = pytest.importorskip("ratchet_cases.tau2", reason="needs the faults extra")

TASKS: list[Task] = registry.get_tasks_loader(faults.TASK_SET)()
RETAIL = registry.get_env_constructor("retail")


def _run(task: Task, policy) -> tuple[Environment, list[Message]]:
    env = RETAIL()
    env.set_faults(task.faults)
    trajectory: list[Message] = []
    for call, result in faults.play(task, env.get_response, policy):
        trajectory.append(
            AssistantMessage(role="assistant", content=None, tool_calls=[call])
        )
        trajectory.append(result)
    return env, trajectory


def _grade(task: Task, env: Environment, trajectory: list[Message]):
    return EnvironmentEvaluator.calculate_reward(
        environment_constructor=RETAIL,
        task=task,
        full_trajectory=trajectory,
        live_db_hash=env.get_db_hash(),
        live_user_db_hash=env.get_user_db_hash(),
    )


def test_the_task_set_covers_every_retail_write_tool_twice():
    tools = {task.faults[0].tool_name for task in TASKS}
    assert tools == set(faults.write_tools(RETAIL()))
    assert len(TASKS) == 2 * len(tools)


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.id)
def test_an_agent_that_checks_before_retrying_passes(task: Task):
    env, trajectory = _run(task, faults.Policy.RECONCILE)
    reward_info = _grade(task, env, trajectory)

    assert reward_info.db_check.db_match is True
    # The recorded timeout never reproduces on replay, so the replay always
    # disagrees with the live run, and the live database decides the reward.
    assert reward_info.info["replay_divergence"] is True


@pytest.mark.parametrize(
    "task",
    [t for t in TASKS if t.faults[0].kind == "timeout_before_commit"],
    ids=lambda t: t.id,
)
def test_an_agent_that_gives_up_after_a_lost_write_fails(task: Task):
    env, trajectory = _run(task, faults.Policy.GIVE_UP)

    assert _grade(task, env, trajectory).db_check.db_match is False
