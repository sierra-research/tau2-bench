"""End-to-end text run of a canonical task through the real orchestrator.

An oracle-style scripted agent and callee play the fixed v4 call shape
(design doc §3) turn by turn — opener, "Hello?", purpose statement, per-field
ask -> get_entity -> read-back -> confirm, one submit_fields, thanks — and the
finished simulation scores reward 1.0 through the environment evaluator. The
script is built from the task's own frozen values, so it holds for whichever
task the canonical set puts first; the task-supplied deterministic opener is
asserted on the trajectory itself.
"""

from collections import deque
from copy import deepcopy
from typing import Optional

import pytest

from tau2.agent.base_agent import HalfDuplexAgent
from tau2.data_model.message import AssistantMessage, Message, ToolCall, UserMessage
from tau2.data_model.simulation import TerminationReason
from tau2.data_model.tasks import Task
from tau2.domains.intake.environment import get_environment, get_tasks
from tau2.evaluator.evaluator_env import EnvironmentEvaluator
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.user.user_simulator_base import STOP, HalfDuplexUser
from tau2.utils.utils import get_now


class ScriptedAgent(HalfDuplexAgent[None]):
    """Plays a fixed sequence of assistant messages, whatever it receives."""

    def __init__(self, script: list[AssistantMessage]):
        super().__init__(tools=[], domain_policy="")
        self._script = deque(script)

    def generate_next_message(self, message, state):
        next_message = deepcopy(self._script.popleft())
        next_message.timestamp = get_now()
        return next_message, state

    def get_init_state(self, message_history: Optional[list[Message]] = None):
        return None

    def is_stop(self, message: AssistantMessage) -> bool:
        return False

    def set_seed(self, seed: int) -> None:
        pass


class ScriptedUser(HalfDuplexUser[None]):
    """Plays a fixed sequence of user messages, whatever it receives."""

    def __init__(self, script: list[UserMessage]):
        super().__init__(instructions=None, tools=None)
        self._script = deque(script)

    def generate_next_message(self, message, state):
        next_message = deepcopy(self._script.popleft())
        next_message.timestamp = get_now()
        return next_message, state

    def get_init_state(self, message_history: Optional[list[Message]] = None):
        return None

    def set_seed(self, seed: int) -> None:
        pass


def agent_says(content: str) -> AssistantMessage:
    return AssistantMessage(role="assistant", content=content, cost=0.0)


def agent_calls(name: str, **arguments) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=None,
        tool_calls=[
            ToolCall(
                id=f"a_{name}", name=name, arguments=arguments, requestor="assistant"
            )
        ],
        cost=0.0,
    )


def user_says(content: str) -> UserMessage:
    return UserMessage(role="user", content=content, cost=0.0)


def user_calls(name: str, **arguments) -> UserMessage:
    return UserMessage(
        role="user",
        content=None,
        tool_calls=[
            ToolCall(id=f"u_{name}", name=name, arguments=arguments, requestor="user")
        ],
        cost=0.0,
    )


@pytest.fixture
def task() -> Task:
    return get_tasks()[0]


def _task_facts(task: Task) -> dict:
    """The frozen values the oracle script speaks and submits."""
    init = task.initial_state.initialization_actions
    record = next(a for a in init if a.func_name == "seed_record").arguments["record"]
    entities = next(a for a in init if a.func_name == "set_entities").arguments[
        "entities"
    ]
    submit = next(
        a for a in task.evaluation_criteria.actions if a.name == "submit_fields"
    )
    (field_name, spoken_value) = next(iter(entities.items()))
    return {
        "record_id": submit.arguments["record_id"],
        "callee": record["callee_full_name"],
        "field_name": field_name,
        "spoken_value": spoken_value,
        "submission": dict(submit.arguments["fields"]),
    }


def test_e2e_text_run_scores_full_reward(task):
    facts = _task_facts(task)
    agent = ScriptedAgent(
        [
            agent_calls("get_callback_order"),
            agent_says(
                f"Hi, am I speaking with {facts['callee']}? I'm calling about "
                f"the form you recently submitted — the {facts['field_name']} "
                "is missing. Could you give it to me?"
            ),
            agent_calls(
                "log_capture",
                field_name=facts["field_name"],
                value=next(iter(facts["submission"].values())),
            ),
            agent_says(
                f"Let me read that back: {facts['spoken_value']} — is that correct?"
            ),
            agent_calls(
                "submit_fields",
                record_id=facts["record_id"],
                fields=facts["submission"],
                confirmed_with_user=True,
            ),
            agent_says("Thanks, that's all I need. Have a great day!"),
        ]
    )
    user = ScriptedUser(
        [
            user_says("Hello?"),
            user_calls("get_entity", field=facts["field_name"]),
            user_says(f"Yes, that's me — sure. It's {facts['spoken_value']}."),
            user_says("Yes, that's right."),
            user_says(f"Thanks, bye! {STOP}"),
        ]
    )
    environment = get_environment()
    orchestrator = Orchestrator(
        domain="intake",
        agent=agent,
        user=user,
        environment=environment,
        task=task,
        max_steps=40,
    )
    result = orchestrator.run()

    # The task-supplied deterministic opener seeded the first assistant turn.
    assert result.messages[0].role == "assistant"
    assert result.messages[0].content == task.agent_opener
    assert result.messages[0].content.startswith("Hello, this is ")
    # The callee opened with the fixed first turn.
    first_user = next(m for m in result.messages if m.role == "user")
    assert first_user.content == "Hello?"
    assert result.termination_reason == TerminationReason.USER_STOP

    reward_info = EnvironmentEvaluator.calculate_reward(
        environment_constructor=lambda solo_mode=False: get_environment(
            solo_mode=solo_mode
        ),
        task=task,
        full_trajectory=result.messages,
    )
    assert reward_info.reward == 1.0
