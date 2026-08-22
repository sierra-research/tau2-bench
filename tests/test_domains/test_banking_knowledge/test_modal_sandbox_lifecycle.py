"""Lifecycle tests for banking environments that own Modal sandboxes."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage
from tau2.data_model.simulation import RewardInfo, SimulationRun, TerminationReason
from tau2.data_model.tasks import EvaluationCriteria, RewardType, make_task
from tau2.domains.banking_knowledge.retrieval_toolkits import (
    KnowledgeToolsAllTools,
    KnowledgeToolsWithShell,
)
from tau2.environment.environment import Environment
from tau2.environment.toolkit import ToolKitBase
from tau2.evaluator import evaluator as evaluator_module
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.evaluator.evaluator_env import (
    EnvironmentEvaluator,
    FullDuplexEnvironmentEvaluator,
)
from tau2.knowledge import modal_sandbox_manager
from tau2.knowledge.modal_sandbox_manager import ModalSandboxManager
from tau2.runner import build as build_module
from tau2.runner import helpers as helpers_module
from tau2.runner import simulation as simulation_module


class _ClosingToolkit(ToolKitBase):
    def __init__(self):
        super().__init__()
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class _HashingShellTools(KnowledgeToolsWithShell):
    def get_db_hash(self) -> str:
        return "unchanged-db"


def _task_with_empty_db_reference():
    return make_task(
        user_instructions="No action is needed.",
        eval_criteria=EvaluationCriteria(
            actions=[],
            reward_basis=[RewardType.DB],
        ),
    )


def _completed_simulation(task_id: str) -> SimulationRun:
    return SimulationRun(
        id="simulation-1",
        task_id=task_id,
        start_time="start",
        end_time="end",
        duration=0.0,
        termination_reason=TerminationReason.AGENT_STOP,
        messages=[],
    )


def _fake_modal():
    process = MagicMock()
    process.stdout.read.return_value = "INDEX.txt\n"
    process.stderr.read.return_value = ""
    process.wait.return_value = 0

    sandbox = MagicMock()
    sandbox.exec.return_value = process

    image = MagicMock()
    image.object_id = "im-lifecycle-test"
    image.pip_install.return_value = image
    image.run_commands.return_value = image
    image.add_local_file.return_value = image
    image.add_local_dir.return_value = image

    modal = SimpleNamespace(
        App=SimpleNamespace(lookup=MagicMock(return_value=object())),
        Image=SimpleNamespace(debian_slim=MagicMock(return_value=image)),
        Sandbox=SimpleNamespace(create=MagicMock(return_value=sandbox)),
        exception=SimpleNamespace(ExecTimeoutError=TimeoutError),
    )
    return modal, sandbox


def _new_modal_environment(tmp_path, managers: list[ModalSandboxManager]):
    manager = ModalSandboxManager(base_temp_dir=str(tmp_path))
    manager.export_documents(
        [{"id": "doc", "title": "Document", "content": "Contents"}]
    )
    managers.append(manager)
    tools = _HashingShellTools(MagicMock(), manager)
    return Environment(
        domain_name="banking_knowledge",
        policy="policy",
        tools=tools,
    )


def test_environment_close_is_idempotent_and_context_managed():
    toolkit = _ClosingToolkit()

    with Environment(
        domain_name="test",
        policy="policy",
        tools=toolkit,
        user_tools=toolkit,
    ) as environment:
        assert environment.tools is toolkit

    environment.close()
    assert toolkit.close_calls == 1


@pytest.mark.parametrize(
    "toolkit_factory",
    [
        lambda sandbox: KnowledgeToolsWithShell(MagicMock(), sandbox),
        lambda sandbox: KnowledgeToolsAllTools(
            MagicMock(), MagicMock(), MagicMock(), sandbox
        ),
    ],
)
def test_shell_toolkits_release_their_owned_sandbox_once(toolkit_factory):
    sandbox = MagicMock()
    toolkit = toolkit_factory(sandbox)

    toolkit.close()
    toolkit.close()

    sandbox.cleanup.assert_called_once_with()
    assert toolkit._sandbox is None


@pytest.mark.parametrize(
    ("evaluator", "trajectory"),
    [
        (EnvironmentEvaluator, []),
        (FullDuplexEnvironmentEvaluator, []),
    ],
)
def test_environment_evaluators_close_predicted_and_gold_environments(
    evaluator, trajectory, base_task
):
    predicted_environment = MagicMock(spec=Environment)
    gold_environment = MagicMock(spec=Environment)
    predicted_environment.get_db_hash.return_value = "same"
    gold_environment.get_db_hash.return_value = "same"
    predicted_environment.get_user_db_hash.return_value = "same-user"
    gold_environment.get_user_db_hash.return_value = "same-user"
    predicted_environment.run_env_assertion.return_value = True
    constructor = MagicMock(side_effect=[predicted_environment, gold_environment])

    evaluator.calculate_reward(
        environment_constructor=constructor,
        task=base_task,
        full_trajectory=trajectory,
    )

    predicted_environment.close.assert_called_once_with()
    gold_environment.close.assert_called_once_with()


@pytest.mark.parametrize(
    ("evaluator", "trajectory"),
    [
        (EnvironmentEvaluator, []),
        (FullDuplexEnvironmentEvaluator, []),
    ],
)
def test_environment_evaluators_close_after_replay_failure(
    evaluator, trajectory, base_task
):
    predicted_environment = MagicMock(spec=Environment)
    predicted_environment.set_state.side_effect = RuntimeError("replay failed")
    constructor = MagicMock(return_value=predicted_environment)

    with pytest.raises(RuntimeError, match="replay failed"):
        evaluator.calculate_reward(
            environment_constructor=constructor,
            task=base_task,
            full_trajectory=trajectory,
        )

    predicted_environment.close.assert_called_once_with()


def test_lazy_regrade_never_creates_remote_and_removes_all_staging(
    tmp_path, monkeypatch
):
    load_modal = MagicMock(side_effect=AssertionError("unexpected Modal call"))
    monkeypatch.setattr(modal_sandbox_manager, "_load_modal", load_modal)

    managers: list[ModalSandboxManager] = []

    def environment_constructor(**_kwargs):
        return _new_modal_environment(tmp_path, managers)

    get_constructor = MagicMock(return_value=environment_constructor)
    monkeypatch.setattr(
        evaluator_module.registry, "get_env_constructor", get_constructor
    )
    task = _task_with_empty_db_reference()
    simulation = _completed_simulation(task.id)
    simulation.messages = [
        AssistantMessage(
            id="assistant-shell",
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(id="shell-call", name="shell", arguments={"command": "ls"})
            ],
        ),
        ToolMessage(
            id="shell-call",
            role="tool",
            content="INDEX.txt\n",
            requestor="assistant",
        ),
    ]

    reward = evaluate_simulation(
        simulation=simulation,
        task=task,
        evaluation_type=EvaluationType.ENV,
        solo_mode=False,
        domain="banking_knowledge",
        env_kwargs={"retrieval_variant": "terminal_use"},
        strict_replay=False,
    )

    assert reward.reward == 1.0
    assert len(managers) == 3  # tool introspection, predicted replay, gold replay
    assert all(not manager.sandbox_dir.exists() for manager in managers)
    assert list(tmp_path.iterdir()) == []
    load_modal.assert_not_called()


def test_live_run_terminates_remote_sandbox_and_removes_staging(tmp_path, monkeypatch):
    modal, remote_sandbox = _fake_modal()
    monkeypatch.setattr(modal_sandbox_manager, "_load_modal", lambda: modal)
    managers: list[ModalSandboxManager] = []
    environment = _new_modal_environment(tmp_path, managers)
    manager = managers[0]
    task = _task_with_empty_db_reference()
    completed = _completed_simulation(task.id)

    def run():
        assert manager.run_command("ls") == (0, "INDEX.txt\n", "")
        return completed

    orchestrator = SimpleNamespace(
        run=run,
        environment=environment,
        task=task,
        solo_mode=False,
    )

    def evaluate_after_live_cleanup(**_kwargs):
        remote_sandbox.terminate.assert_called_once_with(wait=True)
        remote_sandbox.detach.assert_called_once_with()
        assert not manager.sandbox_dir.exists()
        return RewardInfo(reward=1.0)

    monkeypatch.setattr(
        simulation_module,
        "evaluate_simulation",
        MagicMock(side_effect=evaluate_after_live_cleanup),
    )

    result = simulation_module.run_simulation(orchestrator)

    assert result is completed
    remote_sandbox.terminate.assert_called_once_with(wait=True)
    remote_sandbox.detach.assert_called_once_with()
    assert not manager.sandbox_dir.exists()
    assert list(tmp_path.iterdir()) == []


def test_live_run_closes_environment_when_orchestrator_raises(monkeypatch):
    environment = MagicMock(spec=Environment)
    orchestrator = SimpleNamespace(
        run=MagicMock(side_effect=RuntimeError("simulation failed")),
        environment=environment,
    )

    with pytest.raises(RuntimeError, match="simulation failed"):
        simulation_module.run_simulation(orchestrator)

    environment.close.assert_called_once_with()


def test_text_orchestrator_build_failure_closes_environment(monkeypatch):
    environment = MagicMock(spec=Environment)
    config = SimpleNamespace(
        seed=42,
        effective_agent="agent",
        effective_user="user",
        domain="banking_knowledge",
        retrieval_config=None,
        llm_agent="agent-model",
        llm_args_agent={},
    )
    monkeypatch.setattr(
        build_module, "build_environment", lambda *_args, **_kw: environment
    )
    monkeypatch.setattr(
        build_module.registry, "get_agent_metadata", lambda *_args, **_kw: False
    )
    monkeypatch.setattr(
        build_module,
        "build_agent",
        MagicMock(side_effect=RuntimeError("agent build failed")),
    )

    with pytest.raises(RuntimeError, match="agent build failed"):
        build_module.build_text_orchestrator(config, _task_with_empty_db_reference())

    environment.close.assert_called_once_with()


def test_environment_metadata_construction_closes_environment(monkeypatch):
    toolkit = _ClosingToolkit()
    environment = Environment("banking_knowledge", "policy", tools=toolkit)
    constructor = MagicMock(return_value=environment)
    monkeypatch.setattr(
        helpers_module.registry,
        "get_env_constructor",
        MagicMock(return_value=constructor),
    )

    info = helpers_module.get_environment_info("banking_knowledge")

    assert info.domain_name == "banking_knowledge"
    assert toolkit.close_calls == 1
