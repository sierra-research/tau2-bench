# Copyright Sierra
"""Synthetic on-disk run directories in the real ``Results`` layout.

Builders for parity-probe flagging tests, stop-rule simulations, and
tag-pivot analysis tests (factory parallel PRs). Runs are written with
``Results.save(format="dir")`` — the directory layout voice runs use
(results.json metadata + simulations/<id>.json) — so anything written here
round-trips through ``Results.load`` exactly like a real run.

Only the fields analysis code reads are meaningful: task id, trial, reward,
termination reason, and a ``SpeechEnvironment`` carrying the persona_id.
Everything else is minimal-but-valid filler.
"""

from __future__ import annotations

from pathlib import Path

from tau2.data_model.message import AssistantMessage
from tau2.data_model.simulation import (
    AgentInfo,
    CommunicateCheck,
    Info,
    Results,
    RewardInfo,
    SimulationRun,
    TerminationReason,
    UserInfo,
)
from tau2.data_model.tasks import StructuredUserInstructions, Task, UserScenario
from tau2.data_model.voice import SpeechEnvironment
from tau2.environment.environment import EnvironmentInfo
from test_multilingual.factory_testing.toy_language import TOY_DOMAIN, TOY_LANGUAGE

DEFAULT_PERSONA_ID = "tessa_tl_v1"


def _stub_task(task_id: str) -> Task:
    return Task(
        id=task_id,
        user_scenario=UserScenario(
            instructions=StructuredUserInstructions(
                domain=TOY_DOMAIN,
                task_instructions=f"Synthetic task {task_id} for results-factory tests.",
                reason_for_call="Synthetic.",
            )
        ),
    )


def _stub_info(num_trials: int) -> Info:
    return Info(
        git_commit="0000000000000000000000000000000000000000",
        num_trials=num_trials,
        max_steps=50,
        max_errors=10,
        user_info=UserInfo(implementation="fake_user_simulator", llm="fake-llm"),
        agent_info=AgentInfo(implementation="fake_llm_agent", llm="fake-llm"),
        environment_info=EnvironmentInfo(domain_name=TOY_DOMAIN, policy="synthetic"),
    )


def make_run_dir(
    tmp_path: Path,
    arm_name: str,
    rewards: dict[str, list[float]],
    persona_by_task: dict[str, str] | None = None,
    termination_by_task: dict[str, TerminationReason] | None = None,
    language: str | None = TOY_LANGUAGE,
    communicate_checks_by_task: dict[str, list[dict]] | None = None,
    agent_messages_by_task: dict[str, list[str]] | None = None,
) -> Path:
    """Write a synthetic dir-format run and return its directory.

    Args:
        tmp_path: Parent directory (the run dir is ``tmp_path / arm_name``).
        arm_name: Run/arm name; becomes the directory name.
        rewards: task_id -> one reward per trial (list length = trial count;
            lengths may differ across tasks).
        persona_by_task: task_id -> persona_id recorded on each simulation's
            SpeechEnvironment (default: every task uses ``tessa_tl_v1``).
        termination_by_task: task_id -> termination reason (default
            ``USER_STOP``).
        language: language code stamped on the SpeechEnvironment (None for an
            English-baseline arm).
        communicate_checks_by_task: task_id -> CommunicateCheck dicts
            (``{"info", "met", "justification"}``) recorded on every trial's
            ``reward_info.communicate_checks`` (default: none — unchanged
            historical behavior). Used by judge-calibration tests.
        agent_messages_by_task: task_id -> assistant-turn texts written as
            the simulation's messages (default: empty message list).

    Returns:
        The run directory (contains results.json + simulations/).
    """
    persona_by_task = persona_by_task or {}
    termination_by_task = termination_by_task or {}
    communicate_checks_by_task = communicate_checks_by_task or {}
    agent_messages_by_task = agent_messages_by_task or {}
    num_trials = max((len(r) for r in rewards.values()), default=0)

    simulations: list[SimulationRun] = []
    for task_id, task_rewards in rewards.items():
        persona_id = persona_by_task.get(task_id, DEFAULT_PERSONA_ID)
        termination = termination_by_task.get(task_id, TerminationReason.USER_STOP)
        communicate_checks = [
            CommunicateCheck(**check)
            for check in communicate_checks_by_task.get(task_id, [])
        ] or None
        messages = [
            AssistantMessage(role="assistant", content=text)
            for text in agent_messages_by_task.get(task_id, [])
        ]
        for trial, reward in enumerate(task_rewards):
            simulations.append(
                SimulationRun(
                    id=f"{task_id}_trial{trial}",
                    task_id=task_id,
                    trial=trial,
                    start_time="2026-01-01T00:00:00",
                    end_time="2026-01-01T00:05:00",
                    duration=300.0,
                    termination_reason=termination,
                    reward_info=RewardInfo(
                        reward=reward, communicate_checks=communicate_checks
                    ),
                    messages=messages,
                    speech_environment=SpeechEnvironment(
                        persona_name=persona_id,
                        persona_id=persona_id,
                        language=language,
                    ),
                )
            )

    results = Results(
        info=_stub_info(num_trials=num_trials),
        tasks=[_stub_task(task_id) for task_id in rewards],
        simulations=simulations,
    )
    run_dir = tmp_path / arm_name
    run_dir.mkdir(parents=True, exist_ok=True)
    results.save(run_dir, format="dir")
    return run_dir


def paired_run_dirs(
    tmp_path: Path,
    baseline_rewards: dict[str, list[float]],
    deltas: dict[str, float],
    arm_names: tuple[str, str] = ("toylang", "english_baseline"),
    persona_by_task: dict[str, str] | None = None,
) -> tuple[Path, Path]:
    """Two-arm setup with controlled per-task reward deltas.

    The main arm's reward for a task/trial is the baseline reward plus that
    task's delta, clamped to [0, 1] (so e.g. ``deltas={"3": -0.5}`` makes
    task 3 a regression for parity-probe flagging tests).

    Args:
        tmp_path: Parent directory for both run dirs.
        baseline_rewards: task_id -> per-trial rewards for the baseline arm.
        deltas: task_id -> reward delta applied to the main arm (missing
            task ids get delta 0.0, i.e. parity).
        arm_names: (main_arm, baseline_arm) directory names.
        persona_by_task: persona ids for the MAIN arm's simulations (the
            baseline arm is stamped as English: no language, default persona).

    Returns:
        (main_run_dir, baseline_run_dir).
    """
    main_arm, baseline_arm = arm_names
    main_rewards = {
        task_id: [
            min(1.0, max(0.0, reward + deltas.get(task_id, 0.0)))
            for reward in task_rewards
        ]
        for task_id, task_rewards in baseline_rewards.items()
    }
    main_dir = make_run_dir(
        tmp_path,
        main_arm,
        main_rewards,
        persona_by_task=persona_by_task,
    )
    baseline_dir = make_run_dir(
        tmp_path,
        baseline_arm,
        baseline_rewards,
        persona_by_task={task_id: "default" for task_id in baseline_rewards},
        language=None,
    )
    return main_dir, baseline_dir
