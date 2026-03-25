from copy import deepcopy

import pytest

from tau2.config import (
    DEFAULT_LLM_AGENT,
    DEFAULT_LLM_ARGS_AGENT,
    DEFAULT_LLM_ARGS_USER,
    DEFAULT_LLM_USER,
)
from tau2.data_model.simulation import TextRunConfig
from tau2.data_model.tasks import Task, make_task
from tau2.data_model.voice import SynthesisConfig, TranscriptionConfig, VoiceSettings
from tau2.run import EvaluationType, run_task, run_tasks


@pytest.fixture
def transcription_voice_settings() -> VoiceSettings:
    return VoiceSettings(
        synthesis_config=None,
        transcription_config=TranscriptionConfig(),
    )


@pytest.fixture
def synthesis_voice_settings() -> VoiceSettings:
    return VoiceSettings(
        synthesis_config=SynthesisConfig(),
        transcription_config=None,
    )


@pytest.fixture
def run_config_voice(
    transcription_voice_settings: VoiceSettings, synthesis_voice_settings: VoiceSettings
) -> TextRunConfig:
    """Test config for voice-related tests (text agent with voice settings).

    Note: These tests are currently skipped. Voice settings are not part of
    TextRunConfig; full voice simulations should use VoiceRunConfig instead.
    """
    return TextRunConfig(
        domain="mock",
        agent="llm_agent",
        user="user_simulator",
        task_ids=["create_task_1"],
        llm_agent="gpt-3.5-turbo",
        llm_args_agent={},
        llm_user="gpt-3.5-turbo",
        llm_args_user={},
        num_trials=3,
        max_steps=20,
        max_errors=10,
        save_to=None,
        max_concurrency=3,
    )


@pytest.mark.skip(
    reason="Voice agent API not yet integrated into run_tasks/run_task. Use audio_native_config instead."
)
def test_simplified_voice_run(
    domain_name: str,
    transcription_voice_settings: VoiceSettings,
    synthesis_voice_settings: VoiceSettings,
):
    """Test that we can run a task with the mock domain and voice settings"""

    def run_simple_task(user_instruction: str, domain_name: str):
        task = make_task(
            user_instructions=user_instruction,
            eval_criteria=None,
            initialization_data=None,
            message_history=None,
        )
        simulation = run_tasks(
            domain=domain_name,
            tasks=[task],
            agent="voice_llm_agent",
            user="voice_user_simulator",
            llm_agent=DEFAULT_LLM_AGENT,
            llm_args_agent=deepcopy(DEFAULT_LLM_ARGS_AGENT),
            llm_user=DEFAULT_LLM_USER,
            llm_args_user=deepcopy(DEFAULT_LLM_ARGS_USER),
            max_steps=5,
            max_errors=5,
            evaluation_type=EvaluationType.ENV,
            console_display=False,
            max_concurrency=1,
            agent_voice_settings=transcription_voice_settings,
            user_voice_settings=synthesis_voice_settings,
        )
        return simulation

    simulation = run_simple_task(
        user_instruction="create a task called 'test' for user_1",
        domain_name=domain_name,
    )
    assert simulation is not None


@pytest.mark.skip(
    reason="Voice agent API not yet integrated into run_tasks/run_task. Use audio_native_config instead."
)
def test_run_tasks_base_voice(
    domain_name: str,
    base_task: Task,
    transcription_voice_settings: VoiceSettings,
    synthesis_voice_settings: VoiceSettings,
):
    """Test running a task with the mock domain and voice settings"""
    results = run_tasks(
        domain=domain_name,
        tasks=[base_task],
        agent="voice_llm_agent",
        user="voice_user_simulator",
        llm_agent="gpt-3.5-turbo",
        llm_args_agent={},
        llm_user="gpt-3.5-turbo",
        llm_args_user={},
        max_concurrency=1,
        agent_voice_settings=transcription_voice_settings,
        user_voice_settings=synthesis_voice_settings,
    )
    # Check that simulation ran and has expected structure
    assert len(results.simulations) == 1
    simulation = results.simulations[0]
    assert simulation.messages is not None
    assert len(simulation.messages) > 0
    assert simulation.start_time is not None
    assert simulation.end_time is not None
    assert simulation.reward_info.reward is not None


@pytest.mark.skip(
    reason="Voice agent API not yet integrated into run_tasks/run_task. Use audio_native_config instead."
)
def test_run_task_base_voice(
    domain_name: str,
    base_task: Task,
    transcription_voice_settings: VoiceSettings,
    synthesis_voice_settings: VoiceSettings,
):
    """Test running a task with the mock domain"""
    simulation = run_task(
        domain=domain_name,
        task=base_task,
        agent="voice_llm_agent",
        user="voice_user_simulator",
        llm_agent="gpt-3.5-turbo",
        llm_args_agent={},
        llm_user="gpt-3.5-turbo",
        llm_args_user={},
        evaluation_type=EvaluationType.ENV,
        agent_voice_settings=transcription_voice_settings,
        user_voice_settings=synthesis_voice_settings,
    )
    # Check that simulation ran and has expected structure
    assert simulation.messages is not None
    assert len(simulation.messages) > 0
    assert simulation.start_time is not None
    assert simulation.end_time is not None
    assert simulation.reward_info.reward is not None


# def test_run_tasks_message_history_voice(
#     domain_name: str,
#     task_with_message_history: Task,
#     transcription_voice_settings: VoiceSettings,
#     synthesis_voice_settings: VoiceSettings,
# ):
#     """Test running a task with message history"""
#     print(task_with_message_history.model_dump_json(indent=2))
#     simulation = run_task(
#         domain=domain_name,
#         task=task_with_message_history,
#         agent="llm_agent",
#         user="voice_user_simulator",
#         llm_agent="gpt-3.5-turbo",
#         llm_args_agent={},
#         llm_user="gpt-3.5-turbo",
#         llm_args_user={},
#         agent_voice_settings=transcription_voice_settings,
#         user_voice_settings=synthesis_voice_settings,
#     )
#     assert simulation is not None


# def test_run_tasks_initialization_data_voice(
#     domain_name: str,
#     task_with_initialization_data: Task,
#     transcription_voice_settings: VoiceSettings,
#     synthesis_voice_settings: VoiceSettings,
# ):
#     """Test running a task with initialization data"""
#     simulation = run_task(
#         domain=domain_name,
#         task=task_with_initialization_data,
#         agent="llm_agent",
#         user="voice_user_simulator",
#         llm_agent="gpt-3.5-turbo",
#         llm_args_agent={},
#         llm_user="gpt-3.5-turbo",
#         llm_args_user={},
#         agent_voice_settings=transcription_voice_settings,
#         user_voice_settings=synthesis_voice_settings,
#     )
#     assert simulation is not None


# def test_run_tasks_initialization_actions_voice(
#     domain_name: str,
#     task_with_initialization_actions: Task,
#     transcription_voice_settings: VoiceSettings,
#     synthesis_voice_settings: VoiceSettings,
# ):
#     """Test running a task with initialization actions"""
#     simulation = run_task(
#         domain=domain_name,
#         task=task_with_initialization_actions,
#         agent="llm_agent",
#         user="voice_user_simulator",
#         llm_agent="gpt-3.5-turbo",
#         llm_args_agent={},
#         llm_user="gpt-3.5-turbo",
#         llm_args_user={},
#         agent_voice_settings=transcription_voice_settings,
#         user_voice_settings=synthesis_voice_settings,
#     )
#     assert simulation is not None


# def test_run_tasks_env_assertions_voice(
#     domain_name: str,
#     task_with_env_assertions: Task,
#     transcription_voice_settings: VoiceSettings,
#     synthesis_voice_settings: VoiceSettings,
# ):
#     """Test running a task with env assertions"""
#     simulation = run_task(
#         domain=domain_name,
#         task=task_with_env_assertions,
#         agent="llm_agent",
#         user="voice_user_simulator",
#         llm_agent="gpt-3.5-turbo",
#         llm_args_agent={},
#         llm_user="gpt-3.5-turbo",
#         llm_args_user={},
#         evaluation_type=EvaluationType.ENV,
#         agent_voice_settings=transcription_voice_settings,
#         user_voice_settings=synthesis_voice_settings,
#     )
#     # Check that simulation ran and has expected structure
#     assert simulation.messages is not None
#     assert len(simulation.messages) > 0
#     assert simulation.start_time is not None
#     assert simulation.end_time is not None
#     # These assertions can fail if model is not good enough
#     assert simulation.reward_info.reward == 1.0
#     assert len(simulation.reward_info.env_assertions) == 1
#     assert simulation.reward_info.env_assertions[0].met is True
#     # Add an env_assertion that will fail and test that the reward is 0.0
#     task_with_env_assertions.evaluation_criteria.env_assertions.append(
#         EnvAssertion(
#             env_type="assistant",
#             func_name="assert_task_status",
#             arguments={"task_id": "task_1", "expected_status": "made_up_status"},
#         )
#     )
#     simulation = run_task(
#         domain=domain_name,
#         task=task_with_env_assertions,
#         agent="llm_agent",
#         user="voice_user_simulator",
#         llm_agent="gpt-3.5-turbo",
#         llm_args_agent={},
#         llm_user="gpt-3.5-turbo",
#         llm_args_user={},
#         evaluation_type=EvaluationType.ENV,
#         agent_voice_settings=transcription_voice_settings,
#         user_voice_settings=synthesis_voice_settings,
#     )
#     assert simulation.reward_info.reward == 0.0
#     assert len(simulation.reward_info.env_assertions) == 2
#     assert simulation.reward_info.env_assertions[0].met is True
#     assert simulation.reward_info.env_assertions[1].met is False


# def test_run_tasks_history_and_env_assertions(
#     domain_name: str,
#     task_with_history_and_env_assertions: Task,
#     transcription_voice_settings: VoiceSettings,
#     synthesis_voice_settings: VoiceSettings,
# ):
#     """Test running a task with history and env assertions"""
#     simulation = run_task(
#         domain=domain_name,
#         task=task_with_history_and_env_assertions,
#         agent="llm_agent",
#         user="voice_user_simulator",
#         llm_agent="gpt-3.5-turbo",
#         llm_args_agent={},
#         llm_user="gpt-3.5-turbo",
#         llm_args_user={},
#         agent_voice_settings=transcription_voice_settings,
#         user_voice_settings=synthesis_voice_settings,
#     )
#     assert simulation is not None


# def test_run_tasks_nl_assertions_voice(
#     domain_name: str,
#     task_with_nl_assertions: Task,
#     transcription_voice_settings: VoiceSettings,
#     synthesis_voice_settings: VoiceSettings,
# ):
#     """Test running a task with the mock domain"""
#     task = get_tasks(domain_name, task_ids=["create_task_1_nl_eval"])[0]
#     simulation = run_task(
#         domain=domain_name,
#         task=task,
#         agent="llm_agent",
#         user="user_simulator",
#         llm_agent="gpt-3.5-turbo",
#         llm_args_agent={},
#         llm_user="gpt-3.5-turbo",
#         llm_args_user={},
#         evaluation_type=EvaluationType.NL_ASSERTIONS,
#         agent_voice_settings=transcription_voice_settings,
#         user_voice_settings=synthesis_voice_settings,
#     )
#     # Check that simulation ran and has expected structure
#     assert simulation.messages is not None
#     assert len(simulation.messages) > 0
#     assert simulation.start_time is not None
#     assert simulation.end_time is not None
#     assert simulation.reward_info.reward == 1.0
#     assert len(simulation.reward_info.nl_assertions) == 2
#     assert simulation.reward_info.nl_assertions[0].met is True
#     assert simulation.reward_info.nl_assertions[1].met is True

#     # Add an nl_assertion that will fail and test that the reward is 0.0
#     task.evaluation_criteria.nl_assertions.append("The user is not complimented")
#     simulation = run_task(
#         domain=domain_name,
#         task=task,
#         agent="llm_agent",
#         user="voice_user_simulator",
#         llm_agent="gpt-3.5-turbo",
#         llm_args_agent={},
#         llm_user="gpt-3.5-turbo",
#         llm_args_user={},
#         evaluation_type=EvaluationType.NL_ASSERTIONS,
#     )
#     assert simulation.reward_info.reward == 0.0

#     assert len(simulation.reward_info.nl_assertions) == 3
#     assert simulation.reward_info.nl_assertions[0].met is True
#     assert simulation.reward_info.nl_assertions[1].met is True
#     assert simulation.reward_info.nl_assertions[2].met is False


# def test_run_tasks_action_checks_voice(
#     domain_name: str,
#     task_with_action_checks: Task,
#     transcription_voice_settings: VoiceSettings,
#     synthesis_voice_settings: VoiceSettings,
# ):
#     """Test running a task with action checks"""
#     simulation = run_task(
#         domain=domain_name,
#         task=task_with_action_checks,
#         agent="llm_agent",
#         user="voice_user_simulator",
#         llm_agent="gpt-3.5-turbo",
#         llm_args_agent={},
#         llm_user="gpt-3.5-turbo",
#         llm_args_user={},
#         agent_voice_settings=transcription_voice_settings,
#         user_voice_settings=synthesis_voice_settings,
#     )
#     assert simulation is not None
#     # Following assertions can fail if model is not good enough
#     assert simulation.reward_info.reward == 1.0
#     assert simulation.reward_info.reward_breakdown[RewardType.DB] == 1.0
#     assert simulation.reward_info.reward_breakdown[RewardType.ACTION] == 1.0


# def test_run_domain_voice(
#     run_config_voice: RunConfig,
#     transcription_voice_settings: VoiceSettings,
#     synthesis_voice_settings: VoiceSettings,
# ):
#     """Test running a domain with the mock domain
#     Requires environment manager to be running
#     """
#     run_config_voice.agent_voice_settings = transcription_voice_settings
#     run_config_voice.user_voice_settings = synthesis_voice_settings
#     simulation_results = run_domain(run_config_voice)
#     assert simulation_results is not None


# def test_run_gt_agent_voice(
#     domain_name: str,
#     base_task: Task,
#     transcription_voice_settings: VoiceSettings,
#     synthesis_voice_settings: VoiceSettings,
# ):
#     """Test running gt agent"""
#     simulation_results = run_tasks(
#         domain=domain_name,
#         tasks=[base_task],
#         agent="llm_agent_gt",
#         user="voice_user_simulator",
#         llm_agent="gpt-3.5-turbo",
#         llm_args_agent={},
#         llm_user="gpt-3.5-turbo",
#         llm_args_user={},
#         agent_voice_settings=transcription_voice_settings,
#         user_voice_settings=synthesis_voice_settings,
#     )
#     assert simulation_results is not None
