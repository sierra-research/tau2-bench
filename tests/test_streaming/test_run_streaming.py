"""
Tests for full-duplex/audio-native run functionality.

This test suite provides equivalents of the half-duplex run tests in test_run.py
for the audio-native mode using DiscreteTimeAudioNativeAgent and VoiceStreamingUserSimulator.

These are integration tests that actually run full-duplex simulations with real API calls.
They require:
- OpenAI Realtime API access for DiscreteTimeAudioNativeAgent
- TTS synthesis for VoiceStreamingUserSimulator

Use pytest marker to skip these tests if APIs are not configured:
    pytest -m "not full_duplex_integration"
"""

from copy import deepcopy

import pytest

from tau2.config import (
    DEFAULT_LLM_AGENT,
    DEFAULT_LLM_ARGS_AGENT,
    DEFAULT_LLM_ARGS_USER,
    DEFAULT_LLM_USER,
)
from tau2.data_model.simulation import (
    AudioNativeConfig,
    RunConfig,
    TextRunConfig,
    VoiceRunConfig,
)
from tau2.data_model.tasks import EnvAssertion, Task, make_task
from tau2.run import (
    EvaluationType,
    get_tasks,
    run_domain,
    run_task,
    run_tasks,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def domain_name():
    return "mock"


@pytest.fixture
def base_task() -> Task:
    return get_tasks("mock", task_ids=["create_task_1"])[0]


@pytest.fixture
def task_with_env_assertions() -> Task:
    return get_tasks("mock", task_ids=["create_task_1_with_env_assertions"])[0]


@pytest.fixture
def audio_native_config() -> AudioNativeConfig:
    """Create an AudioNativeConfig for testing.

    Voice conversations take longer than text, so we need more time.
    120 seconds = 600 ticks at 0.2s per tick.
    """
    return AudioNativeConfig(
        tick_duration_seconds=0.2,
        max_steps_seconds=120,  # 120 seconds max, 600 ticks
    )


@pytest.fixture
def run_config_audio_native() -> VoiceRunConfig:
    """Create a VoiceRunConfig for testing."""
    return VoiceRunConfig(
        domain="mock",
        task_ids=["create_task_1"],
        llm_user="gpt-4o-mini",
        llm_args_user={},
        num_trials=1,
        max_errors=10,
        save_to=None,
        max_concurrency=1,
        audio_native_config=AudioNativeConfig(
            tick_duration_seconds=0.2,
            max_steps_seconds=120,
        ),
    )


# =============================================================================
# AudioNativeConfig Model Tests (no API calls needed)
# =============================================================================


def test_audio_native_config_custom_values():
    """Test that AudioNativeConfig accepts custom values."""
    config = AudioNativeConfig(
        tick_duration_seconds=0.1,
        max_steps_seconds=120,
        wait_to_respond_threshold_other_seconds=2.0,
    )
    assert config.tick_duration_seconds == 0.1
    assert config.max_steps_seconds == 120
    assert config.tick_duration_ms == 100.0
    assert config.max_steps_ticks == 1200  # 120 / 0.1
    assert config.wait_to_respond_threshold_other_ticks == 20  # 2.0 / 0.1


def test_audio_native_config_derived_properties():
    """Test that derived properties compute correctly."""
    config = AudioNativeConfig(
        tick_duration_seconds=0.5,
        max_steps_seconds=60,
        wait_to_respond_threshold_other_seconds=1.0,
        wait_to_respond_threshold_self_seconds=2.0,
        yield_threshold_when_interrupted_seconds=1.5,
        yield_threshold_when_interrupting_seconds=3.0,
    )
    assert config.tick_duration_ms == 500.0
    assert config.max_steps_ticks == 120  # 60 / 0.5
    assert config.wait_to_respond_threshold_other_ticks == 2  # 1.0 / 0.5
    assert config.wait_to_respond_threshold_self_ticks == 4  # 2.0 / 0.5
    assert config.yield_threshold_when_interrupted_ticks == 3  # 1.5 / 0.5
    assert config.yield_threshold_when_interrupting_ticks == 6  # 3.0 / 0.5


# =============================================================================
# RunConfig with AudioNativeConfig Tests (no API calls needed)
# =============================================================================


def test_voice_run_config():
    """Test that VoiceRunConfig correctly sets effective values."""
    audio_config = AudioNativeConfig()
    run_config = VoiceRunConfig(
        domain="mock",
        audio_native_config=audio_config,
    )

    assert run_config.audio_native_config is not None
    assert run_config.is_voice is True
    assert run_config.effective_agent == "discrete_time_audio_native_agent"
    assert run_config.effective_user == "voice_streaming_user_simulator"


def test_text_run_config():
    """Test that TextRunConfig correctly sets effective values."""
    run_config = TextRunConfig(
        domain="mock",
        agent="llm_agent",
        user="user_simulator",
    )

    assert run_config.is_voice is False
    assert run_config.effective_agent == "llm_agent"
    assert run_config.effective_user == "user_simulator"
    assert not isinstance(run_config, VoiceRunConfig)


def test_run_config_effective_max_steps():
    """Test that config types correctly compute effective max steps."""
    # TextRunConfig uses regular max_steps
    run_config = TextRunConfig(
        domain="mock",
        agent="llm_agent",
        user="user_simulator",
        max_steps=50,
    )
    assert run_config.effective_max_steps == 50

    # VoiceRunConfig uses max_steps_ticks from audio_native_config
    audio_config = AudioNativeConfig(
        tick_duration_seconds=0.2,
        max_steps_seconds=120,  # 120 / 0.2 = 600 ticks
    )
    run_config_audio = VoiceRunConfig(
        domain="mock",
        audio_native_config=audio_config,
    )
    assert run_config_audio.effective_max_steps == 600


# =============================================================================
# Full-Duplex Integration Tests
# These tests require OpenAI Realtime API and TTS access
# =============================================================================


@pytest.mark.full_duplex_integration
def test_simplified_run_audio_native(
    domain_name: str, audio_native_config: AudioNativeConfig
):
    """Test that we can run a simple task with audio-native mode.

    Equivalent of test_simplified_run for full-duplex.
    """
    task = make_task(
        user_instructions="create a task called 'test' for user_1",
        eval_criteria=None,
        initialization_data=None,
        message_history=None,
    )
    simulation = run_tasks(
        domain=domain_name,
        tasks=[task],
        agent="discrete_time_audio_native_agent",
        user="voice_streaming_user_simulator",
        llm_agent=DEFAULT_LLM_AGENT,
        llm_args_agent=deepcopy(DEFAULT_LLM_ARGS_AGENT),
        llm_user=DEFAULT_LLM_USER,
        llm_args_user=deepcopy(DEFAULT_LLM_ARGS_USER),
        max_steps=300,  # 60 seconds at 0.2s ticks
        max_errors=5,
        evaluation_type=EvaluationType.ENV,
        console_display=False,
        max_concurrency=1,
        audio_native_config=audio_native_config,
    )
    assert simulation is not None


@pytest.mark.full_duplex_integration
def test_run_tasks_base_audio_native(
    domain_name: str, base_task: Task, audio_native_config: AudioNativeConfig
):
    """Test running a task with audio-native mode.

    Equivalent of test_run_tasks_base for full-duplex.
    """
    results = run_tasks(
        domain=domain_name,
        tasks=[base_task],
        agent="discrete_time_audio_native_agent",
        user="voice_streaming_user_simulator",
        llm_agent="gpt-4o-mini",
        llm_args_agent={},
        llm_user="gpt-4o-mini",
        llm_args_user={},
        max_concurrency=1,
        audio_native_config=audio_native_config,
    )
    # Check that simulation ran and has expected structure
    assert len(results.simulations) == 1
    simulation = results.simulations[0]
    assert len(simulation.get_messages()) > 0
    assert simulation.start_time is not None
    assert simulation.end_time is not None
    assert simulation.reward_info.reward is not None


@pytest.mark.full_duplex_integration
def test_run_task_base_audio_native(
    domain_name: str, base_task: Task, audio_native_config: AudioNativeConfig
):
    """Test running a single task with audio-native mode.

    Equivalent of test_run_task_base for full-duplex.
    """
    simulation = run_task(
        domain=domain_name,
        task=base_task,
        agent="discrete_time_audio_native_agent",
        user="voice_streaming_user_simulator",
        llm_agent="gpt-4o-mini",
        llm_args_agent={},
        llm_user="gpt-4o-mini",
        llm_args_user={},
        evaluation_type=EvaluationType.ENV,
        audio_native_config=audio_native_config,
    )
    # Check that simulation ran and has expected structure
    assert len(simulation.get_messages()) > 0
    assert simulation.start_time is not None
    assert simulation.end_time is not None
    assert simulation.reward_info.reward is not None


@pytest.mark.full_duplex_integration
@pytest.mark.xfail(
    reason="Test depends on LLM quality and voice timing - may not consistently complete task in time"
)
def test_run_tasks_env_assertions_audio_native(
    domain_name: str,
    task_with_env_assertions: Task,
    audio_native_config: AudioNativeConfig,
):
    """Test running a task with env assertions in audio-native mode.

    Equivalent of test_run_tasks_env_assertions for full-duplex.
    """
    simulation = run_task(
        domain=domain_name,
        task=task_with_env_assertions,
        agent="discrete_time_audio_native_agent",
        user="voice_streaming_user_simulator",
        llm_agent="gpt-4o-mini",
        llm_args_agent={},
        llm_user="gpt-4o-mini",
        llm_args_user={},
        evaluation_type=EvaluationType.ENV,
        audio_native_config=audio_native_config,
    )
    # Check that simulation ran and has expected structure
    assert len(simulation.get_messages()) > 0
    assert simulation.start_time is not None
    assert simulation.end_time is not None
    # These assertions can fail if model is not good enough
    assert simulation.reward_info.reward == 1.0
    assert len(simulation.reward_info.env_assertions) == 1
    assert simulation.reward_info.env_assertions[0].met is True

    # Add an env_assertion that will fail and test that the reward is 0.0
    task_with_env_assertions.evaluation_criteria.env_assertions.append(
        EnvAssertion(
            env_type="assistant",
            func_name="assert_task_status",
            arguments={"task_id": "task_1", "expected_status": "made_up_status"},
        )
    )
    simulation = run_task(
        domain=domain_name,
        task=task_with_env_assertions,
        agent="discrete_time_audio_native_agent",
        user="voice_streaming_user_simulator",
        llm_agent="gpt-4o-mini",
        llm_args_agent={},
        llm_user="gpt-4o-mini",
        llm_args_user={},
        evaluation_type=EvaluationType.ENV,
        audio_native_config=audio_native_config,
    )
    assert simulation.reward_info.reward == 0.0
    assert len(simulation.reward_info.env_assertions) == 2
    assert simulation.reward_info.env_assertions[0].met is True
    assert simulation.reward_info.env_assertions[1].met is False


@pytest.mark.full_duplex_integration
@pytest.mark.xfail(
    reason="Test depends on LLM quality and voice timing - may not consistently complete task in time"
)
def test_run_tasks_nl_assertions_audio_native(
    domain_name: str, audio_native_config: AudioNativeConfig
):
    """Test running a task with NL assertions in audio-native mode.

    Equivalent of test_run_tasks_nl_assertions for full-duplex.
    """
    task = get_tasks(domain_name, task_ids=["create_task_1_nl_eval"])[0]
    simulation = run_task(
        domain=domain_name,
        task=task,
        agent="discrete_time_audio_native_agent",
        user="voice_streaming_user_simulator",
        llm_agent="gpt-4o-mini",
        llm_args_agent={},
        llm_user="gpt-4o-mini",
        llm_args_user={},
        evaluation_type=EvaluationType.NL_ASSERTIONS,
        audio_native_config=audio_native_config,
    )
    # Check that simulation ran and has expected structure
    assert len(simulation.get_messages()) > 0
    assert simulation.start_time is not None
    assert simulation.end_time is not None
    assert simulation.reward_info.reward == 1.0
    assert len(simulation.reward_info.nl_assertions) == 2
    assert simulation.reward_info.nl_assertions[0].met is True
    assert simulation.reward_info.nl_assertions[1].met is True

    # Add an nl_assertion that will fail and test that the reward is 0.0
    task.evaluation_criteria.nl_assertions.append("The user is not complimented")
    simulation = run_task(
        domain=domain_name,
        task=task,
        agent="discrete_time_audio_native_agent",
        user="voice_streaming_user_simulator",
        llm_agent="gpt-4o-mini",
        llm_args_agent={},
        llm_user="gpt-4o-mini",
        llm_args_user={},
        evaluation_type=EvaluationType.NL_ASSERTIONS,
        audio_native_config=audio_native_config,
    )
    assert simulation.reward_info.reward == 0.0
    assert len(simulation.reward_info.nl_assertions) == 3
    assert simulation.reward_info.nl_assertions[0].met is True
    assert simulation.reward_info.nl_assertions[1].met is True
    assert simulation.reward_info.nl_assertions[2].met is False


@pytest.mark.full_duplex_integration
def test_run_domain_audio_native(run_config_audio_native: RunConfig):
    """Test running a domain with audio-native mode.

    Equivalent of test_run_domain for full-duplex.
    """
    simulation_results = run_domain(run_config_audio_native)
    assert simulation_results is not None
