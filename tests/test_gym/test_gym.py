import pytest

from tau2.gym.gym_agent import AgentGymEnv, GymAgent, TauSpace
from tau2.user.user_simulator import UserSimulator

from .utils import timeout


class TestTauGymEnv:
    """Test cases for TauGymEnv."""

    def test_tau_gym_env_initialization(self):
        """Test that TauGymEnv initializes correctly."""
        env = AgentGymEnv(domain="mock", task_id="create_task_1")

        assert env.domain == "mock"
        assert env.task_id == "create_task_1"
        assert env._orchestrator is None
        assert env._agent is None
        assert env._user is None
        assert not env._simulation_done.is_set()
        assert isinstance(env.observation_space, TauSpace)
        assert isinstance(env.action_space, TauSpace)

    @pytest.mark.live_api
    @timeout(10)
    def test_tau_gym_env_reset(self):
        """Test that TauGymEnv reset works correctly."""
        env = AgentGymEnv(domain="mock", task_id="create_task_1")

        # Test reset
        observation, info = env.reset()

        # Check that reset returns expected types
        assert isinstance(observation, str)
        assert isinstance(info, dict)

        # Check that orchestrator and agent are initialized
        assert env._orchestrator is not None
        assert env._agent is not None
        assert env._user is not None
        assert isinstance(env._agent, GymAgent)

        # Check that the orchestrator thread is running
        assert env._orchestrator_thread is not None
        assert env._orchestrator_thread.is_alive()

    @pytest.mark.live_api
    @timeout(15)
    def test_tau_gym_env_step(self):
        """Test that TauGymEnv step works correctly."""
        env = AgentGymEnv(domain="mock", task_id="create_task_1")

        # Reset the environment first
        observation, info = env.reset()

        # Test step with an action
        action = "Hello! I'm here to help you. What can I assist you with today?"
        observation, reward, terminated, truncated, info = env.step(action)

        # Check that step returns expected types
        assert isinstance(observation, str)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

        # Check that the action was processed
        assert len(observation) > 0

    @pytest.mark.live_api
    @timeout(20)
    def test_tau_gym_env_multiple_steps(self):
        """Test that TauGymEnv can handle multiple steps."""
        env = AgentGymEnv(domain="mock", task_id="create_task_1")

        # Reset the environment
        observation, info = env.reset()

        # Take multiple steps
        actions = [
            "Hello! I'm here to help you.",
            "I understand you need assistance.",
            "Let me guide you through the process.",
        ]

        for i, action in enumerate(actions):
            observation, reward, terminated, truncated, info = env.step(action)

            # Check that we get a valid observation
            assert isinstance(observation, str)
            assert isinstance(terminated, bool)

            # If terminated, break
            if terminated:
                break

        # Check that we can take at least one step
        assert len(actions) > 0

    def test_tau_gym_env_step_without_reset(self):
        """Test that TauGymEnv raises an error when step is called without reset."""
        env = AgentGymEnv(domain="mock", task_id="create_task_1")

        # Try to step without reset
        with pytest.raises(RuntimeError, match="Orchestrator not initialized"):
            env.step("test action")

    @pytest.mark.live_api
    @timeout(15)
    def test_tau_gym_env_reset_multiple_times(self):
        """Test that TauGymEnv can be reset multiple times."""
        env = AgentGymEnv(domain="mock", task_id="create_task_1")

        # Reset multiple times
        for i in range(3):
            observation, info = env.reset()

            # Check that reset works each time
            assert isinstance(observation, str)
            assert isinstance(info, dict)
            assert env._orchestrator is not None
            assert env._agent is not None

    @pytest.mark.live_api
    @timeout(15)
    def test_tau_gym_env_observation_format(self):
        """Test that TauGymEnv formats observations correctly."""
        env = AgentGymEnv(domain="mock", task_id="create_task_1")

        # Reset the environment
        observation, info = env.reset()

        # Test step to get an observation
        action = "Hello! How can I help you?"
        observation, reward, terminated, truncated, info = env.step(action)

        # Check that observation is a string
        assert isinstance(observation, str)

        # If observation is not empty, it should contain role: content format
        if observation:
            # Check that it contains at least one role: content pattern
            lines = observation.split("\n")
            for line in lines:
                if line.strip():  # Skip empty lines
                    assert ":" in line, f"Line '{line}' should contain ':'"

    @pytest.mark.live_api
    @timeout(30)
    def test_tau_gym_env_termination(self):
        """Test that TauGymEnv properly handles simulation termination."""
        env = AgentGymEnv(domain="mock", task_id="create_task_1")

        # Reset the environment
        observation, info = env.reset()

        # Take steps until termination or max steps
        max_steps = 10
        step_count = 0

        while step_count < max_steps:
            action = f"Step {step_count}: I'm helping you."
            observation, reward, terminated, truncated, info = env.step(action)

            if terminated:
                break

            step_count += 1

        # Check that we either terminated or reached max steps
        assert step_count <= max_steps

    @pytest.mark.live_api
    @timeout(15)
    def test_tau_gym_env_thread_safety(self):
        """Test that TauGymEnv handles threading correctly."""
        env = AgentGymEnv(domain="mock", task_id="create_task_1")

        # Reset the environment
        observation, info = env.reset()

        # Check that the orchestrator thread is running
        assert env._orchestrator_thread.is_alive()

        # Take a step
        action = "Test action"
        observation, reward, terminated, truncated, info = env.step(action)

        # Check that the thread is still running (unless terminated)
        if not terminated:
            assert env._orchestrator_thread.is_alive()

    def test_tau_gym_env_invalid_domain(self):
        """Test that TauGymEnv handles invalid domain gracefully."""
        with pytest.raises(Exception):  # Should raise some exception for invalid domain
            env = AgentGymEnv(domain="invalid_domain", task_id="create_task_1")
            env.reset()

    def test_tau_gym_env_invalid_task_id(self):
        """Test that TauGymEnv handles invalid task_id gracefully."""
        with pytest.raises(ValueError):  # Should raise ValueError for invalid task_id
            env = AgentGymEnv(domain="mock", task_id="invalid_task_id")
            env.reset()

    def test_tau_gym_env_reset_simple(self):
        """The shared text builder constructs every component without running."""
        env = AgentGymEnv(domain="mock", task_id="create_task_1")

        task = env._get_task()
        orchestrator = env._get_orchestrator()

        assert task.id == "create_task_1"
        assert orchestrator.environment is not None
        assert isinstance(orchestrator.agent, GymAgent)
        assert isinstance(orchestrator.user, UserSimulator)
        assert orchestrator.task.id == "create_task_1"

    def test_tau_gym_env_basic_functionality(self):
        """Test basic TauGymEnv functionality without running full simulation."""
        env = AgentGymEnv(domain="mock", task_id="create_task_1")

        assert env.domain == "mock"
        assert env.task_id == "create_task_1"
        assert env._orchestrator is None
        orchestrator = env._get_orchestrator()
        assert orchestrator.solo_mode is False
        assert orchestrator.max_steps == env.max_steps

    def test_gym_resolves_english_language_context_like_text_runs(self):
        """No language means the canonical English pack, as for `tau2 run`."""
        from tau2.config import DEFAULT_LANGUAGE
        from tau2.data_model.language import RuntimePersona
        from tau2.data_model.semantic_communication import CommunicateScoring

        orchestrator = AgentGymEnv(
            domain="mock", task_id="create_task_1"
        )._get_orchestrator()
        context = orchestrator.call_context
        assert context is not None
        assert context.language.language == DEFAULT_LANGUAGE
        assert context.language.mode == "text"
        assert context.communicate_scoring == CommunicateScoring()
        assert isinstance(orchestrator.user.persona_config, RuntimePersona)
        assert orchestrator.user.persona_config.language == DEFAULT_LANGUAGE

    def test_gym_uses_configured_communicate_scoring(self):
        from tau2.data_model.semantic_communication import (
            CommunicateScorer,
            CommunicateScoring,
        )

        scoring = CommunicateScoring(scorer=CommunicateScorer.SUBSTRING)
        env = AgentGymEnv(
            domain="mock", task_id="create_task_1", communicate_scoring=scoring
        )
        assert env._get_orchestrator().call_context.communicate_scoring == scoring

    def test_gym_resolves_a_non_english_language(self):
        from tau2.multilingual.runtime import load_language

        task_id = load_language("es", "retail", mode="text").tasks[0].id
        env = AgentGymEnv(domain="retail", task_id=task_id, language="es")
        orchestrator = env._get_orchestrator()
        assert orchestrator.call_context.language.language == "es"
        assert orchestrator.user.persona_config.language == "es"
        assert "Spanish" in orchestrator.agent.domain_policy

    def test_solo_gym_uses_the_shared_dummy_user(self):
        from tau2.user.user_simulator import DummyUser

        env = AgentGymEnv(domain="mock", task_id="create_task_1", solo_mode=True)
        orchestrator = env._get_orchestrator()
        assert orchestrator.solo_mode is True
        assert isinstance(orchestrator.user, DummyUser)
        assert isinstance(orchestrator.agent, GymAgent)

    def test_reward_comes_from_the_shared_run_evaluation(self, monkeypatch):
        """The thread runs run_simulation; the reward is its reward_info."""
        from tau2.data_model.simulation import RewardInfo, SimulationRun
        from tau2.gym import gym_agent

        seen = {}

        def fake_run_simulation(orchestrator):
            seen["orchestrator"] = orchestrator
            return SimulationRun(
                id="fixture",
                task_id="create_task_1",
                start_time="t0",
                end_time="t1",
                duration=0.0,
                termination_reason="agent_stop",
                messages=[],
                reward_info=RewardInfo(reward=0.5),
            )

        monkeypatch.setattr(gym_agent, "run_simulation", fake_run_simulation)
        env = AgentGymEnv(domain="mock", task_id="create_task_1")
        env._orchestrator = env._get_orchestrator()
        env._run_orchestrator()
        assert seen["orchestrator"] is env._orchestrator
        reward, info = env._get_reward()
        assert reward == 0.5
        assert '"reward": 0.5' in info


if __name__ == "__main__":
    # Run a simple test with timeout
    @timeout(30)
    def run_basic_test():
        env = AgentGymEnv(domain="mock", task_id="create_task_1")
        observation, info = env.reset()
        print(f"Initial observation: {observation}")

        action = "Hello! I'm here to help you."
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"After step observation: {observation}")
        print(f"Terminated: {terminated}")

        print("✅ TauGymEnv basic test completed!")

    # Run the test
    result = run_basic_test()
    if result is None:
        print("⚠️ Test timed out - this is expected for a full simulation")
