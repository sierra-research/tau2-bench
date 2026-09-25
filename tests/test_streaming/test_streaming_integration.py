import time
from types import SimpleNamespace

import pytest

from tau2 import LLMAgent, Orchestrator, UserSimulator
from tau2.data_model.audio import audio_bytes_to_string
from tau2.data_model.message import AssistantMessage, Tick, UserMessage
from tau2.data_model.simulation import TerminationReason
from tau2.orchestrator.full_duplex_orchestrator import FullDuplexOrchestrator
from tau2.orchestrator.modes import CommunicationMode
from tau2.utils.utils import get_now


class TestBackwardCompatibility:
    """Test that existing code continues to work unchanged."""

    def test_orchestrator_defaults_to_half_duplex(self, mock_setup):
        """Verify orchestrator defaults to HALF_DUPLEX mode for backward compatibility."""
        agent, user, env, task = mock_setup

        orchestrator = Orchestrator(
            domain="mock",
            agent=agent,
            user=user,
            environment=env,
            task=task,
        )

        assert orchestrator.mode == CommunicationMode.HALF_DUPLEX

    def test_existing_message_creation_works(self):
        """Verify old message creation still works."""
        # Old code should work unchanged
        msg = UserMessage(role="user", content="Hello")
        assert msg.content == "Hello"
        assert msg.role == "user"

        # New fields should have defaults
        assert msg.chunk_id is None
        assert msg.is_final_chunk

    def test_message_has_text_content_still_works(self):
        """Verify has_text_content() method still works as before."""
        msg1 = UserMessage(role="user", content="Hello")
        assert msg1.has_text_content()

        msg2 = UserMessage(role="user", content="")
        assert not msg2.has_text_content()

        msg3 = UserMessage(role="user", content=None)
        assert not msg3.has_text_content()


class TestCommunicationModes:
    """Test communication mode selection and validation."""

    def test_half_duplex_mode_with_regular_agent(self, mock_setup):
        """Regular agents work in HALF_DUPLEX mode."""
        agent, user, env, task = mock_setup

        orchestrator = Orchestrator(
            domain="mock",
            agent=agent,
            user=user,
            environment=env,
            task=task,
        )

        assert orchestrator.mode == CommunicationMode.HALF_DUPLEX

    def test_full_duplex_requires_streaming_agent(self, mock_setup):
        """FullDuplexOrchestrator requires streaming-capable agent."""
        agent, user, env, task = mock_setup

        with pytest.raises(ValueError, match="get_next_chunk"):
            FullDuplexOrchestrator(
                domain="mock",
                agent=agent,  # Regular agent doesn't have get_next_chunk
                user=user,
                environment=env,
                task=task,
            )

    def test_full_duplex_requires_streaming_user(self, mock_agent_setup):
        """FullDuplexOrchestrator requires streaming-capable user.

        Note: The full version of this test (using TextStreamingLLMAgent) is in
        src/experiments/tau_voice/tests/test_streaming_integration.py.
        Here we verify the validation using a minimal mock with get_next_chunk.
        """
        from tau2.registry import registry
        from tau2.run import get_tasks

        tools, domain_policy = mock_agent_setup

        # Create regular (non-streaming) user
        user = UserSimulator(
            instructions="Test user",
            llm="gpt-4",
        )

        # Create a minimal mock that satisfies FullDuplexAgent's interface
        class _MockStreamingAgent:
            def get_next_chunk(self, state, chunk):
                pass

            def get_init_state(self, message_history=None):
                pass

        env_constructor = registry.get_env_constructor("mock")
        env = env_constructor()
        tasks = get_tasks("mock", task_ids=["create_task_1"])
        task = tasks[0]

        with pytest.raises(ValueError, match="get_next_chunk"):
            FullDuplexOrchestrator(
                domain="mock",
                agent=_MockStreamingAgent(),
                user=user,  # Regular user doesn't have get_next_chunk
                environment=env,
                task=task,
            )


class TestMessageEnhancements:
    """Test message model enhancements for streaming."""

    def test_message_chunking_fields(self):
        """Test chunk-related fields."""
        chunk = AssistantMessage(
            role="assistant",
            content="First chunk",
            chunk_id=0,
            is_final_chunk=False,
        )

        assert chunk.chunk_id == 0
        assert not chunk.is_final_chunk


class TestFullDuplexAudioDrain:
    """Test final buffered audio handling for full-duplex runs."""

    def test_drain_buffered_agent_audio_appends_to_last_tick(self):
        orchestrator = FullDuplexOrchestrator.__new__(FullDuplexOrchestrator)
        orchestrator.agent = SimpleNamespace(
            adapter=SimpleNamespace(
                _buffered_agent_audio=[(b" tail", "item_1"), (b" audio", "item_1")]
            ),
            audio_format=None,
        )
        orchestrator.ticks = [
            Tick(
                tick_id=0,
                timestamp=get_now(),
                agent_chunk=AssistantMessage(
                    role="assistant",
                    audio_content=audio_bytes_to_string(b"heard"),
                    contains_speech=True,
                ),
            )
        ]

        orchestrator._drain_buffered_agent_audio()

        assert (
            orchestrator.ticks[-1].agent_chunk.get_audio_bytes() == b"heard tail audio"
        )
        assert orchestrator.agent.adapter._buffered_agent_audio == []
        assert orchestrator.ticks[-1].agent_chunk.contains_speech

    def test_finalize_drains_buffer_before_agent_stop_clears_it(self):
        class Agent:
            def __init__(self):
                self.adapter = SimpleNamespace(
                    _buffered_agent_audio=[(b"final words", "item_1")]
                )
                self.audio_format = None
                self.saw_buffer_during_stop = None

            def stop(self, *args, **kwargs):
                self.saw_buffer_during_stop = list(self.adapter._buffered_agent_audio)
                self.adapter._buffered_agent_audio.clear()

        agent = Agent()
        orchestrator = FullDuplexOrchestrator.__new__(FullDuplexOrchestrator)
        orchestrator.agent = agent
        orchestrator.user = SimpleNamespace(stop=lambda *args, **kwargs: None)
        orchestrator.agent_state = object()
        orchestrator.user_state = object()
        orchestrator.pending_agent_tool_results = None
        orchestrator.pending_user_tool_results = None
        orchestrator._run_start_perf = time.perf_counter()
        orchestrator._run_start_time = get_now()
        orchestrator.simulation_id = "sim"
        orchestrator.task = SimpleNamespace(id="task")
        orchestrator.termination_reason = TerminationReason.USER_STOP
        orchestrator.seed = None
        orchestrator.mode = CommunicationMode.FULL_DUPLEX
        orchestrator.ticks = [Tick(tick_id=0, timestamp=get_now())]

        simulation_run = orchestrator._finalize()

        assert agent.saw_buffer_during_stop == []
        assert simulation_run.ticks[-1].agent_chunk.get_audio_bytes() == b"final words"
        assert agent.adapter._buffered_agent_audio == []


# Fixtures


@pytest.fixture
def mock_agent_setup():
    """Setup basic agent requirements."""
    from tau2.registry import registry

    # Get environment to access tools
    env_constructor = registry.get_env_constructor("mock")
    env = env_constructor()
    tools = env.get_tools()

    # Get a minimal domain policy
    domain_policy = "Mock domain for testing"

    return tools, domain_policy


@pytest.fixture
def mock_setup(mock_agent_setup):
    """Setup basic orchestrator components."""
    from tau2.registry import registry
    from tau2.run import get_tasks

    tools, domain_policy = mock_agent_setup

    agent = LLMAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm="gpt-4",
    )

    user = UserSimulator(
        instructions="Test user",
        llm="gpt-4",
    )

    env_constructor = registry.get_env_constructor("mock")
    env = env_constructor()

    # Use an actual task from the mock domain
    tasks = get_tasks("mock", task_ids=["create_task_1"])
    task = tasks[0]

    return agent, user, env, task
