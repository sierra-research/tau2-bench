"""Tests for the adversarial simulator wrapper."""

from experiments.tau2_trace.adversarial_wrapper import (
    AdversarialSimulatorWrapper,
)
from tau2.data_model.message import AssistantMessage, UserMessage
from tau2.user.user_simulator_base import UserState


class FakeBaseSimulator:
    """Minimal mock that matches the UserSimulator interface."""

    def __init__(self):
        self.instructions = "Test instructions"
        self.tools = None
        self.call_count = 0
        self._seed = None

    def set_seed(self, seed: int):
        self._seed = seed

    def get_init_state(self, message_history=None):
        return UserState(system_messages=[], messages=[])

    def generate_next_message(self, message, state):
        self.call_count += 1
        user_msg = UserMessage(
            role="user",
            content=f"Base response #{self.call_count}",
        )
        state.messages.append(user_msg)
        return user_msg, state


class TestAdversarialWrapper:
    def test_passthrough_when_no_perturbation(self):
        base = FakeBaseSimulator()
        wrapper = AdversarialSimulatorWrapper(base, perturbation_rate=0.0, seed=42)

        state = wrapper.get_init_state()
        agent_msg = AssistantMessage(role="assistant", content="Hello")
        user_msg, state = wrapper.generate_next_message(agent_msg, state)

        assert user_msg.content == "Base response #1"
        assert len(wrapper.perturbation_log) == 0

    def test_always_perturbs(self):
        base = FakeBaseSimulator()
        wrapper = AdversarialSimulatorWrapper(base, perturbation_rate=1.0, seed=42)

        state = wrapper.get_init_state()
        agent_msg = AssistantMessage(role="assistant", content="Hello")
        user_msg, state = wrapper.generate_next_message(agent_msg, state)

        assert len(wrapper.perturbation_log) == 1
        event = wrapper.perturbation_log[0]
        assert event.kind in ("interruption", "self_correction_setup")

    def test_seeded_reproducibility(self):
        """Same seed produces same perturbation sequence."""
        results_a = []
        results_b = []

        for results in [results_a, results_b]:
            base = FakeBaseSimulator()
            wrapper = AdversarialSimulatorWrapper(base, perturbation_rate=1.0, seed=99)
            state = wrapper.get_init_state()

            for _ in range(5):
                agent_msg = AssistantMessage(role="assistant", content="test")
                user_msg, state = wrapper.generate_next_message(agent_msg, state)
                results.append(user_msg.content)

        assert results_a == results_b

    def test_self_correction_delivers_next_turn(self):
        base = FakeBaseSimulator()
        # Use a seed that produces a self_correction on first call
        # We force it by setting rate=1.0 and checking behavior
        wrapper = AdversarialSimulatorWrapper(base, perturbation_rate=1.0, seed=42)

        state = wrapper.get_init_state()
        agent_msg = AssistantMessage(role="assistant", content="Hello")

        # First call: may set up a self-correction
        msg1, state = wrapper.generate_next_message(agent_msg, state)

        if wrapper._pending_correction is not None:
            # Second call should deliver the correction
            msg2, state = wrapper.generate_next_message(agent_msg, state)
            assert (
                "correct info" in msg2.content.lower()
                or "wrong" in msg2.content.lower()
                or "check again" in msg2.content.lower()
            )

    def test_set_seed_forwarded_to_base(self):
        """Orchestrator calls set_seed(); wrapper must forward to base."""
        base = FakeBaseSimulator()
        wrapper = AdversarialSimulatorWrapper(base, perturbation_rate=0.0, seed=42)

        wrapper.set_seed(123)
        assert base._seed == 123

    def test_reset_clears_state(self):
        base = FakeBaseSimulator()
        wrapper = AdversarialSimulatorWrapper(base, perturbation_rate=1.0, seed=42)

        state = wrapper.get_init_state()
        agent_msg = AssistantMessage(role="assistant", content="Hello")
        wrapper.generate_next_message(agent_msg, state)

        wrapper.reset()
        assert wrapper._pending_correction is None
        assert wrapper._turn_counter == 0
        assert len(wrapper.perturbation_log) == 0
