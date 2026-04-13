"""
Phase 5: Adversarial Simulator Wrapper.

A proxy around UserSimulator that injects stochastic perturbations to test
agent robustness. Uses the Proxy Pattern (composition) so core tau2-bench
code is never modified.

Usage:
    from tau2.user.user_simulator import UserSimulator
    from experiments.tau2_trace.adversarial_wrapper import AdversarialSimulatorWrapper

    base_sim = UserSimulator(tools=tools, instructions=instructions, llm=llm)
    wrapped = AdversarialSimulatorWrapper(base_sim, perturbation_rate=0.2, seed=42)
    # Pass `wrapped` to the Orchestrator in place of `base_sim`
"""

from __future__ import annotations

import random
from typing import Literal, Optional, Tuple

from pydantic import BaseModel

from tau2.data_model.message import Message, UserMessage
from tau2.user.user_simulator_base import (
    HalfDuplexUser,
    UserState,
    ValidUserInputMessage,
)

PerturbationKind = Literal[
    "interruption", "self_correction_setup", "self_correction_delivery"
]


class PerturbationEvent(BaseModel):
    """Record of a perturbation that was applied."""

    turn: int
    kind: PerturbationKind
    original_content: Optional[str] = None
    injected_content: Optional[str] = None


INTERRUPTION_TEMPLATES = [
    "Wait, before we continue -- is this going to cost me anything extra?",
    "Hold on, my screen just went dark for a second. OK it's back now. What was I supposed to do?",
    "Actually, can we hurry this up? I'm running late for something.",
    "Sorry, I got distracted. Can you repeat what you just said?",
    "One sec -- I have another call coming in. OK I'm back, go ahead.",
]

SELF_CORRECTION_TEMPLATES = [
    "Oh wait, I think I read that wrong. Let me check again.",
    "Actually, sorry -- I was looking at the wrong thing earlier.",
    "Hmm, that doesn't seem right. Let me look at it more carefully.",
]


class AdversarialSimulatorWrapper(HalfDuplexUser):
    """
    Wraps a UserSimulator and injects perturbations at configurable rates.

    Matches the exact interface expected by the Orchestrator:
        generate_next_message(message, state) -> (UserMessage, UserState)
        get_init_state(message_history) -> UserState
        is_stop(message) -> bool

    The perturbation types:
        - Interruption: replaces the user response with an off-topic question.
          The base simulator still advances its internal state, but the agent
          sees an interruption instead. Next turn proceeds normally.
        - Self-correction: lets the base response through but schedules a
          correction message on the following turn that retracts/modifies it.
    """

    def __init__(
        self,
        base_simulator: HalfDuplexUser,
        perturbation_rate: float = 0.20,
        seed: int = 42,
    ):
        super().__init__(
            instructions=base_simulator.instructions,
            tools=base_simulator.tools,
        )
        self.base_simulator = base_simulator
        self.perturbation_rate = perturbation_rate
        self.rng = random.Random(seed)

        self._pending_correction: Optional[str] = None
        self._turn_counter: int = 0
        self.perturbation_log: list[PerturbationEvent] = []

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> UserState:
        return self.base_simulator.get_init_state(message_history)

    @classmethod
    def is_stop(cls, message: UserMessage) -> bool:
        from tau2.user.user_simulator import UserSimulator

        return UserSimulator.is_stop(message)

    def generate_next_message(
        self, message: ValidUserInputMessage, state: UserState
    ) -> Tuple[UserMessage, UserState]:
        """
        Main interception point. The orchestrator calls this exactly like
        it would call the base UserSimulator.
        """
        self._turn_counter += 1

        # Resolve pending self-correction from previous turn
        if self._pending_correction is not None:
            correction_text = self._pending_correction
            self._pending_correction = None

            # Still step the base simulator to keep its state consistent
            base_msg, state = self.base_simulator.generate_next_message(message, state)

            corrected_msg = UserMessage(
                role="user",
                content=correction_text,
            )
            # Replace the last message in state with our correction
            if state.messages and len(state.messages) > 0:
                state.messages[-1] = corrected_msg

            self.perturbation_log.append(
                PerturbationEvent(
                    turn=self._turn_counter,
                    kind="self_correction_delivery",
                    original_content=base_msg.content,
                    injected_content=correction_text,
                )
            )
            return corrected_msg, state

        # Get the ground truth response from the base simulator
        base_msg, state = self.base_simulator.generate_next_message(message, state)

        # Don't perturb stop/transfer messages
        if self.is_stop(base_msg):
            return base_msg, state

        # Don't perturb tool calls (they must go through correctly)
        if base_msg.is_tool_call():
            return base_msg, state

        # Roll for perturbation
        if self.rng.random() < self.perturbation_rate:
            kind = self.rng.choice(["interruption", "self_correction"])

            if kind == "interruption":
                injected = self.rng.choice(INTERRUPTION_TEMPLATES)
                perturbed_msg = UserMessage(role="user", content=injected)

                if state.messages and len(state.messages) > 0:
                    state.messages[-1] = perturbed_msg

                self.perturbation_log.append(
                    PerturbationEvent(
                        turn=self._turn_counter,
                        kind="interruption",
                        original_content=base_msg.content,
                        injected_content=injected,
                    )
                )
                return perturbed_msg, state

            elif kind == "self_correction":
                # Schedule a correction for the next turn
                correction = self.rng.choice(SELF_CORRECTION_TEMPLATES)
                if base_msg.content:
                    correction += f" The correct info is: {base_msg.content}"
                self._pending_correction = correction

                self.perturbation_log.append(
                    PerturbationEvent(
                        turn=self._turn_counter,
                        kind="self_correction_setup",
                        original_content=base_msg.content,
                        injected_content=None,
                    )
                )
                return base_msg, state

        # Happy path: no perturbation
        return base_msg, state

    def set_seed(self, seed: int) -> None:
        """Forward seed to the base simulator (required by Orchestrator)."""
        if hasattr(self.base_simulator, "set_seed"):
            self.base_simulator.set_seed(seed)

    def reset(self) -> None:
        """Reset perturbation state for a new simulation."""
        self._pending_correction = None
        self._turn_counter = 0
        self.perturbation_log.clear()
