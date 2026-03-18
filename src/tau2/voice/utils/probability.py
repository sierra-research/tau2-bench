# Copyright Sierra
"""Probability utilities for voice effects scheduling."""

import math
import random
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, computed_field


def poisson_should_trigger(
    rate_per_second: float, duration_seconds: float, rng: random.Random
) -> bool:
    """Determine if a Poisson-distributed event should trigger."""
    if rate_per_second <= 0:
        return False
    probability = 1.0 - math.exp(-rate_per_second * duration_seconds)
    return rng.random() < probability


class GEState(str, Enum):
    """Gilbert-Elliott model states (GOOD or BAD)."""

    GOOD = "good"
    BAD = "bad"


# Default parameters for Gilbert-Elliott model
GE_DEFAULT_GOOD_STATE_LOSS_PROB = 0.0  # No loss in good state
GE_DEFAULT_BAD_STATE_LOSS_PROB = 0.2  # 20% loss in bad state


class GilbertElliottConfig(BaseModel):
    """Configuration for Gilbert-Elliott bursty packet loss model."""

    target_loss_rate: float = Field(
        ge=0.0,
        lt=GE_DEFAULT_BAD_STATE_LOSS_PROB,
        description="Target average loss rate (0.0 to 0.2), e.g. 0.02 for 2%",
    )
    avg_burst_duration_ms: float = Field(
        gt=0.0,
        description="Average duration of burst (Bad state) in milliseconds",
    )
    good_state_loss_prob: float = Field(
        default=GE_DEFAULT_GOOD_STATE_LOSS_PROB,
        ge=0.0,
        le=1.0,
        description="Loss probability in Good state (k)",
    )
    bad_state_loss_prob: float = Field(
        default=GE_DEFAULT_BAD_STATE_LOSS_PROB,
        gt=0.0,
        le=1.0,
        description="Loss probability in Bad state (h)",
    )

    @computed_field
    @property
    def r_rate(self) -> float:
        """Transition rate from Bad to Good state (B->G)."""
        avg_burst_duration_sec = self.avg_burst_duration_ms / 1000.0
        return 1.0 / avg_burst_duration_sec

    @computed_field
    @property
    def p_rate(self) -> float:
        """Transition rate from Good to Bad state (G->B)."""
        h = self.bad_state_loss_prob
        return self.r_rate * self.target_loss_rate / (h - self.target_loss_rate)

    @computed_field
    @property
    def steady_state_bad_prob(self) -> float:
        """Steady-state probability of being in Bad state (π_B)."""
        return self.p_rate / (self.p_rate + self.r_rate)


class GilbertElliottModel:
    """Gilbert-Elliott model for bursty packet loss simulation."""

    def __init__(
        self,
        target_loss_rate: float,
        avg_burst_duration_ms: float,
        rng: random.Random,
        initial_state: Literal["good", "bad"] = "good",
        good_state_loss_prob: float = GE_DEFAULT_GOOD_STATE_LOSS_PROB,
        bad_state_loss_prob: float = GE_DEFAULT_BAD_STATE_LOSS_PROB,
    ):
        """Initialize the Gilbert-Elliott model."""
        # Validate and store configuration
        self.config = GilbertElliottConfig(
            target_loss_rate=target_loss_rate,
            avg_burst_duration_ms=avg_burst_duration_ms,
            good_state_loss_prob=good_state_loss_prob,
            bad_state_loss_prob=bad_state_loss_prob,
        )

        self.rng = rng
        self.state = GEState.GOOD if initial_state == "good" else GEState.BAD

    def should_drop(self, duration_seconds: float) -> bool:
        """Check if a frame should be dropped for this time chunk."""
        # Handle state transitions using continuous-time approximation
        self._update_state(duration_seconds)

        # Determine if drop occurs based on current state
        loss_prob = (
            self.config.good_state_loss_prob
            if self.state == GEState.GOOD
            else self.config.bad_state_loss_prob
        )
        return self.rng.random() < loss_prob

    def _update_state(self, duration_seconds: float) -> None:
        """Update internal state based on transition probabilities."""
        if self.state == GEState.GOOD:
            # Check for Good -> Bad transition
            p_transition = 1.0 - math.exp(-self.config.p_rate * duration_seconds)
            if self.rng.random() < p_transition:
                self.state = GEState.BAD
        else:
            # Check for Bad -> Good transition
            p_transition = 1.0 - math.exp(-self.config.r_rate * duration_seconds)
            if self.rng.random() < p_transition:
                self.state = GEState.GOOD

    def reset(self, state: Literal["good", "bad"] = "good") -> None:
        """Reset the model to a specific state."""
        self.state = GEState.GOOD if state == "good" else GEState.BAD

    @property
    def is_in_bad_state(self) -> bool:
        """Check if currently in Bad (burst) state."""
        return self.state == GEState.BAD

    @property
    def target_loss_rate(self) -> float:
        """Target average loss rate from config."""
        return self.config.target_loss_rate

    @property
    def avg_burst_duration_ms(self) -> float:
        """Average burst duration in ms from config."""
        return self.config.avg_burst_duration_ms
