"""
Persona configuration for user simulator behavior.

IMPORTANT: User behavior/persona is controlled in THREE places:
1. Global simulation guidelines (data/tau2/user_simulator/*.md) - Base behavior for all users
2. Task-specific persona (UserScenario.persona field) - Baked into task JSON at creation time
3. Runtime persona config (PersonaConfig, this file) - Configurable at simulation time

This allows for:
- Global defaults via guidelines
- Task-specific personas (e.g., "tech-savvy" vs "elderly confused user")
- Runtime variation (e.g., terseness level, interrupt tendency, quirks)

FUTURE: To support different persona attributes for text vs voice users, consider:
- Option A (Inheritance): VoicePersonaConfig(PersonaConfig) with voice-specific attrs (speech_quirks, accent), type-safe
- Option B (Single config): One PersonaConfig with optional mode-specific attrs, simpler but less type-safe
"""

import random
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class Verbosity(str, Enum):
    """How verbose the user is in their responses."""

    STANDARD = "standard"  # Normal conversational responses
    MINIMAL = "minimal"  # 1-2 word responses when sufficient


class InterruptTendency(str, Enum):
    """Whether the user waits for the agent to finish or can interrupt them."""

    WAITS = "waits"  # User waits for agent to complete before responding
    INTERRUPTS = "interrupts"  # User can interrupt agent while they're speaking


class PersonaConfig(BaseModel):
    """
    Runtime configuration for user simulator persona attributes.
    These settings control behavioral aspects that can be varied at simulation time.

    Default behavior (no persona config): Standard verbosity with normal conversational flow.
    """

    verbosity: Verbosity = Field(
        default=Verbosity.STANDARD,
        description="How verbose the user's responses are. Default: STANDARD",
    )

    interrupt_tendency: Optional[InterruptTendency] = Field(
        default=None,
        description="Whether user can interrupt the agent while they're speaking. Only applicable to streaming/voice users. None (default) means no interruption behavior configured.",
    )

    # Future attributes can be added here:
    # technical_skill: TechnicalSkill = TechnicalSkill.AVERAGE
    # speech_quirks: list[str] = []

    def to_guidelines_text(self) -> Optional[str]:
        """
        Convert persona config to additional guidelines text to append to system prompt.
        Returns None if no modifications needed (all defaults).
        """
        guidelines = []

        if self.verbosity == Verbosity.MINIMAL:
            guidelines.append(
                """
## MINIMAL VERBOSITY
You are terse in your responses.

- When a 1-2 word response is sufficient, respond with only those 1-2 words.
  Example: Agent: "Is this a round trip?" → You: "Yes" and NOT "Yes, it is a round trip."

- When a short phrase is sufficient, respond with the phrase instead of the full sentence.
  Example: Agent: "What is your city of origin and destination?" → You: "New York to Los Angeles" and NOT "I want to fly from New York to Los Angeles."

- Avoid filler words, pleasantries, or elaboration unless specifically needed.
  Example: Agent: "You're all set. Please let me know if you need anything else." → You: "Bye." and NOT "Thank you. That's all I needed."

- However, if this is a voice/audio call, you must still sound natural. Do not simply join multiple terse phrases in an unnatural way.
  Example: You should NOT say "Looking for wireless, noise-canceling, over-ear—black." Instead, say "I'm looking for wireless, noise-canceling over-ear headphones in black."
""".strip()
            )

        return "\n\n".join(guidelines) if guidelines else None

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "PersonaConfig":
        """Create a PersonaConfig from a dictionary with support for weighted random values.

        This method allows flexible specification of persona attributes:
        - Explicit values: {"verbosity": "minimal"}
        - Weighted random: {"verbosity": {"minimal": 0.8, "standard": 0.2}}

        Args:
            config: Dictionary mapping attribute names to values or randomization specs.

        Returns:
            PersonaConfig with values either specified or randomly selected.

        Examples:
            # Explicit value
            PersonaConfig.from_dict({"verbosity": "minimal"})

            # Weighted random (80% minimal, 20% standard)
            PersonaConfig.from_dict({"verbosity": {"minimal": 0.8, "standard": 0.2}})

            # Mixed: some explicit, some weighted random
            PersonaConfig.from_dict({
                "verbosity": "minimal",
                "interrupt_tendency": {"waits": 0.3, "interrupts": 0.7}
            })
        """
        resolved_config = {}

        # Generic processing for all fields
        for field_name, value in config.items():
            # Handle explicit value (string or other non-dict types)
            if not isinstance(value, dict):
                resolved_config[field_name] = value
            # Handle weighted random: {"option1": 0.8, "option2": 0.2}
            else:
                # Normalize probabilities to ensure they sum to 1.0
                total = sum(value.values())
                normalized = {k: v / total for k, v in value.items()}

                # Random selection based on weights
                rand_val = random.random()
                cumulative = 0.0
                for option, probability in normalized.items():
                    cumulative += probability
                    if rand_val < cumulative:
                        resolved_config[field_name] = option
                        break

        return cls(**resolved_config)
