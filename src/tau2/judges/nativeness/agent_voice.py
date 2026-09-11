# Copyright Sierra
"""Agent-role context clause for the nativeness judge."""

from typing import Literal, Optional, cast

from tau2.voice.voice_gender import resolve_agent_gender

ParticipantGender = Literal["male", "female"]


def normalize_participant_gender(value: Optional[str]) -> Optional[ParticipantGender]:
    """Validate recorded gender metadata without guessing or name inference."""
    normalized = (value or "").strip().lower()
    if normalized not in {"male", "female"}:
        return None
    return cast(ParticipantGender, normalized)


def agent_context(
    provider: Optional[str],
    voice: Optional[str] = None,
    caller_gender: Optional[str] = None,
) -> Optional[str]:
    """Known grammatical-gender context for the two speakers.

    The agent value comes only from the reviewed provider-voice catalog; the
    caller value comes only from recorded run/task context. Unknown values stay
    unknown rather than being inferred from names.
    """
    agent_gender = normalize_participant_gender(resolve_agent_gender(provider, voice))
    caller_gender = normalize_participant_gender(caller_gender)
    if not provider and not voice and not caller_gender:
        return None
    lines = ["The speaker is an AI customer-service voice agent."]
    if agent_gender:
        lines.append(
            f"Agent gender: {agent_gender}. Use this for the agent's first-person "
            "gender agreement."
        )
    if caller_gender:
        lines.append(
            f"Caller gender: {caller_gender}. Use this only when the agent directly "
            "addresses or refers to the caller with gender-marked language."
        )
    lines.append("Do not infer any missing gender from a name or stereotype.")
    return " ".join(lines)
