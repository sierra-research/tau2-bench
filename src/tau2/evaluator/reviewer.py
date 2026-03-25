"""
Conversation reviewer for analyzing simulation trajectories for errors.

This module provides functionality to review a single simulation using an
LLM judge to identify errors made by the agent and/or user simulator.

Two review modes are supported:
- "full": Review both agent and user simulator errors (also does auth classification)
- "user": Review only user simulator errors

Additionally, a focused hallucination check is available:
- check_hallucination(): Detects user simulator hallucinations (full-duplex only).
  This is used to gate reruns independently of the full review.

This is different from the evaluator which computes task success rewards/metrics.
The reviewer identifies qualitative conversation errors.

Usage:
    from tau2.evaluator.reviewer import review_simulation, ReviewMode, check_hallucination

    # Full review (agent + user errors)
    review, auth = review_simulation(simulation, task, ReviewMode.FULL, ...)
    simulation.review = review
    simulation.auth_classification = auth

    # User-only review
    review, _ = review_simulation(simulation, task, ReviewMode.USER, ...)
    simulation.user_only_review = review

    # Hallucination check (full-duplex only)
    hallucination_check = check_hallucination(simulation, task)
    simulation.hallucination_check = hallucination_check
"""

from enum import Enum
from typing import Optional, Union

from tau2.data_model.simulation import (
    AuthenticationClassification,
    HallucinationCheck,
    Review,
    SimulationRun,
    UserInfo,
    UserOnlyReview,
)
from tau2.data_model.tasks import Task
from tau2.evaluator.auth_classifier import (
    AuthenticationClassifier,
    FullDuplexAuthenticationClassifier,
)
from tau2.evaluator.hallucination_reviewer import FullDuplexHallucinationReviewer
from tau2.evaluator.review_llm_judge import (
    ConversationReviewer,
    FullDuplexConversationReviewer,
)
from tau2.evaluator.review_llm_judge_user_only import (
    FullDuplexUserOnlyReviewer,
    UserOnlyReviewer,
)


class ReviewMode(str, Enum):
    """Review mode."""

    FULL = "full"  # Review both agent and user errors
    USER = "user"  # Review only user simulator errors


def _is_full_duplex(simulation: SimulationRun) -> bool:
    """Check if the simulation used full-duplex mode (has ticks)."""
    return simulation.ticks is not None and len(simulation.ticks) > 0


def review_simulation(
    simulation: SimulationRun,
    task: Task,
    mode: ReviewMode,
    user_info: UserInfo,
    policy: Optional[str] = None,
    interruption_enabled: bool = False,
) -> tuple[Union[Review, UserOnlyReview], Optional[AuthenticationClassification]]:
    """
    Review a single simulation for conversation errors.

    Args:
        simulation: The simulation run to review.
        task: The task specification.
        mode: Review mode - FULL (agent+user) or USER (user only).
        user_info: User info containing simulation guidelines.
        policy: The policy the agent must follow (required for FULL mode).
        interruption_enabled: Whether interruption was enabled (for full-duplex).

    Returns:
        Tuple of (review_result, auth_classification).
        - For FULL mode: (Review, AuthenticationClassification)
        - For USER mode: (UserOnlyReview, None)
    """
    is_full_duplex = _is_full_duplex(simulation)

    if mode == ReviewMode.FULL:
        if not policy:
            raise ValueError("policy is required for FULL review mode")
        # Full review: agent + user errors + auth classification
        if is_full_duplex:
            review = FullDuplexConversationReviewer.review(
                user_info=user_info,
                task=task,
                full_trajectory=simulation.ticks,
                policy=policy,
                interruption_enabled=interruption_enabled,
            )
            auth_classification = FullDuplexAuthenticationClassifier.classify(
                ticks=simulation.ticks,
            )
        else:
            review = ConversationReviewer.review(
                user_info=user_info,
                task=task,
                full_trajectory=simulation.messages,
                policy=policy,
            )
            auth_classification = AuthenticationClassifier.classify(
                messages=simulation.messages,
            )
        return review, auth_classification

    else:  # ReviewMode.USER
        # User-only review: user simulator errors only
        if is_full_duplex:
            review = FullDuplexUserOnlyReviewer.review(
                user_info=user_info,
                task=task,
                full_trajectory=simulation.ticks,
                interruption_enabled=interruption_enabled,
            )
        else:
            review = UserOnlyReviewer.review(
                user_info=user_info,
                task=task,
                full_trajectory=simulation.messages,
            )
        return review, None


def check_hallucination(
    simulation: SimulationRun,
    task: Task,
) -> HallucinationCheck:
    """
    Check a simulation for user simulator hallucinations.

    This is a focused fact-check that only detects fabricated information.
    It does not analyze premature termination, interruption behavior, or
    other guideline violations. Full-duplex only.

    Args:
        simulation: The simulation run to check.
        task: The task specification.

    Returns:
        HallucinationCheck with any hallucinations found.

    Raises:
        ValueError: If the simulation is not full-duplex.
    """
    if not _is_full_duplex(simulation):
        raise ValueError(
            "Hallucination check is only supported for full-duplex simulations."
        )

    return FullDuplexHallucinationReviewer.review(
        task=task,
        full_trajectory=simulation.ticks,
    )


def format_hallucination_feedback(
    hallucination_check: HallucinationCheck,
) -> Optional[str]:
    """Format hallucination errors into feedback for the user simulator.

    Used by the hallucination retry loop: when a hallucination is detected,
    this function builds a feedback string that is appended to the user
    instructions on the next retry attempt.

    Args:
        hallucination_check: Result of check_hallucination().

    Returns:
        A string to append to user instructions, or None if no hallucinations.
    """
    if not hallucination_check.errors:
        return None

    error_lines = []
    for error in hallucination_check.errors:
        if error.correct_behavior:
            error_lines.append(f"- [hallucination] {error.correct_behavior}")
        else:
            error_lines.append(f"- [hallucination] {error.reasoning[:200]}")

    if not error_lines:
        return None

    return (
        "IMPORTANT: In a previous attempt at this conversation, the user simulator "
        "made these errors. You MUST avoid repeating them:\n" + "\n".join(error_lines)
    )
