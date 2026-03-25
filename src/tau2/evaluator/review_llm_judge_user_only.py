import json
from copy import deepcopy

from tau2.config import DEFAULT_LLM_EVAL_USER_SIMULATOR
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    Tick,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import UserInfo, UserOnlyReview, UserOnlyReviewError
from tau2.data_model.tasks import Task
from tau2.utils.display import MarkdownDisplay
from tau2.utils.llm_utils import extract_json_from_llm_response, generate

# =============================================================================
# Prompts for User Simulator Review
# =============================================================================

SYSTEM_PROMPT = """
# Goal
Assess this conversation to determine if the user simulator made any errors during the interaction.

Here are the general guidelines given to the user simulator:
<User Simulator Guidelines>
{user_guidelines}
</User Simulator Guidelines>

# Inputs
- <User Instructions>: Instructions for the user simulator to follow during this specific interaction.
- <Conversation>: The full conversation between the user simulator and the agent.

# Instructions
Read the <User Instructions> and <Conversation>.
Review the entire conversation and identify ALL user messages where the user simulator made an error.

A user simulator error occurs when a user message:
- Does not follow the <User Simulator Guidelines> or <User Instructions>.
- Is not a natural or correct continuation of the conversation.
- Provides incorrect information that contradicts the <User Instructions>.
- Reveals information the user should not know according to the instructions.

## Important Review Principles

1. **Fact-check every user claim**: For every factual detail the user provides (names, emails, zip codes, sizes, colors, product descriptions, etc.), verify it appears in or is derivable from the <User Instructions>. Any detail not grounded in the instructions is a hallucination — even if it sounds plausible. When the user lacks information, the correct behavior is to say "I don't know" or ask the agent.

2. **Do not blame the user for agent failures**: If the agent is unresponsive, repeatedly fails, or makes critical errors, the user giving up or ending the conversation is a reasonable reaction — not a user error. Only flag premature_termination when the agent was actively working and making progress.


## Error Severity
For each user error, classify its severity:
- **critical_helped**: The user error helped the agent succeed inappropriately (e.g., user provided information they shouldn't have, making the task too easy).
- **critical_hindered**: The user error hindered the agent or made the task harder/impossible (e.g., user provided incorrect information or contradicted instructions).
- **minor**: The user made an error but it did NOT influence the simulation outcome.

## Error Tags
For each error, assign one or more tags from the following list:
- **hallucination**: Provided factual details (e.g., zip codes, sizes, product descriptions) not present in or derivable from the user instructions — even if plausible.
- **incorrect_interpretation**: Misinterpreted available information (e.g., misunderstood agent's message or response).
- **guideline_violation**: Message not consistent with the User Simulator Guidelines or User Instructions.
- **revealed_info_early**: Shared information before it was appropriate or before proper verification.
- **inconsistent_behavior**: Statement contradicts earlier statements or actions in the conversation.
- **premature_termination**: Ended the conversation or accepted an incomplete outcome while the agent was actively working and making progress. Do NOT use this tag if the user ended the conversation because the agent was unresponsive or repeatedly failing.
- **missed_required_action**: Did not take a required action that was expected.
- **wrong_sequence**: Performed actions out of the expected order or sequence.
- **other**: Use only when no other tag applies. Include a description of the error type in the reasoning.

## Workflow
Follow these steps to produce your analysis:

1. **Fact-check user claims**: Verify every factual claim the user makes against the <User Instructions>.

2. **Analyze each user message**: Go through the conversation message by message. For each user message, check if it contains an error based on the guidelines above. Note any errors you find.

3. **Assess context for termination**: If the user ended the conversation early, only flag premature_termination if the agent was actively making progress (not stalled or failing).

4. **Summarize**: Summarize what happened at the conversation level, including what errors (if any) affected the outcome.

5. **Format output**: Compile your findings into the expected JSON format. Include only the messages where errors were found (discard messages with no errors from the errors list).

# Output
Structure your answer in the following JSON format:
```json
{{
    "errors": [
        {{
            "turn_idx": <turn number where error occurred>,
            "error_tags": ["<tag1>", "<tag2>", ...],
            "severity": "minor" or "critical_helped" or "critical_hindered",
            "reasoning": "<explanation of why this is an error>",
            "user_message": "<the problematic user message content>",
            "correct_behavior": "<what the user should have said or done instead>"
        }}
    ],
    "summary": "<brief summary of the review>"
}}
```

- "errors": List of errors found. Empty list if no errors.
- "summary": A brief summary of the conversation review highlighting key findings.

Return ONLY the JSON object, no additional text.
""".strip()

USER_PROMPT = """
<User Instructions>
{user_instructions}
</User Instructions>

<Conversation>
{conversation}
</Conversation>
""".strip()


# =============================================================================
# Full-Duplex Prompts (with Interruption Policy)
# =============================================================================

FULL_DUPLEX_SYSTEM_PROMPT = """
# Goal
Assess this full-duplex conversation to determine if the user simulator made any errors during the interaction.
This includes both content errors AND turn-taking/interruption behavior errors.

Here are the general guidelines given to the user simulator:
<User Simulator Guidelines>
{user_guidelines}
</User Simulator Guidelines>

## Conversation Format
The conversation is displayed in a consolidated format where:
- Consecutive speech from the same speaker is grouped together into segments
- Each segment shows **Agent**: or **User**: followed by the speech content
- Because this is full-duplex, both parties may speak in overlapping segments
- Agent tool calls are not shown (internal to the agent)

# Inputs
- <User Instructions>: Instructions for the user simulator to follow during this specific interaction.
- <Conversation>: The full conversation between the user simulator and the agent.
- <Interruption Policy>: Whether the user simulator was configured to interrupt.

# Instructions
Review the entire conversation. Identify ALL errors where the user simulator:

## Content Errors:
- Does not follow the <User Simulator Guidelines> or <User Instructions>.
- Is not a natural or correct continuation of the conversation.
- Provides incorrect information that contradicts the <User Instructions>.
- Reveals information the user should not know according to the instructions.

## Important Review Principles

1. **Fact-check every user claim**: For every factual detail the user provides (names, emails, zip codes, sizes, colors, product descriptions, etc.), verify it appears in or is derivable from the <User Instructions>. Any detail not grounded in the instructions is a hallucination — even if it sounds plausible. When the user lacks information, the correct behavior is to say "I don't know" or ask the agent.

2. **Do not blame the user for agent failures**: If the agent is unresponsive, repeatedly fails, or makes critical errors, the user giving up or ending the conversation is a reasonable reaction — not a user error. Only flag premature_termination when the agent was actively working and making progress.


## Turn-Taking/Interruption Errors (only if interruption is enabled):
- Interrupts the agent too frequently or unnecessarily.
- Interrupts the agent when they have only spoken a few words (less than ~5 words).
- Fails to interrupt when the agent has clearly finished their main point and is rambling.
- Interrupts at inappropriate moments (e.g., mid-sentence when important info is being conveyed).
- Does NOT interrupt when the user has heard enough to respond and continuing to listen adds no value.

Note: If interruption is disabled, do not flag any interruption-related errors.

## Error Severity
For each user error, classify its severity:
- **critical_helped**: The user error helped the agent succeed inappropriately (e.g., user provided information they shouldn't have, making the task too easy).
- **critical_hindered**: The user error hindered the agent or made the task harder/impossible (e.g., user provided incorrect information or contradicted instructions).
- **minor**: The user made an error but it did NOT influence the simulation outcome.

## Error Tags
For each error, assign one or more tags from the following list:
- **hallucination**: Provided factual details (e.g., zip codes, sizes, product descriptions) not present in or derivable from the user instructions — even if plausible.
- **incorrect_interpretation**: Misinterpreted available information (e.g., misunderstood agent's message or response).
- **guideline_violation**: Message not consistent with the User Simulator Guidelines or User Instructions.
- **revealed_info_early**: Shared information before it was appropriate or before proper verification.
- **inconsistent_behavior**: Statement contradicts earlier statements or actions in the conversation.
- **premature_termination**: Ended the conversation or accepted an incomplete outcome while the agent was actively working and making progress. Do NOT use this tag if the user ended the conversation because the agent was unresponsive or repeatedly failing.
- **missed_required_action**: Did not take a required action that was expected.
- **wrong_sequence**: Performed actions out of the expected order or sequence.
- **interruption_error**: Interrupted inappropriately or failed to interrupt when appropriate (only for full-duplex with interruption enabled).
- **other**: Use only when no other tag applies. Include a description of the error type in the reasoning.

## Workflow
Follow these steps to produce your analysis:

1. **Fact-check user claims**: Verify every factual claim the user makes against the <User Instructions>.

2. **Analyze each user segment**: Go through the conversation segment by segment. For each user segment, check if it contains an error based on the guidelines above. Note any errors you find.

3. **Assess context for termination**: If the user ended the conversation early, only flag premature_termination if the agent was actively making progress (not stalled or failing).

4. **Summarize**: Summarize what happened at the conversation level, including what errors (if any) affected the outcome.

5. **Format output**: Compile your findings into the expected JSON format. Include only the segments where errors were found (discard segments with no errors from the errors list).

# Output
Structure your answer in the following JSON format:
```json
{{
    "errors": [
        {{
            "tick_start": <start tick of the segment where error occurred>,
            "tick_end": <end tick of the segment where error occurred>,
            "error_type": "content_error" or "interruption_error",
            "error_tags": ["<tag1>", "<tag2>", ...],
            "severity": "minor" or "critical_helped" or "critical_hindered",
            "reasoning": "<explanation of why this is an error>",
            "user_message": "<the problematic user message content>",
            "correct_behavior": "<what the user should have said or done instead>"
        }}
    ],
    "summary": "<brief summary of the review>"
}}
```

- "errors": List of errors found. Empty list if no errors.
- "summary": A brief summary of the conversation review highlighting key findings.

Return ONLY the JSON object, no additional text.
""".strip()


FULL_DUPLEX_USER_PROMPT = """
<User Instructions>
{user_instructions}
</User Instructions>

<Interruption Policy>
Interruption enabled: {interruption_enabled}
</Interruption Policy>

<Conversation>
{conversation}
</Conversation>
""".strip()


# =============================================================================
# Parsing Helpers
# =============================================================================


def _extract_json_from_response(response: str) -> str:
    """Extract JSON from LLM response. Delegates to shared utility."""
    return extract_json_from_llm_response(response)


def _parse_user_only_review_response(
    response: str,
) -> tuple[str, bool, bool, bool, list[UserOnlyReviewError]]:
    """
    Parse the LLM response in JSON format.

    Returns:
        Tuple of (summary, user_error, critical_user_error, has_errors, list of UserOnlyReviewError)
    """
    json_str = _extract_json_from_response(response)
    result_data = json.loads(json_str)
    summary = result_data.get("summary", "")

    errors: list[UserOnlyReviewError] = []
    for error_data in result_data.get("errors", []):
        # Parse severity
        severity = None
        severity_str = error_data.get("severity")
        if severity_str in ("minor", "critical_helped", "critical_hindered"):
            severity = severity_str

        # Parse error_tags (must be a list)
        error_tags = error_data.get("error_tags", [])
        if isinstance(error_tags, str):
            error_tags = [error_tags]

        errors.append(
            UserOnlyReviewError(
                turn_idx=error_data.get("turn_idx"),
                tick_start=error_data.get("tick_start"),
                tick_end=error_data.get("tick_end"),
                error_type=error_data.get("error_type", "content_error"),
                error_tags=error_tags,
                severity=severity,
                reasoning=error_data.get("reasoning", ""),
                user_message=error_data.get("user_message"),
                correct_behavior=error_data.get("correct_behavior"),
            )
        )

    # Calculate boolean flags from errors list
    user_error = len(errors) > 0
    critical_user_error = any(
        e.severity in ("critical_helped", "critical_hindered") for e in errors
    )
    has_errors = len(errors) > 0

    return summary, user_error, critical_user_error, has_errors, errors


# =============================================================================
# User Simulator Reviewer
# =============================================================================


class UserOnlyReviewer:
    """
    LLM judge that reviews whether the user simulator behaved according to
    the task instructions for the entire conversation.
    """

    @classmethod
    def review(
        cls,
        user_info: UserInfo,
        task: Task,
        full_trajectory: list[Message],
    ) -> UserOnlyReview:
        """
        Review whether the user simulator behaved according to the task instructions.

        Args:
            user_info: Information about the user simulator configuration.
            task: The task containing user scenario instructions.
            full_trajectory: List of messages from the conversation.

        Returns:
            UserOnlyReview with any errors found.
        """
        user_instructions = str(task.user_scenario)

        # Use the guidelines from user_info if available (already computed with correct use_tools),
        # otherwise fall back to default guidelines
        if user_info.global_simulation_guidelines:
            user_guidelines = user_info.global_simulation_guidelines
        else:
            raise ValueError("No global user simulator guidelines provided")

        return cls.review_user_simulation(
            user_guidelines=user_guidelines,
            user_instructions=user_instructions,
            full_trajectory=full_trajectory,
        )

    @classmethod
    def make_user_visible_trajectory(
        cls, full_trajectory: list[Message]
    ) -> list[Message]:
        """
        Extract messages that would be visible to the user from the full trajectory.

        Filters out:
        - Agent tool calls (internal to agent)
        - Tool results for agent tools

        Keeps:
        - User messages
        - Agent messages with content (strips tool_calls if present)
        - User tool results (if user has tools)
        """
        user_trajectory = []
        for msg in full_trajectory:
            if isinstance(msg, UserMessage):
                user_trajectory.append(msg)
            elif isinstance(msg, ToolMessage) and msg.requestor == "user":
                user_trajectory.append(msg)
            elif isinstance(msg, AssistantMessage):
                if msg.is_tool_call():
                    if msg.content is not None:
                        updated_msg = deepcopy(msg)
                        updated_msg.tool_calls = None
                        user_trajectory.append(updated_msg)
                else:
                    user_trajectory.append(msg)
        return user_trajectory

    @classmethod
    def format_trajectory(cls, full_trajectory: list[Message]) -> str:
        """
        Make a string representation of the user-visible trajectory.
        """
        user_trajectory = cls.make_user_visible_trajectory(full_trajectory)
        return MarkdownDisplay.display_messages(user_trajectory)

    @classmethod
    def review_user_simulation(
        cls,
        user_guidelines: str,
        user_instructions: str,
        full_trajectory: list[Message],
    ) -> UserOnlyReview:
        """
        Review whether the user simulator behaved according to the task instructions.

        Args:
            user_guidelines: The global user simulator guidelines.
            user_instructions: The specific user instructions for this task.
            full_trajectory: List of messages from the conversation.

        Returns:
            UserOnlyReview with any errors found.
        """
        conversation_str = cls.format_trajectory(full_trajectory)

        system_prompt = SYSTEM_PROMPT.format(user_guidelines=user_guidelines)
        user_prompt = USER_PROMPT.format(
            user_instructions=user_instructions,
            conversation=conversation_str,
        )

        messages = [
            SystemMessage(role="system", content=system_prompt),
            UserMessage(role="user", content=user_prompt),
        ]

        assistant_message = generate(
            model=DEFAULT_LLM_EVAL_USER_SIMULATOR,
            messages=messages,
            call_name="llm_judge_user_only_review",
        )

        try:
            (
                summary,
                user_error,
                critical_user_error,
                has_errors,
                errors,
            ) = _parse_user_only_review_response(assistant_message.content)
            return UserOnlyReview(
                summary=summary,
                user_error=user_error,
                critical_user_error=critical_user_error,
                has_errors=has_errors,
                errors=errors,
                cost=assistant_message.cost,
            )
        except Exception as e:
            # If parsing fails, return a result indicating the review failed
            return UserOnlyReview(
                summary="",
                user_error=False,
                critical_user_error=False,
                has_errors=False,
                errors=[
                    UserOnlyReviewError(
                        turn_idx=-1,
                        reasoning=f"Failed to parse LLM response: {e}. Response: {assistant_message.content}",
                    )
                ],
                cost=assistant_message.cost,
            )


class FullDuplexUserOnlyReviewer:
    """
    LLM judge that reviews whether the user simulator behaved according to
    the task instructions for a full-duplex (tick-based) trajectory.

    This reviewer also checks for turn-taking/interruption policy errors.
    """

    @classmethod
    def review(
        cls,
        user_info: UserInfo,
        task: Task,
        full_trajectory: list[Tick],
        interruption_enabled: bool = False,
    ) -> UserOnlyReview:
        """
        Review whether the user simulator behaved according to the task instructions.

        This includes both content review and turn-taking/interruption policy review.

        Args:
            user_info: Information about the user simulator configuration.
            task: The task containing user scenario instructions.
            full_trajectory: List of Tick objects from full-duplex simulation.
            interruption_enabled: Whether the user simulator was configured to interrupt.

        Returns:
            UserOnlyReview with any errors found.
        """
        user_instructions = str(task.user_scenario)

        # Use the guidelines from user_info if available
        if user_info.global_simulation_guidelines:
            user_guidelines = user_info.global_simulation_guidelines
        else:
            raise ValueError("No global user simulator guidelines provided")

        # Convert ticks to consolidated conversation display (user-visible only)
        conversation_str = MarkdownDisplay.display_ticks_consolidated(
            full_trajectory, user_visible_only=True
        )

        # Build prompts
        system_prompt = FULL_DUPLEX_SYSTEM_PROMPT.format(
            user_guidelines=user_guidelines
        )
        user_prompt = FULL_DUPLEX_USER_PROMPT.format(
            user_instructions=user_instructions,
            interruption_enabled=interruption_enabled,
            conversation=conversation_str,
        )

        llm_messages = [
            SystemMessage(role="system", content=system_prompt),
            UserMessage(role="user", content=user_prompt),
        ]

        assistant_message = generate(
            model=DEFAULT_LLM_EVAL_USER_SIMULATOR,
            messages=llm_messages,
            call_name="llm_judge_user_only_streaming_review",
        )

        try:
            (
                summary,
                user_error,
                critical_user_error,
                has_errors,
                errors,
            ) = _parse_user_only_review_response(assistant_message.content)
            return UserOnlyReview(
                summary=summary,
                user_error=user_error,
                critical_user_error=critical_user_error,
                has_errors=has_errors,
                errors=errors,
                cost=assistant_message.cost,
            )
        except Exception as e:
            # If parsing fails, return a result indicating the review failed
            return UserOnlyReview(
                summary="",
                user_error=False,
                critical_user_error=False,
                has_errors=False,
                errors=[
                    UserOnlyReviewError(
                        turn_idx=-1,
                        error_type="review_error",
                        reasoning=f"Failed to parse LLM response: {e}. Response: {assistant_message.content}",
                    )
                ],
                cost=assistant_message.cost,
            )
