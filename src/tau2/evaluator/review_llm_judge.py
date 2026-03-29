import json
from typing import Literal

from tau2.config import DEFAULT_LLM_EVAL_USER_SIMULATOR
from tau2.data_model.message import SystemMessage, Tick, UserMessage
from tau2.data_model.simulation import Review, ReviewError, UserInfo
from tau2.data_model.tasks import Task
from tau2.utils.display import MarkdownDisplay
from tau2.utils.llm_utils import extract_json_from_llm_response, generate

# =============================================================================
# Prompts for Full Conversation Review (Both User and Agent)
# =============================================================================

SYSTEM_PROMPT = """
# Goal
Assess this conversation to determine if the user simulator or the agent made any errors during the interaction.

## Overview
We are supervising a Customer Service Agent.
Given a **Scenario**, we simulate interactions between a **User Simulator** and the **Agent**.
During an interaction, the User Simulator sends messages to the Agent.
The Agent responds to the User Simulator's messages, and may also perform actions using **Tools**.

An **interaction trajectory** is a sequence of **turns** that can be one of the following:
- **User Turn**: A message from the User Simulator to the Agent
- **Agent Turn**: A message from the Agent to the User Simulator
- **Tool Turn**: A tool call from the Agent to the Tool Server
- **Tool Response Turn**: A tool response from the Tool Server to the Agent

## Guidelines

### General rules
For all interactions:
- The Agent must follow a **Policy**. This Policy specifies the rules according to which the agent must act.
- The User Simulator must adhere to **Simulation guidelines**. These guidelines specify how the User Simulator should interpret its instructions.

### Scenario specifications for an interaction
A particular interaction is based on a **Scenario**.
A Scenario defines:
- The **User Instructions**.
- The **Actions** that the Agent is expected to take.
- **Natural Language Assertions** that the Agent must satisfy.

## How to assess the conversation

### User Turn Errors:
For each User Turn, check if the User Simulator:
- Followed the Simulation guidelines
- Followed the User instructions
- Performed its task correctly
- Acted consistently with previous turns

### Important User Review Principles:
1. **Fact-check every user claim**: For every factual detail the user provides (names, emails, zip codes, sizes, colors, product descriptions, etc.), verify it appears in or is derivable from the <User Instructions>. Any detail not grounded in the instructions is a hallucination — even if it sounds plausible. When the user lacks information, the correct behavior is to say "I don't know" or ask the agent.

2. **Do not blame the user for agent failures**: If the agent is unresponsive, repeatedly fails, or makes critical errors, the user giving up or ending the conversation is a reasonable reaction — not a user error. Only flag premature_termination when the agent was actively working and making progress.

### User Error Severity:
For each user error, classify its severity:
- **critical_helped**: The user error helped the agent succeed inappropriately (e.g., user provided information they shouldn't have, making the task too easy).
- **critical_hindered**: The user error hindered the agent or made the task harder/impossible (e.g., user provided incorrect information or contradicted instructions).
- **minor**: The user made an error but it did NOT influence the simulation outcome.

### Agent Turn Errors:
For each Agent Turn, check if the Agent:
- Followed the Policy
- Correctly performed its task
- Acted consistently with previous turns

### Agent Error Severity:
For each agent error, classify its severity:
- **critical**: The error directly caused task failure OR violated important security/policy requirements (even if task succeeded).
- **minor**: The error was suboptimal but did NOT affect the final outcome or violate critical policies.

### Error Tags:
For each error, assign one or more tags from the following list:
- **hallucination**: Made up information not grounded in guidelines, instructions, or tool call results. For user errors specifically: provided factual details (e.g., zip codes, sizes, product descriptions) not present in or derivable from the user instructions.
- **incorrect_interpretation**: Misinterpreted available information (e.g., misread a tool result or misunderstood a message).
- **guideline_violation**: Message or action not consistent with the provided guidelines or policy.
- **revealed_info_early**: Shared information before it was appropriate or before proper verification.
- **inconsistent_behavior**: Action or statement contradicts earlier statements or actions in the conversation.
- **tool_call_schema_error**: Made a tool call with invalid tool name, missing arguments, or wrong argument types.
- **tool_call_argument_error**: Made a tool call with correct schema but incorrect argument values.
- **irrelevant_tool_call**: Made a tool call not relevant to the current task or conversation state.
- **premature_termination**: Ended the conversation or accepted an incomplete outcome while the other participant was actively working and making progress. Do NOT use this tag if the user ended the conversation because the agent was unresponsive or repeatedly failing.
- **missed_required_action**: Did not take a required action that was expected.
- **wrong_sequence**: Performed actions out of the expected order or sequence.
- **other**: Use only when no other tag applies. Include a description of the error type in the reasoning.

### Workflow
Follow these steps to produce your analysis:

1. **Fact-check user claims**: Verify every factual claim the user makes against the <User Instructions>.

2. **Analyze each turn**: Go through the conversation turn by turn. For each turn, check if the message contains an error based on the guidelines above. Note any errors you find.

3. **Assess context for termination**: If the user ended the conversation early, only flag premature_termination if the agent was actively making progress (not stalled or failing).

4. **Summarize**: Summarize what happened at the conversation level, including what errors (if any) affected the outcome.

5. **Format output**: Compile your findings into the expected JSON format. Include only the turns where errors were found (discard turns with no errors from the errors list).

# Inputs
- <Policy>: The policy the Agent must follow.
- <Simulation Guidelines>: The guidelines for the User Simulator.
- <User Instructions>: Instructions for the user simulator to follow.
- <Example Action Trajectory>: An example sequence of actions that could complete the task. Note: Other valid approaches may exist, and this example trajectory may include extraneous actions.
- <Natural Language Assertions>: Assertions the Agent must satisfy.
- <Conversation>: The full conversation between the user simulator and the agent.

# Output
Structure your answer in the following JSON format:
```json
{{
    "errors": [
        {{
            "source": "user" or "agent" (who performed the action that has the error),
            "error_tags": ["<tag1>", "<tag2>", ...],
            "severity": "minor" or "critical" (for agent) / "minor" or "critical_helped" or "critical_hindered" (for user),
            "turn_idx": <turn number where error occurred>,
            "reasoning": "<explanation of why this is an error>",
            "correct_behavior": "<what should have been done instead>"
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
<Policy>
{policy}
</Policy>

<Simulation Guidelines>
{simulation_guidelines}
</Simulation Guidelines>

<User Instructions>
{user_instructions}
</User Instructions>

<Example Action Trajectory>
{example_action_trajectory}
</Example Action Trajectory>

<Natural Language Assertions>
{natural_language_assertions}
</Natural Language Assertions>

<Conversation>
{conversation}
</Conversation>
""".strip()


# =============================================================================
# Full-Duplex Prompts (with Interruption Policy)
# =============================================================================

FULL_DUPLEX_SYSTEM_PROMPT = """
# Goal
Assess this full-duplex conversation to determine if the user simulator or the agent made any errors during the interaction.
This includes both content errors AND turn-taking/interruption behavior errors.

## Overview
We are supervising a Customer Service Agent in a full-duplex (real-time) conversation.
Given a **Scenario**, we simulate interactions between a **User Simulator** and the **Agent**.
Both parties can speak simultaneously, and interruptions may occur.

## Conversation Format
The conversation is displayed in a consolidated format where:
- Consecutive speech from the same speaker is grouped together into segments
- Each segment shows **Agent**: or **User**: followed by the speech content
- Tool calls and results are shown separately when they occur
- Because this is full-duplex, both parties may speak in overlapping segments

## Guidelines

### General rules
For all interactions:
- The Agent must follow a **Policy**. This Policy specifies the rules according to which the agent must act.
- The User Simulator must adhere to **Simulation guidelines**. These guidelines specify how the User Simulator should interpret its instructions.

### Scenario specifications for an interaction
A particular interaction is based on a **Scenario**.
A Scenario defines:
- The **User Instructions**.
- The **Actions** that the Agent is expected to take.
- **Natural Language Assertions** that the Agent must satisfy.

## How to assess the conversation

### User Errors (Content):
For each User segment, check if the User Simulator:
- Followed the Simulation guidelines
- Followed the User instructions
- Performed its task correctly
- Acted consistently with previous segments

### Important User Review Principles:
1. **Fact-check every user claim**: For every factual detail the user provides (names, emails, zip codes, sizes, colors, product descriptions, etc.), verify it appears in or is derivable from the <User Instructions>. Any detail not grounded in the instructions is a hallucination — even if it sounds plausible. When the user lacks information, the correct behavior is to say "I don't know" or ask the agent.

2. **Do not blame the user for agent failures**: If the agent is unresponsive, repeatedly fails, or makes critical errors, the user giving up or ending the conversation is a reasonable reaction — not a user error. Only flag premature_termination when the agent was actively working and making progress.

### User Errors (Turn-Taking, only if interruption is enabled):
- Interrupts the agent too frequently or unnecessarily
- Interrupts when the agent has only spoken a few words (less than ~5 words)
- Fails to interrupt when the agent has clearly finished their main point and is rambling
- Interrupts at inappropriate moments (e.g., mid-sentence when important info is being conveyed)
- Does NOT interrupt when the user has heard enough to respond

### User Error Severity:
For each user error, classify its severity:
- **critical_helped**: The user error helped the agent succeed inappropriately (e.g., user provided information they shouldn't have, making the task too easy).
- **critical_hindered**: The user error hindered the agent or made the task harder/impossible (e.g., user provided incorrect information or contradicted instructions).
- **minor**: The user made an error but it did NOT influence the simulation outcome.

### Agent Errors:
For each Agent segment, check if the Agent:
- Followed the Policy
- Correctly performed its task
- Acted consistently with previous segments

### Agent Error Severity:
For each agent error, classify its severity:
- **critical**: The error directly caused task failure OR violated important security/policy requirements (even if task succeeded).
- **minor**: The error was suboptimal but did NOT affect the final outcome or violate critical policies.

### Error Tags:
For each error, assign one or more tags from the following list:
- **hallucination**: Made up information not grounded in guidelines, instructions, or tool call results. For user errors specifically: provided factual details (e.g., zip codes, sizes, product descriptions) not present in or derivable from the user instructions.
- **incorrect_interpretation**: Misinterpreted available information (e.g., misread a tool result or misunderstood a message).
- **guideline_violation**: Message or action not consistent with the provided guidelines or policy.
- **revealed_info_early**: Shared information before it was appropriate or before proper verification.
- **inconsistent_behavior**: Action or statement contradicts earlier statements or actions in the conversation.
- **tool_call_schema_error**: Made a tool call with invalid tool name, missing arguments, or wrong argument types.
- **tool_call_argument_error**: Made a tool call with correct schema but incorrect argument values.
- **irrelevant_tool_call**: Made a tool call not relevant to the current task or conversation state.
- **premature_termination**: Ended the conversation or accepted an incomplete outcome while the other participant was actively working and making progress. Do NOT use this tag if the user ended the conversation because the agent was unresponsive or repeatedly failing.
- **missed_required_action**: Did not take a required action that was expected.
- **wrong_sequence**: Performed actions out of the expected order or sequence.
- **interruption_error**: Interrupted inappropriately or failed to interrupt when appropriate (only for full-duplex with interruption enabled).
- **other**: Use only when no other tag applies. Include a description of the error type in the reasoning.

### Workflow
Follow these steps to produce your analysis:

1. **Fact-check user claims**: Verify every factual claim the user makes against the <User Instructions>.

2. **Analyze each segment**: Go through the conversation segment by segment. For each segment, check if it contains an error based on the guidelines above. Note any errors you find.

3. **Assess context for termination**: If the user ended the conversation early, only flag premature_termination if the agent was actively making progress (not stalled or failing).

4. **Summarize**: Summarize what happened at the conversation level, including what errors (if any) affected the outcome.

5. **Format output**: Compile your findings into the expected JSON format. Include only the segments where errors were found (discard segments with no errors from the errors list).

# Inputs
- <Policy>: The policy the Agent must follow.
- <Simulation Guidelines>: The guidelines for the User Simulator.
- <User Instructions>: Instructions for the user simulator to follow.
- <Example Action Trajectory>: An example sequence of actions that could complete the task. Note: Other valid approaches may exist, and this example may include extraneous steps.
- <Natural Language Assertions>: Assertions the Agent must satisfy.
- <Interruption Policy>: Whether the user simulator was configured to interrupt.
- <Conversation>: The full conversation between the user simulator and the agent.

# Output
Structure your answer in the following JSON format:
```json
{{
    "errors": [
        {{
            "source": "user" or "agent" (who performed the action that has the error),
            "error_type": "content_error" or "interruption_error",
            "error_tags": ["<tag1>", "<tag2>", ...],
            "severity": "minor" or "critical" (for agent) / "minor" or "critical_helped" or "critical_hindered" (for user),
            "tick_start": <start tick of the segment where error occurred>,
            "tick_end": <end tick of the segment where error occurred>,
            "reasoning": "<explanation of why this is an error>",
            "correct_behavior": "<what should have been done instead>"
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
<Policy>
{policy}
</Policy>

<Simulation Guidelines>
{simulation_guidelines}
</Simulation Guidelines>

<User Instructions>
{user_instructions}
</User Instructions>

<Example Action Trajectory>
{example_action_trajectory}
</Example Action Trajectory>

<Natural Language Assertions>
{natural_language_assertions}
</Natural Language Assertions>

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


def _parse_review_response(
    response: str,
) -> tuple[str, bool, bool, bool, bool, list[ReviewError]]:
    """
    Parse the LLM response in JSON format.

    Returns:
        Tuple of (summary, agent_error, user_error, critical_user_error, has_errors, list of ReviewError)
    """
    json_str = _extract_json_from_response(response)
    result_data = json.loads(json_str)
    summary = result_data.get("summary", "")

    errors: list[ReviewError] = []
    for error_data in result_data.get("errors", []):
        source_str = error_data.get("source", "unknown")
        source: Literal["user", "agent", "unknown"] = "unknown"
        if source_str.lower() == "user":
            source = "user"
        elif source_str.lower() == "agent":
            source = "agent"

        # Parse severity
        severity = None
        severity_str = error_data.get("severity")
        if source == "user":
            if severity_str in ("minor", "critical_helped", "critical_hindered"):
                severity = severity_str
        elif source == "agent":
            if severity_str in ("minor", "critical"):
                severity = severity_str

        # Parse error_tags (must be a list)
        error_tags = error_data.get("error_tags", [])
        if isinstance(error_tags, str):
            error_tags = [error_tags]

        errors.append(
            ReviewError(
                source=source,
                error_type=error_data.get("error_type"),
                error_tags=error_tags,
                severity=severity,
                turn_idx=error_data.get("turn_idx"),
                tick_start=error_data.get("tick_start"),
                tick_end=error_data.get("tick_end"),
                reasoning=error_data.get("reasoning", ""),
                correct_behavior=error_data.get("correct_behavior"),
            )
        )

    # Calculate boolean flags from errors list
    agent_error = any(e.source == "agent" for e in errors)
    user_error = any(e.source == "user" for e in errors)
    critical_user_error = any(
        e.source == "user" and e.severity in ("critical_helped", "critical_hindered")
        for e in errors
    )
    has_errors = len(errors) > 0

    return summary, agent_error, user_error, critical_user_error, has_errors, errors


# =============================================================================
# Conversation Reviewer (Both User and Agent)
# =============================================================================


class ConversationReviewer:
    """
    LLM judge that reviews whether both the user simulator and the agent
    behaved according to their respective guidelines for the entire conversation.
    """

    @classmethod
    def review(
        cls,
        user_info: UserInfo,
        task: Task,
        full_trajectory: list,
        policy: str,
    ) -> Review:
        """
        Review whether both the user simulator and the agent behaved correctly.

        Args:
            user_info: Information about the user simulator configuration.
            task: The task containing user scenario and evaluation criteria.
            full_trajectory: List of messages from the conversation.
            policy: The policy the agent must follow.

        Returns:
            Review with any error found.
        """
        user_instructions = str(task.user_scenario)

        # Use the guidelines from user_info if available
        if user_info.global_simulation_guidelines:
            simulation_guidelines = user_info.global_simulation_guidelines
        else:
            raise ValueError("No global user simulator guidelines provided")

        # Get evaluation criteria
        example_action_trajectory = MarkdownDisplay.display_actions(
            task.evaluation_criteria.actions
        )
        nl_assertions = task.evaluation_criteria.nl_assertions or ""

        return cls.review_conversation(
            policy=policy,
            simulation_guidelines=simulation_guidelines,
            user_instructions=user_instructions,
            example_action_trajectory=example_action_trajectory,
            natural_language_assertions=nl_assertions,
            full_trajectory=full_trajectory,
        )

    @classmethod
    def format_trajectory(cls, full_trajectory: list) -> str:
        """
        Make a string representation of the full trajectory.
        """
        return MarkdownDisplay.display_messages(full_trajectory)

    @classmethod
    def review_conversation(
        cls,
        policy: str,
        simulation_guidelines: str,
        user_instructions: str,
        example_action_trajectory: str,
        natural_language_assertions: str,
        full_trajectory: list,
    ) -> Review:
        """
        Review whether the conversation proceeded correctly.

        Args:
            policy: The policy the agent must follow.
            simulation_guidelines: The global user simulator guidelines.
            user_instructions: The specific user instructions for this task.
            example_action_trajectory: An example sequence of actions that could complete the task.
            natural_language_assertions: Assertions the agent must satisfy.
            full_trajectory: List of messages from the conversation.

        Returns:
            Review with any error found.
        """
        conversation_str = cls.format_trajectory(full_trajectory)

        system_prompt = SYSTEM_PROMPT
        user_prompt = USER_PROMPT.format(
            policy=policy,
            simulation_guidelines=simulation_guidelines,
            user_instructions=user_instructions,
            example_action_trajectory=example_action_trajectory,
            natural_language_assertions=natural_language_assertions,
            conversation=conversation_str,
        )

        messages = [
            SystemMessage(role="system", content=system_prompt),
            UserMessage(role="user", content=user_prompt),
        ]

        assistant_message = generate(
            model=DEFAULT_LLM_EVAL_USER_SIMULATOR,
            messages=messages,
            call_name="llm_judge_review",
        )

        try:
            (
                summary,
                agent_error,
                user_error,
                critical_user_error,
                has_errors,
                errors,
            ) = _parse_review_response(assistant_message.content)

            return Review(
                summary=summary,
                agent_error=agent_error,
                user_error=user_error,
                critical_user_error=critical_user_error,
                has_errors=has_errors,
                errors=errors,
                cost=assistant_message.cost,
            )
        except Exception as e:
            # If parsing fails, return a result indicating the review failed
            return Review(
                summary="",
                agent_error=False,
                user_error=False,
                critical_user_error=False,
                has_errors=False,
                errors=[
                    ReviewError(
                        source="unknown",
                        turn_idx=None,
                        reasoning=f"Failed to parse LLM response: {e}. Response: {assistant_message.content}",
                        correct_behavior=None,
                    )
                ],
                cost=assistant_message.cost,
            )


class FullDuplexConversationReviewer:
    """
    LLM judge that reviews whether both the user simulator and the agent
    behaved correctly for a full-duplex (tick-based) trajectory.

    This reviewer also checks for turn-taking/interruption policy errors.
    """

    @classmethod
    def review(
        cls,
        user_info: UserInfo,
        task: Task,
        full_trajectory: list[Tick],
        policy: str,
        interruption_enabled: bool = False,
    ) -> Review:
        """
        Review whether both the user simulator and the agent behaved correctly.

        This includes both content review and turn-taking/interruption policy review.

        Args:
            user_info: Information about the user simulator configuration.
            task: The task containing user scenario and evaluation criteria.
            full_trajectory: List of Tick objects from full-duplex simulation.
            policy: The policy the agent must follow.
            interruption_enabled: Whether the user simulator was configured to interrupt.

        Returns:
            Review with any error found.
        """
        user_instructions = str(task.user_scenario)

        # Use the guidelines from user_info if available
        if user_info.global_simulation_guidelines:
            simulation_guidelines = user_info.global_simulation_guidelines
        else:
            raise ValueError("No global user simulator guidelines provided")

        # Get evaluation criteria
        example_action_trajectory = MarkdownDisplay.display_actions(
            task.evaluation_criteria.actions
        )
        nl_assertions = task.evaluation_criteria.nl_assertions or ""

        # Convert ticks to consolidated conversation display
        conversation_str = MarkdownDisplay.display_ticks_consolidated(full_trajectory)

        # Build prompts
        system_prompt = FULL_DUPLEX_SYSTEM_PROMPT
        user_prompt = FULL_DUPLEX_USER_PROMPT.format(
            policy=policy,
            simulation_guidelines=simulation_guidelines,
            user_instructions=user_instructions,
            example_action_trajectory=example_action_trajectory,
            natural_language_assertions=nl_assertions,
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
            call_name="llm_judge_streaming_review",
        )

        try:
            (
                summary,
                agent_error,
                user_error,
                critical_user_error,
                has_errors,
                errors,
            ) = _parse_review_response(assistant_message.content)

            return Review(
                summary=summary,
                agent_error=agent_error,
                user_error=user_error,
                critical_user_error=critical_user_error,
                has_errors=has_errors,
                errors=errors,
                cost=assistant_message.cost,
            )
        except Exception as e:
            # If parsing fails, return a result indicating the review failed
            return Review(
                summary="",
                agent_error=False,
                user_error=False,
                critical_user_error=False,
                has_errors=False,
                errors=[
                    ReviewError(
                        source="unknown",
                        turn_idx=None,
                        reasoning=f"Failed to parse LLM response: {e}. Response: {assistant_message.content}",
                        correct_behavior=None,
                    )
                ],
                cost=assistant_message.cost,
            )
