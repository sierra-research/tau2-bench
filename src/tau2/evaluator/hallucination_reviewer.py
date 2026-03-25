"""
Hallucination reviewer for detecting fabricated information in user simulator messages.

This module provides a focused fact-checker that identifies instances where the
user simulator provided information not present in or derivable from the user
instructions. It is used to gate reruns — if hallucinations are detected, the
simulation is re-run with feedback.

Only full-duplex (tick-based) mode is supported for now.
"""

import json

from tau2.config import DEFAULT_LLM_EVAL_USER_SIMULATOR
from tau2.data_model.message import SystemMessage, Tick, UserMessage
from tau2.data_model.simulation import HallucinationCheck, HallucinationCheckError
from tau2.data_model.tasks import Task
from tau2.utils.display import MarkdownDisplay
from tau2.utils.llm_utils import extract_json_from_llm_response, generate

# =============================================================================
# Prompts
# =============================================================================

FULL_DUPLEX_SYSTEM_PROMPT = """
# Goal
You are a fact-checker. Your task is to identify every instance where the user \
provided information that is NOT present in or derivable from the user instructions.

This includes fabricated details such as zip codes, order IDs, addresses, phone numbers, \
email addresses, dates of birth, sizes, colors, preferences, past order details, account \
numbers, or any other specific factual claims.

# Valid Sources of Information
The following are legitimate sources of information — the user referencing these is NOT a hallucination:

1. **User instructions**: Everything in the <User Instructions> section, including both the \
persona description and the scenario instructions. Both are valid.
2. **Agent-provided information**: If the agent tells the user something during the conversation \
(e.g., "I see your order #12345 was placed on Monday"), the user may reference that information \
later. This is NOT a hallucination.
3. **User tool call results**: If the user performed a tool call and received results shown in \
the conversation, the user may reference those results. This is NOT a hallucination.

# Unknown Information
Pay special attention to the "Unknown info" section in the user instructions, if present. \
If the instructions explicitly state that the user does not know something (e.g., \
"You do not remember your email address"), and the user provides that information anyway, \
it is definitely a hallucination.

# What IS a Hallucination
A hallucination is when the user states or implies a specific factual detail that:
- Is NOT in the user instructions (including persona)
- Was NOT provided by the agent during the conversation
- Was NOT returned by a user tool call
- Includes picking a specific preference when asked (e.g., brightness, refund method) \
when the instructions don't provide one — the user should say "no preference" or "I don't know"

Common examples: making up a zip code, inventing an order ID, fabricating a size/color \
preference, providing an address not in the instructions, stating past order details that \
were never mentioned, or suggesting a specific payment method not in the instructions.

Simple conversational responses (e.g., "okay", "thank you", "yes", "go ahead") are NOT \
hallucinations.

# Conversation Format
The conversation is displayed in a consolidated format where:
- Consecutive speech from the same speaker is grouped together into segments
- Each segment shows **Agent**: or **User**: followed by the speech content
- Agent tool calls are not shown (internal to the agent)
- User tool calls and results may be shown

# Instructions
1. Read the <User Instructions> carefully — this is your ground truth.
2. Go through each user segment in the conversation.
3. For every factual claim the user makes, verify it against the valid sources above.
4. Flag any claim that cannot be traced to a valid source.
5. For each hallucination, explain what was fabricated and what the user should have said instead.

# Output
Think step-by-step FIRST in the "reasoning" field, then give your verdict.

Structure your answer in the following JSON format:
```json
{{
    "reasoning": "<step-by-step analysis of the conversation: list every factual claim the user made, what source (if any) it maps to, and whether it checks out>",
    "hallucinations": [
        {{
            "reasoning": "<why this specific claim is fabricated — what source was checked and missing>",
            "user_message": "<the problematic user message content>",
            "correct_behavior": "<what the user should have said instead>"
        }}
    ],
    "summary": "<brief summary of findings>"
}}
```

- "reasoning": Your step-by-step fact-checking analysis. This MUST come before the hallucinations list.
- "hallucinations": List of hallucinations found. Empty list if none.
- "summary": A brief summary of the fact-check.

Return ONLY the JSON object, no additional text.
""".strip()


FULL_DUPLEX_USER_PROMPT = """
<User Instructions>
{user_instructions}
</User Instructions>

<Conversation>
{conversation}
</Conversation>
""".strip()


# =============================================================================
# Parsing
# =============================================================================


def _parse_hallucination_response(
    response: str,
) -> tuple[str, bool, list[HallucinationCheckError], str]:
    """
    Parse the LLM response for hallucination check.

    Returns:
        Tuple of (reasoning, hallucination_found, list of HallucinationCheckError, summary)
    """
    json_str = extract_json_from_llm_response(response)
    result_data = json.loads(json_str)
    reasoning = result_data.get("reasoning", "")
    summary = result_data.get("summary", "")

    errors: list[HallucinationCheckError] = []
    for h in result_data.get("hallucinations", []):
        errors.append(
            HallucinationCheckError(
                reasoning=h.get("reasoning", ""),
                user_message=h.get("user_message"),
                correct_behavior=h.get("correct_behavior"),
            )
        )

    hallucination_found = len(errors) > 0
    return reasoning, hallucination_found, errors, summary


# =============================================================================
# Reviewer
# =============================================================================


class FullDuplexHallucinationReviewer:
    """
    LLM judge that checks whether the user simulator hallucinated information
    not present in the task instructions, for a full-duplex (tick-based) trajectory.
    """

    @classmethod
    def review(
        cls,
        task: Task,
        full_trajectory: list[Tick],
    ) -> HallucinationCheck:
        """
        Check whether the user simulator hallucinated information.

        Args:
            task: The task containing user scenario instructions.
            full_trajectory: List of Tick objects from full-duplex simulation.

        Returns:
            HallucinationCheck with any hallucinations found.
        """
        user_instructions = str(task.user_scenario)

        # Convert ticks to consolidated conversation display (user-visible only)
        conversation_str = MarkdownDisplay.display_ticks_consolidated(
            full_trajectory, user_visible_only=True
        )

        # Build prompts
        system_prompt = FULL_DUPLEX_SYSTEM_PROMPT
        user_prompt = FULL_DUPLEX_USER_PROMPT.format(
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
            call_name="llm_judge_hallucination_check",
        )

        try:
            reasoning, hallucination_found, errors, summary = (
                _parse_hallucination_response(assistant_message.content)
            )
            return HallucinationCheck(
                reasoning=reasoning,
                hallucination_found=hallucination_found,
                errors=errors,
                summary=summary,
                cost=assistant_message.cost,
            )
        except Exception as e:
            # If parsing fails, return a safe result (no hallucination detected)
            return HallucinationCheck(
                reasoning=f"Failed to parse LLM response: {e}",
                hallucination_found=False,
                errors=[
                    HallucinationCheckError(
                        reasoning=f"Failed to parse LLM response: {e}. Response: {assistant_message.content}",
                    )
                ],
                summary="",
                cost=assistant_message.cost,
            )
