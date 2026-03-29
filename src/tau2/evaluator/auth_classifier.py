"""
LLM-based classifier for user authentication outcomes in conversations.

This classifier determines whether user authentication:
- Succeeded: The agent successfully authenticated the user
- Failed: The agent attempted but failed to authenticate the user
- Not needed: The task did not require user authentication
"""

import json
import re
from typing import Literal, Optional

from loguru import logger
from rich.console import Console
from rich.panel import Panel

from tau2.config import DEFAULT_LLM_EVAL_USER_SIMULATOR
from tau2.data_model.message import SystemMessage, Tick, UserMessage
from tau2.data_model.simulation import AuthenticationClassification
from tau2.utils.display import MarkdownDisplay
from tau2.utils.llm_utils import generate

# =============================================================================
# Prompts
# =============================================================================

SYSTEM_PROMPT = """
# Goal
Classify the user authentication outcome in this customer service conversation.

## Background
In customer service interactions, agents often need to authenticate users before taking actions on their accounts.
Authentication typically involves verifying the user's identity through:
- Email address lookup
- Name + zip code verification
- Other identity verification methods

## Classification Options

You must classify the authentication outcome as one of:

1. **succeeded**: The agent successfully authenticated/identified the user
   - The agent verified the user's identity through email, name+zip, or other means
   - The agent was able to look up and confirm the user's account
   - The conversation proceeded with the agent having access to user information

2. **failed**: The agent attempted but failed to authenticate the user
   - The agent tried to verify identity but couldn't find a matching account
   - Authentication failed due to mismatched information
   - The user couldn't be identified despite attempts

3. **not_needed**: The task did not require user authentication
   - The conversation was about general information
   - No account-specific actions were needed
   - The task could be completed without knowing who the user is

## Instructions
1. Review the conversation carefully
2. Look for authentication attempts (email lookup, name+zip verification, etc.)
3. Determine the outcome of any authentication attempts
4. If initial authentication failed, but in later attempts the agent was able to authenticate the user, then classify as "succeeded"
5. If no authentication was attempted or needed, classify as "not_needed"

## Output Format
Return ONLY a JSON object:
```json
{
    "status": "succeeded" or "failed" or "not_needed",
    "reasoning": "Brief explanation of why this classification was chosen"
}
```
""".strip()

USER_PROMPT = """
<Conversation>
{conversation}
</Conversation>
""".strip()


# =============================================================================
# Classifier Implementation
# =============================================================================


def _parse_classification_response(
    response: str,
) -> tuple[Literal["succeeded", "failed", "not_needed"], str]:
    """Parse the LLM response into classification result."""
    # Try to extract JSON from the response
    json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
    if not json_match:
        logger.warning(f"Could not find JSON in response: {response[:200]}")
        return "not_needed", "Could not parse response"

    try:
        data = json.loads(json_match.group())
        status = data.get("status", "not_needed")
        reasoning = data.get("reasoning", "")

        # Validate status
        if status not in ("succeeded", "failed", "not_needed"):
            logger.warning(f"Invalid status '{status}', defaulting to 'not_needed'")
            status = "not_needed"

        return status, reasoning
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON: {e}")
        return "not_needed", "Could not parse response"


class AuthenticationClassifier:
    """Classifier for user authentication outcomes in turn-based conversations."""

    @staticmethod
    def classify(
        messages: list,
        model: str = DEFAULT_LLM_EVAL_USER_SIMULATOR,
    ) -> AuthenticationClassification:
        """
        Classify authentication outcome for a turn-based conversation.

        Args:
            messages: List of conversation messages.
            model: LLM model to use for classification.

        Returns:
            AuthenticationClassification with status, reasoning, and cost.
        """
        # Format conversation
        conversation = MarkdownDisplay.display_messages(messages)

        # Build prompt
        user_prompt = USER_PROMPT.format(conversation=conversation)

        # Call LLM
        assistant_message = generate(
            model=model,
            messages=[
                SystemMessage(role="system", content=SYSTEM_PROMPT),
                UserMessage(role="user", content=user_prompt),
            ],
            call_name="classify_authentication",
        )

        # Parse response
        status, reasoning = _parse_classification_response(assistant_message.content)

        return AuthenticationClassification(
            status=status,
            reasoning=reasoning,
            cost=assistant_message.cost,
        )


class FullDuplexAuthenticationClassifier:
    """Classifier for user authentication outcomes in full-duplex conversations."""

    @staticmethod
    def classify(
        ticks: list[Tick],
        model: str = DEFAULT_LLM_EVAL_USER_SIMULATOR,
    ) -> AuthenticationClassification:
        """
        Classify authentication outcome for a full-duplex conversation.

        Args:
            ticks: List of conversation ticks.
            model: LLM model to use for classification.

        Returns:
            AuthenticationClassification with status, reasoning, and cost.
        """
        # Format conversation - use consolidated view for readability
        conversation = MarkdownDisplay.display_ticks_consolidated(ticks)

        # Build prompt
        user_prompt = USER_PROMPT.format(conversation=conversation)

        # Call LLM
        assistant_message = generate(
            model=model,
            messages=[
                SystemMessage(role="system", content=SYSTEM_PROMPT),
                UserMessage(role="user", content=user_prompt),
            ],
            call_name="classify_authentication",
        )

        # Parse response
        status, reasoning = _parse_classification_response(assistant_message.content)

        return AuthenticationClassification(
            status=status,
            reasoning=reasoning,
            cost=assistant_message.cost,
        )


# =============================================================================
# Display Functions
# =============================================================================


def display_auth_classification(
    classification: AuthenticationClassification,
    title: str = "Authentication Classification",
    console: Optional[Console] = None,
) -> None:
    """Display authentication classification result."""
    if console is None:
        console = Console()

    # Status emoji and color
    status_display = {
        "succeeded": ("✅", "green", "Authentication Succeeded"),
        "failed": ("❌", "red", "Authentication Failed"),
        "not_needed": ("➖", "dim", "Authentication Not Needed"),
    }
    emoji, color, label = status_display.get(
        classification.status, ("❓", "yellow", "Unknown")
    )

    content = f"{emoji} **{label}**\n\n{classification.reasoning}"

    if classification.cost:
        content += f"\n\n_Cost: ${classification.cost:.4f}_"

    console.print(Panel(content, title=title, border_style=color))
