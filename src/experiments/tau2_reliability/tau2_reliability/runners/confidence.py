"""Post-hoc confidence elicitation for predictability metrics.

After simulations have been run, this module prompts the agent LLM
to self-assess its confidence on each task, enabling P_cal, P_auroc,
and P_brier computation.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from loguru import logger
from tau2.data_model.message import AssistantMessage, SystemMessage, UserMessage
from tau2.data_model.simulation import Results, SimulationRun
from tau2.utils.llm_utils import generate

from tau2_reliability.models import TaskTrialData

CONFIDENCE_SYSTEM_PROMPT = """You are evaluating your own performance on a customer service task.

You will be shown a conversation between an AI agent and a customer. The agent attempted to resolve the customer's request using the available tools.

Based on the conversation, rate your confidence that the agent correctly and fully resolved the customer's request.

Respond with ONLY a JSON object: {"confidence": <number between 0 and 100>, "reasoning": "<brief explanation>"}

- 0 = completely uncertain the task was handled correctly
- 50 = unsure, could go either way
- 100 = fully confident the task was handled correctly"""


def elicit_confidence(
    sim: SimulationRun,
    model: str = "gpt-4o-mini",
) -> Optional[float]:
    """Elicit confidence score for a single simulation via LLM self-assessment.

    Args:
        sim: A completed SimulationRun with messages.
        model: LLM model to use for confidence elicitation.

    Returns:
        Confidence score normalized to [0, 1], or None on failure.
    """
    messages = sim.messages or []
    if not messages:
        return None

    # Build transcript
    transcript_parts = []
    for msg in messages:
        if isinstance(msg, (UserMessage, AssistantMessage)):
            role = msg.role
            content = msg.content or ""
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_str = ", ".join(f"{tc.name}({tc.arguments})" for tc in msg.tool_calls)
                content = f"{content} [Tool calls: {tool_str}]".strip()
            if content:
                transcript_parts.append(f"{role}: {content}")

    transcript = "\n".join(transcript_parts[-30:])  # Last 30 messages to fit context

    try:
        response = generate(
            model=model,
            messages=[
                SystemMessage(role="system", content=CONFIDENCE_SYSTEM_PROMPT),
                UserMessage(role="user", content=f"Conversation:\n\n{transcript}"),
            ],
        )

        score = _parse_confidence(response.content or "")
        if score is not None:
            return score / 100.0  # Normalize to [0, 1]
        return None

    except Exception as e:
        logger.warning(f"Confidence elicitation failed for sim {sim.id}: {e}")
        return None


def elicit_confidences_for_results(
    results: Results,
    model: str = "gpt-4o-mini",
    max_concurrent: int = 5,
) -> dict[str, list[float]]:
    """Elicit confidence scores for all simulations in a Results object.

    Returns:
        Dict mapping task_id -> list of confidence scores (one per trial).
    """
    from collections import defaultdict

    confidences: dict[str, list[float]] = defaultdict(list)
    total = len(results.simulations)

    for i, sim in enumerate(results.simulations):
        if (i + 1) % 10 == 0:
            logger.info(f"Eliciting confidence: {i + 1}/{total}")

        score = elicit_confidence(sim, model=model)
        if score is not None:
            confidences[sim.task_id].append(score)
        else:
            confidences[sim.task_id].append(0.5)  # Default: maximum uncertainty

    logger.info(f"Elicited {sum(len(v) for v in confidences.values())} confidence scores")
    return dict(confidences)


def attach_confidences(
    task_data: list[TaskTrialData],
    confidences: dict[str, list[float]],
) -> list[TaskTrialData]:
    """Attach confidence scores to existing TaskTrialData objects.

    Returns new TaskTrialData objects with confidence_scores set.
    """
    updated = []
    for td in task_data:
        if td.task_id in confidences:
            scores = confidences[td.task_id]
            # Pad or trim to match number of trials
            if len(scores) < td.num_trials:
                scores = scores + [0.5] * (td.num_trials - len(scores))
            elif len(scores) > td.num_trials:
                scores = scores[:td.num_trials]
            td_new = td.model_copy(update={"confidence_scores": scores})
            updated.append(td_new)
        else:
            updated.append(td)
    return updated


def _parse_confidence(text: str) -> Optional[float]:
    """Parse a confidence score from LLM response."""
    # Try JSON parsing first
    try:
        from tau2.utils.llm_utils import extract_json_from_llm_response
        json_str = extract_json_from_llm_response(text)
        data = json.loads(json_str)
        if "confidence" in data:
            val = float(data["confidence"])
            return max(0.0, min(100.0, val))
    except Exception:
        pass

    # Fallback: find any number 0-100 in the text
    numbers = re.findall(r'\b(\d{1,3}(?:\.\d+)?)\b', text)
    for num_str in numbers:
        val = float(num_str)
        if 0 <= val <= 100:
            return val

    return None
