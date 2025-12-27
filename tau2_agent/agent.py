"""
ADK agent definition for tau2-bench evaluation service.

This agent exposes tau2-bench evaluation capabilities via A2A protocol.
Supports LLMs that use text-based tool calls (JSON format) instead of
native function calling.
"""

import json
import os
import re
import uuid

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import Gemini
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from loguru import logger

from .tools import GetEvaluationResults, ListDomains, RunTau2Evaluation

# Agent instruction prompt
INSTRUCTION = """You are a conversational agent evaluation service powered by tau2-bench.

You can evaluate other conversational agents across multiple customer service domains:
- airline: Flight booking, modifications, cancellations
- retail: Product orders, returns, exchanges
- telecom: Technical support, billing issues
- mock: Simple test scenarios

When a user requests an evaluation:
1. Extract the evaluation parameters from the request (domain, agent endpoint, number of tasks)
2. Use run_tau2_evaluation tool to execute the evaluation immediately
3. Provide clear, actionable feedback on agent performance
4. Offer to retrieve detailed results using get_evaluation_results

IMPORTANT: To use a tool, respond with a JSON object:
{"tool_call": {"name": "tool_name", "arguments": {"param1": "value1"}}}

For example, to run an evaluation:
{"tool_call": {"name": "run_tau2_evaluation", "arguments": {"domain": "mock", "agent_endpoint": "http://localhost:8001/a2a/simple_nebius_agent", "num_tasks": 2}}}

Be helpful in explaining evaluation metrics and suggesting improvements.
"""


def create_model():
    """
    Create the Gemini model for the tau2 agent orchestrator.

    Uses ADK's native Gemini integration for optimal performance on GCP.
    Model can be configured via TAU2_AGENT_MODEL env var (default: gemini-2.0-flash).

    Authentication is handled automatically:
    - GCP: Uses Application Default Credentials (ADC)
    - Local: Reads GOOGLE_API_KEY from environment

    Returns:
        Gemini: ADK Gemini model instance.
    """
    model = os.getenv("TAU2_AGENT_MODEL", "gemini-2.0-flash")

    # Strip 'gemini/' prefix if present - native Gemini uses bare model name
    if model.startswith("gemini/"):
        model = model[7:]

    return Gemini(model=model)


def _extract_tool_call(text: str) -> dict | None:
    """Extract a tool call from text, trying multiple JSON formats.

    Attempts to find and parse JSON that represents a tool call.
    Handles wrapped format {"tool_call": {...}}, code blocks,
    and direct format {"name": "...", "arguments": {...}}.

    Args:
        text: The text to search for tool calls.

    Returns:
        A dict with "name" and optional "arguments" keys, or None if not found.
    """
    # Remove code block markers if present
    cleaned = re.sub(r'```(?:tool_call|json)?\s*', '', text)
    cleaned = re.sub(r'```', '', cleaned)

    # Find all JSON-like objects in the text
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, cleaned, re.DOTALL)

    for match in matches:
        try:
            parsed = json.loads(match)

            # Check for wrapped format: {"tool_call": {"name": ...}}
            if isinstance(parsed.get("tool_call"), dict):
                inner = parsed["tool_call"]
                if "name" in inner:
                    return inner

            # Check for direct format: {"name": ..., "arguments": ...}
            if "name" in parsed:
                return parsed

        except json.JSONDecodeError:
            continue

    return None


def parse_text_tool_call(
    callback_context: CallbackContext,  # noqa: ARG001
    llm_response: LlmResponse,
) -> LlmResponse | None:
    """
    Parse a JSON-formatted text tool call from an LLM response and convert it into a function_call LlmResponse.

    Parameters:
        llm_response (LlmResponse): The model response to inspect for a JSON `tool_call` object; ignored if the response already contains a native function_call.

    Returns:
        LlmResponse | None: A new LlmResponse whose single part is a function_call constructed from the parsed `tool_call` (with a generated id), or `None` if no valid text-based tool call is found.
    """
    if not llm_response.content or not llm_response.content.parts:
        return None

    # Check if response already has a native function_call - if so, pass through
    for part in llm_response.content.parts:
        if part.function_call:
            logger.debug(
                "Response already contains native function_call, passing through"
            )
            return None

    # No native function_call found - try to parse text-based tool call
    full_text = ""
    for part in llm_response.content.parts:
        if part.text:
            full_text += part.text

    if not full_text:
        return None

    # Extract tool call from text - supports multiple formats:
    # 1. {"tool_call": {"name": "...", "arguments": {...}}}
    # 2. ```tool_call\n{"name": "...", "arguments": {...}}```
    # 3. {"name": "...", "arguments": {...}}
    tool_call = _extract_tool_call(full_text)
    if not tool_call:
        return None

    tool_name = tool_call["name"]
    tool_args = tool_call.get("arguments", {})

    logger.info(
        "Parsed text-based tool call (model did not use native function calling)",
        tool_name=tool_name,
        tool_args=tool_args,
    )

    # Create a function_call Part
    function_call_part = types.Part.from_function_call(
        name=tool_name,
        args=tool_args,
    )
    # ADK expects an id on function calls; set post-creation as from_function_call doesn't accept id
    assert function_call_part.function_call is not None
    function_call_part.function_call.id = str(uuid.uuid4())

    # Return a new LlmResponse with the function call
    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[function_call_part],
        ),
        usage_metadata=llm_response.usage_metadata,
    )


# Define the ADK Agent
root_agent = LlmAgent(
    name="tau2_agent",
    model=create_model(),
    instruction=INSTRUCTION,
    description="Agent evaluation service using tau2-bench framework across airline, retail, and telecom domains",
    after_model_callback=parse_text_tool_call,
    tools=[
        RunTau2Evaluation(
            name="run_tau2_evaluation",
            description="""Run a tau2-bench evaluation of a conversational agent.

            Requires X-User-LLM-Model and X-User-LLM-API-Key headers.

            Parameters:
            - domain: Evaluation domain (airline, retail, telecom, mock)
            - agent_endpoint: A2A endpoint of agent to evaluate
            - num_trials: Number of trials per task (default: 1)
            - num_tasks: Number of tasks to evaluate (optional)
            - task_ids: Optional list of specific task IDs to run
            """,
        ),
        ListDomains(
            name="list_domains",
            description="List all available tau2-bench evaluation domains and their descriptions",
        ),
        GetEvaluationResults(
            name="get_evaluation_results",
            description="Get detailed results from a tau2-bench evaluation by evaluation_id",
        ),
    ],
)
