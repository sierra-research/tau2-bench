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
from google.adk.models.lite_llm import LiteLlm
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
    Selects and constructs the LLM model used by the agent.
    
    If the `NEBIUS_API_KEY` environment variable is present, returns a LiteLlm configured for the Nebius Qwen model and uses `NEBIUS_API_BASE` (default: "https://api.tokenfactory.nebius.com/v1/") as the API base. If `NEBIUS_API_KEY` is not set, returns the Gemini model identifier string "gemini-2.0-flash-exp".
    
    Returns:
        A LiteLlm instance configured for Nebius when `NEBIUS_API_KEY` is set; otherwise the Gemini model identifier string `"gemini-2.0-flash-exp"`.
    """
    nebius_key = os.getenv("NEBIUS_API_KEY")
    if nebius_key:
        api_base = os.getenv(
            "NEBIUS_API_BASE", "https://api.tokenfactory.nebius.com/v1/"
        )
        return LiteLlm(
            model="nebius/Qwen/Qwen3-30B-A3B-Thinking-2507",
            api_base=api_base,
            api_key=nebius_key,
        )
    # Fallback to Gemini if no Nebius key
    return "gemini-2.0-flash-exp"


def parse_text_tool_call(
    *, callback_context, llm_response: LlmResponse
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

    # Try to find JSON tool_call in the text
    # Pattern: {"tool_call": {"name": "...", "arguments": {...}}}
    tool_call_match = re.search(
        r'\{\s*"tool_call"\s*:\s*\{[^}]*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^}]*\}\s*\}\s*\}',
        full_text,
        re.DOTALL,
    )

    if not tool_call_match:
        # Try a more lenient pattern for nested arguments
        tool_call_match = re.search(
            r'\{"tool_call":\s*(\{.*?\})\s*\}',
            full_text,
            re.DOTALL,
        )
        if tool_call_match:
            try:
                # Try to parse the full match
                inner = tool_call_match.group(0)
                parsed = json.loads(inner)
                tool_call = parsed.get("tool_call")
            except json.JSONDecodeError:
                return None
        else:
            return None
    else:
        try:
            parsed = json.loads(tool_call_match.group(0))
            tool_call = parsed.get("tool_call")
        except json.JSONDecodeError:
            return None

    if not tool_call or "name" not in tool_call:
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

            Parameters:
            - domain: Evaluation domain (airline, retail, telecom, mock)
            - agent_endpoint: A2A endpoint of agent to evaluate
            - user_llm: LLM model for user simulator (default: gpt-4o)
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