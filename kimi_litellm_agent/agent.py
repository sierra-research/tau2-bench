"""
Kimi K2 Thinking agent using LiteLLM with Nebius TokenFactory API.

This agent uses the Moonshot AI Kimi-K2-Thinking model hosted on Nebius
TokenFactory, accessed via LiteLLM's native Nebius provider.

Authentication:
- Requires NEBIUS_API_KEY environment variable
- API base URL: https://api.tokenfactory.nebius.com/v1/
"""

import os

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm


def create_agent() -> LlmAgent:
    """
    Create an ADK agent configured with Kimi K2 Thinking via LiteLLM.

    Uses the Nebius TokenFactory API endpoint with the NEBIUS_API_KEY
    environment variable for authentication.

    Returns:
        LlmAgent configured to use Kimi K2 Thinking model
    """
    # Nebius TokenFactory configuration
    # Note: TokenFactory uses a different base URL than Nebius AI Studio
    api_base = os.getenv(
        "NEBIUS_API_BASE", "https://api.tokenfactory.nebius.com/v1/"
    )
    api_key = os.getenv("NEBIUS_API_KEY")

    # Model can be overridden via environment variable
    # Default: Kimi K2 Thinking from Moonshot AI
    model_name = os.getenv("KIMI_AGENT_MODEL", "moonshotai/Kimi-K2-Thinking")

    # LiteLLM uses "nebius/" prefix for Nebius provider
    litellm_model = f"nebius/{model_name}"

    # Create LiteLLM model with Nebius TokenFactory endpoint
    llm_model = LiteLlm(
        model=litellm_model,
        api_base=api_base,
        api_key=api_key,
    )

    # Instruction for tau2-bench compatibility
    instruction = """You are a helpful customer service assistant.

When helping customers, you have access to tools that are described in the user's message within <available_tools> tags.

IMPORTANT: To use a tool, you MUST respond with ONLY a JSON object in this exact format:
{"tool_call": {"name": "tool_name", "arguments": {"param1": "value1"}}}

For example, to check network status:
{"tool_call": {"name": "check_network_status", "arguments": {}}}

Rules:
1. Read the available tools carefully from the user's message
2. When you need information, call the appropriate tool using the JSON format above
3. After receiving tool results, provide helpful guidance to the customer
4. Be polite and professional
5. If no tools are needed, respond with helpful text directly

Always respond with either a tool call JSON or a helpful text message - never leave your response empty."""

    # Create agent with LiteLLM configuration
    agent = LlmAgent(
        model=llm_model,
        name="kimi_litellm_agent",
        description="A customer service agent using Kimi K2 Thinking (Nebius) for tau2-bench evaluation",
        instruction=instruction,
    )

    return agent


# Create the agent instance (used by ADK CLI)
# ADK looks for 'root_agent' by default
root_agent = create_agent()
agent = root_agent  # Alias for backward compatibility


if __name__ == "__main__":
    print("Kimi LiteLLM Agent")
    print(f"Name: {agent.name}")
    print(f"Description: {agent.description}")
    print(f"Model: {agent.model}")
