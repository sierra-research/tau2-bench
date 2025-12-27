"""
ListDomains tool for ADK agent.

This tool enables external agents to discover available evaluation domains.
"""

from typing import Any

from google.adk.tools import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from loguru import logger

# Domain descriptions (tau2 doesn't store these in registry)
DOMAIN_DESCRIPTIONS = {
    "airline": "Airline customer service (flights, bookings, cancellations)",
    "retail": "Retail e-commerce (orders, returns, exchanges)",
    "telecom": "Telecommunications support (technical issues, billing)",
    "telecom-workflow": "Telecommunications with workflow-based policy",
    "mock": "Simple test domain for development",
}


class ListDomains(BaseTool):
    """List available tau2-bench evaluation domains"""

    name = "list_domains"
    description = (
        "List all available tau2-bench evaluation domains and their descriptions"
    )

    def _get_declaration(self) -> types.FunctionDeclaration | None:
        """
        Create a FunctionDeclaration describing this tool for integration with GenAI tooling.

        Returns:
            types.FunctionDeclaration | None: A FunctionDeclaration with a minimal but valid
            parameters schema. Gemini requires at least an OBJECT type with properties.
        """
        # Gemini requires a valid OBJECT schema with at least one property defined.
        # We add an optional 'verbose' flag that doesn't affect tool behavior
        # but satisfies Gemini's function calling schema requirements.
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "verbose": types.Schema(
                        type=types.Type.BOOLEAN,
                        description="Return verbose output with task counts (optional, default: true)",
                    ),
                },
                required=[],
            ),
        )

    async def run_async(
        self,
        *,
        args: dict[str, Any],  # noqa: ARG002
        tool_context: ToolContext,  # noqa: ARG002
    ) -> Any:
        """
        Discover available tau2-bench evaluation domains and return metadata for each domain.

        Parameters:
            _args (dict[str, Any]): Ignored; reserved for tool invocation parameters.

        Returns:
            dict: A mapping with key "domains" to a list of domain info objects. Each domain info object contains:
                - name (str): Domain identifier.
                - description (str): Human-readable description (from DOMAIN_DESCRIPTIONS or "<domain> domain" fallback).
                - num_tasks (int | None): Number of tasks in the domain, or `None` if tasks could not be loaded (load failures are logged).
        """
        from tau2.registry import registry
        from tau2.run import load_tasks

        domains_info = []
        for domain_name in registry.get_domains():
            try:
                # Get task count from tau2's task loader
                tasks = load_tasks(domain_name)
                num_tasks = len(tasks)
            except Exception as e:
                logger.warning(f"Could not load tasks for domain {domain_name}: {e}")
                num_tasks = None

            domains_info.append(
                {
                    "name": domain_name,
                    "description": DOMAIN_DESCRIPTIONS.get(
                        domain_name, f"{domain_name} domain"
                    ),
                    "num_tasks": num_tasks,
                }
            )

        return {"domains": domains_info}
