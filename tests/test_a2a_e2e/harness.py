"""Test harness: minimal A2A server backed by gpt-4o.

Provides a TestAgentExecutor that proxies user messages to OpenAI's gpt-4o
and a build_test_server() factory that assembles the full A2A server stack.
"""

from a2a.server.agent_execution.agent_executor import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.apps.jsonrpc.fastapi_app import A2AFastAPIApplication
from a2a.server.events.event_queue import EventQueue
from a2a.server.request_handlers.default_request_handler import DefaultRequestHandler
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from a2a.utils.message import new_agent_text_message
from openai import OpenAI


class TestAgentExecutor(AgentExecutor):
    """Minimal AgentExecutor that proxies messages to gpt-4o.

    Receives user messages containing mock domain tools and policy
    (formatted as <available_tools> and <policy> XML blocks by the
    tau2 A2A translation layer). Forwards the full text to gpt-4o
    and returns the response as an A2A Message.
    """

    __test__ = False  # Prevent pytest collection

    def __init__(self) -> None:
        self._client = OpenAI()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        user_input = context.get_user_input()

        response = self._client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": user_input}],
        )

        response_text = response.choices[0].message.content or ""

        message = new_agent_text_message(
            text=response_text,
            context_id=context.context_id,
            task_id=context.task_id,
        )
        await event_queue.enqueue_event(message)
        await event_queue.close()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        await event_queue.close()


def _build_agent_card(url: str) -> AgentCard:
    """Build the AgentCard for the test server."""
    return AgentCard(
        name="tau2-test-agent",
        description="Test agent for A2A E2E testing",
        url=url,
        version="1.0.0",
        capabilities=AgentCapabilities(),
        skills=[
            AgentSkill(
                id="general",
                name="General Assistance",
                description="General customer service assistance",
                tags=["general"],
            )
        ],
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
    )


def build_test_server(url: str = "http://localhost:0"):
    """Build a complete A2A test server as a FastAPI ASGI app.

    Returns:
        A FastAPI application ready to be served by uvicorn.
    """
    agent_card = _build_agent_card(url)
    executor = TestAgentExecutor()
    task_store = InMemoryTaskStore()
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store,
    )
    a2a_app = A2AFastAPIApplication(
        agent_card=agent_card,
        http_handler=handler,
    )
    return a2a_app.build()
