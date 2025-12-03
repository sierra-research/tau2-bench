"""White agent implementation - the target agent being tested."""

import os
import dotenv
import uvicorn
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from a2a.utils import new_agent_text_message
from litellm import completion
from loguru import logger

dotenv.load_dotenv()


def prepare_white_agent_card(url):
    skill = AgentSkill(
        id="task_fulfillment",
        name="Task Fulfillment",
        description="Handles user requests and completes tasks",
        tags=["general"],
        examples=[],
    )
    card = AgentCard(
        name="general_white_agent",
        description="A general-purpose white agent for task fulfillment.",
        url=url,
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(),
        skills=[skill],
    )
    return card


class GeneralWhiteAgentExecutor(AgentExecutor):
    def __init__(self):
        self.ctx_id_to_messages = {}

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # parse the task
        user_input = context.get_user_input()
        if context.context_id not in self.ctx_id_to_messages:
            self.ctx_id_to_messages[context.context_id] = []
        messages = self.ctx_id_to_messages[context.context_id]
        messages.append(
            {
                "role": "user",
                "content": user_input,
            }
        )
        if os.environ.get("LITELLM_PROXY_API_KEY") is not None:
            response = completion(
                messages=messages,
                model="openrouter/openai/gpt-4o",
                custom_llm_provider="litellm_proxy",
                temperature=0.0,
            )
        else:
            response = completion(
                messages=messages,
                model="openai/gpt-4o",
                custom_llm_provider="openai",
                temperature=0.0,
            )
        next_message = response.choices[0].message.model_dump()  # type: ignore
        messages.append(
            {
                "role": "assistant",
                "content": next_message["content"],
            }
        )
        await event_queue.enqueue_event(
            new_agent_text_message(
                next_message["content"], context_id=context.context_id
            )
        )

    async def cancel(self, context, event_queue) -> None:
        raise NotImplementedError


def start_white_agent(agent_name="general_white_agent", host="localhost", port=9002):
    logger.info("Starting white agent...")
    # url = f"http://{host}:{port}"
    card = prepare_white_agent_card(os.getenv("AGENT_URL"))

    request_handler = DefaultRequestHandler(
        agent_executor=GeneralWhiteAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=card,
        http_handler=request_handler,
    )

    # Increase workers and timeout to handle concurrent requests better
    uvicorn.run(
        app.build(),
        host=host,
        port=port,
        timeout_keep_alive=300,
        limit_concurrency=50,
    )
