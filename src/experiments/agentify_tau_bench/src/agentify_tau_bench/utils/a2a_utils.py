import asyncio
import uuid
from typing import Optional

import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    Message,
    MessageSendParams,
    Part,
    Role,
    SendMessageRequest,
    SendMessageResponse,
    TextPart,
)
from loguru import logger


async def get_agent_card(url: str) -> AgentCard | None:
    """
    Get the agent card from the A2A server.

    Args:
        url: The URL of the A2A server.

    Returns:
        The agent card if found, None otherwise.
    """
    httpx_client = httpx.AsyncClient()
    resolver = A2ACardResolver(httpx_client=httpx_client, base_url=url)

    card: AgentCard | None = await resolver.get_agent_card()

    return card


async def wait_agent_ready(url: str, timeout: int = 10) -> bool:
    """
    Wait until the A2A server is ready, check by getting the agent card.

    Args:
        url: The URL of the A2A server.
        timeout: The timeout in seconds.

    Returns:
        True if the A2A server is ready, False otherwise.
    """
    retry_cnt = 0
    while retry_cnt < timeout:
        retry_cnt += 1
        try:
            card = await get_agent_card(url)
            if card is not None:
                return True
            else:
                logger.info(
                    f"Agent card not available yet..., retrying {retry_cnt}/{timeout}"
                )
        except Exception:
            pass
        await asyncio.sleep(1)
    return False


async def a2a_send_message(
    url: str,
    message: str,
    task_id: Optional[str] = None,
    context_id: Optional[str] = None,
) -> SendMessageResponse:
    """
    Send a message to the A2A server.

    Args:
        url: The URL of the A2A server.
        message: The message to send.
        task_id: The task ID.
        context_id: The context ID.

    Returns:
        The response from the A2A server.
    """
    card = await get_agent_card(url)
    httpx_client = httpx.AsyncClient(timeout=120.0)
    client = A2AClient(httpx_client=httpx_client, agent_card=card)

    message_id = uuid.uuid4().hex
    params = MessageSendParams(
        message=Message(
            role=Role.user,
            parts=[Part(TextPart(text=message))],
            message_id=message_id,
            task_id=task_id,
            context_id=context_id,
        )
    )
    request_id = uuid.uuid4().hex
    req = SendMessageRequest(id=request_id, params=params)
    response = await client.send_message(request=req)
    return response
