"""A2A HTTP client for communicating with remote A2A agents."""

import contextlib
import json
import uuid
from collections.abc import AsyncIterator
from typing import Any, Optional

import httpx
from loguru import logger

from tau2.a2a.exceptions import (
    A2AAuthError,
    A2ADiscoveryError,
    A2AError,
    A2AMessageError,
    A2ATimeoutError,
)
from tau2.a2a.models import A2AConfig, AgentCard


class A2AClient:
    """
    HTTP client for A2A protocol communication.

    Handles agent discovery, message sending, and protocol-level concerns
    like authentication and error handling.
    """

    def __init__(
        self,
        config: A2AConfig,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        """
        Initialize A2A client.

        Args:
            config: A2A configuration bundle
            http_client: Optional pre-configured httpx client for testing
        """
        self.config = config
        self._http_client = http_client
        self._agent_card: Optional[AgentCard] = None
        self._owned_client = http_client is None

    def _create_http_client(self) -> httpx.AsyncClient:
        """Create a configured httpx.AsyncClient."""
        return httpx.AsyncClient(
            timeout=httpx.Timeout(
                self.config.timeout,
                connect=self.config.connect_timeout,
            ),
            verify=self.config.verify_ssl,
            headers=self._build_headers(),
            follow_redirects=True,
        )

    @contextlib.asynccontextmanager
    async def _http_client_context(self) -> AsyncIterator[httpx.AsyncClient]:
        """Yield an httpx client, creating one if not externally provided."""
        if self._http_client is not None:
            yield self._http_client
        else:
            async with self._create_http_client() as client:
                yield client

    def _get_url(self, path: str = "") -> str:
        """Build full URL from endpoint and path."""
        endpoint = self.config.endpoint.rstrip("/")
        if path:
            return f"{endpoint}/{path.lstrip('/')}"
        return endpoint

    def _build_headers(self) -> dict[str, str]:
        """Construct HTTP headers with optional auth."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.config.auth_token:
            headers["Authorization"] = f"Bearer {self.config.auth_token}"
        return headers

    async def discover_agent(self) -> AgentCard:
        """
        Discover A2A agent capabilities via agent card.

        Fetches /.well-known/agent-card.json and caches the result.

        Returns:
            AgentCard with agent metadata and capabilities

        Raises:
            A2ADiscoveryError: If discovery fails
        """
        if self._agent_card is not None:
            return self._agent_card

        try:
            async with self._http_client_context() as client:
                logger.debug(f"Discovering A2A agent at {self.config.endpoint}")

                response = await client.get(
                    self._get_url(".well-known/agent-card.json"),
                    headers=self._build_headers(),
                )

            if response.status_code == 401:
                msg = "Agent discovery requires authentication"
                raise A2AAuthError(msg)

            if response.status_code == 404:
                msg = "Agent card not found at /.well-known/agent-card.json"
                raise A2ADiscoveryError(
                    msg,
                    endpoint=self.config.endpoint,
                )

            if response.status_code >= 400:
                msg = f"Agent discovery failed with status {response.status_code}"
                raise A2ADiscoveryError(
                    msg,
                    endpoint=self.config.endpoint,
                )

            try:
                agent_card_data = response.json()
                agent_card = AgentCard(**agent_card_data)
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse agent card: {e}")
                msg = f"Invalid agent card format: {e}"
                raise A2ADiscoveryError(
                    msg,
                    endpoint=self.config.endpoint,
                ) from e

            self._agent_card = agent_card
            logger.info(
                f"Discovered A2A agent '{agent_card.name}' "
                f"(v{agent_card.version}) at {self.config.endpoint}"
            )
            return agent_card

        except httpx.TimeoutException as e:
            logger.error(f"Agent discovery timed out at {self.config.endpoint}")
            msg = "Agent discovery timed out"
            raise A2ATimeoutError(
                msg,
                timeout=self.config.timeout,
            ) from e

        except httpx.HTTPError as e:
            logger.error(f"Agent discovery failed at {self.config.endpoint}: {e}")
            msg = f"Agent discovery failed: {e}"
            raise A2ADiscoveryError(
                msg,
                endpoint=self.config.endpoint,
            ) from e

    async def send_message(
        self,
        message_content: str,
        context_id: Optional[str] = None,
    ) -> tuple[str, str]:
        """
        Send message to A2A agent using JSON-RPC 2.0 protocol.

        Args:
            message_content: Text content to send to agent
            context_id: Optional session context ID for multi-turn conversations

        Returns:
            Tuple of (response_content, context_id)

        Raises:
            A2AError: If message sending fails
            A2ATimeoutError: If request times out
            A2AAuthError: If authentication fails
        """
        request_id = str(uuid.uuid4())

        try:
            async with self._http_client_context() as client:
                # Build JSON-RPC request
                rpc_request = {
                    "jsonrpc": "2.0",
                    "id": str(uuid.uuid4()),
                    "method": "message/send",
                    "params": {
                        "message": {
                            "messageId": str(uuid.uuid4()),
                            "role": "user",
                            "parts": [{"text": message_content}],
                            "contextId": context_id,
                        }
                    },
                }

                logger.debug(
                    f"Sending A2A message to {self.config.endpoint} "
                    f"(context_id={context_id}, length={len(message_content)})"
                )
                logger.trace(f"A2A request payload (id={request_id}): {rpc_request}")

                # Send request
                response = await client.post(
                    self._get_url(), json=rpc_request, headers=self._build_headers()
                )

            # Handle HTTP errors
            if response.status_code == 401:
                msg = "Authentication failed"
                raise A2AAuthError(msg)

            if response.status_code == 408:
                msg = "Agent response timeout"
                raise A2ATimeoutError(
                    msg,
                    timeout=self.config.timeout,
                )

            if response.status_code >= 400:
                error_msg = f"Message send failed with status {response.status_code}"
                try:
                    error_data = response.json()
                    logger.trace(
                        f"A2A error response (id={request_id}, "
                        f"status={response.status_code}): {error_data}"
                    )
                    if "error" in error_data:
                        error_msg = f"{error_msg}: {error_data['error']}"
                except Exception:
                    logger.trace(
                        f"A2A error response raw (id={request_id}, "
                        f"status={response.status_code}): {response.text[:1000]}"
                    )

                raise A2AError(
                    error_msg,
                    status_code=response.status_code,
                )

            # Parse JSON-RPC response
            try:
                rpc_response = response.json()
                logger.trace(f"A2A response payload (id={request_id}): {rpc_response}")

                # Check for JSON-RPC error
                if "error" in rpc_response:
                    error_info = rpc_response["error"]
                    error_detail = error_info.get("message", "Unknown error")
                    msg = f"Agent returned error: {error_detail}"
                    raise A2AMessageError(msg)

                # Extract result
                result = rpc_response.get("result", {})

                # Extract response content - handle multiple A2A response formats
                response_texts = []

                # Format 1: Google ADK style - artifacts array
                artifacts = result.get("artifacts", [])
                if artifacts:
                    for artifact in artifacts:
                        artifact_parts = artifact.get("parts", [])
                        for part in artifact_parts:
                            if "text" in part:
                                response_texts.append(part["text"])

                # Format 2: Direct Message response - result.parts (per A2A spec)
                if not response_texts:
                    direct_parts = result.get("parts", [])
                    for part in direct_parts:
                        if "text" in part:
                            response_texts.append(part["text"])

                # Format 3: TaskStatusUpdateEvent - status.message.parts
                if not response_texts:
                    status = result.get("status", {})
                    status_message = status.get("message", {})
                    status_parts = status_message.get("parts", [])
                    for part in status_parts:
                        if "text" in part:
                            response_texts.append(part["text"])

                # Format 4: Legacy wrapper format - result.message.parts
                if not response_texts:
                    result_message = result.get("message", {})
                    message_parts = result_message.get("parts", [])
                    for part in message_parts:
                        if "text" in part:
                            response_texts.append(part["text"])

                # Format 5: History-based - last agent message
                if not response_texts:
                    history = result.get("history", [])
                    for msg in reversed(history):
                        if msg.get("role") == "agent":
                            msg_parts = msg.get("parts", [])
                            for part in msg_parts:
                                if "text" in part:
                                    response_texts.append(part["text"])
                            break

                response_content = "\n".join(response_texts)

                if not response_content:
                    logger.warning(
                        f"A2A agent returned empty response (id={request_id}, "
                        f"result_keys={list(result.keys())}, "
                        f"artifacts_count={len(artifacts)})"
                    )

                # Extract context_id - try multiple locations
                response_context_id = (
                    result.get("contextId")  # Google ADK format
                    or result.get("message", {}).get("contextId")  # Standard A2A
                )

                logger.info(
                    f"A2A message exchange completed (id={request_id}, "
                    f"status={response.status_code}, "
                    f"context_id={response_context_id})"
                )

                return response_content, response_context_id

            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.trace(
                    f"A2A response parsing failed (id={request_id}): {e}, "
                    f"raw={response.text[:2000]}"
                )
                logger.error(f"Failed to parse A2A response: {e}")
                error_msg = f"Invalid A2A response format: {e}"
                raise A2AMessageError(error_msg) from e

        except httpx.TimeoutException as e:
            logger.error(
                f"A2A message timeout (id={request_id}, "
                f"endpoint={self.config.endpoint}, timeout={self.config.timeout})"
            )
            error_msg = "Agent response timeout"
            raise A2ATimeoutError(
                error_msg,
                timeout=self.config.timeout,
            ) from e

        except httpx.HTTPError as e:
            logger.error(
                f"A2A message send failed (id={request_id}, "
                f"endpoint={self.config.endpoint}): {e}"
            )
            error_msg = f"Failed to send message: {e}"
            raise A2AError(
                error_msg,
                status_code=getattr(e, "status_code", None),
            ) from e

    async def close(self) -> None:
        """Close HTTP client if owned by this instance."""
        if self._owned_client and self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    async def __aenter__(self) -> "A2AClient":
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> None:
        """Async context manager exit."""
        await self.close()
