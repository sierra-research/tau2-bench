"""A2A HTTP client for communicating with remote A2A agents."""

import contextlib
import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx
from loguru import logger

from tau2.a2a.exceptions import (
    A2AAuthError,
    A2ADiscoveryError,
    A2AError,
    A2AMessageError,
    A2ATimeoutError,
)
from tau2.a2a.metrics import ProtocolMetrics, estimate_tokens
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
        http_client: httpx.AsyncClient | None = None,
    ):
        """
        Initialize A2A client.

        Args:
            config: A2A configuration bundle
            http_client: Optional pre-configured httpx client for testing
        """
        self.config = config
        self._http_client = http_client
        self._agent_card: AgentCard | None = None
        self._owned_client = http_client is None
        self._metrics: list[ProtocolMetrics] = []

    def _create_http_client(self) -> httpx.AsyncClient:
        """
        Create a configured httpx.AsyncClient for performing HTTP requests.

        Returns:
            httpx.AsyncClient: An AsyncClient configured with the client's timeout, SSL verification setting, default JSON headers, and redirects enabled.
        """
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
        """
        Provide an async context manager that yields an httpx.AsyncClient.

        Yields the externally supplied client when one was provided to the instance; otherwise creates a new AsyncClient for the context and ensures it is closed on exit. The caller remains responsible for managing the lifecycle of an external client.
        """
        if self._http_client is not None:
            # External client provided - reuse it (caller manages lifecycle)
            yield self._http_client
        else:
            # Create fresh client for this request
            async with self._create_http_client() as client:
                yield client

    def _get_url(self, path: str = "") -> str:
        """Build full URL from endpoint and path, ensuring no trailing slash issues."""
        endpoint = self.config.endpoint.rstrip("/")
        if path:
            return f"{endpoint}/{path.lstrip('/')}"
        return endpoint

    def _build_headers(self) -> dict[str, str]:
        """
        Construct standard JSON HTTP headers and include an Authorization Bearer header when an auth token is configured.

        Returns:
            dict[str, str]: HTTP headers with "Content-Type" and "Accept" set to "application/json", and "Authorization" set to "Bearer <token>" if present in the client config.
        """
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        # Add authentication if provided
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
        # Return cached agent card if available
        if self._agent_card is not None:
            return self._agent_card

        try:
            async with self._http_client_context() as client:
                logger.debug(
                    "Discovering A2A agent",
                    endpoint=self.config.endpoint,
                )

                # Fetch agent card
                response = await client.get(
                    self._get_url(".well-known/agent-card.json"),
                    headers=self._build_headers(),
                )

            # Handle errors
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

            # Parse agent card
            try:
                agent_card_data = response.json()
                agent_card = AgentCard(**agent_card_data)
            except (json.JSONDecodeError, ValueError) as e:
                logger.error("Failed to parse agent card", error=str(e))
                msg = f"Invalid agent card format: {e}"
                raise A2ADiscoveryError(
                    msg,
                    endpoint=self.config.endpoint,
                ) from e

            # Cache and return
            self._agent_card = agent_card

            logger.info(
                "Successfully discovered A2A agent",
                agent_name=agent_card.name,
                agent_version=agent_card.version,
                endpoint=self.config.endpoint,
            )

            return agent_card

        except httpx.TimeoutException as e:
            logger.error("Agent discovery timed out", endpoint=self.config.endpoint)
            msg = "Agent discovery timed out"
            raise A2ATimeoutError(
                msg,
                timeout=self.config.timeout,
            ) from e

        except httpx.HTTPError as e:
            logger.error(
                "Agent discovery failed",
                endpoint=self.config.endpoint,
                error=str(e),
            )
            msg = f"Agent discovery failed: {e}"
            raise A2ADiscoveryError(
                msg,
                endpoint=self.config.endpoint,
            ) from e

    async def send_message(
        self,
        message_content: str,
        context_id: str | None = None,
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
        # Generate request ID for metrics tracking
        request_id = str(uuid.uuid4())

        # Start latency tracking
        start_time = time.perf_counter()

        # Count input tokens
        input_tokens = estimate_tokens(message_content)

        # Initialize metrics variables
        status_code = None
        output_tokens = None
        error_msg = None
        response_context_id = None
        response_content = ""

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
                    "Sending A2A message",
                    endpoint=self.config.endpoint,
                    context_id=context_id,
                    message_length=len(message_content),
                    input_tokens=input_tokens,
                )

                # Debug: Log full request payload (only at debug level)
                logger.trace(
                    "A2A request payload",
                    request_id=request_id,
                    payload=rpc_request,
                )

                # Send request
                response = await client.post(
                    self._get_url(), json=rpc_request, headers=self._build_headers()
                )

            status_code = response.status_code

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
                    # Debug: Log full error response for troubleshooting
                    logger.trace(
                        "A2A error response",
                        request_id=request_id,
                        status_code=response.status_code,
                        error_data=error_data,
                    )
                    if "error" in error_data:
                        error_msg = f"{error_msg}: {error_data['error']}"
                except Exception:
                    # Debug: Log raw response text if JSON parsing fails
                    logger.trace(
                        "A2A error response (raw)",
                        request_id=request_id,
                        status_code=response.status_code,
                        raw_text=response.text[:1000],  # Limit to 1000 chars
                    )

                raise A2AError(
                    error_msg,
                    status_code=response.status_code,
                )

            # Parse JSON-RPC response
            try:
                rpc_response = response.json()

                # Debug: Log full response payload (only at trace level)
                logger.trace(
                    "A2A response payload",
                    request_id=request_id,
                    payload=rpc_response,
                )

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
                # When server returns a Message (not Task), parts are at result level
                if not response_texts:
                    direct_parts = result.get("parts", [])
                    for part in direct_parts:
                        if "text" in part:
                            response_texts.append(part["text"])

                # Format 3: TaskStatusUpdateEvent - status.message.parts (ADK streaming)
                if not response_texts:
                    status = result.get("status", {})
                    status_message = status.get("message", {})
                    status_parts = status_message.get("parts", [])
                    for part in status_parts:
                        if "text" in part:
                            response_texts.append(part["text"])

                # Format 4: Legacy wrapper format - result.message.parts
                # Some implementations wrap the message in a 'message' field
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

                # Debug: Log actual response content
                if response_content:
                    logger.debug(
                        "A2A agent response content",
                        content_preview=response_content[:500],
                        content_length=len(response_content),
                    )
                else:
                    logger.warning(
                        "A2A agent returned empty response",
                        request_id=request_id,
                        result_keys=list(result.keys()),
                        artifacts_count=len(artifacts),
                    )

                # Count output tokens
                output_tokens = estimate_tokens(response_content)

                # Extract context_id - try multiple locations
                response_context_id = (
                    result.get("contextId")  # Google ADK format
                    or result.get("message", {}).get("contextId")  # Standard A2A
                )

                # Calculate latency
                latency_ms = (time.perf_counter() - start_time) * 1000

                # Log structured metrics
                logger.info(
                    "A2A message exchange completed",
                    request_id=request_id,
                    endpoint=self.config.endpoint,
                    status_code=status_code,
                    latency_ms=round(latency_ms, 2),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    context_id=response_context_id,
                )

                # Create and store metrics
                metrics = ProtocolMetrics(
                    request_id=request_id,
                    endpoint=self.config.endpoint,
                    method="POST",
                    status_code=status_code,
                    latency_ms=latency_ms,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    context_id=response_context_id,
                )
                self._metrics.append(metrics)

                return response_content, response_context_id

            except (json.JSONDecodeError, KeyError, TypeError) as e:
                error_msg = f"Invalid A2A response format: {e}"
                # Debug: Log raw response for parsing errors
                logger.trace(
                    "A2A response parsing failed",
                    request_id=request_id,
                    error=str(e),
                    raw_response=response.text[:2000],  # Limit to 2000 chars
                )
                logger.error("Failed to parse A2A response", error=str(e))
                raise A2AMessageError(error_msg) from e

        except httpx.TimeoutException as e:
            # Calculate latency even for timeout
            latency_ms = (time.perf_counter() - start_time) * 1000
            error_msg = "Agent response timeout"

            # Log error with metrics
            logger.error(
                "A2A message timeout",
                request_id=request_id,
                endpoint=self.config.endpoint,
                timeout=self.config.timeout,
                latency_ms=round(latency_ms, 2),
                input_tokens=input_tokens,
            )

            # Record metrics for timeout
            metrics = ProtocolMetrics(
                request_id=request_id,
                endpoint=self.config.endpoint,
                method="POST",
                status_code=status_code,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                context_id=context_id,
                error=error_msg,
            )
            self._metrics.append(metrics)

            raise A2ATimeoutError(
                error_msg,
                timeout=self.config.timeout,
            ) from e

        except httpx.HTTPError as e:
            # Calculate latency for HTTP errors
            latency_ms = (time.perf_counter() - start_time) * 1000
            error_msg = f"Failed to send message: {e}"

            # Log error with metrics
            logger.error(
                "A2A message send failed",
                request_id=request_id,
                endpoint=self.config.endpoint,
                error=str(e),
                latency_ms=round(latency_ms, 2),
                input_tokens=input_tokens,
                status_code=getattr(e, "status_code", None),
            )

            # Record metrics for HTTP error
            metrics = ProtocolMetrics(
                request_id=request_id,
                endpoint=self.config.endpoint,
                method="POST",
                status_code=status_code or getattr(e, "status_code", None),
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                context_id=context_id,
                error=error_msg,
            )
            self._metrics.append(metrics)

            raise A2AError(
                error_msg,
                status_code=getattr(e, "status_code", None),
            ) from e

    def get_metrics(self) -> list[ProtocolMetrics]:
        """
        Get all collected protocol metrics.

        Returns:
            List of ProtocolMetrics for all requests made by this client
        """
        return self._metrics.copy()

    def clear_metrics(self) -> None:
        """Clear all collected metrics."""
        self._metrics.clear()

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
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit."""
        await self.close()
