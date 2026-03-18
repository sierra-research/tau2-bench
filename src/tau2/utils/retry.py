"""Retry utilities for external API calls using tenacity."""

import httpx
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from tau2.config import (
    DEFAULT_RETRY_ATTEMPTS,
    DEFAULT_RETRY_MAX_WAIT,
    DEFAULT_RETRY_MIN_WAIT,
    DEFAULT_RETRY_MULTIPLIER,
)

try:
    import websockets

    _HAS_WEBSOCKETS = True
except ImportError:
    websockets = None
    _HAS_WEBSOCKETS = False


def log_retry_attempt(retry_state):
    """Custom retry logger that safely handles exceptions with curly braces."""
    exception = retry_state.outcome.exception()
    wait_time = retry_state.next_action.sleep
    attempt = retry_state.attempt_number
    max_attempts = retry_state.retry_object.stop.max_attempt_number

    # Include status code if available (e.g., from ElevenLabs ApiError)
    status_code = getattr(exception, "status_code", None)
    status_info = f" (HTTP {status_code})" if status_code else ""

    logger.warning(
        f"[Retry {attempt}/{max_attempts}] {retry_state.fn.__name__} failed{status_info}, "
        f"retrying in {wait_time:.1f}s. Error: {exception.__class__.__name__}: {str(exception)[:200]}"
    )


# Standard retry decorator for external API calls
# Retries on connection errors and timeouts with exponential backoff
api_retry = retry(
    stop=stop_after_attempt(DEFAULT_RETRY_ATTEMPTS),
    wait=wait_exponential(
        multiplier=DEFAULT_RETRY_MULTIPLIER,
        min=DEFAULT_RETRY_MIN_WAIT,
        max=DEFAULT_RETRY_MAX_WAIT,
    ),
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    before_sleep=log_retry_attempt,
    reraise=True,
)


# TTS-specific retry decorator
# Retries on connection errors, timeouts, and transient server errors from TTS APIs
# Works with any API that has a status_code attribute (ElevenLabs, etc.)
# Also handles httpx-specific timeout and connection errors
def _is_retryable_tts_error(e: Exception) -> bool:
    """Check if an exception should trigger a TTS retry."""
    # Connection and timeout errors (built-in)
    if isinstance(e, (ConnectionError, TimeoutError, OSError)):
        return True
    # httpx-specific exceptions (used by ElevenLabs client)
    if isinstance(e, (httpx.TimeoutException, httpx.ConnectError)):
        return True
    # Catch timeout errors from any library by checking class name
    # (covers httpcore.ReadTimeout, urllib3.ReadTimeoutError, requests.ReadTimeout, etc.)
    exc_name = type(e).__name__.lower()
    if "timeout" in exc_name or "readtimeout" in exc_name:
        return True
    # HTTP status code based retries
    if hasattr(e, "status_code") and e.status_code is not None:
        # Retry on rate limiting (429), conflict (409), and server errors (5xx)
        if e.status_code == 429 or e.status_code == 409 or e.status_code >= 500:
            return True
        # Log why we're NOT retrying for other HTTP errors
        logger.debug(
            f"TTS error not retried (HTTP {e.status_code} is not retryable): "
            f"{e.__class__.__name__}"
        )
    return False


tts_retry = retry(
    stop=stop_after_attempt(DEFAULT_RETRY_ATTEMPTS),
    wait=wait_exponential(
        multiplier=DEFAULT_RETRY_MULTIPLIER,
        min=DEFAULT_RETRY_MIN_WAIT,
        max=DEFAULT_RETRY_MAX_WAIT,
    ),
    retry=retry_if_exception(_is_retryable_tts_error),
    before_sleep=log_retry_attempt,
    reraise=True,
)


# WebSocket-specific retry decorator for async functions
# Retries on WebSocket connection failures, network issues, and timeouts
# Does NOT retry on authentication failures (InvalidStatusCode 401, 403)
# Requires: pip install tau2[voice]
if _HAS_WEBSOCKETS:
    websocket_retry = retry(
        stop=stop_after_attempt(DEFAULT_RETRY_ATTEMPTS),
        wait=wait_exponential(
            multiplier=DEFAULT_RETRY_MULTIPLIER,
            min=DEFAULT_RETRY_MIN_WAIT,
            max=DEFAULT_RETRY_MAX_WAIT,
        ),
        retry=retry_if_exception(
            lambda e: (
                isinstance(e, (ConnectionError, TimeoutError, OSError))
                or isinstance(e, websockets.exceptions.WebSocketException)
                and not (
                    isinstance(e, websockets.exceptions.InvalidStatusCode)
                    and e.status_code in (401, 403)
                )
            )
        ),
        before_sleep=log_retry_attempt,
        reraise=True,
    )
else:
    websocket_retry = retry(
        stop=stop_after_attempt(DEFAULT_RETRY_ATTEMPTS),
        wait=wait_exponential(
            multiplier=DEFAULT_RETRY_MULTIPLIER,
            min=DEFAULT_RETRY_MIN_WAIT,
            max=DEFAULT_RETRY_MAX_WAIT,
        ),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        before_sleep=log_retry_attempt,
        reraise=True,
    )
