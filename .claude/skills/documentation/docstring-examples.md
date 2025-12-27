# Google-Style Docstring Examples

Complete examples following the Google Python Style Guide.

---

## Module Docstring

```python
"""HTTP client utilities for A2A protocol communication.

This module provides async HTTP client functionality for sending and
receiving A2A protocol messages. It handles connection pooling, retry
logic, and response validation.

Example:
    Basic usage with the default client::

        from tau2_agent.http import send_message

        response = await send_message(url, payload)

Attributes:
    DEFAULT_TIMEOUT: Default request timeout in seconds.
    MAX_RETRIES: Maximum retry attempts for transient failures.
"""

DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3
```

---

## Function Docstrings

### Simple Function

```python
def calculate_hash(data: bytes) -> str:
    """Compute SHA-256 hash of the given data.

    Args:
        data: Raw bytes to hash.

    Returns:
        Hexadecimal string representation of the hash.
    """
```

### Function with Multiple Parameters

```python
def send_request(
    url: str,
    payload: dict[str, Any],
    timeout: float = 30.0,
    headers: dict[str, str] | None = None,
) -> Response:
    """Send an HTTP POST request with JSON payload.

    Handles connection errors with exponential backoff retry logic.
    The request is considered failed after MAX_RETRIES attempts.

    Args:
        url: Target endpoint URL.
        payload: JSON-serializable request body.
        timeout: Request timeout in seconds.
        headers: Additional HTTP headers to include.

    Returns:
        Response object containing status code and parsed body.

    Raises:
        ConnectionError: If all retry attempts fail.
        ValidationError: If the response body is not valid JSON.
    """
```

### Function with Complex Types

```python
def batch_process(
    items: list[dict[str, Any]],
    processor: Callable[[dict[str, Any]], T],
    max_concurrent: int = 10,
) -> list[T]:
    """Process items concurrently with bounded parallelism.

    Items are processed in batches to limit memory usage. Failed items
    are collected and reported but do not stop processing.

    Args:
        items: List of items to process.
        processor: Callable that transforms each item.
        max_concurrent: Maximum parallel operations.

    Returns:
        List of successfully processed results in original order.
        Failed items are replaced with None.

    Raises:
        ValueError: If max_concurrent is less than 1.
    """
```

### Generator Function

```python
def iter_chunks(data: bytes, chunk_size: int = 1024) -> Iterator[bytes]:
    """Yield fixed-size chunks from a byte sequence.

    The final chunk may be smaller than chunk_size if the data
    length is not evenly divisible.

    Args:
        data: Source bytes to chunk.
        chunk_size: Maximum size of each chunk.

    Yields:
        Byte chunks of up to chunk_size length.

    Example:
        >>> list(iter_chunks(b"hello", 2))
        [b'he', b'll', b'o']
    """
```

### Async Function

```python
async def fetch_user(user_id: int, client: AsyncClient) -> User:
    """Retrieve user data from the remote API.

    Makes an authenticated request to the user service. Results are
    cached for 5 minutes to reduce API load.

    Args:
        user_id: Unique identifier of the user to fetch.
        client: Configured HTTP client with authentication.

    Returns:
        User model populated with remote data.

    Raises:
        UserNotFoundError: If no user exists with the given ID.
        AuthenticationError: If the client credentials are invalid.
    """
```

---

## Class Docstrings

### Basic Class

```python
class EvaluationSession:
    """A single evaluation session tracking agent performance.

    Represents an in-progress or completed evaluation run against
    a specific task. Tracks all messages exchanged and the final
    outcome.

    Attributes:
        session_id: Unique identifier for this session.
        task_id: The task being evaluated.
        status: Current session state (pending, running, completed).
        messages: Ordered list of exchanged messages.
    """

    def __init__(self, task_id: str, config: EvalConfig) -> None:
        """Initialize a new evaluation session.

        Args:
            task_id: Identifier of the task to evaluate.
            config: Evaluation parameters and constraints.
        """
```

### Class with Properties

```python
class ConnectionPool:
    """Manages a pool of reusable HTTP connections.

    Connections are created lazily and returned to the pool after use.
    Idle connections are closed after the configured timeout.

    Attributes:
        max_size: Maximum connections in the pool.
        timeout: Idle timeout before connection cleanup.
    """

    @property
    def active_count(self) -> int:
        """int: Number of connections currently in use."""
        return len(self._active)

    @property
    def available(self) -> bool:
        """bool: Whether the pool can accept new requests."""
        return self.active_count < self.max_size
```

### Dataclass

```python
@dataclass
class TaskResult:
    """Result of a completed evaluation task.

    Attributes:
        task_id: Identifier of the evaluated task.
        success: Whether the task completed successfully.
        score: Numeric score from 0.0 to 1.0.
        duration_ms: Execution time in milliseconds.
        error: Error message if success is False.
    """

    task_id: str
    success: bool
    score: float
    duration_ms: int
    error: str | None = None
```

---

## Method Docstrings

### Instance Method

```python
def add_message(self, role: str, content: str) -> Message:
    """Append a message to the session history.

    Args:
        role: Message sender role (user, assistant, system).
        content: Message text content.

    Returns:
        The created Message object with assigned ID.

    Raises:
        SessionClosedError: If the session has already completed.
    """
```

### Class Method

```python
@classmethod
def from_file(cls, path: Path) -> "Config":
    """Load configuration from a JSON file.

    Args:
        path: Path to the configuration file.

    Returns:
        Config instance with loaded values.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValidationError: If the file contains invalid configuration.
    """
```

### Static Method

```python
@staticmethod
def validate_id(value: str) -> bool:
    """Check if a string is a valid session ID format.

    Valid IDs are 32 hexadecimal characters (UUID without hyphens).

    Args:
        value: String to validate.

    Returns:
        True if the value matches the expected format.
    """
```

---

## Special Cases

### No Return Value

```python
def configure_logging(level: str, output: Path | None = None) -> None:
    """Set up application logging configuration.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        output: Optional file path for log output.
    """
```

### Boolean Return

```python
def is_valid_email(address: str) -> bool:
    """Check if an email address has valid format.

    Performs basic format validation only. Does not verify
    the address exists or can receive mail.

    Args:
        address: Email address to validate.

    Returns:
        True if the format is valid, False otherwise.
    """
```

### Context Manager

```python
@contextmanager
def transaction(self) -> Iterator[Connection]:
    """Execute operations within a database transaction.

    Changes are committed on successful exit. Any exception
    triggers a rollback before re-raising.

    Yields:
        Database connection bound to the transaction.

    Raises:
        DatabaseError: If commit or rollback fails.

    Example:
        >>> with db.transaction() as conn:
        ...     conn.execute("INSERT INTO users ...")
    """
```
