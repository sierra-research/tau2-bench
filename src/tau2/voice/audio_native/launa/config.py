"""Public endpoint configuration; no model paths or private service defaults."""

from urllib.parse import urlsplit


def validate_endpoint(endpoint: str, allow_insecure_local: bool = False) -> str:
    if not endpoint:
        raise ValueError("LAUNA_WS_URL is required")
    parsed = urlsplit(endpoint)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Do not put credentials, queries, or fragments in the URL")
    loopback = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    if not parsed.hostname:
        raise ValueError("Endpoint hostname is required")
    if parsed.scheme != "wss" and not (
        parsed.scheme == "ws" and loopback and allow_insecure_local
    ):
        raise ValueError("Use wss; plaintext is restricted to explicit loopback tests")
    return endpoint
