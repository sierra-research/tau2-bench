"""Apriel transport authentication shared by Tau's provider and launcher."""

import http.client
import json
import os
import re
import ssl
import urllib.request
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

FORBIDDEN_QUERY_KEYS = {
    "api_key",
    "access_token",
    "token",
    "authorization",
    "client_secret",
}


def normalize_realtime_url(value):
    try:
        parts = urlsplit(value)
        port = parts.port
    except (TypeError, ValueError):
        raise ValueError("Invalid Apriel realtime URL") from None
    scheme = {"http": "ws", "https": "wss", "ws": "ws", "wss": "wss"}.get(parts.scheme)
    if (
        not scheme
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.fragment
    ):
        raise ValueError(
            "Apriel realtime URL must be HTTP(S) or WS(S), without user credentials or a fragment"
        )
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("Invalid Apriel realtime URL port")
    if any(
        k.lower() in FORBIDDEN_QUERY_KEYS
        for k, _ in parse_qsl(parts.query, keep_blank_values=True)
    ):
        raise ValueError(
            "Send Apriel credentials in headers or subprotocols, never the URL"
        )
    path = parts.path.rstrip("/")
    if not path:
        path = "/v1/realtime"
    elif path.endswith("/v1"):
        path += "/realtime"
    elif not path.endswith("/v1/realtime"):
        raise ValueError("Apriel URL must be an origin, /v1, or /v1/realtime")
    return urlunsplit((scheme, parts.netloc, path, parts.query, ""))


def _secret_file(name):
    filename = os.environ.get(name)
    if not filename:
        return None
    try:
        value = Path(filename).read_text().splitlines()[0].strip()
    except (OSError, IndexError):
        raise ValueError("Cannot read " + name) from None
    return value or None


def resolve_credentials(api_key=None, cloud_auth_token=None):
    key = (
        api_key
        or os.environ.get("APRIEL_REALTIME_API_KEY")
        or _secret_file("APRIEL_REALTIME_API_KEY_FILE")
        or _secret_file("APRIEL_API_KEY_FILE")
        or os.environ.get("APRIEL_API_KEY")
    )
    cloud = (
        os.environ.get("APRIEL_CLOUD_AUTH_TOKEN", "")
        if cloud_auth_token is None
        else cloud_auth_token
    )
    if not key:
        raise ValueError("Set APRIEL_REALTIME_API_KEY")
    if not isinstance(key, str) or not re.fullmatch(
        r"[!#$%&'*+.^_`|~0-9A-Za-z-]{1,4096}", key
    ):
        raise ValueError("Invalid APRIEL_REALTIME_API_KEY format")
    if (
        not isinstance(cloud, str)
        or len(cloud) > 16384
        or any(ord(c) < 33 or ord(c) > 126 for c in cloud)
    ):
        raise ValueError("Invalid APRIEL_CLOUD_AUTH_TOKEN format")
    return key, cloud


def tls_server_name(url, override=None):
    parts = urlsplit(url)
    if parts.scheme not in {"https", "wss"}:
        return None
    host = parts.hostname or ""
    name = override or os.environ.get("APRIEL_REALTIME_TLS_SERVER_NAME")
    if name:
        if not re.fullmatch(r"[A-Za-z0-9.-]+", name):
            raise ValueError("Invalid APRIEL_REALTIME_TLS_SERVER_NAME")
        return name
    return host.replace("_", "-") if "_" in host else None


def websocket_request(base_url, model, api_key=None, cloud_auth_token=None):
    key, cloud = resolve_credentials(api_key, cloud_auth_token)
    parts = urlsplit(normalize_realtime_url(base_url))
    query = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if k != "model"
    ]
    query.append(("model", model))
    url = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ""))
    options = {"additional_headers": {"Authorization": "Bearer " + (cloud or key)}}
    if cloud:
        options["subprotocols"] = ["realtime", "openai-insecure-api-key." + key]
    server_name = tls_server_name(url)
    if server_name:
        options["server_hostname"] = server_name
    return url, options


def model_request(base_url, api_key=None, cloud_auth_token=None):
    key, cloud = resolve_credentials(api_key, cloud_auth_token)
    parts = urlsplit(normalize_realtime_url(base_url))
    url = urlunsplit(
        (
            "https" if parts.scheme == "wss" else "http",
            parts.netloc,
            parts.path.removesuffix("/realtime") + "/models",
            "",
            "",
        )
    )
    headers = {"Authorization": "Bearer " + (cloud or key)}
    if cloud:
        headers["X-Apriel-Authorization"] = "Bearer " + key
    return urllib.request.Request(url, headers=headers)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Apriel model discovery refused an HTTP redirect")


def resolve_model(base_url, api_key=None, cloud_auth_token=None):
    request = model_request(base_url, api_key, cloud_auth_token)
    context = ssl.create_default_context()
    name = tls_server_name(request.full_url)

    class Connection(http.client.HTTPSConnection):
        def connect(self):
            http.client.HTTPConnection.connect(self)
            self.sock = self._context.wrap_socket(
                self.sock, server_hostname=name or self._tunnel_host or self.host
            )

    class Handler(urllib.request.HTTPSHandler):
        def https_open(self, req):
            return self.do_open(
                Connection,
                req,
                context=self._context,
            )

    opener = urllib.request.build_opener(NoRedirect(), Handler(context=context))
    with opener.open(request, timeout=30) as response:
        models = json.load(response).get("data", [])
    ids = [
        m["id"] for m in models if isinstance(m, dict) and isinstance(m.get("id"), str)
    ]
    if len(ids) != 1:
        raise RuntimeError(
            "Expected exactly one attached realtime model from /v1/models"
        )
    return ids[0]
