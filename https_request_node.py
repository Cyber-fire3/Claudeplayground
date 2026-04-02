"""
HTTPS Request Node — Python implementation

Makes arbitrary HTTPS (and HTTP) requests, mirroring n8n's HTTP Request node.

Features:
  - All HTTP methods  (GET, POST, PUT, PATCH, DELETE, HEAD, OPTIONS)
  - Request body formats: JSON, form-data, multipart, raw bytes
  - Auth methods: None, API Key (header/query), Bearer token, Basic, OAuth2 token
  - Custom headers, query params, cookies
  - Redirect following, timeout, SSL verification toggles
  - Automatic response parsing (JSON → dict, else → text/bytes)
  - Pagination helper (follow nextPageUrl or offset/cursor patterns)
  - Retry with exponential back-off on transient errors

Install:
  pip install httpx          # primary transport (sync + async)
  pip install requests       # optional fallback
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlencode, urljoin


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

AuthType = Literal["none", "api_key", "bearer", "basic", "oauth2_token"]
BodyFormat = Literal["json", "form", "multipart", "raw"]


@dataclass
class RequestSettings:
    """
    All knobs for an HTTP request live here.

    Parameters
    ----------
    base_url:
        Optional base URL prepended to every request path.
    timeout:
        Seconds before a request times out. None = no timeout.
    verify_ssl:
        Set False to skip certificate verification (dev only).
    follow_redirects:
        Automatically follow 3xx redirects.
    max_retries:
        Number of retry attempts on 5xx or connection errors.
    retry_statuses:
        HTTP status codes that trigger a retry.
    headers:
        Default headers merged into every request.
    cookies:
        Default cookies merged into every request.
    """
    base_url: str = ""
    timeout: float | None = 30.0
    verify_ssl: bool = True
    follow_redirects: bool = True
    max_retries: int = 3
    retry_statuses: list[int] = field(default_factory=lambda: [429, 500, 502, 503, 504])
    headers: dict[str, str] = field(default_factory=dict)
    cookies: dict[str, str] = field(default_factory=dict)


@dataclass
class AuthSettings:
    """
    Authentication configuration.

    Examples
    --------
    API key in header:
        AuthSettings(type="api_key", token="mykey", api_key_name="X-API-Key", api_key_in="header")

    API key as query param:
        AuthSettings(type="api_key", token="mykey", api_key_name="api_key", api_key_in="query")

    Bearer token:
        AuthSettings(type="bearer", token="eyJ...")

    HTTP Basic:
        AuthSettings(type="basic", username="alice", password="s3cr3t")

    Pre-fetched OAuth2 access token:
        AuthSettings(type="oauth2_token", token="ya29...")
    """
    type: AuthType = "none"
    token: str = ""                    # API key, bearer, or OAuth2 token value
    api_key_name: str = "X-API-Key"   # Header or param name for api_key auth
    api_key_in: Literal["header", "query"] = "header"
    username: str = ""                 # Basic auth
    password: str = ""                 # Basic auth


# ---------------------------------------------------------------------------
# Response wrapper
# ---------------------------------------------------------------------------

@dataclass
class Response:
    """
    Normalised HTTP response — provider-agnostic wrapper around httpx/requests.
    """
    status_code: int
    headers: dict[str, str]
    body: Any              # parsed JSON dict/list, or str, or bytes
    raw_bytes: bytes
    url: str
    elapsed_ms: float

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    def raise_for_status(self) -> None:
        if not self.ok:
            raise HTTPError(f"HTTP {self.status_code}: {self.url}\n{self.raw_bytes[:400]!r}")

    def __repr__(self) -> str:
        body_preview = str(self.body)[:80] if self.body else ""
        return f"Response(status={self.status_code}, url={self.url!r}, body={body_preview!r})"


class HTTPError(Exception):
    pass


# ---------------------------------------------------------------------------
# HTTPRequestNode
# ---------------------------------------------------------------------------

class HTTPRequestNode:
    """
    Provider-agnostic HTTP/HTTPS request node.

    Parameters
    ----------
    settings:
        Connection-level settings (base URL, timeout, retries, …).
    auth:
        Authentication settings.

    Usage
    -----
    node = HTTPRequestNode(
        settings=RequestSettings(base_url="https://api.example.com", timeout=10),
        auth=AuthSettings(type="bearer", token="my-token"),
    )
    resp = node.get("/users", params={"page": 1})
    resp.raise_for_status()
    print(resp.body)          # already-parsed JSON dict or list
    """

    def __init__(
        self,
        settings: RequestSettings | None = None,
        auth: AuthSettings | None = None,
    ) -> None:
        self.settings = settings or RequestSettings()
        self.auth = auth or AuthSettings()
        self._client = None   # lazily created httpx client

    # ------------------------------------------------------------------
    # Client lifecycle
    # ------------------------------------------------------------------

    def _get_client(self):
        if self._client is None:
            try:
                import httpx
            except ImportError:
                raise ImportError("pip install httpx")
            self._client = httpx.Client(
                verify=self.settings.verify_ssl,
                follow_redirects=self.settings.follow_redirects,
                timeout=self.settings.timeout,
                headers=self.settings.headers,
                cookies=self.settings.cookies,
            )
        return self._client

    def close(self) -> None:
        """Close the underlying connection pool."""
        if self._client:
            self._client.close()
            self._client = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------
    # Auth injection
    # ------------------------------------------------------------------

    def _apply_auth(
        self,
        headers: dict[str, str],
        params: dict[str, Any],
    ) -> tuple[dict, dict]:
        headers = dict(headers)
        params = dict(params)
        a = self.auth

        if a.type == "bearer" or a.type == "oauth2_token":
            headers["Authorization"] = f"Bearer {a.token}"

        elif a.type == "basic":
            encoded = base64.b64encode(f"{a.username}:{a.password}".encode()).decode()
            headers["Authorization"] = f"Basic {encoded}"

        elif a.type == "api_key":
            if a.api_key_in == "header":
                headers[a.api_key_name] = a.token
            else:
                params[a.api_key_name] = a.token

        return headers, params

    # ------------------------------------------------------------------
    # Core request
    # ------------------------------------------------------------------

    def request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        body: Any = None,
        body_format: BodyFormat = "json",
        cookies: dict[str, str] | None = None,
        files: dict[str, Any] | None = None,
        raw_body: bytes | None = None,
    ) -> Response:
        """
        Make an HTTP request.

        Parameters
        ----------
        method:
            HTTP verb: GET, POST, PUT, PATCH, DELETE, HEAD, OPTIONS.
        url:
            Full URL or a path appended to settings.base_url.
        params:
            Query-string parameters.
        headers:
            Per-request headers (merged with defaults).
        body:
            Request body. Interpreted according to body_format.
        body_format:
            "json"       — body is serialised as JSON  (default)
            "form"       — body is URL-encoded form data
            "multipart"  — body is multipart/form-data (use files= for file parts)
            "raw"        — body sent as-is (bytes)
        cookies:
            Per-request cookies.
        files:
            Multipart file parts: {"field": (filename, bytes, mime_type)}.
        raw_body:
            Bypass body_format and send raw bytes directly.
        """
        # Build full URL
        full_url = urljoin(self.settings.base_url, url) if self.settings.base_url else url

        merged_headers, merged_params = self._apply_auth(headers or {}, params or {})
        merged_cookies = {**self.settings.cookies, **(cookies or {})}

        # Build body kwargs
        send_kwargs: dict[str, Any] = {}
        if raw_body is not None:
            send_kwargs["content"] = raw_body
        elif body is not None:
            if body_format == "json":
                send_kwargs["json"] = body
            elif body_format == "form":
                send_kwargs["data"] = body
            elif body_format == "multipart":
                send_kwargs["data"] = body
                if files:
                    send_kwargs["files"] = files
            elif body_format == "raw":
                send_kwargs["content"] = body if isinstance(body, bytes) else str(body).encode()
        elif files:
            send_kwargs["files"] = files

        # Retry loop
        last_exc: Exception | None = None
        backoff = 1.0
        attempts = self.settings.max_retries + 1

        for attempt in range(attempts):
            try:
                t0 = time.monotonic()
                raw = self._get_client().request(
                    method.upper(),
                    full_url,
                    params=merged_params or None,
                    headers=merged_headers or None,
                    cookies=merged_cookies or None,
                    **send_kwargs,
                )
                elapsed = (time.monotonic() - t0) * 1000

                resp = self._parse_response(raw, elapsed)

                if resp.status_code in self.settings.retry_statuses and attempt < attempts - 1:
                    time.sleep(backoff)
                    backoff *= 2
                    continue

                return resp

            except Exception as exc:
                last_exc = exc
                if attempt < attempts - 1:
                    time.sleep(backoff)
                    backoff *= 2

        raise HTTPError(f"Request failed after {attempts} attempts: {last_exc}") from last_exc

    @staticmethod
    def _parse_response(raw, elapsed_ms: float) -> Response:
        import json as _json

        raw_bytes = raw.content
        body: Any = raw_bytes

        content_type = raw.headers.get("content-type", "")
        if "application/json" in content_type or "application/ld+json" in content_type:
            try:
                body = _json.loads(raw_bytes)
            except Exception:
                body = raw_bytes.decode(errors="replace")
        elif "text/" in content_type:
            body = raw_bytes.decode(errors="replace")

        return Response(
            status_code=raw.status_code,
            headers=dict(raw.headers),
            body=body,
            raw_bytes=raw_bytes,
            url=str(raw.url),
            elapsed_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def get(self, url: str, *, params: dict | None = None, headers: dict | None = None, **kw) -> Response:
        return self.request("GET", url, params=params, headers=headers, **kw)

    def post(self, url: str, *, body: Any = None, body_format: BodyFormat = "json", **kw) -> Response:
        return self.request("POST", url, body=body, body_format=body_format, **kw)

    def put(self, url: str, *, body: Any = None, body_format: BodyFormat = "json", **kw) -> Response:
        return self.request("PUT", url, body=body, body_format=body_format, **kw)

    def patch(self, url: str, *, body: Any = None, body_format: BodyFormat = "json", **kw) -> Response:
        return self.request("PATCH", url, body=body, body_format=body_format, **kw)

    def delete(self, url: str, *, params: dict | None = None, **kw) -> Response:
        return self.request("DELETE", url, params=params, **kw)

    def head(self, url: str, **kw) -> Response:
        return self.request("HEAD", url, **kw)

    # ------------------------------------------------------------------
    # Pagination helper
    # ------------------------------------------------------------------

    def paginate(
        self,
        url: str,
        *,
        method: str = "GET",
        body: Any = None,
        params: dict | None = None,
        headers: dict | None = None,
        # --- pagination strategy ---
        next_url_path: str | None = "next",          # JSON key that holds the next page URL
        items_path: str | None = None,               # JSON key that holds the items list
        offset_param: str | None = None,             # e.g. "offset" or "page"
        limit_param: str | None = None,              # e.g. "limit" or "per_page"
        limit_value: int = 100,
        max_pages: int = 50,
    ) -> list[Any]:
        """
        Automatically follow paginated responses and return all items.

        Two strategies:
        1. next_url_path — follow a "next" URL in the response JSON
        2. offset_param  — increment an offset/page parameter

        Parameters
        ----------
        next_url_path:
            Dot-notation key in the JSON response that contains the next page
            URL, e.g. "next" or "pagination.next_url".
        items_path:
            Dot-notation key whose value is the list of items, e.g. "data"
            or "results". If None, the response body itself is expected to be
            a list.
        offset_param:
            Query parameter name for the offset / page number.
        limit_param:
            Query parameter name for the page size.
        limit_value:
            Value sent for limit_param on each request.
        max_pages:
            Hard stop to avoid infinite loops.
        """
        all_items: list[Any] = []
        current_url: str | None = url
        current_params = dict(params or {})
        offset = 0

        if offset_param:
            current_params[offset_param] = offset
        if limit_param:
            current_params[limit_param] = limit_value

        for _ in range(max_pages):
            if not current_url:
                break

            resp = self.request(method, current_url, params=current_params, headers=headers, body=body)
            resp.raise_for_status()

            # Extract items
            data = resp.body
            if items_path:
                for key in items_path.split("."):
                    if isinstance(data, dict):
                        data = data.get(key, [])
            items = data if isinstance(data, list) else [data]
            all_items.extend(items)

            if not items:
                break

            # Determine next page
            if next_url_path:
                nxt = resp.body
                for key in next_url_path.split("."):
                    if isinstance(nxt, dict):
                        nxt = nxt.get(key)
                    else:
                        nxt = None
                        break
                current_url = nxt
                current_params = {}    # next URL usually already includes params
            elif offset_param:
                offset += limit_value
                current_params[offset_param] = offset
            else:
                break

        return all_items

    def __repr__(self) -> str:
        return (
            f"HTTPRequestNode(base_url={self.settings.base_url!r}, "
            f"auth={self.auth.type}, "
            f"timeout={self.settings.timeout}s)"
        )


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    # --- Basic GET (public API, no auth) ---
    node = HTTPRequestNode(
        settings=RequestSettings(timeout=10),
    )

    print("=== GET https://httpbin.org/get ===")
    resp = node.get("https://httpbin.org/get", params={"hello": "world"})
    print(f"Status : {resp.status_code}")
    print(f"Elapsed: {resp.elapsed_ms:.1f} ms")
    print(f"Args   : {resp.body.get('args') if isinstance(resp.body, dict) else resp.body}")

    # --- POST JSON ---
    print("\n=== POST https://httpbin.org/post ===")
    resp = node.post(
        "https://httpbin.org/post",
        body={"name": "Alice", "score": 42},
    )
    print(f"Status : {resp.status_code}")
    if isinstance(resp.body, dict):
        print(f"JSON   : {json.dumps(resp.body.get('json'), indent=2)}")

    # --- Bearer auth example (token not validated by httpbin) ---
    print("\n=== Bearer auth ===")
    secure_node = HTTPRequestNode(
        settings=RequestSettings(base_url="https://httpbin.org"),
        auth=AuthSettings(type="bearer", token="my-secret-token"),
    )
    resp = secure_node.get("/bearer")
    print(f"Status : {resp.status_code}  authenticated={resp.body.get('authenticated') if isinstance(resp.body, dict) else '?'}")

    # --- API key in query param ---
    print("\n=== API key (query param) ===")
    api_node = HTTPRequestNode(
        settings=RequestSettings(timeout=5),
        auth=AuthSettings(
            type="api_key",
            token="supersecret",
            api_key_name="api_key",
            api_key_in="query",
        ),
    )
    resp = api_node.get("https://httpbin.org/get")
    if isinstance(resp.body, dict):
        print(f"api_key in args: {resp.body.get('args', {}).get('api_key')}")

    # --- Context manager ---
    print("\n=== Context manager + paginate stub ===")
    with HTTPRequestNode(settings=RequestSettings(timeout=5)) as n:
        print(n)

    node.close()
    print("\nAll requests completed.")
