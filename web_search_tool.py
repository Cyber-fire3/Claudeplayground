"""
Web Search Tool — plug into any AIAgentNode

Provides a web_search Tool that can be attached to any agent so it can
search the internet at runtime.

Supported providers (pick one, set the matching API key):
  - SerpAPI    https://serpapi.com          (SERPAPI_KEY)
  - Brave      https://api.search.brave.com (BRAVE_KEY)
  - Tavily     https://tavily.com           (TAVILY_KEY)  ← AI-focused, generous free tier

Usage
-----
from web_search_tool import make_web_search_tool
from main import AIAgentNode, AISettings

agent = AIAgentNode(
    provider="gemini",
    api_key="AIza...",
    settings=AISettings(model="gemini-2.0-flash"),
)
agent.add_tool(make_web_search_tool(provider="tavily", api_key="tvly-..."))
print(agent.run("What happened in the news today?"))
"""

from __future__ import annotations

from https_request_node import AuthSettings, HTTPRequestNode, RequestSettings
from n8n_ai_agent_node import Tool


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

def _serpapi_search(http: HTTPRequestNode, query: str, num_results: int) -> str:
    resp = http.get(
        "https://serpapi.com/search",
        params={"q": query, "num": num_results, "hl": "en"},
    )
    resp.raise_for_status()
    results = resp.body.get("organic_results", [])
    if not results:
        return "No results found."
    return "\n\n".join(
        f"{r.get('title', '')}\n{r.get('link', '')}\n{r.get('snippet', '')}"
        for r in results[:num_results]
    )


def _brave_search(http: HTTPRequestNode, query: str, num_results: int) -> str:
    resp = http.get(
        "https://api.search.brave.com/res/v1/web/search",
        params={"q": query, "count": num_results},
    )
    resp.raise_for_status()
    results = resp.body.get("web", {}).get("results", [])
    if not results:
        return "No results found."
    return "\n\n".join(
        f"{r.get('title', '')}\n{r.get('url', '')}\n{r.get('description', '')}"
        for r in results[:num_results]
    )


def _tavily_search(http: HTTPRequestNode, query: str, num_results: int) -> str:
    resp = http.post(
        "https://api.tavily.com/search",
        body={
            "query": query,
            "max_results": num_results,
            "include_answer": True,
        },
    )
    resp.raise_for_status()
    body = resp.body

    # Tavily returns a direct answer plus source snippets
    parts = []
    if body.get("answer"):
        parts.append(f"Summary: {body['answer']}")
    for r in body.get("results", [])[:num_results]:
        parts.append(f"{r.get('title', '')}\n{r.get('url', '')}\n{r.get('content', '')}")
    return "\n\n".join(parts) if parts else "No results found."


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_PROVIDERS = {
    "serpapi": {
        "fn": _serpapi_search,
        "auth_type": "api_key",
        "api_key_name": "api_key",
        "api_key_in": "query",
    },
    "brave": {
        "fn": _brave_search,
        "auth_type": "api_key",
        "api_key_name": "X-Subscription-Token",
        "api_key_in": "header",
    },
    "tavily": {
        "fn": _tavily_search,
        "auth_type": "bearer",
        "api_key_name": "",
        "api_key_in": "header",
    },
}


def make_web_search_tool(
    api_key: str,
    provider: str = "tavily",
    num_results: int = 5,
    timeout: float = 15.0,
) -> Tool:
    """
    Build and return a web_search Tool for use with AIAgentNode.

    Parameters
    ----------
    api_key:
        API key for the chosen search provider.
    provider:
        "tavily" | "brave" | "serpapi"
    num_results:
        Maximum number of results to return to the model.
    timeout:
        HTTP timeout in seconds.
    """
    cfg = _PROVIDERS.get(provider.lower())
    if cfg is None:
        raise ValueError(f"Unknown search provider '{provider}'. Choose from: {list(_PROVIDERS)}")

    if cfg["auth_type"] == "bearer":
        auth = AuthSettings(type="bearer", token=api_key)
    else:
        auth = AuthSettings(
            type="api_key",
            token=api_key,
            api_key_name=cfg["api_key_name"],
            api_key_in=cfg["api_key_in"],
        )

    http = HTTPRequestNode(
        settings=RequestSettings(timeout=timeout),
        auth=auth,
    )

    search_fn = cfg["fn"]

    def _search(query: str) -> str:
        return search_fn(http, query, num_results)

    return Tool(
        name="web_search",
        description=(
            "Search the internet for up-to-date information. "
            "Use this whenever you need current news, facts, prices, or "
            "anything that may have changed after your training cutoff."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A concise search query, e.g. 'latest Python 4 release date'",
                }
            },
            "required": ["query"],
        },
        fn=_search,
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from main import AIAgentNode, AISettings, WindowBufferMemory

    agent = AIAgentNode(
        provider="gemini",
        api_key="AIza-YOUR-KEY-HERE",
        settings=AISettings(
            model="gemini-2.0-flash",
            temperature=0.3,
            system_prompt=(
                "You are a helpful research assistant. "
                "Always search the web when you need current information."
            ),
        ),
        memory=WindowBufferMemory(window_size=6),
    )

    agent.add_tool(make_web_search_tool(provider="tavily", api_key="tvly-YOUR-KEY-HERE"))

    print(agent)
    print("\nReplace the API keys above and call:")
    print('  agent.run("What is the latest version of Python?")')
