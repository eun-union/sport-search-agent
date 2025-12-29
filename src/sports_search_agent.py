"""
Sports Search Agent - Flyte v2 Demo

A multi-step AI agent that:
1. Takes a sports query from the user
2. Searches the web using Tavily API
3. Synthesizes results using LiteLLM (supports multiple LLM providers)
4. Returns a formatted response with sources

Demonstrates:
- Multi-step orchestration
- Caching for cost reduction
- Observability of each step
- External API integration
- Provider-agnostic LLM calls via LiteLLM
"""

import asyncio
from dataclasses import dataclass
from typing import List
import flyte

# =============================================================================
# Task Environments
# =============================================================================

# Tavily search environment - lightweight, CPU-only
search_env = flyte.TaskEnvironment(
    name="sports-search-tavily",
    description="Web search using Tavily API for sports queries",
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    image=flyte.Image.from_debian_base().with_pip_packages(
        "tavily-python>=0.3.0",
        "unionai-reuse>=0.1.9",
    ),
    secrets=[
        flyte.Secret(key="EWH_TAVILY_API_KEY", as_env_var="TAVILY_API_KEY"),
    ],
    reusable=flyte.ReusePolicy(
        replicas=(1, 2),
        idle_ttl=120,
        concurrency=5,
        scaledown_ttl=60,
    ),
)

# LiteLLM environment - for response synthesis (supports multiple LLM providers)
llm_env = flyte.TaskEnvironment(
    name="sports-search-llm-anthropic",
    description="LLM synthesis using LiteLLM (Gemini, OpenAI, Claude, etc.)",
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    image=flyte.Image.from_debian_base().with_pip_packages(
        "litellm>=1.0.0",
    ),
    secrets=[
        flyte.Secret(key="NEW_ANTHROPIC_API_KEY", as_env_var="ANTHROPIC_API_KEY")
    ],
    # Temporarily disabled reusable to force fresh pods with new secret
)

# Driver environment - orchestrates the pipeline
driver_env = flyte.TaskEnvironment(
    name="sports-search-driver",
    description="Orchestrates the sports search pipeline",
    resources=flyte.Resources(cpu=1, memory="256Mi"),
    image=flyte.Image.from_debian_base().with_pip_packages(
        "unionai-reuse>=0.1.9",
    ),
    depends_on=[search_env, llm_env],
    reusable=flyte.ReusePolicy(
        replicas=(1, 2),
        idle_ttl=120,
        concurrency=10,
        scaledown_ttl=60,
    ),
)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SearchResult:
    """A single search result from Tavily."""
    title: str
    url: str
    content: str
    score: float


@dataclass
class AgentResponse:
    """The final response from the sports search agent."""
    query: str
    answer: str
    sources: List[str]
    search_results_count: int


# =============================================================================
# Traced LLM Helpers (for observability in Union UI)
# =============================================================================

@flyte.trace
async def llm_normalize(query: str) -> str:
    """Traced LLM call for query normalization."""
    from litellm import acompletion

    response = await acompletion(
        model="claude-3-haiku-20240307",
        messages=[{
            "role": "user",
            "content": f"""Rewrite this sports question into a standard, canonical format.
Rules:
- Use proper capitalization (e.g., "NBA", "NFL", "FIFA")
- Use full names for championships (e.g., "NBA Championship" not "nba title")
- Include the year if mentioned
- Keep it as a clear question
- Only return the normalized question, nothing else

Original question: {query}

Normalized question:"""
        }],
        temperature=0,
        max_tokens=100,
    )
    return response.choices[0].message.content.strip()


@flyte.trace
async def llm_synthesize(system_prompt: str, user_prompt: str) -> str:
    """Traced LLM call for response synthesis."""
    from litellm import acompletion

    response = await acompletion(
        model="claude-3-haiku-20240307",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=500,
    )
    return response.choices[0].message.content


# =============================================================================
# Tasks
# =============================================================================

@llm_env.task(retries=flyte.RetryStrategy(count=2))
async def normalize_query(query: str) -> str:
    """
    Normalize a sports query to a canonical form for better cache hits.

    This enables semantic caching - different phrasings of the same question
    will normalize to the same canonical form and hit the cache.

    Examples:
        "who was 2023 nba champion?" -> "Who won the 2023 NBA Championship?"
        "who won the nba championship in 2023?" -> "Who won the 2023 NBA Championship?"

    Args:
        query: The user's original query

    Returns:
        A normalized, canonical form of the query
    """
    print(f"[Normalize] Original query: {query}")

    # Use traced helper for observability
    normalized = await llm_normalize(query)

    print(f"[Normalize] Normalized query: {normalized}")

    return normalized


@search_env.task(
    cache=flyte.Cache(behavior="auto", version_override="v2"),
    retries=flyte.RetryStrategy(count=3),
)
async def search_sports_web(query: str, max_results: int = 5) -> List[SearchResult]:
    """
    Search the web for sports-related information using Tavily API.

    This task is cached to avoid redundant API calls for the same query.

    Args:
        query: The sports-related search query
        max_results: Maximum number of results to return

    Returns:
        List of search results with title, URL, content, and relevance score
    """
    import os
    from tavily import TavilyClient

    print(f"[Tavily] Searching for: {query}")

    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY not found in environment")

    client = TavilyClient(api_key=api_key)

    # Add "sports" context if not already present
    search_query = query if "sport" in query.lower() else f"{query} sports"

    response = client.search(
        query=search_query,
        max_results=max_results,
        search_depth="advanced",
        include_answer=False,
    )

    results = []
    for result in response.get("results", []):
        results.append(SearchResult(
            title=result.get("title", "No title"),
            url=result.get("url", ""),
            content=result.get("content", ""),
            score=result.get("score", 0.0),
        ))

    print(f"[Tavily] Found {len(results)} results")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r.title[:60]}... (score: {r.score:.2f})")

    return results


@llm_env.task(
    cache=flyte.Cache(behavior="auto", version_override="v7"),
    retries=flyte.RetryStrategy(count=3),
)
async def synthesize_response(
    query: str,
    search_results: List[SearchResult],
) -> str:
    """
    Use LiteLLM to synthesize search results into a coherent answer.

    LiteLLM provides a unified interface for multiple LLM providers:
    - gemini/gemini-pro (Google Gemini)
    - gpt-4o-mini (OpenAI)
    - claude-3-haiku-20240307 (Anthropic)

    This task is cached to avoid redundant LLM calls for identical inputs.

    Args:
        query: The original user query
        search_results: List of search results from Tavily

    Returns:
        A synthesized, well-formatted answer
    """
    print(f"[LiteLLM] Synthesizing response for: {query}")

    # Format search results for the prompt
    context_parts = []
    for i, result in enumerate(search_results, 1):
        context_parts.append(
            f"Source {i}: {result.title}\n"
            f"URL: {result.url}\n"
            f"Content: {result.content}\n"
        )

    context = "\n---\n".join(context_parts)

    system_prompt = """You are a sports expert assistant. Your job is to answer questions
about sports using the provided search results.

Guidelines:
- Be concise but informative (2-3 paragraphs max)
- Include specific facts, stats, or details when available
- If the search results don't contain enough information, say so
- Write in an engaging, conversational tone
- Focus on the most recent and relevant information"""

    user_prompt = f"""Question: {query}

Here are the search results to use:

{context}

Please provide a helpful, accurate answer based on these sources."""

    # Use traced helper for observability in Union UI
    answer = await llm_synthesize(system_prompt, user_prompt)

    print(f"[LiteLLM] Generated response ({len(answer)} chars)")

    return answer


@driver_env.task(report=True)
async def sports_search_agent(
    query: str,
    max_results: int = 5,
) -> AgentResponse:
    """
    Main sports search agent workflow.

    Orchestrates the multi-step pipeline:
    1. Normalize the query for semantic caching
    2. Search the web using Tavily
    3. Synthesize results using LiteLLM (Claude)
    4. Format and return the response

    Args:
        query: The user's sports-related question
        max_results: Maximum number of search results to use

    Returns:
        AgentResponse with the answer and source URLs
    """
    print(f"[Agent] Starting sports search for: {query}")

    # Step 1: Normalize query for semantic caching
    with flyte.group("query-normalization"):
        normalized_query = await normalize_query(query)
        print(f"[Agent] Normalized: '{query}' -> '{normalized_query}'")

    # Step 2: Search the web (uses normalized query for better cache hits)
    with flyte.group("web-search"):
        search_results = await search_sports_web(normalized_query, max_results)

    if not search_results:
        return AgentResponse(
            query=query,
            answer="I couldn't find any relevant information for your query. Please try rephrasing your question.",
            sources=[],
            search_results_count=0,
        )

    # Step 3: Synthesize response using LLM (uses normalized query for cache)
    with flyte.group("llm-synthesis"):
        answer = await synthesize_response(normalized_query, search_results)

    # Step 4: Format response
    sources = [r.url for r in search_results if r.url]

    print(f"[Agent] Completed with {len(sources)} sources")

    # Step 5: Generate HTML report for Union UI
    sources_html = "\n".join([
        f'<li><a href="{url}" target="_blank">{url}</a></li>'
        for url in sources
    ])

    html_report = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px;">
        <h1 style="color: #1a1a1a; border-bottom: 2px solid #e0e0e0; padding-bottom: 10px;">
            🏆 Sports Search Results
        </h1>

        <div style="background: #f5f5f5; padding: 15px; border-radius: 8px; margin: 20px 0;">
            <strong>Original Query:</strong> {query}<br>
            <strong>Normalized Query:</strong> {normalized_query}
        </div>

        <h2 style="color: #333;">Answer</h2>
        <div style="background: #e3f2fd; padding: 20px; border-radius: 8px; border-left: 4px solid #1976d2;">
            {answer.replace(chr(10), '<br>')}
        </div>

        <h2 style="color: #333; margin-top: 30px;">Sources ({len(sources)})</h2>
        <ul style="line-height: 1.8;">
            {sources_html}
        </ul>

        <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #e0e0e0; color: #666; font-size: 12px;">
            Powered by Flyte v2 | Tavily Search | Anthropic Claude
        </div>
    </div>
    """

    await flyte.report.replace.aio(html_report)

    return AgentResponse(
        query=query,
        answer=answer,
        sources=sources,
        search_results_count=len(search_results),
    )


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    flyte.init_from_config()

    # Example query for testing
    test_query = "Who won the latest NBA championship and who was the MVP?"

    run = flyte.run(sports_search_agent, query=test_query)

    print(f"\nWorkflow started!")
    print(f"Run name: {run.name}")
    print(f"Run URL: {run.url}")
    print(f"\nQuery: {test_query}")
    print(f"\nMonitor progress:")
    print(f"  flyte get logs {run.name}")
    print(f"\nGet result in Python:")
    print(f'''
import asyncio
import flyte

flyte.init_from_config()
run = flyte.remote.Run.get(name="{run.name}")

async def get_result():
    result = await run.result()
    print(f"Answer: {{result.answer}}")
    print(f"\\nSources:")
    for url in result.sources:
        print(f"  - {{url}}")

asyncio.run(get_result())
''')
