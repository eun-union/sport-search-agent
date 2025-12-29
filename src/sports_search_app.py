"""
Sports Search Agent - Flyte App (Deployed Web Service)

A hosted API endpoint powered by Flyte v2 that anyone can access via URL.
No local setup required for end users.
"""

import flyte
from flyte.app import AppEnvironment, Scaling
from flyte.app.extras import FastAPIAppEnvironment
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import our existing workflow
from src.sports_search_agent import sports_search_agent, AgentResponse

# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="Sports Search Agent",
    description="AI-powered sports search using Tavily + Claude",
    version="1.0.0",
)


class SearchRequest(BaseModel):
    query: str
    max_results: int = 5


class SearchResponse(BaseModel):
    query: str
    answer: str
    sources: list[str]
    search_results_count: int
    run_url: str


@app.get("/")
async def root():
    return {
        "service": "Sports Search Agent",
        "docs": "/docs",
        "usage": "POST /search with {'query': 'your sports question'}",
    }


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Search for sports information and get an AI-synthesized answer.
    """
    try:
        # Run the Flyte workflow
        run = flyte.run(
            sports_search_agent,
            query=request.query,
            max_results=request.max_results,
        )

        # Wait for result
        result: AgentResponse = await run.result()

        return SearchResponse(
            query=result.query,
            answer=result.answer,
            sources=result.sources,
            search_results_count=result.search_results_count,
            run_url=run.url,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "healthy"}


# =============================================================================
# Flyte App Environment
# =============================================================================

app_env = FastAPIAppEnvironment(
    name="sports-search-app",
    app=app,
    image=flyte.Image.from_debian_base().with_pip_packages(
        "fastapi",
        "uvicorn",
        "flyte",
        "tavily-python>=0.3.0",
        "litellm>=1.0.0",
    ),
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    secrets=[
        flyte.Secret(key="EWH_TAVILY_API_KEY", as_env_var="TAVILY_API_KEY"),
        flyte.Secret(key="NEW_ANTHROPIC_API_KEY", as_env_var="ANTHROPIC_API_KEY"),
    ],
    scaling=Scaling(min_replicas=1, max_replicas=3),
    requires_auth=False,  # Public access (set True for auth-required)
)


# =============================================================================
# Deploy
# =============================================================================

if __name__ == "__main__":
    flyte.init_from_config()

    print("Deploying Sports Search App...")
    print("This may take a few minutes on first deploy (building image).\n")

    # Deploy the app
    deployed = flyte.deploy(app_env)

    print(f"\n✅ App deployed successfully!")
    print(f"📍 Endpoint: {deployed.endpoint}")
    print(f"\nUsage:")
    print(f"  curl -X POST {deployed.endpoint}/search \\")
    print(f"    -H 'Content-Type: application/json' \\")
    print(f"    -d '{{\"query\": \"Who won the 2024 NBA Finals?\"}}'")
    print(f"\nAPI Docs: {deployed.endpoint}/docs")
