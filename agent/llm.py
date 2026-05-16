import os
from functools import lru_cache

from langchain_openai import ChatOpenAI

from agent.config import GENERATION_MODEL, NEBIUS_BASE_URL, ROUTER_MODEL


def _build(model: str) -> ChatOpenAI:
    key = os.environ.get("NEBIUS_API_KEY")
    if not key:
        raise EnvironmentError(
            "NEBIUS_API_KEY is not set. Copy .env.example → .env and fill in your key."
        )
    return ChatOpenAI(model=model, base_url=NEBIUS_BASE_URL, api_key=key)


@lru_cache(maxsize=1)
def get_generation_llm() -> ChatOpenAI:
    """70B model — agent reasoning, tool selection, and summarization."""
    return _build(GENERATION_MODEL)


@lru_cache(maxsize=1)
def get_router_llm() -> ChatOpenAI:
    """8B model — query routing and user-profile updates (Task 2)."""
    return _build(ROUTER_MODEL)
