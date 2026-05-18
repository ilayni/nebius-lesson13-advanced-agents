"""User profile — persistent, distilled facts about each user (Task 2b)."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from agent.config import PROFILES_DIR


# ── Schema ────────────────────────────────────────────────────────────────────

class UserProfile(BaseModel):
    """Distilled facts about a user — name, interests, preferences, freeform notes."""
    name:        str | None  = None
    interests:   list[str]   = []
    preferences: list[str]   = []
    notes:       list[str]   = []


# ── Storage ───────────────────────────────────────────────────────────────────

def load_profile(user_id: str) -> UserProfile:
    """Load the profile for *user_id*, or return an empty profile on first use."""
    path = PROFILES_DIR / f"{user_id}.json"
    try:
        return UserProfile.model_validate_json(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        return UserProfile()


def save_profile(user_id: str, profile: UserProfile) -> None:
    """Persist the updated profile to disk."""
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    (PROFILES_DIR / f"{user_id}.json").write_text(
        profile.model_dump_json(indent=2), encoding="utf-8"
    )


# ── LLM-driven updater ────────────────────────────────────────────────────────

_UPDATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You maintain a brief user profile for a customer-service data analyst assistant.

Current profile (JSON):
{current_profile}

Latest exchange:
User: {human_message}
Agent: {ai_message}

Update the profile with any new *personal* facts revealed about the user:
their name, job role, recurring interests, stated preferences, or goals.

Rules:
- Do NOT record which dataset queries the user ran — only facts about the person.
- Remove or correct anything contradicted by the new exchange.
- If nothing new was learned, return the profile unchanged.
- Keep each list concise (≤5 items per field). Drop the least relevant item when a list is full.
- Return a valid UserProfile object."""),
])


def update_profile(
    user_id: str,
    human_message: str,
    ai_message: str,
) -> None:
    """Call the 8B router LLM to distill the latest exchange into the user profile and save it."""
    from agent.llm import get_router_llm  # local import avoids circular init

    current = load_profile(user_id)
    try:
        chain = _UPDATE_PROMPT | get_router_llm().with_structured_output(UserProfile)
        updated: UserProfile = chain.invoke({
            "current_profile": current.model_dump_json(indent=2),
            "human_message":   human_message,
            "ai_message":      ai_message,
        })
        save_profile(user_id, updated)
    except Exception:
        pass  # profile update is best-effort; never crash the agent


# ── Formatting helper (for system-prompt injection) ───────────────────────────

def format_profile_block(profile: UserProfile) -> str | None:
    """Return a human-readable [User Profile] block, or None if the profile is empty."""
    parts: list[str] = []
    if profile.name:
        parts.append(f"Name: {profile.name}")
    if profile.interests:
        parts.append(f"Interests: {', '.join(profile.interests)}")
    if profile.preferences:
        parts.append(f"Preferences: {', '.join(profile.preferences)}")
    if profile.notes:
        parts.append(f"Notes: {', '.join(profile.notes)}")
    if not parts:
        return None
    return "[User Profile]\n" + "\n".join(parts)
