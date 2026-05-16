from __future__ import annotations

import textwrap
from typing import Literal

import pandas as pd
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from agent.config import SUMMARIZE_SAMPLE_SIZE
from agent.dataset import load


# ── Pydantic input schemas ────────────────────────────────────────────────────
# Every tool exposes its schema so the LLM knows exactly what arguments to pass.

class ListIntentsInput(BaseModel):
    category: str | None = Field(
        None,
        description=(
            "Category name to filter by (e.g. 'REFUND', 'SHIPPING'). "
            "Pass null to return intents for every category."
        ),
    )


class CountRowsInput(BaseModel):
    category: str | None = Field(
        None,
        description="Category to count rows for (case-insensitive). Null = no category filter.",
    )
    intent: str | None = Field(
        None,
        description="Intent to count rows for (case-insensitive). Can be combined with category.",
    )


class ShowExamplesInput(BaseModel):
    n: int = Field(5, ge=1, le=20, description="Number of example rows to return.")
    category: str | None = Field(None, description="Category to sample from.")
    intent: str | None = Field(None, description="Intent to sample from.")
    offset: int = Field(
        0,
        ge=0,
        description=(
            "Skip this many matching rows before returning n results. "
            "Use to fetch a different batch for follow-up 'show me more' requests "
            "(e.g. first call offset=0, second call offset=n)."
        ),
    )


class IntentDistributionInput(BaseModel):
    category: str | None = Field(
        None,
        description=(
            "Category to analyse (e.g. 'ACCOUNT'). "
            "Pass null to get a distribution across all categories."
        ),
    )


class SummarizeSubsetInput(BaseModel):
    category: str | None = Field(None, description="Category to summarize.")
    intent: str | None = Field(None, description="Intent to summarize. Can be combined with category.")
    focus: Literal["instructions", "responses", "both"] = Field(
        "both",
        description=(
            "Which text columns to include in the summary: "
            "'instructions' (customer queries only), "
            "'responses' (agent replies only), or "
            "'both' (full interaction)."
        ),
    )


# ── Internal helper ───────────────────────────────────────────────────────────

def _filter(category: str | None, intent: str | None) -> pd.DataFrame:
    """Return the dataset filtered by category and/or intent (case-insensitive)."""
    df = load()
    if category:
        df = df[df["category"].str.upper() == category.strip().upper()]
    if intent:
        df = df[df["intent"].str.lower() == intent.strip().lower()]
    return df.reset_index(drop=True)


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def list_categories() -> list[str]:
    """Return every distinct top-level category in the Bitext dataset
    (e.g. ACCOUNT, CANCEL, FEEDBACK, ORDER, REFUND, SHIPPING).
    Call this first when the user asks what topics or groupings exist."""
    return sorted(load()["category"].unique().tolist())


@tool(args_schema=ListIntentsInput)
def list_intents(category: str | None = None) -> dict[str, list[str]]:
    """Return the distinct intents available, grouped by category.
    Use this to discover valid intent names before filtering or counting.
    Narrow to a single category with the category argument, or omit it to see all."""
    df = load() if not category else load()[
        load()["category"].str.upper() == category.strip().upper()
    ]
    result: dict[str, list[str]] = {}
    for cat, grp in df.groupby("category"):
        result[str(cat)] = sorted(grp["intent"].unique().tolist())
    return result


@tool(args_schema=CountRowsInput)
def count_rows(category: str | None = None, intent: str | None = None) -> int:
    """Count how many dataset rows match the given category and/or intent filters.
    Chain with list_intents to answer 'how many X requests?' questions.
    Pass no arguments to get the total dataset size."""
    return len(_filter(category, intent))


@tool(args_schema=ShowExamplesInput)
def show_examples(
    n: int = 5,
    category: str | None = None,
    intent: str | None = None,
    offset: int = 0,
) -> list[dict]:
    """Return example rows from the dataset, each with fields:
    category, intent, instruction (customer query), response (agent reply).
    Use for 'show me examples of X' questions.
    For follow-up 'show me more', increase offset by n (e.g. first call offset=0, next offset=5)."""
    df = _filter(category, intent)
    if df.empty:
        return []
    batch = df.iloc[offset: offset + n]
    return batch[["category", "intent", "instruction", "response"]].to_dict(orient="records")


@tool(args_schema=IntentDistributionInput)
def intent_distribution(category: str | None = None) -> dict[str, int]:
    """Return intent → row count, sorted by count descending.
    Use for 'what is the distribution of intents in X?' questions."""
    df = _filter(category, None)
    if df.empty:
        return {}
    return {str(k): int(v) for k, v in df["intent"].value_counts().items()}


@tool(args_schema=SummarizeSubsetInput)
def summarize_subset(
    category: str | None = None,
    intent: str | None = None,
    focus: Literal["instructions", "responses", "both"] = "both",
) -> str:
    """Summarize a subset of the dataset in natural language.
    Samples up to 20 rows and asks the LLM to synthesize key patterns and themes.
    Use for open-ended questions like 'summarize the FEEDBACK category'
    or 'how do agents typically handle cancellation requests?'."""
    from agent.llm import get_generation_llm  # local import avoids circular init

    df = _filter(category, intent)
    if df.empty:
        return "No rows match the specified filters."

    sample = df.sample(min(SUMMARIZE_SAMPLE_SIZE, len(df)))
    entries: list[str] = []
    for _, row in sample.iterrows():
        parts = [f"Intent: {row['intent']}"]
        if focus in ("instructions", "both"):
            parts.append(f"Customer: {row['instruction']}")
        if focus in ("responses", "both"):
            parts.append(f"Agent: {row['response']}")
        entries.append("\n".join(parts))

    label = (category or "all categories") + (f" / {intent}" if intent else "")
    prompt = textwrap.dedent(f"""
        Analyse these {len(sample)} customer-service interactions from: {label}.
        Summarize the key patterns, common themes, and notable characteristics in 3–5 concise sentences.
        Be factual — do not invent details not present in the data.

        DATA:
        {"\\n---\\n".join(entries)}
    """).strip()

    return get_generation_llm().invoke([HumanMessage(content=prompt)]).content


# Exported list — imported by graph.py
ALL_TOOLS = [
    list_categories,
    list_intents,
    count_rows,
    show_examples,
    intent_distribution,
    summarize_subset,
]
