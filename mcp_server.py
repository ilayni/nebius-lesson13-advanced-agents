#!/usr/bin/env python3
"""FastMCP server — exposes Bitext dataset tools over the Model Context Protocol.

Run:
    python mcp_server.py                  # stdio transport (default, use with MCP clients)
    python mcp_server.py --http           # HTTP/SSE transport on http://localhost:8000
"""

import argparse
import sys
from pathlib import Path

# Ensure repo root is on the path when invoked directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv()

from fastmcp import FastMCP

from agent.dataset import load
from agent.tools import _filter


mcp = FastMCP(
    "bitext-analyst",
    instructions=(
        "Tools for querying the Bitext Customer Service dataset — "
        "a collection of customer support queries and agent responses "
        "grouped by category and intent."
    ),
)


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
def list_categories() -> list[str]:
    """Return every distinct top-level category in the Bitext dataset
    (e.g. ACCOUNT, CANCEL, FEEDBACK, ORDER, REFUND, SHIPPING).
    Call this first when the user asks what topics or groupings exist."""
    return sorted(load()["category"].unique().tolist())


@mcp.tool()
def list_intents(category: str | None = None) -> dict[str, list[str]]:
    """Return distinct intents grouped by category.
    Narrow to a single category with the *category* argument, or omit it to see all.
    Use this to discover valid intent names before counting or filtering."""
    df = load() if not category else load()[
        load()["category"].str.upper() == category.strip().upper()
    ]
    result: dict[str, list[str]] = {}
    for cat, grp in df.groupby("category"):
        result[str(cat)] = sorted(grp["intent"].unique().tolist())
    return result


@mcp.tool()
def count_rows(category: str | None = None, intent: str | None = None) -> int:
    """Count dataset rows matching the given category and/or intent filters.
    Pass no arguments to get the total dataset size."""
    return len(_filter(category, intent))


@mcp.tool()
def show_examples(
    n: int = 5,
    category: str | None = None,
    intent: str | None = None,
    offset: int = 0,
) -> list[dict]:
    """Return example rows from the dataset.
    Each row contains: category, intent, instruction (customer query), response (agent reply).
    Use *offset* to page through results (e.g. first call offset=0, next call offset=n)."""
    df = _filter(category, intent)
    if df.empty:
        return []
    batch = df.iloc[offset: offset + n]
    return batch[["category", "intent", "instruction", "response"]].to_dict(orient="records")


@mcp.tool()
def intent_distribution(category: str | None = None) -> dict[str, int]:
    """Return intent → row count mapping, sorted by count descending.
    Pass *category* to scope to one category, or omit for the full dataset."""
    df = _filter(category, None)
    if df.empty:
        return {}
    return {str(k): int(v) for k, v in df["intent"].value_counts().items()}


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Bitext Analyst MCP Server")
    parser.add_argument("--http", action="store_true", help="Use HTTP/SSE transport instead of stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.http:
        mcp.run(transport="streamable-http", host=args.host, port=args.port)
    else:
        mcp.run()  # stdio — default MCP transport


if __name__ == "__main__":
    main()
