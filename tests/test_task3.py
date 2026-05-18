#!/usr/bin/env python3
"""Task 3 test — verifies the FastMCP server exposes and runs tools correctly."""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from fastmcp import Client

SERVER = str(Path(__file__).resolve().parent.parent / "mcp_server.py")


# ── Helpers ───────────────────────────────────────────────────────────────────

async def call(client: Client, tool: str, args: dict):
    """Call a tool and return its .data (already-parsed Python value)."""
    result = await client.call_tool(tool, args)
    if result.is_error:
        raise RuntimeError(f"Tool {tool!r} returned an error: {result.content}")
    return result.data


# ── Test cases ────────────────────────────────────────────────────────────────

async def run_tests() -> list[tuple[str, str, object, bool]]:
    results = []

    async with Client(SERVER) as client:

        # TC-01 — server exposes at least the required tools
        tools = await client.list_tools()
        tool_names = {t.name for t in tools}
        required = {"list_categories", "list_intents", "count_rows", "show_examples", "intent_distribution"}
        passed = required.issubset(tool_names)
        results.append(("TC-3-01", "Server exposes ≥5 expected tools", sorted(tool_names), passed))

        # TC-02 — list_categories returns known categories
        cats = await call(client, "list_categories", {})
        passed = isinstance(cats, list) and "REFUND" in cats and "SHIPPING" in cats
        results.append(("TC-3-02", "list_categories returns known categories", cats, passed))

        # TC-03 — count_rows total > 0
        total = await call(client, "count_rows", {})
        passed = isinstance(total, int) and total > 0
        results.append(("TC-3-03", "count_rows() total > 0", total, passed))

        # TC-04 — count_rows with filter returns fewer rows than total
        refund_count = await call(client, "count_rows", {"category": "REFUND"})
        passed = isinstance(refund_count, int) and 0 < refund_count < total
        results.append(("TC-3-04", "count_rows(REFUND) < total", refund_count, passed))

        # TC-05 — show_examples returns dicts with expected fields
        examples = await call(client, "show_examples", {"n": 3, "category": "SHIPPING"})
        passed = (
            isinstance(examples, list)
            and len(examples) == 3
            and all("instruction" in r and "response" in r for r in examples)
        )
        results.append(("TC-3-05", "show_examples returns 3 rows with instruction/response", len(examples), passed))

        # TC-06 — intent_distribution returns known intents for ACCOUNT
        dist = await call(client, "intent_distribution", {"category": "ACCOUNT"})
        passed = isinstance(dist, dict) and any("account" in k.lower() for k in dist)
        results.append(("TC-3-06", "intent_distribution(ACCOUNT) returns known intents", list(dist.keys())[:5], passed))

        # TC-07 — list_intents scoped to REFUND includes refund intents
        intents = await call(client, "list_intents", {"category": "REFUND"})
        refund_intents = intents.get("REFUND", [])
        passed = isinstance(intents, dict) and any("refund" in i.lower() for i in refund_intents)
        results.append(("TC-3-07", "list_intents(REFUND) includes refund intents", refund_intents, passed))

    return results


# ── Markdown output ───────────────────────────────────────────────────────────

def render_md(results: list, now: datetime) -> str:
    passed = sum(1 for *_, ok in results if ok)
    failed = len(results) - passed
    lines = [
        "# Task 3 — MCP Server Test Run",
        "",
        f"**Date/Time:** {now.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Total:** {len(results)} | **Passed:** {passed} | **Failed:** {failed}",
        "",
        "---",
        "",
    ]
    for tc_id, desc, result, ok in results:
        lines += [
            f"### {tc_id} — {desc}",
            "",
            f"**Status:** {'PASS' if ok else 'FAIL'}",
            "",
            "**Result:**",
            "```",
            str(result)[:500],
            "```",
            "",
            "---",
            "",
        ]
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    now = datetime.now()
    output_path = Path(__file__).resolve().parent / f"task3_test_{now.strftime('%Y-%m-%d_%H')}.md"

    print("Starting MCP server and running tests…")
    results = asyncio.run(run_tests())

    for tc_id, desc, _, ok in results:
        print(f"  {tc_id}: {'PASS' if ok else 'FAIL'} — {desc}")

    output_path.write_text(render_md(results, now), encoding="utf-8")
    print(f"\nResults written to: {output_path}")


if __name__ == "__main__":
    main()
