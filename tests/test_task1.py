#!/usr/bin/env python3
"""Task 1 test runner — runs a suite of suggested questions and writes results to a dated .md file."""

import sys
import os
import io
from datetime import datetime
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.graph import build_graph

# ── Test questions covering every tool and routing path ────────────────────────

TEST_CASES = [
    {
        "id": "TC-01",
        "description": "List all top-level categories (list_categories tool)",
        "query": "What categories are available in the dataset?",
    },
    {
        "id": "TC-02",
        "description": "List intents for a specific category (list_intents tool)",
        "query": "What intents exist under the REFUND category?",
    },
    {
        "id": "TC-03",
        "description": "Count total rows (count_rows tool, no filters)",
        "query": "How many total rows does the dataset have?",
    },
    {
        "id": "TC-04",
        "description": "Count rows for a specific intent (multi-step: list_intents → count_rows)",
        "query": "How many refund requests are in the dataset?",
    },
    {
        "id": "TC-05",
        "description": "Show examples from a category (show_examples tool)",
        "query": "Show me 3 examples of shipping-related customer queries.",
    },
    {
        "id": "TC-06",
        "description": "Intent distribution for a category (intent_distribution tool)",
        "query": "What is the distribution of intents in the ACCOUNT category?",
    },
    {
        "id": "TC-07",
        "description": "Summarize a category (summarize_subset tool — unstructured query)",
        "query": "Summarize the FEEDBACK category. What are the main themes?",
    },
    {
        "id": "TC-08",
        "description": "Out-of-scope query — router should decline without calling any tool",
        "query": "What is the weather in Tel Aviv today?",
    },
]


# ── Run one query and collect structured results ───────────────────────────────

def run_query(query: str, graph, config: dict) -> dict:
    """Run a single query and return a dict with tool_calls, observations, and answer."""
    initial = {
        "messages":   [HumanMessage(content=query)],
        "query_type": "",
        "iterations": 0,
    }

    result = {
        "router_classification": None,
        "tool_calls": [],       # list of {"tool": name, "args": args}
        "observations": [],     # list of {"tool": name, "content": str}
        "answer": None,
    }

    for chunk in graph.stream(initial, config=config, stream_mode="updates"):
        for node_name, output in chunk.items():
            if node_name == "router":
                result["router_classification"] = output.get("query_type", "?")
                continue
            if output is None:  # profile_updater and similar nodes return {}
                continue

            for msg in output.get("messages", []):
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for tc in msg.tool_calls:
                        result["tool_calls"].append({"tool": tc["name"], "args": tc["args"]})
                elif isinstance(msg, ToolMessage):
                    preview = str(msg.content)
                    if len(preview) > 800:
                        preview = preview[:800] + " …[truncated]"
                    result["observations"].append({"tool": msg.name, "content": preview})
                elif isinstance(msg, AIMessage) and msg.content:
                    result["answer"] = msg.content

    return result


# ── Markdown formatting ────────────────────────────────────────────────────────

def format_args(args: dict) -> str:
    if not args:
        return "*(none)*"
    parts = [f"`{k}={repr(v)}`" for k, v in args.items()]
    return ", ".join(parts)


def render_result_md(tc: dict, result: dict) -> str:
    lines = []
    lines.append(f"### {tc['id']} — {tc['description']}")
    lines.append("")
    lines.append(f"**Query:** {tc['query']}")
    lines.append("")
    lines.append(f"**Router classification:** `{result['router_classification']}`")
    lines.append("")

    if result["tool_calls"]:
        lines.append("**Tool calls:**")
        for call in result["tool_calls"]:
            lines.append(f"- `{call['tool']}({format_args(call['args'])})`")
        lines.append("")

    if result["observations"]:
        lines.append("**Observations:**")
        for obs in result["observations"]:
            lines.append(f"- **{obs['tool']}** →")
            lines.append(f"  ```")
            lines.append(f"  {obs['content']}")
            lines.append(f"  ```")
        lines.append("")

    lines.append("**Answer:**")
    lines.append("")
    lines.append(result["answer"] or "*(no answer produced)*")
    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H")
    output_filename = f"task1_test_{timestamp}.md"
    output_path = Path(__file__).resolve().parent / output_filename

    print(f"Building graph…")
    graph = build_graph()

    md_sections = []
    md_sections.append(f"# Task 1 — Agent Test Run")
    md_sections.append(f"")
    md_sections.append(f"**Date/Time:** {now.strftime('%Y-%m-%d %H:%M:%S')}")
    md_sections.append(f"**Total test cases:** {len(TEST_CASES)}")
    md_sections.append(f"")
    md_sections.append(f"---")
    md_sections.append(f"")

    passed = 0
    failed = 0

    for tc in TEST_CASES:
        print(f"Running {tc['id']}: {tc['description']} …", end=" ", flush=True)
        config = {"configurable": {"thread_id": f"test-{tc['id']}"}}
        try:
            result = run_query(tc["query"], graph, config)
            md_sections.append(render_result_md(tc, result))
            status = "OK" if result["answer"] else "NO_ANSWER"
            print(status)
            if result["answer"]:
                passed += 1
            else:
                failed += 1
        except Exception as exc:
            print(f"ERROR: {exc}")
            failed += 1
            md_sections.append(f"### {tc['id']} — {tc['description']}")
            md_sections.append(f"")
            md_sections.append(f"**Query:** {tc['query']}")
            md_sections.append(f"")
            md_sections.append(f"**ERROR:** `{exc}`")
            md_sections.append(f"")
            md_sections.append("---")
            md_sections.append("")

    # Summary
    summary_lines = [
        f"## Summary",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total  | {len(TEST_CASES)} |",
        f"| Passed | {passed} |",
        f"| Failed | {failed} |",
        f"",
    ]
    md_sections.insert(5, "\n".join(summary_lines))

    output_path.write_text("\n".join(md_sections), encoding="utf-8")
    print(f"\nResults written to: {output_path}")


if __name__ == "__main__":
    main()
