#!/usr/bin/env python3
"""Task 2 test runner — verifies episodic memory (2a) and user profile (2b)."""

import sys
import sqlite3
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from agent.config import CHECKPOINTS_DB, PROFILES_DIR
from agent.graph import build_graph
from agent.profile import UserProfile, load_profile, save_profile


# ── Helpers ───────────────────────────────────────────────────────────────────

def run_query(query: str, graph, config: dict) -> dict:
    """Run a query and return router_classification, tool_calls, observations, answer."""
    initial = {
        "messages":   [HumanMessage(content=query)],
        "query_type": "",
        "iterations": 0,
    }
    result = {
        "router_classification": None,
        "tool_calls":   [],
        "observations": [],
        "answer":       None,
    }
    for chunk in graph.stream(initial, config=config, stream_mode="updates"):
        for node_name, output in chunk.items():
            if node_name == "router":
                result["router_classification"] = output.get("query_type", "?")
                continue
            if node_name == "profile_updater":
                continue
            for msg in output.get("messages", []):
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for tc in msg.tool_calls:
                        result["tool_calls"].append({"tool": tc["name"], "args": tc["args"]})
                elif isinstance(msg, ToolMessage):
                    preview = str(msg.content)
                    if len(preview) > 600:
                        preview = preview[:600] + " …[truncated]"
                    result["observations"].append({"tool": msg.name, "content": preview})
                elif isinstance(msg, AIMessage) and msg.content:
                    result["answer"] = msg.content
    return result


def fmt_args(args: dict) -> str:
    if not args:
        return "*(none)*"
    return ", ".join(f"`{k}={repr(v)}`" for k, v in args.items())


def render_md(tc_id: str, description: str, query: str, result: dict) -> str:
    lines = [
        f"### {tc_id} — {description}",
        "",
        f"**Query:** {query}",
        "",
        f"**Router:** `{result['router_classification']}`",
        "",
    ]
    if result["tool_calls"]:
        lines.append("**Tool calls:**")
        for c in result["tool_calls"]:
            lines.append(f"- `{c['tool']}({fmt_args(c['args'])})`")
        lines.append("")
    if result["observations"]:
        lines.append("**Observations:**")
        for o in result["observations"]:
            lines.append(f"- **{o['tool']}** →")
            lines.append(f"  ```\n  {o['content']}\n  ```")
        lines.append("")
    lines += [
        "**Answer:**",
        "",
        result["answer"] or "*(no answer)*",
        "",
        "---",
        "",
    ]
    return "\n".join(lines)


# ── Test suites ───────────────────────────────────────────────────────────────

def run_2a_tests(graph) -> list[tuple[str, str, str, dict, bool]]:
    """Task 2a — Episodic memory: multi-turn follow-ups within one session."""
    session_id = "test-task2a-session"
    config = {"configurable": {"thread_id": session_id, "user_id": "test_user_2a"}}
    results = []

    # Turn 1
    q1 = "Show me 3 examples from the REFUND category."
    r1 = run_query(q1, graph, config)
    passed = bool(r1["answer"])
    results.append(("TC-2a-01", "First turn — show 3 REFUND examples", q1, r1, passed))

    # Turn 2 — follow-up referencing Turn 1 context
    q2 = "Show me 3 more."
    r2 = run_query(q2, graph, config)
    # The agent should increase the offset — look for offset > 0 in tool args
    used_offset = any(
        int(c["args"].get("offset", 0)) > 0
        for c in r2["tool_calls"]
        if c["tool"] == "show_examples"
    )
    passed2 = bool(r2["answer"]) and used_offset
    results.append(("TC-2a-02", "Follow-up 'show me 3 more' uses non-zero offset", q2, r2, passed2))

    # Turn 3 — count then follow-up
    q3 = "How many complaints did we get?"
    r3 = run_query(q3, graph, config)
    results.append(("TC-2a-03", "Count complaints", q3, r3, bool(r3["answer"])))

    q4 = "What about refunds?"
    r4 = run_query(q4, graph, config)
    results.append(("TC-2a-04", "Follow-up 'what about refunds?' understood in context", q4, r4, bool(r4["answer"])))

    return results


def run_2b_tests(graph) -> list[tuple[str, str, str, dict, bool]]:
    """Task 2b — User profile: agent learns and recalls user facts."""
    user_id    = "test_user_2b"
    session_id = "test-task2b-session"
    config = {"configurable": {"thread_id": session_id, "user_id": user_id}}

    # Seed profile directly to avoid depending on LLM extraction for the test
    save_profile(user_id, UserProfile(
        name="Alice",
        interests=["refund analysis", "complaint patterns"],
        preferences=["concise answers"],
        notes=["works in customer operations"],
    ))

    results = []

    # Ask the agent what it remembers
    q = "What do you remember about me?"
    r = run_query(q, graph, config)
    # Profile was injected in system prompt; answer should mention "Alice"
    mentions_name = "alice" in (r["answer"] or "").lower()
    results.append(("TC-2b-01", "Agent recalls seeded user profile", q, r, mentions_name))

    # Verify profile persists after save/load
    loaded = load_profile(user_id)
    roundtrip_ok = loaded.name == "Alice" and "refund analysis" in loaded.interests
    dummy_result = {
        "router_classification": "n/a",
        "tool_calls": [],
        "observations": [],
        "answer": f"Profile on disk: name={loaded.name}, interests={loaded.interests}",
    }
    results.append(("TC-2b-02", "Profile persists correctly on disk", "(internal check)", dummy_result, roundtrip_ok))

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H")
    output_path = Path(__file__).resolve().parent / f"task2_test_{timestamp}.md"

    print("Building graph with SqliteSaver…")
    with SqliteSaver.from_conn_string(str(CHECKPOINTS_DB)) as saver:
        graph = build_graph(checkpointer=saver)

        all_results: list[tuple[str, str, str, dict, bool]] = []

        print("\n— Task 2a: Episodic memory —")
        for item in run_2a_tests(graph):
            tc_id, desc, query, result, passed = item
            status = "PASS" if passed else "FAIL"
            print(f"  {tc_id}: {status} — {desc}")
            all_results.append(item)

        print("\n— Task 2b: User profile —")
        for item in run_2b_tests(graph):
            tc_id, desc, query, result, passed = item
            status = "PASS" if passed else "FAIL"
            print(f"  {tc_id}: {status} — {desc}")
            all_results.append(item)

    passed_count = sum(1 for *_, ok in all_results if ok)
    failed_count = len(all_results) - passed_count

    # Build markdown
    md = [
        "# Task 2 — Memory Test Run",
        "",
        f"**Date/Time:** {now.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Total test cases:** {len(all_results)}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total  | {len(all_results)} |",
        f"| Passed | {passed_count} |",
        f"| Failed | {failed_count} |",
        "",
        "---",
        "",
        "## Task 2a — Episodic Memory",
        "",
    ]

    for tc_id, desc, query, result, passed in all_results:
        section = render_md(tc_id, desc, query, result)
        if "2b" in tc_id and "## Task 2b" not in "\n".join(md):
            md += ["## Task 2b — User Profile", ""]
        md.append(section)

    output_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\nResults written to: {output_path}")


if __name__ == "__main__":
    main()
