#!/usr/bin/env python3
"""Bitext Customer Service Data Analyst Agent — interactive CLI."""

import argparse
import sys

from dotenv import load_dotenv

load_dotenv()  # must run before any agent imports that read env vars

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402

from agent.graph import build_graph  # noqa: E402

console = Console()


# ── Pretty printers ───────────────────────────────────────────────────────────

def _print_tool_call(name: str, args: dict) -> None:
    console.print(Panel(
        f"[bold]Tool:[/bold] {name}\n[bold]Args:[/bold] {args}",
        title="[yellow]⚙ Tool Call[/yellow]",
        border_style="yellow",
    ))


def _print_observation(name: str, content: str) -> None:
    preview = content if len(content) <= 600 else content[:600] + " …"
    console.print(Panel(
        f"[bold]Tool:[/bold] {name}\n[bold]Result:[/bold] {preview}",
        title="[green]👁 Observation[/green]",
        border_style="green",
    ))


def _print_answer(content: str) -> None:
    console.print(Panel(content, title="[bold blue]Agent[/bold blue]", border_style="blue"))


# ── Core loop ─────────────────────────────────────────────────────────────────

def run_query(query: str, graph, config: dict) -> None:
    """Stream one user query through the graph, printing every reasoning step."""
    initial = {
        "messages":   [HumanMessage(content=query)],
        "query_type": "",
        "iterations": 0,
    }

    for chunk in graph.stream(initial, config=config, stream_mode="updates"):
        for node_name, output in chunk.items():
            if node_name == "router":
                qt = output.get("query_type", "?")
                console.print(f"  [dim][router] → {qt}[/dim]")
                continue

            for msg in output.get("messages", []):
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for tc in msg.tool_calls:
                        _print_tool_call(tc["name"], tc["args"])
                elif isinstance(msg, ToolMessage):
                    _print_observation(msg.name, str(msg.content))
                elif isinstance(msg, AIMessage) and msg.content:
                    _print_answer(msg.content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bitext Customer Service Data Analyst Agent")
    parser.add_argument(
        "--session", default="default",
        help="Session ID — persists the conversation across restarts. Same ID = same history.",
    )
    args = parser.parse_args()

    graph  = build_graph()
    config = {"configurable": {"thread_id": args.session}}

    console.print(Panel(
        "[bold]Bitext Customer Service Data Analyst Agent[/bold]\n"
        f"Session: [cyan]{args.session}[/cyan]   "
        "Type [bold red]exit[/bold red] to quit.",
        border_style="bright_blue",
    ))

    while True:
        try:
            query = console.input("\n[bold]You:[/bold] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            sys.exit(0)

        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            console.print("[dim]Goodbye.[/dim]")
            break

        run_query(query, graph, config)


if __name__ == "__main__":
    main()
