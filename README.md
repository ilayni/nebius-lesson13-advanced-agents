# Bitext Customer Service Data Analyst Agent

Assignment 3 · Nebius Academy — Advanced Agents  
**Students:** [First Last] · [First Last]

---

## What it does

A LangGraph ReAct agent that answers questions about the [Bitext Customer Service dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset).

- **Structured** — counts, lists, examples, distributions ("How many refund requests?")
- **Unstructured** — summaries and synthesis ("Summarize the FEEDBACK category")
- **Out-of-scope** — politely declined

---

## Setup

**Prerequisites:** Python 3.11+, [uv](https://docs.astral.sh/uv/), Nebius Token Factory API key.

```bash
git clone <repo-url> && cd <repo-name>
uv venv .venv && source .venv/bin/activate
uv pip install -e .
cp .env.example .env   # then fill in NEBIUS_API_KEY
```

---

## Running

### `run.sh` — launch script

```bash
# Agent CLI only
python main.py

# Agent CLI + MCP server (HTTP on http://127.0.0.1:8000)
./run.sh --mcp

# Agent CLI + MCP server + Streamlit UI (http://localhost:8501)
./run.sh --ui

# Pass extra arguments to the agent (session/user persistence)
./run.sh --mcp --session alice_refunds --user alice
./run.sh --ui  --session alice_refunds --user alice
```

`Ctrl+C` stops all processes cleanly.

### CLI only

```bash
python main.py
```

Every reasoning step (tool calls and observations) is printed before the final answer.

**Persistent sessions + user profile** — the same session ID restores the full conversation on restart; the same user ID restores your profile:

```bash
python main.py --session alice_refunds --user alice
```

---

## Architecture

```
User → CLI → LangGraph StateGraph
              │
              ├─ router_node (8B)  ──  out_of_scope → decline → END
              │
              └─ agent_node (70B) ◄──► ToolNode (6 tools)
                      │
                      └── iterations ≥ 12 → fallback → END
```

### Models

| Role | Model |
|------|-------|
| Router | `meta-llama/Meta-Llama-3.1-8B-Instruct` — fast 3-way classifier, cheap (~$0.03/1M tokens) |
| Agent + Summarization | `meta-llama/Llama-3.3-70B-Instruct` — reliable tool-calling, predictable JSON output |

Only Nebius Token Factory models are used.

### Tools

| Tool | Purpose |
|------|---------|
| `list_categories` | All distinct categories |
| `list_intents(category)` | Intents per category — used before filtering |
| `count_rows(category, intent)` | Answers "how many X?" |
| `show_examples(n, category, intent, offset)` | N example rows; `offset` enables "show me more" |
| `intent_distribution(category)` | Intent → count map |
| `summarize_subset(category, intent, focus)` | LLM-generated summary of a sampled subset |

Each tool has a Pydantic input schema and a typed return value.

---

## MCP Server (Task 3)

Five dataset tools are exposed over the [Model Context Protocol](https://modelcontextprotocol.io) via [FastMCP](https://github.com/PrefectHQ/fastmcp).

**Start the server (stdio transport — default for MCP clients):**

```bash
python mcp_server.py
```

**Start with HTTP/SSE transport (for remote clients):**

```bash
python mcp_server.py --http          # listens on http://127.0.0.1:8000
python mcp_server.py --http --port 9000
```

### Exposed tools

| Tool | Description |
|------|-------------|
| `list_categories` | All distinct top-level categories |
| `list_intents(category)` | Intents per category |
| `count_rows(category, intent)` | Row count with optional filters |
| `show_examples(n, category, intent, offset)` | N example rows; offset for paging |
| `intent_distribution(category)` | Intent → count map |

### Connecting a client

```python
import asyncio
from fastmcp import Client

SERVER = "mcp_server.py"   # path to the server script (stdio transport)

async def main():
    async with Client(SERVER) as client:
        # List available tools
        tools = await client.list_tools()
        print("Tools:", [t.name for t in tools])

        # Call a tool — result.data is already a parsed Python value
        cats = await client.call_tool("list_categories", {})
        print("Categories:", cats.data)

        examples = await client.call_tool(
            "show_examples", {"n": 3, "category": "REFUND"}
        )
        for row in examples.data:
            print(row["intent"], "—", row["instruction"][:60])

asyncio.run(main())
```

---

## Streamlit UI (Bonus A)

```bash
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`. The sidebar lets you set a **Session ID** and **User ID** — both persist across browser refreshes and are interchangeable with the CLI sessions. Reasoning steps (tool calls and observations) appear as collapsible blocks above each answer.

---

## Project structure

```
agent/          config, llm, dataset, router, tools, graph
main.py         CLI entrypoint
mcp_server.py   MCP server (Task 3)
streamlit_app.py  Streamlit UI (Bonus A)
run.sh          Launch script (--mcp / --ui flags)
pyproject.toml  dependencies
.env.example    environment variable template
```
