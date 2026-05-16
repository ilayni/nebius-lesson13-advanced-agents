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

## Running the CLI

```bash
python main.py
```

Every reasoning step (tool calls and observations) is printed before the final answer.

**Persistent sessions** — the same session ID restores the full conversation on restart:

```bash
python main.py --session alice_refunds
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

_Coming soon — see `mcp_server.py`._

---

## Project structure

```
agent/          config, llm, dataset, router, tools, graph
main.py         CLI entrypoint
mcp_server.py   MCP server (Task 3)
pyproject.toml  dependencies
.env.example    environment variable template
```
