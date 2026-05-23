# Learning Notes — Assignment 3: From AI Model to AI Agent

---

## Q1: How does the router's 3-way classification affect agent flow?

**Short answer:** `out_of_scope` short-circuits to a hardcoded decline. `structured` and `unstructured` both hit the same agent node — the distinction only matters for tool selection, not graph routing.

**Detail:**

The router classifies every query into one of three types using an 8B LLM with structured output:
- `structured` — concrete data questions (counts, lists, examples)
- `unstructured` — open-ended summarisation/synthesis
- `out_of_scope` — unrelated to the dataset

Only `out_of_scope` changes the flow: `route_after_router` sends it straight to `decline_node`, which returns a pre-written refusal string. No 70B agent call, no tool use — fast and cheap.

Both `structured` and `unstructured` route to the same `agent_node`. The `query_type` value is stored in state but never injected into the agent's system prompt. The LLM never explicitly "knows" it was classified as structured or unstructured.

**What actually differentiates the two** is the agent's own tool selection guided by tool descriptions:
- Structured → `count_rows`, `list_categories`, `show_examples`, `intent_distribution`
- Unstructured → `summarize_subset`

**Key takeaway:** The structured/unstructured split is a design affordance — it prepares the graph for future routing (e.g., different prompts or models per type) without currently changing the flow. The real intelligence lives in the tool descriptions, not the router label.

---

## Q2: How does the agent enforce different tool calls for structured vs unstructured queries?

**Short answer:** It doesn't enforce anything in code. Tool selection is entirely soft guidance through tool descriptions and system prompt rules — the LLM decides.

**Detail:**

All 6 tools are bound to the LLM at once via `bind_tools(ALL_TOOLS)` in `graph.py`. There is no conditional filtering by `query_type`. The `query_type` value stored in state is never injected into the agent's messages or system prompt.

The LLM is guided to pick the right tool through two layers:

**Layer 1 — Tool docstrings** (sent to the LLM as tool descriptions):
- `summarize_subset`: *"Use for open-ended questions like 'summarize the FEEDBACK category'"* → signals unstructured
- `count_rows`: *"Chain with list_intents to answer 'how many X requests?' questions"* → signals structured
- `show_examples`: *"Use for 'show me examples of X' questions"* → signals structured

**Layer 2 — System prompt rules** in `_BASE_SYSTEM_PROMPT`:
- `"For 'how many X?' questions: use list_intents to confirm the intent name, then count_rows"`
- `"For summarisation questions: use summarize_subset"`
- `"For distribution questions: use intent_distribution"`

**Key insight:** This is how LLM agents work generally — you guide tool selection with natural language descriptions, not with if/else code. The "enforcement" is probabilistic, not deterministic.

**Bonus subtlety:** `summarize_subset` makes a second LLM call *inside* the tool to synthesize the sampled rows. Unstructured queries therefore cost two LLM calls: one for the agent's reasoning step + one inside the tool for the actual summary.

---

## Q3: Where is the session saved? Why session in DB and user in JSON? How is user info distilled? How is user connected to a session?

### Where is the session saved?
Automatically by LangGraph's `SqliteSaver` checkpointer, attached at compile time:
```python
with SqliteSaver.from_conn_string(str(CHECKPOINTS_DB)) as saver:
    graph = build_graph(checkpointer=saver)   # main.py
return builder.compile(checkpointer=checkpointer)  # graph.py
```
Every `graph.stream()` call writes a checkpoint to `data/checkpoints.sqlite` after each node — no explicit "save" call. On the next turn LangGraph reads it and prepends the full message history automatically. Key = `thread_id`.

### Why session in SQLite and user in JSON?
Different shapes and access patterns:
- **Session** = append-only message list, grows every turn, needs LangGraph's internal serialization → SQLite (LangGraph manages the schema)
- **User profile** = small flat dict (name, interests, preferences), read once per turn, written once per turn → plain JSON file is sufficient, no concurrency needed

### How is user info distilled?
`profile_updater_node` fires after **every completed in-scope turn** (last node before END each turn, not just end of session). It extracts the last human message + last AI answer from state and calls the 8B LLM to extract only *personal facts* (name, role, interests, preferences) — explicitly told NOT to record query history. Returns a structured `UserProfile` Pydantic object, saved to `data/profiles/<user_id>.json`. Because the system prompt is also reloaded each turn, profile updates are visible to the agent on the very next message.

### How is user connected to a session?
They are intentionally independent — linked only at runtime via the `config` dict:
```python
config = {"configurable": {"thread_id": session_id, "user_id": user_id}}
```
- `thread_id` → SQLite checkpoint lookup (what was said)
- `user_id` → profile JSON file loaded into system prompt (who is talking)

No foreign key between them. Same session can be used with different users and vice versa. The profile is injected at the start of each agent turn via `_build_system_prompt(user_id)`.

### What does the agent actually receive each turn?
`agent_node` builds the LLM input as:
```python
messages = [system] + state["messages"]
```
Where `state["messages"]` is the full restored list + the current message already appended by `add_messages`. So the agent sees:
- **System prompt** — rebuilt by `_build_system_prompt(user_id)` on every `agent_node` call (i.e., every ReAct step). Re-reads profile JSON each time so profile updates from the previous turn are immediately reflected in the next turn. Not persisted in the checkpoint.
  - **Design note:** rebuilding per ReAct step is wasteful — the profile can only change between turns (after `profile_updater_node`), so mid-turn rebuilds are redundant file reads. Better design would build it once per turn (e.g., in `router_node` or store in state). Building once per session would be cheapest but would miss same-session profile updates.
- **All previous messages** — every `HumanMessage`, `AIMessage`, `ToolMessage` from prior turns (restored from SQLite checkpoint)
- **Current message** — the new `HumanMessage` for this turn (already merged into the list by `add_messages`)

The current message is not separate — `add_messages` appends it to the restored list before `agent_node` runs, so from the agent's perspective it's one flat continuous message list.

---

## Q5: What does the MCP server add over direct tool calls? How is the path decided?

**Short answer:** MCP is a separate public interface for external consumers — not an alternative path for the agent. There is no runtime routing decision.

**Detail:**

The agent's tools (`agent/tools.py`) are LangChain `@tool` objects, bound in-process to the LangGraph agent via `bind_tools(ALL_TOOLS)`. They only work inside this agent.

The MCP server (`mcp_server.py`) re-wraps the same underlying data functions (`agent/dataset.py`, `_filter`) as a **standard protocol service** using `@mcp.tool()`. Any MCP-compatible client — Claude Desktop, another agent, a different app — can connect to it over stdio or HTTP/SSE.

**Two completely separate entry points:**
- LangGraph agent → calls `agent/tools.py` directly in-process, never touches MCP
- External MCP client → connects to `mcp_server.py` as a subprocess or HTTP service, never goes through the agent

There is no runtime decision between the two paths. The test (`test_task3.py`) uses `fastmcp.Client` to talk directly to the MCP server — it completely bypasses the LangGraph agent.

**What MCP actually adds:**
1. **Interoperability** — any MCP client can use the tools, not just this agent
2. **Transport flexibility** — stdio or HTTP/SSE, accessible remotely
3. **Separation of concerns** — data access layer becomes a standalone reusable service
4. **Discoverability** — clients can call `client.list_tools()` to inspect available tools

**Key takeaway:** The point of task 3 is to practice exposing tools as a reusable service. MCP does not change how the agent works internally.

### Can you connect the MCP server back to the ReAct agent?

Yes — `langchain-mcp-adapters` converts MCP tools into LangChain tools, so they can be passed directly to `bind_tools()`:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

async with MultiServerMCPClient({
    "bitext": {"command": "python", "args": ["mcp_server.py"]}  # stdio
    # or HTTP: {"url": "http://127.0.0.1:8000/mcp"}
}) as mcp_client:
    tools = mcp_client.get_tools()
    llm_with_tools = get_generation_llm().bind_tools(tools)
```

The agent would then call tools over the MCP protocol to a separate process instead of in-process.

**When to use MCP vs direct tools:**

| Scenario | Use MCP | Use direct tools |
|---|---|---|
| Tools run on a remote server | Yes | No |
| Tools shared across multiple agents | Yes | No |
| Same process, single agent | Overkill | Yes |

For this project the direct approach is correct — MCP server and agent are in the same repo. The MCP server's value is for external clients, not for wiring back into the same agent.

---

## Q4: How is follow-up question context implemented?

**Short answer:** The previous AI answer (up to 400 chars) is passed as `context` to `classify_query` in `router_node`. That is the only mechanism.

**Detail:**

In `router_node` (`graph.py:80`):
```python
context = str(last_ai.content)[:400] if (last_ai and prev_was_in_scope) else ""
qt = classify_query(last_human.content, context=context)
```

The router receives the current human query and the last AI response as context. This lets it correctly classify short follow-ups like "show me 3 more" as `structured` instead of `out_of_scope`.

Only the AI answer is included — not the previous human question — for two reasons:
1. The AI response already describes what was being discussed, which is sufficient for classification.
2. If the prior human turn was out-of-scope, passing it could trick the router into misclassifying an unrelated new query as a continuation (the `prev_was_in_scope` guard also handles this).

**Key takeaway:** Follow-up context in the router = last AI answer only. Full message history (both human and AI) is always available to the agent via state, but the router only needs the last AI response to detect dataset-related follow-ups.

---
