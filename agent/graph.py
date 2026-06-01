from __future__ import annotations

from typing import Annotated, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from agent.config import MAX_ITERATIONS
from agent.llm import get_generation_llm
from agent.profile import format_profile_block, load_profile, update_profile
from agent.router import QueryType, classify_query
from agent.tools import ALL_TOOLS


# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages:   Annotated[list, add_messages]  # accumulated conversation messages
    query_type: str                             # set by router_node each turn
    iterations: int                             # reset to 0 each turn by router_node


# ── Fixed strings ─────────────────────────────────────────────────────────────

_BASE_SYSTEM_PROMPT = (
    "You are a data analyst assistant for the Bitext Customer Service dataset — "
    "a collection of customer support queries and agent responses grouped by category and intent.\n"
    "Rules:\n"
    "• Always use your tools to retrieve data. Never guess counts, categories, or examples.\n"
    "• For 'how many X?' questions: use list_intents to confirm the intent name, then count_rows.\n"
    "• For 'show me examples' follow-ups: increment the offset argument of show_examples by n.\n"
    "• For summarisation questions: use summarize_subset.\n"
    "• For distribution questions: use intent_distribution.\n"
    "• If the user asks what you remember or know about them, answer from the "
    "[User Profile] block below — do not use tools for that."
)

_DECLINE_MSG = (
    "I'm sorry — I can only answer questions about the Bitext Customer Service dataset. "
    "That question is outside my scope. Try asking about categories, intents, "
    "row counts, examples, or patterns in the data."
)

_FALLBACK_MSG = (
    f"I wasn't able to produce a complete answer within {MAX_ITERATIONS} reasoning steps. "
    "Please try rephrasing or breaking the question into smaller parts."
)


def _build_system_prompt(user_id: str) -> str:
    profile = load_profile(user_id)
    block = format_profile_block(profile)
    if block:
        return f"{_BASE_SYSTEM_PROMPT}\n\n{block}"
    return _BASE_SYSTEM_PROMPT


# ── Nodes ─────────────────────────────────────────────────────────────────────

def router_node(state: AgentState) -> dict:
    """Classify the latest user query and reset the iteration counter.

    Passes the previous agent response as context so short follow-ups like
    'show me 3 more' or 'what about refunds?' are not misclassified as out_of_scope.
    """
    last_human = next(m for m in reversed(state["messages"]) if isinstance(m, HumanMessage))
    last_ai    = next(
        (m for m in reversed(state["messages"])
         if isinstance(m, AIMessage) and m.content and not m.tool_calls),
        None,
    )
    # Only pass previous context when the last turn was in-scope; passing an
    # out_of_scope context could cause a genuinely unrelated follow-up query
    # to be misclassified as a dataset continuation.
    prev_was_in_scope = state.get("query_type") != QueryType.out_of_scope
    context = str(last_ai.content)[:400] if (last_ai and prev_was_in_scope) else ""
    qt = classify_query(last_human.content, context=context)
    return {"query_type": qt, "iterations": 0}


def decline_node(_state: AgentState) -> dict:
    """Return a polite refusal for out-of-scope queries — no LLM call."""
    return {"messages": [AIMessage(content=_DECLINE_MSG)]}


def fallback_node(_state: AgentState) -> dict:
    """Return a graceful message when MAX_ITERATIONS is reached."""
    return {"messages": [AIMessage(content=_FALLBACK_MSG)]}


def profile_updater_node(state: AgentState, config: RunnableConfig) -> dict:
    """Distil the latest exchange into the persistent user profile (Task 2b)."""
    user_id = config["configurable"].get("user_id", "default_user")

    human_msg = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )
    ai_msg = next(
        (m.content for m in reversed(state["messages"])
         if isinstance(m, AIMessage) and m.content and not m.tool_calls),
        None,
    )

    if human_msg and ai_msg:
        update_profile(user_id, human_msg, str(ai_msg))

    return {}


# ── Graph factory ─────────────────────────────────────────────────────────────

def build_graph(checkpointer=None):
    """Compile and return the ReAct agent StateGraph.

    Args:
        checkpointer: Optional LangGraph checkpointer (e.g. SqliteSaver) for
                      persistent conversation memory (Task 2a).
                      Pass None (default) for stateless behaviour.

    Returns:
        A compiled LangGraph CompiledStateGraph ready for invoke/stream calls.
    """
    llm_with_tools = get_generation_llm().bind_tools(ALL_TOOLS)

    def agent_node(state: AgentState, config: RunnableConfig) -> dict:
        """One ReAct step: LLM decides to call a tool or emit the final answer."""
        user_id  = config["configurable"].get("user_id", "default_user")
        system   = SystemMessage(content=_build_system_prompt(user_id))
        messages = [system] + state["messages"]
        response  = llm_with_tools.invoke(messages)
        return {"messages": [response], "iterations": state["iterations"] + 1}

    # ── Conditional routing ───────────────────────────────────────────────────

    def route_after_router(state: AgentState) -> Literal["decline", "agent"]:
        return "decline" if state["query_type"] == QueryType.out_of_scope else "agent"

    def route_after_agent(
        state: AgentState,
    ) -> Literal["tools", "fallback", "profile_updater"]:
        if state["iterations"] >= MAX_ITERATIONS:
            return "fallback"
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "profile_updater"

    # ── Assemble ──────────────────────────────────────────────────────────────

    builder = StateGraph(AgentState)

    builder.add_node("router",          router_node)
    builder.add_node("decline",         decline_node)
    builder.add_node("agent",           agent_node)
    builder.add_node("tools",           ToolNode(ALL_TOOLS))
    builder.add_node("fallback",        fallback_node)
    builder.add_node("profile_updater", profile_updater_node)

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router", route_after_router,
        {"decline": "decline", "agent": "agent"},
    )
    builder.add_edge("decline", END)
    builder.add_conditional_edges(
        "agent", route_after_agent,
        {"tools": "tools", "fallback": "fallback", "profile_updater": "profile_updater"},
    )
    builder.add_edge("tools",           "agent")
    builder.add_edge("fallback",        END)
    builder.add_edge("profile_updater", END)

    return builder.compile(checkpointer=checkpointer)


# Module-level graph for langgraph dev / LangSmith Studio
graph = build_graph()
