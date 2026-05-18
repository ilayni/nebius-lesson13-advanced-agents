"""Streamlit chat UI for the Bitext Customer Service Data Analyst Agent.

Run:
    streamlit run streamlit_app.py

The CLI (python main.py) continues to work unchanged — both share the same
data/checkpoints.sqlite and data/profiles/ directory.
"""

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from agent.config import CHECKPOINTS_DB
from agent.graph import build_graph


# ── Cached graph (created once per process, shared across all reruns) ─────────

@st.cache_resource
def get_graph():
    """Build and cache the graph with a persistent SqliteSaver connection."""
    conn = sqlite3.connect(str(CHECKPOINTS_DB), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return build_graph(checkpointer=saver)


# ── Session state initialisation ──────────────────────────────────────────────

def _init_state():
    if "session_id" not in st.session_state:
        st.session_state.session_id = "default"
    if "user_id" not in st.session_state:
        st.session_state.user_id = "default_user"
    if "display_messages" not in st.session_state:
        st.session_state.display_messages = []


# ── Streaming helper ──────────────────────────────────────────────────────────

def stream_query(query: str) -> dict:
    """Run one query through the graph and return a structured result dict."""
    graph  = get_graph()
    config = {
        "configurable": {
            "thread_id": st.session_state.session_id,
            "user_id":   st.session_state.user_id,
        }
    }
    initial = {
        "messages":   [HumanMessage(content=query)],
        "query_type": "",
        "iterations": 0,
    }

    result = {"steps": [], "answer": None}

    for chunk in graph.stream(initial, config=config, stream_mode="updates"):
        for node_name, output in chunk.items():
            if node_name in ("router", "profile_updater") or output is None:
                continue
            for msg in output.get("messages", []):
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for tc in msg.tool_calls:
                        result["steps"].append({
                            "type": "tool_call",
                            "name": tc["name"],
                            "args": tc["args"],
                        })
                elif isinstance(msg, ToolMessage):
                    preview = str(msg.content)
                    if len(preview) > 800:
                        preview = preview[:800] + " …[truncated]"
                    result["steps"].append({
                        "type":    "observation",
                        "name":    msg.name,
                        "content": preview,
                    })
                elif isinstance(msg, AIMessage) and msg.content:
                    result["answer"] = msg.content

    return result


# ── Rendering helpers ─────────────────────────────────────────────────────────

def _render_steps(steps: list):
    """Render tool calls and observations as collapsible expanders."""
    for step in steps:
        if step["type"] == "tool_call":
            args_str = ", ".join(f"{k}={v!r}" for k, v in step["args"].items()) or "—"
            with st.expander(f"⚙ Tool: `{step['name']}({args_str})`", expanded=False):
                st.json(step["args"])
        elif step["type"] == "observation":
            with st.expander(f"👁 Result: `{step['name']}`", expanded=False):
                st.text(step["content"])


def _render_history():
    """Replay all display_messages into the chat container."""
    for msg in st.session_state.display_messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                _render_steps(msg.get("steps", []))
                st.markdown(msg["content"])


# ── App ───────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Bitext Analyst Agent",
        page_icon="📊",
        layout="wide",
    )

    _init_state()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("📊 Bitext Analyst")
        st.caption("Customer Service Data Analyst Agent")
        st.divider()

        new_session = st.text_input(
            "Session ID",
            value=st.session_state.session_id,
            help="Same ID restores the full conversation after a restart.",
        )
        new_user = st.text_input(
            "User ID",
            value=st.session_state.user_id,
            help="Your persistent profile is keyed to this ID.",
        )

        # Reset display history when session or user changes
        if new_session != st.session_state.session_id or new_user != st.session_state.user_id:
            st.session_state.session_id = new_session
            st.session_state.user_id    = new_user
            st.session_state.display_messages = []
            st.rerun()

        if st.button("🗑 Clear display", use_container_width=True):
            st.session_state.display_messages = []
            st.rerun()

        st.divider()
        st.caption(
            "Reasoning steps (tool calls and results) appear as collapsible "
            "blocks above each answer."
        )

    # ── Chat area ─────────────────────────────────────────────────────────────
    _render_history()

    if query := st.chat_input("Ask something about the Bitext dataset…"):
        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.display_messages.append({"role": "user", "content": query})

        # Run the agent and display the response
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                result = stream_query(query)

            _render_steps(result["steps"])
            answer = result["answer"] or "*(no answer produced)*"
            st.markdown(answer)

        st.session_state.display_messages.append({
            "role":    "assistant",
            "content": answer,
            "steps":   result["steps"],
        })


if __name__ == "__main__":
    main()
