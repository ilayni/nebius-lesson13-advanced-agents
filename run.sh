#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 [--mcp] [--ui] [-- agent args...]"
  echo ""
  echo "  --mcp   Start the agent CLI + MCP server (HTTP on port 8000)"
  echo "  --ui    Start the agent CLI + MCP server + Streamlit UI"
  echo ""
  echo "  Any extra arguments are forwarded to main.py (e.g. --session foo --user bar)"
  exit 1
}

START_MCP=false
START_UI=false
AGENT_ARGS=()
PIDS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mcp) START_MCP=true; shift ;;
    --ui)  START_UI=true; START_MCP=true; shift ;;
    -h|--help) usage ;;
    *) AGENT_ARGS=("$@"); break ;;
  esac
done

if ! $START_MCP && ! $START_UI; then
  usage
fi

cleanup() {
  echo ""
  echo "Shutting down..."
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

wait_for_pid() {
  local pid=$1 name=$2 timeout=${3:-5}
  for _ in $(seq 1 "$timeout"); do
    sleep 1
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "ERROR: $name failed to start."
      exit 1
    fi
  done
}

if $START_MCP; then
  echo "Starting MCP server (HTTP on http://127.0.0.1:8000)..."
  python mcp_server.py --http &
  MCP_PID=$!
  PIDS+=("$MCP_PID")
  wait_for_pid "$MCP_PID" "MCP server" 2
  echo "MCP server running (PID $MCP_PID)"
fi

if $START_UI; then
  echo "Starting Streamlit UI (http://localhost:8501)..."
  streamlit run streamlit_app.py --server.headless true &
  UI_PID=$!
  PIDS+=("$UI_PID")
  wait_for_pid "$UI_PID" "Streamlit UI" 5
  echo "Streamlit UI running (PID $UI_PID)"
fi

echo ""
echo "Starting agent CLI..."
python main.py "${AGENT_ARGS[@]}"
