#!/usr/bin/env bash
# Demo workspace bootstrap for docs/demo/demo.tape (sourced by the tape's
# hidden section; also runnable standalone). Creates a throwaway Codex
# project wired to the openreality MCP server against the LOCAL offline
# simulator: no account, no GPU, deterministic backend data.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEMO="$SCRIPT_DIR/.demo-workspace"

# Refuse to fight a stale simulator for the port, and keep its PID file intact
# so the operator can identify it.
if curl -s -o /dev/null -m 1 http://127.0.0.1:8973/ 2>/dev/null; then
  echo "DEMO FAILED: port 8973 already in use (stale simulator?)"
  return 1 2>/dev/null || exit 1
fi

rm -rf "$DEMO"
mkdir -p "$DEMO/state" "$DEMO/artifacts"

# Prefer the CLI built in this checkout; fall back to the published package.
if [ -f "$REPO_ROOT/mcp/dist/cli.js" ]; then
  CLI_CMD="node"; CLI_BASE_ARGS="$REPO_ROOT/mcp/dist/cli.js"
else
  CLI_CMD="npx"; CLI_BASE_ARGS="-y openreality-mcp"
fi

# Offline simulator (the same mock backend `openreality-mcp simulator` serves).
$CLI_CMD $CLI_BASE_ARGS simulator --port 8973 >"$DEMO/sim.log" 2>&1 &
SIMULATOR_PID=$!
echo "$SIMULATOR_PID" > "$DEMO/sim.pid"
code=""
for _ in $(seq 1 50); do
  code=$(curl -s -o /dev/null -w '%{http_code}' -H 'Authorization: Bearer sim-token' http://127.0.0.1:8973/api/scenes 2>/dev/null)
  [ "$code" = "200" ] && break
  sleep 0.2
done
if [ "$code" != "200" ]; then
  kill "$SIMULATOR_PID" >/dev/null 2>&1 || true
  echo "DEMO FAILED: simulator did not become ready; see $DEMO/sim.log"
  return 1 2>/dev/null || exit 1
fi

# VHS may itself be launched from an agent session. Do not make the recorded
# Codex process inherit markers that identify it as a nested/sandboxed session.
# Keep CODEX_HOME and package-location variables intact.
for DEMO_ENV_NAME in \
  CODEX_CI CODEX_SANDBOX CODEX_SANDBOX_NETWORK_DISABLED \
  CODEX_SESSION_ID CODEX_THREAD_ID CODEX_COMPANION_SESSION_ID \
  CODEX_COMPANION_TRANSCRIPT_PATH CLAUDECODE CLAUDE_PLUGIN_DATA TMUX; do
  unset "$DEMO_ENV_NAME" 2>/dev/null || true
done

# Pre-warm the npx cache so the on-camera `npx -y openreality-mcp serve` boots
# instantly (whoami fails without credentials; the download is what matters).
npx -y openreality-mcp whoami >/dev/null 2>&1 || true

# The tape owns a demo-only MCP entry and never touches the user's normal
# `openreality` entry. Remove only a stale entry left by an interrupted take.
command -v codex >/dev/null 2>&1 && codex mcp remove openreality-demo >/dev/null 2>&1

# A stand-in capture for the upload beat (the simulator does not decode video).
head -c 65536 /dev/urandom > "$DEMO/office-walkthrough.mp4"

cd "$DEMO" || exit 1
echo "DEMO READY in $DEMO"
