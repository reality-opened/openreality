#!/usr/bin/env bash
# Demo workspace bootstrap, sourced by the hidden section of every tape in
# docs/demo/tapes/ (also runnable standalone). Creates a throwaway project
# folder wired to the openreality MCP server against the LOCAL offline
# simulator: no account, no GPU, deterministic backend data.
#
# Inputs (environment, all optional; render.sh sets them):
#   DEMO_AGENT      codex | claude | cursor        (default: codex)
#   DEMO_PORT       simulator port                 (default: 8973)
#   DEMO_WORKSPACE  throwaway project folder       (default: /Users/Shared/openreality-demo
#                   on macOS, else /tmp/openreality-demo: a path that does not
#                   contain the user name, since the folder shows on camera)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEMO_AGENT="${DEMO_AGENT:-codex}"
DEMO_PORT="${DEMO_PORT:-8973}"
if [ -z "${DEMO_WORKSPACE:-}" ]; then
  if [ -d /Users/Shared ] && [ -w /Users/Shared ]; then
    DEMO_WORKSPACE=/Users/Shared/openreality-demo
  else
    DEMO_WORKSPACE=/tmp/openreality-demo
  fi
fi
DEMO="$DEMO_WORKSPACE"
MARKER="$DEMO/.openreality-demo-workspace"

demo_fail() {
  echo "DEMO FAILED: $1"
  return 1 2>/dev/null || exit 1
}

case "$DEMO_AGENT" in
  codex|claude|cursor) ;;
  *) demo_fail "unknown DEMO_AGENT '$DEMO_AGENT' (codex, claude, or cursor)"; return 1 2>/dev/null || exit 1 ;;
esac

# Refuse to fight a stale simulator for the port, and keep its PID file intact
# so the operator can identify it.
if curl -s -o /dev/null -m 1 "http://127.0.0.1:$DEMO_PORT/" 2>/dev/null; then
  demo_fail "port $DEMO_PORT already in use (stale simulator?)"; return 1 2>/dev/null || exit 1
fi

# The workspace is recreated from scratch on every take. Only ever delete a
# folder this script created (it carries the marker file).
if [ -e "$DEMO" ] && [ ! -e "$MARKER" ]; then
  demo_fail "$DEMO exists and is not a demo workspace; set DEMO_WORKSPACE elsewhere"; return 1 2>/dev/null || exit 1
fi
rm -rf "$DEMO"
mkdir -p "$DEMO/state" "$DEMO/artifacts"
touch "$MARKER"

# Prefer the CLI built in this checkout; fall back to the published package.
if [ -f "$REPO_ROOT/mcp/dist/cli.js" ]; then
  CLI_CMD="node"; CLI_BASE_ARGS="$REPO_ROOT/mcp/dist/cli.js"
else
  CLI_CMD="npx"; CLI_BASE_ARGS="-y openreality-mcp"
fi

# Offline simulator (the same mock backend `openreality-mcp simulator` serves).
$CLI_CMD $CLI_BASE_ARGS simulator --port "$DEMO_PORT" >"$DEMO/sim.log" 2>&1 &
SIMULATOR_PID=$!
echo "$SIMULATOR_PID" > "$DEMO/sim.pid"
code=""
for _ in $(seq 1 100); do
  code=$(curl -s -o /dev/null -w '%{http_code}' -H 'Authorization: Bearer sim-token' "http://127.0.0.1:$DEMO_PORT/api/scenes" 2>/dev/null)
  [ "$code" = "200" ] && break
  sleep 0.2
done
if [ "$code" != "200" ]; then
  kill "$SIMULATOR_PID" >/dev/null 2>&1 || true
  demo_fail "simulator did not become ready; see $DEMO/sim.log"; return 1 2>/dev/null || exit 1
fi

# VHS is usually launched from an agent session. Do not let the recorded agent
# inherit markers that identify it as a nested or sandboxed session (Claude
# Code, Codex, tmux). CODEX_HOME and package-location variables stay intact.
for DEMO_ENV_NAME in $(env | grep -oE '^(CLAUDE[A-Z_]*|CODEX_CI|CODEX_SANDBOX[A-Z_]*|CODEX_SESSION_ID|CODEX_THREAD_ID|CODEX_COMPANION[A-Z_]*|TMUX)'); do
  unset "$DEMO_ENV_NAME" 2>/dev/null || true
done

# Pre-warm the npx cache so the on-camera `npx -y openreality-mcp serve` boots
# instantly (whoami fails without credentials; the download is what matters).
npx -y openreality-mcp whoami >/dev/null 2>&1 || true

# A stand-in capture for the upload beat (the simulator does not decode video).
head -c 65536 /dev/urandom > "$DEMO/office-walkthrough.mp4"

# Point the MCP server at the simulator and keep its state and fetched
# artifacts inside the throwaway folder. The server started by the assistant
# inherits these from this shell, so the on-camera install lines are the
# normal ones from the README (no login step: the simulator needs none).
export OPENREALITY_URL="http://127.0.0.1:$DEMO_PORT"
export OPENREALITY_TOKEN="sim-token"
export OPENREALITY_DIR="$DEMO/state"
export OPENREALITY_ARTIFACTS_DIR="$DEMO/artifacts"

cd "$DEMO" || { demo_fail "cannot enter $DEMO"; return 1 2>/dev/null || exit 1; }

# Every tape owns a demo-only MCP entry named `openreality-demo` and never
# touches the user's normal `openreality` entry. Remove only a stale demo entry
# left by an interrupted take, and settle the one-time dialogs that would
# otherwise appear on camera.
case "$DEMO_AGENT" in
  codex)
    command -v codex >/dev/null 2>&1 && codex mcp remove openreality-demo >/dev/null 2>&1
    # Codex asks "Do you trust the contents of this directory?" in a folder it
    # has not seen. Pre-trust the throwaway workspace only, exactly as
    # answering "Yes, continue" would (it writes the same config entry).
    # (Codex ignores a project-level .codex/config.toml here, and 0.15x
    # rejects --profile with an inline profile table, so the take passes its
    # reasoning-effort override on the command line instead.)
    python3 - "$DEMO" <<'PY'
import os, sys
path = os.path.expanduser("~/.codex/config.toml")
demo = sys.argv[1]
header = '[projects."%s"]' % demo
try:
    with open(path) as fh:
        text = fh.read()
except FileNotFoundError:
    text = ""
if header not in text:
    if text and not text.endswith("\n"):
        text += "\n"
    text += "\n%s\ntrust_level = \"trusted\"\n" % header
    with open(path, "w") as fh:
        fh.write(text)
PY
    ;;
  claude)
    # On camera the take runs the README's own lines: `claude mcp add --scope
    # project openreality -- npx -y openreality-mcp serve` (writes ./.mcp.json,
    # and prints that path rather than a home-folder path) and then
    # `claude --model opus`. This shell function makes that launch load the
    # project's .mcp.json and nothing else, so the recording machine's other
    # MCP servers (and their "needs authentication" warning) stay out of the
    # take. `claude mcp ...` subcommands pass through untouched.
    claude() {
      if [ "$1" = "mcp" ] || [ ! -f .mcp.json ]; then
        command claude "$@"
      else
        command claude --strict-mcp-config --mcp-config .mcp.json "$@"
      fi
    }
    # Project settings for the take: the server's tools plus unzip and ls
    # are pre-approved (no permission prompt on camera) and the effort level
    # is medium so Opus answers in seconds rather than deliberating.
    mkdir -p "$DEMO/.claude"
    cat > "$DEMO/.claude/settings.json" <<'JSON'
{
  "effortLevel": "medium",
  "permissions": {
    "allow": ["mcp__openreality", "Bash(unzip:*)", "Bash(ls:*)"]
  }
}
JSON
    # The folder-trust prompt is per folder: pre-accept it for the throwaway
    # workspace only, exactly as answering "Yes" would.
    python3 - "$DEMO" <<'PY'
import json, os, sys, tempfile
path = os.path.expanduser("~/.claude.json")
demo = sys.argv[1]
try:
    with open(path) as fh:
        cfg = json.load(fh)
except FileNotFoundError:
    cfg = {}
project = cfg.setdefault("projects", {}).setdefault(demo, {})
project["hasTrustDialogAccepted"] = True
fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".claude.json.")
with os.fdopen(fd, "w") as fh:
    json.dump(cfg, fh, indent=2)
os.replace(tmp, path)
PY
    ;;
  cursor)
    # Cursor reads MCP servers from .cursor/mcp.json in the project. The tape
    # shows this file on camera in place of an `mcp add` command.
    mkdir -p "$DEMO/.cursor"
    cat > "$DEMO/.cursor/mcp.json" <<JSON
{
  "mcpServers": {
    "openreality-demo": {
      "command": "npx",
      "args": ["-y", "openreality-mcp", "serve"],
      "env": {
        "OPENREALITY_URL": "http://127.0.0.1:$DEMO_PORT",
        "OPENREALITY_TOKEN": "sim-token",
        "OPENREALITY_DIR": "$DEMO/state",
        "OPENREALITY_ARTIFACTS_DIR": "$DEMO/artifacts"
      }
    }
  }
}
JSON
    # Approve the demo server so the agent does not stop to ask on camera.
    command -v cursor-agent >/dev/null 2>&1 && cursor-agent mcp enable openreality-demo >/dev/null 2>&1
    ;;
esac

echo "DEMO READY in $DEMO (agent: $DEMO_AGENT, simulator: http://127.0.0.1:$DEMO_PORT)"
