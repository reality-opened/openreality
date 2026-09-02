#!/usr/bin/env bash
# Demo workspace bootstrap for docs/demo/demo.tape (sourced by the tape's
# hidden section; also runnable standalone). Creates a throwaway Claude Code
# project wired to the openreality MCP server against the LOCAL offline
# simulator: no account, no GPU, deterministic backend data.

DEMO="${OPENREALITY_DEMO_DIR:-/tmp/openreality-vhs-demo}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

rm -rf "$DEMO"
mkdir -p "$DEMO/state" "$DEMO/artifacts" "$DEMO/.claude"

# Prefer the CLI built in this checkout; fall back to the published package.
if [ -f "$REPO_ROOT/mcp/dist/cli.js" ]; then
  CLI_CMD="node"; CLI_BASE_ARGS="$REPO_ROOT/mcp/dist/cli.js"
else
  CLI_CMD="npx"; CLI_BASE_ARGS="-y openreality-mcp"
fi

# Refuse to fight a stale simulator for the port; fail loud instead.
if curl -s -o /dev/null -m 1 http://127.0.0.1:8973/ 2>/dev/null; then
  echo "DEMO FAILED: port 8973 already in use (stale simulator?)"; return 1 2>/dev/null || exit 1
fi

# Offline simulator (the same mock backend `openreality-mcp simulator` serves).
$CLI_CMD $CLI_BASE_ARGS simulator --port 8973 >"$DEMO/sim.log" 2>&1 &
echo $! > "$DEMO/sim.pid"
for _ in $(seq 1 50); do
  code=$(curl -s -o /dev/null -w '%{http_code}' -H 'Authorization: Bearer sim-token' http://127.0.0.1:8973/api/scenes 2>/dev/null)
  [ "$code" = "200" ] && break
  sleep 0.2
done

# Permissions for a prompt-free session. The MCP server itself is registered
# ON CAMERA by the demo's visible `claude mcp add ...` line.
python3 - "$DEMO" <<'PY'
import json, os, sys
demo = sys.argv[1]
json.dump({
    "permissions": {"allow": ["mcp__openreality", "Bash(ls:*)", "Bash(unzip:*)"]},
}, open(os.path.join(demo, ".claude", "settings.json"), "w"), indent=2)
# Pre-trust the workspace so the TUI opens straight into the session.
cfg_path = os.path.expanduser("~/.claude.json")
cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
cfg.setdefault("projects", {}).setdefault(demo, {})["hasTrustDialogAccepted"] = True
json.dump(cfg, open(cfg_path, "w"), indent=2)
PY

# A nested Claude session must not inherit the recorder session's markers
# (they trigger child-session warnings and tmux hints in the TUI).
for v in $(env | cut -d= -f1 | grep -E '^(CLAUDECODE|CLAUDE_CODE_|TMUX)'); do
  unset "$v" 2>/dev/null
done

# Pre-warm the npx cache so the on-camera `npx -y openreality-mcp serve` boots
# instantly (whoami fails without credentials; the download is what matters).
npx -y openreality-mcp whoami >/dev/null 2>&1 || true

# Codex client hygiene: the demo registers the MCP server ON CAMERA, so drop
# any stale entry first, and give the workspace a git root so Codex skips its
# untracked-folder warning.
command -v codex >/dev/null 2>&1 && codex mcp remove openreality >/dev/null 2>&1
git init -q "$DEMO" 2>/dev/null

# A stand-in capture for the upload beat (the simulator does not decode video).
head -c 65536 /dev/urandom > "$DEMO/office-walkthrough.mp4"

cd "$DEMO" || exit 1
echo "DEMO READY in $DEMO"
