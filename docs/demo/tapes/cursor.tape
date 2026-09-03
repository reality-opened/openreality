# Open Reality CLI demo, Cursor take (VHS: https://github.com/charmbracelet/vhs)
#
# A real Cursor CLI (`cursor-agent`) session driving the openreality MCP tools
# against the offline simulator (deterministic backend, no account, no GPU).
# Do not run this file directly: ./render.sh cursor <green|black|grey> adds
# the Output, Env and Theme header and writes the finished tape to build/.
#
# Needs: vhs, ttyd, ffmpeg, a logged-in `cursor-agent` CLI (run
# `cursor-agent login` once). Cursor has no `mcp add` command: setup.sh writes
# the project's .cursor/mcp.json and the take shows that file on camera.
#
# STATUS: the beats and waits mirror the validated Codex take, but this tape
# has not been rendered end to end yet (no logged-in Cursor CLI on the
# machine that wrote it). The first render will show whether Cursor's
# startup needs an extra keypress; adjust the "open Cursor" block below.

Set Shell bash
Set FontSize 18
Set Width 1200
Set Height 700
Set Padding 20
Set Framerate 15
Set PlaybackSpeed 2.5
Set TypingSpeed 50ms

# ---- hidden bootstrap: simulator + throwaway Cursor project ----------------
Hide
Type "source ./setup.sh"
Enter
Wait+Screen@120s /DEMO READY|DEMO FAILED/
Type "clear"
Enter
Show

# ---- install lines, on camera ----------------------------------------------
Sleep 800ms
Type "# Install Open Reality for Cursor: one entry in .cursor/mcp.json (this take uses the bundled offline simulator)"
Sleep 300ms
Enter
Type@15ms "cat .cursor/mcp.json"
Sleep 400ms
Enter
Wait+Screen@10s /openreality-mcp/
Sleep 2500ms

# ---- open Cursor -------------------------------------------------------------
# --force lets the agent run shell commands (unzip -l) without stopping to ask.
Type@15ms "cursor-agent @MODEL_FLAG@--force"
Sleep 500ms
Enter
Sleep 12s

# ---- beat 1: video in, scene out --------------------------------------------
Type "use only the openreality-demo MCP server: upload office-walkthrough.mp4, wait for reconstruction, and tell me the new scan id"
Sleep 400ms
Enter
Wait+Screen@240s /sim-scan-/
Sleep 10s

# ---- beat 2: honest relative units ------------------------------------------
Type "how far is the desk from the office chair in it?"
Sleep 400ms
Enter
Wait+Screen@180s /relative/
Sleep 14s

# ---- beat 3: calibrate, then metres -----------------------------------------
Type "the desk is actually 1.6m from the chair. calibrate the scan with that, then measure desk to chair again"
Sleep 400ms
Enter
Sleep 55s

# ---- beat 4: robot-training export, artifact on disk ------------------------
Type "export the scan as robot training data, save the zip in this folder, and show me what is inside it"
Sleep 400ms
Enter
Wait+Screen@240s /trajectory\.npz/
Sleep 16s

Sleep 5s

# ---- hidden teardown --------------------------------------------------------
Hide
Ctrl+C
Sleep 500ms
Ctrl+C
Sleep 500ms
Type "exit"
Enter
Sleep 500ms
Type "kill $(cat $PWD/sim.pid) 2>/dev/null; rm -f $PWD/sim.pid; true"
Enter
