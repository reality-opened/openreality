# Open Reality CLI demo, Claude Code take (VHS: https://github.com/charmbracelet/vhs)
#
# A real Claude Code session (Opus by default; DEMO_MODEL overrides) driving
# the openreality MCP tools against the offline simulator (deterministic
# backend, no account, no GPU). Do not run this file directly:
# ./render.sh claude <green|black|grey> adds the Output and Theme header and
# writes the finished tape to build/.
#
# Needs: vhs, ttyd, ffmpeg, a logged-in `claude` CLI. A local mcp/ build is
# preferred; setup falls back to npm.
#
# The opening shows the README's normal install lines. setup.sh (hidden)
# exports the simulator URL and token and defines a `claude` shell function
# that makes the launch load only this project's .mcp.json; see README.md.
#
# Screen notes (Claude Code 2.1.x): MCP calls render as "Called <server>"
# without the tool name, and the answer can mention any number before the
# tool call happens, so waits key on the idle status line instead. While a
# turn runs that line reads "... (shift+tab to cycle) · esc to interrupt ...";
# when idle it reads "... (shift+tab to cycle)" optionally followed by an
# agent count and the "/rc" hint, or "? for shortcuts" on default settings.
#
# Typing: prompts type at the default speed; command lines use Type@15ms so
# a long line lands in about a second of video.

Set Shell bash
Set FontSize 18
Set Width 1200
Set Height 700
Set Padding 20
Set Framerate 15
Set PlaybackSpeed 2.5
Set TypingSpeed 50ms

# ---- hidden bootstrap: simulator + throwaway Claude Code project ----------
Hide
Type "source ./setup.sh"
Enter
Wait+Screen@120s /DEMO READY|DEMO FAILED/
Type "clear"
Enter
Show

# ---- install lines, on camera (the README's own) ---------------------------
Sleep 800ms
Type@15ms "# Install Open Reality for Claude Code (offline simulator take: no account, no login)"
Sleep 300ms
Enter
Sleep 1200ms
Type@15ms "claude mcp add --scope project openreality -- npx -y openreality-mcp serve"
Sleep 400ms
Enter
Wait+Screen@60s /Added stdio MCP server/
# Hold the install lines so viewers can read them (Claude Code clears the
# screen when it starts); this pause also lets the command finish, since
# keystrokes sent while it is still exiting get swallowed.
Sleep 4s

# ---- open Claude Code --------------------------------------------------------
Type@15ms "claude @MODEL_FLAG@"
Sleep 1500ms
Enter
Wait+Screen@60s /shift\+tab to cycle|for shortcuts/
Sleep 3s

# ---- beat 1: video in, scene out --------------------------------------------
Type "use only the openreality MCP server and keep answers short: upload office-walkthrough.mp4, wait for reconstruction, and tell me the new scan id"
Sleep 400ms
Enter
Sleep 2s
Wait+Screen@300s /(?m)shift\+tab to cycle\)( · ← \d+ agents?)?( +.rc)? *$|\? for shortcuts *$/
Sleep 10s

# ---- beat 2: honest relative units ------------------------------------------
Type "how far is the desk from the office chair in it?"
Sleep 400ms
Enter
Sleep 2s
Wait+Screen@300s /(?m)shift\+tab to cycle\)( · ← \d+ agents?)?( +.rc)? *$|\? for shortcuts *$/
Sleep 12s

# ---- beat 3: calibrate, then metres -----------------------------------------
Type "the desk is actually 1.2m from the chair. calibrate the scan with that, then measure desk to chair again"
Sleep 400ms
Enter
Sleep 2s
Wait+Screen@300s /(?m)shift\+tab to cycle\)( · ← \d+ agents?)?( +.rc)? *$|\? for shortcuts *$/
Sleep 12s

# ---- beat 4: robot-training export, artifact on disk ------------------------
Type "export the scan as robot training data, save the zip in this folder, and list the files inside it"
Sleep 400ms
Enter
Sleep 2s
Wait+Screen@300s /(?m)shift\+tab to cycle\)( · ← \d+ agents?)?( +.rc)? *$|\? for shortcuts *$/
Sleep 14s

Sleep 5s

# ---- hidden teardown --------------------------------------------------------
Hide
Type "/exit"
Enter
Sleep 2s
Type "kill $(cat $PWD/sim.pid) 2>/dev/null; rm -f $PWD/sim.pid; true"
Enter
