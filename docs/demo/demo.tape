# Open Reality CLI demo (VHS: https://github.com/charmbracelet/vhs)
#
# Renders a REAL Claude Code session driving the openreality MCP tools against
# the offline simulator (deterministic backend, no account, no GPU). Re-render
# any time with:  cd docs/demo && vhs demo.tape
#
# Needs: vhs, ttyd, ffmpeg, a logged-in `claude` CLI, and `npm run build` done
# in mcp/ (setup.sh falls back to the published npm package otherwise).

Output demo.gif
Output demo.mp4

Set Shell bash
Set FontSize 18
Set Width 1200
Set Height 700
Set Padding 20
Set Framerate 15
Set PlaybackSpeed 2.5
Set TypingSpeed 100ms
Set Theme { "name": "openreality", "background": "#071c15", "foreground": "#e8e3d5", "cursor": "#d4a017", "selection": "#0b3d2c", "black": "#0b3d2c", "red": "#e07070", "green": "#3fa06f", "yellow": "#d4a017", "blue": "#7fa8c9", "magenta": "#b48ead", "cyan": "#7fc7bd", "white": "#e8e3d5", "brightBlack": "#3a5c4c", "brightRed": "#f0a0a0", "brightGreen": "#66c294", "brightYellow": "#f0c75e", "brightBlue": "#a3c6e0", "brightMagenta": "#d0b8d8", "brightCyan": "#a5ded6", "brightWhite": "#ffffff" }

# ---- hidden bootstrap: simulator + throwaway Claude project ----------------
Hide
Type "source ./setup.sh"
Enter
Wait+Screen@60s /DEMO READY/
Type "clear"
Enter
Show

# ---- open Claude Code -------------------------------------------------------
Sleep 800ms
Type "claude --model sonnet"
Sleep 500ms
Enter
Wait+Screen@120s /❯/
Sleep 2s

# ---- beat 1: video in, scene out --------------------------------------------
Type "upload office-walkthrough.mp4, wait for the reconstruction, and tell me the new scan id"
Sleep 400ms
Enter
Wait+Screen@240s /sim-scan-/
Sleep 8s

# ---- beat 2: honest relative units ------------------------------------------
Type "how far is the desk from the office chair in it?"
Sleep 400ms
Enter
Wait+Screen@180s /relative/
Sleep 10s

# ---- beat 3: calibrate, then metres -----------------------------------------
# (No content wait here: this Claude build renders MCP calls without the tool
# name, and every distinctive completion word also appears in the typed prompt
# or the previous turn. A generous fixed sleep covers the three-tool turn.)
Type "the desk is actually 1.6m from the chair. calibrate the scan with that, then measure desk to chair again"
Sleep 400ms
Enter
Sleep 40s

# ---- beat 4: robot-training export, artifact on disk ------------------------
Type "export the scan as robot training data, save the zip in this folder, and show me what is inside it"
Sleep 400ms
Enter
Wait+Screen@240s /trajectory\.npz/
Sleep 12s

Sleep 5s

# ---- hidden teardown --------------------------------------------------------
Hide
Ctrl+C
Sleep 500ms
Type "exit"
Enter
Sleep 500ms
Type "kill $(cat /tmp/openreality-vhs-demo/sim.pid) 2>/dev/null; true"
Enter
