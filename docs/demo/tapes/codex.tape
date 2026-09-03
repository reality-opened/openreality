# Open Reality CLI demo, Codex take (VHS: https://github.com/charmbracelet/vhs)
#
# A real Codex session driving the openreality MCP tools against the offline
# simulator (deterministic backend, no account, no GPU). Do not run this file
# directly: ./render.sh codex <green|black|grey> adds the Output and Theme
# header and writes the finished tape to build/.
#
# Needs: vhs, ttyd, ffmpeg, a logged-in `codex` CLI. A local mcp/ build is
# preferred; setup falls back to npm.
#
# Timing notes (Codex 0.15x): Codex draws its transcript into the terminal's
# scrollback, and VHS can only read the first screenful of the buffer, so a
# text wait stops matching after the first turn. The beats therefore use
# fixed pauses sized for medium reasoning effort (the -c override on the
# launch line). If a turn outlasts its pause, Codex queues the
# next prompt and sends it when the turn ends; lengthen the pause if a take
# shows that.

Set Shell bash
Set FontSize 18
Set Width 1200
Set Height 700
Set Padding 20
Set Framerate 15
Set PlaybackSpeed 2.5
Set TypingSpeed 50ms

# ---- hidden bootstrap: simulator + throwaway Codex project -----------------
Hide
Type "source ./setup.sh"
Enter
Wait+Screen@120s /DEMO READY|DEMO FAILED/
Type "clear"
Enter
Show

# ---- install lines, on camera ----------------------------------------------
Sleep 800ms
Type "# Install Open Reality for Codex (this take uses the bundled offline simulator)"
Sleep 300ms
Enter
Type@15ms "codex mcp add openreality-demo --env OPENREALITY_URL=http://127.0.0.1:@PORT@ --env OPENREALITY_TOKEN=sim-token --env OPENREALITY_DIR=$PWD/state --env OPENREALITY_ARTIFACTS_DIR=$PWD/artifacts -- npx -y openreality-mcp serve"
Sleep 400ms
Enter
Wait+Screen@60s /Added global MCP/
# Let the command finish before typing again; keystrokes sent while it is
# still exiting get swallowed.
Sleep 1500ms

# ---- open Codex --------------------------------------------------------------
Type@15ms "codex --approve-for-me -c model_reasoning_effort=medium"
Sleep 500ms
Enter
Wait+Screen@60s /Ask Codex to do anything/
Sleep 3s

# ---- beat 1: video in, scene out --------------------------------------------
Type "use only the openreality-demo MCP server and keep answers short: upload office-walkthrough.mp4, wait for reconstruction, and tell me the new scan id"
Sleep 400ms
Enter
Sleep 60s

# ---- beat 2: honest relative units ------------------------------------------
Type "how far is the desk from the office chair in it?"
Sleep 400ms
Enter
Sleep 35s

# ---- beat 3: calibrate, then metres -----------------------------------------
Type "the desk is actually 1.2m from the chair. calibrate the scan with that, then measure desk to chair again"
Sleep 400ms
Enter
Sleep 45s

# ---- beat 4: robot-training export, artifact on disk ------------------------
Type "export the scan as robot training data, save the zip in this folder, and list the files inside it"
Sleep 400ms
Enter
Sleep 55s

Sleep 5s

# ---- hidden teardown --------------------------------------------------------
Hide
Ctrl+C
Sleep 500ms
Ctrl+C
Sleep 1s
Type "exit"
Enter
Sleep 500ms
Type "kill $(cat $PWD/sim.pid) 2>/dev/null; rm -f $PWD/sim.pid; codex mcp remove openreality-demo; true"
Enter
