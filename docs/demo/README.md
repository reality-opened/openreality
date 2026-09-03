# CLI demo recording

The README's terminal demo is rendered with
[VHS](https://github.com/charmbracelet/vhs): a real AI-assistant session
driving the openreality MCP tools against the **offline simulator**
(deterministic backend data, no account, no GPU). There is one tape per
assistant in `tapes/` (`codex.tape`, `claude.tape`, `cursor.tape`) and three
terminal backgrounds (dark green, black, dark grey). Every take has the same
four beats:

1. show the assistant's MCP installation step, then upload a video
2. measure desk to chair: the result is honestly labeled **relative units**
3. give one real-world distance, calibrate, measure again: now **metres**
4. export robot-training data and list the zip contents on disk

## Re-render

```bash
# macOS prerequisites
brew install vhs ttyd ffmpeg

# one logged-in assistant CLI per take you want:
#   codex   (npm i -g @openai/codex; codex login)
#   claude  (Claude Code; claude auth login)
#   cursor  (Cursor CLI; cursor-agent login)

cd docs/demo
./render.sh claude grey       # one take: out/claude-grey.mp4 + out/claude-grey.gif
./render.sh claude all        # the same take on green, black and grey backgrounds
./render.sh codex grey
./render.sh cursor grey
```

`render.sh <agent> <background>` builds the finished tape in `build/`
(header with outputs, environment and theme, followed by `tapes/<agent>.tape`)
and records it. Outputs land in `out/` and are not committed; copy the take
you like to a versioned name next to this file (the root README embeds
`demo-gif-v1.gif` and links `demo-video-v1.mp4`; bump the number for a new
take so old links keep working).

Environment knobs: `DEMO_MODEL` (Claude take: `opus` by default; Cursor take:
the account's default model), `DEMO_PORT` (simulator port, 8973),
`DEMO_WORKSPACE` (throwaway project folder; the default is
`/Users/Shared/openreality-demo` on macOS and `/tmp/openreality-demo`
elsewhere, deliberately a path without the user name because the folder shows
on camera), `DEMO_SKIP_RECORD=1` (reuse the existing terminal take and only
render the scene panel).

## Scene panel (optional)

Add the four times (seconds at which the take reaches each visual beat) to
render the synchronized 3D panel and composite it beside the terminal:

```bash
python3 -m venv /tmp/openreality-demo-render-env
source /tmp/openreality-demo-render-env/bin/activate
python -m pip install "numpy<2" "matplotlib>=3.8"

./render.sh claude grey 17.5 25.5 37.5 53 --t-start 8 --metric-distance 1.2   # adds out/claude-grey-scene.mp4 + .gif
```

Assistant response timing varies. Review the terminal take first, note when
the four beats land (a timestamped contact sheet helps:
`ffmpeg -i out/claude-grey.mp4 -vf "fps=1/2,drawtext=fontfile=/System/Library/Fonts/Supplemental/Arial.ttf:text='%{pts\:hms}':fontsize=40:fontcolor=yellow:x=20:y=20,scale=480:-1,tile=4x11" -frames:v 1 build/sheet.png`),
then pass those seconds with `DEMO_SKIP_RECORD=1` so the same take is used:

```bash
DEMO_SKIP_RECORD=1 ./render.sh claude grey 17.5 25.5 37.5 53 --t-start 8 --metric-distance 1.2
```

`--t-start` is the second at which the upload prompt is sent; the panel
stays idle ("no scan yet") until then, so the opening install lines are not
paired with a "reconstructing" label. The Claude and Codex prompts calibrate
with 1.2 m, so pass `--metric-distance 1.2`; the panel's default is the older
1.6 m.

### Use an actual splat

The offline simulator does not reconstruct the placeholder upload and serves
only a one-point PLY stub. The default side panel therefore uses a deterministic
fixture visualization and says so on-screen. Do not present it as a render of a
real reconstruction.

For a real scan, first ask the assistant to fetch the scan's `splat.ply` and
obtain the two world-space points used by `scene_measure_distance`. Then render
the same take with that PLY and those exact endpoints:

```bash
DEMO_SKIP_RECORD=1 ./render.sh claude grey 17.5 25.5 37.5 53 --t-start 8 \
  --splat /path/to/splat.ply \
  --scan-id scan-123 \
  --point-a 1.2 0.38 0.4 \
  --point-b 0.3 0.45 1.1 \
  --point-a-label desk \
  --point-b-label chair \
  --metric-distance 1.6
```

`render_scene_panel.py` samples the actual ASCII or binary little-endian PLY,
uses its RGB or spherical-harmonic colors, and draws the highlighted segment
between the supplied measurement points. It is a lightweight point-splat
preview, not a full anisotropic Gaussian renderer.

## How a take works

- `setup.sh` (sourced by every tape's hidden section) starts the simulator,
  creates the throwaway project folder, pre-warms the MCP package, and settles
  the assistant's one-time prompts so they never appear on camera:
  - Codex: removes a stale `openreality-demo` entry and marks the throwaway
    folder as trusted in `~/.codex/config.toml` (the same thing answering
    "Yes, continue" to its directory-trust question does). The take launches
    Codex with `-c model_reasoning_effort=medium` so turns take seconds
    rather than a minute.
  - Claude Code: marks the throwaway folder as trusted in `~/.claude.json`
    (the same thing answering "Yes, I trust this folder" does) and writes the
    project's `.claude/settings.json`, which pre-approves the server's tools
    plus `unzip` and `ls` (no permission prompt on camera) and sets a medium
    effort level (Opus answers in seconds instead of deliberating). The
    on-camera lines are the README's own: `claude mcp add --scope project
    openreality -- npx -y openreality-mcp serve` (project scope, so the
    printed path is the project's `.mcp.json` rather than a home folder) and
    `claude --model opus`. A shell function defined by `setup.sh` makes that
    launch add `--strict-mcp-config --mcp-config .mcp.json`, so the recording
    machine's other MCP servers (and their warnings) stay out of the take;
    the simulator URL and token reach the server through the environment.
  - Cursor: writes the project's `.cursor/mcp.json` (Cursor has no `mcp add`
    command; the take shows that file instead) and approves the server with
    `cursor-agent mcp enable`.
- The Codex and Cursor tapes own a demo-only MCP entry named
  `openreality-demo` and never touch a normal `openreality` entry; the Claude
  Code take's entry is the throwaway project's own `.mcp.json`. `render.sh` removes the demo entry,
  stops the simulator and deletes the throwaway folder when the take ends,
  even on failure.
- The Claude Code take waits on Claude Code's idle status line, so it either
  completes or fails loudly. The Codex take uses fixed pauses instead: Codex
  draws its transcript into the terminal's scrollback, and VHS can only read
  the first screenful of that, so text waits stop matching after the first
  turn. If a Codex turn outlasts its pause, the next prompt is queued and
  sent when the turn ends; re-render with longer pauses in `tapes/codex.tape`
  if that shows. Assistant wording varies between
  renders; the backend data does not. Re-render until you like the take.
- Take status (2026-09-02): the Claude Code take is validated on all three
  backgrounds. The Codex take records end to end, but with gpt-5.6 the
  model routed the measure and calibrate beats through the server-side
  scene agent instead of `scene_measure_distance` and `scene_anchor`, and
  then reported that it could not calibrate; review a Codex take before
  publishing it, and expect to tune its prompts. The Cursor tape has not
  been rendered yet (no logged-in Cursor CLI on the machine that wrote it);
  its beats mirror the Codex take and the first render will show whether
  Cursor's startup needs an extra keypress.
- On headless servers Chromium may need extra shared libraries
  (`ldd ~/.cache/rod/browser/*/chrome | grep "not found"`, fetch the debs, set
  `LD_LIBRARY_PATH`) and `VHS_NO_SANDBOX=true` where unprivileged user
  namespaces are restricted (Ubuntu 24.04 default).
