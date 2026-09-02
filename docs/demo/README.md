# CLI demo recording

`demo.tape` renders the README's terminal demo with
[VHS](https://github.com/charmbracelet/vhs): a real Codex session driving
the openreality MCP tools against the **offline simulator** (deterministic
backend data, no account, no GPU). The finished video puts Codex on the left
and the measured scene on the right. The beats:

1. show the Codex MCP installation command, then upload a video
2. measure desk to chair: the result is honestly labeled **relative units**
3. give one real-world distance, calibrate, measure again: now **metres**
4. export robot-training data and list the zip contents on disk

## Re-render

```bash
# macOS prerequisites; also requires a logged-in `codex` CLI
brew install vhs ttyd ffmpeg
python3 -m venv /tmp/openreality-demo-render-env
source /tmp/openreality-demo-render-env/bin/activate
python -m pip install "numpy<2" "matplotlib>=3.8"

# optional: use this checkout for the simulator instead of the npm fallback
cd mcp
npm install
npm run build
cd ..

cd docs/demo
./render.sh           # writes terminal.mp4, panel.mp4, demo.mp4, and demo.gif
```

Codex response timing varies. If a take reaches the four scene beats at
different times, pass the observed seconds explicitly:

```bash
./render.sh 20 30 50 58
```

## Use an actual splat

The offline simulator does not reconstruct the placeholder upload and serves
only a one-point PLY stub. The default side panel therefore uses a deterministic
fixture visualization and says so on-screen. Do not present it as a render of a
real reconstruction.

For a real scan, first ask Codex to fetch the scan's `splat.ply` and obtain the
two world-space points used by `scene_measure_distance`. Then render the same
take with that PLY and those exact endpoints:

```bash
./render.sh 20 30 50 58 \
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

Notes:

- `setup.sh` (sourced by the tape's hidden section) creates a throwaway Codex
  workspace in `docs/demo/.demo-workspace`, starts the simulator on port 8973,
  and pre-warms the MCP package. The visible install command creates the
  temporary `openreality-demo` Codex MCP entry; the tape and `render.sh` remove
  only that entry afterward, leaving a normal `openreality` entry untouched.
- On headless servers Chromium may need extra shared libraries
  (`ldd ~/.cache/rod/browser/*/chrome | grep "not found"`, fetch the debs, set
  `LD_LIBRARY_PATH`) and `VHS_NO_SANDBOX=true` where unprivileged user
  namespaces are restricted (Ubuntu 24.04 default).
- Codex's wording varies between renders; the backend data does not. Re-render
  until you like the take. Waits key on stable strings (scan id, "relative",
  tool names, zip entries), so a take either completes or fails loudly.
- Publish the rendered GIF as the root README's release asset after reviewing
  that the four side-panel transitions align with the Codex responses.
