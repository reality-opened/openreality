#!/usr/bin/env bash
set -euo pipefail

# The four times are where the Codex take reaches each visual beat. Defaults
# match the current simulator script; pass adjusted values after reviewing a
# new take. Any remaining arguments go to render_scene_panel.py, including an
# actual splat and its measured endpoints.
if [ "$#" -eq 0 ]; then
  T_CLOUD=20
  T_MEASURE=30
  T_METRIC=50
  T_EXPORT=58
elif [ "$#" -ge 4 ]; then
  T_CLOUD="$1"
  T_MEASURE="$2"
  T_METRIC="$3"
  T_EXPORT="$4"
  shift 4
else
  echo "usage: ./render.sh [t_cloud t_measure t_metric t_export [panel options...]]"
  exit 2
fi

for tool in vhs ttyd ffmpeg ffprobe codex; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "missing required command: $tool"
    exit 1
  fi
done

cleanup() {
  codex mcp remove openreality-demo >/dev/null 2>&1 || true
  if [ -f .demo-workspace/sim.pid ]; then
    DEMO_PID=$(sed -n '1p' .demo-workspace/sim.pid)
    case "$DEMO_PID" in
      ''|*[!0-9]*) ;;
      *) kill "$DEMO_PID" >/dev/null 2>&1 || true ;;
    esac
  fi
}
trap cleanup EXIT INT TERM

vhs demo.tape
TERMINAL_DURATION=$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 terminal.mp4)

if python3 -c 'import numpy, matplotlib' >/dev/null 2>&1; then
  RENDER_PYTHON=(python3)
elif command -v uv >/dev/null 2>&1; then
  RENDER_PYTHON=(uv --cache-dir /tmp/openreality-demo-uv-cache run --isolated --with "numpy<2" --with "matplotlib>=3.8" python)
else
  echo 'rendering requires compatible NumPy + Matplotlib; see docs/demo/README.md'
  exit 1
fi

"${RENDER_PYTHON[@]}" render_scene_panel.py \
  --duration "$TERMINAL_DURATION" \
  --t-cloud "$T_CLOUD" \
  --t-measure "$T_MEASURE" \
  --t-metric "$T_METRIC" \
  --t-export "$T_EXPORT" \
  --terminal terminal.mp4 \
  --out panel.mp4 \
  --composite demo.mp4 \
  "$@"

ffmpeg -y -v error -i demo.mp4 \
  -vf "fps=12,scale=1600:-2:flags=lanczos" \
  -loop 0 demo.gif

echo "wrote demo.mp4 and demo.gif"
