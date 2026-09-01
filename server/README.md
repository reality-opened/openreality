# Open Reality server

The Open Reality broker: a Flask + python-socketio ASGI server that turns
uploaded videos into persisted 3D scenes and serves the whole workspace API
that [openreality-mcp](../mcp) speaks. Scene cards, measurement, plane detection, path planning, metric
anchoring, LOD builds, scene agents, and robot-training exports
(`openreality`, `groot_lerobot_v2`). BSD-2-Clause.

This is the `server/` component of the public `reality-opened/openreality`
monorepo, a curated mirror of the internal server repo, published so you can
**self-host the whole workflow**:

## Self-host it

Read **[docs/self-hosting.md](docs/self-hosting.md)**. Two paths, one code path:

```bash
# your own GPU box: one process, local disk
python -m server.selfhost --data-dir ~/openreality-data --port 8000

# or your own Modal account: CPU broker + GPU job worker
modal deploy modal_selfhost.py
```

Both run without Clerk (`OPENREALITY_AUTH=local`, a static bearer bootstraps
identity; `openreality-mcp login` then mints durable API keys as usual) and
without the web SPA (headless broker; the MCP client is the interface).

**Licensing:** the SLAM backbone (MIT-SPARK VGGT_SPARK code + facebook/VGGT-1B
weights) and the metric-anchor depth model are **CC BY-NC 4.0: non-commercial
use only**, fetched from upstream at build/run time and never redistributed
here. For commercial use, use the hosted service at
[open-reality.io](https://open-reality.io) or obtain your own model licenses.
Details in [docs/self-hosting.md](docs/self-hosting.md).

## What's in the box

- `server/app.py` and friends: the broker (REST + socket contract, auth,
  session/share tokens, API keys)
- `server/oreos/`: the workspace blueprint (ingest, jobs, measure, nav,
  planes, agents, LOD, exports)
- `server/scene_report/`, `server/export/`, `server/agent/`, `server/llm/`
- `server/selfhost.py` + `modal_selfhost.py`: the self-host compositions
- `tests/`: the GPU-free suite (CI runs it on every push)

Docs: [streaming-server](docs/streaming-server.md),
[scene-report](docs/scene-report.md), [spatial-agent](docs/spatial-agent.md),
[dataset-export](docs/dataset-export.md), [isaac-export](docs/isaac-export.md),
[access-control](docs/access-control.md), [testing](docs/testing.md).
Some docs reference internal repo names; see [MIRROR.md](MIRROR.md) for the
mapping and the sync manifest.
