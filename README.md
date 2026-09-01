# Open Reality

Turn phone video into an AI-queryable 3D scene: reconstruct, measure, plan
paths, run scene agents, and export robot-training data. This repo is the
open-source release of the stack behind [open-reality.io](https://open-reality.io),
in three parts:

| Directory | What it is | Ships as |
|---|---|---|
| [`mcp/`](mcp/) | Open Reality as MCP tools for Claude Code, Claude desktop, Codex, and Cursor: 41 tools + scene resources + an offline simulator | npm [`openreality-mcp`](https://www.npmjs.com/package/openreality-mcp) |
| [`server/`](server/) | The broker: videos in, persisted scenes + measurements + agents + exports out. **Self-hostable** on your own GPU or your own Modal account | source (public mirror) |
| [`core/`](core/) | The SLAM library (VGGT-SLAM 2.0 line): dense feed-forward monocular SLAM, metric anchoring, open-set detection, splat export | source (public mirror) |

## Use the hosted service (fastest)

```bash
claude mcp add openreality -- npx -y openreality-mcp serve
npx -y openreality-mcp login
```

Per-client setup (Claude desktop, Codex, Cursor):
[open-reality.io/mcp](https://open-reality.io/mcp) or [`mcp/README.md`](mcp/README.md).

## Self-host the whole workflow

Read [`server/docs/self-hosting.md`](server/docs/self-hosting.md). Two paths,
one code path:

```bash
# your own GPU box: one process, local disk
cd server && python -m server.selfhost --data-dir ~/openreality-data

# or your own Modal account: CPU broker + GPU job worker
cd server && modal deploy modal_selfhost.py
```

Self-hosted brokers run without accounts (`OPENREALITY_AUTH=local`) and the MCP
client connects to them with just `OPENREALITY_URL` + your token.

**Licensing:** this repo is BSD-2-Clause. The self-host SLAM backbone
(MIT-SPARK `VGGT_SPARK` code and the `facebook/VGGT-1B` weights) and the
metric-anchor depth model are **CC BY-NC 4.0: non-commercial use only**,
fetched from upstream at build/run time and never redistributed here. The
hosted service runs a commercially licensed backbone; for commercial self-use,
obtain your own model licenses. Details in
[`server/docs/self-hosting.md`](server/docs/self-hosting.md).

## Development

```bash
# mcp: build + full test suite (unit, contract, cordis lifecycle, stdio e2e)
cd mcp && npm install && npm test

# server: GPU-free suite
cd server && pip install -r requirements.txt && python -m pytest tests/

# core: standalone SLAM over a video
cd core && python main.py --help
```

`core/` and `server/` are curated public mirrors of the internal working repos;
each carries a `MIRROR.md` with the sync manifest. `mcp/` is developed here
directly. Issues and PRs on any component are welcome.
