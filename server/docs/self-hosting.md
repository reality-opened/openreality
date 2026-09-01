# Self-hosting the Open Reality broker

Run the whole MCP workflow on infrastructure you control: upload a video, get a
reconstructed scene, measure it, run exports, fetch artifacts. Two supported
paths, one code path: both wire the same `server/` package through
`server/selfhost.py`; only the storage and job execution backends differ.

| | Path A: your own GPU box | Path B: your own Modal account |
|---|---|---|
| Process model | one process, jobs as worker threads | CPU broker + GPU job function |
| Storage | local disk (`--data-dir`) | Modal Volume + Dicts in your account |
| Entry point | `python -m server.selfhost` | `modal deploy modal_selfhost.py` |

## Licensing: read this first

The self-host backbone is the MIT-SPARK `VGGT_SPARK` fork (pinned commit) with
the `facebook/VGGT-1B` weights. Both are **CC BY-NC 4.0: non-commercial use
only**. The metric-anchor depth model (Depth-Anything-V2 Large) is also
CC BY-NC 4.0. This repo is BSD-2-Clause and never redistributes any of them;
they are fetched from upstream at build/run time under their own terms. If you
need commercial use, either use the hosted service at
[open-reality.io](https://open-reality.io) or obtain your own licenses from the
model owners. The optional detection stack adds
`facebookresearch/perception_models` (Apache-2.0 code) and
`facebookresearch/sam3` (SAM License: commercial use allowed, with
ITAR/acceptable-use restrictions you accept from Meta directly).

## Auth: `OPENREALITY_AUTH=local`

Self-host deployments run without Clerk. One static bearer
(`OPENREALITY_LOCAL_TOKEN`, at least 16 characters) bootstraps identity;
everything without it still gets 401. Broker session tokens and durable `ork_`
API keys work on top of it, so the standard MCP sign-in works unchanged:

```bash
OPENREALITY_URL=<your broker> npx -y openreality-mcp login --token <your local token>
# mints and stores a durable ork_ API key against YOUR broker
```

Single-user by construction: every authorized caller is the same account
(`sub=local`). Do not expose the broker publicly without understanding that.

## What works, what does not (v1)

Works: video upload + reconstruction, splat import (.ply/.spz, inline and
chunked), scene cards/measure/planes/nav/ground frame, metric anchor, LOD
builds, dataset exports (`openreality`, `groot_lerobot_v2`), artifact
downloads, share links, API keys, scene agents (bring an `OPENROUTER_API_KEY`),
SAM 3 segmentation routes (bring a `FAL_API_KEY`).

Not in v1, each refusing honestly instead of hanging: robot-recording ingest
(DimOS pipeline), the `isaac_usd` export lane (usd-core + Poisson meshing),
live phone-streaming sessions (run `python -m server.app` directly for a local
live scan; the hosted service schedules those onto per-user GPU workers).

## Path A: your own GPU box

Requirements: CUDA GPU with 24 GB or more VRAM recommended, 48 GB or more
host RAM for reconstruction, Python 3.11.

```bash
git clone https://github.com/reality-opened/openreality
cd openreality/server
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.3.1 torchvision==0.18.1
pip install -r requirements.txt          # or: pip install -e ../core for the SLAM library

# the NC backbone, cloned from upstream at the pinned commit:
git clone https://github.com/MIT-SPARK/VGGT_SPARK.git third_party/vggt
git -C third_party/vggt checkout 6e6e16107b88e8e76c751826af10d4295d87ecd2
pip install -e third_party/vggt

python -m server.selfhost --data-dir ~/openreality-data --port 8000
```

On first run it mints a local token, stores it at
`~/openreality-data/local_token` (0600), and prints the exact `openreality-mcp`
connect commands. VGGT-1B weights download on first reconstruction (or fetch
`https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt` into
`~/.cache/torch/hub/checkpoints/model.pt` yourself). Optional environment:
`OPENROUTER_API_KEY` (scene agents), `FAL_API_KEY` (SAM 3 routes),
`GEMINI_API_KEY` (depth-camera lanes). Data layout under `--data-dir`:
`records/` (scene metadata), `blobs/` (artifacts), `keys/` (hashed API keys),
`local_token`.

## Path B: your own Modal account

Costs land on your account; the broker is a scale-to-zero CPU container and
each job runs on `OPENREALITY_GPU` (default A10G).

```bash
git clone https://github.com/reality-opened/openreality
cd openreality/server
pip install modal && modal setup     # once, for your Modal account

modal secret create openreality-selfhost \
  OPENREALITY_LOCAL_TOKEN=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))') \
  OPENROUTER_API_KEY="" GEMINI_API_KEY="" FAL_API_KEY=""

modal deploy modal_selfhost.py       # prints the web URL
modal run modal_selfhost.py::download_models
```

Connect the MCP client with `OPENREALITY_URL=<web URL>` and
`OPENREALITY_TOKEN=<your OPENREALITY_LOCAL_TOKEN>`, or mint a durable key with
`openreality-mcp login --token <local token>`.

## Keeping the job glue in sync

`server/selfhost.py` mirrors the hosted job wrappers
(`modal_oreos_{ingest,lod,anchor,export}.py`). A change to one of those bodies
changes its twin in the same commit; `tests/test_selfhost.py` covers the auth
mode, the stores, and the spawner-to-jobs-store wire.
