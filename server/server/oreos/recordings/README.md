# Oreos — camera-only scene pipeline for robot recordings

Take a DimensionalOS robot recording end-to-end with one command: extract its RGB and
odometry, reconstruct the scene with camera-only VGGT-SLAM on an A100, score the result
against the robot's own odometry, pass it through a measured QC gate, and get a
one-page `report.html` that tells you exactly what happened — then an export bundle
*only if the gate allows it*.

Provenance: EXP-36 (`platform/experiments/exp36_oreos_dimos_spike/` — 32 scored runs on
Dimensional's public recordings) calibrated the QC gate and paid for every hardening in
the recon worker. Strategy: `platform/experiments/research/2026-07-24-oreos-on-dimos-plan.md`.

## Quickstart (the walkthrough)

```bash
cd server   # repo root; needs: pip deps incl. modal (authed), numpy, opencv, matplotlib

# 1. A recording that PASSES the gate (small office scan, ~8 min wall total):
python -m server.oreos.recordings run /path/to/go2_short.db --out sessions/go2_short --with-submaps

# 2. A recording the gate REFUSES (long outdoor tour — scale collapse):
python -m server.oreos.recordings run /path/to/hk_village3.db --out sessions/hk_village3

# 3. Open the story:
open sessions/go2_short/report.html      # green banner, map, trajectory overlay
open sessions/hk_village3/report.html    # red banner + the measured reasons
cat  sessions/hk_village3/EXPORT_REFUSED.md
```

Recordings: any DimOS memory2 `.db` (their public LFS corpus works — see EXP-36's
fetch tooling). Both era formats supported (pose-anchor columns and LCM-payload
odometry are handled automatically; `reference_usable` in the manifest tells you
which recordings can be consistency-scored at all).

## What each file in a session dir means

| File | Meaning |
|---|---|
| `ingest_manifest.json` | what the recording contained (streams, frames, reference usability) |
| `results/est_tum.txt` | our camera-only trajectory (TUM format) |
| `results/recon_summary.json` | solver stats: submaps, loop closures, sign repairs, errors, GPU time, **`core_source`** (which SLAM library the image ran) |
| `results/map_preview.ply` | the dense cloud (voxel-downsampled) |
| `results/submaps/*.npz` | per-submap bundles (feed the DimOS replay node) |
| `consistency.json` | vs the robot's own odometry — **NOT ground truth**; Sim(3)-aligned |
| `confidence.json` | the QC verdict gating everything downstream |
| `report.html` | the whole story on one page — start here |
| `bundle/` or `EXPORT_REFUSED.md` | gated output, or the written reasons it was withheld |
| `pipeline_log.jsonl` | stage-by-stage record (rendered in the report) |

## Honesty model (read this once)

- `high_confidence` means **no known failure signature detected** — not "verified
  accurate." The v1 signals catch the collapse class measured in EXP-36; moderate
  uniform drift passes them (pinned limitation, `server/qc/confidence.py`).
- The consistency numbers compare against the robot's own drifting odometry, Sim(3)-
  aligned. Metres are shown with % of extent and % of path so scale isn't hidden.
- Camera-only reconstruction is **scan-class** technology today: short/small-area
  sessions work; long loopy tours drift 14–38 % of extent (EXP-36, config-robust).
  The gate exists precisely because of that boundary.

## Inside DimOS's runtime (the replay-node demo)

The batch pipeline needs no dimos install. To watch the results play back inside
DimensionalOS's own module runtime (typed ports, LCM, rerun), use the EXP-36 node:
`platform/experiments/exp36_oreos_dimos_spike/scripts/oreos_replay_blueprint.py`
(recipe in that experiment's RUNBOOK; needs `pip install dimos` + loopback multicast).
Its productization is Phase 0/2 of the adoption plan.

## Which core does a run use? (`OREOS_CORE_SOURCE`)

All three GPU workers (`modal_recon.py`, `live_probe/modal_live.py`,
`live_node/modal_stream.py`) build their image from one recipe,
`modal_image.py`:

| `OREOS_CORE_SOURCE` | what ships | when |
|---|---|---|
| `tag` (default) | `openreality-core @ git+…/core@v2.2.2` — reproducible, needs the `github-token` Modal secret | deploys, anything whose numbers get quoted |
| `local` | your sibling `../core` checkout, mounted + editable-installed (the old, only behaviour) | dev loop, and **required today** — see below |

⚠️ **core `v2.2.0` is a LOCAL tag** (cut 2026-07-24, not yet on origin). Until the
founder pushes core with `--follow-tags`, deploy with `OREOS_CORE_SOURCE=local`;
`tag` mode fails at image build and the build log says exactly this.

Either way the choice is stamped into the image env and echoed into the run
artifacts — `core_source` in `results/recon_summary.json` (and the report's
Reconstruction table), in the live node's `hello`/`/health`, and in the probe's
results file. A dirty local checkout is rendered as a red warning: a number whose
SLAM library can't be named isn't attributable.

## Architecture

`pipeline.py` (stage orchestrator, resumable, logged) · `modal_image.py` (shared image
recipe + core provenance) · `modal_recon.py` (A100 worker, app `oreos-recon`, volume
`oreos-sessions`; EXP-36-hardened) · `consistency.py` (numpy scorer) · `server/qc`
(the gate) · `report.py` (the dashboard) · `server/ingest` (recording reader,
dimos-free).
