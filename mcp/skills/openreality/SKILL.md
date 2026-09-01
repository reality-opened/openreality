---
name: openreality
description: Drive Open Reality OS through the openreality MCP tools — scan upload, reconstruction, scene sync/query, scene agents, measurements, navigation, and robot-training exports. Use whenever the user mentions their Open Reality account, scans/scenes, splats, reconstructions, scene measurements, or LeRobot/GR00T/Isaac exports of captured spaces.
---

# Open Reality OS

You have MCP tools (server name `openreality`) for a user's Open Reality account —
persisted 3D scenes reconstructed from phone captures, imported splats, and robot
recordings.

## Workflow

1. **Sync first**: `workspace_list_scenes` — the account is the sync layer; scans made on
   the user's phone or the web appear here. `new_since_last_sync` tells you what's new.
2. **Context second**: `scene_card <scan_id>` — small, honest summary (source, metric
   state, objects, artifacts). @-mention `openreality://scene/<scan_id>` works too.
3. **Then act**:
   - Deterministic questions (distances, angles, room structure, paths, exports) →
     `scene_*` / `export_*` tools. These are pure server routes — no LLM cost.
   - Narrated / visual questions ("describe this room", "key features") →
     `agent_chat` / `agent_annotate`. These run the SERVER-side scene agent, which spends
     a bounded LLM budget from the account. Prefer `agent_replay` (always $0) when a
     recorded run already answers the question (`agent_runs` lists them).
4. **Uploads**: `workspace_upload_video` (≤1 GB) / `workspace_upload_recording` (robot
   .db) / `workspace_upload_splat` (.ply/.spz) → then `workspace_job_wait` on the job_id.
5. **Artifacts go to disk, never into context**: `artifact_fetch` / `artifact_fetch_splat`
   return local file paths. Exports: `export_manifest` (dry-run) → `export_prepare` →
   `artifact_fetch` of the returned key. Formats: `openreality`, `groot_lerobot_v2`,
   `isaac_usd`.

## Honesty doctrine (non-negotiable — mirrors the product's own guardrails)

- **Units**: a length is metres ONLY when the response says `units: "m"` (a metric anchor
  was applied — `scale_source` names it). `units: "relative"` means relative SLAM units:
  never call them metres, never convert. To unlock metres: `scene_anchor` with two picked
  points and their real distance.
- **Closed world**: the objects from `scene_list_objects` are the only objects that exist.
  Don't invent, merge, or rename them beyond their labels.
- **Provenance rides along**: keyframes are photos; synthetic views are RENDERS of the
  splat (say so); generated assets (SAM-3D completions, variants, floor patches) are
  generated — their meta.json says `generated: true`. Nav plans carry "not certified
  navigation" — repeat it when presenting a path.
- **Degraded is a state, not a failure**: robot-recording scenes persist degraded (no
  detections, preview cloud) with a QC report under `derived/demo/recordings/` — report
  the QC verdict rather than hiding it.
- **The Isaac export is metric-gated**: without an anchor it is refused. That refusal is
  correct behavior — offer the anchor flow, don't fight the gate.
- Surface server refusals as-is: 409 `agent_run_active` (one run per scene — attach or
  wait), 422 `unreachable_goal` (includes the nearest reachable point), 409
  `export_job_active`.

## Troubleshooting

- `No Open Reality credentials` → run `openreality-mcp login --token <token>` (token from
  the deployment's `/os` page URL hash after dashboard sign-in) or set
  `OPENREALITY_TOKEN`.
- Tokens roll automatically at half-life; a stale (>12 h idle) credential needs a fresh
  login.
- To develop without the real backend: `openreality-mcp simulator`, then point the server
  at it with `OPENREALITY_URL=http://127.0.0.1:8973 OPENREALITY_TOKEN=sim-token`.
