# MIRROR.md

This directory (`server/` of `reality-opened/openreality`) is a **curated
public mirror** of the internal `reality-opened/server` repo (private). Development happens there; reviewed
states are synced here. Issues and PRs are welcome and get folded back
upstream.

## Manifest

| Field | Value |
|---|---|
| Upstream repo | `reality-opened/server` (private) |
| Synced at commit | `14b1f43d936c7bf6b2e521be0f46947063ef4880` (main, 2026-09-02) |
| Included | `server/`, `tests/` (minus 3 files tied to internal deploy harnesses), `docs/` subset, `scripts/{demo_ingest_gemini.py,fetch_demo_videos.sh,ply_to_spz.mjs}`, `modal_selfhost.py`, `requirements.txt`, `LICENSE` |
| Excluded | the hosted Modal deploy + experiment harnesses (`modal_*.py` except `modal_selfhost.py`), internal scripts, the `web` SPA submodule, billing/deploy docs, internal agent docs |
| Local modifications | `requirements.txt` (core pin points at the public `openreality-core` mirror), two comment lines de-identifying pilot customers (`server/app.py`, `server/reconstruct_pilot.py`), this file, `README.md`, `.github/` |

Name mapping for docs and help strings written upstream:
`reality-opened/core` corresponds to the sibling [../core](../core);
`reality-opened/server` corresponds to this directory. References to the hosted
Modal deploy (`modal_streaming.py`, per-user GPU workers, `web/` SPA builds)
describe the hosted service, not this mirror; the self-host twin is
`modal_selfhost.py` + `server/selfhost.py`.

## Sync procedure (maintainers)

From a checkout of the private repo:

```bash
git archive origin/main server tests LICENSE requirements.txt modal_selfhost.py \
  docs/self-hosting.md docs/streaming-server.md docs/scene-report.md docs/spatial-agent.md \
  docs/dataset-export.md docs/isaac-export.md docs/access-control.md docs/testing.md \
  scripts/demo_ingest_gemini.py scripts/fetch_demo_videos.sh scripts/ply_to_spz.mjs \
  | tar -x -C <this-repo>/server
rm <this-repo>/server/tests/{test_oreos_modal_image.py,test_pilot_reconstruct.py,test_modal_entry_sources.py}
# re-apply the requirements pin + de-identification edits, update the manifest
# commit above, run the test suite.
```
