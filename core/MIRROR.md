# MIRROR.md

This directory (`core/` of `reality-opened/openreality`) is a **curated
public mirror** of the internal `reality-opened/core` repo (private). Development happens there; reviewed
states are synced here. Issues and PRs are welcome and get folded back
upstream.

## Manifest

| Field | Value |
|---|---|
| Upstream repo | `reality-opened/core` (private) |
| Synced at commit | `32e3e5c981853005a2db87991c7661d59d56ae43` (main, 2026-09-02) |
| Included | `vggt_slam/`, `main.py`, `requirements.txt`, `setup.py`, `LICENSE`, `docs/` (minus d4rt), `tests/` (minus d4rt), `visualize_results.py` |
| Excluded | the commercially licensed VGGT-Omega backbone shim, internal eval harnesses (`evals/`, `modal_*.py`), shelved research (`d4rt_temporal/`, `Open-d4rt`), internal agent docs |
| Local modifications | `setup.py` (version `0.1.0`; packages trimmed to `vggt_slam`), this file, `README.md`, `.github/` |

Docs and help strings that mention `reality-opened/server` correspond to the
sibling [../server](../server) mirror.

## Sync procedure (maintainers)

From a checkout of the private repo:

```bash
git archive origin/main vggt_slam main.py requirements.txt setup.py LICENSE docs tests visualize_results.py \
  | tar -x -C <this-repo>/core
rm <this-repo>/core/tests/test_d4rt_phase0.py <this-repo>/docs/d4rt-temporal.md
# re-apply the setup.py curation, update the manifest commit above, run CI.
```
