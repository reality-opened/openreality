# CLAUDE.md - openreality (public monorepo)

The public release of the Open Reality stack (`reality-opened/openreality`),
three components in one repo:

- `mcp/`: the MCP server (npm `openreality-mcp`). Developed HERE directly; an
  all-plugin cordis app. Read `mcp/CLAUDE.md` before changing it.
- `server/`: curated public mirror of the private `reality-opened/server` repo.
  Do not develop here; sync manifest + procedure in `server/MIRROR.md`.
- `core/`: curated public mirror of the private `reality-opened/core` repo.
  Same rule; see `core/MIRROR.md`.
Rules that span the repo:

- Documentation uses no em dashes.
- Documentation should avoid non-obvious jargon.
- Documentation should be understandable to people with no prior experience in the field. 
- The self-host licensing posture (CC BY-NC backbone, fetched never vendored)
  is stated in the root README and `server/docs/self-hosting.md`; keep both in
  sync and never soften it.
- CI (`.github/workflows/ci.yml`) runs all three component suites on every
  push; keep each job equivalent to its component's standalone check.
- `README.md` and `README.zh.md` are translations of each other; change both in
  the same commit. The hero banner is `docs/assets/hero.svg` (brand palette:
  deep green #0b3d2c, gold #d4a017, coral #e07070).

Commands: `cd mcp && npm test`, `cd server && python -m pytest tests/`,
`cd core && python -m compileall vggt_slam`.
