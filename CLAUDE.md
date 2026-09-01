# CLAUDE.md - openreality-mcp

Public repo (`reality-opened/openreality-mcp`, npm `openreality-mcp`): Open
Reality OS as MCP tools. An **all-plugin [cordis](https://github.com/cordiverse/cordis)
app**. Read the Architecture section of [README.md](README.md) before changing
`src/`.

## Rules

- **Registrations are effects.** Every tool/resource contribution goes through
  `ctx.mcp.tool(ctx, ...)` / `ctx.mcp.resource(ctx, ...)`, which install via
  `ctx.effect()`; disposal must unregister (proven by `test/cordis.test.ts`).
- **Plugins, not assembly changes.** New behavior is a new plugin (named
  exports `name` / `inject` / `apply`, no default export) or a new tool in an
  existing namespace; `src/app.ts` stays pure composition. Service classes
  (`src/services/`) default-export the class.
- **The wire contract is vendored, not owned.** `vendor/protocol/` is a pinned
  copy of the private `reality-opened/web` `packages/protocol`. Never edit it
  here; follow the sync procedure in [vendor/protocol/README.md](vendor/protocol/README.md).
- **Honesty doctrine passes through verbatim**: `units`, `scale_source`,
  `provenance`, `degraded`, and typed server refusals (409/422 bodies) reach
  the model unedited. Never soften or convert them.
- **Artifacts go to disk, never into context**: fetch tools return paths.
- **Persisted contracts keep historical names**: API routes, `derived/demo/*`
  keys, and `DEMO_*` env fallbacks spell `demo` on purpose; never rename.

## Commands

```bash
npm install
npm run build       # tsup → dist/cli.js (vendored protocol bundled in)
npm run typecheck
npm test            # build + unit/contract/lifecycle/stdio-e2e; regenerates examples/transcript.md
npm run simulator   # mock broker on :8973 for offline development
```
