# vendor/protocol — `@reality/protocol` (pinned copy)

Vendored source copy of the Open Reality cross-repo wire contract
(`@reality/protocol`): REST route builders, auth/token helpers, and the typed
payload shapes + guards the broker and every client speak. It is consumed as
TypeScript source (bundled into `dist/cli.js` at build time) and the simulator
is contract-tested against its guards.

**Do not edit these files here.** The source of truth lives in the
`reality-opened/web` workspace at `packages/protocol`; this copy exists so the
public repo is self-contained.

## Manifest

| Field | Value |
|---|---|
| Upstream repo | `reality-opened/web` (private) |
| Upstream path | `packages/protocol/*.ts` (implementation files; tests stay upstream) |
| Synced at commit | `55e4b4d` (web main, 2026-09-01) |
| Last upstream change to the package | share-access + event additions (rest.ts/events.ts/types.ts, 2026-08) |
| Local modifications | none |

## Sync procedure

```bash
# from a checkout of reality-opened/web
cp packages/protocol/{auth,binary,oreos,events,exportHub,objectLayer,rest,types,index}.ts \
   <this-repo>/vendor/protocol/
# then update the manifest commit above and rerun:
npm run typecheck && npm test
```
