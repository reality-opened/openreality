# openreality-mcp

**Open Reality OS as MCP tools.** Drive the whole OS from Claude Code, Claude
desktop, or any MCP client: upload scans into your account, follow
reconstruction jobs, sync scenes down as context, measure, plan paths, run the
scene agents, and pull robot-training exports (`openreality` /
`groot_lerobot_v2` / `isaac_usd`) to disk.

```bash
npm install -g openreality-mcp   # or run it ad hoc with npx
```

This is a **thin typed client of the deployed broker**: the same REST contract
the OS web client speaks (the vendored [`@reality/protocol`](vendor/protocol)
is the single source of truth for routes and payload shapes). No orchestration,
guardrails, or storage logic is duplicated here.

## What runs where

- **The MCP server always runs on your machine**: a stdio process started by
  your client. Credentials and fetched artifacts stay local.
- **Every tool call is a REST call to an Open Reality broker**
  (`OPENREALITY_URL`). Reconstruction, scene agents, storage, and exports
  execute there; real use needs an account (`openreality-mcp login`).
- **Fully offline development works without an account**:
  `openreality-mcp simulator` fakes the whole workflow with fixture data (see
  below). It does not reconstruct anything.
- The broker contract is plain REST, so `OPENREALITY_URL` can point at any
  deployment that speaks it, **including your own**: the broker
  ([`../server`](../server)) and the SLAM library ([`../core`](../core)) live
  in this repo and are self-hostable (your own GPU box, or your own Modal
  account). Self-hosted stacks are non-commercial by upstream model licensing;
  see [`../server/docs/self-hosting.md`](../server/docs/self-hosting.md).

## Install: Claude Code

```bash
claude mcp add openreality -- npx -y openreality-mcp serve
```

or in a project `.mcp.json` (see [`.mcp.json.example`](.mcp.json.example)):

```json
{
  "mcpServers": {
    "openreality": {
      "command": "npx",
      "args": ["-y", "openreality-mcp", "serve"]
    }
  }
}
```

Optionally install the skill so Claude knows the workflow + honesty doctrine:
copy `skills/openreality/` into `~/.claude/skills/` (or your project's
`.claude/skills/`).

## Install: Claude desktop

Add to `claude_desktop_config.json` (Settings → Developer → Edit Config):

```json
{
  "mcpServers": {
    "openreality": {
      "command": "npx",
      "args": ["-y", "openreality-mcp", "serve"]
    }
  }
}
```

## Install: Codex

```bash
codex mcp add openreality -- npx -y openreality-mcp serve
```

or in `~/.codex/config.toml` (the Codex IDE extension reads the same file;
verified against codex-cli 0.146.0):

```toml
[mcp_servers.openreality]
command = "npx"
args = ["-y", "openreality-mcp", "serve"]
```

Manage with `codex mcp list` / `codex mcp get openreality` / `codex mcp remove openreality`.

## Install: Cursor

One-click: [Add to Cursor](cursor://anysphere.cursor-deeplink/mcp/install?name=openreality&config=eyJjb21tYW5kIjoibnB4IiwiYXJncyI6WyIteSIsIm9wZW5yZWFsaXR5LW1jcCIsInNlcnZlIl19)
(the deeplink carries the base64 of the server config below).

Or add to `~/.cursor/mcp.json` (global) or `.cursor/mcp.json` (per project):

```json
{
  "mcpServers": {
    "openreality": {
      "command": "npx",
      "args": ["-y", "openreality-mcp", "serve"]
    }
  }
}
```

Sign-in is shared: `npx -y openreality-mcp login` once, and every client that starts
the server picks up the stored credentials.

## Sign in

The broker authenticates with a durable **API key** (`ork_...`, opaque,
revocable server-side, minted once via a browser login; no expiry, no
rolling). The older broker session-token flow (12 h, auto-rolled at half-life)
still works as a fallback for brokers that predate API keys.

```bash
openreality-mcp login
# opens your browser to sign in, then mints and stores a durable ork_ API key.
# Headless / no browser? The URL is always printed too; open it anywhere.
```

Credentials land in `~/.config/openreality/credentials.json` (0600).
Alternatively set `OPENREALITY_TOKEN` in the environment; this also accepts an
`ork_...` key directly. The broker URL defaults to the deployed app; override
with `OPENREALITY_URL`. The browser hand-off page defaults to
`https://open-reality.io`; override with `OPENREALITY_LOGIN_URL`.

Already have a token (an `ork_...` key, or a short-lived sign-in token from
your deployment's `/os` page `#token=…` hash)?

```bash
openreality-mcp login --token <token>
```

Manage keys with `openreality-mcp keys list` / `openreality-mcp keys revoke
<key_id>`; `openreality-mcp whoami` sanity-checks the stored credential. A
revoked or unknown key answers every request with a clear "run `login` again"
error; it never loops retrying.

## Architecture: an all-plugin [cordis](https://github.com/cordiverse/cordis) app

The server is assembled from cordis plugins around two services; nothing
registers anything globally.

```
src/services/broker.ts   ctx.broker: resolved config + credential session + typed HTTP client
src/services/mcp.ts      ctx.mcp:     the MCP server; effect-scoped tool/resource registration
src/plugins/workspace.ts   workspace_* tools     (inject: ['mcp', 'broker'])
src/plugins/scene.ts       scene_* tools
src/plugins/agent.ts       agent_* tools
src/plugins/export.ts      export_* / artifact_* tools
src/plugins/resources.ts   openreality:// resources
src/app.ts               the composition (createApp), and nothing else
```

The rules this buys, mechanically enforced by the framework:

- **Registrations are reversible effects.** Every tool/resource goes through
  `ctx.effect()` with the SDK handle's `remove()` as the disposer: disposing a
  namespace fiber removes exactly its tools from the live server, remounting
  restores them (proven in `test/cordis.test.ts`).
- **Load order is service requirements, not boot sequencing.** A tool namespace
  declares `inject: ['mcp', 'broker']` and stays pending until those services
  exist.
- **Misconfiguration fails loud at mount.** Service configs are validated
  (Standard Schema via zod) when the plugin loads, not on the first tool call.
- **Compositions are data.** `createApp()` mounts the full surface; embedders
  and tests mount any subset on a fresh `Context`.

## Develop against the simulator (no backend needed)

```bash
openreality-mcp simulator --port 8973
# then in another shell / your MCP config:
OPENREALITY_URL=http://127.0.0.1:8973 OPENREALITY_TOKEN=sim-token openreality-mcp serve
```

The simulator is a dependency-free mock broker whose responses are typechecked
against `@reality/protocol` and contract-tested with the protocol's own guards
(`test/simulator.contract.test.ts`). It simulates the honesty doctrine too:
un-anchored scenes measure in `relative` units, the Isaac export is
metric-gated, one agent run per scene (409), replays are $0. It also implements
the durable API-key routes against the same `ork_...` wire contract as the real
broker. [`examples/transcript.md`](examples/transcript.md) is a full recorded
session.

## Tool surface (41 tools)

| Namespace | Tools |
|---|---|
| `workspace_*` | list_scenes (sync cursor), upload_video, upload_recording, upload_splat (inline + chunked), job_status, job_wait |
| `scene_*` | card, list_objects, measure_distance, measure_angle, plan_path, planes, ground_frame, ground_frame_fit, anchor, keyframe_image, synthetic_views, synthetic_view_image, lod, lod_build, imported_objects, imported_objects_run, object_complete, object_variants, segment, job_status, job_wait, share, share_access (who opened a share link, visits, returns, questions) |
| `agent_*` | annotate, replay, pilot, chat (waits + renders the event transcript), run_events, runs |
| `export_*` / `artifact_*` | export_manifest, export_prepare, export_status, artifact_fetch, artifact_fetch_splat, artifact_fetch_cloud |

Resources: `openreality://scenes` (index) and `openreality://scene/{scan_id}`
(context card), @-mentionable in Claude Code.

Design rules: artifacts are written to disk
(`~/.config/openreality/artifacts/...`), never into context; honesty fields
(`units`, `scale_source`, `provenance`, `degraded`) pass through verbatim;
server refusals (409/422 with typed bodies) surface as structured tool errors.

## Build & test

```bash
npm install
npm run build        # tsup → dist/cli.js (vendored protocol bundled in)
npm run typecheck
npm test             # builds + runs unit, contract, cordis-lifecycle, and stdio e2e
                     # suites; regenerates examples/transcript.md against the simulator
```

## License

[BSD-2-Clause](LICENSE).
