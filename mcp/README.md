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
claude mcp add --scope user openreality -- npx -y openreality-mcp serve
```

`--scope user` registers the server for every project on your machine. Drop
the flag to register it for the current folder only (Claude Code's default).

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

Or install both at once as a **Claude Code plugin** (this repo is a plugin
marketplace: [`.claude-plugin/marketplace.json`](../.claude-plugin/marketplace.json)
at the repo root, plugin manifest in [`.claude-plugin/plugin.json`](.claude-plugin/plugin.json)):

```
/plugin marketplace add reality-opened/openreality
/plugin install openreality@openreality
```

## Install: Claude desktop

**One-click extension (no terminal):** download `openreality-mcp-<version>.mcpb`
from the [latest release](https://github.com/reality-opened/openreality/releases/latest)
and double-click it, or drag it into the Claude desktop window. The extension's
settings panel takes an optional server URL (for self-hosters) and an optional
API token. The bundle is built from this package by `npm run build:mcpb`
([`scripts/build-mcpb.sh`](scripts/build-mcpb.sh), manifest in
[`manifest.json`](manifest.json)).

**Or by hand:** add to `claude_desktop_config.json` (Settings → Developer → Edit Config):

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

## Privacy policy

This section is the privacy policy for the `openreality-mcp` package and the
Claude desktop extension built from it. It is written so that a person with no
technical background can read it.

**What runs where.** The MCP server is a small program that runs on your own
computer, started by your AI assistant. It does not run on our servers. It has
no analytics, no telemetry, and no crash reporting: nothing is sent anywhere
unless a tool call needs it.

**Data collection.** The program sends data only to the Open Reality server it
is configured to talk to: the hosted service at `open-reality.io` (operated by
reality-opened) by default, or your own self-hosted server if you set
`OPENREALITY_URL`. What it sends is exactly the content of the tool calls your
assistant makes: video files, splat files, or robot recordings you ask it to
upload; scene identifiers; measurement points; questions for the scene agent.
It does not read files you did not ask it to upload.

**Usage and storage on your computer.** Your sign-in credential is stored in
`~/.config/openreality/credentials.json` (readable only by your user account).
Downloaded artifacts (point clouds, splats, export bundles) are written under
`~/.config/openreality/artifacts/`. A small sync cursor (which scene ids this
machine has already listed) is kept next to them. Delete these folders to
remove everything the program stored.

**Usage and storage on the hosted service.** Uploads become scenes in your Open
Reality account and stay there so your assistant can query them later. The
scene agent's conversations are recorded as event logs so they can be replayed
without cost. Data is retained until you delete the scene or the account.

**Third-party sharing.** The hosted service uses infrastructure providers to
process your data (GPU compute for reconstruction, model providers for the
scene agent and segmentation). They process data on our behalf under their
terms and do not receive it for their own use. We do not sell your data and do
not share it with advertisers. Share links you mint are read-only and single
scene; the `scene_share_access` tool lets you see who used them.

**Data retention.** Local files stay until you delete them. Hosted scenes stay
until you delete them or ask us to. Revoke a sign-in key at any time with
`openreality-mcp keys revoke <key_id>`.

**Contact.** Open an issue at
[github.com/reality-opened/openreality/issues](https://github.com/reality-opened/openreality/issues)
or use the contact address published at [open-reality.io](https://open-reality.io).

## License

[BSD-2-Clause](LICENSE).
