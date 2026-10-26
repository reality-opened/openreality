# openreality-mcp: architecture

_Writers: TBD, TBD · Last reviewed: TBD_

> **Writers:** replace every _italic prompt_ below. Explain components and the methods they use, not individual files. Link to code with `path:Symbol` instead of pasting it. Aim for a doc someone can read in 20 minutes.
>
> **This repo is public.** Write this doc so it can be read publicly. Report design concerns as GitHub issues and keep the internal discussion there. Leave out anything about customers or unreleased plans.
>
> **Repo doc rules** (from [`../../CLAUDE.md`](../../CLAUDE.md)): no em dashes; explain jargon; write for someone new to the field.
>
> **Existing docs to start from:** the "Architecture: an all-plugin cordis app" section of [`../README.md`](../README.md) (cordis is the plugin framework the server is built on), the rules in [`../CLAUDE.md`](../CLAUDE.md), and [`../vendor/protocol/README.md`](../vendor/protocol/README.md) (the vendored wire contract). Expand on the README section here; don't repeat it.

## 1. What this package does

_One paragraph. An MCP server published on npm as `openreality-mcp`. It gives a coding agent tools to upload video, query a 3D scene and export. It talks to the Open Reality server API. Name the clients it supports (Claude Code, Claude Desktop extension, Codex, Cursor) and how each one installs it._

## 2. Key components

_For each: **responsibility** · **entry points** · **where it lives** · **key method**. The list below comes from the directory layout. Merge, split or drop entries as needed._

- **CLI and server start-up**: `src/cli.ts`, `src/index.ts`, `src/app.ts`
- **Config, login and session**: `src/config.ts`, `src/login.ts`, `src/session.ts`
- **HTTP client to the server**: `src/http.ts`, `src/services/broker.ts`
- **Plugin composition (cordis)**: `src/app.ts`. _Explain how plugins, `inject` and `ctx.effect()` work, and why disposing a plugin unregisters its tools._
- **Tool registration**: `src/toolkit.ts`, `src/services/mcp.ts`
- **Vendored wire contract**: `vendor/protocol/`, a pinned copy of `web/packages/protocol`. _Explain the sync procedure._
- **Tool groups (plugins)**: `src/plugins/{workspace,scene,agent,export,resources}.ts`
- **Hardware bridge**: `src/hardware.ts` (the `hardware.json` device list), `src/plugins/hardware.ts` (one plugin fiber per device: starts the device's MCP server and forwards its tools as `<device>_*`), `src/jsonschema.ts` (device argument schemas to zod)
- **Scene card**: `src/card.ts`
- **Skills shipped with the package**: `skills/`
- **Simulator**: `simulator/`
- **Packaging and registries**: `manifest.json`, `server.json`, `dist-mcpb/`, `scripts/`

## 3. Data flow

_Trace one tool call: the agent calls `scene_measure_distance` → tool handler → HTTP to the server → response → what the agent sees. Include how auth is attached and how long jobs (upload, then `*_job_wait`) get polled._

## 4. How to make common changes

_Short recipes. Suggested starters:_
- _Add a new tool: which plugin it goes in, how its schema is defined, and what tests to add._
- _Expose a new server route through the MCP._
- _Cut a release: build, publish to npm, update the registries._
- _Test against a local server or the simulator._

## 5. Gotchas and design concerns

_Gotchas: traps a new engineer will hit, such as the rules on honest units and provenance in tool output._

_Design concerns: one line each, linked to a GitHub issue labelled `design-concern`._
