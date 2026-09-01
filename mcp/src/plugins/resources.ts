/**
 * MCP resources — the "@-mention a scene" surface: the account's scene index and
 * per-scene context cards, addressable as openreality:// URIs so users can pin
 * them into context without a tool call.
 */
import { REST_PATHS, type SceneListItem } from '@reality/protocol';
import { ResourceTemplate } from '@modelcontextprotocol/sdk/server/mcp.js';
import type { Context } from 'cordis';
import { fetchSceneCard } from '../card.js';

/** Cordis plugin name used in loader/fiber diagnostics. */
export const name = 'scene-resources';

/** Services required before this plugin activates. */
export const inject = ['mcp', 'broker'];

export function apply(ctx: Context): void {
  const { mcp } = ctx;
  const { client } = ctx.broker;
  mcp.resource(
    ctx,
    'scenes',
    'openreality://scenes',
    {
      title: 'Open Reality scenes',
      description: 'Index of the signed-in account\'s persisted scenes (newest first).',
      mimeType: 'text/markdown',
    },
    async (uri) => {
      const data = await client.json<{ scenes?: SceneListItem[] }>('GET', REST_PATHS.SCENES);
      const scenes = Array.isArray(data.scenes) ? data.scenes : [];
      const lines = [
        '# Open Reality scenes',
        '',
        ...scenes.map(
          (s) =>
            `- \`${s.scan_id}\` — ${s.room_type || 'unknown room'} · ${s.object_count} objects · ` +
            `${new Date(s.created_at * 1000).toISOString()} → openreality://scene/${s.scan_id}`,
        ),
      ];
      return {
        contents: [{ uri: uri.href, mimeType: 'text/markdown', text: lines.join('\n') }],
      };
    },
  );

  mcp.resource(
    ctx,
    'scene',
    new ResourceTemplate('openreality://scene/{scan_id}', {
      list: undefined,
      complete: {
        scan_id: async () => {
          try {
            const data = await client.json<{ scenes?: SceneListItem[] }>('GET', REST_PATHS.SCENES);
            return (data.scenes ?? []).map((s) => s.scan_id);
          } catch {
            return [];
          }
        },
      },
    }),
    {
      title: 'Scene context card',
      description: 'Tier-1 context card for one scene (same content as the scene_card tool).',
      mimeType: 'text/markdown',
    },
    async (uri, variables) => {
      const scanId = String(variables.scan_id);
      const { markdown } = await fetchSceneCard(client, scanId);
      return { contents: [{ uri: uri.href, mimeType: 'text/markdown', text: markdown }] };
    },
  );
}
