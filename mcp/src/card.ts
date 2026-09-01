/**
 * Scene context card — the Tier-1 context object: a compact,
 * honesty-preserving markdown summary composed from the scene detail + the demo
 * manifest. This is what "a scene synced into Claude Code" looks like; heavy
 * artifacts stay behind tools.
 *
 * Honesty rules (mirrors the server's posture, never relaxed here):
 *   - lengths are metres ONLY under an anchor pointer; otherwise "relative units";
 *   - object labels are the closed-world inventory (model query + operator relabel);
 *   - degraded / QC states are stated, never smoothed over.
 */
import {
  OREOS_MANIFEST_KEY,
  OREOS_REST_PATHS,
  REST_PATHS,
  emptyOreosManifest,
  isOreosManifest,
  type OreosManifest,
  type PersistedScene,
} from '@reality/protocol';
import { ApiClient, ApiError } from './http.js';

export interface SceneCard {
  markdown: string;
  scene: PersistedScene;
  manifest: OreosManifest;
}

export async function fetchSceneCard(client: ApiClient, scanId: string): Promise<SceneCard> {
  const scene = await client.json<PersistedScene>('GET', REST_PATHS.SCENE_DETAIL(scanId));
  let manifest = emptyOreosManifest();
  try {
    const raw = await client.json<unknown>(
      'GET',
      REST_PATHS.SCENE_DERIVED(scanId, OREOS_MANIFEST_KEY),
    );
    if (isOreosManifest(raw)) manifest = raw;
  } catch (err) {
    if (!(err instanceof ApiError && err.status === 404)) throw err;
  }
  return { markdown: renderCard(scene, manifest), scene, manifest };
}

function metricLine(scene: PersistedScene): string {
  const dl = scene.derived_latest;
  if (dl && dl.kind === 'anchor') {
    const sf =
      typeof dl.scale_factor === 'number' ? ` (scale ${dl.scale_factor.toPrecision(4)} m/unit)` : '';
    return `**metric: anchored** — lengths are metres${sf}, via ${dl.source_key}`;
  }
  return '**metric: un-anchored** — all lengths are RELATIVE units, not metres (anchor the scene to unlock metres)';
}

export function renderCard(scene: PersistedScene, manifest: OreosManifest): string {
  const report = scene.report;
  const facts = scene.facts ?? report?.facts;
  const source = scene.source ?? 'recon_video';
  const created = scene.created_at ? new Date(scene.created_at * 1000).toISOString() : 'unknown';
  const lines: string[] = [];

  lines.push(`# Scene ${scene.scan_id}`);
  lines.push('');
  lines.push(
    `- source: ${source} · created: ${created} · points: ${scene.point_count ?? 'n/a'} · splat: ${scene.has_splat ? 'yes' : 'no'}`,
  );
  lines.push(`- ${metricLine(scene)}`);
  if (report) {
    lines.push(`- room: ${report.room_type || 'unknown'}${report.degraded ? ' · **report degraded**' : ''}`);
    if (report.summary) lines.push(`- summary: ${report.summary}`);
    if (report.coverage_note) lines.push(`- coverage: ${report.coverage_note}`);
  }

  const m = facts?.metrics;
  if (m) {
    const dims = Array.isArray(m.dimensions) ? m.dimensions.map((d) => d.toFixed(2)).join(' × ') : 'n/a';
    lines.push(
      `- extent: ${dims} · keyframes: ${m.num_keyframes ?? 0} · submaps: ${m.num_submaps ?? 0} · vertical axis ${m.vertical_axis_known ? 'known' : 'UNKNOWN'}`,
    );
    if (typeof m.room_height === 'number') {
      lines.push(`- ground frame: room height ${m.room_height.toFixed(2)} (${m.derivation ?? 'derived'})`);
    }
  }

  const objects = (facts?.objects ?? []).filter((o) => !o.dismissed);
  if (objects.length) {
    lines.push('');
    lines.push(`## Objects (closed world, ${objects.length})`);
    const top = objects.slice(0, 12);
    for (const o of top) {
      const label = o.human_label ? `${o.human_label} (model: ${o.query})` : o.query;
      const center = Array.isArray(o.center) ? o.center.map((c) => c.toFixed(2)).join(', ') : '?';
      lines.push(`- ${label} · conf ${o.confidence?.toFixed?.(2) ?? o.confidence} · center [${center}]`);
    }
    if (objects.length > top.length) lines.push(`- … ${objects.length - top.length} more`);
  }

  const runs = manifest.agent_runs ?? [];
  const counts = [
    runs.length ? `${runs.length} agent run(s)` : null,
    manifest.measurements?.length ? `${manifest.measurements.length} measurement(s)` : null,
    manifest.paths?.length ? `${manifest.paths.length} planned path(s)` : null,
    manifest.edits?.length ? `${manifest.edits.length} edit(s)` : null,
    manifest.variations?.length ? `${manifest.variations.length} variation(s)` : null,
  ].filter(Boolean);
  lines.push('');
  lines.push('## Workspace artifacts');
  lines.push(`- ${counts.length ? counts.join(' · ') : 'none yet'}`);
  if (runs.length) {
    const last = runs[runs.length - 1]!;
    lines.push(`- last agent run: ${last.run_id} (${last.kind}) → events at ${last.events_key}`);
  }
  if (scene.synthetic_views?.length) {
    lines.push(`- synthetic views: ${scene.synthetic_views.length} (renders of the splat, NOT photographs)`);
  }
  lines.push('');
  lines.push(
    `_Routes: measure/nav/agents via the scene tools; exports via export_prepare (formats: openreality, groot_lerobot_v2, isaac_usd). Manifest key: ${OREOS_MANIFEST_KEY}; agent events persist under derived/demo/agent_runs/. Paths built from ${Object.keys(OREOS_REST_PATHS).length} contract routes._`,
  );
  return lines.join('\n');
}
