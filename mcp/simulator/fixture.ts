/**
 * Seeded fixture data for the simulator: one anchored office scene with a prior
 * agent run, so the full query surface answers on first boot. Shapes are typed
 * against @reality/protocol — the sim cannot drift from the contract silently.
 */
import type {
  OreosManifest,
  PersistedScene,
  SceneListItem,
} from '@reality/protocol';

/** 1×1 PNG (red) — stands in for keyframes and synthetic-view renders. */
export const PNG_1PX = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==',
  'base64',
);

/** Tiny valid-ish ASCII PLY so splat/cloud fetches produce a recognizable file. */
export const PLY_STUB = Buffer.from(
  'ply\nformat ascii 1.0\ncomment openreality simulator stub\nelement vertex 1\n' +
    'property float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n',
  'utf8',
);

export const FIXTURE_SCAN_ID = 'sim-fixture-01';
export const FIXTURE_SCALE = 0.048; // metres per SLAM unit (anchored)
export const FIXTURE_ANCHOR_KEY = 'derived/anchor/20260810T0900';
export const FIXTURE_RUN_ID = 'run-fixture-001';
const CREATED_AT = 1_754_800_000; // 2026-08-10ish

export function fixtureScene(): PersistedScene {
  const objects = [
    {
      query: 'desk',
      center: [1.2, 0.38, 0.4],
      extent: [1.4, 0.75, 0.7],
      confidence: 0.92,
      evidence: [],
    },
    {
      query: 'office chair',
      center: [0.3, 0.45, 1.1],
      extent: [0.6, 0.95, 0.6],
      confidence: 0.81,
      evidence: [],
      human_label: 'desk chair',
    },
    {
      query: 'monitor',
      center: [1.25, 0.95, 0.35],
      extent: [0.62, 0.4, 0.08],
      confidence: 0.77,
      evidence: [],
    },
  ];
  const metrics = {
    dimensions: [4.2, 2.6, 3.1],
    bbox_min: [-1.9, -0.1, -1.4],
    bbox_max: [2.3, 2.5, 1.7],
    trajectory_length: 11.4,
    num_submaps: 3,
    num_keyframes: 24,
    point_count: 1_850_000,
    vertical_axis_known: true,
    up_axis: [0, 1, 0],
    floor_height: -0.02,
    ceiling_height: 2.4,
    room_height: 2.42,
    floor_extent: [4.1, 3.0],
    floor_area: 11.2,
    derivation: 'plane-fit',
  };
  return {
    scan_id: FIXTURE_SCAN_ID,
    created_at: CREATED_AT,
    report: {
      summary: 'Small office with a desk against the east wall, a chair, and a monitor.',
      room_type: 'office',
      objects: objects.map((o) => ({ name: o.query, location: 'office', evidence: [] })),
      observations: ['Desk surface is clear.', 'Chair partially occludes the desk from the door.'],
      coverage_note: 'Full loop of the room; ceiling sparsely covered.',
      facts: {
        metrics,
        objects,
        object_counts: { desk: 1, 'office chair': 1, monitor: 1 },
        relations: [],
        coverage_estimate: 0.82,
        scene_center: [0.2, 1.2, 0.15],
        units_note: 'lengths in metres via metric anchor',
      },
      model: 'simulator',
      degraded: false,
      generated_at: CREATED_AT,
    },
    facts: {
      metrics,
      objects,
      object_counts: { desk: 1, 'office chair': 1, monitor: 1 },
      relations: [],
      coverage_estimate: 0.82,
      scene_center: [0.2, 1.2, 0.15],
      units_note: 'lengths in metres via metric anchor',
    },
    keyframes: [
      { submap_id: 0, frame_idx: 4, blob_key: 'kf_0_4.png' },
      { submap_id: 1, frame_idx: 12, blob_key: 'kf_1_12.png' },
      { submap_id: 2, frame_idx: 3, blob_key: 'kf_2_3.png' },
    ],
    points_key: 'points.npz',
    point_count: 1_850_000,
    scene_center: [0.2, 1.2, 0.15],
    source: 'recon_video',
    has_splat: true,
    derived_latest: {
      kind: 'anchor',
      cloud_key: `${FIXTURE_ANCHOR_KEY}/cloud.npz`,
      trajectory_key: `${FIXTURE_ANCHOR_KEY}/trajectory.npz`,
      splat_key: `${FIXTURE_ANCHOR_KEY}/splat.ply`,
      source_key: `${FIXTURE_ANCHOR_KEY}/cloud.npz`,
      applied_at: '2026-08-10T09:00:00Z',
      scale_factor: FIXTURE_SCALE,
    },
  };
}

export function toListItem(scene: PersistedScene): SceneListItem {
  return {
    scan_id: scene.scan_id,
    created_at: scene.created_at,
    room_type: scene.report?.room_type ?? 'unknown',
    summary: scene.report?.summary ?? '',
    object_count: scene.facts?.objects?.length ?? 0,
    point_count: scene.point_count,
    thumbnail: scene.keyframes[0] ?? null,
    has_splat: scene.has_splat ?? false,
  };
}

export function fixtureManifest(): OreosManifest {
  return {
    version: 1,
    updated_at: CREATED_AT + 600,
    annotations_key: null,
    agent_runs: [
      {
        run_id: FIXTURE_RUN_ID,
        kind: 'annotate',
        created_at: CREATED_AT + 500,
        events_key: `derived/demo/agent_runs/${FIXTURE_RUN_ID}/events.json`,
      },
    ],
    edits: [],
    paths: [],
    variations: [],
    measurements: [
      { key: 'derived/demo/measurements/20260810T0910/measure.json', created_at: CREATED_AT + 550 },
    ],
  };
}

/** A fresh (un-anchored) scene persisted by a simulated video/recording ingest. */
export function ingestedScene(scanId: string, label: string | null, source: PersistedScene['source']): PersistedScene {
  const base = fixtureScene();
  const objects = base.facts.objects.slice(0, 2);
  const facts = {
    ...base.facts,
    objects,
    object_counts: { desk: 1, 'office chair': 1 },
    units_note: 'lengths are relative (no metric anchor yet)',
  };
  return {
    ...base,
    scan_id: scanId,
    created_at: Math.floor(Date.now() / 1000),
    source,
    derived_latest: null,
    facts,
    report: {
      ...base.report,
      summary: label
        ? `${label} — reconstructed by the simulator.`
        : 'Reconstructed by the simulator.',
      facts,
      degraded: source === 'robot_recording',
    },
  };
}
