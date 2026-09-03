/**
 * scene.* — the deterministic OS feature routes as tools. Every handler is a thin mirror of one contract route; honesty
 * fields (units / scale_source / provenance / degraded) pass through verbatim.
 */
import {
  OREOS_REST_PATHS,
  REST_PATHS,
  type GroundFrameResponse,
  type ImportedObjectsStatus,
  type MeasureResponse,
  type NavPlanRequest,
  type NavPlanResponse,
  type OreosLodResponse,
  type PersistedScene,
  type PlanesResponse,
  type ShareAccessResponse,
  type ShareSceneResponse,
} from '@reality/protocol';
import { z } from 'zod';
import type { Context } from 'cordis';
import { fetchSceneCard } from '../card.js';
import { guard, imageResult, jsonResult, pollSceneJob, progressTick, textResult, vec3 } from '../toolkit.js';

/** Inline waypoint budget for a planned path; the rest stays in the path doc. */
const DEFAULT_WAYPOINT_PREVIEW = 24;

/** Evenly sample `rows` down to `max` entries, always keeping the first and last. */
function sampleEvenly<T>(rows: readonly T[], max: number): T[] {
  if (rows.length <= max) return [...rows];
  const step = (rows.length - 1) / (max - 1);
  return Array.from({ length: max }, (_, i) => rows[Math.round(i * step)]!);
}

/**
 * A planned path is a trajectory, not a summary: the server returns one waypoint AND
 * one 4x4 camera pose per frame (166 frames on a 16.5 s plan ≈ 110 KB of JSON, enough
 * to blow a tool-result budget on its own). That is artifact-shaped data, and the
 * server already persists the whole plan at `doc_key`, so the default response keeps
 * the summary — stats, floor, grid, notes, provenance, units — and trims the bulk.
 *
 * Nothing is dropped silently: the true counts and the artifact key ride along, so a
 * caller can always tell a preview from a complete path and go fetch the rest.
 */
function compactNavPlan(
  plan: NavPlanResponse,
  includePoses: boolean,
  maxWaypoints: number,
): Record<string, unknown> {
  const { poses, waypoints_world, ...rest } = plan;
  const waypoints = waypoints_world ?? [];
  const kept = sampleEvenly(waypoints, maxWaypoints);
  const truncated = kept.length < waypoints.length;
  const fetchHint = plan.doc_key
    ? `artifact_fetch(scan_id, "${plan.doc_key}") for every waypoint and frame`
    : 'the server did not persist a path doc for this plan';
  return {
    ...rest,
    waypoints_world: kept,
    waypoints_total: waypoints.length,
    ...(truncated
      ? {
          waypoints_truncated: true,
          waypoints_note:
            `preview only — ${kept.length} of ${waypoints.length} waypoints, sampled evenly ` +
            `(first and last kept); ${fetchHint}`,
        }
      : {}),
    ...(includePoses
      ? { poses }
      : {
          poses_omitted: {
            n_poses: poses?.length ?? 0,
            reason:
              'per-frame 4x4 c2w camera flythrough — artifact-shaped, kept out of context ' +
              'by default; pass include_poses:true to inline it',
            fetch: plan.doc_key ?? null,
          },
        }),
  };
}

/** Cordis plugin name used in loader/fiber diagnostics. */
export const name = 'scene-tools';

/** Services required before this plugin activates. */
export const inject = ['mcp', 'broker'];

export function apply(ctx: Context): void {
  const { mcp } = ctx;
  const { client } = ctx.broker;
  mcp.tool(
    ctx,
    'scene_card',
    {
      title: 'Scene context card',
      description:
        'The Tier-1 context card for a scene: source, metric/anchor state, room + summary, ' +
        'closed-world object inventory, ground frame, and workspace artifacts (agent runs, ' +
        'measurements, paths). Small by design — pull details with the other scene tools.',
      inputSchema: { scan_id: z.string() },
      annotations: { readOnlyHint: true },
    },
    async ({ scan_id }) =>
      guard(async () => textResult((await fetchSceneCard(client, scan_id)).markdown)),
  );

  mcp.tool(
    ctx,
    'scene_list_objects',
    {
      title: 'List scene objects',
      description:
        'The closed-world object inventory from the persisted record (facts.objects): model ' +
        'query, operator relabel, confidence, world center/extent. These uids/labels are the ' +
        'ONLY objects that exist — never invent others.',
      inputSchema: {
        scan_id: z.string(),
        include_dismissed: z.boolean().default(false),
      },
      annotations: { readOnlyHint: true },
    },
    async ({ scan_id, include_dismissed }) =>
      guard(async () => {
        const scene = await client.json<PersistedScene>('GET', REST_PATHS.SCENE_DETAIL(scan_id));
        const objects = (scene.facts?.objects ?? []).map((o, i) => ({
          object_id: i,
          label: o.human_label ?? o.query,
          model_query: o.query,
          confidence: o.confidence,
          center: o.center,
          extent: o.extent,
          dismissed: o.dismissed ?? false,
          imported: o.imported_detection ? o.imported_detection.provenance : undefined,
        }));
        const kept = include_dismissed ? objects : objects.filter((o) => !o.dismissed);
        return jsonResult({
          scan_id,
          count: kept.length,
          closed_world: true,
          units_note: scene.facts?.units_note,
          objects: kept,
        });
      }),
  );

  mcp.tool(
    ctx,
    'scene_measure_distance',
    {
      title: 'Measure a distance',
      description:
        'Distance between two WORLD-frame points (POST /measure). The response units are ' +
        '"m" ONLY when the scene has a metric anchor — otherwise "relative". ALWAYS report ' +
        'the units and scale_source with the value; never say metres for relative units.',
      inputSchema: {
        scan_id: z.string(),
        point_a: vec3,
        point_b: vec3,
      },
      annotations: { readOnlyHint: true },
    },
    async ({ scan_id, point_a, point_b }) =>
      guard(async () =>
        jsonResult(
          await client.json<MeasureResponse>('POST', OREOS_REST_PATHS.SCENE_MEASURE(scan_id), {
            kind: 'distance',
            points_world: [point_a, point_b],
          }),
        ),
      ),
  );

  mcp.tool(
    ctx,
    'scene_measure_angle',
    {
      title: 'Measure an angle',
      description:
        'Angle at the vertex (second point) of three WORLD-frame points (POST /measure, ' +
        'kind=angle). Returns degrees.',
      inputSchema: {
        scan_id: z.string(),
        points_world: z.tuple([vec3, vec3, vec3]),
      },
      annotations: { readOnlyHint: true },
    },
    async ({ scan_id, points_world }) =>
      guard(async () =>
        jsonResult(
          await client.json<MeasureResponse>('POST', OREOS_REST_PATHS.SCENE_MEASURE(scan_id), {
            kind: 'angle',
            points_world,
          }),
        ),
      ),
  );

  mcp.tool(
    ctx,
    'scene_plan_path',
    {
      title: 'Plan a robot path',
      description:
        'Plan a path through scanned free space to an object (by uid) or a world point ' +
        '(POST /nav/plan). Provide exactly one of object_uid | point_world. Honest failures: ' +
        '422 no_geometry / no_floor / unreachable_goal (with a nearest-reachable suggestion). ' +
        'The response provenance line ("not certified navigation") must ride along with any use. ' +
        'The per-frame camera flythrough (poses) is left on disk by default and the waypoint ' +
        'list comes back as an evenly-sampled preview — both counts and the full-trajectory ' +
        'artifact key ride along; artifact_fetch(doc_key) for the whole thing.',
      inputSchema: {
        scan_id: z.string(),
        object_uid: z.string().optional(),
        point_world: vec3.optional(),
        start: vec3.optional(),
        clearance: z.number().optional(),
        up_override: z.enum(['+y', '-y', '+z', '-z']).optional(),
        include_poses: z
          .boolean()
          .default(false)
          .describe('Inline the per-frame 4x4 c2w flythrough (large — one pose per frame)'),
        max_waypoints: z
          .number()
          .int()
          .min(2)
          .max(2000)
          .default(DEFAULT_WAYPOINT_PREVIEW)
          .describe('Cap on waypoints returned inline; the plan is sampled evenly to fit'),
      },
      annotations: { readOnlyHint: true },
    },
    async ({
      scan_id,
      object_uid,
      point_world,
      start,
      clearance,
      up_override,
      include_poses,
      max_waypoints,
    }) =>
      guard(async () => {
        if (!!object_uid === !!point_world) {
          throw new Error('Provide exactly one of object_uid | point_world.');
        }
        const body: NavPlanRequest = {
          goal: (object_uid ? { object_uid } : { point_world: point_world! }) as NavPlanRequest['goal'],
        };
        if (start) body.start = start;
        if (clearance != null || up_override) {
          body.params = { ...(clearance != null ? { clearance } : {}), ...(up_override ? { up_override } : {}) };
        }
        const plan = await client.json<NavPlanResponse>(
          'POST',
          OREOS_REST_PATHS.SCENE_NAV_PLAN(scan_id),
          body,
        );
        return jsonResult(compactNavPlan(plan, include_poses, max_waypoints));
      }),
  );

  mcp.tool(
    ctx,
    'scene_planes',
    {
      title: 'Detect floor/wall planes',
      description:
        'Floor + wall plane candidates from the scene geometry (POST /planes; cached ' +
        'per-scene). Used for restyle scoping and understanding room structure.',
      inputSchema: {
        scan_id: z.string(),
        force: z.boolean().optional().describe('Recompute even when a cached result exists'),
        max_walls: z.number().int().min(0).max(16).optional(),
      },
      annotations: { readOnlyHint: true },
    },
    async ({ scan_id, force, max_walls }) =>
      guard(async () =>
        jsonResult(
          await client.json<PlanesResponse>('POST', OREOS_REST_PATHS.SCENE_PLANES(scan_id), {
            ...(force != null ? { force } : {}),
            ...(max_walls != null ? { max_walls } : {}),
          }),
        ),
      ),
  );

  mcp.tool(
    ctx,
    'scene_ground_frame',
    {
      title: 'Read the ground frame',
      description:
        'Read the recorded floor/ceiling frame (GET /ground_frame). If vertical_axis_known is ' +
        'false, the numbers are ABSENT on purpose — say so rather than guessing heights.',
      inputSchema: { scan_id: z.string() },
      annotations: { readOnlyHint: true },
    },
    async ({ scan_id }) =>
      guard(async () =>
        jsonResult(
          await client.json<GroundFrameResponse>('GET', OREOS_REST_PATHS.SCENE_GROUND_FRAME(scan_id)),
        ),
      ),
  );

  mcp.tool(
    ctx,
    'scene_ground_frame_fit',
    {
      title: 'Fit the ground frame',
      description:
        '(Re)fit the floor plane and write the result onto the scene metrics ' +
        '(POST /ground_frame). dry_run=true computes without persisting.',
      inputSchema: {
        scan_id: z.string(),
        dry_run: z.boolean().default(true),
      },
      annotations: { destructiveHint: false },
    },
    async ({ scan_id, dry_run }) =>
      guard(async () =>
        jsonResult(
          await client.json<GroundFrameResponse>(
            'POST',
            OREOS_REST_PATHS.SCENE_GROUND_FRAME(scan_id),
            { dry_run },
          ),
        ),
      ),
  );

  mcp.tool(
    ctx,
    'scene_anchor',
    {
      title: 'Anchor the scene to metres',
      annotations: { readOnlyHint: false, destructiveHint: false },
      description:
        'Metric-anchor calibration (POST /anchor): two picked world points + their real ' +
        'distance in metres. Non-destructive (writes NEW derived/anchor/* artifacts + the ' +
        'derived_latest pointer) and it is what unlocks units:"m" for measure/nav/export. ' +
        'materialize:"job" returns 202 with a job_id to poll via workspace_job_wait.',
      inputSchema: {
        scan_id: z.string(),
        point_a: vec3,
        point_b: vec3,
        distance_m: z.number().positive(),
        materialize: z.enum(['job', 'sync']).default('job'),
      },
    },
    async ({ scan_id, point_a, point_b, distance_m, materialize }) =>
      guard(async () =>
        jsonResult(
          await client.json<Record<string, unknown>>('POST', REST_PATHS.SCENE_ANCHOR(scan_id), {
            point_a,
            point_b,
            distance_m,
            materialize,
          }),
        ),
      ),
  );

  mcp.tool(
    ctx,
    'scene_keyframe_image',
    {
      title: 'Fetch a keyframe image',
      description:
        'One capture keyframe as an image (GET /keyframes/<blob_key>). blob_key comes from ' +
        'the scene detail keyframes[] refs. Real capture photos — cite as photo evidence.',
      inputSchema: { scan_id: z.string(), blob_key: z.string() },
      annotations: { readOnlyHint: true },
    },
    async ({ scan_id, blob_key }) =>
      guard(async () => {
        const { bytes, contentType } = await client.bytes(
          REST_PATHS.SCENE_KEYFRAME(scan_id, blob_key),
        );
        return imageResult(bytes, contentType, `keyframe ${blob_key} of ${scan_id}`);
      }),
  );

  mcp.tool(
    ctx,
    'scene_synthetic_views',
    {
      title: 'List synthetic views',
      description:
        'Renders of the scene\'s own splat registered as evidence (GET /synthetic_views). ' +
        'These are NOT photographs — always carry the "synthetic view" provenance when citing.',
      inputSchema: { scan_id: z.string() },
      annotations: { readOnlyHint: true },
    },
    async ({ scan_id }) =>
      guard(async () =>
        jsonResult(
          await client.json<Record<string, unknown>>(
            'GET',
            OREOS_REST_PATHS.SCENE_SYNTHETIC_VIEWS(scan_id),
          ),
        ),
      ),
  );

  mcp.tool(
    ctx,
    'scene_synthetic_view_image',
    {
      title: 'Fetch a synthetic view image',
      description:
        'One synthetic view PNG (a RENDER of the splat, not a photo — say so when citing).',
      inputSchema: { scan_id: z.string(), view_id: z.string() },
      annotations: { readOnlyHint: true },
    },
    async ({ scan_id, view_id }) =>
      guard(async () => {
        const { bytes, contentType } = await client.bytes(
          OREOS_REST_PATHS.SCENE_SYNTHETIC_VIEW_PNG(scan_id, view_id),
        );
        return imageResult(bytes, contentType, `synthetic view ${view_id} of ${scan_id} (render, not a photo)`);
      }),
  );

  mcp.tool(
    ctx,
    'scene_lod',
    {
      title: 'LOD status',
      description:
        'Level-of-detail index status for the scene splat (GET /lod). Always 200 — read ' +
        '`status` (ready|none|running|error|not_needed).',
      inputSchema: { scan_id: z.string() },
      annotations: { readOnlyHint: true },
    },
    async ({ scan_id }) =>
      guard(async () =>
        jsonResult(await client.json<OreosLodResponse>('GET', OREOS_REST_PATHS.SCENE_LOD(scan_id))),
      ),
  );

  mcp.tool(
    ctx,
    'scene_lod_build',
    {
      title: 'Build LOD variants',
      annotations: { readOnlyHint: false, destructiveHint: false },
      description:
        'Spawn the LOD decimation job (POST /lod → 202 {job_id}; 409 lod_job_active when one ' +
        'is already running). Poll with workspace_job_wait.',
      inputSchema: { scan_id: z.string() },
    },
    async ({ scan_id }) =>
      guard(async () =>
        jsonResult(
          await client.json<Record<string, unknown>>('POST', OREOS_REST_PATHS.SCENE_LOD(scan_id)),
        ),
      ),
  );

  mcp.tool(
    ctx,
    'scene_imported_objects',
    {
      title: 'Imported-splat objects status',
      description:
        'Status of geometry-derived objects for an imported splat (GET /imported_objects).',
      inputSchema: { scan_id: z.string() },
      annotations: { readOnlyHint: true },
    },
    async ({ scan_id }) =>
      guard(async () =>
        jsonResult(
          await client.json<ImportedObjectsStatus>(
            'GET',
            OREOS_REST_PATHS.SCENE_IMPORTED_OBJECTS(scan_id),
          ),
        ),
      ),
  );

  mcp.tool(
    ctx,
    'scene_imported_objects_run',
    {
      title: 'Cluster imported-splat objects',
      annotations: { readOnlyHint: false, destructiveHint: false },
      description:
        'Cluster an imported splat\'s geometry into facts.objects (POST /imported_objects). ' +
        'Labels carry geometry/synthetic-view provenance — confidence 0.0 is honest, not a bug.',
      inputSchema: { scan_id: z.string() },
    },
    async ({ scan_id }) =>
      guard(async () =>
        jsonResult(
          await client.json<Record<string, unknown>>(
            'POST',
            OREOS_REST_PATHS.SCENE_IMPORTED_OBJECTS(scan_id),
          ),
        ),
      ),
  );

  mcp.tool(
    ctx,
    'scene_object_complete',
    {
      title: 'Complete an object to 3D',
      annotations: { readOnlyHint: false, destructiveHint: false },
      description:
        'SAM-3D-Objects mesh completion for a segmented object (POST /objects/<uid>/complete ' +
        '→ 202 {job_id}, broker-thread lane). Poll with scene_job_wait. Output is GENERATED ' +
        'geometry (honesty envelope in the artifact meta.json).',
      inputSchema: { scan_id: z.string(), uid: z.string() },
    },
    async ({ scan_id, uid }) =>
      guard(async () =>
        jsonResult(
          await client.json<Record<string, unknown>>(
            'POST',
            OREOS_REST_PATHS.SCENE_OBJECT_COMPLETE(scan_id, uid),
          ),
        ),
      ),
  );

  mcp.tool(
    ctx,
    'scene_object_variants',
    {
      title: 'Generate object variants',
      annotations: { readOnlyHint: false, destructiveHint: false },
      description:
        'TRELLIS variant generation for a completed object (POST /objects/<uid>/variants → ' +
        '202 {job_id}). Optional body fields pass through (see server docs). GENERATED assets.',
      inputSchema: {
        scan_id: z.string(),
        uid: z.string(),
        body: z.record(z.unknown()).optional().describe('Optional route body passthrough'),
      },
    },
    async ({ scan_id, uid, body }) =>
      guard(async () =>
        jsonResult(
          await client.json<Record<string, unknown>>(
            'POST',
            OREOS_REST_PATHS.SCENE_OBJECT_VARIANTS(scan_id, uid),
            body ?? {},
          ),
        ),
      ),
  );

  mcp.tool(
    ctx,
    'scene_segment',
    {
      title: 'Segment an object (advanced)',
      annotations: { readOnlyHint: false, destructiveHint: false },
      description:
        'SAM-3 click segmentation (POST /segment). Advanced: the body must match the server ' +
        'contract (click point + view selection — see routes_sam3d.py / the OS UI). Provided ' +
        'as a raw passthrough so the full surface stays reachable.',
      inputSchema: {
        scan_id: z.string(),
        body: z.record(z.unknown()).describe('Raw request body for the segment route'),
      },
    },
    async ({ scan_id, body }) =>
      guard(async () =>
        jsonResult(
          await client.json<Record<string, unknown>>(
            'POST',
            OREOS_REST_PATHS.SCENE_SEGMENT(scan_id),
            body,
          ),
        ),
      ),
  );

  mcp.tool(
    ctx,
    'scene_job_status',
    {
      title: 'Check a scene job',
      description:
        'One status read of a broker-thread scene job (SAM-3D completion, variants, view ' +
        'renders). NOTE: these jobs honestly 404 across a broker restart.',
      inputSchema: { scan_id: z.string(), job_id: z.string() },
      annotations: { readOnlyHint: true },
    },
    async ({ scan_id, job_id }) =>
      guard(async () =>
        jsonResult(
          await client.json<Record<string, unknown>>(
            'GET',
            OREOS_REST_PATHS.SCENE_JOB(scan_id, job_id),
          ),
        ),
      ),
  );

  mcp.tool(
    ctx,
    'scene_job_wait',
    {
      title: 'Wait for a scene job',
      description: 'Poll a broker-thread scene job until done/error/timeout.',
      inputSchema: {
        scan_id: z.string(),
        job_id: z.string(),
        timeout_s: z.number().min(1).max(3600).default(600),
        poll_s: z.number().min(0.05).max(60).default(2),
      },
      annotations: { readOnlyHint: true },
    },
    async ({ scan_id, job_id, timeout_s, poll_s }, extra) =>
      guard(async () =>
        jsonResult(await pollSceneJob(client, scan_id, job_id, timeout_s, poll_s, progressTick(extra))),
      ),
  );

  mcp.tool(
    ctx,
    'scene_share_access',
    {
      title: 'What each share link was used for',
      annotations: { readOnlyHint: true },
      description:
        'Owner-only read of the share-link access log (GET /share/access): per minted link ' +
        '(access_id) — opened, request count, visits (new visit after 30 min idle), ' +
        'first/last seen, returned (a visit >= 24 h after the first), devices, and the Q&A ' +
        'questions the viewer asked. This is the behavioural evidence for customer discovery ' +
        '(L5 = returned). Match access_id to the person you sent the link to. ' +
        'events:true appends the raw event list.',
      inputSchema: {
        scan_id: z.string(),
        events: z.boolean().optional(),
      },
    },
    async ({ scan_id, events }) =>
      guard(async () =>
        jsonResult(
          await client.json<ShareAccessResponse>(
            'GET',
            REST_PATHS.SCENE_SHARE_ACCESS(scan_id) + (events ? '?events=1' : ''),
          ),
        ),
      ),
  );

  mcp.tool(
    ctx,
    'scene_share',
    {
      title: 'Mint a read-only share link',
      annotations: { readOnlyHint: false, destructiveHint: false },
      description:
        'Mint a read-only, single-scene share token + embed URL (POST /share; default TTL ' +
        '30 days). Share tokens can NEVER reach the OS feature routes — viewer only.',
      inputSchema: {
        scan_id: z.string(),
        ttl_seconds: z.number().int().positive().optional(),
      },
    },
    async ({ scan_id, ttl_seconds }) =>
      guard(async () =>
        jsonResult(
          await client.json<ShareSceneResponse>('POST', REST_PATHS.SCENE_SHARE(scan_id), {
            ...(ttl_seconds ? { ttl_seconds } : {}),
          }),
        ),
      ),
  );
}
