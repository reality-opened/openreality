/**
 * Open Reality mock-backend simulator — the offline dev target.
 *
 * A dependency-free node:http server implementing the subset of the broker
 * contract the MCP tools (and the OS web client's REST layer) speak. Response
 * shapes are typed against @reality/protocol, so contract drift fails the
 * build/tests instead of lying at runtime. Deliberately deterministic: jobs
 * advance one stage per status poll and agent runs release a few events per
 * cursor poll — no wall-clock flakiness in CI.
 *
 * The real server's honesty doctrine is simulated too: un-anchored scenes
 * measure in relative units, the Isaac export is metric-gated, agent runs cost
 * "$" while replays are $0. The point of the sim is to develop against true
 * shapes AND true refusals.
 */
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import type { AddressInfo } from 'node:net';
import type {
  AgentRunEventsResponse,
  AgentRunListItem,
  ApiKeySummary,
  ExportManifestResponse,
  GroundFrameResponse,
  ListApiKeysResponse,
  ManifestTreeEntry,
  MeasureResponse,
  MintApiKeyResponse,
  NavPlanResponse,
  OreosAgentEvent,
  OreosJobStatus,
  OreosLodResponse,
  OreosManifest,
  PersistedScene,
  PlanesResponse,
  RevokeApiKeyResponse,
} from '@reality/protocol';
import { API_KEY_PREFIX } from '@reality/protocol';
import {
  FIXTURE_RUN_ID,
  FIXTURE_SCAN_ID,
  PLY_STUB,
  PNG_1PX,
  fixtureManifest,
  fixtureScene,
  ingestedScene,
  toListItem,
} from './fixture.js';
import { buildZip } from './zip.js';

// ── state ──

interface SimJob {
  states: OreosJobStatus[];
  idx: number;
  doneFired: boolean;
  onDone?: () => void;
}

interface SimRun {
  run_id: string;
  kind: AgentRunListItem['kind'];
  events: OreosAgentEvent[];
  released: number;
  perPoll: number;
  started_at: number;
  replay: boolean;
  source_run_id?: string;
  cost_usd: number;
  finishedRecorded: boolean;
}

interface SimApiKey {
  key_id: string;
  name: string;
  prefix: string;
  created_at: number;
  last_used_at: number | null;
  revoked_at: number | null;
  /** The full `ork_...` secret. Never serialized back out except once, in the mint
   *  response — GET/DELETE only ever return the `ApiKeySummary` projection. */
  secret: string;
}

interface SimSceneBundle {
  scene: PersistedScene;
  manifest: OreosManifest;
  derived: Map<string, Buffer>;
  blobs: Map<string, Buffer>;
  syntheticViews: Array<Record<string, unknown>>;
  lod: 'none' | 'running' | 'ready';
  planesCached: boolean;
}

interface SimState {
  scenes: Map<string, SimSceneBundle>;
  wsJobs: Map<string, SimJob>;
  sceneJobs: Map<string, Map<string, SimJob>>;
  runs: Map<string, Map<string, SimRun>>;
  splatUploads: Map<string, Buffer[]>;
  prepared: Map<string, { status: 'running' | 'ready' | 'error'; artifact?: Record<string, unknown>; job_id: string }>;
  /** Minted durable API keys, keyed by key_id (account-wide — the simulator only ever
   *  models one account). Looked up by secret on every request's auth check. */
  apiKeys: Map<string, SimApiKey>;
  seq: { job: number; scan: number; run: number; token: number; upload: number; path: number; apiKey: number };
}

function nowS(): number {
  return Math.floor(Date.now() / 1000);
}

/** Deterministic (no crypto/wall-clock) opaque secret generator — same "no flakiness"
 *  philosophy as the rest of the sim's ids. Not a real capability-bearing secret. */
function mintKeySecret(n: number): string {
  const filler = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
  let body = `sim${String(n).padStart(4, '0')}`;
  while (body.length < 43) body += filler[body.length % filler.length];
  return `${API_KEY_PREFIX}${body}`;
}

function bundleFor(scene: PersistedScene, manifest?: OreosManifest): SimSceneBundle {
  const derived = new Map<string, Buffer>();
  const blobs = new Map<string, Buffer>();
  for (const kf of scene.keyframes) blobs.set(kf.blob_key, PNG_1PX);
  return {
    scene,
    manifest:
      manifest ??
      ({
        version: 1,
        updated_at: 0,
        annotations_key: null,
        agent_runs: [],
        edits: [],
        paths: [],
        variations: [],
        measurements: [],
      } as OreosManifest),
    derived,
    blobs,
    syntheticViews: [],
    lod: 'none',
    planesCached: false,
  };
}

function initialState(): SimState {
  const state: SimState = {
    scenes: new Map(),
    wsJobs: new Map(),
    sceneJobs: new Map(),
    runs: new Map(),
    splatUploads: new Map(),
    prepared: new Map(),
    apiKeys: new Map(),
    seq: { job: 0, scan: 0, run: 0, token: 0, upload: 0, path: 0, apiKey: 0 },
  };
  const fixture = bundleFor(fixtureScene(), fixtureManifest());
  // the fixture's prior run is persisted, replayable for $0
  const priorEvents = annotateScript(FIXTURE_SCAN_ID, fixture.scene);
  fixture.derived.set(
    `derived/demo/agent_runs/${FIXTURE_RUN_ID}/events.json`,
    Buffer.from(JSON.stringify(priorEvents), 'utf8'),
  );
  fixture.derived.set(
    'derived/demo/measurements/20260810T0910/measure.json',
    Buffer.from(JSON.stringify({ kind: 'distance', value: 1.02, units: 'm' }), 'utf8'),
  );
  state.scenes.set(FIXTURE_SCAN_ID, fixture);
  const runs = new Map<string, SimRun>();
  runs.set(FIXTURE_RUN_ID, {
    run_id: FIXTURE_RUN_ID,
    kind: 'annotate',
    events: priorEvents,
    released: priorEvents.length,
    perPoll: 4,
    started_at: fixture.scene.created_at + 500,
    replay: false,
    cost_usd: 0.041,
    finishedRecorded: true,
  });
  state.runs.set(FIXTURE_SCAN_ID, runs);
  return state;
}

// ── event scripts ──

let evSeq = 0;
function ev(type: string, payload: Record<string, unknown>, phase?: string): OreosAgentEvent {
  evSeq += 1;
  return { seq: evSeq, ts: nowS(), type, payload, ...(phase ? { phase } : {}) };
}

function resetScriptSeq(): void {
  evSeq = 0;
}

function annotateScript(scanId: string, scene: PersistedScene): OreosAgentEvent[] {
  resetScriptSeq();
  const objs = scene.facts.objects;
  const anchored = scene.derived_latest?.kind === 'anchor';
  return [
    ev('run_meta', { replay: false, model: 'simulator', run_kind: 'annotate' }),
    ev('agent_state', { phase: 'survey', enabled: true }, 'survey'),
    ev('agent_thought', { type: 'thinking', content: `Surveying ${scanId}: ${objs.length} persisted objects.` }, 'survey'),
    ev('agent_tool_event', { tool: 'list_scene_objects', status: 'started', args: {} }, 'labels'),
    ev('agent_tool_event', { tool: 'list_scene_objects', status: 'succeeded', latency_ms: 12, result: { count: objs.length } }, 'labels'),
    ev('agent_finding', { query: objs[0]?.query ?? 'object', description: 'Largest object in the inventory.', confidence: objs[0]?.confidence ?? 0.5, position: objs[0]?.center }, 'labels'),
    ev('agent_thought', { type: 'observation', content: `Room reads as an ${scene.report.room_type}. ${anchored ? 'Metric anchor present — dimensions in metres.' : 'No metric anchor — relative units only.'}` }, 'description'),
    ev('agent_ui_command', { name: 'show_dimension_labels', args: { visible: true } }, 'dimensions'),
    ev('run_done', { llm_calls: 3, cost_usd: 0.041, findings_emitted: 1 }),
  ];
}

function chatScript(scanId: string, scene: PersistedScene, message: string): OreosAgentEvent[] {
  resetScriptSeq();
  const objs = scene.facts.objects;
  const anchored = scene.derived_latest?.kind === 'anchor';
  const unitsLine = anchored
    ? `Lengths are metres (anchored, scale ${scene.derived_latest?.scale_factor ?? '?'} m/unit).`
    : 'Lengths are RELATIVE units — no metric anchor on this scene.';
  return [
    ev('run_meta', { replay: false, model: 'simulator', run_kind: 'chat' }),
    ev('agent_thought', { type: 'chat_response', author: 'user', content: message }),
    ev('agent_tool_event', { tool: 'list_scene_objects', status: 'started', args: {} }),
    ev('agent_tool_event', { tool: 'list_scene_objects', status: 'succeeded', latency_ms: 9, result: { count: objs.length } }),
    ev('agent_thought', {
      type: 'chat_response',
      content:
        `The scene has ${objs.length} closed-world objects: ` +
        `${objs.map((o) => o.human_label ?? o.query).join(', ')}. ${unitsLine}`,
    }),
    ev('run_done', { llm_calls: 2, cost_usd: 0.0123, findings_emitted: 0 }),
  ];
}

function pilotScript(scanId: string, scene: PersistedScene, instruction: string): OreosAgentEvent[] {
  resetScriptSeq();
  return [
    ev('run_meta', { replay: false, model: 'simulator', run_kind: 'pilot' }),
    ev('agent_thought', { type: 'thinking', content: `Pilot instruction: ${instruction}` }),
    ev('agent_tool_event', { tool: 'plan_path', status: 'started', args: {} }),
    ev('agent_tool_event', { tool: 'plan_path', status: 'succeeded', latency_ms: 31, result: { waypoints: 8 } }),
    ev('agent_ui_command', { name: 'drive_path', args: { path_id: 'path-pilot' } }),
    ev('run_done', { llm_calls: 2, cost_usd: 0.008, findings_emitted: 0 }),
  ];
}

// ── helpers ──

function sendJson(res: ServerResponse, code: number, body: unknown): void {
  const buf = Buffer.from(JSON.stringify(body), 'utf8');
  res.writeHead(code, { 'Content-Type': 'application/json', 'Content-Length': buf.length });
  res.end(buf);
}

function sendBytes(res: ServerResponse, buf: Buffer, type: string): void {
  res.writeHead(200, { 'Content-Type': type, 'Content-Length': buf.length });
  res.end(buf);
}

function readBody(req: IncomingMessage): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on('data', (c: Buffer) => chunks.push(c));
    req.on('end', () => resolve(Buffer.concat(chunks)));
    req.on('error', reject);
  });
}

function advance(job: SimJob): OreosJobStatus {
  const status = job.states[Math.min(job.idx, job.states.length - 1)]!;
  if (job.idx < job.states.length - 1) job.idx += 1;
  const state = status.status;
  if ((state === 'done' || state === 'error') && !job.doneFired) {
    job.doneFired = true;
    job.onDone?.();
  }
  return status;
}

function dist3(a: number[], b: number[]): number {
  return Math.hypot(a[0]! - b[0]!, a[1]! - b[1]!, a[2]! - b[2]!);
}

function activeRun(runs: Map<string, SimRun> | undefined): SimRun | null {
  if (!runs) return null;
  for (const r of runs.values()) if (r.released < r.events.length) return r;
  return null;
}

function exportTree(scene: PersistedScene, format: string): ManifestTreeEntry[] {
  const p = format === 'groot_lerobot_v2' ? `${scene.scan_id}_groot_lerobot_v2` : scene.scan_id;
  const base: ManifestTreeEntry[] = [
    { path: `${p}/cloud.npz`, size: 4096 },
    { path: `${p}/trajectory.npz`, size: 1024 },
    { path: `${p}/meta/info.json`, size: 256 },
  ];
  if (scene.has_splat) base.push({ path: `${p}/splat.ply`, size: PLY_STUB.length });
  if (format === 'groot_lerobot_v2') base.push({ path: `${p}/meta/modality.json`, size: 128 });
  if (format === 'isaac_usd') base.push({ path: `${p}/isaac/scene.usda`, size: 2048 });
  return base;
}

// ── the server ──

export interface SimulatorOptions {
  port?: number;
}

export interface RunningSimulator {
  url: string;
  server: Server;
  close: () => Promise<void>;
}

export function startSimulator(opts: SimulatorOptions = {}): Promise<RunningSimulator> {
  const state = initialState();

  const server = createServer((req, res) => {
    handle(req, res).catch((err) => {
      sendJson(res, 500, { error: 'simulator_crash', message: String(err?.stack ?? err) });
    });
  });

  async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const url = new URL(req.url ?? '/', 'http://sim');
    const seg = url.pathname.split('/').filter(Boolean);
    const method = req.method ?? 'GET';

    if (url.pathname === '/health') return sendJson(res, 200, { status: 'ok', gpu: false });

    // auth: any bearer except the literal 'expired-token' (which only /refresh accepts)
    const auth = req.headers.authorization ?? '';
    const bearer = auth.startsWith('Bearer ') ? auth.slice(7) : url.searchParams.get('auth_token') ?? '';
    if (url.pathname === '/api/session/refresh' && method === 'POST') {
      if (!bearer) return sendJson(res, 401, { error: 'unauthorized' });
      state.seq.token += 1;
      return sendJson(res, 200, { token: `sim-token-${state.seq.token}`, expires_at: nowS() + 12 * 3600 });
    }
    // Only a Clerk JWT / session-token bearer may mint a key — the wire contract rejects
    // ANY ork_-SHAPED bearer here with 403, even one this sim never minted (an unknown
    // API key trying to mint is still "an API key trying to mint", not "unauthenticated");
    // this must run before the general bearer-validity gate below, which would otherwise
    // 401 an unregistered ork_ bearer first and this 403 would never be reached.
    if (seg[0] === 'api' && seg[1] === 'keys' && method === 'POST' && bearer.startsWith(API_KEY_PREFIX)) {
      return sendJson(res, 403, { error: 'api_key_cannot_mint' });
    }
    if (bearer.startsWith(API_KEY_PREFIX)) {
      // A durable API key must be one this sim minted, and not revoked — unlike the
      // blanket "any nonempty bearer works" leniency below (which exists for the
      // arbitrary session/JWT-shaped tokens the rest of the test suite passes around).
      const rec = [...state.apiKeys.values()].find((k) => k.secret === bearer);
      if (!rec || rec.revoked_at !== null) return sendJson(res, 401, { error: 'unauthorized' });
      rec.last_used_at = nowS();
    } else if (!bearer || bearer === 'expired-token') {
      return sendJson(res, 401, { error: 'unauthorized' });
    }

    // ── API keys ──
    if (seg[0] === 'api' && seg[1] === 'keys') {
      if (method === 'POST' && seg.length === 2) {
        const body = JSON.parse((await readBody(req)).toString('utf8') || '{}') as { name?: string };
        if (body.name !== undefined && (typeof body.name !== 'string' || body.name.length > 64)) {
          return sendJson(res, 422, { error: 'invalid_name' });
        }
        state.seq.apiKey += 1;
        const secret = mintKeySecret(state.seq.apiKey);
        const rec: SimApiKey = {
          key_id: `key_${String(state.seq.apiKey).padStart(4, '0')}`,
          name: body.name ?? '',
          prefix: secret.slice(0, 10),
          created_at: nowS(),
          last_used_at: null,
          revoked_at: null,
          secret,
        };
        state.apiKeys.set(rec.key_id, rec);
        const out: MintApiKeyResponse = {
          key: rec.secret,
          key_id: rec.key_id,
          name: rec.name,
          prefix: rec.prefix,
          created_at: rec.created_at,
        };
        return sendJson(res, 200, out);
      }

      if (method === 'GET' && seg.length === 2) {
        const keys: ApiKeySummary[] = [...state.apiKeys.values()]
          .sort((a, b) => b.created_at - a.created_at)
          .map(({ secret: _secret, ...summary }) => summary);
        const out: ListApiKeysResponse = { keys };
        return sendJson(res, 200, out);
      }

      if (method === 'DELETE' && seg[2]) {
        const rec = state.apiKeys.get(decodeURIComponent(seg[2]));
        if (!rec) return sendJson(res, 404, { error: 'key_not_found' });
        if (rec.revoked_at === null) rec.revoked_at = nowS();
        const out: RevokeApiKeyResponse = { key_id: rec.key_id, revoked_at: rec.revoked_at };
        return sendJson(res, 200, out);
      }

      return sendJson(res, 404, { error: 'not_found' });
    }

    // ── workspace ──
    if (seg[0] === 'api' && seg[1] === 'workspace') {
      if (seg[2] === 'jobs' && seg[3] && method === 'GET') {
        const job = state.wsJobs.get(seg[3]);
        if (!job) return sendJson(res, 404, { error: 'job_not_found' });
        return sendJson(res, 200, advance(job));
      }
      if (seg[2] === 'ingest' && method === 'POST') {
        const body = await readBody(req);
        const filename = String(req.headers['x-upload-filename'] ?? '');
        const label = req.headers['x-scene-label'] ? String(req.headers['x-scene-label']) : null;
        if (seg[3] === 'video' || seg[3] === 'recording') {
          if (!filename) return sendJson(res, 400, { error: 'missing_filename' });
          if (body.length === 0) return sendJson(res, 400, { error: 'empty_body' });
          const isRec = seg[3] === 'recording';
          const demoSource = String(req.headers['x-demo-source'] ?? '');
          const source = isRec
            ? ('robot_recording' as const)
            : demoSource === 'gemini2'
              ? ('recon_gemini2' as const)
              : demoSource === 'depthcam'
                ? ('recon_depthcam' as const)
                : ('recon_video' as const);
          state.seq.scan += 1;
          state.seq.job += 1;
          const scanId = `sim-scan-${String(state.seq.scan).padStart(2, '0')}`;
          const jobId = `job-${String(state.seq.job).padStart(3, '0')}`;
          const mk = (status: OreosJobStatus['status'], stage?: OreosJobStatus['stage'], extra?: Partial<OreosJobStatus>): OreosJobStatus => ({
            job_id: jobId,
            status,
            scan_id: scanId,
            updated_at: nowS(),
            ...(stage ? { stage } : {}),
            ...extra,
          });
          state.wsJobs.set(jobId, {
            idx: 0,
            doneFired: false,
            states: [
              mk('queued', 'upload'),
              mk('running', 'recon', { progress: 0.4 }),
              mk('running', 'report', { progress: 0.8 }),
              mk('running', 'persist', { progress: 0.95 }),
              mk('done', 'persist', { result: { scan_id: scanId } }),
            ],
            onDone: () => {
              const bundle = bundleFor(ingestedScene(scanId, label, source));
              if (isRec) {
                bundle.derived.set(
                  'derived/demo/recordings/qc_report.json',
                  Buffer.from(JSON.stringify({ verdict: 'PASS', consistency_mm: 21.3, provenance: 'simulator' })),
                );
              }
              state.scenes.set(scanId, bundle);
            },
          });
          return sendJson(res, 202, { job_id: jobId, scan_id: scanId });
        }
        if (seg[3] === 'splat' && seg.length === 4) {
          if (!body.subarray(0, 3).equals(Buffer.from('ply'))) {
            return sendJson(res, 422, { error: 'not_a_gaussian_splat' });
          }
          state.seq.scan += 1;
          const scanId = `sim-splat-${String(state.seq.scan).padStart(2, '0')}`;
          state.scenes.set(scanId, bundleFor(ingestedScene(scanId, label, 'imported_splat')));
          return sendJson(res, 201, { scan_id: scanId, gaussian_count: 123_456 });
        }
        if (seg[3] === 'splat' && seg[4] === 'init') {
          state.seq.upload += 1;
          const id = `up-${state.seq.upload}`;
          state.splatUploads.set(id, []);
          return sendJson(res, 200, { upload_id: id });
        }
        if (seg[3] === 'splat' && seg[5] === 'chunk') {
          const chunks = state.splatUploads.get(seg[4]!);
          if (!chunks) return sendJson(res, 404, { error: 'upload_not_found' });
          chunks[Number(seg[6])] = body;
          return sendJson(res, 200, { ok: true, index: Number(seg[6]) });
        }
        if (seg[3] === 'splat' && seg[5] === 'finalize') {
          const chunks = state.splatUploads.get(seg[4]!);
          if (!chunks) return sendJson(res, 404, { error: 'upload_not_found' });
          const whole = Buffer.concat(chunks.filter(Boolean));
          if (!whole.subarray(0, 3).equals(Buffer.from('ply'))) {
            return sendJson(res, 422, { error: 'not_a_gaussian_splat' });
          }
          state.seq.scan += 1;
          state.seq.job += 1;
          const scanId = `sim-splat-${String(state.seq.scan).padStart(2, '0')}`;
          const jobId = `job-${String(state.seq.job).padStart(3, '0')}`;
          state.wsJobs.set(jobId, {
            idx: 0,
            doneFired: false,
            states: [
              { job_id: jobId, status: 'queued', scan_id: scanId },
              { job_id: jobId, status: 'running', stage: 'persist', scan_id: scanId },
              { job_id: jobId, status: 'done', scan_id: scanId, result: { gaussian_count: whole.length } },
            ],
            onDone: () => {
              state.scenes.set(scanId, bundleFor(ingestedScene(scanId, null, 'imported_splat')));
            },
          });
          return sendJson(res, 202, { job_id: jobId, scan_id: scanId });
        }
      }
      return sendJson(res, 404, { error: 'not_found' });
    }

    // ── scenes ──
    if (seg[0] === 'api' && seg[1] === 'scenes' && method === 'GET' && seg.length === 2) {
      const scenes = [...state.scenes.values()]
        .map((b) => toListItem(b.scene))
        .sort((a, b) => b.created_at - a.created_at);
      return sendJson(res, 200, { scenes });
    }

    if (seg[0] === 'api' && seg[1] === 'demo' && seg[2] === 'scene' && seg[3]) {
      // synthetic views namespace (persisted demo route prefix)
      const bundle = state.scenes.get(seg[3]);
      if (!bundle) return sendJson(res, 404, { error: 'scene_not_found' });
      if (seg[4] === 'synthetic_views' && seg.length === 5) {
        if (method === 'GET') {
          return sendJson(res, 200, {
            views: bundle.syntheticViews.map((v, i) => ({ ...v, url: `/api/demo/scene/${seg[3]}/synthetic_views/${v.view_id}.png` })),
            count: bundle.syntheticViews.length,
            provenance: 'synthetic view',
            provenance_detail: 'rendered from the imported splat',
          });
        }
        if (method === 'POST') {
          const body = JSON.parse((await readBody(req)).toString('utf8') || '{}') as {
            replace?: boolean;
            views?: Array<Record<string, unknown>>;
          };
          if (body.replace) bundle.syntheticViews = [];
          const added = (body.views ?? []).map((v, i) => {
            const viewId = `view-${bundle.syntheticViews.length + i + 1}`;
            const b64 = typeof v.image_b64 === 'string' ? v.image_b64 : '';
            bundle.blobs.set(`${viewId}.png`, b64 ? Buffer.from(b64, 'base64') : PNG_1PX);
            return { view_id: viewId, index: bundle.syntheticViews.length + i, ...v, image_b64: undefined };
          });
          bundle.syntheticViews.push(...added);
          return sendJson(res, 200, {
            views: added.map((v, i) => ({ view_id: v.view_id, index: i })),
            count: bundle.syntheticViews.length,
            provenance: 'synthetic view',
            provenance_detail: 'rendered from the imported splat',
          });
        }
      }
      if (seg[4] === 'synthetic_views' && seg[5]?.endsWith('.png') && method === 'GET') {
        const key = seg[5];
        const buf = bundle.blobs.get(key) ?? PNG_1PX;
        return sendBytes(res, buf, 'image/png');
      }
      return sendJson(res, 404, { error: 'not_found' });
    }

    if (!(seg[0] === 'api' && seg[1] === 'scenes' && seg[2])) {
      return sendJson(res, 404, { error: 'not_found' });
    }

    const scanId = seg[2];
    const bundle = state.scenes.get(scanId);
    if (!bundle) return sendJson(res, 404, { error: 'scene_not_found' });
    const scene = bundle.scene;
    const anchored = scene.derived_latest?.kind === 'anchor';
    const scale = anchored ? (scene.derived_latest?.scale_factor ?? 1) : 1;
    const scaleSource = anchored ? `anchor:${scene.derived_latest!.source_key}` : 'none';
    const rest = seg.slice(3);

    // detail
    if (rest.length === 0 && method === 'GET') return sendJson(res, 200, scene);

    // blobs
    if (rest[0] === 'keyframes' && rest[1] && method === 'GET') {
      const buf = bundle.blobs.get(decodeURIComponent(rest[1]));
      if (!buf) return sendJson(res, 404, { error: 'keyframe_not_found' });
      return sendBytes(res, buf, 'image/png');
    }
    if (rest[0] === 'splat.ply' && method === 'GET') return sendBytes(res, PLY_STUB, 'application/octet-stream');
    if (rest[0] === 'cloud.ply' && method === 'GET') return sendBytes(res, PLY_STUB, 'application/octet-stream');

    if (rest[0] === 'derived' && method === 'GET') {
      const key = 'derived/' + rest.slice(1).map(decodeURIComponent).join('/');
      if (key === 'derived/demo/manifest.json') {
        return sendJson(res, 200, bundle.manifest);
      }
      const buf = bundle.derived.get(key);
      if (!buf) return sendJson(res, 404, { error: 'artifact_not_found', key });
      const type = key.endsWith('.json') ? 'application/json' : key.endsWith('.png') ? 'image/png' : 'application/octet-stream';
      return sendBytes(res, buf, type);
    }

    // share
    if (rest[0] === 'share' && method === 'POST') {
      const body = JSON.parse((await readBody(req)).toString('utf8') || '{}') as { ttl_seconds?: number };
      const ttl = body.ttl_seconds ?? 30 * 24 * 3600;
      state.seq.token += 1;
      const tok = `sim-share-${state.seq.token}`;
      return sendJson(res, 200, {
        share_token: tok,
        embed_url: `http://sim/embed.html#scan_id=${scanId}&share_token=${tok}`,
        scan_id: scanId,
        expires_at: nowS() + ttl,
      });
    }

    // anchor
    if (rest[0] === 'anchor' && method === 'POST') {
      const body = JSON.parse((await readBody(req)).toString('utf8') || '{}') as {
        point_a: number[];
        point_b: number[];
        distance_m: number;
        materialize?: string;
      };
      const measured = dist3(body.point_a, body.point_b);
      if (!(measured > 0) || !(body.distance_m > 0)) return sendJson(res, 422, { error: 'bad_anchor_points' });
      const sf = body.distance_m / measured;
      const stamp = `derived/anchor/${Date.now()}`;
      const applyAnchor = () => {
        bundle.derived.set(`${stamp}/cloud.npz`, Buffer.from('npz-stub'));
        bundle.derived.set(`${stamp}/splat.ply`, PLY_STUB);
        scene.derived_latest = {
          kind: 'anchor',
          cloud_key: `${stamp}/cloud.npz`,
          trajectory_key: null,
          splat_key: `${stamp}/splat.ply`,
          source_key: `${stamp}/cloud.npz`,
          applied_at: new Date().toISOString(),
          scale_factor: sf,
        };
        scene.facts.units_note = 'lengths in metres via metric anchor';
      };
      if (body.materialize === 'job') {
        state.seq.job += 1;
        const jobId = `job-${String(state.seq.job).padStart(3, '0')}`;
        state.wsJobs.set(jobId, {
          idx: 0,
          doneFired: false,
          states: [
            { job_id: jobId, status: 'queued', scan_id: scanId },
            { job_id: jobId, status: 'running', stage: 'persist', scan_id: scanId },
            { job_id: jobId, status: 'done', scan_id: scanId },
          ],
          onDone: applyAnchor,
        });
        return sendJson(res, 202, {
          scale_factor: sf,
          measured_distance: measured,
          distance_m: body.distance_m,
          applied_at: new Date().toISOString(),
          pending: true,
          job_id: jobId,
          gauge_span_before: measured,
          gauge_span_after_m: body.distance_m,
        });
      }
      applyAnchor();
      return sendJson(res, 200, {
        scale_factor: sf,
        measured_distance: measured,
        distance_m: body.distance_m,
        applied_at: new Date().toISOString(),
        calibrated_cloud_key: `${stamp}/cloud.npz`,
        calibrated_trajectory_key: null,
        calibrated_splat_key: `${stamp}/splat.ply`,
        gauge_span_before: measured,
        gauge_span_after_m: body.distance_m,
        cloud_extent_before: 5.9,
        cloud_extent_after_m: 5.9 * sf,
      });
    }

    // measure
    if (rest[0] === 'measure' && method === 'POST') {
      const body = JSON.parse((await readBody(req)).toString('utf8') || '{}') as {
        kind: 'distance' | 'angle';
        points_world: number[][];
      };
      if (body.kind === 'distance') {
        if (body.points_world?.length !== 2) return sendJson(res, 422, { error: 'bad_points' });
        const raw = dist3(body.points_world[0]!, body.points_world[1]!);
        const out: MeasureResponse = {
          kind: 'distance',
          value: raw * scale,
          units: anchored ? 'm' : 'relative',
          scale_factor: anchored ? scale : null,
          scale_source: scaleSource,
        };
        return sendJson(res, 200, out);
      }
      if (body.kind === 'angle') {
        if (body.points_world?.length !== 3) return sendJson(res, 422, { error: 'bad_points' });
        const [a, b, c] = body.points_world as [number[], number[], number[]];
        const v1 = [a[0]! - b[0]!, a[1]! - b[1]!, a[2]! - b[2]!];
        const v2 = [c[0]! - b[0]!, c[1]! - b[1]!, c[2]! - b[2]!];
        const dot = v1[0]! * v2[0]! + v1[1]! * v2[1]! + v1[2]! * v2[2]!;
        const deg =
          (Math.acos(Math.min(1, Math.max(-1, dot / (Math.hypot(...v1) * Math.hypot(...v2))))) * 180) / Math.PI;
        const out: MeasureResponse = {
          kind: 'angle',
          value: deg,
          units: anchored ? 'm' : 'relative',
          scale_factor: anchored ? scale : null,
          scale_source: scaleSource,
          degrees: deg,
        };
        return sendJson(res, 200, out);
      }
      return sendJson(res, 422, { error: 'bad_kind' });
    }

    // nav plan
    if (rest[0] === 'nav' && rest[1] === 'plan' && method === 'POST') {
      const body = JSON.parse((await readBody(req)).toString('utf8') || '{}') as {
        goal?: { object_uid?: string; point_world?: number[] };
        start?: number[];
      };
      let goal: number[] | null = body.goal?.point_world ?? null;
      if (body.goal?.object_uid) {
        const idx = Number(body.goal.object_uid.replace(/^det:/, ''));
        const obj = scene.facts.objects[idx];
        if (!obj) return sendJson(res, 422, { error: 'unreachable_goal', message: `unknown object_uid ${body.goal.object_uid}` });
        goal = obj.center;
      }
      if (!goal) return sendJson(res, 422, { error: 'no_goal' });
      const start = body.start ?? scene.scene_center;
      // One waypoint + one pose per frame at 10 fps, like the real planner — a plan is a
      // resampled trajectory, not a corner list, so it is long enough to need trimming.
      const n = 40;
      const waypoints = Array.from({ length: n + 1 }, (_, i) => [
        start[0]! + ((goal![0]! - start[0]!) * i) / n,
        scene.facts.metrics.floor_height ?? 0,
        start[2]! + ((goal![2]! - start[2]!) * i) / n,
      ]) as [number, number, number][];
      state.seq.path += 1;
      const pathId = `path-${String(state.seq.path).padStart(3, '0')}`;
      const key = `derived/demo/nav/paths/${pathId}.json`;
      const length = dist3(start, goal) * scale;
      const out: NavPlanResponse = {
        path_id: pathId,
        doc_key: key,
        waypoints_world: waypoints,
        poses: waypoints.map((w, i) => ({
          c2w: [
            [1, 0, 0, w[0]],
            [0, 1, 0, (scene.facts.metrics.floor_height ?? 0) + 0.35],
            [0, 0, 1, w[2]],
            [0, 0, 0, 1],
          ],
          t: i / 10,
        })),
        floor: {
          point: [0, scene.facts.metrics.floor_height ?? 0, 0],
          normal: [0, 1, 0],
          up: [0, 1, 0],
          up_source: 'pose-derived',
        },
        grid: { n_free: 5210, n_components: 1, n_navigable: 5100, cell_size: 0.05, cell_size_units: anchored ? 'm' : 'relative' },
        units: anchored ? 'm' : 'relative',
        units_basis: anchored ? scaleSource : 'extent_fraction',
        stats: { path_length: length, duration_s: n / 10, n_frames: n + 1, goal_snap: 0, start_snap: 0 },
        notes: ['simulator: straight-line plan over a synthetic occupancy grid'],
        cache: { source: 'built', grid_rebuilt: true },
        provenance: 'Planned in scanned free space of the simulator fixture — not certified navigation.',
      };
      const doc = Buffer.from(JSON.stringify(out), 'utf8');
      bundle.derived.set(key, doc);
      bundle.manifest.paths.push({ key, created_at: nowS() });
      bundle.manifest.updated_at = nowS();
      return sendJson(res, 200, out);
    }

    // planes
    if (rest[0] === 'planes' && method === 'POST') {
      const fh = scene.facts.metrics.floor_height ?? 0;
      const out: PlanesResponse = {
        planes: [
          {
            id: 'plane-floor',
            kind: 'floor',
            point: [0, fh, 0],
            normal: [0, 1, 0],
            u_axis: [1, 0, 0],
            v_axis: [0, 0, 1],
            uv_bounds: [[-2, 2.3], [-1.5, 1.7]],
            thickness: 0.03,
            inlier_count: 210_000,
          },
          {
            id: 'plane-wall-1',
            kind: 'wall',
            point: [2.3, 1.2, 0],
            normal: [-1, 0, 0],
            u_axis: [0, 0, 1],
            v_axis: [0, 1, 0],
            uv_bounds: [[-1.4, 1.7], [0, 2.4]],
            thickness: 0.04,
            inlier_count: 88_000,
          },
        ],
        up: [0, 1, 0],
        up_source: 'pose-derived',
        provenance: 'RANSAC over the persisted cloud (simulator)',
        cached: bundle.planesCached,
      };
      bundle.planesCached = true;
      return sendJson(res, 200, out);
    }

    // ground frame
    if (rest[0] === 'ground_frame') {
      const m = scene.facts.metrics;
      const frame = {
        vertical_axis_known: m.vertical_axis_known,
        derivation: m.derivation ?? 'plane-fit',
        up_axis: m.up_axis ?? [0, 1, 0],
        floor_height: m.floor_height ?? null,
        ceiling_height: m.ceiling_height ?? null,
        room_height: m.room_height ?? null,
        floor_extent: m.floor_extent ?? null,
        floor_area: m.floor_area ?? null,
        units: (anchored ? 'm' : 'relative') as 'm' | 'relative',
        units_basis: anchored ? scaleSource : 'extent_fraction',
      };
      if (method === 'GET') {
        const out: GroundFrameResponse = { scan_id: scanId, frame, computed: false };
        return sendJson(res, 200, out);
      }
      const body = JSON.parse((await readBody(req)).toString('utf8') || '{}') as { dry_run?: boolean };
      const out: GroundFrameResponse = {
        scan_id: scanId,
        frame: { ...frame, floor_inliers: 210_000, floor_inlier_ratio: 0.31, up_source: 'plane-fit' },
        computed: true,
        persisted: !body.dry_run,
      };
      return sendJson(res, 200, out);
    }

    // segment / complete / variants
    if (rest[0] === 'segment' && method === 'POST') {
      const uid = `sel:${Date.now()}`;
      const maskKey = `derived/demo/objects/${uid.replace(':', '_')}/mask.png`;
      bundle.derived.set(maskKey, PNG_1PX);
      return sendJson(res, 200, {
        uid,
        obb: {
          center: [0.5, 0.4, 0.5],
          extents: [0.4, 0.4, 0.4],
          rotation: [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
          ],
        },
        n_points: 4200,
        mask_key: maskKey,
        prompt_source: 'indexed_keyframe',
      });
    }
    if (rest[0] === 'objects' && rest[1] && rest[2] === 'complete' && method === 'POST') {
      state.seq.job += 1;
      const jobId = `sjob-${String(state.seq.job).padStart(3, '0')}`;
      const uidKey = decodeURIComponent(rest[1]).replace(':', '_');
      const jobs = state.sceneJobs.get(scanId) ?? new Map<string, SimJob>();
      jobs.set(jobId, {
        idx: 0,
        doneFired: false,
        states: [
          { job_id: jobId, status: 'queued' },
          { job_id: jobId, status: 'running' },
          { job_id: jobId, status: 'done', result: { asset_key: `derived/demo/objects/${uidKey}/completed/asset.glb` } },
        ],
        onDone: () => {
          bundle.derived.set(`derived/demo/objects/${uidKey}/completed/asset.glb`, Buffer.from('glTF-stub'));
          bundle.derived.set(
            `derived/demo/objects/${uidKey}/completed/meta.json`,
            Buffer.from(JSON.stringify({ provenance: 'simulator', generated: true, generator: 'sam3d' })),
          );
        },
      });
      state.sceneJobs.set(scanId, jobs);
      return sendJson(res, 202, { job_id: jobId });
    }
    if (rest[0] === 'objects' && rest[1] && rest[2] === 'variants' && method === 'POST') {
      state.seq.job += 1;
      const jobId = `sjob-${String(state.seq.job).padStart(3, '0')}`;
      const jobs = state.sceneJobs.get(scanId) ?? new Map<string, SimJob>();
      jobs.set(jobId, {
        idx: 0,
        doneFired: false,
        states: [
          { job_id: jobId, status: 'queued' },
          { job_id: jobId, status: 'done', result: { variants: 1 } },
        ],
      });
      state.sceneJobs.set(scanId, jobs);
      return sendJson(res, 202, { job_id: jobId });
    }
    if (rest[0] === 'objects' && rest[1] && rest[2] === 'floor_patch' && method === 'POST') {
      const uidKey = decodeURIComponent(rest[1]).replace(':', '_');
      const patchKey = `derived/demo/objects/${uidKey}/floor_patch/patch.ply`;
      const metaKey = `derived/demo/objects/${uidKey}/floor_patch/meta.json`;
      bundle.derived.set(patchKey, PLY_STUB);
      bundle.derived.set(metaKey, Buffer.from(JSON.stringify({ generated: true, generator: 'planner' })));
      return sendJson(res, 200, { patch_key: patchKey, meta_key: metaKey, n_gaussians: 1200 });
    }

    // imported objects
    if (rest[0] === 'imported_objects') {
      if (method === 'GET') {
        return sendJson(res, 200, {
          scan_id: scanId,
          status: 'none',
          eligible: scene.source === 'imported_splat',
          reason: scene.source === 'imported_splat' ? undefined : 'scene is not an imported splat',
        });
      }
      if (scene.source !== 'imported_splat') return sendJson(res, 409, { error: 'not_imported_splat' });
      return sendJson(res, 200, {
        scan_id: scanId,
        run_id: `imp-${Date.now()}`,
        count: 2,
        replaced: 0,
        objects: scene.facts.objects,
        labels: { mode: 'geometry', attempted: 2, labelled: 2, declined: 0, errors: 0 },
        provenance: 'geometry clustering (simulator)',
        caveats: ['labels are size-based, not detections'],
        up: [0, 1, 0],
        up_source: 'heuristic',
        diagnostics: {},
        rejected: {},
        derived_key: 'derived/demo/objects/index.json',
      });
    }

    // lod
    if (rest[0] === 'lod') {
      if (method === 'GET') {
        const out: OreosLodResponse = {
          scan_id: scanId,
          status: bundle.lod === 'ready' ? 'ready' : bundle.lod === 'running' ? 'running' : 'none',
          index:
            bundle.lod === 'ready'
              ? {
                  version: 1,
                  scan_id: scanId,
                  source: { key: 'splat.ply', gaussians: 18_850_000, bytes: 1_190_000_000 },
                  default_level: 2_000_000,
                  levels: [
                    {
                      name: '2000k',
                      budget: 2_000_000,
                      gaussians: 1_985_412,
                      key: 'derived/demo/lod/splat_2000k.ply',
                      bytes: 125_000_000,
                      spz_key: 'derived/demo/lod/splat_2000k.spz',
                      spz_bytes: 13_800_000,
                    },
                  ],
                  provenance: 'voxel decimation (simulator)',
                  generated: true,
                  generator: 'lod',
                  caveats: [],
                }
              : null,
        };
        return sendJson(res, 200, out);
      }
      if (bundle.lod === 'running') return sendJson(res, 409, { error: 'lod_job_active' });
      bundle.lod = 'running';
      state.seq.job += 1;
      const jobId = `job-${String(state.seq.job).padStart(3, '0')}`;
      state.wsJobs.set(jobId, {
        idx: 0,
        doneFired: false,
        states: [
          { job_id: jobId, status: 'queued', scan_id: scanId },
          { job_id: jobId, status: 'running', scan_id: scanId },
          { job_id: jobId, status: 'done', scan_id: scanId },
        ],
        onDone: () => {
          bundle.lod = 'ready';
          bundle.derived.set('derived/demo/lod/splat_2000k.ply', PLY_STUB);
          bundle.derived.set('derived/demo/lod/splat_2000k.spz', PLY_STUB);
          bundle.derived.set('derived/demo/lod/index.json', Buffer.from('{}'));
        },
      });
      return sendJson(res, 202, { job_id: jobId });
    }

    // scene jobs
    if (rest[0] === 'jobs' && rest[1] && method === 'GET') {
      const job = state.sceneJobs.get(scanId)?.get(rest[1]);
      if (!job) return sendJson(res, 404, { error: 'job_not_found' });
      return sendJson(res, 200, advance(job));
    }

    // ── demo namespace: agent + export + doc ──
    if (rest[0] === 'demo' && rest[1] === 'agent') {
      const runs = state.runs.get(scanId) ?? new Map<string, SimRun>();
      state.runs.set(scanId, runs);

      const startRun = (kind: SimRun['kind'], events: OreosAgentEvent[], opts?: Partial<SimRun>): SimRun => {
        state.seq.run += 1;
        const run: SimRun = {
          run_id: `run-${String(state.seq.run).padStart(3, '0')}`,
          kind,
          events,
          released: 0,
          perPoll: 3,
          started_at: nowS(),
          replay: false,
          cost_usd: kind === 'replay' ? 0 : 0.0123,
          finishedRecorded: false,
          ...opts,
        };
        runs.set(run.run_id, run);
        return run;
      };

      if (rest[2] === 'annotate' && method === 'POST') {
        const body = JSON.parse((await readBody(req)).toString('utf8') || '{}') as {
          mode?: string;
          replay_of?: string;
          attach_if_active?: boolean;
        };
        const active = activeRun(runs);
        if (active && !body.replay_of) {
          if (body.attach_if_active) return sendJson(res, 202, { run_id: active.run_id, attached: true });
          return sendJson(res, 409, { error: 'agent_run_active', active_run_id: active.run_id });
        }
        if (body.replay_of) {
          const src = runs.get(body.replay_of);
          if (!src) return sendJson(res, 404, { error: 'run_not_found' });
          const run = startRun('replay', src.events.map((e) => ({ ...e })), {
            replay: true,
            source_run_id: src.run_id,
          });
          return sendJson(res, 202, { run_id: run.run_id, replay: true, source_run_id: src.run_id });
        }
        const run = startRun('annotate', annotateScript(scanId, scene));
        return sendJson(res, 202, { run_id: run.run_id });
      }

      if (rest[2] === 'chat' && method === 'POST') {
        const body = JSON.parse((await readBody(req)).toString('utf8') || '{}') as { message?: string };
        const active = activeRun(runs);
        if (active) return sendJson(res, 409, { error: 'agent_run_active', active_run_id: active.run_id });
        const run = startRun('chat', chatScript(scanId, scene, body.message ?? ''));
        return sendJson(res, 202, { run_id: run.run_id });
      }

      if (rest[2] === 'pilot' && method === 'POST') {
        const body = JSON.parse((await readBody(req)).toString('utf8') || '{}') as { instruction?: string };
        const active = activeRun(runs);
        if (active) return sendJson(res, 409, { error: 'agent_run_active', active_run_id: active.run_id });
        const run = startRun('pilot', pilotScript(scanId, scene, body.instruction ?? ''));
        return sendJson(res, 202, { run_id: run.run_id });
      }

      if (rest[2] === 'runs' && rest.length === 3 && method === 'GET') {
        const items: AgentRunListItem[] = [...runs.values()].map((r) => ({
          run_id: r.run_id,
          kind: r.kind,
          status: r.released >= r.events.length ? 'done' : 'running',
          replay: r.replay,
          started_at: r.started_at,
          n_events: r.events.length,
          cost_usd: r.cost_usd,
          llm_calls: r.released >= r.events.length ? 3 : null,
          source_run_id: r.source_run_id ?? null,
          persisted: true,
        }));
        items.sort((a, b) => b.started_at - a.started_at);
        return sendJson(res, 200, { runs: items, active_run_id: activeRun(runs)?.run_id ?? null });
      }

      if (rest[2] === 'runs' && rest[3] && rest[4] === 'events' && method === 'GET') {
        const run = runs.get(rest[3]);
        if (!run) return sendJson(res, 404, { error: 'run_not_found' });
        const after = Number(url.searchParams.get('after') ?? 0);
        run.released = Math.min(run.released + run.perPoll, run.events.length);
        const visible = run.events.filter((e) => e.seq > after && e.seq <= run.events[run.released - 1]!.seq);
        const status = run.released >= run.events.length ? 'done' : 'running';
        if (status === 'done' && !run.finishedRecorded) {
          run.finishedRecorded = true;
          const key = `derived/demo/agent_runs/${run.run_id}/events.json`;
          bundle.derived.set(key, Buffer.from(JSON.stringify(run.events), 'utf8'));
          if (run.kind === 'annotate' || run.kind === 'pilot') {
            bundle.manifest.agent_runs.push({
              run_id: run.run_id,
              kind: run.kind,
              created_at: run.started_at,
              events_key: key,
            });
            bundle.manifest.updated_at = nowS();
          }
        }
        const next = visible.length ? visible[visible.length - 1]!.seq : after;
        const out: AgentRunEventsResponse = {
          events: visible,
          status,
          next,
          run_meta: { run_id: run.run_id, replay: run.replay, started_at: run.started_at, model: 'simulator', cost_usd: run.cost_usd },
          error: null,
          ...(run.replay ? { replay: true } : {}),
        };
        return sendJson(res, 200, out);
      }
      return sendJson(res, 404, { error: 'not_found' });
    }

    if (rest[0] === 'demo' && rest[1] === 'export') {
      const format = url.searchParams.get('format') ?? 'openreality';
      const sourceKey = url.searchParams.get('source') ?? 'original';
      const preparedKey = `${scanId}:${format}:${sourceKey}`;
      const metricBlocked = format === 'isaac_usd' && !anchored;

      if (rest[2] === 'manifest' && method === 'GET') {
        const tree = exportTree(scene, format);
        const out: ExportManifestResponse = {
          scan_id: scanId,
          format: format as ExportManifestResponse['format'],
          source_key: sourceKey,
          complete: !metricBlocked,
          absent: metricBlocked
            ? [{ component: 'metric scale', reason: 'isaac_usd is metric-gated: anchor the scene first' }]
            : [],
          tree,
          file_count: tree.length,
          total_bytes: tree.reduce((s, e) => s + e.size, 0),
          info: { generator: 'simulator' },
          episode: null,
          modality: format === 'groot_lerobot_v2' ? { rgb: 'keyframes' } : null,
          episodes: null,
          tasks: null,
          zip_available: !metricBlocked,
          zip_blocked_reason: metricBlocked ? 'metric_gate: no anchor on this scene' : null,
          ...(format === 'isaac_usd' && anchored
            ? { scale: { scale, scale_source: scaleSource, anchor_scale: scale } }
            : {}),
        };
        return sendJson(res, 200, out);
      }

      if (rest[2] === 'prepare' && method === 'POST') {
        const body = JSON.parse((await readBody(req)).toString('utf8') || '{}') as {
          format?: string;
          source?: string;
        };
        const fmt = body.format ?? format;
        const pKey = `${scanId}:${fmt}:${body.source ?? 'original'}`;
        if (fmt === 'isaac_usd' && !anchored) {
          return sendJson(res, 422, { error: 'export_blocked', message: 'metric_gate: no anchor on this scene' });
        }
        const existing = state.prepared.get(pKey);
        if (existing?.status === 'running') {
          return sendJson(res, 409, { error: 'export_job_active', job_id: existing.job_id });
        }
        state.seq.job += 1;
        const jobId = `job-${String(state.seq.job).padStart(3, '0')}`;
        const tree = exportTree(scene, fmt);
        const treeBytes = tree.reduce((s, e) => s + e.size, 0);
        state.prepared.set(pKey, { status: 'running', job_id: jobId });
        state.wsJobs.set(jobId, {
          idx: 0,
          doneFired: false,
          states: [
            { job_id: jobId, status: 'queued', scan_id: scanId },
            { job_id: jobId, status: 'running', stage: 'report', scan_id: scanId, tree_bytes: treeBytes },
            { job_id: jobId, status: 'running', stage: 'persist', scan_id: scanId, tree_bytes: treeBytes },
            { job_id: jobId, status: 'done', scan_id: scanId, tree_bytes: treeBytes, file_count: tree.length },
          ],
          onDone: () => {
            const zip = buildZip(tree.map((e) => ({ path: e.path, data: Buffer.alloc(Math.min(e.size, 64), 0x5a) })));
            const key = `derived/demo/export/${fmt}.zip`;
            bundle.derived.set(key, zip);
            state.prepared.set(pKey, {
              status: 'ready',
              job_id: jobId,
              artifact: {
                format: fmt,
                key,
                source_key: body.source ?? 'original',
                bytes: zip.length,
                tree_bytes: treeBytes,
                file_count: tree.length,
                built_at: nowS(),
                age_s: 0,
                build_seconds: 1.2,
                download_path: `/api/scenes/${scanId}/derived/demo/export/${fmt}.zip`,
                download_filename: `${scanId}_${fmt}.zip`,
              },
            });
          },
        });
        return sendJson(res, 202, { job_id: jobId, status: 'queued', scan_id: scanId, format: fmt, source_key: body.source ?? 'original', queued_at: nowS() });
      }

      if (rest[2] === 'prepared' && method === 'GET') {
        const p = state.prepared.get(preparedKey);
        if (!p) return sendJson(res, 200, { status: 'none' });
        return sendJson(res, 200, { status: p.status, ...(p.artifact ? { artifact: p.artifact } : {}), job_id: p.job_id });
      }
      return sendJson(res, 404, { error: 'not_found' });
    }

    if (rest[0] === 'demo' && rest[1] === 'doc' && method === 'PUT') {
      const body = JSON.parse((await readBody(req)).toString('utf8') || '{}') as {
        key_suffix?: string;
        json?: Record<string, unknown>;
      };
      if (!body.key_suffix) return sendJson(res, 422, { error: 'missing_key_suffix' });
      const key = `derived/demo/${body.key_suffix}`;
      bundle.derived.set(key, Buffer.from(JSON.stringify(body.json ?? {}), 'utf8'));
      return sendJson(res, 200, { key });
    }

    return sendJson(res, 404, { error: 'not_found' });
  }

  return new Promise((resolve) => {
    server.listen(opts.port ?? 0, '127.0.0.1', () => {
      const addr = server.address() as AddressInfo;
      resolve({
        url: `http://127.0.0.1:${addr.port}`,
        server,
        close: () => new Promise<void>((r) => server.close(() => r())),
      });
    });
  });
}
