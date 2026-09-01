/**
 * The simulator is only useful if it cannot drift from the contract: every
 * response here is validated with @reality/protocol's OWN type guards.
 */
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import {
  OREOS_MANIFEST_KEY,
  OREOS_REST_PATHS,
  REST_PATHS,
  isAgentRunEventsResponse,
  isExportManifestResponse,
  isListApiKeysResponse,
  isMeasureResponse,
  isMintApiKeyResponse,
  isNavPlanResponse,
  isOreosJobStatus,
  isOreosManifest,
  isRevokeApiKeyResponse,
} from '@reality/protocol';
import { startSimulator, type RunningSimulator } from '../simulator/sim.js';
import { FIXTURE_SCAN_ID } from '../simulator/fixture.js';

let sim: RunningSimulator;
const H = { Authorization: 'Bearer test-token' };

async function get(path: string): Promise<unknown> {
  const res = await fetch(sim.url + path, { headers: H });
  return res.json();
}
async function post(path: string, body: unknown): Promise<{ status: number; json: unknown }> {
  const res = await fetch(sim.url + path, {
    method: 'POST',
    headers: { ...H, 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return { status: res.status, json: await res.json() };
}

beforeAll(async () => {
  sim = await startSimulator({ port: 0 });
});
afterAll(async () => {
  await sim.close();
});

describe('simulator speaks the protocol', () => {
  it('lists scenes in the {scenes: []} envelope', async () => {
    const data = (await get(REST_PATHS.SCENES)) as { scenes: unknown[] };
    expect(Array.isArray(data.scenes)).toBe(true);
    expect(data.scenes.length).toBeGreaterThan(0);
  });

  it('serves a manifest that passes isOreosManifest', async () => {
    const data = await get(REST_PATHS.SCENE_DERIVED(FIXTURE_SCAN_ID, OREOS_MANIFEST_KEY));
    expect(isOreosManifest(data)).toBe(true);
  });

  it('measures with a valid MeasureResponse (anchored → metres)', async () => {
    const { json } = await post(OREOS_REST_PATHS.SCENE_MEASURE(FIXTURE_SCAN_ID), {
      kind: 'distance',
      points_world: [
        [0, 0, 0],
        [2, 0, 0],
      ],
    });
    expect(isMeasureResponse(json)).toBe(true);
    expect((json as { units: string }).units).toBe('m');
    expect((json as { value: number }).value).toBeCloseTo(2 * 0.048, 6);
  });

  it('plans a path with a valid NavPlanResponse', async () => {
    const { json } = await post(OREOS_REST_PATHS.SCENE_NAV_PLAN(FIXTURE_SCAN_ID), {
      goal: { object_uid: 'det:1' },
    });
    expect(isNavPlanResponse(json)).toBe(true);
  });

  it('rejects an unknown nav goal honestly (422 unreachable_goal)', async () => {
    const { status, json } = await post(OREOS_REST_PATHS.SCENE_NAV_PLAN(FIXTURE_SCAN_ID), {
      goal: { object_uid: 'det:99' },
    });
    expect(status).toBe(422);
    expect((json as { error: string }).error).toBe('unreachable_goal');
  });

  it('runs ingest jobs through valid OreosJobStatus states to done', async () => {
    const res = await fetch(sim.url + OREOS_REST_PATHS.INGEST_VIDEO, {
      method: 'POST',
      headers: { ...H, 'X-Upload-Filename': 'clip.mp4' },
      body: Buffer.from('stub'),
    });
    expect(res.status).toBe(202);
    const { job_id, scan_id } = (await res.json()) as { job_id: string; scan_id: string };
    let last: unknown;
    for (let i = 0; i < 10; i++) {
      last = await get(OREOS_REST_PATHS.WORKSPACE_JOB(job_id));
      expect(isOreosJobStatus(last)).toBe(true);
      if ((last as { status: string }).status === 'done') break;
    }
    expect((last as { status: string }).status).toBe('done');
    const scene = (await get(REST_PATHS.SCENE_DETAIL(scan_id))) as { scan_id: string; derived_latest: unknown };
    expect(scene.scan_id).toBe(scan_id);
    expect(scene.derived_latest).toBeNull(); // fresh scenes are un-anchored
  });

  it('serves agent events that pass isAgentRunEventsResponse and 409s a second run', async () => {
    const { status, json } = await post(OREOS_REST_PATHS.SCENE_AGENT_CHAT(FIXTURE_SCAN_ID), {
      message: 'hi',
    });
    expect(status).toBe(202);
    const runId = (json as { run_id: string }).run_id;
    const second = await post(OREOS_REST_PATHS.SCENE_AGENT_ANNOTATE(FIXTURE_SCAN_ID), { mode: 'full' });
    expect(second.status).toBe(409);
    expect((second.json as { error: string }).error).toBe('agent_run_active');
    let cursor = 0;
    for (let i = 0; i < 10; i++) {
      const ev = await get(OREOS_REST_PATHS.SCENE_AGENT_RUN_EVENTS(FIXTURE_SCAN_ID, runId, cursor));
      expect(isAgentRunEventsResponse(ev)).toBe(true);
      cursor = (ev as { next: number }).next;
      if ((ev as { status: string }).status !== 'running') break;
    }
  });

  it('gates the Isaac export behind the metric anchor', async () => {
    // upload a fresh (un-anchored) scene
    const res = await fetch(sim.url + OREOS_REST_PATHS.INGEST_VIDEO, {
      method: 'POST',
      headers: { ...H, 'X-Upload-Filename': 'clip2.mp4' },
      body: Buffer.from('stub'),
    });
    const { job_id, scan_id } = (await res.json()) as { job_id: string; scan_id: string };
    for (let i = 0; i < 10; i++) {
      const s = (await get(OREOS_REST_PATHS.WORKSPACE_JOB(job_id))) as { status: string };
      if (s.status === 'done') break;
    }
    const manifest = await get(OREOS_REST_PATHS.SCENE_EXPORT_MANIFEST(scan_id, 'isaac_usd'));
    expect(isExportManifestResponse(manifest)).toBe(true);
    expect((manifest as { zip_available: boolean }).zip_available).toBe(false);
    expect((manifest as { zip_blocked_reason: string | null }).zip_blocked_reason).toMatch(/metric_gate/);
    const prep = await post(OREOS_REST_PATHS.SCENE_EXPORT_PREPARE(scan_id), { format: 'isaac_usd' });
    expect(prep.status).toBe(422);
  });

  it('mints an API key with isMintApiKeyResponse validating the response, and refuses an ork_ bearer minting another (403 api_key_cannot_mint)', async () => {
    const { status, json } = await post(REST_PATHS.API_KEYS, { name: 'claude-mcp@test-host' });
    expect(status).toBe(200);
    expect(isMintApiKeyResponse(json)).toBe(true);
    const minted = json as { key: string; key_id: string; prefix: string };
    expect(minted.key.startsWith('ork_')).toBe(true);
    expect(minted.prefix).toBe(minted.key.slice(0, 10));

    const res = await fetch(sim.url + REST_PATHS.API_KEYS, {
      method: 'POST',
      headers: { Authorization: `Bearer ${minted.key}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    });
    expect(res.status).toBe(403);
    expect(((await res.json()) as { error: string }).error).toBe('api_key_cannot_mint');
  });

  it('403s an ork_-shaped bearer minting even when that key is unregistered (not 401)', async () => {
    const res = await fetch(sim.url + REST_PATHS.API_KEYS, {
      method: 'POST',
      headers: { Authorization: 'Bearer ork_never-minted-by-this-sim-00000000000000', 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    });
    expect(res.status).toBe(403);
    expect(((await res.json()) as { error: string }).error).toBe('api_key_cannot_mint');
  });

  it('lists API keys newest-first (revoked included) with isListApiKeysResponse validating the response', async () => {
    await post(REST_PATHS.API_KEYS, { name: 'first' });
    await post(REST_PATHS.API_KEYS, { name: 'second' });
    const data = await get(REST_PATHS.API_KEYS);
    expect(isListApiKeysResponse(data)).toBe(true);
    const keys = (data as { keys: Array<{ created_at: number }> }).keys;
    expect(keys.length).toBeGreaterThanOrEqual(2);
    for (let i = 1; i < keys.length; i++) {
      expect(keys[i - 1]!.created_at).toBeGreaterThanOrEqual(keys[i]!.created_at);
    }
  });

  it('revokes an API key idempotently with isRevokeApiKeyResponse validating the response; unknown id 404s', async () => {
    const { json: minted } = await post(REST_PATHS.API_KEYS, { name: 'to-revoke' });
    const keyId = (minted as { key_id: string }).key_id;

    const del = await fetch(sim.url + REST_PATHS.API_KEY(keyId), { method: 'DELETE', headers: H });
    const first = await del.json();
    expect(del.status).toBe(200);
    expect(isRevokeApiKeyResponse(first)).toBe(true);

    // idempotent: revoking an already-revoked key returns the same revoked_at, still 200
    const del2 = await fetch(sim.url + REST_PATHS.API_KEY(keyId), { method: 'DELETE', headers: H });
    const second = await del2.json();
    expect(del2.status).toBe(200);
    expect((second as { revoked_at: number }).revoked_at).toBe((first as { revoked_at: number }).revoked_at);

    const missing = await fetch(sim.url + REST_PATHS.API_KEY('key_does_not_exist'), {
      method: 'DELETE',
      headers: H,
    });
    expect(missing.status).toBe(404);
  });

  it('401s an unknown or revoked ork_ bearer on a normal route, and accepts a live minted one', async () => {
    const { json: minted } = await post(REST_PATHS.API_KEYS, { name: 'live-bearer' });
    const key = (minted as { key: string; key_id: string }).key;
    const keyId = (minted as { key: string; key_id: string }).key_id;

    const unknown = await fetch(sim.url + REST_PATHS.SCENES, {
      headers: { Authorization: 'Bearer ork_totally-unknown-0000000000000000000000000' },
    });
    expect(unknown.status).toBe(401);

    const ok = await fetch(sim.url + REST_PATHS.SCENES, { headers: { Authorization: `Bearer ${key}` } });
    expect(ok.status).toBe(200);

    await fetch(sim.url + REST_PATHS.API_KEY(keyId), { method: 'DELETE', headers: H });
    const revoked = await fetch(sim.url + REST_PATHS.SCENES, { headers: { Authorization: `Bearer ${key}` } });
    expect(revoked.status).toBe(401);
  });

  it('401s bad bearers but lets /api/session/refresh roll them', async () => {
    const bad = await fetch(sim.url + REST_PATHS.SCENES, {
      headers: { Authorization: 'Bearer expired-token' },
    });
    expect(bad.status).toBe(401);
    const roll = await fetch(sim.url + REST_PATHS.SESSION_REFRESH, {
      method: 'POST',
      headers: { Authorization: 'Bearer expired-token' },
    });
    expect(roll.status).toBe(200);
    const { token } = (await roll.json()) as { token: string };
    const ok = await fetch(sim.url + REST_PATHS.SCENES, {
      headers: { Authorization: `Bearer ${token}` },
    });
    expect(ok.status).toBe(200);
  });
});
