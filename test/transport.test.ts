/**
 * Transport-resilience contract: transient network failures must not surface as
 * tool failures for reads, and one blip must not abort a long job wait.
 * Motivated by a measured 5.6% tool-failure day (EXP-46, 2026-08-18): undici
 * "fetch failed" on stale keep-alive sockets after broker recycles, and a
 * healthy 30-minute workspace_job_wait dying to a single failed poll GET.
 */
import { describe, expect, it } from 'vitest';
import { ApiClient } from '../src/http.js';
import type { Session } from '../src/session.js';
import { pollWorkspaceJob, progressTick } from '../src/toolkit.js';

const session = {
  token: async () => 'test-token',
  refresh: async () => false,
  isApiKeyCredential: () => true,
} as unknown as Session;

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

describe('ApiClient.send transport retry', () => {
  it('retries a GET whose socket dies and succeeds on the next attempt', async () => {
    let calls = 0;
    const client = new ApiClient('http://broker', session, (async () => {
      calls += 1;
      if (calls === 1) throw new TypeError('fetch failed');
      return jsonResponse({ ok: true });
    }) as typeof fetch);
    await expect(client.json('GET', '/api/x')).resolves.toEqual({ ok: true });
    expect(calls).toBe(2);
  });

  it('gives up on a GET after three attempts with a diagnosable error', async () => {
    let calls = 0;
    const client = new ApiClient('http://broker', session, (async () => {
      calls += 1;
      throw new TypeError('fetch failed');
    }) as typeof fetch);
    await expect(client.json('GET', '/api/x')).rejects.toThrow(/network failure on GET \/api\/x \(3 attempts\): fetch failed/);
    expect(calls).toBe(3);
  });

  it('never retries a POST (bodies may be one-shot streams; routes may not be idempotent)', async () => {
    let calls = 0;
    const client = new ApiClient('http://broker', session, (async () => {
      calls += 1;
      throw new TypeError('fetch failed');
    }) as typeof fetch);
    await expect(client.json('POST', '/api/x', { a: 1 })).rejects.toThrow(/not retried/);
    expect(calls).toBe(1);
  });

  it('does not retry HTTP error statuses (they are answers, not blips)', async () => {
    let calls = 0;
    const client = new ApiClient('http://broker', session, (async () => {
      calls += 1;
      return jsonResponse({ error: 'not_found' }, 404);
    }) as typeof fetch);
    await expect(client.json('GET', '/api/x')).rejects.toMatchObject({ status: 404 });
    expect(calls).toBe(1);
  });
});

describe('pollWorkspaceJob resilience', () => {
  it('survives transient poll failures and still reaches the terminal state', async () => {
    const script = [
      () => jsonResponse({ job_id: 'j', status: 'running', stage: 'recon' }),
      () => {
        throw new TypeError('fetch failed');
      },
      () => jsonResponse({ job_id: 'j', status: 'oops', stage: '' }, 503),
      () => jsonResponse({ job_id: 'j', status: 'done', stage: 'done' }),
    ];
    let calls = 0;
    // Each thrown fetch is retried 3x by send(); make the whole burst fail by
    // throwing for every attempt while the script step is the "outage".
    const client = new ApiClient('http://broker', session, (async () => {
      const step = script[Math.min(calls, script.length - 1)];
      calls += 1;
      return step();
    }) as typeof fetch);
    const outcome = await pollWorkspaceJob(client, 'j', 30, 0.01);
    expect(outcome.timed_out).toBe(false);
    expect(outcome.final?.status).toBe('done');
    expect(outcome.trail).toContain('poll-retry');
  });

  it('throws immediately on a semantic 4xx (unknown job is an answer)', async () => {
    const client = new ApiClient('http://broker', session, (async () =>
      jsonResponse({ error: 'job_not_found' }, 404)) as typeof fetch);
    await expect(pollWorkspaceJob(client, 'j', 5, 0.01)).rejects.toMatchObject({ status: 404 });
  });

  it('times out with final:null when no poll ever succeeded', async () => {
    const client = new ApiClient('http://broker', session, (async () =>
      jsonResponse({ raw: 'unavailable' }, 503)) as typeof fetch);
    const outcome = await pollWorkspaceJob(client, 'j', 0.03, 0.02);
    expect(outcome.timed_out).toBe(true);
    expect(outcome.final).toBeNull();
  });
});

describe('progressTick', () => {
  it('is inert without a progressToken', () => {
    expect(progressTick(undefined)).toBeUndefined();
    expect(progressTick({ _meta: {} })).toBeUndefined();
  });

  it('reports elapsed seconds against the token on each tick', async () => {
    const sent: unknown[] = [];
    const tick = progressTick({
      _meta: { progressToken: 't1' },
      sendNotification: async (n) => {
        sent.push(n);
      },
    })!;
    await tick({ job_id: 'j', status: 'running', stage: 'recon' } as never, 12.4);
    expect(sent).toHaveLength(1);
    expect(sent[0]).toMatchObject({
      method: 'notifications/progress',
      params: { progressToken: 't1', progress: 12.4, message: 'recon (12s)' },
    });
  });

  it('swallows notification failures (progress must never kill the wait)', async () => {
    const tick = progressTick({
      _meta: { progressToken: 't1' },
      sendNotification: async () => {
        throw new Error('closed');
      },
    })!;
    await expect(tick(null, 1)).resolves.toBeUndefined();
  });
});
