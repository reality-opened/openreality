/**
 * Session lifecycle: 401 → one refresh + retry (via ApiClient), and the
 * half-life proactive roll — the headless mirror of OreosSession's behavior.
 */
import { mkdtempSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { REST_PATHS } from '@reality/protocol';
import { Session } from '../src/session.js';
import { ApiClient } from '../src/http.js';
import { startSimulator } from '../simulator/sim.js';

function tempCreds(): string {
  return join(mkdtempSync(join(tmpdir(), 'orl-mcp-')), 'credentials.json');
}

describe('Session', () => {
  it('retries once through a refresh on 401 (expired-token flow)', async () => {
    const sim = await startSimulator({ port: 0 });
    try {
      const session = new Session(sim.url, tempCreds(), 'expired-token');
      const client = new ApiClient(sim.url, session);
      const data = await client.json<{ scenes: unknown[] }>('GET', REST_PATHS.SCENES);
      expect(Array.isArray(data.scenes)).toBe(true); // 401 → refresh → retry succeeded
    } finally {
      await sim.close();
    }
  });

  it('rolls proactively at half the token life and persists the credential', async () => {
    let refreshes = 0;
    let clock = 1000;
    const fakeFetch: typeof fetch = async (url, init) => {
      if (String(url).endsWith(REST_PATHS.SESSION_REFRESH)) {
        refreshes += 1;
        return new Response(JSON.stringify({ token: `rolled-${refreshes}`, expires_at: clock + 1200 }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      throw new Error(`unexpected fetch ${url}`);
    };
    const credsPath = tempCreds();
    const session = new Session('http://broker', credsPath, null, fakeFetch, () => clock);
    await session.adopt('initial-token'); // adopt tries one roll
    expect(refreshes).toBe(1);

    await session.token();
    expect(refreshes).toBe(1); // fresh — no roll

    clock += 700; // past half of the 1200 s life
    const tok = await session.token();
    expect(refreshes).toBe(2);
    expect(tok).toBe('rolled-2');

    const stored = JSON.parse(readFileSync(credsPath, 'utf8')) as { token: string };
    expect(stored.token).toBe('rolled-2');
  });

  it('surfaces a login hint when no credential exists', async () => {
    const session = new Session('http://broker', tempCreds(), null);
    await expect(session.token()).rejects.toThrow(/login/);
  });
});

describe('Session — durable API key (ork_...) credentials', () => {
  const API_KEY = 'ork_abcdefghijklmnopqrstuvwxyz0123456789ABCD';

  it('returns the key as-is from token() — no half-life roll, ever', async () => {
    let fetchCalls = 0;
    const fakeFetch: typeof fetch = async () => {
      fetchCalls += 1;
      throw new Error('token() must never make a network call for an API key credential');
    };
    let clock = 1000;
    const session = new Session('http://broker', tempCreds(), API_KEY, fakeFetch, () => clock);
    expect(await session.token()).toBe(API_KEY);
    clock += 10_000_000; // far past any plausible half-life — still must not roll
    expect(await session.token()).toBe(API_KEY);
    expect(fetchCalls).toBe(0);
    expect(session.isApiKeyCredential()).toBe(true);
  });

  it('refresh() no-ops to false without a network call', async () => {
    let fetchCalls = 0;
    const fakeFetch: typeof fetch = async () => {
      fetchCalls += 1;
      throw new Error('refresh() must never make a network call for an API key credential');
    };
    const session = new Session('http://broker', tempCreds(), API_KEY, fakeFetch);
    await expect(session.refresh()).resolves.toBe(false);
    expect(fetchCalls).toBe(0);
  });

  it('adopt() stores an ork_ token statically (no roll attempt) and persists it', async () => {
    const credsPath = tempCreds();
    let fetchCalls = 0;
    const fakeFetch: typeof fetch = async () => {
      fetchCalls += 1;
      throw new Error('adopt() must never roll an API key');
    };
    const session = new Session('http://broker', credsPath, null, fakeFetch);
    const rolled = await session.adopt(API_KEY);
    expect(rolled).toBe(false);
    expect(fetchCalls).toBe(0);
    const stored = JSON.parse(readFileSync(credsPath, 'utf8')) as { token: string };
    expect(stored.token).toBe(API_KEY);
  });

  it('does not loop a 401 retry through ApiClient, and surfaces a clear revoked-key error', async () => {
    let calls = 0;
    const fakeFetch: typeof fetch = async () => {
      calls += 1;
      return new Response(JSON.stringify({ error: 'unauthorized' }), {
        status: 401,
        headers: { 'Content-Type': 'application/json' },
      });
    };
    const session = new Session('http://broker', tempCreds(), API_KEY, fakeFetch);
    const client = new ApiClient('http://broker', session, fakeFetch);
    await expect(client.json('GET', REST_PATHS.SCENES)).rejects.toThrow(/revoked|login/i);
    expect(calls).toBe(1); // exactly one attempt — no refresh-and-retry loop
  });
});
