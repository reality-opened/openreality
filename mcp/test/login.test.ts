/**
 * Unit tests for the browser-login loopback callback server (src/login.ts) — plain
 * HTTP requests against a real (but ephemeral, localhost-only) server, no real
 * browser and no real broker involved.
 */
import { describe, expect, it, vi } from 'vitest';
import { EventEmitter } from 'node:events';
import { openBrowser, randomState, startLoopbackServer } from '../src/login.js';

const spawnMock = vi.hoisted(() => vi.fn());
vi.mock('node:child_process', () => ({ spawn: spawnMock }));

describe('openBrowser', () => {
  it('survives a missing opener binary (async ENOENT arrives as an error EVENT on a headless box)', async () => {
    const child = new EventEmitter() as unknown as { unref: () => void; emit: (e: string, err: Error) => boolean };
    (child as { unref: () => void }).unref = () => {};
    spawnMock.mockReturnValue(child);
    openBrowser('https://example.test/cli-auth');
    // Without a handler attached, this emit throws synchronously (and in the real
    // CLI crashes the process, killing the loopback listener) — the regression.
    child.emit('error', new Error('spawn xdg-open ENOENT'));
    expect(spawnMock).toHaveBeenCalledOnce();
  });
});

describe('randomState', () => {
  it('generates a base64url nonce (no padding/URL-unsafe characters) that differs each call', () => {
    const state = randomState();
    expect(state.length).toBeGreaterThan(20);
    expect(state).toMatch(/^[A-Za-z0-9_-]+$/);
    expect(randomState()).not.toBe(randomState());
  });
});

describe('startLoopbackServer', () => {
  it('resolves result with the token on a matching-state callback, and serves a close-this-tab page', async () => {
    const loopback = await startLoopbackServer('state-abc', 5_000);
    try {
      const res = await fetch(
        `http://127.0.0.1:${loopback.port}/callback?state=state-abc&token=clerk-jwt-xyz`,
      );
      expect(res.status).toBe(200);
      const html = await res.text();
      expect(html).toMatch(/close this tab/i);
      await expect(loopback.result).resolves.toBe('clerk-jwt-xyz');
    } finally {
      await loopback.close();
    }
  });

  it('rejects a state-mismatched callback (400) without settling the result, then accepts the real one', async () => {
    const loopback = await startLoopbackServer('state-real', 5_000);
    try {
      const bad = await fetch(
        `http://127.0.0.1:${loopback.port}/callback?state=state-wrong&token=attacker-token`,
      );
      expect(bad.status).toBe(400);

      const good = await fetch(
        `http://127.0.0.1:${loopback.port}/callback?state=state-real&token=real-token`,
      );
      expect(good.status).toBe(200);
      await expect(loopback.result).resolves.toBe('real-token');
    } finally {
      await loopback.close();
    }
  });

  it('400s a callback missing the token param, then accepts the real callback', async () => {
    const loopback = await startLoopbackServer('state-notok', 5_000);
    try {
      const noToken = await fetch(`http://127.0.0.1:${loopback.port}/callback?state=state-notok`);
      expect(noToken.status).toBe(400);

      const good = await fetch(
        `http://127.0.0.1:${loopback.port}/callback?state=state-notok&token=tok-1`,
      );
      expect(good.status).toBe(200);
      await expect(loopback.result).resolves.toBe('tok-1');
    } finally {
      await loopback.close();
    }
  });

  it('times out cleanly (rejecting result) when no callback ever arrives, and close() is idempotent after', async () => {
    const loopback = await startLoopbackServer('state-timeout', 50);
    await expect(loopback.result).rejects.toThrow(/timed out/i);
    await loopback.close(); // server already self-closed on timeout — must not throw/hang
  });

  it('ignores requests to any other path', async () => {
    const loopback = await startLoopbackServer('state-other', 5_000);
    try {
      const res = await fetch(`http://127.0.0.1:${loopback.port}/not-callback`);
      expect(res.status).toBe(404);
    } finally {
      await loopback.close();
    }
  });
});
