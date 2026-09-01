/**
 * Browser-based `openreality-mcp login` (no --token): a loopback HTTP callback server
 * plus a best-effort "open the user's browser" helper. Split out of cli.ts so the
 * callback handler — state verification, happy path, timeout — can be unit-tested
 * directly (test/login.test.ts) without a real browser or a real broker.
 *
 * Flow (cli.ts orchestrates): mint a state nonce → start this server → open
 * `${loginBaseUrl}/cli-auth?port=<port>&state=<state>` (and always print it too) → the
 * dashboard signs the user in and redirects the browser to
 * `http://127.0.0.1:<port>/callback?state=<state>&token=<clerk-jwt>` → we verify the
 * state, hand the short-lived Clerk JWT back to the caller, and show a "you can close
 * this tab" page.
 */
import { randomBytes } from 'node:crypto';
import { createServer, type Server } from 'node:http';
import type { AddressInfo } from 'node:net';
import { spawn } from 'node:child_process';
import { platform } from 'node:os';

/** 32 random bytes, base64url — the CSRF-style state nonce for the loopback callback. */
export function randomState(): string {
  return randomBytes(32).toString('base64url');
}

const SUCCESS_HTML = `<!doctype html>
<html>
<head><title>Open Reality — signed in</title></head>
<body style="font-family: system-ui, sans-serif; text-align: center; padding: 4rem;">
<h1>Signed in.</h1>
<p>openreality-mcp received your credentials. You can close this tab.</p>
</body>
</html>
`;

export interface LoopbackServer {
  /** The ephemeral 127.0.0.1 port the server bound to. */
  port: number;
  /**
   * Settles exactly once: resolves with the callback's `token` on the first
   * state-matching request, or rejects after the timeout if none ever arrives. A
   * state-MISMATCHED request does not settle this — the server just answers it 400
   * and keeps waiting for the real callback.
   */
  result: Promise<string>;
  /** Idempotent — safe to call even after `result` has already settled (which closes
   *  the server itself). */
  close: () => Promise<void>;
}

/**
 * Start a 127.0.0.1 loopback server for exactly one GET /callback?state=&token=.
 */
export function startLoopbackServer(expectedState: string, timeoutMs = 180_000): Promise<LoopbackServer> {
  return new Promise((resolveStart) => {
    let settled = false;
    let resolveResult!: (token: string) => void;
    let rejectResult!: (err: Error) => void;
    const result = new Promise<string>((res, rej) => {
      resolveResult = res;
      rejectResult = rej;
    });

    const finish = (settle: () => void): void => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      settle();
      server.close();
    };

    const server: Server = createServer((req, res) => {
      const url = new URL(req.url ?? '/', 'http://127.0.0.1');
      if (url.pathname !== '/callback') {
        res.writeHead(404, { 'Content-Type': 'text/plain' }).end('not found');
        return;
      }
      const state = url.searchParams.get('state');
      const token = url.searchParams.get('token');
      if (!state || state !== expectedState) {
        // Not the login flow we started (stray hit, or a forged callback) — refuse it
        // but keep listening for the real one.
        res.writeHead(400, { 'Content-Type': 'text/plain' }).end('state mismatch');
        return;
      }
      if (!token) {
        res.writeHead(400, { 'Content-Type': 'text/plain' }).end('missing token');
        return;
      }
      res.writeHead(200, { 'Content-Type': 'text/html' }).end(SUCCESS_HTML);
      finish(() => resolveResult(token));
    });

    const timer = setTimeout(() => {
      finish(() =>
        rejectResult(
          new Error(
            `Timed out after ${Math.round(timeoutMs / 1000)}s waiting for the browser login callback.`,
          ),
        ),
      );
    }, timeoutMs);
    timer.unref?.();

    server.listen(0, '127.0.0.1', () => {
      const { port } = server.address() as AddressInfo;
      resolveStart({
        port,
        result,
        close: () =>
          new Promise<void>((r) => {
            if (!server.listening) {
              r();
              return;
            }
            server.close(() => r());
          }),
      });
    });
  });
}

/**
 * Best-effort open `url` in the platform's default browser (detached, ignored stdio —
 * never blocks or pipes output back to us). The caller must ALWAYS also print the URL:
 * this can silently fail (headless boxes, missing xdg-open, sandboxed CI).
 */
export function openBrowser(url: string): void {
  try {
    const plat = platform();
    const [cmd, args] =
      plat === 'darwin'
        ? ['open', [url]]
        : plat === 'win32'
          ? ['cmd', ['/c', 'start', '""', url]]
          : ['xdg-open', [url]];
    const child = spawn(cmd as string, args as string[], { detached: true, stdio: 'ignore' });
    // spawn failures (ENOENT on a headless box with no xdg-open) arrive as an
    // async 'error' EVENT, not a throw — without a handler it would crash the
    // whole CLI and take the loopback listener down with it.
    child.on('error', () => {});
    child.unref();
  } catch {
    // best-effort only — the printed URL is the real fallback.
  }
}
