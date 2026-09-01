/**
 * Headless port of the OS page's session lifecycle (apps/webserver/src/oreos/session.ts):
 * hold a durable broker session token, roll it proactively at half its remaining life via
 * POST /api/session/refresh (which trades a still-valid session token — or a fresh Clerk
 * JWT — for a new 12 h one), and let the HTTP layer force one refresh+retry on a 401.
 *
 * Durable API keys (`ork_...`, minted via POST /api/keys — see cli.ts's `login`) are a
 * SECOND, simpler credential kind this class also holds: they carry no `exp`, so `token()`
 * never rolls them and `refresh()` is a no-op that returns `false` immediately (no network
 * call, no half-life bookkeeping). A revoked/invalid key still 401s like any bad bearer;
 * ApiClient (http.ts) is what turns that into a "run `login` again" message instead of
 * retry-looping — see its `send()`.
 *
 * Credentials live at ~/.config/openreality/credentials.json (0600). A token passed via
 * OPENREALITY_TOKEN / DEMO_AUTH_TOKEN is used as the starting credential and any rolled
 * replacement is persisted to the file.
 */
import { mkdirSync, readFileSync, writeFileSync, chmodSync } from 'node:fs';
import { dirname } from 'node:path';
import { REST_PATHS, decodeJwtPayload, isApiKeyToken } from '@reality/protocol';

interface StoredCredentials {
  token: string;
  /** Epoch seconds. */
  expires_at?: number | null;
  /** Epoch seconds at which we obtained this token (drives the half-life roll). */
  obtained_at?: number | null;
  base_url?: string;
}

interface RefreshResponse {
  token?: string;
  expires_at?: number;
}

export class Session {
  private current: StoredCredentials | null = null;
  private refreshing: Promise<boolean> | null = null;

  constructor(
    private readonly baseUrl: string,
    private readonly credentialsPath: string,
    envToken: string | null = null,
    private readonly fetchImpl: typeof fetch = fetch,
    private readonly now: () => number = () => Date.now() / 1000,
  ) {
    if (envToken) {
      this.current = isApiKeyToken(envToken)
        ? { token: envToken, expires_at: null, obtained_at: null }
        : { token: envToken, expires_at: jwtExp(envToken), obtained_at: this.now() };
    } else {
      this.current = this.load();
    }
  }

  private load(): StoredCredentials | null {
    try {
      const raw = JSON.parse(readFileSync(this.credentialsPath, 'utf8')) as StoredCredentials;
      return typeof raw.token === 'string' && raw.token ? raw : null;
    } catch {
      return null;
    }
  }

  private persist(): void {
    if (!this.current) return;
    try {
      mkdirSync(dirname(this.credentialsPath), { recursive: true });
      writeFileSync(
        this.credentialsPath,
        JSON.stringify({ ...this.current, base_url: this.baseUrl }, null, 2) + '\n',
      );
      chmodSync(this.credentialsPath, 0o600);
    } catch {
      // A read-only home dir must not take the tool surface down with it.
    }
  }

  hasCredentials(): boolean {
    return !!this.current?.token;
  }

  /** True when the held credential is a durable `ork_...` API key rather than a Clerk
   *  JWT / broker session token — static, never rolled or refreshed. */
  isApiKeyCredential(): boolean {
    return !!this.current && isApiKeyToken(this.current.token);
  }

  /** Current bearer. Session/JWT tokens roll first when past half their remaining life;
   *  API keys carry no `exp` and are returned as-is. */
  async token(): Promise<string> {
    if (!this.current?.token) {
      throw new Error(
        'No Open Reality credentials. Run `openreality-mcp login` (browser sign-in) or ' +
          '`openreality-mcp login --token <token>`, or set OPENREALITY_TOKEN.',
      );
    }
    if (isApiKeyToken(this.current.token)) {
      return this.current.token;
    }
    const { token, expires_at, obtained_at } = this.current;
    if (expires_at && obtained_at) {
      const remaining = expires_at - this.now();
      const total = expires_at - obtained_at;
      if (total > 0 && remaining < total / 2) await this.refresh();
    }
    return this.current.token;
  }

  /**
   * Trade the current bearer for a fresh durable token. Single-flight (concurrent 401s
   * collapse onto one refresh, mirroring OreosSession). Returns false when the broker
   * refuses — callers surface the login hint, they don't loop. A no-op for an API key
   * credential (there is nothing to roll): resolves `false` immediately without a
   * network call, so a revoked/invalid key's 401 is never retried.
   */
  refresh(): Promise<boolean> {
    if (this.current && isApiKeyToken(this.current.token)) {
      return Promise.resolve(false);
    }
    if (this.refreshing) return this.refreshing;
    this.refreshing = this.refreshOnce().finally(() => {
      this.refreshing = null;
    });
    return this.refreshing;
  }

  private async refreshOnce(): Promise<boolean> {
    const bearer = this.current?.token;
    if (!bearer) return false;
    try {
      const res = await this.fetchImpl(this.baseUrl + REST_PATHS.SESSION_REFRESH, {
        method: 'POST',
        headers: { Authorization: `Bearer ${bearer}` },
      });
      if (!res.ok) return false;
      const data = (await res.json()) as RefreshResponse;
      if (typeof data.token !== 'string' || !data.token) return false;
      this.current = {
        token: data.token,
        expires_at:
          typeof data.expires_at === 'number' ? data.expires_at : jwtExp(data.token),
        obtained_at: this.now(),
      };
      this.persist();
      return true;
    } catch {
      return false;
    }
  }

  /**
   * `login`: adopt a credential and persist it. An `ork_...` API key is stored as-is —
   * it's already durable, there is nothing to roll. Anything else is treated as a
   * Clerk JWT / session token and rolled into a durable session token when possible
   * (returns whether that roll succeeded; the caller keeps the pasted token otherwise).
   */
  async adopt(token: string): Promise<boolean> {
    if (isApiKeyToken(token)) {
      this.current = { token, expires_at: null, obtained_at: null };
      this.persist();
      return false;
    }
    this.current = { token, expires_at: jwtExp(token), obtained_at: this.now() };
    const rolled = await this.refresh();
    if (!rolled) this.persist(); // keep the pasted token; it may still authorize requests
    return rolled;
  }
}

function jwtExp(token: string): number | null {
  const claims = decodeJwtPayload(token);
  return typeof claims?.exp === 'number' ? claims.exp : null;
}
