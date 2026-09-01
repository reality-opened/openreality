export const DEFAULT_SCANS = 2;

/** Claims carried by a read-only scene share token (embed delivery). The server signs
 *  these as an HS256 JWT under a distinct issuer; the embed page treats the token as an
 *  opaque capability and does not verify it client-side. See rest.ts + embed-delivery.md. */
export interface ShareTokenClaims {
  /** The single scan this token grants read access to. */
  scan_id: string;
  /** Always `'scene:read'` — read + Q&A only, one scan. */
  scope: 'scene:read';
  /** The owning Clerk user id, carried so the server's user-scoped lookups resolve. */
  sub: string;
  /** Expiry, epoch seconds. */
  exp: number;
}

type AtobLike = (input: string) => string;
type TextDecoderLike = new () => { decode(input: Uint8Array): string };
type URLLike = { protocol: string; origin: string };
type URLConstructorLike = new (input: string) => URLLike;
type BufferLike = {
  from(input: string, encoding: 'base64'): Uint8Array;
};

function decodeBase64ToBytes(base64: string): Uint8Array {
  const atobFn = (globalThis as { atob?: AtobLike }).atob;
  if (typeof atobFn === 'function') {
    const binary = atobFn(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) {
      bytes[i] = binary.charCodeAt(i);
    }
    return bytes;
  }

  const buffer = (globalThis as unknown as { Buffer?: BufferLike }).Buffer;
  if (buffer) {
    return new Uint8Array(buffer.from(base64, 'base64'));
  }

  throw new Error('No base64 decoder is available in this runtime.');
}

function decodeUtf8(bytes: Uint8Array): string {
  const TextDecoderCtor = (globalThis as { TextDecoder?: TextDecoderLike }).TextDecoder;
  if (TextDecoderCtor) {
    return new TextDecoderCtor().decode(bytes);
  }

  let escaped = '';
  for (const byte of bytes) {
    escaped += `%${byte.toString(16).padStart(2, '0')}`;
  }
  return decodeURIComponent(escaped);
}

export function decodeJwtPayload(token: string): Record<string, unknown> | null {
  const parts = token.split('.');
  if (parts.length < 2 || !parts[1]) {
    return null;
  }

  try {
    let base64 = parts[1].replace(/-/g, '+').replace(/_/g, '/');
    const pad = base64.length % 4;
    if (pad) {
      base64 += '='.repeat(4 - pad);
    }
    return JSON.parse(decodeUtf8(decodeBase64ToBytes(base64))) as Record<string, unknown>;
  } catch {
    return null;
  }
}

export function getAuthorizedPartyOrigin(token: string | null): string | null {
  if (!token) {
    return null;
  }

  const payload = decodeJwtPayload(token);
  const azp = payload?.azp;
  if (typeof azp !== 'string' || !azp.trim()) {
    return null;
  }

  try {
    const URLCtor = (globalThis as { URL?: URLConstructorLike }).URL;
    if (!URLCtor) {
      return null;
    }
    const url = new URLCtor(azp);
    if (url.protocol !== 'http:' && url.protocol !== 'https:') {
      return null;
    }
    return url.origin;
  } catch {
    return null;
  }
}

/**
 * @deprecated Reads the `scansRemaining` JWT claim, which is a lagging mirror of
 * the credit ledger and is no longer minted into broker session tokens at all
 * (a 12-hour token carrying a quota was a 12-hour grant). Use
 * `GET /api/credits/balance` for a live balance. Kept because the Clerk JWT
 * template still carries the claim during the migration.
 */
export function getScansRemainingClaim(token: string | null): number | null {
  if (!token) {
    return DEFAULT_SCANS;
  }

  const payload = decodeJwtPayload(token);
  if (payload?.tier === 'approved') {
    return null;
  }

  const value = payload?.scansRemaining;
  return typeof value === 'number' ? value : DEFAULT_SCANS;
}

/** Durable API key bearer prefix (opaque, ~47 chars total, first 10 chars is the display
 *  `prefix` field). Minted once via `POST /api/keys` after a browser login; revocable
 *  server-side. Unlike a Clerk JWT or broker session token, an API key carries no `exp` —
 *  it is never rolled or refreshed (see rest.ts's API_KEYS routes + Session in the
 *  openreality-mcp repo). */
export const API_KEY_PREFIX = 'ork_' as const;

/** True when `token` looks like a durable Open Reality API key (`ork_...`) rather than a
 *  Clerk JWT or broker session token. A cheap prefix check — the opaque suffix is never
 *  decoded or validated client-side (never throws). */
export function isApiKeyToken(token: string | null | undefined): boolean {
  return typeof token === 'string' && token.startsWith(API_KEY_PREFIX);
}
