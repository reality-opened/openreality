/**
 * Typed HTTP client for the broker — the headless twin of OreosApi
 * (apps/webserver/src/oreos/api.ts): every call carries `Authorization: Bearer`,
 * a 401 triggers exactly one session refresh + retry, and error bodies are
 * surfaced verbatim (the server's {error, message} shapes are part of the
 * contract — 409 agent_run_active carries active_run_id, 422 carries the
 * planner's reason, and the model needs to see them).
 */
import { createReadStream, statSync } from 'node:fs';
import { Readable } from 'node:stream';
import type { BodyInit } from 'undici-types';
import type { Session } from './session.js';

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly body: unknown,
    message?: string,
  ) {
    super(message ?? `HTTP ${status}`);
    this.name = 'ApiError';
  }
}

function parseBody(text: string): unknown {
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    return { raw: text.slice(0, 2000) };
  }
}

export class ApiClient {
  constructor(
    public readonly baseUrl: string,
    private readonly session: Session,
    private readonly fetchImpl: typeof fetch = fetch,
  ) {}

  private async send(
    method: string,
    path: string,
    init: { headers?: Record<string, string>; body?: BodyInit; duplex?: 'half' } = {},
    retried = false,
  ): Promise<Response> {
    const token = await this.session.token();
    // Transport-level failures (undici "fetch failed": a stale keep-alive socket
    // the broker closed, a container recycle mid-connect, a DNS blip) throw before
    // any HTTP status exists. Retry them for GETs only — GETs here carry no body
    // and are safe to repeat, while POST bodies may be one-shot streams
    // (uploadFile) and non-idempotent routes must surface the failure instead.
    const attempts = method === 'GET' ? 3 : 1;
    let res: Response | undefined;
    for (let attempt = 1; ; attempt++) {
      try {
        res = await this.fetchImpl(this.baseUrl + path, {
          method,
          headers: { ...init.headers, Authorization: `Bearer ${token}` },
          body: init.body,
          duplex: init.duplex,
        } as RequestInit);
        break;
      } catch (err) {
        if (attempt >= attempts) {
          const detail = err instanceof Error ? err.message : String(err);
          throw new Error(
            `network failure on ${method} ${path} (${attempts === 1 ? 'not retried' : `${attempts} attempts`}): ${detail}`,
            { cause: err },
          );
        }
        await new Promise((r) => setTimeout(r, 250 * 2 ** (attempt - 1)));
      }
    }
    if (res.status === 401 && !retried) {
      if (await this.session.refresh()) {
        return this.send(method, path, init, true);
      }
      // API keys are static (Session.refresh() no-ops to `false` for them without a
      // network call) — a 401 here means the key was revoked or never existed, not a
      // transient expiry. Don't loop; say so plainly instead of a bare "HTTP 401".
      if (this.session.isApiKeyCredential()) {
        throw new ApiError(
          401,
          parseBody(await res.text()),
          'This Open Reality API key was revoked (or is invalid). Run `openreality-mcp login` again.',
        );
      }
    }
    return res;
  }

  private async finish<T>(res: Response): Promise<T> {
    const parsed = parseBody(await res.text());
    if (!res.ok) throw new ApiError(res.status, parsed);
    return parsed as T;
  }

  /** JSON in/out. Throws ApiError (carrying the parsed body) on any non-2xx. */
  async json<T>(method: string, path: string, body?: unknown): Promise<T> {
    const res = await this.send(method, path, {
      headers: body !== undefined ? { 'Content-Type': 'application/json' } : {},
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
    return this.finish<T>(res);
  }

  /** Raw bytes GET (artifacts, keyframes, view PNGs). */
  async bytes(path: string): Promise<{ bytes: Buffer; contentType: string }> {
    const res = await this.send('GET', path);
    if (!res.ok) throw new ApiError(res.status, parseBody(await res.text()));
    return {
      bytes: Buffer.from(await res.arrayBuffer()),
      contentType: res.headers.get('content-type') ?? 'application/octet-stream',
    };
  }

  /**
   * Stream a local file as a raw octet-stream body (the ingest routes' contract:
   * never multipart, filename in X-Upload-Filename, explicit Content-Length so the
   * server can enforce its byte cap while reading). Mirrors
   * scripts/demo_ingest_gemini.py::post_file_stream.
   */
  async uploadFile<T>(path: string, filePath: string, headers: Record<string, string>): Promise<T> {
    const size = statSync(filePath).size;
    const stream = Readable.toWeb(createReadStream(filePath)) as unknown as BodyInit;
    const res = await this.send('POST', path, {
      headers: {
        ...headers,
        'Content-Type': 'application/octet-stream',
        'Content-Length': String(size),
      },
      body: stream,
      duplex: 'half',
    });
    return this.finish<T>(res);
  }

  /** POST raw bytes from memory (chunked splat upload). */
  async rawPost<T>(path: string, bytes: Buffer, headers: Record<string, string> = {}): Promise<T> {
    const res = await this.send('POST', path, {
      headers: {
        'Content-Type': 'application/octet-stream',
        'Content-Length': String(bytes.byteLength),
        ...headers,
      },
      body: new Uint8Array(bytes),
    });
    return this.finish<T>(res);
  }
}
