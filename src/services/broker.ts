/**
 * `ctx.broker` — the broker capability: resolved runtime config + credential
 * session + typed HTTP client, provided to every tool plugin via cordis
 * injection. Feature plugins declare `inject: ['broker']` and never construct
 * their own client, so alternate compositions (tests, the simulator, a future
 * HTTP transport) swap the backend by mounting Broker with different config.
 *
 * The config is validated at mount (Standard Schema via zod) — a bad
 * composition fails loud at load, not on the first tool call.
 */
import { Context, Service } from 'cordis';
import { z } from 'zod';
import { Session } from '../session.js';
import { ApiClient } from '../http.js';
import type { Config } from '../config.js';

declare module 'cordis' {
  interface Context {
    broker: Broker;
  }
}

const configSchema = z.object({
  baseUrl: z.string().min(1),
  loginBaseUrl: z.string().min(1),
  envToken: z.string().nullable(),
  configDir: z.string().min(1),
  credentialsPath: z.string().min(1),
  statePath: z.string().min(1),
  artifactsDir: z.string().min(1),
});

export class Broker extends Service {
  static provide = 'broker';
  static Config = configSchema;

  readonly config: Config;
  readonly session: Session;
  readonly client: ApiClient;

  constructor(ctx: Context, config: Config) {
    super(ctx, 'broker');
    this.config = config;
    this.session = new Session(config.baseUrl, config.credentialsPath, config.envToken);
    this.client = new ApiClient(config.baseUrl, this.session);
  }
}

export default Broker;
