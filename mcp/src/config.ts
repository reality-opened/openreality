/**
 * Runtime configuration for the MCP server.
 *
 * Environment (first hit wins; the DEMO_* names are the persisted-contract
 * spellings the deployed broker already speaks — persisted contract, never
 * rename them):
 *   OPENREALITY_URL       | DEMO_BROKER_URL — broker base URL
 *   OPENREALITY_TOKEN     | DEMO_AUTH_TOKEN — bearer token (skips the credentials file)
 *   OPENREALITY_LOGIN_URL                    — browser hand-off page base URL, `login` w/ no --token
 *   OPENREALITY_DIR                          — state dir (default ~/.config/openreality)
 *   OPENREALITY_ARTIFACTS_DIR                — where artifact fetches land
 */
import { homedir } from 'node:os';
import { join } from 'node:path';

/** The deployed broker (Modal app vggt-slam-streaming, function `web`). */
export const DEFAULT_BROKER = 'https://galois77777--vggt-slam-streaming-web.modal.run';

/** The dashboard/marketing site hosting the `/cli-auth` browser hand-off page that
 *  `openreality-mcp login` (no --token) opens. */
export const DEFAULT_LOGIN_URL = 'https://open-reality.io';

export interface Config {
  baseUrl: string;
  /** Base URL of the browser hand-off page for `openreality-mcp login` (no --token). */
  loginBaseUrl: string;
  /** Token supplied via env — takes precedence over the credentials file. */
  envToken: string | null;
  configDir: string;
  credentialsPath: string;
  /** Sync-cursor state (last-seen scenes) — see workspace_list_scenes. */
  statePath: string;
  artifactsDir: string;
}

export function loadConfig(env: NodeJS.ProcessEnv = process.env): Config {
  const baseUrl = (env.OPENREALITY_URL || env.DEMO_BROKER_URL || DEFAULT_BROKER).replace(/\/$/, '');
  const loginBaseUrl = (env.OPENREALITY_LOGIN_URL || DEFAULT_LOGIN_URL).replace(/\/$/, '');
  const configDir = env.OPENREALITY_DIR || join(homedir(), '.config', 'openreality');
  return {
    baseUrl,
    loginBaseUrl,
    envToken: env.OPENREALITY_TOKEN || env.DEMO_AUTH_TOKEN || null,
    configDir,
    credentialsPath: join(configDir, 'credentials.json'),
    statePath: join(configDir, 'state.json'),
    artifactsDir: env.OPENREALITY_ARTIFACTS_DIR || join(configDir, 'artifacts'),
  };
}
