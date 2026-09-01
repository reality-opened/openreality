/**
 * openreality-mcp — CLI entry.
 *
 *   openreality-mcp                 serve MCP over stdio (what Claude Code/desktop run)
 *   openreality-mcp login           browser sign-in; mints + stores a durable API key
 *   openreality-mcp login --token T validate + store a credential (mints a durable API
 *                                    key from it when the broker supports that)
 *   openreality-mcp keys list       list this account's API keys (incl. revoked)
 *   openreality-mcp keys revoke K   revoke an API key by id
 *   openreality-mcp simulator       run the mock-backend simulator (offline dev target)
 *   openreality-mcp whoami          quick credential/broker sanity check
 */
import { hostname } from 'node:os';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  REST_PATHS,
  isApiKeyToken,
  isListApiKeysResponse,
  isMintApiKeyResponse,
  isRevokeApiKeyResponse,
  type ApiKeySummary,
  type SceneListItem,
} from '@reality/protocol';
import { createApp } from './app.js';
import { loadConfig, type Config } from './config.js';
import { Session } from './session.js';
import { ApiClient, ApiError } from './http.js';
import { startSimulator } from '../simulator/sim.js';
import { openBrowser, randomState, startLoopbackServer } from './login.js';

function argValue(args: string[], flag: string): string | undefined {
  const i = args.indexOf(flag);
  return i >= 0 ? args[i + 1] : undefined;
}

/** Name every API key this CLI mints carries, so `keys list` reads legibly across a
 *  fleet of machines. */
function keyName(): string {
  return `claude-mcp@${hostname()}`;
}

/** `login --token T`: T is already a durable API key — validate it with a lightweight
 *  authed call (the broker has no dedicated whoami-for-keys route; GET /api/scenes
 *  doubles as one, same as the legacy JWT/session-token path below) and store as-is. */
async function loginWithApiKey(config: Config, token: string): Promise<void> {
  const session = new Session(config.baseUrl, config.credentialsPath, null);
  await session.adopt(token);
  const client = new ApiClient(config.baseUrl, session);
  try {
    const data = await client.json<{ scenes?: SceneListItem[] }>('GET', REST_PATHS.SCENES);
    console.log(
      `Signed in against ${config.baseUrl} with API key ${token.slice(0, 10)}… — ` +
        `${data.scenes?.length ?? 0} scene(s) visible.`,
    );
    console.log(`Credentials: ${config.credentialsPath}`);
  } catch (err) {
    console.error(`API key stored, but a validation call failed: ${err instanceof Error ? err.message : err}`);
    process.exitCode = 1;
  }
}

/**
 * `login --token T` where T is a Clerk JWT / broker session token (the pre-API-key
 * shape): attempt to mint a durable API key from it, falling back to the legacy
 * roll-it-durable behavior when the broker can't or won't mint (older broker, or any
 * other mint failure).
 */
async function loginWithLegacyToken(config: Config, token: string): Promise<void> {
  const mintSession = new Session(config.baseUrl, config.credentialsPath, token);
  const mintClient = new ApiClient(config.baseUrl, mintSession);
  try {
    const minted = await mintClient.json<unknown>('POST', REST_PATHS.API_KEYS, { name: keyName() });
    if (!isMintApiKeyResponse(minted)) throw new Error('malformed mint response');
    const session = new Session(config.baseUrl, config.credentialsPath, null);
    await session.adopt(minted.key);
    console.log(
      `Signed in against ${config.baseUrl} — minted durable API key ${minted.prefix}… ` +
        `("${minted.name}") and stored it.`,
    );
    console.log(`Credentials: ${config.credentialsPath}`);
    return;
  } catch (err) {
    console.warn(
      `Could not mint a durable API key (${err instanceof Error ? err.message : err}); ` +
        'falling back to the legacy session-token flow.',
    );
  }

  const session = new Session(config.baseUrl, config.credentialsPath, null);
  const rolled = await session.adopt(token);
  const client = new ApiClient(config.baseUrl, session);
  try {
    const data = await client.json<{ scenes?: SceneListItem[] }>('GET', REST_PATHS.SCENES);
    const n = data.scenes?.length ?? 0;
    console.log(
      `Signed in against ${config.baseUrl} — ${n} scene(s) visible.` +
        (rolled
          ? ' Token rolled to a durable session token and stored.'
          : ' Stored the provided token as-is (broker did not roll it).'),
    );
    console.log(`Credentials: ${config.credentialsPath}`);
  } catch (err) {
    console.error(`Token stored, but a test call failed: ${err instanceof Error ? err.message : err}`);
    process.exitCode = 1;
  }
}

async function loginWithToken(config: Config, token: string): Promise<void> {
  if (isApiKeyToken(token)) {
    await loginWithApiKey(config, token);
  } else {
    await loginWithLegacyToken(config, token);
  }
}

/**
 * `login` with no --token: mint a state nonce, open `${loginBaseUrl}/cli-auth` in the
 * browser (always also printing the URL), wait on the loopback callback for the
 * short-lived Clerk JWT it hands back, mint a durable API key from it, and store that.
 * Falls back to storing a rolled session token when the broker predates API keys.
 */
async function loginWithBrowser(config: Config): Promise<void> {
  const state = randomState();
  const loopback = await startLoopbackServer(state, 180_000);
  const authUrl =
    `${config.loginBaseUrl}/cli-auth?port=${loopback.port}&state=${encodeURIComponent(state)}`;
  console.log('Opening your browser to sign in. If it does not open, visit:');
  console.log(`  ${authUrl}`);
  openBrowser(authUrl);

  let clerkToken: string;
  try {
    clerkToken = await loopback.result;
  } catch (err) {
    console.error(err instanceof Error ? err.message : String(err));
    console.error('Try again, or run `openreality-mcp login --token <token>`.');
    process.exitCode = 1;
    return;
  } finally {
    await loopback.close();
  }

  const mintSession = new Session(config.baseUrl, config.credentialsPath, clerkToken);
  const mintClient = new ApiClient(config.baseUrl, mintSession);
  try {
    const minted = await mintClient.json<unknown>('POST', REST_PATHS.API_KEYS, { name: keyName() });
    if (!isMintApiKeyResponse(minted)) throw new Error('malformed mint response');
    const session = new Session(config.baseUrl, config.credentialsPath, null);
    await session.adopt(minted.key);
    console.log(`Signed in — minted durable API key ${minted.prefix}… ("${minted.name}") and stored it.`);
    console.log(`Credentials: ${config.credentialsPath}`);
    return;
  } catch (err) {
    const status = err instanceof ApiError ? err.status : null;
    if (status === 404 || status === 501) {
      console.warn("This broker doesn't support API keys yet — falling back to a rolled session token.");
    } else {
      console.warn(
        `Could not mint a durable API key (${err instanceof Error ? err.message : err}); ` +
          'falling back to a rolled session token.',
      );
    }
  }

  const session = new Session(config.baseUrl, config.credentialsPath, null);
  const rolled = await session.adopt(clerkToken);
  console.log(
    `Signed in against ${config.baseUrl}.` +
      (rolled ? ' Token rolled to a durable session token and stored.' : ' Stored the provided token as-is.'),
  );
  console.log(`Credentials: ${config.credentialsPath}`);
}

function printKeysTable(keys: ApiKeySummary[]): void {
  if (keys.length === 0) {
    console.log('No API keys.');
    return;
  }
  const fmt = (epoch: number | null): string => (epoch == null ? '—' : new Date(epoch * 1000).toISOString());
  const rows = keys.map((k) => ({
    key_id: k.key_id,
    name: k.name || '(unnamed)',
    prefix: k.prefix,
    created_at: fmt(k.created_at),
    last_used_at: fmt(k.last_used_at),
    status: k.revoked_at ? `revoked ${fmt(k.revoked_at)}` : 'active',
  }));
  const columns: Array<[keyof (typeof rows)[number], string]> = [
    ['key_id', 'KEY ID'],
    ['name', 'NAME'],
    ['prefix', 'PREFIX'],
    ['created_at', 'CREATED'],
    ['last_used_at', 'LAST USED'],
    ['status', 'STATUS'],
  ];
  const widths = columns.map(([key, header]) =>
    Math.max(header.length, ...rows.map((r) => r[key].length)),
  );
  const line = (cells: string[]): string => cells.map((c, i) => c.padEnd(widths[i]!)).join('  ');
  console.log(line(columns.map(([, header]) => header)));
  for (const r of rows) console.log(line(columns.map(([key]) => r[key])));
}

async function main(): Promise<void> {
  const [cmd = 'serve', ...rest] = process.argv.slice(2);

  if (cmd === 'serve') {
    const ctx = await createApp();
    await ctx.mcp.connect(new StdioServerTransport());
    return; // stdio transport keeps the process alive
  }

  if (cmd === 'login') {
    const config = loadConfig();
    const token = argValue(rest, '--token') ?? process.env.OPENREALITY_TOKEN ?? '';
    if (token) {
      await loginWithToken(config, token);
    } else {
      await loginWithBrowser(config);
    }
    return;
  }

  if (cmd === 'keys') {
    const config = loadConfig();
    const session = new Session(config.baseUrl, config.credentialsPath, config.envToken);
    if (!session.hasCredentials()) {
      console.error('No credentials. Run: openreality-mcp login');
      process.exitCode = 1;
      return;
    }
    const client = new ApiClient(config.baseUrl, session);

    if (rest[0] === 'list') {
      const data = await client.json<unknown>('GET', REST_PATHS.API_KEYS);
      if (!isListApiKeysResponse(data)) {
        console.error('Unexpected response from the broker.');
        process.exitCode = 1;
        return;
      }
      printKeysTable(data.keys);
      return;
    }

    if (rest[0] === 'revoke') {
      const keyId = rest[1];
      if (!keyId) {
        console.error('Usage: openreality-mcp keys revoke <key_id>');
        process.exitCode = 2;
        return;
      }
      const data = await client.json<unknown>('DELETE', REST_PATHS.API_KEY(keyId));
      if (!isRevokeApiKeyResponse(data)) {
        console.error('Unexpected response from the broker.');
        process.exitCode = 1;
        return;
      }
      console.log(`Revoked ${data.key_id} at ${new Date(data.revoked_at * 1000).toISOString()}.`);
      return;
    }

    console.error('Usage: openreality-mcp keys list | openreality-mcp keys revoke <key_id>');
    process.exitCode = 2;
    return;
  }

  if (cmd === 'whoami') {
    const config = loadConfig();
    const session = new Session(config.baseUrl, config.credentialsPath, config.envToken);
    const client = new ApiClient(config.baseUrl, session);
    if (!session.hasCredentials()) {
      console.error('No credentials. Run: openreality-mcp login');
      process.exitCode = 1;
      return;
    }
    const data = await client.json<{ scenes?: SceneListItem[] }>('GET', REST_PATHS.SCENES);
    console.log(`broker: ${config.baseUrl}`);
    console.log(`credential: ${session.isApiKeyCredential() ? 'durable API key' : 'session token'}`);
    console.log(`scenes visible: ${data.scenes?.length ?? 0}`);
    return;
  }

  if (cmd === 'simulator') {
    const port = Number(argValue(rest, '--port') ?? 8973);
    const { url } = await startSimulator({ port });
    console.log(`Open Reality mock-backend simulator listening at ${url}`);
    console.log('Point the MCP server at it with:');
    console.log(`  OPENREALITY_URL=${url} OPENREALITY_TOKEN=sim-token openreality-mcp`);
    return; // http server keeps the process alive
  }

  console.error(`Unknown command: ${cmd} (expected serve | login | keys | whoami | simulator)`);
  process.exitCode = 2;
}

main().catch((err) => {
  console.error(err instanceof Error ? (err.stack ?? err.message) : String(err));
  process.exit(1);
});
