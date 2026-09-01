import { defineConfig } from 'tsup';

// The bin entry is fully bundled including the vendored `@reality/protocol`
// contract source (a TS `file:` dep), so `dist/` runs with only the published
// npm deps (cordis, @modelcontextprotocol/sdk, zod) installed.
export default defineConfig({
  entry: { cli: 'src/cli.ts' },
  format: ['esm'],
  platform: 'node',
  target: 'node20',
  sourcemap: true,
  clean: true,
  noExternal: [/^@reality\/protocol/],
  banner: { js: '#!/usr/bin/env node' },
});
