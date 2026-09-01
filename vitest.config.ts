import { defineConfig } from 'vitest/config';
import { resolve } from 'node:path';

export default defineConfig({
  resolve: {
    alias: {
      // The vendored contract source is consumed as TS source, never as a
      // build artifact (same pattern the upstream web workspace uses).
      '@reality/protocol': resolve(__dirname, 'vendor/protocol/index.ts'),
    },
  },
  test: {
    include: ['test/**/*.test.ts'],
    testTimeout: 30_000,
    hookTimeout: 30_000,
  },
});
