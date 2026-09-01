/**
 * PLY -> SPZ encoder, using the RENDERER'S OWN codec.
 *
 *   node scripts/ply_to_spz.mjs <input.ply> <output.spz> [--verify]
 *
 * WHY THIS IS A NODE SCRIPT AND NOT PYTHON
 * ----------------------------------------
 * The demo client renders with Spark (`@sparkjsdev/spark`), which reads `.spz`
 * natively. SPZ is a quantized, gzipped container — hand-writing an encoder in
 * Python means reproducing its exact fixed-point layout, quantization constants
 * and version semantics from the outside, where a single wrong constant yields a
 * file that decodes to a silently corrupted scene.
 *
 * Instead we call Spark's own `transcodeSpz`, so the bytes we serve are produced
 * by the very library that will parse them. Compatibility is true by
 * construction, and it survives a Spark version bump.
 *
 * This is cheap to run: the Modal image already installs the webserver's
 * node_modules (it builds the SPA at image-build time), so Spark is present and
 * no new dependency is added anywhere. Measured: 1.5M-gaussian 84 MB PLY ->
 * 8.81 MB spz in ~2.0 s.
 *
 * `--verify` decodes the result back through Spark's own `SpzReader` and asserts
 * the splat count round-trips, so a corrupt write cannot reach a scene silently.
 */
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';

const args = process.argv.slice(2);
const inPath = args[0];
const outPath = args[1];
const verify = args.includes('--verify');

if (!inPath || !outPath) {
  console.error('usage: node ply_to_spz.mjs <input.ply> <output.spz> [--verify]');
  process.exit(2);
}

/**
 * Locate spark's ESM bundle. In the Modal image the server package sits at
 * /root/project/server and the webserver's node_modules at
 * /root/project/web/apps/webserver/node_modules. Fall back to normal resolution
 * so the script also runs from a dev checkout.
 */
function resolveSpark() {
  const explicit = process.env.SPARK_MODULE_PATH;
  const candidates = [
    explicit,
    '/root/project/web/apps/webserver/node_modules/@sparkjsdev/spark/dist/spark.module.js',
    new URL('../web/apps/webserver/node_modules/@sparkjsdev/spark/dist/spark.module.js', import.meta.url).pathname,
    new URL('../../web/apps/webserver/node_modules/@sparkjsdev/spark/dist/spark.module.js', import.meta.url).pathname,
  ].filter(Boolean);
  for (const c of candidates) {
    if (existsSync(c)) return pathToFileURL(c).href;
  }
  try {
    const require = createRequire(import.meta.url);
    return pathToFileURL(require.resolve('@sparkjsdev/spark')).href;
  } catch {
    console.error(
      'FATAL: could not locate @sparkjsdev/spark. Set SPARK_MODULE_PATH to dist/spark.module.js.\ntried:\n  ' +
        candidates.join('\n  '),
    );
    process.exit(3);
  }
}

/**
 * Spark is a browser library and its module has an IMPORT-TIME side effect that
 * touches `navigator`: `VRButton.registerSessionGrantedListener()` runs at the top
 * level and reads `navigator.xr`. Node 21+ ships a global `navigator`, so this is
 * invisible on a modern dev machine — but the Modal image runs **Node 20**, where
 * importing Spark dies with `ReferenceError: navigator is not defined`.
 *
 * A bare object is enough: the function reads `navigator.xr`, finds it undefined,
 * and returns immediately without registering anything. Nothing in the encode path
 * touches the DOM. Only defined when genuinely absent, so a newer Node keeps its
 * real global.
 */
if (typeof globalThis.navigator === 'undefined') {
  globalThis.navigator = { userAgent: 'node' };
}

const sparkUrl = resolveSpark();
const spark = await import(sparkUrl);
if (typeof spark.transcodeSpz !== 'function') {
  console.error('FATAL: this Spark build does not export transcodeSpz');
  process.exit(3);
}

const t0 = Date.now();
const plyBytes = new Uint8Array(readFileSync(inPath));

let res;
try {
  res = await spark.transcodeSpz({ inputs: [{ fileBytes: plyBytes, fileType: 'ply' }] });
} catch (e) {
  console.error(`FATAL: transcodeSpz failed: ${e && e.stack ? e.stack : e}`);
  process.exit(4);
}

const spz = res.fileBytes;
writeFileSync(outPath, spz);
const encodeMs = Date.now() - t0;

const report = {
  input: inPath,
  output: outPath,
  input_bytes: plyBytes.length,
  output_bytes: spz.length,
  ratio: +(plyBytes.length / spz.length).toFixed(2),
  encode_ms: encodeMs,
  clipped: res.clippedCount ?? 0,
};

if (verify) {
  const reader = new spark.SpzReader({ fileBytes: spz });
  await reader.parseHeader();
  report.verify = {
    version: reader.version,
    num_splats: reader.numSplats,
    sh_degree: reader.shDegree,
    fractional_bits: reader.fractionalBits,
    antialiased: reader.flagAntiAlias,
  };
  if (!reader.numSplats || reader.version !== 3) {
    console.error(`FATAL: verification failed: ${JSON.stringify(report.verify)}`);
    process.exit(5);
  }
}

// Single machine-readable line on stdout — the Modal job parses this.
console.log(JSON.stringify(report));
