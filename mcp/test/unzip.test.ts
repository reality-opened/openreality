import { mkdtempSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { deflateRawSync } from 'node:zlib';
import { describe, expect, it } from 'vitest';
import { buildZip } from '../simulator/zip.js';
import { extractZip, readZip } from '../src/unzip.js';

/** Rewrite a stored zip's single entry as deflated (method 8) — what Python's
 *  shutil.make_archive produces, which the store-only simulator writer never does. */
function deflatedZip(path: string, data: Buffer): Buffer {
  const stored = buildZip([{ path, data }]);
  const packed = deflateRawSync(data);
  const nameLen = Buffer.byteLength(path);
  const local = Buffer.from(stored.subarray(0, 30));
  local.writeUInt16LE(8, 8);
  local.writeUInt32LE(packed.length, 18);
  const centralAt = 30 + nameLen + data.length;
  const central = Buffer.from(stored.subarray(centralAt, centralAt + 46 + nameLen));
  central.writeUInt16LE(8, 10);
  central.writeUInt32LE(packed.length, 20);
  const eocd = Buffer.from(stored.subarray(stored.length - 22));
  eocd.writeUInt32LE(central.length, 12);
  eocd.writeUInt32LE(30 + nameLen + packed.length, 16);
  return Buffer.concat([local, Buffer.from(path), packed, central, eocd]);
}

describe('unzip', () => {
  it('round-trips the simulator writer (stored entries)', () => {
    const zip = buildZip([
      { path: 'scan1/mujoco/scene.xml', data: Buffer.from('<mujoco/>') },
      { path: 'scan1/mujoco/manifest.json', data: Buffer.from('{"a":1}') },
    ]);
    expect(readZip(zip).map((f) => [f.path, f.data.toString()])).toEqual([
      ['scan1/mujoco/scene.xml', '<mujoco/>'],
      ['scan1/mujoco/manifest.json', '{"a":1}'],
    ]);
  });

  it('inflates deflated entries', () => {
    const body = Buffer.from('v 0 0 0\n'.repeat(500));
    const [file] = readZip(deflatedZip('a/scan_mesh.obj', body));
    expect(file!.data.equals(body)).toBe(true);
  });

  it('extracts under the destination and refuses escaping entries', () => {
    const dest = mkdtempSync(join(tmpdir(), 'unzip-'));
    const written = extractZip(buildZip([{ path: 's/mujoco/scene.xml', data: Buffer.from('x') }]), dest);
    expect(written).toEqual([join(dest, 's/mujoco/scene.xml')]);
    expect(readFileSync(written[0]!, 'utf8')).toBe('x');
    expect(() => extractZip(buildZip([{ path: '../evil', data: Buffer.from('x') }]), dest)).toThrow(/escapes/);
    expect(() => extractZip(buildZip([{ path: '/abs/evil', data: Buffer.from('x') }]), dest)).toThrow(/escapes/);
  });

  it('rejects non-zip input', () => {
    expect(() => readZip(Buffer.from('not a zip at all, definitely not'))).toThrow(/not a zip/);
  });
});
