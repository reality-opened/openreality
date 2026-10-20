/**
 * Minimal ZIP reader — enough to unpack an export archive onto disk without
 * pulling a dependency in (the mirror of simulator/zip.ts, which writes them).
 * Handles stored + deflated entries (what Python's shutil.make_archive and the
 * simulator emit). ZIP64 archives are refused with a clear message; exports
 * that unpack through here are well under 4 GB.
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, isAbsolute, join, normalize, sep } from 'node:path';
import { inflateRawSync } from 'node:zlib';

const EOCD_SIG = 0x06054b50;
const CENTRAL_SIG = 0x02014b50;
const LOCAL_SIG = 0x04034b50;

export interface ZipFile {
  path: string;
  data: Buffer;
}

function findEocd(buf: Buffer): number {
  // EOCD is 22 bytes + an optional comment of up to 65535 bytes.
  const stop = Math.max(0, buf.length - 22 - 0xffff);
  for (let i = buf.length - 22; i >= stop; i--) {
    if (buf.readUInt32LE(i) === EOCD_SIG) return i;
  }
  throw new Error('not a zip archive (no end-of-central-directory record)');
}

/** Every file entry of a zip, decompressed (directories are skipped). */
export function readZip(buf: Buffer): ZipFile[] {
  const eocd = findEocd(buf);
  const count = buf.readUInt16LE(eocd + 10);
  let offset = buf.readUInt32LE(eocd + 16);
  if (count === 0xffff || offset === 0xffffffff) {
    throw new Error('ZIP64 archives are not supported by this reader');
  }
  const files: ZipFile[] = [];
  for (let n = 0; n < count; n++) {
    if (buf.readUInt32LE(offset) !== CENTRAL_SIG) throw new Error('corrupt zip central directory');
    const method = buf.readUInt16LE(offset + 10);
    const compressed = buf.readUInt32LE(offset + 20);
    const nameLen = buf.readUInt16LE(offset + 28);
    const extraLen = buf.readUInt16LE(offset + 30);
    const commentLen = buf.readUInt16LE(offset + 32);
    const localAt = buf.readUInt32LE(offset + 42);
    const name = buf.toString('utf8', offset + 46, offset + 46 + nameLen);
    offset += 46 + nameLen + extraLen + commentLen;
    if (name.endsWith('/')) continue;

    if (buf.readUInt32LE(localAt) !== LOCAL_SIG) throw new Error(`corrupt zip entry ${name}`);
    const start = localAt + 30 + buf.readUInt16LE(localAt + 26) + buf.readUInt16LE(localAt + 28);
    const raw = buf.subarray(start, start + compressed);
    if (method === 0) files.push({ path: name, data: Buffer.from(raw) });
    else if (method === 8) files.push({ path: name, data: inflateRawSync(raw) });
    else throw new Error(`zip entry ${name} uses unsupported compression method ${method}`);
  }
  return files;
}

/** Unpack ``buf`` under ``destDir``; returns the absolute paths written. Entries that would
 *  land outside ``destDir`` (absolute paths, ``..``) are refused — never written. */
export function extractZip(buf: Buffer, destDir: string): string[] {
  const root = normalize(destDir);
  const written: string[] = [];
  for (const { path, data } of readZip(buf)) {
    const rel = normalize(path);
    if (isAbsolute(rel) || rel === '..' || rel.startsWith(`..${sep}`)) {
      throw new Error(`zip entry escapes the destination: ${path}`);
    }
    const target = join(root, rel);
    mkdirSync(dirname(target), { recursive: true });
    writeFileSync(target, data);
    written.push(target);
  }
  return written;
}
