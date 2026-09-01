type AtobLike = (input: string) => string;
type BtoaLike = (input: string) => string;
type BufferLike = {
  from(input: string, encoding: 'base64'): Uint8Array;
};
type BufferCtorLike = {
  from(input: ArrayBuffer | Uint8Array): { toString(encoding: 'base64'): string };
};

function base64ToBytes(base64: string): Uint8Array {
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

export function base64ToFloat32Array(base64: string): Float32Array {
  const bytes = base64ToBytes(base64);
  const buffer = bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength
    ? bytes.buffer
    : bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  return new Float32Array(buffer);
}

export function base64ColorsToFloat32(base64: string): Float32Array {
  const bytes = base64ToBytes(base64);
  const out = new Float32Array(bytes.length);
  for (let i = 0; i < bytes.length; i += 1) {
    out[i] = bytes[i] / 255.0;
  }
  return out;
}

/** Encode raw bytes as a base64 string — the inverse of `base64ToBytes`. Used to hand a
 *  browser-parsed cloud (e.g. a re-fetched derived/calibrated PLY) to the viewer via the same
 *  `points_b64`/`colors_b64` (`SLAMUpdate`) path the server payload uses. Works in the browser
 *  (`btoa`) and in Node (`Buffer`); chunked so a large typed array doesn't blow the call stack. */
export function bytesToBase64(bytes: Uint8Array): string {
  const btoaFn = (globalThis as { btoa?: BtoaLike }).btoa;
  if (typeof btoaFn === 'function') {
    let binary = '';
    const CHUNK = 0x8000; // 32k — avoids "too many function arguments" on String.fromCharCode
    for (let i = 0; i < bytes.length; i += CHUNK) {
      binary += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
    }
    return btoaFn(binary);
  }

  const buffer = (globalThis as unknown as { Buffer?: BufferCtorLike }).Buffer;
  if (buffer) {
    return buffer.from(bytes).toString('base64');
  }

  throw new Error('No base64 encoder is available in this runtime.');
}

export function concatFloat32Arrays(arrays: Float32Array[]): Float32Array {
  if (arrays.length === 0) return new Float32Array(0);
  if (arrays.length === 1) return arrays[0];

  let totalLength = 0;
  for (const array of arrays) {
    totalLength += array.length;
  }

  const result = new Float32Array(totalLength);
  let offset = 0;
  for (const array of arrays) {
    result.set(array, offset);
    offset += array.length;
  }
  return result;
}
