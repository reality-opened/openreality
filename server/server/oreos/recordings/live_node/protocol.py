"""Wire protocol for the Oreos live node (client <-> Modal WebSocket endpoint).

Deliberately dependency-free (stdlib only) so the SAME file runs in three places
untouched: the local client, the hermetic tests, and the Modal container (the
whole ``live_node`` package is mounted at ``/root/oreos_live_node`` by
``modal_stream.py``'s image, so intra-package relative imports work in both).

Framing (every WebSocket message is binary):

    [4-byte big-endian header length][UTF-8 JSON header][blob 0][blob 1]...

The header's ``blob_lens`` key (added by :func:`encode_message`) records each
blob's byte length, in order. Blobs carry jpeg bytes (client->server) and are
empty for control/result messages.

Modal caps WebSocket messages at 2 MiB, and a 17-frame go2 chunk can reach
~1.6 MB — so a message that exceeds :data:`MAX_WS_MESSAGE_BYTES` is split into
``part`` frames (each itself a well-formed message wrapping a byte fragment) and
reassembled by :class:`Assembler` on the far side. Fragments of one message are
contiguous: WebSocket delivery is ordered, and each peer direction has a single
writer.

Message types (``header["type"]``):

    client -> server:  ``chunk``   {chunk_id, names, client_send_ts} + jpeg blobs
                       ``reset``   {}          (reset SLAM state, keep models warm)
                       ``end``     {}          (all chunks sent; server replies ``bye``)
    server -> client:  ``hello``   {server_ts, container_boot_s, ...} (on accept)
                       ``result``  {chunk_id, n_poses, poses_tum_lines, cloud_summary,
                                    gpu_ms, server_* timing split, ...}
                       ``reset_ok``/``bye``/``busy``/``error``
    both:              ``part``    {seq, last} + [fragment]  (transport-level only)
"""

from __future__ import annotations

import json
import struct

# Modal's web endpoints cap WebSocket messages at 2 MiB; stay well under it so
# the part-envelope overhead can never push a frame over the cap.
MAX_WS_MESSAGE_BYTES = 1_800_000
_PART_OVERHEAD = 256  # generous bound on the part envelope (json header + length prefix)

# Assembler safety caps (2026-07-24 audit, finding L8). A peer that keeps sending
# `part` frames and never sets `last` used to grow ``Assembler._pieces`` without
# bound — an unauthenticated OOM of the one GPU container. Both caps are far above
# anything the real client produces: a 17-frame go2 chunk is ~1.6 MB, i.e. ONE
# frame; the largest legitimate message the node ever assembles is a chunk of
# jpegs, and 512 MiB is ~300x that. Exceeding either cap is a protocol error, not
# a truncation: the session is closed rather than fed a half message.
MAX_ASSEMBLED_BYTES = 512 * 1024 * 1024
MAX_ASSEMBLED_PARTS = 4096

_LEN = struct.Struct(">I")

# -- message types -------------------------------------------------------------
T_CHUNK = "chunk"
T_RESET = "reset"
T_END = "end"
T_HELLO = "hello"
T_RESULT = "result"
T_RESET_OK = "reset_ok"
T_BYE = "bye"
T_BUSY = "busy"
T_ERROR = "error"
T_PART = "part"


class ProtocolError(ValueError):
    """Malformed or out-of-sequence wire data."""


def encode_message(header: dict, blobs: list | tuple = ()) -> bytes:
    """Serialize one protocol message (header dict + binary blobs) to bytes."""
    blobs = [bytes(b) for b in blobs]
    hdr = dict(header)
    hdr["blob_lens"] = [len(b) for b in blobs]
    hj = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
    return b"".join([_LEN.pack(len(hj)), hj] + blobs)


def decode_message(data: bytes) -> tuple[dict, list[bytes]]:
    """Inverse of :func:`encode_message`; raises :class:`ProtocolError` on bad data."""
    data = bytes(data)
    if len(data) < _LEN.size:
        raise ProtocolError(f"message shorter than length prefix ({len(data)} bytes)")
    (hlen,) = _LEN.unpack_from(data, 0)
    if len(data) < _LEN.size + hlen:
        raise ProtocolError(f"truncated header: need {hlen} bytes, have {len(data) - _LEN.size}")
    try:
        header = json.loads(data[_LEN.size : _LEN.size + hlen].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ProtocolError(f"header is not valid JSON: {e}") from e
    if not isinstance(header, dict):
        raise ProtocolError(f"header must be a JSON object, got {type(header).__name__}")
    off = _LEN.size + hlen
    blobs: list[bytes] = []
    for blen in header.get("blob_lens", []):
        if not isinstance(blen, int) or blen < 0 or off + blen > len(data):
            raise ProtocolError(f"blob length {blen} exceeds message ({len(data)} bytes)")
        blobs.append(data[off : off + blen])
        off += blen
    if off != len(data):
        raise ProtocolError(f"{len(data) - off} trailing bytes after declared blobs")
    return header, blobs


def iter_wire_frames(message: bytes, max_frame: int = MAX_WS_MESSAGE_BYTES) -> list[bytes]:
    """Split an encoded message into WebSocket-safe frames.

    A message that fits is sent as-is (single frame). An oversized message is cut
    into ``part`` frames whose fragments concatenate back to the original bytes.
    """
    if max_frame <= _PART_OVERHEAD:
        raise ValueError(f"max_frame must exceed {_PART_OVERHEAD}")
    if len(message) <= max_frame:
        return [message]
    step = max_frame - _PART_OVERHEAD
    pieces = [message[i : i + step] for i in range(0, len(message), step)]
    return [
        encode_message({"type": T_PART, "seq": i, "last": i == len(pieces) - 1}, [piece])
        for i, piece in enumerate(pieces)
    ]


class Assembler:
    """Feed raw WebSocket frames; get back complete ``(header, blobs)`` messages.

    Non-``part`` frames pass straight through. ``part`` frames accumulate until
    ``last``, then the concatenation is decoded as one message. Stateful per
    connection direction — use one Assembler per peer.

    Accumulation is capped (:data:`MAX_ASSEMBLED_BYTES` / :data:`MAX_ASSEMBLED_PARTS`):
    a peer that never sets ``last`` gets a :class:`ProtocolError`, not an OOM.
    """

    def __init__(self, max_bytes: int = MAX_ASSEMBLED_BYTES,
                 max_parts: int = MAX_ASSEMBLED_PARTS) -> None:
        self.max_bytes = int(max_bytes)
        self.max_parts = int(max_parts)
        self._pieces: list[bytes] = []
        self._nbytes = 0
        self._next_seq = 0

    def _drop(self) -> None:
        """Forget the partial message (the connection is about to be closed)."""
        self._pieces, self._nbytes, self._next_seq = [], 0, 0

    def feed(self, frame: bytes) -> tuple[dict, list[bytes]] | None:
        header, blobs = decode_message(frame)
        if header.get("type") != T_PART:
            if self._pieces:
                self._drop()
                raise ProtocolError("complete message interleaved inside a part sequence")
            return header, blobs
        if header.get("seq") != self._next_seq:
            expected = self._next_seq
            self._drop()
            raise ProtocolError(f"part seq {header.get('seq')} != expected {expected}")
        if len(blobs) != 1:
            self._drop()
            raise ProtocolError(f"part frame must carry exactly 1 fragment, got {len(blobs)}")
        if len(self._pieces) + 1 > self.max_parts:
            n = len(self._pieces)
            self._drop()
            raise ProtocolError(
                f"part sequence exceeds {self.max_parts} fragments ({n} accumulated) "
                f"without a `last` frame — refusing to buffer more"
            )
        if self._nbytes + len(blobs[0]) > self.max_bytes:
            total = self._nbytes + len(blobs[0])
            self._drop()
            raise ProtocolError(
                f"part sequence exceeds {self.max_bytes} bytes ({total} accumulated) "
                f"without a `last` frame — refusing to buffer more"
            )
        self._pieces.append(blobs[0])
        self._nbytes += len(blobs[0])
        self._next_seq += 1
        if header.get("last"):
            data = b"".join(self._pieces)
            self._drop()
            return decode_message(data)
        return None
