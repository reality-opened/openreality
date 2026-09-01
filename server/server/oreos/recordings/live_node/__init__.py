"""Oreos live node: PERSISTENT-STREAM remote-GPU VGGT-SLAM (Task #5).

The production transport the live probe's verdict demanded — chunks stream over
ONE WebSocket into the warm A100 container instead of spawn-per-chunk RPC
(which lost ~1.15 s/chunk to Modal per-call overhead).

Five parts (the middle three are modal-free and unit-tested — the 2026-07-24
lifecycle audit landed on code that had no tests at all):
  * ``protocol.py``     — dependency-free wire framing (length-prefixed JSON +
    jpeg blobs, fragmentation under Modal's 2 MiB WS message cap, capped
    reassembly); runs unchanged in client, tests, and container.
  * ``session.py``      — pure session/lifecycle state: chunk-id ordering, the
    LRU results cache, the /health snapshot, the stall watchdog and the
    cross-session bleed guard.
  * ``asgi_app.py``     — the framework-free ASGI app (``/health``, ``/reset``,
    ``/ws``) driving a node object; testable against a fake node.
  * ``modal_stream.py`` — Modal app ``oreos-live-node``: ``@app.cls`` GPU
    container (``max_containers=1``) serving the ASGI app via
    ``@modal.asgi_app()``; VGGT + core Solver loaded once in ``@modal.enter``.
  * ``client.py``       — ``OreosLiveClient``: replay-clock frame streamer +
    sender/reader threads; writes ``live_node_results.json`` in the probe's
    schema (plus the server-side timing split that makes the architecture
    verdict transport-independent).

``DIMOS-NODE.md`` documents the diff-plan for upgrading the EXP-36
``OreosVisionNode`` replay demo to live mode with this client.
"""
