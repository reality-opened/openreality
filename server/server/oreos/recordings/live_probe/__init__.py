"""Oreos live probe: measures per-chunk round-trip latency of remote-GPU
VGGT-SLAM against a robot's real frame rate (Phase 2 of the Oreos-on-DimOS plan).

Two halves:
  * ``modal_live.py`` — Modal app ``oreos-live-probe``: one warm A100 container
    holding VGGT + a core ``Solver``; ``process_chunk`` runs one submap cycle.
  * ``driver.py``     — local replayer: streams a recorded session's frames at
    their RECORDED rate, fires 16+1-frame chunks at the GPU, measures round
    trip + lag behind the replay clock, writes ``live_probe_results.json``.
"""
