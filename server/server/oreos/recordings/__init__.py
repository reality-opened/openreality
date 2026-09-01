"""Oreos: OpenReality's vision pipeline for robot recordings (DimOS-first).

One command takes a robot recording end-to-end:

    ingest -> recon (Modal GPU) -> consistency scoring -> QC gate -> report -> export

Design provenance: EXP-36 (platform/experiments/exp36_oreos_dimos_spike) and the
Oreos-on-DimOS plan (platform/experiments/research/2026-07-24-oreos-on-dimos-plan.md).
Rules inherited from the plan: consume-don't-fork (dimos is a pinned dependency of the
replay-node demo only; the batch pipeline needs no dimos at all), and nothing ships
un-gated (every session ends at a ScanConfidence verdict; exports refuse on
out-of-class with written reasons).
"""

from server.oreos.recordings.pipeline import OreosPipeline, StageResult

__all__ = ["OreosPipeline", "StageResult"]
