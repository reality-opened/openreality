"""Scan-confidence QC (Oreos gating).

GT-free self-consistency scoring for reconstructions, so products (reports,
exports, embeds) can be gated on measured confidence instead of hope. Signal
choice and initial thresholds are calibrated on EXP-36's measured failure class
(platform/experiments/exp36_oreos_dimos_spike/) — recalibrate as pilot-scan
positives accumulate.
"""

from server.qc.confidence import ConfidenceLevel, ScanConfidence, compute_confidence

__all__ = ["ConfidenceLevel", "ScanConfidence", "compute_confidence"]
