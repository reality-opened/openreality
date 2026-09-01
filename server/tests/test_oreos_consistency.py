"""Tests for server.oreos.recordings.consistency — the product-grade trajectory scorer.

Focus: association correctness. `associate()` binary-searches the reference
timestamps with np.searchsorted, which is only meaningful on sorted input, but
`load_tum()` used to hand back rows in file order. Nothing guarantees that order:
modal_recon.py emits est_tum.txt submap-by-submap and loop closures reorder
frames. Mis-association is SILENT — it produces a plausible-looking ATE and a
wrong `umeyama_scale`, which is the number gating the metric Isaac export.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

from server.oreos.recordings.consistency import associate, load_tum, score_consistency


# -- helpers -----------------------------------------------------------------

N = 60
DT = 0.1


def _reference_positions(n: int = N) -> np.ndarray:
    """An open, non-degenerate path (spread well above the 0.05 m usability bar)."""
    t = np.linspace(0.0, 4.0, n)
    return np.stack([np.cos(t) * 2.0, np.sin(t) * 2.0, t * 0.25], axis=1)


def _tum_lines(ts: np.ndarray, pos: np.ndarray) -> list[str]:
    return [
        f"{t:.6f} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} 0 0 0 1" for t, p in zip(ts, pos)
    ]


def _write(path: Path, ts: np.ndarray, pos: np.ndarray, shuffle: bool = False) -> Path:
    lines = _tum_lines(ts, pos)
    if shuffle:
        random.Random(1234).shuffle(lines)
        assert lines != _tum_lines(ts, pos), "shuffle must actually reorder"
    path.write_text("\n".join(lines) + "\n")
    return path


def _rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _session(tmp_path: Path, shuffle_est: bool, shuffle_ref: bool) -> tuple[Path, Path]:
    """est = a known Sim(3) image of the reference, so a correct association
    recovers scale exactly and leaves ~0 residual."""
    ts = np.arange(N) * DT
    ref = _reference_positions()
    # est -> ref needs s = 2.0, so build est as the half-scale rotated copy
    est = (0.5 * (_rot_z(0.7) @ ref.T)).T + np.array([1.5, -0.5, 0.25])
    return (
        _write(tmp_path / "est_tum.txt", ts, est, shuffle=shuffle_est),
        _write(tmp_path / "reference.txt", ts, ref, shuffle=shuffle_ref),
    )


# -- load_tum ----------------------------------------------------------------

def test_load_tum_sorts_by_timestamp(tmp_path: Path) -> None:
    ts = np.arange(N) * DT
    pos = _reference_positions()
    p = _write(tmp_path / "shuffled.txt", ts, pos, shuffle=True)

    out_ts, out_pos, out_q = load_tum(p)

    assert np.all(np.diff(out_ts) > 0), "timestamps must come back ascending"
    np.testing.assert_allclose(out_ts, ts, atol=1e-6)
    # rows stay glued to their own timestamp (not just the ts column sorted)
    np.testing.assert_allclose(out_pos, pos, atol=1e-6)
    assert out_q.shape == (N, 4)


def test_associate_requires_sorted_reference_and_load_tum_provides_it(
    tmp_path: Path,
) -> None:
    ts = np.arange(N) * DT
    pos = _reference_positions()
    p = _write(tmp_path / "shuffled.txt", ts, pos, shuffle=True)

    ts_sorted, _, _ = load_tum(p)
    ia, ib = associate(ts_sorted, ts_sorted)

    assert len(ia) == N
    np.testing.assert_array_equal(ia, ib)  # self-association is the identity


# -- score_consistency end to end --------------------------------------------

@pytest.mark.parametrize(
    "shuffle_est,shuffle_ref",
    [(False, False), (True, False), (False, True), (True, True)],
    ids=["both-sorted", "est-shuffled", "ref-shuffled", "both-shuffled"],
)
def test_scoring_is_invariant_to_file_row_order(
    tmp_path: Path, shuffle_est: bool, shuffle_ref: bool
) -> None:
    est, ref = _session(tmp_path, shuffle_est, shuffle_ref)

    out = score_consistency(est, ref, out_json=tmp_path / "consistency.json")

    assert out["reference_available"] is True
    assert out["n_matched"] == N, "every sample must pair with its own timestamp"
    # exact Sim(3) relation -> the alignment must be essentially perfect
    assert out["ate_rmse_m"] == pytest.approx(0.0, abs=1e-3)
    assert out["umeyama_scale"] == pytest.approx(2.0, abs=1e-3)
    # a constant global scale means no windowed scale drift
    assert out["windowed_scale_cv"] == pytest.approx(0.0, abs=1e-3)
    # and the json sidecar carries the same numbers
    saved = json.loads((tmp_path / "consistency.json").read_text())
    assert saved["umeyama_scale"] == out["umeyama_scale"]


def test_shuffled_and_sorted_inputs_score_identically(tmp_path: Path) -> None:
    a = tmp_path / "sorted"
    b = tmp_path / "shuffled"
    a.mkdir()
    b.mkdir()
    sorted_out = score_consistency(*_session(a, False, False))
    shuffled_out = score_consistency(*_session(b, True, True))

    assert sorted_out == shuffled_out
