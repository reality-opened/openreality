"""Stage-0 contract tests for the OpenReality export schema (no GPU / model weights).

See docs/dataset-export.md §8 Stage 0. Asserts the package imports and the info/episode
manifests round-trip with the non-negotiable coordinate-frame caveats present.
"""

from __future__ import annotations

import json

from server.export import SCHEMA_VERSION
from server.export.schema import (
    ExportInfo,
    build_episode_summary,
    build_info,
    standard_features,
)


def test_schema_version_constant():
    assert SCHEMA_VERSION.startswith("openreality-export/")


def test_build_info_declares_non_metric_caveats():
    info = build_info("scan123", height=480, width=640, fps=10.0)
    # The world frame is up-to-scale / not gravity-aligned (docs/gotchas.md) — must be flagged.
    assert info.up_to_scale is True
    assert info.gravity_aligned is False
    assert info.metric is False
    assert info.scan_id == "scan123"
    assert info.fps == 10.0


def test_standard_features_shapes():
    feats = standard_features(480, 640)
    assert feats["observation.state"].shape == [7]
    assert feats["action"].shape == [6]
    assert feats["observation.intrinsics"].shape == [4]
    assert feats["observation.images.ego_view"].shape == [480, 640, 3]


def test_info_json_roundtrip():
    info = build_info("s", height=2, width=3)
    dumped = json.loads(info.model_dump_json())
    restored = ExportInfo.model_validate(dumped)
    assert restored == info


def test_episode_summary():
    ep = build_episode_summary("s", num_keyframes=12, trajectory_length=3.4, num_objects=5)
    assert ep.num_keyframes == 12
    assert ep.num_objects == 5
