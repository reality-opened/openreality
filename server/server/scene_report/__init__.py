"""Scene report subsystem.

Turns a finished (or in-progress) VGGT-SLAM reconstruction into structured,
evidence-grounded facts and an LLM-authored report of the scanned space.

Public surface:
- ``SceneFeatureExtractor`` (features.py): reconstruction -> ``SceneFacts``.
- ``SceneReportBuilder`` (report.py): ``SceneFacts`` + keyframes -> ``SceneReport``.
- ``InMemoryScenePersistence`` / ``ModalScenePersistence`` (store.py): non-durable
  default + durable per-user (Modal Dict + Volume) store.
- schemas: ``SceneFacts``, ``ObjectInstance``, ``SpatialRelation``, ``SceneReport``.
"""

from __future__ import annotations

from server.scene_report.schemas import (
    EvidenceRef,
    ObjectDescription,
    ObjectInstance,
    ProductMatch,
    SceneFacts,
    SceneMetrics,
    SceneReport,
    SpatialRelation,
)
from server.scene_report.features import SceneFeatureExtractor
from server.scene_report.report import SceneReportBuilder
from server.scene_report.object_enricher import ObjectEnricher, SerpApiLensProvider
from server.scene_report.store import InMemoryScenePersistence, ModalScenePersistence

__all__ = [
    "EvidenceRef",
    "ObjectDescription",
    "ObjectInstance",
    "ProductMatch",
    "SceneFacts",
    "SceneMetrics",
    "SceneReport",
    "SpatialRelation",
    "SceneFeatureExtractor",
    "SceneReportBuilder",
    "ObjectEnricher",
    "SerpApiLensProvider",
    "InMemoryScenePersistence",
    "ModalScenePersistence",
]
