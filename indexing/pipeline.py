"""
indexing/pipeline.py
Core indexing pipeline: serial_number + image → processed record.

This is the single-image processing unit. It does NOT handle DB writes
or batching — those are in batch.py and worker.py respectively.

Flow (sequential, gate-first):
    image → split path (white_bg + cropped)
      → CLIP encode (white_bg)
      → is_jewelry gate (zero-shot CLIP, narrow binary check)
          no  → reject: skip DINOv2, skip trained classifier, skip material
          yes → DINOv2 encode (cropped)
                → trained category classifier (CLIP + DINO per embedding_config)
                → material classifier (CLIP-only, unchanged)

DINOv2 encoding, the trained category classifier, and material
classification only run once the gate passes — they are no longer run
unconditionally in parallel with classification.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

from classifiers.category_classifier import CategoryClassifier
from classifiers.material_classifier import MaterialClassifier
from classifiers.supervised_category_classifier import SupervisedCategoryClassifier
from engines.clip_engine import CLIPEngine
from engines.dinov2_engine import DINOv2Engine
from preprocess.image_processor import ImageProcessor

logger = logging.getLogger(__name__)


@dataclass
class IndexRecord:
    """Single indexed image record, ready for DB insertion."""
    serial_number: str
    tenant_id: str
    category: str
    category_confidence: float
    material: str
    material_confidence: float
    clip_embedding: list[float]
    dino_embedding: list[float]
    error: Optional[str] = None
    # Set when the is_jewelry gate rejected the image. Callers (batch/worker
    # code) must skip the DB write for rejected records — this mirrors the
    # search pipeline's `rejected` / `rejected_reason` fields.
    rejected: bool = False
    rejected_reason: Optional[str] = None


class IndexingPipeline:
    """
    Processes a single image through the full indexing pipeline:
        image → split path → CLIP encode → is_jewelry gate →
        [reject] or [DINOv2 encode → trained classify → material classify]
        → IndexRecord

    All components (engines, classifiers, preprocessor) are injected
    at construction and shared across calls.
    """

    def __init__(
        self,
        image_processor: ImageProcessor,
        clip_engine: CLIPEngine,
        dinov2_engine: DINOv2Engine,
        zero_shot_gate: CategoryClassifier,
        category_classifier: SupervisedCategoryClassifier,
        material_classifier: MaterialClassifier,
    ):
        self.processor = image_processor
        self.clip = clip_engine
        self.dinov2 = dinov2_engine
        self.gate = zero_shot_gate
        self.cat_clf = category_classifier
        self.mat_clf = material_classifier

    def process(self, serial_number: str, image: Image.Image,
                tenant_id: str) -> IndexRecord:
        """
        Process a single image through the full pipeline.

        Args:
            serial_number: unique product identifier.
            image: RGB PIL image.
            tenant_id: tenant identifier for multi-tenant isolation.

        Returns:
            IndexRecord ready for DB insertion.
            On is_jewelry gate rejection, `rejected` is True — the caller
            must not insert a row for this record.
            On failure, returns IndexRecord with error field set.
        """
        try:
            # Step 1: Preprocess — split into white_bg (CLIP) + cropped (DINOv2)
            processed = self.processor.process(image)

            # Step 2: CLIP embedding (uncropped white-bg)
            clip_emb = self.clip.run(processed.white_bg)
            norm_clip = clip_emb / (np.linalg.norm(clip_emb) + 1e-8)

            # Step 3: is_jewelry gate — must pass before any downstream step
            is_jewelry, gate_cat, gate_conf = self.gate.is_jewelry(norm_clip)
            if not is_jewelry:
                return IndexRecord(
                    serial_number=serial_number,
                    tenant_id=tenant_id,
                    category=gate_cat,
                    category_confidence=round(float(gate_conf), 4),
                    material="unknown",
                    material_confidence=0.0,
                    clip_embedding=clip_emb.tolist(),
                    dino_embedding=[],
                    rejected=True,
                    rejected_reason=(
                        f"Image classified as '{gate_cat}' by the is_jewelry "
                        f"gate — not a jewelry product photo."
                    ),
                )

            # Step 4: DINOv2 embedding (cropped) — only for images that passed the gate
            dino_emb = self.dinov2.run(processed.cropped)

            # Step 5: Trained category classifier (CLIP + DINO per embedding_config)
            cat, cat_conf, _ = self.cat_clf.classify(clip_emb, dino_emb)

            # Step 6: Material classifier (CLIP-only, unchanged)
            mat, mat_conf, _ = self.mat_clf.classify(norm_clip)

            return IndexRecord(
                serial_number=serial_number,
                tenant_id=tenant_id,
                category=cat,
                category_confidence=round(float(cat_conf), 4),
                material=mat,
                material_confidence=round(float(mat_conf), 4),
                clip_embedding=clip_emb.tolist(),
                dino_embedding=dino_emb.tolist(),
            )

        except Exception as e:
            logger.error(f"Failed to process {serial_number}: {e}")
            return IndexRecord(
                serial_number=serial_number,
                tenant_id=tenant_id,
                category="unknown",
                category_confidence=0.0,
                material="unknown",
                material_confidence=0.0,
                clip_embedding=[],
                dino_embedding=[],
                error=str(e),
            )
