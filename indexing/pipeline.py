"""
indexing/pipeline.py
Core indexing pipeline: serial_number + image → processed record.

This is the single-image processing unit. It does NOT handle DB writes
or batching — those are in batch.py and worker.py respectively.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image

from classifiers.category_classifier import CategoryClassifier
from classifiers.material_classifier import MaterialClassifier
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


class IndexingPipeline:
    """
    Processes a single image through the full indexing pipeline:
        image → BiRefNet → split path → CLIP + DINOv2 → classify → IndexRecord

    All components (engines, classifiers, preprocessor) are injected
    at construction and shared across calls.
    """

    def __init__(
        self,
        image_processor: ImageProcessor,
        clip_engine: CLIPEngine,
        dinov2_engine: DINOv2Engine,
        category_classifier: CategoryClassifier,
        material_classifier: MaterialClassifier,
    ):
        self.processor = image_processor
        self.clip = clip_engine
        self.dinov2 = dinov2_engine
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
            On failure, returns IndexRecord with error field set.
        """
        try:
            # Step 1: Preprocess — BiRefNet → white_bg + cropped
            processed = self.processor.process(image)

            # Step 2: CLIP embedding (uncropped white-bg)
            clip_emb = self.clip.run(processed.white_bg)

            # Step 3: DINOv2 embedding (cropped)
            dino_emb = self.dinov2.run(processed.cropped)

            # Step 4: Classify category + material
            norm_clip = clip_emb / (np.linalg.norm(clip_emb) + 1e-8)
            cat, cat_conf, _ = self.cat_clf.classify(norm_clip)
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
