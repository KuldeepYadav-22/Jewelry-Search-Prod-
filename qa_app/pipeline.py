"""
qa_app/pipeline.py
Per-request inference pipeline for the QA tool:

  raw upload -> BiRefNet.get_rgba() -> ImageProcessor.process()
             -> CLIP embedding (once, shared by gate + trained classifier)
             -> zero-shot is_jewelry gate
             -> DINOv2 embedding (on the crop)
             -> SupervisedCategoryClassifier -> top category + full breakdown
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
from PIL import Image

from qa_app.model_loader import ModelBundle

# Single GPU, single process: serialize inference calls so concurrent QA
# uploads don't race on the same CUDA context / model state.
_INFERENCE_LOCK = threading.Lock()


@dataclass
class QAResult:
    predicted_category: str
    predicted_confidence: float
    all_class_scores: dict[str, float]
    is_jewelry_gate: str          # "yes" | "no"
    gate_category: str            # raw zero-shot category, for the badge text
    gate_confidence: float
    model_type: str
    embedding_config: str


def run_pipeline(bundle: ModelBundle, image: Image.Image) -> QAResult:
    """Run the full pipeline on a single already-loaded PIL image."""
    with _INFERENCE_LOCK:
        rgba = bundle.birefnet.get_rgba(image)
        processed = bundle.processor.process(rgba)

        # CLIP encoded once, reused for both the gate and the trained classifier.
        clip_emb = bundle.clip.run(processed.white_bg)
        norm_clip = clip_emb / (np.linalg.norm(clip_emb) + 1e-8)

        gate_pass, gate_cat, gate_conf = bundle.zero_shot_gate.is_jewelry(norm_clip)
        is_jewelry = "yes" if gate_pass else "no"

        dino_emb = bundle.dinov2.run(processed.cropped)

        pred_cat, pred_conf, scores = bundle.supervised_clf.classify(clip_emb, dino_emb)

    return QAResult(
        predicted_category=pred_cat,
        predicted_confidence=round(float(pred_conf), 4),
        all_class_scores={k: round(float(v), 4) for k, v in scores.items()},
        is_jewelry_gate=is_jewelry,
        gate_category=gate_cat,
        gate_confidence=round(float(gate_conf), 4),
        model_type=bundle.model_type,
        embedding_config=bundle.embedding_config,
    )
