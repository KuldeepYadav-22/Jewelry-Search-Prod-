"""
classifiers/category_classifier.py
Zero-shot category classification using CLIP text-image similarity.
Uses per-prompt soft voting instead of centroid averaging.

Also doubles as the narrow "is_jewelry" gate that runs immediately after
CLIP encoding, ahead of the trained category classifier — see is_jewelry().
"""
from __future__ import annotations
import numpy as np

# Zero-shot outcomes that mean "not a jewelry product photo": the explicit
# "paper" reject prompts, or a best score too weak to trust ("unknown").
NOT_JEWELRY_CATEGORIES = {"paper", "unknown"}


class CategoryClassifier:
    def __init__(self, clip_engine, prompts: dict[str, list[str]],
                 threshold: float = 0.20):
        self.threshold = threshold
        self.prompt_vecs = {}

        for cat, prompt_list in prompts.items():
            vecs = clip_engine.encode_text(prompt_list, normalize=True)
            self.prompt_vecs[cat] = vecs

        print(f"  [CategoryClassifier] Ready — {len(self.prompt_vecs)} categories")

    def classify(self, clip_embedding: np.ndarray) -> tuple:
        norm = clip_embedding / (np.linalg.norm(clip_embedding) + 1e-8)

        scores = {}
        for cat, vecs in self.prompt_vecs.items():
            per_prompt_scores = vecs @ norm
            top_k = min(2, len(per_prompt_scores))
            top_scores = np.sort(per_prompt_scores)[-top_k:]
            scores[cat] = float(top_scores.mean())

        best = max(scores, key=scores.get)

        if scores[best] < self.threshold:
            return "unknown", scores[best], scores

        return best, scores[best], scores

    def is_jewelry(self, clip_embedding: np.ndarray) -> tuple[bool, str, float]:
        """
        Narrow binary gate: is this a jewelry product photo at all?

        Wraps classify() rather than re-scoring — "paper" and "unknown"
        (low-confidence best match) both mean "no". Runs first, ahead of
        DINOv2 encoding and the trained category classifier: those only
        run when this gate passes.

        Returns:
            (is_jewelry, raw_category, confidence) — raw_category/confidence
            are the underlying zero-shot classify() result, kept for logging
            and for populating rejected-result fields.
        """
        category, confidence, _ = self.classify(clip_embedding)
        return category not in NOT_JEWELRY_CATEGORIES, category, confidence