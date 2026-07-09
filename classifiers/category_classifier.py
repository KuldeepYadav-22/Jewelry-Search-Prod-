"""
classifiers/category_classifier.py
Zero-shot category classification using CLIP text-image similarity.
Uses per-prompt soft voting instead of centroid averaging.
"""
from __future__ import annotations
import numpy as np


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