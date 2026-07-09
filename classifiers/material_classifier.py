"""
classifiers/material_classifier.py
Zero-shot material classification using CLIP text-image similarity.
Uses per-prompt soft voting instead of centroid averaging.
"""
from __future__ import annotations
import numpy as np


class MaterialClassifier:
    def __init__(self, clip_engine, prompts: dict[str, list[str]],
                 threshold: float = 0.25):
        self.threshold = threshold
        self.prompt_vecs = {}

        for mat, prompt_list in prompts.items():
            vecs = clip_engine.encode_text(prompt_list, normalize=True)
            self.prompt_vecs[mat] = vecs

        print(f"  [MaterialClassifier] Ready — {len(self.prompt_vecs)} materials")

    def classify(self, clip_embedding: np.ndarray) -> tuple:
        norm = clip_embedding / (np.linalg.norm(clip_embedding) + 1e-8)

        scores = {}
        for mat, vecs in self.prompt_vecs.items():
            per_prompt_scores = vecs @ norm
            top_k = min(2, len(per_prompt_scores))
            top_scores = np.sort(per_prompt_scores)[-top_k:]
            scores[mat] = float(top_scores.mean())

        best = max(scores, key=scores.get)
        return best, scores[best], scores