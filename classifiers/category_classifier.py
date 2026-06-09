"""
classifiers/category_classifier.py
Zero-shot category classification using CLIP text-image similarity.
Prompts loaded from config/prompts.yaml.
"""
from __future__ import annotations

import numpy as np


class CategoryClassifier:

    def __init__(self, clip_engine, prompts: dict[str, list[str]],
                 threshold: float = 0.20):
        """
        Args:
            clip_engine: CLIPEngine instance (used for text encoding).
            prompts: dict mapping category name → list of text prompts.
                     Loaded from config/prompts.yaml['categories'].
            threshold: minimum score to accept classification.
        """
        self.threshold = threshold
        self.centroids = {}

        for cat, prompt_list in prompts.items():
            vecs = clip_engine.encode_text(prompt_list, normalize=True)
            centroid = vecs.mean(axis=0)
            self.centroids[cat] = centroid / (np.linalg.norm(centroid) + 1e-8)

        print(f"  [CategoryClassifier] Ready — {len(self.centroids)} categories")

    def classify(self, clip_embedding: np.ndarray) -> tuple:
        """
        Classify an image embedding.

        Args:
            clip_embedding: (768,) float32, L2-normalised.

        Returns:
            (category_name, confidence, all_scores_dict)
        """
        norm = clip_embedding / (np.linalg.norm(clip_embedding) + 1e-8)
        scores = {
            cat: float(np.dot(norm, centroid))
            for cat, centroid in self.centroids.items()
        }
        best = max(scores, key=scores.get)
        if scores[best] < self.threshold:
            return "unknown", scores[best], scores
        return best, scores[best], scores
