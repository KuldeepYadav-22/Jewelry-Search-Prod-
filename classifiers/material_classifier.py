"""
classifiers/material_classifier.py
Zero-shot material classification using CLIP text-image similarity.
Prompts loaded from config/prompts.yaml.
"""
from __future__ import annotations

import numpy as np


class MaterialClassifier:

    def __init__(self, clip_engine, prompts: dict[str, list[str]],
                 threshold: float = 0.25):
        """
        Args:
            clip_engine: CLIPEngine instance.
            prompts: dict mapping material name → list of text prompts.
                     Loaded from config/prompts.yaml['materials'].
            threshold: minimum score to accept classification.
        """
        self.threshold = threshold
        self.centroids = {}

        for mat, prompt_list in prompts.items():
            vecs = clip_engine.encode_text(prompt_list, normalize=True)
            centroid = vecs.mean(axis=0)
            self.centroids[mat] = centroid / (np.linalg.norm(centroid) + 1e-8)

        print(f"  [MaterialClassifier] Ready — {len(self.centroids)} materials")

    def classify(self, clip_embedding: np.ndarray) -> tuple:
        """
        Classify material from an image embedding.

        Args:
            clip_embedding: (768,) float32, L2-normalised.

        Returns:
            (material_name, confidence, all_scores_dict)
        """
        norm = clip_embedding / (np.linalg.norm(clip_embedding) + 1e-8)
        scores = {
            mat: float(np.dot(norm, centroid))
            for mat, centroid in self.centroids.items()
        }
        best = max(scores, key=scores.get)
        return best, scores[best], scores
