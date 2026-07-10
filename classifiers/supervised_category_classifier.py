"""
classifiers/supervised_category_classifier.py
Supervised 10-class jewelry category classifier.
Drop-in replacement for CategoryClassifier (zero-shot CLIP).

Trained on precomputed CLIP + DINOv2 embeddings.
Same .classify() interface: returns (category, confidence, scores_dict).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import joblib

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


class SupervisedCategoryClassifier:
    """
    Supervised replacement for zero-shot CategoryClassifier.

    Usage:
        clf = SupervisedCategoryClassifier(model_dir="path/to/models/")
        category, confidence, all_scores = clf.classify(
            clip_embedding,          # (768,) raw CLIP embedding
            dino_embedding=None,     # (1024,) raw DINOv2 embedding (if needed)
        )
    """

    def __init__(self, model_dir: str | Path, threshold: float = 0.20):
        model_dir = Path(model_dir)
        self.threshold = threshold

        # Load metadata
        with open(model_dir / "model_metadata.json") as f:
            self.meta = json.load(f)

        # Load label encoder
        self.le = joblib.load(model_dir / "label_encoder.joblib")

        # Load model
        self.model_type = self.meta["model_type"]
        self.embedding_config = self.meta["embedding_config"]

        if self.model_type == "XGBoost":
            assert HAS_XGB, "xgboost is required for this model"
            self.model = xgb.Booster()
            self.model.load_model(str(model_dir / "category_classifier.xgb"))
        else:
            self.model = joblib.load(model_dir / "category_classifier.joblib")

        print(f"  [SupervisedCategoryClassifier] Ready — "
              f"{self.model_type} + {self.embedding_config}, "
              f"{len(self.le.classes_)} classes")

    def _build_features(self, clip_embedding: np.ndarray,
                        dino_embedding: np.ndarray | None) -> np.ndarray:
        """Build feature vector based on the winning embedding config."""
        clip_norm = clip_embedding / (np.linalg.norm(clip_embedding) + 1e-8)

        if "CLIP+DINO" in self.embedding_config:
            assert dino_embedding is not None, \
                "DINOv2 embedding required for CLIP+DINO config"
            dino_norm = dino_embedding / (np.linalg.norm(dino_embedding) + 1e-8)
            return np.concatenate([clip_norm, dino_norm])

        elif "DINOv2" in self.embedding_config:
            assert dino_embedding is not None, \
                "DINOv2 embedding required for DINOv2 config"
            dino_norm = dino_embedding / (np.linalg.norm(dino_embedding) + 1e-8)
            return dino_norm

        else:  # CLIP only
            return clip_norm

    def classify(self, clip_embedding: np.ndarray,
                 dino_embedding: np.ndarray | None = None) -> tuple:
        """
        Classify a jewelry image.

        Args:
            clip_embedding: (768,) raw CLIP embedding (un-normalized OK).
            dino_embedding: (1024,) raw DINOv2 embedding (needed if config uses it).

        Returns:
            (category_str, confidence_float, {category: score} dict)
            Same interface as the old CategoryClassifier.
        """
        features = self._build_features(clip_embedding, dino_embedding)
        X = features.reshape(1, -1)

        if self.model_type == "XGBoost":
            dmat = xgb.DMatrix(X)
            probs = self.model.predict(dmat)[0]  # (num_class,)
        else:
            probs = self.model.predict_proba(X)[0]

        scores = {self.le.classes_[i]: float(probs[i])
                  for i in range(len(probs))}

        best_idx = int(np.argmax(probs))
        best_cat = self.le.classes_[best_idx]
        best_conf = float(probs[best_idx])

        if best_conf < self.threshold:
            return "unknown", best_conf, scores

        return best_cat, best_conf, scores
