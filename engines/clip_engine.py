"""
engines/clip_engine.py
CLIP ViT-L/14 for image embedding + text encoding.

Used for:
- Category classification (via classifiers/)
- Material classification (via classifiers/)
- Semantic similarity search embeddings
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

from engines.base import BaseEngine


class CLIPEngine(BaseEngine):

    def __init__(self, config):
        super().__init__(config)

        model_path = getattr(config, "clip_model_path", "models/clip-ViT-L-14")
        print(f"  [CLIPEngine] Loading from '{model_path}' on {self.device} ...")

        self.model = SentenceTransformer(
            model_path, device=str(self.device),
        )
        self.embedding_dim = 768
        print(f"  [CLIPEngine] Ready (dim={self.embedding_dim})")

    def run(self, image: Image.Image) -> np.ndarray:
        """Encode a PIL image → (768,) float32, NOT L2-normalised."""
        embedding = self.model.encode(
            [image], normalize_embeddings=False, show_progress_bar=False,
        )
        return embedding[0].astype(np.float32)

    def encode_text(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        """Encode text strings → (N, 768) float32."""
        return self.model.encode(
            texts, normalize_embeddings=normalize, show_progress_bar=False,
        ).astype(np.float32)
