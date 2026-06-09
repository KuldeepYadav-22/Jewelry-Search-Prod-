"""
engines/dinov2_engine.py
DINOv2 ViT-L/14 for visual similarity embeddings.

Captures shape, texture, design density, spatial layout.
Used ONLY for similarity search, not classification.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from engines.base import BaseEngine


class DINOv2Engine(BaseEngine):

    _MODEL_VARIANTS = {
        "dinov2_vitl14": {"hub_name": "dinov2_vitl14", "dim": 1024},
        "dinov2_vitb14": {"hub_name": "dinov2_vitb14", "dim": 768},
        "dinov2_vits14": {"hub_name": "dinov2_vits14", "dim": 384},
    }

    def __init__(self, config):
        super().__init__(config)

        model_key = getattr(config, "dinov2_model", "dinov2_vitl14")
        variant = self._MODEL_VARIANTS.get(model_key)
        if variant is None:
            raise ValueError(f"Unknown DINOv2 variant '{model_key}'. "
                             f"Choose from: {list(self._MODEL_VARIANTS.keys())}")

        self.embedding_dim = variant["dim"]
        print(f"  [DINOv2Engine] Loading {model_key} on {self.device} "
              f"(dim={self.embedding_dim}) ...")

        self.model = torch.hub.load("facebookresearch/dinov2", variant["hub_name"])
        self.model.to(self.device).float().eval()

        self._transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        print("  [DINOv2Engine] Ready.")

    def run(self, image: Image.Image) -> np.ndarray:
        """Encode a PIL image → (embedding_dim,) float32, L2-normalised."""
        tensor = self._transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(tensor)
        emb = embedding.cpu().numpy().flatten().astype(np.float32)
        return emb / (np.linalg.norm(emb) + 1e-8)
