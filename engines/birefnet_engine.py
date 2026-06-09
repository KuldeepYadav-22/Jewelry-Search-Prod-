"""
engines/birefnet_engine.py
Background removal using BiRefNet.

Key methods:
- run(image)      -> PIL image composited onto blue background (legacy).
- get_rgba(image) -> RGBA PIL image with alpha mask (used by pipeline).
- get_mask(image) -> Grayscale PIL mask.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

from engines.base import BaseEngine


class BiRefNetEngine(BaseEngine):

    def __init__(self, config):
        super().__init__(config)

        model_path = getattr(config, "birefnet_model_path", "models/BiRefNet")
        self._use_fp16 = getattr(config, "use_fp16", False)
        self._vfx_blue = (0, 0, 255)

        print(f"  [BiRefNetEngine] Loading from '{model_path}' on {self.device} "
              f"(fp16={self._use_fp16}) ...")

        self.model = AutoModelForImageSegmentation.from_pretrained(
            model_path, trust_remote_code=True
        )
        if self._use_fp16:
            self.model.half()
        else:
            self.model.float()
        self.model.to(self.device).eval()

        self._transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        print("  [BiRefNetEngine] Ready.")

    def run(self, image: Image.Image) -> Image.Image:
        """Remove background, composite onto blue."""
        mask_pil = self.get_mask(image)
        return self._composite(image, mask_pil, self._vfx_blue)

    def get_rgba(self, image: Image.Image) -> Image.Image:
        """Remove background, return RGBA with alpha mask."""
        mask_pil = self.get_mask(image)
        rgba = image.convert("RGBA")
        rgba.putalpha(mask_pil)
        return rgba

    def get_mask(self, image: Image.Image) -> Image.Image:
        """Return grayscale segmentation mask."""
        input_tensor = self._transform(image).unsqueeze(0).to(self.device)
        if self._use_fp16:
            input_tensor = input_tensor.half()

        with torch.no_grad():
            preds = self.model(input_tensor)[-1].sigmoid()

        mask = preds[0].squeeze().float()
        mask = torch.nn.functional.interpolate(
            mask.unsqueeze(0).unsqueeze(0),
            size=(image.height, image.width),
            mode="bilinear",
            align_corners=False,
        ).squeeze()

        mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(mask_np)

    @staticmethod
    def _composite(image, mask, bg_color):
        background = Image.new("RGB", image.size, bg_color)
        background.paste(image, mask=mask)
        return background
