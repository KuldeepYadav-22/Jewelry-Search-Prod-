"""
preprocess/image_processor.py
Handles BiRefNet → split path: white_bg (uncropped) + cropped.

Used identically by both indexing and inference pipelines.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class ProcessedImage:
    """Output of ImageProcessor.process()."""
    white_bg: Image.Image    # uncropped, white background — for CLIP
    cropped: Image.Image     # tight crop to subject — for DINOv2
    rgba: Image.Image        # raw RGBA with alpha mask


class ImageProcessor:
    """
    Single entry point for all image preprocessing.
    Runs BiRefNet once, produces two output variants.
    """

    def __init__(self, birefnet_engine, crop_padding: int = 2,
                 crop_alpha_threshold: int = 30,
                 bg_color: tuple = (255, 255, 255)):
        self.birefnet = birefnet_engine
        self.crop_padding = crop_padding
        self.crop_alpha_threshold = crop_alpha_threshold
        self.bg_color = bg_color

    def process(self, image: Image.Image) -> ProcessedImage:
        """
        Process a single image through the full preprocessing pipeline.

        Steps:
            1. BiRefNet → RGBA with alpha mask
            2. Composite on white background (uncropped) → for CLIP
            3. Crop to subject bounding box → for DINOv2

        Returns:
            ProcessedImage with white_bg, cropped, and rgba fields.
        """
        rgba = self.birefnet.get_rgba(image)

        # Path A: uncropped white background
        white_bg = Image.new("RGB", rgba.size, self.bg_color)
        white_bg.paste(rgba, mask=rgba.split()[3])

        # Path B: cropped to subject
        cropped = self._crop_to_subject(rgba)

        return ProcessedImage(white_bg=white_bg, cropped=cropped, rgba=rgba)

    def _crop_to_subject(self, rgba_image: Image.Image) -> Image.Image:
        """
        Crop tightly to non-transparent pixels.

        Uses alpha channel to find bounding box. Adds minimal padding
        and fills with bg_color.
        """
        alpha = np.array(rgba_image.split()[3])

        rows = np.any(alpha > self.crop_alpha_threshold, axis=1)
        cols = np.any(alpha > self.crop_alpha_threshold, axis=0)

        if not rows.any():
            return rgba_image.convert("RGB")

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        h, w = alpha.shape
        rmin = max(0, rmin - self.crop_padding)
        rmax = min(h, rmax + self.crop_padding)
        cmin = max(0, cmin - self.crop_padding)
        cmax = min(w, cmax + self.crop_padding)

        cropped_rgba = rgba_image.crop((cmin, rmin, cmax, rmax))
        result = Image.new("RGB", cropped_rgba.size, self.bg_color)
        result.paste(cropped_rgba, mask=cropped_rgba.split()[3])
        return result
