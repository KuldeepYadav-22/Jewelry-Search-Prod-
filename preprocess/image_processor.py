"""
preprocess/image_processor.py
Handles image preprocessing: RGBA → white_bg (uncropped) + cropped.

The input image is expected to be RGBA with background already removed
on-device. BiRefNet is no longer called here.

Used identically by both indexing and inference pipelines.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────────
# NOTE: BiRefNet background removal is now handled on-device before the image
# reaches this pipeline. The input to process() is an RGBA image with
# transparent background.
#
# If you need to re-enable server-side background removal, uncomment the
# BiRefNet import and the _remove_background() method, then call it
# inside process() before the white_bg and crop steps.
# ──────────────────────────────────────────────────────────────────────────────

# from engines.birefnet_engine import BiRefNetEngine


@dataclass
class ProcessedImage:
    """Output of ImageProcessor.process()."""
    white_bg: Image.Image    # uncropped, white background — for CLIP
    cropped: Image.Image     # tight crop to subject — for DINOv2
    rgba: Image.Image        # original RGBA input


class ImageProcessor:
    """
    Single entry point for all image preprocessing.
    Accepts RGBA input (background already removed on-device),
    produces two output variants for CLIP and DINOv2.
    """

    def __init__(self, crop_padding: int = 2,
                 crop_alpha_threshold: int = 30,
                 bg_color: tuple = (255, 255, 255)):
        # ──────────────────────────────────────────────────────────────
        # BiRefNet is no longer needed since BG removal happens on-device.
        # If you need to re-enable it, uncomment and pass birefnet_engine:
        #
        # def __init__(self, birefnet_engine, crop_padding=2, ...):
        #     self.birefnet = birefnet_engine
        # ──────────────────────────────────────────────────────────────
        self.crop_padding = crop_padding
        self.crop_alpha_threshold = crop_alpha_threshold
        self.bg_color = bg_color

    def process(self, image: Image.Image) -> ProcessedImage:
        """
        Process a single RGBA image through the preprocessing pipeline.

        Args:
            image: PIL Image in RGBA mode (background already removed on-device).
                   If RGB is passed, it will be converted to RGBA with full opacity.

        Steps:
            1. Ensure RGBA format
            2. Composite on white background (uncropped) → for CLIP
            3. Crop to subject bounding box → for DINOv2

        Returns:
            ProcessedImage with white_bg, cropped, and rgba fields.
        """
        # ──────────────────────────────────────────────────────────────
        # Previously BiRefNet was called here to get RGBA:
        #   rgba = self.birefnet.get_rgba(image)
        # Now the input IS already RGBA from on-device processing.
        # ──────────────────────────────────────────────────────────────

        if image.mode != "RGBA":
            # Fallback: if RGB is passed, add full opacity alpha channel
            rgba = image.convert("RGBA")
        else:
            rgba = image

        # Path A: uncropped white background (for CLIP)
        white_bg = Image.new("RGB", rgba.size, self.bg_color)
        white_bg.paste(rgba, mask=rgba.split()[3])

        # Path B: cropped to subject (for DINOv2)
        cropped = self._crop_to_subject(rgba)

        return ProcessedImage(white_bg=white_bg, cropped=cropped, rgba=rgba)

    # ──────────────────────────────────────────────────────────────────
    # Uncomment to re-enable server-side background removal:
    #
    # def _remove_background(self, image: Image.Image) -> Image.Image:
    #     """Run BiRefNet to get RGBA with alpha mask."""
    #     return self.birefnet.get_rgba(image)
    # ──────────────────────────────────────────────────────────────────

    def _crop_to_subject(self, rgba_image: Image.Image) -> Image.Image:
        """
        Crop tightly to non-transparent pixels.
        Uses alpha channel to find bounding box.
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
