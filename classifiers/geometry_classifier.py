"""
classifiers/geometry_classifier.py
Geometry-based pre-classifier using BiRefNet alpha mask.
Tuned based on actual confusion matrix results.
"""
from __future__ import annotations
import numpy as np
import cv2
from PIL import Image


class GeometryClassifier:

    def classify(self, rgba_image: Image.Image) -> tuple[str | None, float]:
        features = self._extract_features(rgba_image)
        if features is None:
            return None, 0.0
        return self._apply_rules(features)

    def _extract_features(self, rgba_image: Image.Image) -> dict | None:
        try:
            alpha = np.array(rgba_image)[:, :, 3]
            img_h, img_w = alpha.shape
            img_area = img_h * img_w
            _, binary = cv2.threshold(alpha, 30, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            if area < 100:
                return None
            perimeter = cv2.arcLength(cnt, True)
            x, y, w, h = cv2.boundingRect(cnt)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            circularity = (4 * np.pi * area / (perimeter ** 2 + 1e-6))
            solidity = area / (hull_area + 1e-6)
            aspect_ratio = w / (h + 1e-6)
            relative_size = area / img_area
            elongation = max(w, h) / (min(w, h) + 1e-6)
            return {
                "circularity": circularity,
                "solidity": solidity,
                "aspect_ratio": aspect_ratio,
                "relative_size": relative_size,
                "elongation": elongation,
                "width": w,
                "height": h,
            }
        except Exception:
            return None

    def _apply_rules(self, f: dict) -> tuple[str | None, float]:
        circularity   = f["circularity"]
        solidity      = f["solidity"]
        relative_size = f["relative_size"]
        elongation    = f["elongation"]

        # Bangle: very circular + hollow + medium-large size
        # Key: solidity low means hollow center
        if (circularity > 0.60 and
                solidity < 0.72 and
                relative_size > 0.08):
            return "bangle", round(min(0.80, circularity * (1 - solidity) * 2.2), 3)

        # Ring: circular + solid + very small
        if (circularity > 0.60 and
                solidity > 0.80 and
                relative_size < 0.12):
            return "ring", round(min(0.75, circularity * solidity * 0.85), 3)

        # Anklet: thin elongated chain + small relative size
        # Anklets are thin delicate chains, more elongated than bracelets
        if (elongation > 1.8 and
                relative_size < 0.15 and
                solidity > 0.30):
            return "anklet", round(min(0.65, (elongation / 5) * 0.8), 3)

        # Chain: very elongated OR very low circularity + larger size than anklet
        if (elongation > 2.0 or circularity < 0.25):
            if relative_size > 0.15:
                return "chain", round(min(0.65, (1 - circularity) * 0.85), 3)
            else:
                return "anklet", round(min(0.60, (1 - circularity) * 0.75), 3)

        # Bracelet: medium size + flexible looking + not hollow
        if (relative_size > 0.08 and
                relative_size < 0.35 and
                solidity > 0.65 and
                circularity < 0.75):
            return "bracelet", round(min(0.60, solidity * 0.7), 3)

        # Earring: small + any shape
        if relative_size < 0.10:
            return "earring", 0.45

        return None, 0.0