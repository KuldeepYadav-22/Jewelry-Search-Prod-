"""
qa_app/model_loader.py
Loads every model exactly once at Flask startup:
  - BiRefNet, CLIP, DINOv2 (shared engines, download-if-missing as usual)
  - The zero-shot CLIP category classifier (reused only as the is_jewelry gate)
  - The trained SupervisedCategoryClassifier (fails fast if its artifacts
    are missing or invalid — the app must not partially start).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from classifiers.category_classifier import CategoryClassifier
from classifiers.supervised_category_classifier import SupervisedCategoryClassifier
from engines.birefnet_engine import BiRefNetEngine
from engines.clip_engine import CLIPEngine
from engines.dinov2_engine import DINOv2Engine
from preprocess.image_processor import ImageProcessor

logger = logging.getLogger(__name__)

REQUIRED_CLASSIFIER_FILES = [
    "category_classifier.xgb",
    "label_encoder.joblib",
    "model_metadata.json",
]


@dataclass
class ModelBundle:
    birefnet: BiRefNetEngine
    clip: CLIPEngine
    dinov2: DINOv2Engine
    zero_shot_gate: CategoryClassifier
    supervised_clf: SupervisedCategoryClassifier
    processor: ImageProcessor
    device: torch.device
    model_type: str
    embedding_config: str
    target_classes: list[str]


def _check_classifier_artifacts(model_dir: Path) -> None:
    missing = [f for f in REQUIRED_CLASSIFIER_FILES if not (model_dir / f).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Cannot start QA app — missing trained classifier artifact(s) in "
            f"'{model_dir}': {', '.join(missing)}. Place all three files "
            f"({', '.join(REQUIRED_CLASSIFIER_FILES)}) in that folder before starting."
        )


def _describe_local_path(path_str: str) -> str:
    return ("found locally" if Path(path_str).exists()
            else "not found locally — will attempt to download")


def _describe_torch_hub_cache() -> str:
    try:
        hub_dir = Path(torch.hub.get_dir())
    except Exception:
        return "unknown"
    return ("torch hub cache present" if any(hub_dir.glob("facebookresearch_dinov2*"))
            else "torch hub cache not found — will attempt to download")


def load_models(inference_config, prompts: dict, model_dir: str | Path) -> ModelBundle:
    """
    Load BiRefNet, CLIP, DINOv2, the zero-shot gate, and the trained
    supervised classifier once. Raises immediately (no partial startup) if
    the three trained-classifier artifacts are missing or unreadable.
    """
    model_dir = Path(model_dir)

    logger.info("Checking trained classifier artifacts in '%s' ...", model_dir)
    _check_classifier_artifacts(model_dir)

    logger.info("Loading BiRefNet (%s) ...", _describe_local_path(
        getattr(inference_config, "birefnet_model_path", "models/BiRefNet")))
    t0 = time.perf_counter()
    birefnet = BiRefNetEngine(inference_config)
    t_birefnet = time.perf_counter() - t0

    logger.info("Loading CLIP (%s) ...", _describe_local_path(
        getattr(inference_config, "clip_model_path", "models/clip-ViT-L-14")))
    t0 = time.perf_counter()
    clip = CLIPEngine(inference_config)
    t_clip = time.perf_counter() - t0

    logger.info("Loading DINOv2 (%s) ...", _describe_torch_hub_cache())
    t0 = time.perf_counter()
    dinov2 = DINOv2Engine(inference_config)
    t_dinov2 = time.perf_counter() - t0

    logger.info("Loading zero-shot CLIP gate (classifiers/category_classifier.py) ...")
    zero_shot_gate = CategoryClassifier(
        clip, prompts["categories"],
        threshold=getattr(inference_config, "category_confidence_threshold", 0.20),
    )

    logger.info("Loading trained supervised category classifier from '%s' ...", model_dir)
    try:
        # threshold=0.0: always surface the true top-1 of the 10 fixed
        # categories instead of "unknown" — the is_jewelry gate already
        # carries the low-trust signal, and the UI shows the full
        # confidence breakdown so QA can judge trust for themselves. The
        # correction dropdown only offers the 10 fixed classes, so the
        # top prediction must always be one of them.
        supervised_clf = SupervisedCategoryClassifier(model_dir=model_dir, threshold=0.0)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Cannot start QA app — '{model_dir / 'model_metadata.json'}' exists but "
            f"is not valid JSON ({e}). Re-export/re-upload this file before starting."
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Cannot start QA app — failed to load classifier artifacts from "
            f"'{model_dir}': {e}"
        ) from e

    processor = ImageProcessor(
        crop_padding=getattr(inference_config, "crop_padding", 2),
        crop_alpha_threshold=getattr(inference_config, "crop_alpha_threshold", 30),
        bg_color=tuple(getattr(inference_config, "bg_color", (255, 255, 255))),
    )

    target_classes = list(supervised_clf.meta.get(
        "target_classes", list(supervised_clf.le.classes_)))

    bundle = ModelBundle(
        birefnet=birefnet, clip=clip, dinov2=dinov2,
        zero_shot_gate=zero_shot_gate, supervised_clf=supervised_clf,
        processor=processor, device=birefnet.device,
        model_type=supervised_clf.model_type,
        embedding_config=supervised_clf.embedding_config,
        target_classes=target_classes,
    )

    logger.info(
        "Startup summary: device=%s | BiRefNet %.1fs | CLIP %.1fs | DINOv2 %.1fs | "
        "trained classifier: %s + %s (%d classes: %s)",
        bundle.device, t_birefnet, t_clip, t_dinov2,
        bundle.model_type, bundle.embedding_config,
        len(target_classes), ", ".join(target_classes),
    )
    return bundle
