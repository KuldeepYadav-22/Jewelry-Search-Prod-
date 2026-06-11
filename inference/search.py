"""
inference/search.py
Pool-based similarity search with threshold pools and serial deduplication.

The input image is expected to be RGBA with background already removed
on-device. BiRefNet is no longer called here.

Architecture:
    RGBA Image → split path (white_bg + crop)
      ├→ white bg (uncropped) → CLIP → classify category + material
      └→ crop to subject      → DINOv2 → visual embedding

    Filter candidates by category + material (via DB ORM)
    Threshold-based pools → merge → dedup by serial_number → top-K
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

from classifiers.category_classifier import CategoryClassifier
from classifiers.material_classifier import MaterialClassifier
from engines.clip_engine import CLIPEngine
from engines.dinov2_engine import DINOv2Engine
from preprocess.image_processor import ImageProcessor
from db.repository import JewelryRepository

# ──────────────────────────────────────────────────────────────────────────────
# BiRefNet is no longer used — background removal happens on-device.
# Uncomment if you need to re-enable server-side BG removal:
#
# from engines.birefnet_engine import BiRefNetEngine
# ──────────────────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Single search result — one unique product."""
    serial_number: str
    rank: int
    score: float
    clip_score: float
    dino_score: float
    pool: str              # "both", "dino", "clip"
    category: str
    material: str
    image_url: str = ""


@dataclass
class SearchResponse:
    """Complete search response."""
    results: list[SearchResult]
    query_category: str
    query_confidence: float
    query_material: str
    material_confidence: float
    n_images: int              # total candidate images after filtering
    n_products: int            # unique products after filtering
    pool_stats: dict
    timing_ms: dict
    rejected: bool = False
    rejected_reason: Optional[str] = None


@dataclass
class SearchConfig:
    """Search parameters — loaded from config.yaml or overridden per request."""
    top_k: int = 10
    clip_pool_threshold: float = 0.50
    dino_pool_threshold: float = 0.30
    final_threshold: float = 0.50
    pool_fallback_size: int = 20
    use_category_filter: bool = True
    use_material_filter: bool = True
    material_confidence_threshold: float = 0.20
    category_confidence_threshold: float = 0.20
    dino_only_penalty: float = 0.9
    clip_only_penalty: float = 0.85

    @classmethod
    def from_dict(cls, d: dict) -> "SearchConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class SearchPipeline:
    """
    Threshold-pool search with serial deduplication.

    Reuses the same engines, classifiers, and preprocessor as indexing.
    Input is RGBA (background already removed on-device).
    """

    def __init__(
        self,
        image_processor: ImageProcessor,
        clip_engine: CLIPEngine,
        dinov2_engine: DINOv2Engine,
        category_classifier: CategoryClassifier,
        material_classifier: MaterialClassifier,
        repository: JewelryRepository,
        config: SearchConfig = None,
        # ──────────────────────────────────────────────────────────────
        # BiRefNet no longer needed. If re-enabling:
        # birefnet_engine: BiRefNetEngine = None,
        # ──────────────────────────────────────────────────────────────
    ):
        self.processor = image_processor
        self.clip = clip_engine
        self.dinov2 = dinov2_engine
        self.cat_clf = category_classifier
        self.mat_clf = material_classifier
        self.repo = repository
        self.config = config or SearchConfig()

    def search(
        self,
        query_image: Image.Image,
        tenant_id: str,
        config: SearchConfig = None,
    ) -> SearchResponse:
        """
        Run threshold-pool search with serial deduplication.

        Args:
            query_image: PIL image in RGBA mode (background already removed on-device).
                         RGB images are accepted and converted with full opacity.
            tenant_id: tenant identifier.
            config: optional per-request config override.

        Returns:
            SearchResponse with ranked unique products.
        """
        cfg = config or self.config
        timing = {}
        t_total = time.perf_counter()

        # ── Stage 1: Preprocess (RGBA → white_bg + crop) ─────────────
        # BiRefNet is skipped — BG removal already done on-device.
        # Previously:
        #   t0 = time.perf_counter()
        #   rgba = self.birefnet.get_rgba(query_image)
        #   timing["birefnet_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        t0 = time.perf_counter()
        processed = self.processor.process(query_image)
        timing["preprocess_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # ── Stage 2: CLIP embedding (uncropped white-bg) ─────────────
        t0 = time.perf_counter()
        clip_emb = self.clip.run(processed.white_bg)
        timing["clip_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # ── Stage 3: Classify ─────────────────────────────────────────
        t0 = time.perf_counter()
        norm_clip = clip_emb / (np.linalg.norm(clip_emb) + 1e-8)
        q_cat, q_cat_conf, _ = self.cat_clf.classify(norm_clip)
        q_mat, q_mat_conf, _ = self.mat_clf.classify(norm_clip)
        timing["classify_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # Paper rejection
        if q_cat == "paper":
            timing["total_ms"] = round((time.perf_counter() - t_total) * 1000, 2)
            return SearchResponse(
                results=[], query_category=q_cat, query_confidence=q_cat_conf,
                query_material=q_mat, material_confidence=q_mat_conf,
                n_images=0, n_products=0, pool_stats={}, timing_ms=timing,
                rejected=True, rejected_reason="Image classified as paper/document.",
            )

        # ── Stage 4: DINOv2 embedding (cropped) ──────────────────────
        t0 = time.perf_counter()
        dino_emb = self.dinov2.run(processed.cropped)
        timing["dinov2_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # ── Stage 5: Fetch candidates from DB ─────────────────────────
        # TODO: If you need custom SQL filtering for category + material

        t0 = time.perf_counter()

        cat_filter = None
        if cfg.use_category_filter and q_cat not in ("unknown", "other"):
            cat_filter = q_cat

        mat_filter = None
        if cfg.use_material_filter and q_mat_conf >= cfg.material_confidence_threshold:
            mat_filter = q_mat

        candidates = self.repo.get_candidates(
            tenant_id=tenant_id,
            category=cat_filter,
            material=mat_filter,
        )
        n_images = len(candidates["serial_numbers"])

        # Fallback: if material filter too restrictive
        if n_images < 3 and mat_filter is not None:
            # TODO: Consider logging material fallback events for monitoring
            #       to track how often the material classifier is unreliable
            candidates = self.repo.get_candidates(
                tenant_id=tenant_id,
                category=cat_filter,
                material=None,
            )
            n_images = len(candidates["serial_numbers"])
            q_mat = q_mat + " (fallback)"

        timing["db_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        if n_images == 0:
            timing["total_ms"] = round((time.perf_counter() - t_total) * 1000, 2)
            return SearchResponse(
                results=[], query_category=q_cat, query_confidence=q_cat_conf,
                query_material=q_mat, material_confidence=q_mat_conf,
                n_images=0, n_products=0, pool_stats={}, timing_ms=timing,
            )

        n_products = len(set(candidates["serial_numbers"]))

        # ── Stage 6: Threshold-based pools ────────────────────────────
        t0 = time.perf_counter()

        clip_scores = self._cosine_similarity(clip_emb, candidates["clip_embeddings"])
        dino_scores = self._cosine_similarity(dino_emb, candidates["dino_embeddings"])

        clip_pool = set(np.where(clip_scores >= cfg.clip_pool_threshold)[0])
        dino_pool = set(np.where(dino_scores >= cfg.dino_pool_threshold)[0])

        # Fallback: if both pools empty, take top-N from each
        if len(clip_pool) == 0 and len(dino_pool) == 0:
            fallback_n = min(cfg.pool_fallback_size, n_images)
            clip_pool = set(np.argsort(clip_scores)[::-1][:fallback_n])
            dino_pool = set(np.argsort(dino_scores)[::-1][:fallback_n])

        merged = clip_pool | dino_pool
        in_both = clip_pool & dino_pool

        # ── Stage 7: Score + deduplicate ──────────────────────────────

        # Score each image in merged pool
        scored_images = []
        for idx in merged:
            cs = float(clip_scores[idx])
            ds = float(dino_scores[idx])

            if idx in in_both:
                score = (cs + ds) / 2
                pool_tag = "both"
            elif idx in dino_pool:
                score = ds * cfg.dino_only_penalty
                pool_tag = "dino"
            else:
                score = cs * cfg.clip_only_penalty
                pool_tag = "clip"

            scored_images.append({
                "idx": idx,
                "serial_number": candidates["serial_numbers"][idx],
                "score": score,
                "clip_score": cs,
                "dino_score": ds,
                "pool": pool_tag,
                "category": candidates["categories"][idx],
                "material": candidates["materials"][idx],
                "image_url": candidates["image_urls"][idx],
            })

        # Deduplicate: keep best scoring image per serial_number
        best_per_product = {}
        for item in scored_images:
            sn = item["serial_number"]
            if sn not in best_per_product or item["score"] > best_per_product[sn]["score"]:
                best_per_product[sn] = item

        # ── Stage 8: Final threshold + top-K ──────────────────────────
        # TODO: If you need to log or store search results for analytics,
        #       add a repository method like repo.log_search() here with
        #       query metadata, result serial_numbers, and scores

        ranked = sorted(best_per_product.values(), key=lambda x: x["score"], reverse=True)

        results = []
        rank = 0
        for item in ranked:
            if item["score"] < cfg.final_threshold:
                continue
            rank += 1
            if rank > cfg.top_k:
                break
            results.append(SearchResult(
                serial_number=item["serial_number"],
                rank=rank,
                score=round(item["score"], 4),
                clip_score=round(item["clip_score"], 4),
                dino_score=round(item["dino_score"], 4),
                pool=item["pool"],
                category=item["category"],
                material=item["material"],
                image_url=item["image_url"],
            ))

        timing["search_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        timing["total_ms"] = round((time.perf_counter() - t_total) * 1000, 2)

        pool_stats = {
            "clip_pool_size": len(clip_pool),
            "dino_pool_size": len(dino_pool),
            "merged_images": len(merged),
            "in_both": len(in_both),
            "unique_products_in_pool": len(best_per_product),
            "products_above_threshold": len(results),
        }

        return SearchResponse(
            results=results,
            query_category=q_cat,
            query_confidence=round(q_cat_conf, 4),
            query_material=q_mat,
            material_confidence=round(q_mat_conf, 4),
            n_images=n_images,
            n_products=n_products,
            pool_stats=pool_stats,
            timing_ms=timing,
        )

    @staticmethod
    def _cosine_similarity(query_vec: np.ndarray,
                            matrix: np.ndarray) -> np.ndarray:
        qn = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        mn = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
        return mn @ qn
