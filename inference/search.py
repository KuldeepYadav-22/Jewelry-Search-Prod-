"""
inference/search.py
Pool-based similarity search pipeline.

Takes a query image + tenant_id, returns ranked results with
serial_number, score, and rank.

Architecture:
    Query → BiRefNet → split path
      ├→ white bg (uncropped) → CLIP → classify category + material
      └→ crop to subject      → DINOv2 → visual embedding

    Filter candidates by category + material (via DB)

    Pool-based ranking:
      CLIP pool:   top-N by semantic similarity
      DINOv2 pool: top-N by visual similarity
      Merge → score → threshold → top-K results
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image

from classifiers.category_classifier import CategoryClassifier
from classifiers.material_classifier import MaterialClassifier
from engines.clip_engine import CLIPEngine
from engines.dinov2_engine import DINOv2Engine
from preprocess.image_processor import ImageProcessor

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Single search result."""
    serial_number: str
    rank: int
    score: float
    clip_score: float
    dino_score: float
    pool: str           # "both", "dino", "clip"
    category: str
    material: str


@dataclass
class SearchResponse:
    """Complete search response."""
    results: list[SearchResult]
    query_category: str
    query_confidence: float
    query_material: str
    material_confidence: float
    n_candidates: int
    pool_stats: dict
    timing_ms: dict
    rejected: bool = False
    rejected_reason: Optional[str] = None


@dataclass
class SearchConfig:
    """Search parameters — loaded from config.yaml or overridden per request."""
    top_k: int = 10
    pool_size: int = 10
    final_threshold: float = 0.7
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
    Pool-based similarity search.

    Uses the same engines, classifiers, and preprocessor as the indexing
    pipeline — initialized once, shared across requests.
    """

    def __init__(
        self,
        image_processor: ImageProcessor,
        clip_engine: CLIPEngine,
        dinov2_engine: DINOv2Engine,
        category_classifier: CategoryClassifier,
        material_classifier: MaterialClassifier,
        db_repo,
        config: SearchConfig = None,
    ):
        self.processor = image_processor
        self.clip = clip_engine
        self.dinov2 = dinov2_engine
        self.cat_clf = category_classifier
        self.mat_clf = material_classifier
        self.db = db_repo
        self.config = config or SearchConfig()

    def search(
        self,
        query_image: Image.Image,
        tenant_id: str,
        config: SearchConfig = None,
    ) -> SearchResponse:
        """
        Run pool-based similarity search.

        Args:
            query_image: PIL image (RGB).
            tenant_id: tenant identifier.
            config: optional per-request config override.

        Returns:
            SearchResponse with ranked results and metadata.
        """
        cfg = config or self.config
        timing = {}
        total_start = time.perf_counter()

        if query_image.mode != "RGB":
            query_image = query_image.convert("RGB")

        # ── Stage 1: Preprocess (BiRefNet → split path) ──────────────
        t0 = time.perf_counter()
        processed = self.processor.process(query_image)
        timing["birefnet_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # ── Stage 2: CLIP embedding (uncropped white-bg) ─────────────
        t0 = time.perf_counter()
        clip_emb = self.clip.run(processed.white_bg)
        timing["clip_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # ── Stage 3: Classify category + material ────────────────────
        t0 = time.perf_counter()
        norm_clip = clip_emb / (np.linalg.norm(clip_emb) + 1e-8)
        q_cat, q_cat_conf, _ = self.cat_clf.classify(norm_clip)
        q_mat, q_mat_conf, _ = self.mat_clf.classify(norm_clip)
        timing["classify_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # Paper rejection
        if q_cat == "paper":
            timing["total_ms"] = round((time.perf_counter() - total_start) * 1000, 2)
            return SearchResponse(
                results=[], query_category=q_cat, query_confidence=q_cat_conf,
                query_material=q_mat, material_confidence=q_mat_conf,
                n_candidates=0, pool_stats={}, timing_ms=timing,
                rejected=True, rejected_reason="Image classified as paper/document.",
            )

        # ── Stage 4: DINOv2 embedding (cropped) ──────────────────────
        t0 = time.perf_counter()
        dino_emb = self.dinov2.run(processed.cropped)
        timing["dinov2_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # ── Stage 5: Fetch candidates from DB ────────────────────────
        t0 = time.perf_counter()

        # Determine filters
        cat_filter = None
        if cfg.use_category_filter and q_cat not in ("unknown", "other"):
            cat_filter = q_cat

        mat_filter = None
        if cfg.use_material_filter and q_mat_conf >= cfg.material_confidence_threshold:
            mat_filter = q_mat

        candidates = self.db.get_candidates(
            tenant_id=tenant_id,
            category=cat_filter,
            material=mat_filter,
        )

        n_candidates = len(candidates["serial_numbers"])

        # Fallback: if material filter too restrictive
        if n_candidates < 3 and mat_filter is not None:
            candidates = self.db.get_candidates(
                tenant_id=tenant_id,
                category=cat_filter,
                material=None,
            )
            n_candidates = len(candidates["serial_numbers"])
            q_mat = q_mat + " (fallback)"

        timing["db_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        if n_candidates == 0:
            timing["total_ms"] = round((time.perf_counter() - total_start) * 1000, 2)
            return SearchResponse(
                results=[], query_category=q_cat, query_confidence=q_cat_conf,
                query_material=q_mat, material_confidence=q_mat_conf,
                n_candidates=0, pool_stats={}, timing_ms=timing,
            )

        # ── Stage 6: Pool-based scoring ──────────────────────────────
        t0 = time.perf_counter()

        cand_clip = candidates["clip_embeddings"]     # (N, 768)
        cand_dino = candidates["dino_embeddings"]      # (N, 1024)

        clip_scores = self._cosine_similarity(clip_emb, cand_clip)
        dino_scores = self._cosine_similarity(dino_emb, cand_dino)

        # Two independent pools
        pool_n = min(cfg.pool_size, n_candidates)
        clip_pool = set(np.argsort(clip_scores)[::-1][:pool_n])
        dino_pool = set(np.argsort(dino_scores)[::-1][:pool_n])

        # Merge
        merged = clip_pool | dino_pool
        in_both = clip_pool & dino_pool

        # Score each candidate
        scored = []
        for idx in merged:
            cs = float(clip_scores[idx])
            ds = float(dino_scores[idx])

            if idx in in_both:
                rank_score = (cs + ds) / 2
                pool_tag = "both"
            elif idx in dino_pool:
                rank_score = ds * cfg.dino_only_penalty
                pool_tag = "dino"
            else:
                rank_score = cs * cfg.clip_only_penalty
                pool_tag = "clip"

            scored.append((idx, rank_score, cs, ds, pool_tag))

        scored.sort(key=lambda x: x[1], reverse=True)
        timing["search_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # ── Stage 7: Threshold + top-K ───────────────────────────────
        results = []
        rank = 0
        for idx, score, cs, ds, pool_tag in scored:
            if score < cfg.final_threshold:
                continue
            rank += 1
            if rank > cfg.top_k:
                break

            results.append(SearchResult(
                serial_number=candidates["serial_numbers"][idx],
                rank=rank,
                score=round(score, 4),
                clip_score=round(cs, 4),
                dino_score=round(ds, 4),
                pool=pool_tag,
                category=candidates["categories"][idx],
                material=candidates["materials"][idx],
            ))

        timing["total_ms"] = round((time.perf_counter() - total_start) * 1000, 2)

        pool_stats = {
            "clip_pool_size": len(clip_pool),
            "dino_pool_size": len(dino_pool),
            "merged_size": len(merged),
            "in_both": len(in_both),
        }

        return SearchResponse(
            results=results,
            query_category=q_cat,
            query_confidence=round(q_cat_conf, 4),
            query_material=q_mat,
            material_confidence=round(q_mat_conf, 4),
            n_candidates=n_candidates,
            pool_stats=pool_stats,
            timing_ms=timing,
        )

    @staticmethod
    def _cosine_similarity(query_vec: np.ndarray,
                            matrix: np.ndarray) -> np.ndarray:
        """Cosine similarity between one vector and a matrix of vectors."""
        qn = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        mn = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
        return mn @ qn
