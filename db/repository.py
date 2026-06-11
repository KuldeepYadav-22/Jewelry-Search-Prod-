"""
db/repository.py
Data access layer using SQLAlchemy ORM.

Provides all DB operations for both indexing and search pipelines.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sqlalchemy import select, and_, func
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert as pg_insert

from db.models import JewelryEmbedding

logger = logging.getLogger(__name__)


class JewelryRepository:
    """
    ORM-based repository for jewelry embeddings.

    Injected into IndexingPipeline and SearchPipeline.
    """

    def __init__(self, session: Session):
        self.session = session

    # ── Indexing operations ────────────────────────────────────────────

    # TODO: Add bulk insert method using COPY for large batch indexing
    #       (INSERT ... VALUES is slow for 10k+ records)
    #       Consider: session.execute(insert(JewelryEmbedding), records_list)

    # TODO: Add delete_by_serial(tenant_id, serial_number) for re-indexing
    #       a single product (delete old images, insert new ones)
    #       Query: DELETE FROM jewelry_embeddings
    #              WHERE tenant_id = :tid AND serial_number = :sn

    # TODO: Add delete_by_tenant(tenant_id) for full tenant re-index
    #       Query: DELETE FROM jewelry_embeddings WHERE tenant_id = :tid

    def upsert(self, record: dict) -> None:
        """
        Insert or update a single image record.

        Args:
            record: dict with keys matching JewelryEmbedding columns.
                    Must include: id, serial_number, tenant_id,
                    category, category_confidence, material,
                    material_confidence, clip_embedding, dino_embedding.
        """
        stmt = pg_insert(JewelryEmbedding).values(**record)
        stmt = stmt.on_conflict_do_update(
            index_elements=["id"],
            set_={
                "serial_number": stmt.excluded.serial_number,
                "category": stmt.excluded.category,
                "category_confidence": stmt.excluded.category_confidence,
                "material": stmt.excluded.material,
                "material_confidence": stmt.excluded.material_confidence,
                "clip_embedding": stmt.excluded.clip_embedding,
                "dino_embedding": stmt.excluded.dino_embedding,
                "image_url": stmt.excluded.image_url,
                "updated_at": func.now(),
            },
        )
        self.session.execute(stmt)
        self.session.commit()

    def upsert_batch(self, records: list[dict]) -> None:
        """Insert or update multiple records in a single transaction."""
        for record in records:
            stmt = pg_insert(JewelryEmbedding).values(**record)
            stmt = stmt.on_conflict_do_update(
                index_elements=["id"],
                set_={
                    "serial_number": stmt.excluded.serial_number,
                    "category": stmt.excluded.category,
                    "category_confidence": stmt.excluded.category_confidence,
                    "material": stmt.excluded.material,
                    "material_confidence": stmt.excluded.material_confidence,
                    "clip_embedding": stmt.excluded.clip_embedding,
                    "dino_embedding": stmt.excluded.dino_embedding,
                    "image_url": stmt.excluded.image_url,
                    "updated_at": func.now(),
                },
            )
            self.session.execute(stmt)
        self.session.commit()

    # ── Search operations ─────────────────────────────────────────────

    # TODO: Add pgvector approximate nearest neighbor search using HNSW index
    #       For large catalogs (600k+), replace full-table cosine similarity with:
    #       Query: SELECT * FROM jewelry_embeddings
    #              WHERE tenant_id = :tid AND category = :cat
    #              ORDER BY clip_embedding <=> :query_embedding
    #              LIMIT :pool_size
    #       This uses the HNSW index for sub-millisecond retrieval.

    # TODO: Add get_candidates_by_serial_numbers(tenant_id, serial_numbers)
    #       For fetching full records after deduplication (if needed):
    #       Query: SELECT * FROM jewelry_embeddings
    #              WHERE tenant_id = :tid AND serial_number IN (:sns)

    # TODO: Add search logging for analytics:
    #       CREATE TABLE search_logs (
    #           id SERIAL PRIMARY KEY,
    #           tenant_id TEXT,
    #           query_category TEXT,
    #           query_material TEXT,
    #           n_candidates INT,
    #           n_results INT,
    #           top_score FLOAT,
    #           latency_ms FLOAT,
    #           created_at TIMESTAMP DEFAULT NOW()
    #       )

    def get_candidates(
        self,
        tenant_id: str,
        category: Optional[str] = None,
        material: Optional[str] = None,
    ) -> dict:
        """
        Fetch candidate images for similarity search.

        Filters by tenant_id (always), category (optional),
        and material (optional).

        Args:
            tenant_id: tenant identifier (required).
            category: if provided, filter WHERE category = category.
            material: if provided, filter WHERE material = material.

        Returns:
            dict with:
                ids:                  list[str]
                serial_numbers:       list[str]
                clip_embeddings:      np.ndarray (N, 768)
                dino_embeddings:      np.ndarray (N, 1024)
                categories:           list[str]
                materials:            list[str]
                category_confidences: list[float]
                material_confidences: list[float]
                image_urls:           list[str]
        """
        # Build query
        conditions = [JewelryEmbedding.tenant_id == tenant_id]

        if category is not None:
            conditions.append(JewelryEmbedding.category == category)

        if material is not None:
            conditions.append(JewelryEmbedding.material == material)

        stmt = (
            select(JewelryEmbedding)
            .where(and_(*conditions))
        )

        rows = self.session.execute(stmt).scalars().all()

        if not rows:
            return {
                "ids": [],
                "serial_numbers": [],
                "clip_embeddings": np.empty((0, 768), dtype=np.float32),
                "dino_embeddings": np.empty((0, 1024), dtype=np.float32),
                "categories": [],
                "materials": [],
                "category_confidences": [],
                "material_confidences": [],
                "image_urls": [],
            }

        return {
            "ids":                  [r.id for r in rows],
            "serial_numbers":       [r.serial_number for r in rows],
            "clip_embeddings":      np.array(
                                        [r.clip_embedding for r in rows],
                                        dtype=np.float32,
                                    ),
            "dino_embeddings":      np.array(
                                        [r.dino_embedding for r in rows],
                                        dtype=np.float32,
                                    ),
            "categories":           [r.category for r in rows],
            "materials":            [r.material for r in rows],
            "category_confidences": [r.category_confidence for r in rows],
            "material_confidences": [r.material_confidence for r in rows],
            "image_urls":           [r.image_url or "" for r in rows],
        }

    def count_by_tenant(self, tenant_id: str) -> dict:
        """
        Get image and product counts for a tenant.

        Returns:
            dict with total_images, unique_products, categories, materials.
        """
        base = select(JewelryEmbedding).where(
            JewelryEmbedding.tenant_id == tenant_id
        )

        total = self.session.execute(
            select(func.count()).select_from(base.subquery())
        ).scalar()

        unique_products = self.session.execute(
            select(func.count(func.distinct(JewelryEmbedding.serial_number)))
            .where(JewelryEmbedding.tenant_id == tenant_id)
        ).scalar()

        cat_counts = self.session.execute(
            select(
                JewelryEmbedding.category,
                func.count().label("count"),
            )
            .where(JewelryEmbedding.tenant_id == tenant_id)
            .group_by(JewelryEmbedding.category)
        ).all()

        mat_counts = self.session.execute(
            select(
                JewelryEmbedding.material,
                func.count().label("count"),
            )
            .where(JewelryEmbedding.tenant_id == tenant_id)
            .group_by(JewelryEmbedding.material)
        ).all()

        return {
            "total_images": total,
            "unique_products": unique_products,
            "categories": {row[0]: row[1] for row in cat_counts},
            "materials": {row[0]: row[1] for row in mat_counts},
        }

    # TODO: Add get_product_images(tenant_id, serial_number) to fetch
    #       all images for a single product (for detail view after search):
    #       Query: SELECT * FROM jewelry_embeddings
    #              WHERE tenant_id = :tid AND serial_number = :sn
    #              ORDER BY created_at

