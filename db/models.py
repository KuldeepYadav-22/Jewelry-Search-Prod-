"""
db/models.py
SQLAlchemy ORM model for jewelry embeddings table.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Column, String, Float, DateTime, Text, Index,
)
from sqlalchemy.orm import DeclarativeBase
from pgvector.sqlalchemy import Vector


class Base(DeclarativeBase):
    pass


class JewelryEmbedding(Base):
    """
    Stores one image's embeddings and classification.
    A single product (serial_number) may have multiple rows
    (one per orientation/photo).
    """
    __tablename__ = "jewelry_embeddings"

    id              = Column(String, primary_key=True)  # unique image id (uuid)
    serial_number   = Column(String, nullable=False, index=True)
    tenant_id       = Column(String, nullable=False, index=True)
    category        = Column(String)
    category_confidence = Column(Float)
    material        = Column(String)
    material_confidence = Column(Float)
    clip_embedding  = Column(Vector(768))
    dino_embedding  = Column(Vector(1024))
    image_url       = Column(Text)
    created_at      = Column(DateTime, default=datetime.utcnow)
    updated_at      = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # TODO: Add deleted_at column for soft delete support:
    # deleted_at    = Column(DateTime, nullable=True)
    # Then add to get_candidates: .where(JewelryEmbedding.deleted_at.is_(None))

    # TODO: Add stock_id or variant_id if products have variants:
    # stock_id      = Column(String, nullable=True)

    __table_args__ = (
        Index("idx_tenant_category", "tenant_id", "category"),
        Index("idx_tenant_material", "tenant_id", "material"),
        Index("idx_tenant_serial", "tenant_id", "serial_number"),
        # TODO: Add HNSW indexes for pgvector approximate nearest neighbor search.
        #       These cannot be created via SQLAlchemy and must be run manually:
        #
        #       CREATE INDEX idx_clip_hnsw ON jewelry_embeddings
        #           USING hnsw (clip_embedding vector_ip_ops)
        #           WITH (m=24, ef_construction=128);
        #
        #       CREATE INDEX idx_dino_hnsw ON jewelry_embeddings
        #           USING hnsw (dino_embedding vector_ip_ops)
        #           WITH (m=24, ef_construction=128);
    )

    def __repr__(self):
        return (f"<JewelryEmbedding(serial_number={self.serial_number!r}, "
                f"category={self.category!r}, material={self.material!r})>")
