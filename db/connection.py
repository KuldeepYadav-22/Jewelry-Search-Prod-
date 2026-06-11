"""
db/connection.py
SQLAlchemy engine and session factory.

Usage:
    from db.connection import get_engine, get_session

    engine = get_engine("postgresql://user:pass@host:5432/dbname")
    session = get_session(engine)
"""
from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session

from db.models import Base


def get_engine(
    database_url: str = None,
    pool_size: int = 5,
    max_overflow: int = 10,
    echo: bool = False,
) -> Engine:
    """
    Create a SQLAlchemy engine with connection pooling.

    Args:
        database_url: PostgreSQL connection string.
                      Falls back to DATABASE_URL env var.
        pool_size: Number of persistent connections in the pool.
        max_overflow: Max additional connections beyond pool_size.
        echo: If True, log all SQL statements.

    Returns:
        SQLAlchemy Engine instance.
    """
    url = database_url or os.environ.get("DATABASE_URL")
    if not url:
        raise ValueError(
            "No database URL provided. Pass database_url or set DATABASE_URL env var."
        )

    engine = create_engine(
        url,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=True,   # verify connections before use
        echo=echo,
    )
    return engine


def get_session(engine: Engine) -> Session:
    """Create a new session from the engine."""
    SessionFactory = sessionmaker(bind=engine)
    return SessionFactory()


def create_tables(engine: Engine) -> None:
    """Create all tables if they don't exist."""
    Base.metadata.create_all(engine)
