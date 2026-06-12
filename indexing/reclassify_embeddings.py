"""
indexing/reclassify_embeddings.py
Reclassify precomputed CLIP embeddings and return DB-ready records.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from classifiers.category_classifier import CategoryClassifier
from classifiers.material_classifier import MaterialClassifier
from engines.clip_engine import CLIPEngine


DEFAULT_EMBEDDING_DIM = 768


@dataclass
class ReclassificationRecord:
    """DB-ready record for a reclassified embedding row."""

    serial_number: Optional[str]
    tenant_id: Optional[str]
    category: str
    category_confidence: float
    material: str
    material_confidence: float
    error: Optional[str] = None


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _build_clip_engine(config_dir: Path) -> tuple[CLIPEngine, CategoryClassifier, MaterialClassifier]:
    config_data = _load_yaml(config_dir / "config.yaml")
    prompts_data = _load_yaml(config_dir / "prompts.yaml")

    inference_config = SimpleNamespace(**config_data.get("inference", {}))
    clip_engine = CLIPEngine(inference_config)
    category_classifier = CategoryClassifier(clip_engine, prompts_data.get("categories", {}))
    material_classifier = MaterialClassifier(clip_engine, prompts_data.get("materials", {}))
    return clip_engine, category_classifier, material_classifier


def reclassify_embeddings_csv(
    input_csv: str | Path,
    config_dir: str | Path = "./config",
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    serial_number_column: str = "serial_number",
    tenant_id_column: str = "tenant_id",
) -> list[ReclassificationRecord]:
    """Load a CSV of CLIP embeddings and return DB-ready classification records."""
    source_df = pd.read_csv(Path(input_csv))
    return reclassify_embeddings_frame(
        source_df=source_df,
        config_dir=config_dir,
        embedding_dim=embedding_dim,
        serial_number_column=serial_number_column,
        tenant_id_column=tenant_id_column,
    )

# TO DO 
def reclassify_embeddings_frame(
    source_df: pd.DataFrame,
    config_dir: str | Path = "./config",
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    serial_number_column: str = "serial_number",
    tenant_id_column: str = "tenant_id",
) -> list[ReclassificationRecord]:
    """Reclassify CLIP embeddings from a DataFrame and return DB-ready records."""
    config_path = Path(config_dir)

    clip_cols = [f"clip_{i}" for i in range(embedding_dim)]
    missing_cols = [column for column in clip_cols if column not in source_df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required CLIP embedding columns: {', '.join(missing_cols[:10])}"
        )

    _, classifier, material_classifier = _build_clip_engine(config_path)
    clip_embs = source_df[clip_cols].to_numpy(dtype=np.float32, copy=True)
    has_serial_number = serial_number_column in source_df.columns
    has_tenant_id = tenant_id_column in source_df.columns
    records: list[ReclassificationRecord] = []

    def _clean_optional_value(column_name: str, row_index: int) -> Optional[str]:
        if column_name not in source_df.columns:
            return None
        value = source_df.at[row_index, column_name]
        if pd.isna(value):
            return None
        return str(value)

    for index, embedding in enumerate(clip_embs):
        norm = embedding / (np.linalg.norm(embedding) + 1e-8)
        category, category_confidence, _ = classifier.classify(norm)
        material, material_confidence, _ = material_classifier.classify(norm)

        records.append(
            ReclassificationRecord(
                serial_number=_clean_optional_value(serial_number_column, index) if has_serial_number else None,
                tenant_id=_clean_optional_value(tenant_id_column, index) if has_tenant_id else None,
                category=category,
                category_confidence=round(float(category_confidence), 4),
                material=material,
                material_confidence=round(float(material_confidence), 4),
            )
        )

    return records


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reclassify rows in a CLIP embedding CSV and print DB-ready records."
    )
    parser.add_argument("input_csv", help="Path to the source CSV file.")
    parser.add_argument(
        "--config-dir",
        default="./config",
        help="Directory containing config.yaml and prompts.yaml.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=DEFAULT_EMBEDDING_DIM,
        help="Number of CLIP embedding columns to read (default: 768).",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    records = reclassify_embeddings_csv(
        input_csv=args.input_csv,
        config_dir=args.config_dir,
        embedding_dim=args.embedding_dim,
    )

    print(f"Re-classified {len(records)} items")
    categories = pd.Series([record.category for record in records]).value_counts().to_string()
    materials = pd.Series([record.material for record in records]).value_counts().to_string()
    print(f"Categories:\n{categories}")
    print(f"\nMaterials:\n{materials}")


if __name__ == "__main__":
    main()