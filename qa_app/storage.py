"""
qa_app/storage.py
Saves uploaded QA images to disk and appends review rows to a CSV log.
Both operations are safe under concurrent writes from multiple QA users
hitting the single Flask process at once.
"""
from __future__ import annotations

import csv
import fcntl
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

CSV_HEADER = [
    "timestamp",
    "saved_image_path",
    "predicted_category",
    "predicted_confidence",
    "all_class_scores",
    "is_jewelry_gate",
    "is_correct",
    "actual_label",
    "model_type",
    "embedding_config",
]


def save_upload(upload_dir: Path, file_storage: FileStorage) -> Path:
    """Save an uploaded image, never overwriting on filename collision."""
    upload_dir.mkdir(parents=True, exist_ok=True)
    original_name = secure_filename(file_storage.filename or "upload") or "upload"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    dest = upload_dir / f"{stamp}_{uuid.uuid4().hex[:8]}_{original_name}"
    file_storage.save(dest)
    return dest


def append_log_row(csv_path: Path, row: dict) -> None:
    """
    Append one reviewed-image row to the CSV log, creating the header on
    first run. Uses an exclusive file lock so concurrent submissions from
    different QA users (and threads within this process) can't interleave
    writes or double-write the header.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    line = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "saved_image_path": row["saved_image_path"],
        "predicted_category": row["predicted_category"],
        "predicted_confidence": row["predicted_confidence"],
        "all_class_scores": json.dumps(row["all_class_scores"], separators=(",", ":")),
        "is_jewelry_gate": row["is_jewelry_gate"],
        "is_correct": row["is_correct"],
        "actual_label": row.get("actual_label") or "",
        "model_type": row["model_type"],
        "embedding_config": row["embedding_config"],
    }

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0, os.SEEK_END)
            is_new = f.tell() == 0
            writer = csv.DictWriter(f, fieldnames=CSV_HEADER, quoting=csv.QUOTE_MINIMAL)
            if is_new:
                writer.writeheader()
            writer.writerow(line)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
