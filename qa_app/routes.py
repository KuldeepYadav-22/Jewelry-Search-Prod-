"""
qa_app/routes.py
Flask routes for the jewelry category classifier QA tool.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from flask import (
    Blueprint, current_app, flash, redirect, render_template,
    request, send_from_directory, url_for,
)
from PIL import Image, ImageOps, UnidentifiedImageError

from qa_app.pipeline import QAResult, run_pipeline
from qa_app.storage import append_log_row, save_upload

logger = logging.getLogger(__name__)
bp = Blueprint("qa", __name__)


@bp.route("/", methods=["GET"])
def index():
    return render_template("index.html", target_classes=current_app.config["TARGET_CLASSES"])


@bp.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if file is None or file.filename == "":
        flash("Please choose or capture an image first.")
        return redirect(url_for("qa.index"))

    upload_dir = Path(current_app.config["UPLOAD_DIR"])
    try:
        saved_path = save_upload(upload_dir, file)
    except Exception:
        logger.exception("Failed to save upload")
        flash("Could not save the uploaded file.")
        return redirect(url_for("qa.index"))

    try:
        image = Image.open(saved_path)
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
    except (UnidentifiedImageError, OSError):
        flash("That file doesn't look like a valid image.")
        return redirect(url_for("qa.index"))

    bundle = current_app.config["MODEL_BUNDLE"]
    result: QAResult = run_pipeline(bundle, image)

    sorted_scores = sorted(result.all_class_scores.items(), key=lambda kv: kv[1], reverse=True)

    return render_template(
        "index.html",
        target_classes=current_app.config["TARGET_CLASSES"],
        result=result,
        sorted_scores=sorted_scores,
        image_url=url_for("qa.uploaded_file", filename=saved_path.name),
        saved_image_path=str(saved_path),
    )


@bp.route("/uploads/<path:filename>")
def uploaded_file(filename):
    upload_dir = Path(current_app.config["UPLOAD_DIR"]).resolve()
    return send_from_directory(upload_dir, filename)


@bp.route("/submit", methods=["POST"])
def submit():
    form = request.form
    target_classes = current_app.config["TARGET_CLASSES"]

    is_correct = form.get("verdict") == "correct"
    actual_label = "" if is_correct else form.get("actual_label", "")

    if not is_correct and actual_label not in target_classes:
        flash("Please select the actual category from the dropdown.")
        return redirect(url_for("qa.index"))

    try:
        all_class_scores = json.loads(form.get("all_class_scores", "{}"))
    except json.JSONDecodeError:
        all_class_scores = {}

    append_log_row(Path(current_app.config["LOG_CSV"]), {
        "saved_image_path": form.get("saved_image_path", ""),
        "predicted_category": form.get("predicted_category", ""),
        "predicted_confidence": form.get("predicted_confidence", ""),
        "all_class_scores": all_class_scores,
        "is_jewelry_gate": form.get("is_jewelry_gate", ""),
        "is_correct": is_correct,
        "actual_label": actual_label,
        "model_type": form.get("model_type", ""),
        "embedding_config": form.get("embedding_config", ""),
    })

    flash("Logged — thank you. Upload the next image whenever you're ready.")
    return redirect(url_for("qa.index"))
