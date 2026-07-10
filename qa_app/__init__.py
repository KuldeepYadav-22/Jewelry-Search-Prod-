"""
qa_app/__init__.py
Flask application factory for the jewelry category classifier QA tool.

All models (BiRefNet, CLIP, DINOv2, zero-shot gate, trained classifier)
are loaded exactly once here, at app creation time, and stashed on
app.config for routes to reuse across requests.
"""
from __future__ import annotations

import logging
import types
from pathlib import Path

import yaml
from flask import Flask

from qa_app.model_loader import load_models

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_yaml_config():
    with open(REPO_ROOT / "config" / "config.yaml") as f:
        raw = yaml.safe_load(f)
    inference_cfg = types.SimpleNamespace(**raw.get("inference", {}))
    qa_cfg = raw.get("qa_app", {})
    with open(REPO_ROOT / "config" / "prompts.yaml") as f:
        prompts = yaml.safe_load(f)
    return inference_cfg, qa_cfg, prompts


def create_app() -> Flask:
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    inference_cfg, qa_cfg, prompts = _load_yaml_config()

    app = Flask(__name__)
    app.secret_key = qa_cfg.get("secret_key", "jewelry-qa-dev-key")

    upload_dir = REPO_ROOT / qa_cfg.get("upload_dir", "data/qa_uploads")
    log_csv = REPO_ROOT / qa_cfg.get("log_csv", "data/qa_log.csv")
    model_dir = REPO_ROOT / qa_cfg.get("category_classifier_model_dir", "model")
    upload_dir.mkdir(parents=True, exist_ok=True)

    app.config["UPLOAD_DIR"] = str(upload_dir)
    app.config["LOG_CSV"] = str(log_csv)
    app.config["MAX_CONTENT_LENGTH"] = qa_cfg.get("max_upload_mb", 20) * 1024 * 1024

    logger.info("Loading models — this happens once at startup, not per-request ...")
    bundle = load_models(inference_cfg, prompts, model_dir)
    app.config["MODEL_BUNDLE"] = bundle
    app.config["TARGET_CLASSES"] = bundle.target_classes

    from qa_app.routes import bp
    app.register_blueprint(bp)

    return app
