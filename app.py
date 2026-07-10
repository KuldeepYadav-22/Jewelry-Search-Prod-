"""
app.py
Entry point for the jewelry category classifier QA web UI.

Run:
    python app.py

Binds to the host/port configured under `qa_app:` in config/config.yaml
(defaults: 0.0.0.0:5000). Point `cloudflared tunnel --url http://localhost:5000`
(or an equivalent named tunnel config) at this port for external QA access.

All models load once here at process startup (see qa_app/model_loader.py);
this is a single-process app with no worker pool — it's an internal QA
tool, not a production search endpoint.
"""
from __future__ import annotations

import yaml

from qa_app import create_app

if __name__ == "__main__":
    with open("config/config.yaml") as f:
        qa_cfg = yaml.safe_load(f).get("qa_app", {})

    app = create_app()
    app.run(
        host=qa_cfg.get("host", "0.0.0.0"),
        port=qa_cfg.get("port", 5000),
        debug=qa_cfg.get("debug", False),
        threaded=True,
    )
