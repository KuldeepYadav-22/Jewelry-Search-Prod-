# Jewelry Image Search

Multi-model jewelry image similarity search with threshold-based pool ranking and SQLAlchemy ORM.

## Architecture

```
Image → BiRefNet → RGBA
  ├→ white bg (uncropped) → CLIP → classify + semantic embedding
  └→ crop to subject      → DINOv2 → visual embedding

Search: filter → threshold pools → merge → dedup by serial_number → top-K
```

See `docs/SEARCH_ARCHITECTURE.md` for scoring math and diagrams.

## Setup

```bash
pip install -r requirements.txt
```

## Database Setup

```python
from db import get_engine, create_tables

engine = get_engine("postgresql://user:pass@localhost:5432/jewelry_db")
create_tables(engine)  # creates jewelry_embeddings table + indexes
```

## Indexing

```python
from indexing import IndexingPipeline, IndexRecord
from engines import BiRefNetEngine, CLIPEngine, DINOv2Engine
from classifiers import CategoryClassifier, MaterialClassifier
from preprocess import ImageProcessor
from db import get_engine, get_session, JewelryRepository
import yaml

# Load prompts
with open("config/prompts.yaml") as f:
    prompts = yaml.safe_load(f)

# Init engines
birefnet = BiRefNetEngine(config)
clip = CLIPEngine(config)
dinov2 = DINOv2Engine(config)

# Init pipeline
processor = ImageProcessor(birefnet)
cat_clf = CategoryClassifier(clip, prompts["categories"])
mat_clf = MaterialClassifier(clip, prompts["materials"])
pipeline = IndexingPipeline(processor, clip, dinov2, cat_clf, mat_clf)

# Process single image
record = pipeline.process("SERIAL001", image, "tenant_abc")

# Save via ORM
engine = get_engine("postgresql://...")
session = get_session(engine)
repo = JewelryRepository(session)
repo.upsert(record.__dict__)
```

## Search

```python
from inference import SearchPipeline, SearchConfig
from db import get_engine, get_session, JewelryRepository

# Init (same engines as indexing)
session = get_session(engine)
repo = JewelryRepository(session)
search = SearchPipeline(processor, clip, dinov2, cat_clf, mat_clf, repo)

# Search — returns unique products
response = search.search(query_image, tenant_id="tenant_abc")

for r in response.results:
    print(f"#{r.rank} {r.serial_number} score={r.score:.3f} pool={r.pool}")

# Custom config per request
custom = SearchConfig(top_k=5, clip_pool_threshold=0.60, final_threshold=0.65)
response = search.search(query_image, "tenant_abc", config=custom)
```

## Folder Structure

```
jewelry-search/
├── config/
│   ├── config.yaml          # model paths, search params, DB URL, qa_app settings
│   └── prompts.yaml         # classification prompts
├── engines/                 # BiRefNet, CLIP, DINOv2 (shared)
├── classifiers/             # Category + Material (shared) + trained SupervisedCategoryClassifier
├── preprocess/              # Image processing (shared)
├── db/
│   ├── models.py            # SQLAlchemy ORM model (pgvector)
│   ├── connection.py        # Engine + session factory
│   └── repository.py        # Upsert + search queries
├── indexing/
│   └── pipeline.py          # serial + image → IndexRecord
├── inference/
│   └── search.py            # image + tenant → SearchResponse
├── qa_app/                  # standalone QA web UI (see below)
├── model/                   # trained category classifier artifacts (see below)
├── app.py                   # QA web UI entry point
├── docs/
│   └── SEARCH_ARCHITECTURE.md
└── README.md
```

## QA Web UI (Category Classifier)

A standalone Flask app for internally reviewing the trained 10-class
supervised category classifier (`classifiers/supervised_category_classifier.py`).
It is separate from the main search pipeline (`indexing/`, `inference/`) — it
does not touch the database and does not affect production search traffic.

### Model artifacts

Place these three files (produced by `jewelry_category_classifier (1).ipynb`)
in `model/` before starting the app:

```
model/category_classifier.xgb
model/label_encoder.joblib
model/model_metadata.json
```

The app **fails fast at startup** and names exactly which file is missing (or,
if a file exists but isn't valid, exactly which file and why) — it will not
partially start.

### Running it

```bash
pip install -r requirements.txt
python app.py
```

- Binds to `0.0.0.0:5000` by default (override `host`/`port`/`debug` under
  `qa_app:` in `config/config.yaml`).
- Runs as a single process on GPU if available (`inference.device: auto` in
  `config/config.yaml`) — no worker pool, this is an internal QA tool, not a
  production endpoint.
- All models (BiRefNet, CLIP, DINOv2, zero-shot gate, trained classifier)
  load once at startup; the startup log reports device, per-model
  local-cache/download status, and the winning `model_type` +
  `embedding_config` from `model_metadata.json`.

### Exposing it externally

Point `cloudflared` at the local port the app is bound to, e.g.:

```bash
cloudflared tunnel --url http://localhost:5000
```

(Tunnel setup itself — named tunnels, DNS, auth — is external to this app;
the app only needs to bind to a stable local host/port.)

### What it does

Upload a photo (drag-and-drop, file picker, or mobile camera capture) →
BiRefNet background removal → the same CLIP/DINOv2 preprocessing and
embeddings used everywhere else in this repo → the trained classifier's
top prediction plus a full 10-class score breakdown, with a low-trust badge
if the existing zero-shot CLIP gate doesn't think the image is jewelry.
QA can then confirm the prediction or pick the actual category from a fixed
10-class dropdown; either way the review is appended to `data/qa_log.csv`
(concurrency-safe across multiple QA users) and the uploaded image is saved
to `data/qa_uploads/` (both gitignored).

### Known gap

No authentication — anyone with the tunnel URL can use the tool. Acceptable
for an internal QA tool per the task scope, but worth knowing before sharing
the link widely.
