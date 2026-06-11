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
│   ├── config.yaml          # model paths, search params, DB URL
│   └── prompts.yaml         # classification prompts
├── engines/                 # BiRefNet, CLIP, DINOv2 (shared)
├── classifiers/             # Category + Material (shared)
├── preprocess/              # Image processing (shared)
├── db/
│   ├── models.py            # SQLAlchemy ORM model (pgvector)
│   ├── connection.py        # Engine + session factory
│   └── repository.py        # Upsert + search queries
├── indexing/
│   └── pipeline.py          # serial + image → IndexRecord
├── inference/
│   └── search.py            # image + tenant → SearchResponse
├── docs/
│   └── SEARCH_ARCHITECTURE.md
└── README.md
```
