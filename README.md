# Jewelry Image Search

Multi-model jewelry image similarity search with pool-based ranking.

## Architecture

```
Image → BiRefNet → RGBA
  ├→ white bg (uncropped) → CLIP → classify category + material + semantic embedding
  └→ crop to subject      → DINOv2 → visual similarity embedding

Search: category filter → material filter → pool merge → threshold → top-K
```

See `docs/SEARCH_ARCHITECTURE.md` for full scoring math and diagrams.

## Setup

```bash
pip install -r requirements.txt
```

PyTorch: install separately from https://pytorch.org for your CUDA version.

## Indexing

```python
from indexing import IndexingPipeline, IndexRecord
from engines import BiRefNetEngine, CLIPEngine, DINOv2Engine
from classifiers import CategoryClassifier, MaterialClassifier
from preprocess import ImageProcessor
from PIL import Image
import yaml

# Load prompts
with open("config/prompts.yaml") as f:
    prompts = yaml.safe_load(f)

# Init engines (once)
config = ...  # your inference config
birefnet = BiRefNetEngine(config)
clip = CLIPEngine(config)
dinov2 = DINOv2Engine(config)

# Init pipeline
processor = ImageProcessor(birefnet)
cat_clf = CategoryClassifier(clip, prompts["categories"])
mat_clf = MaterialClassifier(clip, prompts["materials"])
pipeline = IndexingPipeline(processor, clip, dinov2, cat_clf, mat_clf)

# Index single image
record = pipeline.process("SERIAL001", Image.open("ring.jpg"), "tenant_abc")
# → IndexRecord with .clip_embedding (768), .dino_embedding (1024), .category, .material
```

## Search

```python
from inference import SearchPipeline, SearchConfig
# ... same engine/classifier init as above ...

search = SearchPipeline(processor, clip, dinov2, cat_clf, mat_clf, db_repo)

# Search with default config
response = search.search(Image.open("query.jpg"), tenant_id="tenant_abc")

for result in response.results:
    print(f"#{result.rank} {result.serial_number} "
          f"score={result.score} pool={result.pool}")

# Search with custom config
custom = SearchConfig(top_k=5, pool_size=20, final_threshold=0.6)
response = search.search(query_image, "tenant_abc", config=custom)
```

## DB Schema

```sql
CREATE TABLE jewelry_embeddings (
    serial_number       TEXT PRIMARY KEY,
    tenant_id           TEXT NOT NULL,
    category            TEXT,
    category_confidence FLOAT,
    material            TEXT,
    material_confidence FLOAT,
    clip_embedding      VECTOR(768),
    dino_embedding      VECTOR(1024),
    created_at          TIMESTAMP DEFAULT NOW(),
    updated_at          TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_tenant_cat ON jewelry_embeddings(tenant_id, category);
CREATE INDEX idx_tenant_mat ON jewelry_embeddings(tenant_id, material);
CREATE INDEX idx_clip_hnsw ON jewelry_embeddings
    USING hnsw (clip_embedding vector_ip_ops) WITH (m=24, ef_construction=128);
CREATE INDEX idx_dino_hnsw ON jewelry_embeddings
    USING hnsw (dino_embedding vector_ip_ops) WITH (m=24, ef_construction=128);
```

## Config

- `config/config.yaml` — model paths, device, search parameters
- `config/prompts.yaml` — classification prompts (update without redeploy)

## Folder Structure

```
jewelry-search/
├── config/           # YAML configs
├── engines/          # BiRefNet, CLIP, DINOv2 (shared)
├── classifiers/      # Category + Material (shared)
├── preprocess/       # Image processing (shared)
├── db/               # DB layer (your team)
├── indexing/         # Index pipeline
├── inference/        # Search pipeline
├── docs/             # Architecture docs
└── README.md
```
