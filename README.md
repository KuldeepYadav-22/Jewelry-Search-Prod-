# Jewelry Image Search — Indexing Pipeline

## Setup

```bash
pip install -r requirements.txt
```

PyTorch: install separately from https://pytorch.org for your CUDA version.

## Usage

### Single image

```python
from indexing.worker import IndexingWorker
from PIL import Image

worker = IndexingWorker(config_dir="./config")

image = Image.open("product.jpg")
record = worker.index_one(
    serial_number="EGBR040893",
    image=image,
    tenant_id="tenant_abc",
)

print(record.serial_number)         # "EGBR040893"
print(record.category)              # "ring"
print(record.category_confidence)   # 0.2841
print(record.material)              # "gold"
print(record.material_confidence)   # 0.2563
print(len(record.clip_embedding))   # 768
print(len(record.dino_embedding))   # 1024
print(record.error)                 # None (or error message)
```

### Batch

```python
items = [
    {"serial_number": "EGBR040893", "image": Image.open("img1.jpg")},
    {"serial_number": "EGBN017123", "image": Image.open("img2.jpg")},
    # ...
]

results = worker.index_batch(items, tenant_id="tenant_abc")

for record in results:
    if record.error:
        print(f"FAILED: {record.serial_number} — {record.error}")
    else:
        # Insert into your DB
        db.upsert(record.__dict__)
```

### With progress callback

```python
def on_progress(p):
    print(f"[{p.processed}/{p.total}] {p.images_per_second:.1f} img/s, "
          f"ETA {p.eta_seconds:.0f}s, ok={p.succeeded}, fail={p.failed}")

results = worker.index_batch(items, tenant_id="tenant_abc", on_progress=on_progress)
```

## Architecture

```
Image → BiRefNet → RGBA
  ├→ white bg (uncropped) → CLIP → category + material + clip_embedding
  └→ crop to subject      → DINOv2 → dino_embedding
```

## Config

Edit `config/config.yaml` for model paths and device.
Edit `config/prompts.yaml` for classification prompts (no redeploy needed).

## DB Schema (for your team)

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

CREATE INDEX idx_category ON jewelry_embeddings(tenant_id, category);
CREATE INDEX idx_material ON jewelry_embeddings(tenant_id, material);
CREATE INDEX idx_clip_hnsw ON jewelry_embeddings
    USING hnsw (clip_embedding vector_ip_ops) WITH (m=24, ef_construction=128);
CREATE INDEX idx_dino_hnsw ON jewelry_embeddings
    USING hnsw (dino_embedding vector_ip_ops) WITH (m=24, ef_construction=128);
```
