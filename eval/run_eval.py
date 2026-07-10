"""
eval/run_eval.py
Runs the full indexing pipeline on eval set and produces:
- Accuracy per head (category + material)
- Confusion matrix per head
- Misclassified items list
"""
import os
import sys
import yaml
import types
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engines.clip_engine import CLIPEngine
from engines.dinov2_engine import DINOv2Engine
from engines.birefnet_engine import BiRefNetEngine
from preprocess.image_processor import ImageProcessor
from classifiers.category_classifier import CategoryClassifier
from classifiers.material_classifier import MaterialClassifier
from indexing.pipeline import IndexingPipeline

# ── PATHS ────────────────────────────────────────────────────────
GROUND_TRUTH_CSV = "eval/ground_truth_clean.csv"
IMAGES_ROOT      = r"C:\Users\ishit\Downloads\labelled_data\labelled_data"
CONFIG_YAML      = "config/config.yaml"
PROMPTS_YAML     = "config/prompts.yaml"
OUTPUT_DIR       = "eval/results"
# ─────────────────────────────────────────────────────────────────

def load_config():
    with open(CONFIG_YAML) as f:
        raw = yaml.safe_load(f)
    cfg = types.SimpleNamespace(**raw.get("inference", {}))
    return cfg

def load_prompts():
    with open(PROMPTS_YAML) as f:
        return yaml.safe_load(f)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading config...")
    config = load_config()
    prompts = load_prompts()

    print("Loading engines...")
    birefnet = BiRefNetEngine(config)
    clip = CLIPEngine(config)
    dinov2 = DINOv2Engine(config)

    print("Loading classifiers...")
    cat_clf = CategoryClassifier(clip, prompts["categories"])
    mat_clf = MaterialClassifier(clip, prompts["materials"])

    processor = ImageProcessor(birefnet)
    pipeline = IndexingPipeline(processor, clip, dinov2, cat_clf, mat_clf)

    print("Loading ground truth...")
    gt = pd.read_csv('eval/ground_truth_fixed.csv')
    print(f"Eval set: {len(gt)} images")

    # Run predictions
    predictions = []
    for i, row in gt.iterrows():
        img_path = os.path.join(IMAGES_ROOT, row['category'], row['filename'])
        if not os.path.exists(img_path):
            print(f"  MISSING: {img_path}")
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            record = pipeline.process(
                serial_number=row['serial_number'],
                image=image,
                tenant_id="eval"
            )
            predictions.append({
                "serial_number": row['serial_number'],
                "predicted_category": record.category,
                "category_confidence": record.category_confidence,
                "predicted_material": record.material,
                "material_confidence": record.material_confidence,
                "error": record.error
            })
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(gt)}")
        except Exception as e:
            print(f"  ERROR {row['serial_number']}: {e}")

    preds_df = pd.DataFrame(predictions)
    preds_df.to_csv(f"{OUTPUT_DIR}/predictions.csv", index=False)

    # Merge with ground truth
    df = gt.merge(preds_df, on="serial_number")
    df = df[df['error'].isna()]
    print(f"\nSuccessfully processed: {len(df)}/{len(gt)}")

    # ── Category Head ──
    print("\n=== CATEGORY HEAD ===")
    cat_acc = accuracy_score(df['category'], df['predicted_category'])
    print(f"Accuracy: {cat_acc:.1%}")
    print(classification_report(df['category'], df['predicted_category']))

    cat_labels = sorted(df['category'].unique())
    cm_cat = confusion_matrix(df['category'], df['predicted_category'], labels=cat_labels)
    plt.figure(figsize=(14, 10))
    sns.heatmap(cm_cat, annot=True, fmt='d',
                xticklabels=cat_labels, yticklabels=cat_labels, cmap="Reds")
    plt.title(f"Category Confusion Matrix (Accuracy: {cat_acc:.1%})")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix_category.png", dpi=150)
    print(f"Saved: {OUTPUT_DIR}/confusion_matrix_category.png")

    # ── Material Head ──
    mat_df = df[df['material'] != 'unknown']
    if len(mat_df) > 0:
        print("\n=== MATERIAL HEAD ===")
        mat_acc = accuracy_score(mat_df['material'], mat_df['predicted_material'])
        print(f"Accuracy: {mat_acc:.1%}")
        print(classification_report(mat_df['material'], mat_df['predicted_material']))

        mat_labels = sorted(mat_df['material'].unique())
        cm_mat = confusion_matrix(mat_df['material'], mat_df['predicted_material'], labels=mat_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_mat, annot=True, fmt='d',
                    xticklabels=mat_labels, yticklabels=mat_labels, cmap="Blues")
        plt.title(f"Material Confusion Matrix (Accuracy: {mat_acc:.1%})")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/confusion_matrix_material.png", dpi=150)
        print(f"Saved: {OUTPUT_DIR}/confusion_matrix_material.png")

    # ── Misclassified ──
    misclassified = df[
        (df['category'] != df['predicted_category']) |
        (df['material'] != df['predicted_material'])
    ][['serial_number', 'category', 'predicted_category',
       'category_confidence', 'material', 'predicted_material',
       'material_confidence']]
    misclassified.to_csv(f"{OUTPUT_DIR}/misclassified.csv", index=False)
    print(f"\nMisclassified: {len(misclassified)}/{len(df)}")
    print(f"\n✅ Done! Results in {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()