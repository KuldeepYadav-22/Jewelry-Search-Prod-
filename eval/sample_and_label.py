"""
eval/sample_and_label.py
Randomly samples 300 images from the source folder and creates
a CSV ready for manual labelling.
"""
import os
import random
import shutil
import csv

# ── CHANGE THESE TWO PATHS ──────────────────────────────────────
SOURCE_FOLDER = r"C:\Users\ishit\Downloads\images_part1"  # where your images are
EVAL_FOLDER   = r"C:\Users\ishit\Documents\Projects\Jewelry-Search-Prod-\eval_images"
# ────────────────────────────────────────────────────────────────

SAMPLE_SIZE = 300
OUTPUT_CSV  = "eval/ground_truth.csv"

def main():
    # Step 1: Get all images
    all_images = [
        f for f in os.listdir(SOURCE_FOLDER)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    print(f"Total images found: {len(all_images)}")

    # Step 2: Random sample
    random.seed(42)  # fixed seed so same sample every run
    sampled = random.sample(all_images, min(SAMPLE_SIZE, len(all_images)))
    print(f"Sampled: {len(sampled)} images")

    # Step 3: Copy to eval folder
    os.makedirs(EVAL_FOLDER, exist_ok=True)
    for fname in sampled:
        src = os.path.join(SOURCE_FOLDER, fname)
        dst = os.path.join(EVAL_FOLDER, fname)
        shutil.copy2(src, dst)
    print(f"Copied to: {EVAL_FOLDER}")

    # Step 4: Create CSV for labelling
    os.makedirs("eval", exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "serial_number",
            "true_category",
            "true_material"
        ])
        for fname in sampled:
            serial = fname.rsplit(".", 1)[0]  # remove extension
            writer.writerow([serial, "", ""])  # empty labels to fill

    print(f"CSV created: {OUTPUT_CSV}")
    print(f"\nNext step: open {OUTPUT_CSV} and fill in the labels!")
    print("Categories: ring, bangle, necklace, earring, bracelet, chain, pendant, mangalsutra, anklet, other, paper")
    print("Materials:  gold, silver, rose_gold, diamond, oxidized, platinum, other")

if __name__ == "__main__":
    main()