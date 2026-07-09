import pandas as pd
import os

CSV_PATH = r"C:\Users\ishit\Downloads\rebuilt_labelled_metadata (1).csv"
IMAGES_ROOT = r"C:\Users\ishit\Downloads\labelled_data\labelled_data"
OUTPUT_CSV = "eval/ground_truth_clean.csv"
SAMPLE_PER_CLASS = 30

def main():
    df = pd.read_csv(CSV_PATH)
    print(f"Total rows: {len(df)}")
    print(f"Category distribution:")
    print(df['category'].value_counts())

    df = df[df['category'].notna() & (df['category'] != '')]
    df['material'] = df['material'].fillna('unknown')
    df['serial_number'] = df['labelled_path'].apply(
        lambda x: os.path.basename(str(x)).rsplit('.', 1)[0]
    )

    frames = []
    for cat, group in df.groupby('category'):
        frames.append(group.sample(min(SAMPLE_PER_CLASS, len(group)), random_state=42))
    sampled = pd.concat(frames).reset_index(drop=True)

    print(f"Sampled: {len(sampled)}")
    print(sampled['category'].value_counts())

    sampled[['serial_number', 'category', 'material', 'labelled_path']].to_csv(OUTPUT_CSV, index=False)
    print(f"Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()