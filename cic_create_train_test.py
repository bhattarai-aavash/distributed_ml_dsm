import os
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse


def load_cic_folder(data_dir: str) -> pd.DataFrame:
    csv_files = [os.path.join(data_dir, f) for f in sorted(os.listdir(data_dir)) if f.lower().endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    frames = []
    total = 0
    for path in csv_files:
        print(f"Loading {os.path.basename(path)} ...")
        df = pd.read_csv(path, low_memory=False)
        frames.append(df)
        total += len(df)
        print(f"  -> {df.shape}")
    print(f"Total rows: {total:,}")
    combined = pd.concat(frames, ignore_index=True)
    return combined


def normalize_label_column(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize column names: strip whitespace and collapse inner spaces
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)

    # Try common label columns (case-insensitive)
    colmap = {c.lower(): c for c in df.columns}
    candidates = [
        "label", "attack", "attack_cat", "attack category", "attack_category", "class"
    ]
    label_col = None
    for cand in candidates:
        if cand in colmap:
            label_col = colmap[cand]
            break

    # Heuristic fallback: find a column with BENIGN/attack-like strings
    if label_col is None:
        for c in df.columns:
            if df[c].dtype == object:
                sample = df[c].astype(str).str.strip().str.lower()
                if sample.isin(["benign", "normal"]).sum() > 0 and sample.nunique() < 200:
                    label_col = c
                    break

    # Final fallback: use the last column (user confirmed it's the label)
    if label_col is None:
        label_col = df.columns[-1]
        print(f"Warning: Falling back to last column as label: '{label_col}'")

    # Map to binary: 0 Benign/Normal, 1 Attack
    col = df[label_col].astype(str).str.strip()
    benign_tokens = {"benign", "normal"}
    is_benign = col.str.lower().isin(benign_tokens) | (col.astype(str).isin(["0"]))
    is_attack = (~is_benign) | (col.astype(str).isin(["1"]))
    df["label"] = (is_attack & ~is_benign).astype(int)
    print(f"Label column detected: '{label_col}' -> 'label' (0=Benign/Normal, 1=Attack)")

    # Drop original label columns if present
    if label_col in df.columns:
        df = df.drop(columns=[label_col])

    return df


def main():
    parser = argparse.ArgumentParser(description="Create train/test split for CIC-IDS2017")
    parser.add_argument("--data_dir", default="CIC-IDS-Data", help="Directory containing CIC-IDS2017 CSV files")
    parser.add_argument("--output_dir", default="cic_split", help="Output directory for split CSVs")
    parser.add_argument("--test_size", type=float, default=0.3, help="Test size fraction (default 0.3)")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = load_cic_folder(args.data_dir)
    df = normalize_label_column(df)

    # Prepare X/y
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols]
    y = df["label"]

    # Stratified split
    print("Creating stratified train/test split ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    train_df = X_train.copy()
    train_df["label"] = y_train
    test_df = X_test.copy()
    test_df["label"] = y_test

    # Save
    train_path = os.path.join(args.output_dir, "CIC_training_set.csv")
    test_path = os.path.join(args.output_dir, "CIC_testing_set.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Saved train: {train_path} -> {train_df.shape}")
    print(f"Saved test:  {test_path} -> {test_df.shape}")


if __name__ == "__main__":
    main()
