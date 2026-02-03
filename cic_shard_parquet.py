import os
import argparse
import pandas as pd
import numpy as np
from typing import List, Tuple


def stratified_split_indices(y: np.ndarray, num_shards: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(y))
    shards: List[List[int]] = [[] for _ in range(num_shards)]

    # Group indices by class
    classes = np.unique(y)
    for c in classes:
        cls_idx = indices[y == c]
        rng.shuffle(cls_idx)
        # Round-robin assign to shards
        for i, idx in enumerate(cls_idx):
            shards[i % num_shards].append(idx)

    # Convert to numpy arrays and shuffle within each shard
    shard_arrays: List[np.ndarray] = []
    for s in shards:
        arr = np.array(s, dtype=int)
        rng.shuffle(arr)
        shard_arrays.append(arr)
    return shard_arrays


def ensure_single_binary_label(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure there is exactly one binary label column named 'label' with values 0/1.
    Drops any alternative label columns and avoids creating duplicates.
    """
    df = df.copy()

    # If a 'label' column exists, try to coerce to numeric 0/1
    if 'label' in df.columns:
        coerced = pd.to_numeric(df['label'], errors='coerce')
        if coerced.notna().all():
            df['label'] = coerced.astype(int).clip(lower=0, upper=1)
            return df
        # else fallthrough to string mapping
        col = df['label'].astype(str)
        is_benign = col.str.lower().isin({'benign', 'normal'}) | col.str.strip().eq('0')
        df['label'] = (~is_benign).astype(int)
        return df

    # Otherwise, look for common label-like columns, map to 0/1, and rename to 'label'
    for c in ["Label", "Attack", "attack_cat", "Attack_category", "class", "Class"]:
        if c in df.columns:
            col = df[c].astype(str)
            is_benign = col.str.lower().isin({'benign', 'normal'}) | col.str.strip().eq('0')
            df['label'] = (~is_benign).astype(int)
            # drop the original label-like column
            df = df.drop(columns=[c])
            return df

    raise ValueError("Could not determine label column to stratify by")


def shard_csv_to_parquet(input_csv: str, output_dir: str, prefix: str, num_shards: int, seed: int) -> None:
    print(f"Loading {input_csv} ...")
    df = pd.read_csv(input_csv, low_memory=False)

    df = ensure_single_binary_label(df)
    y = df['label'].to_numpy()
    shard_indices = stratified_split_indices(y, num_shards=num_shards, seed=seed)

    os.makedirs(output_dir, exist_ok=True)

    for shard_id, idx in enumerate(shard_indices):
        shard_df = df.iloc[idx].reset_index(drop=True)
        out_path = os.path.join(output_dir, f"{prefix}.shard{shard_id}.parquet")
        shard_df.to_parquet(out_path, index=False)
        print(f"Wrote {out_path} -> {shard_df.shape}")


def main():
    parser = argparse.ArgumentParser(description="Shard CIC-IDS2017 split CSVs into parquet shards")
    parser.add_argument("--train_csv", default="cic_split/CIC_training_set.csv", help="Path to training CSV (from cic_create_train_test.py)")
    parser.add_argument("--test_csv", default="cic_split/CIC_testing_set.csv", help="Path to testing CSV (from cic_create_train_test.py)")
    parser.add_argument("--output_dir", default="cic_shard", help="Output directory for shards")
    parser.add_argument("--num_shards", type=int, default=3, help="Number of shards")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    shard_csv_to_parquet(args.train_csv, args.output_dir, prefix="cic_train", num_shards=args.num_shards, seed=args.seed)
    shard_csv_to_parquet(args.test_csv, args.output_dir, prefix="cic_test", num_shards=args.num_shards, seed=args.seed)

    print("Done sharding CIC-IDS2017 train/test into parquet shards.")


if __name__ == "__main__":
    main()
