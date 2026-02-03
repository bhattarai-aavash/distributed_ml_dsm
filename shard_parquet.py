import pandas as pd
import numpy as np
from typing import List


def stratified_shards(df: pd.DataFrame, label_col: str, num_shards: int, seed: int = 42) -> List[pd.DataFrame]:
    if label_col not in df.columns:
        raise ValueError(f"Missing label column '{label_col}' in DataFrame")
    rng = np.random.default_rng(seed)
    shards = [list() for _ in range(num_shards)]

    # Group indices by class label to preserve distribution across shards
    for cls, group in df.groupby(label_col).groups.items():
        idx = np.array(list(group))
        rng.shuffle(idx)
        # Split approximately equal chunks
        splits = np.array_split(idx, num_shards)
        for s, split_idx in zip(shards, splits):
            s.extend(split_idx.tolist())

    # Build shard DataFrames in a deterministic order
    shard_dfs: List[pd.DataFrame] = []
    for s in shards:
        s_sorted = sorted(s)
        shard_dfs.append(df.iloc[s_sorted].copy())
    return shard_dfs


def print_shard_stats(shards: List[pd.DataFrame], label_col: str, name: str):
    total = sum(len(s) for s in shards)
    print(f"{name}: total rows = {total}")
    for i, s in enumerate(shards):
        counts = s[label_col].value_counts().to_dict()
        frac = len(s) / total if total else 0
        print(f"  shard {i}: rows={len(s)} ({frac:.2%}) labels={counts}")


def shard_parquet(input_parquet: str, output_prefix: str, num_shards: int = 3, seed: int = 42):
    print(f"Loading cleaned parquet: {input_parquet}")
    df = pd.read_parquet(input_parquet, engine="pyarrow")

    # Ensure label is int for grouping
    if df['Label'].dtype != 'int64':
        df['Label'] = df['Label'].astype('int64')

    shards = stratified_shards(df, label_col='Label', num_shards=num_shards, seed=seed)
    print_shard_stats(shards, label_col='Label', name=output_prefix)

    for i, shard_df in enumerate(shards):
        out_path = f"{output_prefix}.shard{i}.parquet"
        print(f"Saving shard {i} -> {out_path} (rows={len(shard_df)})")
        shard_df.to_parquet(out_path, engine="pyarrow", index=False)


def main():
    # Paths to cleaned parquet files
    train_in = "/home/abhattar/ml_cars/train_ember_2018_v2_features.cleaned.parquet"
    test_in = "/home/abhattar/ml_cars/test_ember_2018_v2_features.cleaned.parquet"

    train_out_prefix = "/home/abhattar/ml_cars/train_ember_2018_v2_features.cleaned"
    test_out_prefix = "/home/abhattar/ml_cars/test_ember_2018_v2_features.cleaned"

    shard_parquet(train_in, train_out_prefix, num_shards=3, seed=42)
    shard_parquet(test_in, test_out_prefix, num_shards=3, seed=42)


if __name__ == "__main__":
    main()


