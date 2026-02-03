import pandas as pd
import numpy as np
import os
from typing import List


def stratified_shards(df: pd.DataFrame, label_col: str, num_shards: int, seed: int = 42) -> List[pd.DataFrame]:
    """Create stratified shards preserving class distribution across shards"""
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
    """Print statistics about the created shards"""
    total = sum(len(s) for s in shards)
    print(f"{name}: total rows = {total}")
    for i, s in enumerate(shards):
        counts = s[label_col].value_counts().to_dict()
        frac = len(s) / total if total else 0
        print(f"  shard {i}: rows={len(s)} ({frac:.2%}) labels={counts}")


def clean_unsw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess UNSW-NB15 data"""
    print(f"Original data shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(f"Missing values found: {missing_values[missing_values > 0]}")
        # Fill missing values with 0 for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)
    
    # Check for infinite values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(df[numeric_columns]).sum().sum()
    if inf_count > 0:
        print(f"Found {inf_count} infinite values, replacing with 0")
        df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], 0)
    
    print(f"Cleaned data shape: {df.shape}")
    return df


def shard_unsw_parquet(input_csv: str, output_prefix: str, num_shards: int = 3, seed: int = 42):
    """Process UNSW CSV file and create stratified shards"""
    print(f"Loading UNSW CSV: {input_csv}")
    
    # Read CSV file
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Clean the data
    df = clean_unsw_data(df)
    
    # Find the label column (should be 'label' or 'Label' or similar)
    label_col = None
    possible_labels = ['label', 'Label', 'labels', 'Labels', 'class', 'Class', 'attack_cat', 'Attack_cat']
    for col in possible_labels:
        if col in df.columns:
            label_col = col
            break
    
    if label_col is None:
        print("Available columns:", list(df.columns))
        raise ValueError("Could not find label column. Expected one of: " + str(possible_labels))
    
    print(f"Using label column: '{label_col}'")
    print(f"Label distribution:")
    print(df[label_col].value_counts())
    
    # Ensure label is int for grouping
    if df[label_col].dtype != 'int64':
        # If labels are strings, convert to numeric
        if df[label_col].dtype == 'object':
            unique_labels = df[label_col].unique()
            label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
            print(f"Label mapping: {label_mapping}")
            df[label_col] = df[label_col].map(label_mapping)
        df[label_col] = df[label_col].astype('int64')
    
    # Create stratified shards
    shards = stratified_shards(df, label_col=label_col, num_shards=num_shards, seed=seed)
    print_shard_stats(shards, label_col=label_col, name=output_prefix)
    
    # Save shards as parquet files
    for i, shard_df in enumerate(shards):
        out_path = f"{output_prefix}.shard{i}.parquet"
        print(f"Saving shard {i} -> {out_path} (rows={len(shard_df)})")
        shard_df.to_parquet(out_path, engine="pyarrow", index=False)


def main():
    """Main function to process UNSW-NB15 dataset"""
    
    # Create output directory
    output_dir = "/home/abhattar/ml_cars_main/unsw_shard"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")
    
    # Paths to UNSW-NB15 CSV files
    train_csv = "/home/abhattar/ml_cars_main/unsw_data/UNSW_NB15_training-set.csv"
    test_csv = "/home/abhattar/ml_cars_main/unsw_data/UNSW_NB15_testing-set.csv"
    
    # Output prefixes for sharded parquet files
    train_out_prefix = os.path.join(output_dir, "train_unsw_nb15")
    test_out_prefix = os.path.join(output_dir, "test_unsw_nb15")
    
    print("=" * 60)
    print("Processing UNSW-NB15 Training Set")
    print("=" * 60)
    shard_unsw_parquet(train_csv, train_out_prefix, num_shards=3, seed=42)
    
    print("\n" + "=" * 60)
    print("Processing UNSW-NB15 Testing Set")
    print("=" * 60)
    shard_unsw_parquet(test_csv, test_out_prefix, num_shards=3, seed=42)
    
    print("\n" + "=" * 60)
    print("UNSW-NB15 Sharding Complete!")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print("Generated files:")
    for i in range(3):
        print(f"  - train_unsw_nb15.shard{i}.parquet")
        print(f"  - test_unsw_nb15.shard{i}.parquet")


if __name__ == "__main__":
    main()
