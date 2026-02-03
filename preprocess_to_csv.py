import pandas as pd
import numpy as np


def sanitize_features(df: pd.DataFrame) -> pd.DataFrame:
    # Replace inf/-inf with NaN and fill with 0
    return df.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def load_parquet_clean_and_save(
    parquet_path: str,
    csv_path: str,
    drop_negative_one: bool = False,
) -> None:
    print(f"Loading parquet: {parquet_path}")
    df = pd.read_parquet(parquet_path, engine="pyarrow")
    if 'Label' not in df.columns:
        raise ValueError("Input must contain a 'Label' column")

    # Ensure numeric labels; drop NaNs
    labels = pd.to_numeric(df['Label'], errors='coerce')
    mask = labels.notna()
    df = df.loc[mask].copy()
    labels = labels.loc[mask]

    # Optionally drop label == -1 rows for training set
    if drop_negative_one:
        keep = labels != -1
        df = df.loc[keep].copy()
        labels = labels.loc[keep]

    # Map labels to {0,1} if needed
    if set(pd.unique(labels)).issubset({0, 1, 0.0, 1.0}):
        df['Label'] = labels.astype('int64')
    else:
        df['Label'] = (labels > 0.5).astype('int64')

    # Sanitize features
    feature_cols = [c for c in df.columns if c != 'Label']
    df[feature_cols] = sanitize_features(df[feature_cols]).astype('float32')

    print(f"Saving cleaned CSV to: {csv_path}")
    df.to_csv(csv_path, index=False)
    print(f"Done. Rows: {len(df)}, Features: {len(feature_cols)}")


def main():
    train_parquet = "/home/abhattar/ml_cars/train_ember_2018_v2_features.parquet"
    test_parquet = "/home/abhattar/ml_cars/test_ember_2018_v2_features.parquet"
    train_csv = "/home/abhattar/ml_cars/train_ember_2018_v2_features.cleaned.csv"
    test_csv = "/home/abhattar/ml_cars/test_ember_2018_v2_features.cleaned.csv"

    load_parquet_clean_and_save(train_parquet, train_csv, drop_negative_one=True)
    load_parquet_clean_and_save(test_parquet, test_csv, drop_negative_one=False)


if __name__ == "__main__":
    main()


