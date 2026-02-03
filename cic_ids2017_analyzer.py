import os
import pandas as pd
import numpy as np
from typing import Dict, List


def analyze_csv(file_path: str) -> Dict:
    print(f"\n📊 Analyzing: {file_path}")
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except Exception as e:
        print(f"  ❌ Error reading: {e}")
        return {"file_path": file_path, "error": str(e)}

    result: Dict = {
        "file_path": file_path,
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "memory_mb": float(df.memory_usage(deep=True).sum()) / (1024 * 1024),
        "categorical_cols": list(df.select_dtypes(include=["object"]).columns),
        "numeric_cols": list(df.select_dtypes(include=[np.number]).columns),
        "missing_counts": df.isnull().sum().to_dict(),
        "label_distributions": {},
    }

    # Common label columns across CIC-IDS2017 variants
    candidate_labels = [
        "Label", "label", "Attack", "attack_cat", "Attack_category",
        "class", "Class", "Outcome"
    ]
    for col in candidate_labels:
        if col in df.columns:
            vc = df[col].value_counts(dropna=False)
            result["label_distributions"][col] = vc.to_dict()
            print(f"  🏷️  {col} distribution: {dict(vc)}")

    print(f"  📏 Shape: {df.shape}")
    print(f"  💾 Memory: {result['memory_mb']:.2f} MB")
    print(f"  🔤 Categorical: {len(result['categorical_cols'])}")
    print(f"  🔢 Numeric: {len(result['numeric_cols'])}")

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="CIC-IDS2017 dataset analyzer")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing CIC-IDS2017 CSV files")
    parser.add_argument("--report", type=str, default="cic_ids2017_summary.txt", help="Output report path")
    args = parser.parse_args()

    data_dir = args.data_dir
    if not os.path.isdir(data_dir):
        print(f"❌ Directory not found: {data_dir}")
        return

    csv_files: List[str] = [
        os.path.join(data_dir, f) for f in sorted(os.listdir(data_dir)) if f.lower().endswith(".csv")
    ]

    if not csv_files:
        print("❌ No CSV files found in the directory")
        return

    print("=" * 60)
    print("🔍 CIC-IDS2017 Dataset Analysis")
    print("=" * 60)
    print(f"📁 Directory: {data_dir}")
    print(f"🗂️  CSV files: {len(csv_files)}")

    analyses: List[Dict] = []
    total_rows = 0
    total_memory_mb = 0.0

    for path in csv_files:
        analysis = analyze_csv(path)
        analyses.append(analysis)
        if "shape" in analysis:
            total_rows += analysis["shape"][0]
        if "memory_mb" in analysis:
            total_memory_mb += analysis["memory_mb"]

    # Write summary report
    with open(args.report, "w") as f:
        f.write("CIC-IDS2017 Dataset Analysis Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Directory: {data_dir}\n")
        f.write(f"CSV files: {len(csv_files)}\n")
        f.write(f"Total rows (sum across files): {total_rows:,}\n")
        f.write(f"Approx memory (sum across files): {total_memory_mb:.2f} MB\n\n")

        for a in analyses:
            f.write(f"File: {os.path.basename(a.get('file_path',''))}\n")
            if "error" in a:
                f.write(f"  Error: {a['error']}\n\n")
                continue
            f.write(f"  Shape: {a['shape']}\n")
            f.write(f"  Memory: {a['memory_mb']:.2f} MB\n")
            f.write(f"  Columns: {len(a['columns'])}\n")
            f.write(f"  Categorical: {len(a['categorical_cols'])}\n")
            f.write(f"  Numeric: {len(a['numeric_cols'])}\n")
            if a.get("label_distributions"):
                f.write("  Label distributions:\n")
                for col, dist in a["label_distributions"].items():
                    f.write(f"    {col}: {dist}\n")
            miss = {k: v for k, v in a.get("missing_counts", {}).items() if v > 0}
            if miss:
                f.write("  Missing values (non-zero):\n")
                for col, cnt in miss.items():
                    f.write(f"    {col}: {cnt}\n")
            f.write("\n")

    print("\n✅ Analysis complete")
    print(f"📝 Report saved to: {args.report}")


if __name__ == "__main__":
    main()
