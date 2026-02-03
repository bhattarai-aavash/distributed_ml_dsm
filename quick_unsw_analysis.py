import pandas as pd
import os

print("🔍 UNSW-NB15 Complete Dataset Analysis")
print("=" * 60)

# List all files in unsw_data
files = os.listdir("unsw_data/")
print(f"📁 Files found: {len(files)}")
for f in sorted(files):
    print(f"  - {f}")

print(f"\n📊 ANALYZING EACH FILE:")
print("=" * 40)

# Analyze main dataset files (1-4)
main_files = []
total_main_records = 0

for i in range(1, 5):
    file_path = f"unsw_data/UNSW-NB15_{i}.csv"
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            main_files.append(df)
            total_main_records += len(df)
            print(f"\n📚 UNSW-NB15_{i}.csv:")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)[:5]}...")  # First 5 columns
            if 'label' in df.columns:
                label_dist = df['label'].value_counts()
                print(f"  Label distribution: {dict(label_dist)}")
        except Exception as e:
            print(f"  ❌ Error reading {file_path}: {e}")

print(f"\n📊 MAIN DATASET SUMMARY:")
print(f"  Total records across all 4 files: {total_main_records:,}")

# Analyze pre-split files
print(f"\n🎯 PRE-SPLIT FILES:")
for filename in ["UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv"]:
    file_path = f"unsw_data/{filename}"
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            print(f"\n📁 {filename}:")
            print(f"  Shape: {df.shape}")
            if 'label' in df.columns:
                label_dist = df['label'].value_counts()
                print(f"  Label distribution: {dict(label_dist)}")
        except Exception as e:
            print(f"  ❌ Error reading {file_path}: {e}")

# Analyze metadata files
print(f"\n📋 METADATA FILES:")
metadata_files = ["NUSW-NB15_features.csv", "UNSW-NB15_LIST_EVENTS.csv"]
for filename in metadata_files:
    file_path = f"unsw_data/{filename}"
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            print(f"\n📁 {filename}:")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
        except Exception as e:
            print(f"  ❌ Error reading {file_path}: {e}")

print(f"\n💡 RECOMMENDATIONS:")
print("=" * 30)

# Check if main files sum to expected total
expected_total = 2540044  # 2,540,044 records as mentioned in documentation
if total_main_records > 0:
    print(f"📊 Main dataset files contain {total_main_records:,} records")
    if total_main_records == expected_total:
        print("  ✅ Matches expected total of 2,540,044 records")
    else:
        print(f"  ⚠️  Expected 2,540,044 records, got {total_main_records:,}")

# Check train/test split
train_file = "unsw_data/UNSW_NB15_training-set.csv"
test_file = "unsw_data/UNSW_NB15_testing-set.csv"

if os.path.exists(train_file) and os.path.exists(test_file):
    try:
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        
        print(f"\n🎯 Train/Test Split Analysis:")
        print(f"  Training file: {train_df.shape[0]:,} records")
        print(f"  Testing file: {test_df.shape[0]:,} records")
        print(f"  Total: {train_df.shape[0] + test_df.shape[0]:,} records")
        
        # Check if they match expected sizes
        if train_df.shape[0] == 175341:
            print("  ✅ Training file has correct size (175,341)")
        else:
            print(f"  ⚠️  Training file size mismatch: expected 175,341, got {train_df.shape[0]:,}")
            
        if test_df.shape[0] == 82332:
            print("  ✅ Testing file has correct size (82,332)")
        else:
            print(f"  ⚠️  Testing file size mismatch: expected 82,332, got {test_df.shape[0]:,}")
            
    except Exception as e:
        print(f"  ❌ Error analyzing train/test files: {e}")

print(f"\n🚀 NEXT STEPS:")
print("  1. Use the 4 main files (UNSW-NB15_1.csv to UNSW-NB15_4.csv) for complete dataset")
print("  2. Or use the pre-split files for immediate training")
print("  3. Check if train/test files are correctly labeled")
