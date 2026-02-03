import os
import pandas as pd

print("🔍 Checking UNSW-NB15 files in unsw_data/")
print("=" * 50)

# Check what files exist
files = os.listdir("unsw_data/")
print(f"📁 Files found: {files}")

# Expected files
expected_files = [
    "UNSW-NB15_1.csv",
    "UNSW-NB15_2.csv", 
    "UNSW-NB15_3.csv",
    "UNSW-NB15_4.csv",
    "UNSW_NB15_training-set.csv",
    "UNSW_NB15_testing-set.csv",
    "UNSW-NB15_features.csv",
    "UNSW-NB15_GT.csv",
    "UNSW-NB15_LIST_EVENTS.csv"
]

print(f"\n📋 Expected vs Found:")
for expected in expected_files:
    if expected in files:
        print(f"  ✅ {expected}")
        # Get file size
        file_path = f"unsw_data/{expected}"
        try:
            df = pd.read_csv(file_path)
            print(f"     Shape: {df.shape}")
        except Exception as e:
            print(f"     Error reading: {e}")
    else:
        print(f"  ❌ {expected}")

print(f"\n📊 Summary:")
print(f"  Found: {len([f for f in expected_files if f in files])}/{len(expected_files)} expected files")
print(f"  Total files in directory: {len(files)}")






