import pandas as pd
import numpy as np
import os
from typing import Dict, List

def analyze_unsw_file(file_path: str) -> Dict:
    """Analyze a single UNSW CSV file"""
    print(f"\n📊 Analyzing: {file_path}")
    
    try:
        # Try to read the file
        df = pd.read_csv(file_path)
        
        analysis = {
            'file_path': file_path,
            'exists': True,
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'label_distribution': {},
            'attack_cat_distribution': {},
            'categorical_columns': [],
            'numeric_columns': [],
            'missing_values': df.isnull().sum().to_dict()
        }
        
        # Analyze column types
        for col in df.columns:
            if df[col].dtype == 'object':
                analysis['categorical_columns'].append(col)
            else:
                analysis['numeric_columns'].append(col)
        
        # Analyze label distributions
        if 'label' in df.columns:
            analysis['label_distribution'] = df['label'].value_counts().to_dict()
        
        if 'attack_cat' in df.columns:
            analysis['attack_cat_distribution'] = df['attack_cat'].value_counts().to_dict()
        
        # Print summary
        print(f"  ✅ File exists and loaded successfully")
        print(f"  📏 Shape: {df.shape}")
        print(f"  💾 Memory usage: {analysis['memory_usage'] / 1024 / 1024:.2f} MB")
        print(f"  📊 Columns: {len(df.columns)}")
        print(f"  🔤 Categorical: {len(analysis['categorical_columns'])}")
        print(f"  🔢 Numeric: {len(analysis['numeric_columns'])}")
        
        if analysis['label_distribution']:
            print(f"  🏷️  Label distribution: {analysis['label_distribution']}")
        
        if analysis['attack_cat_distribution']:
            print(f"  🎯 Attack categories: {len(analysis['attack_cat_distribution'])} types")
            for cat, count in list(analysis['attack_cat_distribution'].items())[:5]:
                print(f"      {cat}: {count}")
            if len(analysis['attack_cat_distribution']) > 5:
                print(f"      ... and {len(analysis['attack_cat_distribution']) - 5} more")
        
        return analysis
        
    except Exception as e:
        print(f"  ❌ Error loading file: {e}")
        return {
            'file_path': file_path,
            'exists': False,
            'error': str(e)
        }

def main():
    print("🔍 Complete UNSW-NB15 Dataset Analysis")
    print("=" * 60)
    
    # List of expected UNSW-NB15 files
    expected_files = [
        # Main dataset files (4 parts)
        "unsw_data/UNSW-NB15_1.csv",
        "unsw_data/UNSW-NB15_2.csv", 
        "unsw_data/UNSW-NB15_3.csv",
        "unsw_data/UNSW-NB15_4.csv",
        
        # Pre-split training/testing sets
        "unsw_data/UNSW_NB15_training-set.csv",
        "unsw_data/UNSW_NB15_testing-set.csv",
        
        # Metadata files
        "unsw_data/UNSW-NB15_features.csv",
        "unsw_data/UNSW-NB15_GT.csv",
        "unsw_data/UNSW-NB15_LIST_EVENTS.csv"
    ]
    
    # Check what files actually exist
    print("📁 Checking for UNSW-NB15 files...")
    existing_files = []
    missing_files = []
    
    for file_path in expected_files:
        if os.path.exists(file_path):
            existing_files.append(file_path)
            print(f"  ✅ Found: {file_path}")
        else:
            missing_files.append(file_path)
            print(f"  ❌ Missing: {file_path}")
    
    print(f"\n📊 Summary:")
    print(f"  Found: {len(existing_files)} files")
    print(f"  Missing: {len(missing_files)} files")
    
    if not existing_files:
        print("\n❌ No UNSW-NB15 files found!")
        return
    
    # Analyze existing files
    print(f"\n🔍 Analyzing existing files...")
    analyses = []
    
    for file_path in existing_files:
        analysis = analyze_unsw_file(file_path)
        analyses.append(analysis)
    
    # Summary comparison
    print(f"\n" + "=" * 60)
    print("📋 COMPARISON SUMMARY")
    print("=" * 60)
    
    # Group by file type
    main_files = [a for a in analyses if 'UNSW-NB15_' in a['file_path'] and a['file_path'].endswith('.csv') and not 'training' in a['file_path'] and not 'testing' in a['file_path']]
    train_test_files = [a for a in analyses if 'training' in a['file_path'] or 'testing' in a['file_path']]
    metadata_files = [a for a in analyses if any(x in a['file_path'] for x in ['features', 'GT', 'LIST_EVENTS'])]
    
    if main_files:
        print(f"\n📚 MAIN DATASET FILES (UNSW-NB15_1.csv to UNSW-NB15_4.csv):")
        total_records = 0
        for analysis in main_files:
            if analysis['exists']:
                filename = os.path.basename(analysis['file_path'])
                print(f"  {filename}: {analysis['shape']}")
                total_records += analysis['shape'][0]
        print(f"  📊 Total records across main files: {total_records:,}")
    
    if train_test_files:
        print(f"\n🎯 PRE-SPLIT TRAINING/TESTING FILES:")
        for analysis in train_test_files:
            if analysis['exists']:
                filename = os.path.basename(analysis['file_path'])
                print(f"  {filename}: {analysis['shape']}")
                
                # Check if this matches expected sizes
                if 'training' in filename:
                    expected_size = 175341
                    actual_size = analysis['shape'][0]
                    if actual_size == expected_size:
                        print(f"    ✅ Correct size for training set")
                    else:
                        print(f"    ⚠️  Expected {expected_size:,}, got {actual_size:,}")
                
                elif 'testing' in filename:
                    expected_size = 82332
                    actual_size = analysis['shape'][0]
                    if actual_size == expected_size:
                        print(f"    ✅ Correct size for test set")
                    else:
                        print(f"    ⚠️  Expected {expected_size:,}, got {actual_size:,}")
    
    if metadata_files:
        print(f"\n📋 METADATA FILES:")
        for analysis in metadata_files:
            if analysis['exists']:
                filename = os.path.basename(analysis['file_path'])
                print(f"  {filename}: {analysis['shape']}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if len(main_files) == 4:
        print("  ✅ You have all 4 main dataset files - you can create your own train/test split")
    elif len(main_files) > 0:
        print(f"  ⚠️  You have {len(main_files)}/4 main dataset files - incomplete dataset")
    else:
        print("  ❌ No main dataset files found - you only have pre-split files")
    
    if len(train_test_files) == 2:
        print("  ✅ You have both training and testing files")
        # Check if they're swapped
        train_file = next((a for a in train_test_files if 'training' in a['file_path']), None)
        test_file = next((a for a in train_test_files if 'testing' in a['file_path']), None)
        
        if train_file and test_file and train_file['exists'] and test_file['exists']:
            if train_file['shape'][0] < test_file['shape'][0]:
                print("  ⚠️  WARNING: Training file is smaller than test file - they might be swapped!")
                print("     Consider renaming them to fix the train/test split")
    else:
        print("  ⚠️  Missing training or testing files")
    
    if len(metadata_files) > 0:
        print("  ✅ You have metadata files for feature descriptions")

if __name__ == "__main__":
    main()
