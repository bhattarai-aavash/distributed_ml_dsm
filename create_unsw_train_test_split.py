import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def create_unsw_train_test_split():
    """Create proper train/test split from the 4 main UNSW-NB15 files"""
    
    print("🔍 Creating UNSW-NB15 Train/Test Split")
    print("=" * 50)
    
    # Load all 4 main files
    main_files = [
        "unsw_data/UNSW-NB15_1.csv",
        "unsw_data/UNSW-NB15_2.csv", 
        "unsw_data/UNSW-NB15_3.csv",
        "unsw_data/UNSW-NB15_4.csv"
    ]
    
    print("📚 Loading main dataset files...")
    all_dataframes = []
    total_records = 0
    
    for i, file_path in enumerate(main_files, 1):
        print(f"  Loading UNSW-NB15_{i}.csv...")
        try:
            df = pd.read_csv(file_path, low_memory=False)
            all_dataframes.append(df)
            total_records += len(df)
            print(f"    ✅ Loaded {len(df):,} records")
        except Exception as e:
            print(f"    ❌ Error loading {file_path}: {e}")
            return False
    
    print(f"\n📊 Combined dataset: {total_records:,} records")
    
    # Combine all dataframes
    print("🔗 Combining all dataframes...")
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"✅ Combined dataset shape: {combined_df.shape}")
    
    # Check columns
    print(f"\n📋 Dataset columns: {list(combined_df.columns)}")
    
    # Check for label columns
    label_cols = ['label', 'Label', 'attack_cat', 'Attack_cat']
    found_labels = [col for col in label_cols if col in combined_df.columns]
    print(f"🏷️  Found label columns: {found_labels}")
    
    if not found_labels:
        print("❌ No label columns found!")
        return False
    
    # Use 'label' column for binary classification
    if 'label' in combined_df.columns:
        label_col = 'label'
        print(f"✅ Using '{label_col}' column for binary classification")
        
        # Check label distribution
        label_dist = combined_df[label_col].value_counts()
        print(f"📊 Label distribution: {dict(label_dist)}")
        
        # Create train/test split (70/30 split)
        print(f"\n🎯 Creating train/test split...")
        X = combined_df.drop(columns=[label_col, 'id'] if 'id' in combined_df.columns else [label_col])
        y = combined_df[label_col]
        
        # Stratified split to maintain label distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.3, 
            random_state=42, 
            stratify=y
        )
        
        # Combine features and labels
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        print(f"✅ Train set: {train_df.shape[0]:,} records")
        print(f"✅ Test set: {test_df.shape[0]:,} records")
        
        # Check train/test label distributions
        train_label_dist = train_df[label_col].value_counts()
        test_label_dist = test_df[label_col].value_counts()
        
        print(f"\n📊 Train label distribution: {dict(train_label_dist)}")
        print(f"📊 Test label distribution: {dict(test_label_dist)}")
        
        # Save the split datasets
        train_file = "unsw_data/UNSW_NB15_training-set.csv"
        test_file = "unsw_data/UNSW_NB15_testing-set.csv"
        
        print(f"\n💾 Saving datasets...")
        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)
        
        print(f"✅ Training set saved: {train_file}")
        print(f"✅ Testing set saved: {test_file}")
        
        # Verify saved files
        print(f"\n🔍 Verifying saved files...")
        train_verify = pd.read_csv(train_file)
        test_verify = pd.read_csv(test_file)
        
        print(f"✅ Training file: {train_verify.shape}")
        print(f"✅ Testing file: {test_verify.shape}")
        
        return True
        
    else:
        print("❌ 'label' column not found!")
        return False

if __name__ == "__main__":
    success = create_unsw_train_test_split()
    if success:
        print("\n🎉 Train/test split created successfully!")
        print("🚀 You can now proceed with distributed training")
    else:
        print("\n❌ Failed to create train/test split")





