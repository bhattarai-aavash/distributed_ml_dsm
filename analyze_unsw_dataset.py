import pandas as pd
import numpy as np

print('🔍 UNSW-NB15 Dataset Analysis')
print('=' * 50)

# Load training file
train_file = 'unsw_data/UNSW_NB15_training-set.csv'
test_file = 'unsw_data/UNSW_NB15_testing-set.csv'

print(f'📁 Loading: {train_file}')
df_train = pd.read_csv(train_file)

print(f'📁 Loading: {test_file}')
df_test = pd.read_csv(test_file)

print(f'\n📏 DATASET SHAPES:')
print(f'  Training: {df_train.shape}')
print(f'  Testing:  {df_test.shape}')

print(f'\n📊 TOTAL FEATURES: {df_train.shape[1]} columns')

print(f'\n📝 COLUMN NAMES AND DATA TYPES:')
for i, (col, dtype) in enumerate(df_train.dtypes.items(), 1):
    print(f'  {i:2d}. {col:<25} -> {dtype}')

print(f'\n🏷️  LABEL ANALYSIS:')
# Check for label columns
label_cols = ['label', 'Label', 'attack_cat', 'Attack_cat']
for col in label_cols:
    if col in df_train.columns:
        print(f'\n  📋 Column: {col}')
        unique_vals = df_train[col].unique()
        print(f'    Unique values: {len(unique_vals)}')
        print(f'    Values: {list(unique_vals)}')
        
        # Value counts
        value_counts = df_train[col].value_counts()
        print(f'    Distribution:')
        for val, count in value_counts.items():
            pct = count/len(df_train)*100
            print(f'      {val}: {count:,} ({pct:.1f}%)')

print(f'\n🔤 CATEGORICAL COLUMNS:')
categorical_cols = df_train.select_dtypes(include=['object']).columns
print(f'  Found {len(categorical_cols)} categorical columns:')
for col in categorical_cols:
    unique_vals = df_train[col].unique()
    print(f'    {col}: {len(unique_vals)} unique values')
    if len(unique_vals) <= 10:
        print(f'      Values: {list(unique_vals)}')
    else:
        print(f'      Sample values: {list(unique_vals[:5])}...')

print(f'\n🔢 NUMERIC COLUMNS:')
numeric_cols = df_train.select_dtypes(include=[np.number]).columns
print(f'  Found {len(numeric_cols)} numeric columns:')
for col in numeric_cols:
    print(f'    {col}: {df_train[col].dtype}')

print(f'\n📈 MISSING VALUES:')
missing = df_train.isnull().sum()
if missing.sum() > 0:
    print('  Columns with missing values:')
    for col, count in missing[missing > 0].items():
        pct = count/len(df_train)*100
        print(f'    {col}: {count} ({pct:.1f}%)')
else:
    print('  ✅ No missing values found!')
