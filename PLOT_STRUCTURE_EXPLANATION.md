# Explanation: Structure for `plot_throughput_and_training.py`

## Overview

The `plot_throughput_and_training.py` script creates a **combined plot** with two subplots:
1. **Top subplot**: Server throughput (commands/sec) vs elapsed time
2. **Bottom subplot**: ML training metrics (accuracy & loss) vs elapsed time

## Directory Structure

```
experiment_directory/                    # e.g., /home/abhattar/ml_cars_main/ember_mlp_sync
├── average_metrics.csv                 # REQUIRED: ML training metrics
├── average_elapsed_time_all_iterations.csv  # OPTIONAL: Time mapping (only if metrics lacks time)
└── <experiment_name>_keydb/            # REQUIRED: Must end with "_keydb"
    └── <any_subdirectory>/              # Can be any name, but must exist
        ├── <server1>_breakdown.csv      # REQUIRED: Throughput data per server
        ├── <server2>_breakdown.csv
        └── <server3>_breakdown.csv
```

### Example:
```
ember_mlp_sync/
├── average_metrics.csv
└── ember_mlp_sync_keydb/          ← Must end with "_keydb"
    └── ember_mlp_sync/             ← Any subdirectory name
        ├── mteverest1_breakdown.csv
        ├── mteverest3_breakdown.csv
        └── mteverest4_breakdown.csv
```

## How the Script Works

### Step 1: Finding Throughput Data (Lines 36-46)

```python
# Looks for directories ending with "_keydb"
for d in os.listdir(exp_dir):
    if d.endswith('_keydb'):
        keydb_root = os.path.join(exp_dir, d)
        break

# Then searches for breakdown files in: <experiment>_keydb/*/*_breakdown.csv
breakdown_paths = glob.glob(os.path.join(keydb_root, '*', '*_breakdown.csv'))
```

**Key Points:**
- Searches for directories ending with `_keydb` in the experiment directory
- Then looks for `*_breakdown.csv` files **one level deep**: `*_keydb/*/*_breakdown.csv`
- Pattern: `<experiment>_keydb/<subdir>/*_breakdown.csv`

### Step 2: Loading ML Metrics (Lines 24-34)

```python
# Looks for average_metrics.csv directly in experiment directory
metrics_csv = os.path.join(exp_dir, 'average_metrics.csv')
```

**Key Points:**
- Must be in the **root** of the experiment directory
- Not in a subdirectory

### Step 3: Processing Throughput Data (Lines 61-91)

For each `*_breakdown.csv` file:

1. **Reads CSV** with columns:
   - `experiment`, `server`, `second`, `elapsed_seconds` (metadata)
   - `client`, `<master_ip1>`, `<master_ip2>`, ..., `total` (data columns)

2. **Extracts source columns** (everything except metadata):
   ```python
   meta_cols = {"experiment", "server", "second", "elapsed_seconds"}
   source_cols = [c for c in df.columns if c not in meta_cols]
   ```

3. **Orders columns** for plotting:
   - First: `client`
   - Middle: All master IPs (sorted)
   - Last: `total`

4. **Plots each source** as a separate line:
   - X-axis: `elapsed_seconds`
   - Y-axis: Command count (from source column)
   - Label: `{server}:{source}` (e.g., "mteverest1:client", "mteverest1:10.22.12.93")

### Step 4: Processing ML Metrics (Lines 93-217)

The script has **two modes** for handling time:

#### Mode A: Direct Time Column (Preferred)
If `average_metrics.csv` has a time column:
- Looks for: `time_elapsed`, `elapsed_seconds`, `elapsed_time`, or `seconds`
- Uses it directly for X-axis
- No mapping file needed

#### Mode B: Iteration-Based (Fallback)
If `average_metrics.csv` only has `iteration`:
- Requires `average_elapsed_time_all_iterations.csv` to map iterations → time
- Merges on `iteration` column
- Then uses elapsed time for X-axis

**Plots:**
- **Left Y-axis (blue)**: Accuracy metrics
  - `avg_train_acc`, `avg_val_acc`, `avg_test_acc`
- **Right Y-axis (red)**: Loss metrics
  - `avg_train_loss`, `avg_val_loss`, `avg_test_loss`

## File Format Requirements

### 1. `*_breakdown.csv` Format

**Required Columns:**
```csv
experiment,server,second,elapsed_seconds,client,<master_ip1>,<master_ip2>,...,total
```

**Example:**
```csv
experiment,server,second,elapsed_seconds,client,10.22.12.93,10.22.13.174,10.22.14.63,unknown_master,total
keydb_logs,mteverest1,1762469217,1444,10,8,0,0,0,18
keydb_logs,mteverest1,1762468926,1153,0,0,0,10,0,10
```

**Notes:**
- `elapsed_seconds` = time from start (0, 1, 2, ...)
- `second` = Unix timestamp
- Master IP columns are **dynamic** (depends on your setup)
- Each row = one second of data

### 2. `average_metrics.csv` Format

**Option A: With Time Column (Preferred)**
```csv
iteration,time_elapsed,avg_train_acc,avg_val_acc,avg_train_loss,avg_val_loss
1,1.13,0.46875,0.55638,0.76798,0.70089
50,102.35,0.73917,0.78910,0.52068,0.45755
```

**Option B: Without Time Column (Requires Mapping)**
```csv
iteration,avg_train_acc,avg_val_acc,avg_train_loss,avg_val_loss
1,0.46875,0.55638,0.76798,0.70089
50,0.73917,0.78910,0.52068,0.45755
```

### 3. `average_elapsed_time_all_iterations.csv` Format (Optional)

**Required Columns:**
```csv
iteration,avg_time_elapsed
1,1.13
50,102.35
100,202.91
```

**Note:** Only needed if `average_metrics.csv` doesn't have a time column.

## Common Issues

### Issue 1: "No *_keydb directory found"
- **Problem**: Directory doesn't end with `_keydb`
- **Solution**: Rename `keydb_logs/` → `<experiment>_keydb/`

### Issue 2: "No breakdown CSV files found"
- **Problem**: Files not in the right subdirectory level
- **Current**: `*_keydb/*_breakdown.csv` ❌
- **Expected**: `*_keydb/*/*_breakdown.csv` ✅
- **Solution**: Add one more subdirectory level

### Issue 3: "average_metrics.csv not found"
- **Problem**: File not in experiment root directory
- **Solution**: Move `average_metrics.csv` to experiment root

### Issue 4: "Cannot plot metrics vs elapsed time"
- **Problem**: Metrics CSV has no time column AND no mapping file
- **Solution**: Either add `time_elapsed` to metrics CSV, or provide mapping file

## Summary

**Required Structure:**
```
experiment_dir/
├── average_metrics.csv                    ← Root level
└── <name>_keydb/                         ← Must end with "_keydb"
    └── <subdir>/                          ← Any subdirectory
        └── *_breakdown.csv                ← One level deep
```

**Key Search Pattern:**
- Throughput: `glob.glob(os.path.join(keydb_root, '*', '*_breakdown.csv'))`
  - This means: `*_keydb/*/*_breakdown.csv` (two wildcards = two directory levels)
- Metrics: `os.path.join(exp_dir, 'average_metrics.csv')`
  - This means: directly in experiment root





