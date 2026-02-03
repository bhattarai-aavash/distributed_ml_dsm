# Instructions to Reorganize Directory Structure

## Quick Method: Run the Script

```bash
cd /home/abhattar/ml_cars_main
./reorganize_structure.sh
```

This will automatically reorganize:
- `ember_mlp_sync/`
- `ember_mlp_async/`
- `ember_cnn_sync/`

## Manual Method: Step-by-Step

### For `ember_mlp_sync`:

```bash
cd /home/abhattar/ml_cars_main/ember_mlp_sync

# Create the new directory structure
mkdir -p ember_mlp_sync_keydb/ember_mlp_sync

# Move breakdown CSV files
mv keydb_logs/*_breakdown.csv ember_mlp_sync_keydb/ember_mlp_sync/

# Verify the move
ls -la ember_mlp_sync_keydb/ember_mlp_sync/
```

### For `ember_mlp_async`:

```bash
cd /home/abhattar/ml_cars_main/ember_mlp_async

# Create the new directory structure
mkdir -p ember_mlp_async_keydb/ember_mlp_async

# Move breakdown CSV files
mv keydb_logs/*_breakdown.csv ember_mlp_async_keydb/ember_mlp_async/

# Verify the move
ls -la ember_mlp_async_keydb/ember_mlp_async/
```

### For `ember_cnn_sync`:

```bash
cd /home/abhattar/ml_cars_main/ember_cnn_sync

# Create the new directory structure
mkdir -p ember_cnn_sync_keydb/ember_cnn_sync

# Move breakdown CSV files
mv keydb_logs/*_breakdown.csv ember_cnn_sync_keydb/ember_cnn_sync/

# Verify the move
ls -la ember_cnn_sync_keydb/ember_cnn_sync/
```

## What Gets Changed

**Before:**
```
ember_mlp_sync/
├── average_metrics.csv
└── keydb_logs/
    ├── mteverest1_breakdown.csv
    ├── mteverest3_breakdown.csv
    └── mteverest4_breakdown.csv
```

**After:**
```
ember_mlp_sync/
├── average_metrics.csv
├── keydb_logs/                    (other files remain here)
└── ember_mlp_sync_keydb/         (NEW)
    └── ember_mlp_sync/            (NEW)
        ├── mteverest1_breakdown.csv  (MOVED)
        ├── mteverest3_breakdown.csv  (MOVED)
        └── mteverest4_breakdown.csv  (MOVED)
```

## Verification

After reorganization, verify the structure:

```bash
# Check ember_mlp_sync
ls -R /home/abhattar/ml_cars_main/ember_mlp_sync/ember_mlp_sync_keydb/

# Should show:
# ember_mlp_sync_keydb/ember_mlp_sync/:
# mteverest1_breakdown.csv
# mteverest3_breakdown.csv
# mteverest4_breakdown.csv
```

## Notes

- Only `*_breakdown.csv` files are moved
- Other files in `keydb_logs/` (like `*.csv` without `_breakdown`) remain untouched
- The `keydb_logs/` directory itself is not deleted (in case you need other files)
- `average_metrics.csv` stays in the root experiment directory (correct location)

## Testing

After reorganization, test the plotting script:

```python
from plot_throughput_and_training import plot_throughput_and_training

plot_throughput_and_training(
    exp_dir="/home/abhattar/ml_cars_main/ember_mlp_sync",
    output_filename=None,
    y_max_throughput=None,
    throughput_skip_seconds=0
)
```





