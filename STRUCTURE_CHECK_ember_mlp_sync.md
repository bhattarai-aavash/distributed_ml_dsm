# Structure Check for `ember_mlp_sync`

## Current Structure

```
ember_mlp_sync/
├── average_metrics.csv                    ✅ CORRECT
├── keydb_logs/                            ❌ WRONG NAME
│   ├── mteverest1_breakdown.csv          ✅ CORRECT FORMAT
│   ├── mteverest3_breakdown.csv          ✅ CORRECT FORMAT
│   └── mteverest4_breakdown.csv          ✅ CORRECT FORMAT
├── throughput_keydb_logs.csv
└── throughput_summary_keydb_logs.csv

training_plots_ember_mlp_sync/
├── average_elapsed_time_all_iterations.csv  ✅ EXISTS (but in wrong location)
└── average_metrics.csv                      (duplicate)
```

## What's Correct ✅

1. **`average_metrics.csv`** - ✅ Present and has correct format:
   - Has `time_elapsed` column (so no mapping file needed)
   - Has `iteration` column
   - Has metrics: `avg_train_loss`, `avg_train_acc`, `avg_val_loss`, `avg_val_acc`, etc.

2. **Breakdown CSV files** - ✅ Correct format:
   - Columns: `experiment,server,second,elapsed_seconds,client,<master_ips>,total`
   - Files exist: `mteverest1_breakdown.csv`, `mteverest3_breakdown.csv`, `mteverest4_breakdown.csv`

## What's Wrong ❌

1. **Directory name** - ❌ Should be `ember_mlp_sync_keydb/` but is `keydb_logs/`
   - Script looks for directories ending with `_keydb`
   - Current: `ember_mlp_sync/keydb_logs/`
   - Expected: `ember_mlp_sync/ember_mlp_sync_keydb/`

2. **Subdirectory structure** - ❌ Missing subdirectory level
   - Current: `keydb_logs/*_breakdown.csv`
   - Expected: `ember_mlp_sync_keydb/ember_mlp_sync/*_breakdown.csv`

3. **Optional file location** - ⚠️ `average_elapsed_time_all_iterations.csv` exists but in wrong place
   - Current: `training_plots_ember_mlp_sync/average_elapsed_time_all_iterations.csv`
   - Expected: `ember_mlp_sync/average_elapsed_time_all_iterations.csv` (optional, not needed since metrics has time_elapsed)

## Expected Structure

```
ember_mlp_sync/
├── average_metrics.csv                    ✅
├── average_elapsed_time_all_iterations.csv  (optional, not needed)
└── ember_mlp_sync_keydb/                  ❌ MISSING
    └── ember_mlp_sync/                     ❌ MISSING
        ├── mteverest1_breakdown.csv        ✅ (exists but wrong location)
        ├── mteverest3_breakdown.csv        ✅ (exists but wrong location)
        └── mteverest4_breakdown.csv        ✅ (exists but wrong location)
```

## Fix Options

### Option 1: Reorganize Directories (Recommended)
```bash
cd /home/abhattar/ml_cars_main/ember_mlp_sync
mkdir -p ember_mlp_sync_keydb/ember_mlp_sync
mv keydb_logs/*_breakdown.csv ember_mlp_sync_keydb/ember_mlp_sync/
# Keep other files in keydb_logs if needed
```

### Option 2: Modify Script
Update `plot_throughput_and_training.py` to also look for `keydb_logs/` directory.





