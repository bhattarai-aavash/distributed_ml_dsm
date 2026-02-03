#!/bin/bash

# Script to reorganize directory structure for plot_throughput_and_training.py
# This reorganizes ember_mlp_sync, ember_mlp_async, and ember_cnn_sync directories

BASE_DIR="/home/abhattar/ml_cars_main"

echo "📁 Reorganizing directory structure for plotting script..."
echo ""

# Function to reorganize a single experiment directory
reorganize_experiment() {
    local exp_dir="$1"
    local exp_name=$(basename "$exp_dir")
    
    echo "🔧 Processing: $exp_name"
    
    if [ ! -d "$exp_dir" ]; then
        echo "   ⚠️  Directory not found: $exp_dir"
        return
    fi
    
    # Check if keydb_logs directory exists
    if [ ! -d "$exp_dir/keydb_logs" ]; then
        echo "   ⚠️  No keydb_logs directory found in $exp_dir"
        return
    fi
    
    # Create the new directory structure
    new_keydb_dir="$exp_dir/${exp_name}_keydb"
    new_subdir="$new_keydb_dir/$exp_name"
    
    echo "   Creating: $new_subdir"
    mkdir -p "$new_subdir"
    
    # Move breakdown CSV files
    breakdown_count=0
    for breakdown_file in "$exp_dir/keydb_logs"/*_breakdown.csv; do
        if [ -f "$breakdown_file" ]; then
            filename=$(basename "$breakdown_file")
            echo "   Moving: $filename"
            mv "$breakdown_file" "$new_subdir/"
            ((breakdown_count++))
        fi
    done
    
    if [ $breakdown_count -eq 0 ]; then
        echo "   ⚠️  No breakdown CSV files found to move"
        # Remove empty directory
        rmdir "$new_subdir" 2>/dev/null
        rmdir "$new_keydb_dir" 2>/dev/null
    else
        echo "   ✅ Moved $breakdown_count breakdown file(s)"
    fi
    
    echo ""
}

# Reorganize each experiment directory
reorganize_experiment "$BASE_DIR/ember_mlp_sync"
reorganize_experiment "$BASE_DIR/ember_mlp_async"
reorganize_experiment "$BASE_DIR/ember_cnn_sync"

echo "✅ Reorganization complete!"
echo ""
echo "📋 Summary of changes:"
echo "   - Created: <experiment>_keydb/<experiment>/ directories"
echo "   - Moved: *_breakdown.csv files to new locations"
echo "   - Original keydb_logs/ directories remain (other files untouched)"
echo ""
echo "💡 You can now run plot_throughput_and_training.py on these directories"





