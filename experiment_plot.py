import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from datetime import datetime

# ⚙️ CONFIGURATION: Time binning for throughput data averaging
# Set the time interval (in seconds) for averaging throughput data to reduce noise
# Smaller values = more granular data, larger values = more averaged/smooth data
THROUGHPUT_TIME_BIN_SECONDS = 1  # Change this value to adjust averaging interval

def load_ml_metrics(ml_logs_dir):
    """Load ML training metrics from CSV files for all experiments"""
    experiment_data = {}
    
    # Look for plots directory first (contains average_metrics.csv files)
    plots_dir = os.path.join(ml_logs_dir, 'plots')
    if os.path.exists(plots_dir):
        # Load from plots directory (averaged data)
        for exp_dir in os.listdir(plots_dir):
            exp_path = os.path.join(plots_dir, exp_dir)
            if os.path.isdir(exp_path) and exp_dir.startswith('experiment_'):
                csv_file = os.path.join(exp_path, 'average_metrics.csv')
                if os.path.exists(csv_file):
                    try:
                        df = pd.read_csv(csv_file)
                        experiment_data[exp_dir] = df
                        print(f"✅ Loaded averaged data for {exp_dir}: {len(df)} rows")
                    except Exception as e:
                        print(f"❌ Error loading {csv_file}: {e}")
    else:
        # Fallback: Look for individual server data in timestamped directories
        for exp_dir in os.listdir(ml_logs_dir):
            exp_path = os.path.join(ml_logs_dir, exp_dir)
            if os.path.isdir(exp_path) and not exp_dir.startswith('.'):
                experiment_data[exp_dir] = {}
                
                # Look for server directories within each experiment
                for server_dir in os.listdir(exp_path):
                    server_path = os.path.join(exp_path, server_dir)
                    if os.path.isdir(server_path):
                        csv_files = [f for f in os.listdir(server_path) if f.endswith('.csv')]
                        
                        if csv_files:
                            csv_file = os.path.join(server_path, csv_files[0])
                            try:
                                df = pd.read_csv(csv_file)
                                df['timestamp'] = pd.to_datetime(df['timestamp'])
                                experiment_data[exp_dir][server_dir] = df
                                print(f"✅ Loaded {exp_dir}/{server_dir}: {len(df)} rows")
                            except Exception as e:
                                print(f"❌ Error loading {csv_file}: {e}")
    
    return experiment_data

def load_keydb_throughput(keydb_analysis_dir):
    """Load KeyDB throughput data from log analysis files"""
    throughput_data = {}
    
    # Find all server_counts files
    pattern = os.path.join(keydb_analysis_dir, "server_counts_*_experiment_*.txt")
    files = glob.glob(pattern)
    
    for file_path in files:
        # Extract experiment name from filename
        filename = os.path.basename(file_path)
        parts = filename.replace('.txt', '').split('_')
        # Look for "experiment_" in the filename and extract the number
        exp_name = None
        for i, part in enumerate(parts):
            if part == 'experiment' and i + 1 < len(parts):
                exp_name = f"experiment_{parts[i+1]}"
                break
        
        if exp_name is None:
            print(f"⚠️ Could not extract experiment name from {filename}")
            continue
        
        print(f"🔍 Extracted experiment name: {exp_name} from {filename}")
        
        times = []
        totals = []
        
        with open(file_path) as f:
            for line in f:
                if "TOTAL" in line and not line.startswith("Timestamp"):
                    parts = line.split()
                    ts = " ".join(parts[:4])
                    count = int(parts[-1])
                    times.append(ts)
                    totals.append(count)
        
        if times:
            df = pd.DataFrame({"timestamp": times, "throughput": totals})
            print(f"🔍 Debug {exp_name}: Sample timestamp: {times[0] if times else 'None'}")
            
            # Parse timestamp and calculate elapsed time properly
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d %b %Y %H:%M:%S")
                print(f"✅ Parsed timestamps with format '%d %b %Y %H:%M:%S'")
            except Exception as e1:
                print(f"⚠️ Format 1 failed: {e1}")
                try:
                    # Try without year
                    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d %b %H:%M:%S")
                    print(f"✅ Parsed timestamps with format '%d %b %H:%M:%S'")
                except Exception as e2:
                    print(f"⚠️ Format 2 failed: {e2}")
                    try:
                        # Fallback to automatic parsing
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                        print(f"✅ Parsed timestamps with automatic parsing")
                    except Exception as e3:
                        print(f"❌ All timestamp parsing failed: {e3}")
                        continue
            
            # Calculate elapsed time from the first timestamp
            if not df.empty and 'timestamp' in df.columns:
                # Sort by timestamp to ensure chronological order
                df = df.sort_values('timestamp').reset_index(drop=True)
                df['elapsed_sec'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()
                print(f"📊 {exp_name}: Time range {df['timestamp'].min()} to {df['timestamp'].max()}, Elapsed: {df['elapsed_sec'].min():.1f}s to {df['elapsed_sec'].max():.1f}s")
            else:
                print(f"❌ {exp_name}: No valid timestamps found")
                continue
            
            throughput_data[exp_name] = df
    
    return throughput_data

def prepare_ml_data(experiment_data):
    """Prepare ML data for plotting - data is already averaged"""
    prepared_data = {}
    
    for exp_name, df in experiment_data.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            # Data is already averaged, just ensure time_elapsed column exists
            if 'time_elapsed' not in df.columns and 'iteration' in df.columns:
                # If no time_elapsed, create a simple time progression
                df['time_elapsed'] = df['iteration'] * 10  # Assume 10 seconds per iteration
            prepared_data[exp_name] = df
        elif isinstance(df, dict):
            # Data is per-server, need to calculate averages
            print(f"⚠️ {exp_name} contains per-server data, calculating averages...")
            # This would be the old averaging logic, but we expect pre-averaged data
            prepared_data[exp_name] = pd.DataFrame()
    
    return prepared_data

def align_time_ranges(ml_data, throughput_data):
    """Align KeyDB throughput data with ML training time range and average it"""
    aligned_throughput = {}
    
    print(f"🔍 Available ML experiments: {list(ml_data.keys())}")
    print(f"🔍 Available throughput experiments: {list(throughput_data.keys())}")
    print(f"⚙️ Using {THROUGHPUT_TIME_BIN_SECONDS}-second time bins for throughput averaging")
    
    for exp_name, ml_df in ml_data.items():
        if ml_df.empty or 'time_elapsed' not in ml_df.columns:
            continue
            
        # Get ML training time range
        ml_start_time = ml_df['time_elapsed'].min()
        ml_end_time = ml_df['time_elapsed'].max()
        
        print(f"📊 {exp_name} ML training time range: {ml_start_time:.1f}s to {ml_end_time:.1f}s")
        
        # Find corresponding throughput data
        if exp_name in throughput_data and not throughput_data[exp_name].empty:
            throughput_df = throughput_data[exp_name].copy()
            
            if 'elapsed_sec' in throughput_df.columns:
                # Filter throughput data to ML training time range
                mask = (throughput_df['elapsed_sec'] >= ml_start_time) & (throughput_df['elapsed_sec'] <= ml_end_time)
                filtered_throughput = throughput_df[mask].copy()
                
                if not filtered_throughput.empty:
                    # Adjust elapsed time to start from 0 (relative to ML training start)
                    filtered_throughput['elapsed_sec'] = filtered_throughput['elapsed_sec'] - ml_start_time
                    
                    # Average throughput data to reduce noise (group by configurable time intervals)
                    filtered_throughput['time_bin'] = (filtered_throughput['elapsed_sec'] // THROUGHPUT_TIME_BIN_SECONDS) * THROUGHPUT_TIME_BIN_SECONDS
                    averaged_throughput = filtered_throughput.groupby('time_bin')['throughput'].mean().reset_index()
                    averaged_throughput.rename(columns={'time_bin': 'elapsed_sec'}, inplace=True)
                    
                    aligned_throughput[exp_name] = averaged_throughput
                    print(f"✅ {exp_name} throughput: {len(filtered_throughput)} points -> {len(averaged_throughput)} averaged points")
                else:
                    print(f"⚠️ {exp_name} throughput: No data in ML training time range")
            else:
                print(f"⚠️ {exp_name} throughput: No elapsed_sec column")
        else:
            print(f"⚠️ {exp_name} throughput: No data available")
    
    return aligned_throughput

def create_comprehensive_plots(ml_data, throughput_data, output_dir):
    """Create comprehensive plots showing ML metrics and throughput with dual y-axes"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare ML data (already averaged from plots directory)
    averaged_ml_data = prepare_ml_data(ml_data)
    
    # Align throughput data with ML training time ranges
    aligned_throughput = align_time_ranges(averaged_ml_data, throughput_data)
    
    # Plot 1: Throughput vs Accuracy (Dual Y-Axis) - All Experiments
    fig, ax1 = plt.subplots(1, 1, figsize=(16, 10))
    
    # Left Y-axis: Throughput
    ax1.set_xlabel('Elapsed Time (seconds)')
    ax1.set_ylabel('KeyDB Throughput (commands/sec)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Plot throughput for each experiment
    for exp_name, df in aligned_throughput.items():
        if not df.empty and 'elapsed_sec' in df.columns:
            ax1.plot(df['elapsed_sec'], df['throughput'], 
                    label=f'Throughput {exp_name}', color='blue', alpha=0.7, linewidth=2)
    
    # Right Y-axis: Accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel('ML Accuracy', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Plot accuracy for each experiment (averaged across servers)
    colors = ['red', 'darkred', 'maroon']
    for i, (exp_name, df) in enumerate(averaged_ml_data.items()):
        if not df.empty:
            color = colors[i % len(colors)]
            valid_data = df.dropna(subset=['avg_train_acc', 'avg_val_acc'])
            if len(valid_data) > 0:
                ax2.plot(valid_data['time_elapsed'], valid_data['avg_train_acc'], 
                        label=f'{exp_name} Train Acc', color=color, alpha=0.7, linestyle='-')
                ax2.plot(valid_data['time_elapsed'], valid_data['avg_val_acc'], 
                        label=f'{exp_name} Val Acc', color=color, alpha=0.7, linestyle='--')
    
    ax1.set_title('KeyDB Throughput vs ML Accuracy Over Time (All Experiments)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(0, 1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'throughput_vs_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Throughput vs Loss (Dual Y-Axis) - All Experiments
    fig, ax1 = plt.subplots(1, 1, figsize=(16, 10))
    
    # Left Y-axis: Throughput
    ax1.set_xlabel('Elapsed Time (seconds)')
    ax1.set_ylabel('KeyDB Throughput (commands/sec)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Plot throughput for each experiment
    for exp_name, df in aligned_throughput.items():
        if not df.empty and 'elapsed_sec' in df.columns:
            ax1.plot(df['elapsed_sec'], df['throughput'], 
                    label=f'Throughput {exp_name}', color='blue', alpha=0.7, linewidth=2)
    
    # Right Y-axis: Loss
    ax2 = ax1.twinx()
    ax2.set_ylabel('ML Loss', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    # Plot loss for each experiment (averaged across servers)
    colors = ['green', 'darkgreen', 'forestgreen']
    for i, (exp_name, df) in enumerate(averaged_ml_data.items()):
        if not df.empty:
            color = colors[i % len(colors)]
            valid_data = df.dropna(subset=['avg_train_loss', 'avg_val_loss'])
            if len(valid_data) > 0:
                ax2.plot(valid_data['time_elapsed'], valid_data['avg_train_loss'], 
                        label=f'{exp_name} Train Loss', color=color, alpha=0.7, linestyle='-')
                ax2.plot(valid_data['time_elapsed'], valid_data['avg_val_loss'], 
                        label=f'{exp_name} Val Loss', color=color, alpha=0.7, linestyle='--')
    
    ax1.set_title('KeyDB Throughput vs ML Loss Over Time (All Experiments)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(0, 1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'throughput_vs_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: ML Metrics Comparison (like result_analyzer.py)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('ML Training Metrics Comparison (All Experiments)', fontsize=16, fontweight='bold')
    
    # Define metrics and their positions
    metrics = [
        ('avg_train_loss', 'Average Training Loss', axes[0, 0]),
        ('avg_train_acc', 'Average Training Accuracy', axes[0, 1]),
        ('avg_val_loss', 'Average Validation Loss', axes[1, 0]),
        ('avg_val_acc', 'Average Validation Accuracy', axes[1, 1])
    ]
    
    # Define colors for different experiments
    exp_colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for metric, title, ax in metrics:
        for i, (exp_name, df) in enumerate(averaged_ml_data.items()):
            if not df.empty and metric in df.columns:
                data = df.dropna(subset=[metric])
                if not data.empty:
                    color = exp_colors[i % len(exp_colors)]
                    ax.plot(data['time_elapsed'], data[metric], 
                           color=color, linewidth=2, 
                           label=f'{exp_name}', alpha=0.8)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Elapsed Time (seconds)', fontsize=10)
        ax.set_ylabel(title.replace('Average ', ''), fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ml_metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Individual Experiment Analysis with Throughput
    for exp_name, df in averaged_ml_data.items():
        if df.empty:
            continue
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle(f'Experiment {exp_name}: ML Metrics vs KeyDB Throughput', fontsize=14, fontweight='bold')
        
        # Top plot: Accuracy vs Throughput
        ax1_twin = ax1.twinx()
        
        # Plot ML accuracy
        valid_data = df.dropna(subset=['avg_train_acc', 'avg_val_acc'])
        if len(valid_data) > 0:
            ax1.plot(valid_data['time_elapsed'], valid_data['avg_train_acc'], 
                    label='Train Acc', color='red', linewidth=2, alpha=0.8)
            ax1.plot(valid_data['time_elapsed'], valid_data['avg_val_acc'], 
                    label='Val Acc', color='red', linewidth=2, alpha=0.8, linestyle='--')
        
        # Plot throughput for this experiment
        if exp_name in aligned_throughput and not aligned_throughput[exp_name].empty:
            throughput_df = aligned_throughput[exp_name]
            ax1_twin.plot(throughput_df['elapsed_sec'], throughput_df['throughput'], 
                         label='KeyDB Throughput', color='blue', linewidth=2, alpha=0.8)
        
        ax1.set_xlabel('Elapsed Time (seconds)')
        ax1.set_ylabel('ML Accuracy', color='red')
        ax1_twin.set_ylabel('KeyDB Throughput (commands/sec)', color='blue')
        ax1.set_title(f'{exp_name}: Accuracy vs Throughput')
        ax1.grid(True, alpha=0.3)
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Bottom plot: Loss vs Throughput
        ax2_twin = ax2.twinx()
        
        # Plot ML loss
        valid_data = df.dropna(subset=['avg_train_loss', 'avg_val_loss'])
        if len(valid_data) > 0:
            ax2.plot(valid_data['time_elapsed'], valid_data['avg_train_loss'], 
                    label='Train Loss', color='green', linewidth=2, alpha=0.8)
            ax2.plot(valid_data['time_elapsed'], valid_data['avg_val_loss'], 
                    label='Val Loss', color='green', linewidth=2, alpha=0.8, linestyle='--')
        
        # Plot throughput for this experiment
        if exp_name in aligned_throughput and not aligned_throughput[exp_name].empty:
            throughput_df = aligned_throughput[exp_name]
            ax2_twin.plot(throughput_df['elapsed_sec'], throughput_df['throughput'], 
                         label='KeyDB Throughput', color='blue', linewidth=2, alpha=0.8, marker='o', markersize=3)
        
        ax2.set_xlabel('Elapsed Time (seconds)')
        ax2.set_ylabel('ML Loss', color='green')
        ax2_twin.set_ylabel('KeyDB Throughput (commands/sec)', color='blue')
        ax2.set_title(f'{exp_name}: Loss vs Throughput')
        ax2.grid(True, alpha=0.3)
        
        # Combine legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'experiment_{exp_name}_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✅ Plots saved to: {output_dir}")

# Main execution
if __name__ == "__main__":
    # Configuration
    ml_logs_dir = "/home/abhattar/serverlogs/ember_results/mlp_async"  # Update this path
    keydb_analysis_dir = "/home/abhattar/serverlogs/ember_results/mlp_async/keydb_logs"  # Current directory where log analysis files are
    output_dir = f"comprehensive_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("Loading ML metrics...")
    ml_data = load_ml_metrics(ml_logs_dir)
    print(f"Loaded ML data from {len(ml_data)} experiments")
    
    print("Loading KeyDB throughput data...")
    throughput_data = load_keydb_throughput(keydb_analysis_dir)
    print(f"Loaded throughput data from {len(throughput_data)} experiments")
    
    print("Creating comprehensive plots...")
    create_comprehensive_plots(ml_data, throughput_data, output_dir)