import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import glob
from typing import List

def plot_throughput_and_training(exp_dir: str, output_filename: str | None = None, y_max_throughput: int | None = None):
    """
    Create a combined figure with:
      - Top: Throughput over elapsed time (all servers and sources), from *_breakdown.csv
      - Bottom: Training and validation metrics (accuracy and loss) over iterations, from average_metrics.csv

    exp_dir layout (example):
      /path/to/experiment_1/
        average_metrics.csv
        experiment_1_keydb/
          experiment_1/
            mteverest1_breakdown.csv
            mteverest3_breakdown.csv
            mteverest4_breakdown.csv
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    import glob

    # 1) Load metrics
    metrics_csv = os.path.join(exp_dir, 'average_metrics.csv')
    if not os.path.isfile(metrics_csv):
        print(f"⚠️ average_metrics.csv not found in {exp_dir}; skipping metrics panel")
        metrics_df = None
    else:
        try:
            metrics_df = pd.read_csv(metrics_csv)
        except Exception as e:
            print(f"❌ Failed to read {metrics_csv}: {e}")
            metrics_df = None

    # 2) Find throughput breakdown CSVs
    keydb_root = None
    for d in os.listdir(exp_dir):
        if d.endswith('_keydb') and os.path.isdir(os.path.join(exp_dir, d)):
            keydb_root = os.path.join(exp_dir, d)
            break
    if keydb_root is None:
        print(f"⚠️ No *_keydb directory found under {exp_dir}; skipping throughput panel")
        breakdown_paths: List[str] = []
    else:
        breakdown_paths = sorted(glob.glob(os.path.join(keydb_root, '*', '*_breakdown.csv')))

    # 3) Build figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})

    # Top: Throughput combined
    ax_top = axes[0]
    any_series = False
    for csv_path in breakdown_paths:
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        server = str(df['server'].iloc[0])
        # Identify series columns
        meta_cols = {"experiment", "server", "second", "elapsed_seconds"}
        source_cols = [c for c in df.columns if c not in meta_cols]
        source_cols = (
            ([c for c in source_cols if c == 'client']) +
            [c for c in source_cols if c not in ('client', 'total')] +
            ([c for c in source_cols if c == 'total'])
        )
        df = df.sort_values(['elapsed_seconds']).reset_index(drop=True)
        for col in source_cols:
            label = f"{server}:{col}"
            ax_top.plot(df['elapsed_seconds'], df[col], label=label, linewidth=1.5, alpha=0.9)
            any_series = True
    ax_top.set_title('Throughput over elapsed time (all servers & sources)')
    ax_top.set_xlabel('Elapsed seconds')
    ax_top.set_ylabel('Commands per second (GET/SET)')
    ax_top.grid(True, alpha=0.3)
    if any_series:
        ax_top.legend(loc='upper right', ncol=3, fontsize=8)
    if y_max_throughput is not None:
        ax_top.set_ylim(0, y_max_throughput)

    # Bottom: Training metrics
    ax_bottom = axes[1]
    if metrics_df is None or metrics_df.empty:
        ax_bottom.text(0.5, 0.5, 'No metrics available', ha='center', va='center')
        ax_bottom.axis('off')
    else:
        # Plot accuracy on left y-axis, loss on right y-axis
        dfm = metrics_df.dropna(subset=['iteration'])
        dfm = dfm.sort_values(['iteration'])
        ax_acc = ax_bottom
        ax_loss = ax_bottom.twinx()

        # Accuracy
        if 'avg_train_acc' in dfm.columns:
            ax_acc.plot(dfm['iteration'], dfm['avg_train_acc'], label='Train Acc', color='tab:blue', linewidth=2)
        if 'avg_val_acc' in dfm.columns:
            ax_acc.plot(dfm['iteration'], dfm['avg_val_acc'], label='Val Acc', color='tab:cyan', linewidth=2, linestyle='--')
        ax_acc.set_ylabel('Accuracy', color='tab:blue')
        ax_acc.tick_params(axis='y', labelcolor='tab:blue')

        # Loss
        if 'avg_train_loss' in dfm.columns:
            ax_loss.plot(dfm['iteration'], dfm['avg_train_loss'], label='Train Loss', color='tab:red', linewidth=2)
        if 'avg_val_loss' in dfm.columns:
            ax_loss.plot(dfm['iteration'], dfm['avg_val_loss'], label='Val Loss', color='tab:orange', linewidth=2, linestyle='--')
        ax_loss.set_ylabel('Loss', color='tab:red')
        ax_loss.tick_params(axis='y', labelcolor='tab:red')

        ax_acc.set_xlabel('Iteration')
        ax_acc.grid(True, alpha=0.3)

        # Compose legend
        lines = []
        labels = []
        for ax in (ax_acc, ax_loss):
            h, l = ax.get_legend_handles_labels()
            lines += h; labels += l
        ax_acc.legend(lines, labels, loc='upper right')
        ax_bottom.set_title('Training/Validation Metrics')

    plt.tight_layout()
    if output_filename is None:
        output_filename = os.path.join(exp_dir, 'combined_throughput_and_training.png')
    plt.savefig(output_filename, dpi=200)
    plt.close()
    print(f"✅ Saved combined throughput+metrics: {output_filename}")

def load_experiment_data(result_data_dir):
    """
    Load all experiment data from the Ember results directory structure
    
    Expected structure:
    ~/serverlogs/ember_results/
    ├── cnn_async/plots/
    │   ├── experiment_1/average_metrics.csv
    │   ├── experiment_2/average_metrics.csv
    │   └── experiment_3/average_metrics.csv
    ├── cnn_sync/plots/
    │   ├── experiment_1/average_metrics.csv
    │   ├── experiment_2/average_metrics.csv
    │   └── experiment_3/average_metrics.csv
    ├── mlp_async/plots/
    │   ├── experiment_1/average_metrics.csv
    │   ├── experiment_2/average_metrics.csv
    │   └── experiment_3/average_metrics.csv
    └── mlp_sync/plots/
        ├── experiment_1/average_metrics.csv
        ├── experiment_2/average_metrics.csv
        └── experiment_3/average_metrics.csv
    
    Returns:
        dict: Nested dictionary with structure:
        {
            'sync_cnn': {'experiment_1': df, 'experiment_2': df, 'experiment_3': df},
            'sync_mlp': {'experiment_1': df, 'experiment_2': df, 'experiment_3': df},
            'async_cnn': {'experiment_1': df, 'experiment_2': df, 'experiment_3': df},
            'async_mlp': {'experiment_1': df, 'experiment_2': df, 'experiment_3': df}
        }
    """
    data = {}
    
    # Define the four experiment types and their directory names
    experiment_types = {
        'sync_cnn': 'cnn_sync',
        'sync_mlp': 'mlp_sync', 
        'async_cnn': 'cnn_async',
        'async_mlp': 'mlp_async'
    }
    
    for exp_type, dir_name in experiment_types.items():
        data[exp_type] = {}
        exp_dir = os.path.join(result_data_dir, dir_name, 'plots')
        
        if os.path.exists(exp_dir):
            # Find all experiment subdirectories (handle both 'experiment_' and 'expirement_' typos)
            experiment_dirs = [d for d in os.listdir(exp_dir) 
                             if os.path.isdir(os.path.join(exp_dir, d)) and 
                             (d.startswith('experiment_') or d.startswith('expirement_'))]
            
            for exp_dir_name in sorted(experiment_dirs):
                # Extract experiment number from directory name (handle both spellings)
                exp_num = exp_dir_name.replace('experiment_', '').replace('expirement_', '')
                
                # Look for average_metrics.csv in the experiment directory
                csv_file = os.path.join(exp_dir, exp_dir_name, 'average_metrics.csv')
                
                if os.path.exists(csv_file):
                    try:
                        df = pd.read_csv(csv_file)
                        data[exp_type][f'experiment_{exp_num}'] = df
                        print(f"✅ Loaded {exp_type}/{exp_dir_name}: {len(df)} rows")
                    except Exception as e:
                        print(f"❌ Error loading {csv_file}: {e}")
                else:
                    print(f"⚠️ No average_metrics.csv found in {csv_file}")
        else:
            print(f"⚠️ Directory not found: {exp_dir}")
    
    return data

def get_experiment_data_for_plotting(experiment_dict):
    """
    Get experiment data ready for plotting - each experiment is already averaged
    
    Args:
        experiment_dict: Dictionary with experiment dataframes (already averaged)
        
    Returns:
        dict: Dictionary with experiment dataframes ready for plotting
    """
    return experiment_dict

def plot_individual_model_comparison(data, output_dir, max_iteration=None):
    """
    Create plots comparing sync vs async for CNN and MLP separately
    
    Args:
        data: Dictionary containing all experiment data
        output_dir: Directory to save plots
        max_iteration: Maximum iteration to plot (None = plot all)
    """
    print("\n📊 Creating individual model comparison plots...")
    
    # CNN Comparison (sync vs async)
    if 'sync_cnn' in data and 'async_cnn' in data:
        plot_sync_vs_async(data['sync_cnn'], data['async_cnn'], 'CNN', output_dir, max_iteration)
    
    # MLP Comparison (sync vs async)
    if 'sync_mlp' in data and 'async_mlp' in data:
        plot_sync_vs_async(data['sync_mlp'], data['async_mlp'], 'MLP', output_dir, max_iteration)

def plot_sync_vs_async(sync_experiments, async_experiments, model_type, output_dir, max_iteration=None):
    """
    Plot sync vs async comparison for a specific model type
    
    Args:
        sync_experiments: Dictionary of sync experiment dataframes
        async_experiments: Dictionary of async experiment dataframes
        model_type: String indicating model type (CNN/MLP)
        output_dir: Directory to save plots
        max_iteration: Maximum iteration to plot (None = plot all)
    """
    if not sync_experiments or not async_experiments:
        print(f"⚠️ No data available for {model_type} comparison")
        return
    
    # Create figure with 4 subplots (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Ember {model_type} Training: Sync vs Async Comparison', fontsize=16, fontweight='bold')
    
    # Define metrics and their positions
    metrics = [
        ('avg_train_loss', 'Average Training Loss', axes[0, 0]),
        ('avg_train_acc', 'Average Training Accuracy', axes[0, 1]),
        ('avg_val_loss', 'Average Validation Loss', axes[1, 0]),
        ('avg_val_acc', 'Average Validation Accuracy', axes[1, 1])
    ]
    
    # Define colors for different experiments
    sync_colors = ['blue', 'navy', 'darkblue']
    async_colors = ['red', 'darkred', 'maroon']
    
    for metric, title, ax in metrics:
        # Plot sync experiments
        for i, (exp_name, df) in enumerate(sync_experiments.items()):
            if not df.empty and metric in df.columns:
                data = df.dropna(subset=[metric])
                # Filter by max_iteration if specified
                if max_iteration is not None:
                    data = data[data['iteration'] <= max_iteration]
                if not data.empty:
                    color = sync_colors[i % len(sync_colors)]
                    ax.plot(data['iteration'], data[metric], 
                           color=color, linewidth=2, 
                           label=f'Sync {model_type} {exp_name}', alpha=0.8)
        
        # Plot async experiments
        for i, (exp_name, df) in enumerate(async_experiments.items()):
            if not df.empty and metric in df.columns:
                data = df.dropna(subset=[metric])
                # Filter by max_iteration if specified
                if max_iteration is not None:
                    data = data[data['iteration'] <= max_iteration]
                if not data.empty:
                    color = async_colors[i % len(async_colors)]
                    ax.plot(data['iteration'], data[metric], 
                           color=color, linewidth=2, 
                           label=f'Async {model_type} {exp_name}', alpha=0.8)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Iterations', fontsize=10)
        ax.set_ylabel(title.replace('Average ', ''), fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    
    # Save combined plot
    filename = os.path.join(output_dir, f'{model_type.lower()}_sync_vs_async_comparison.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved {model_type} sync vs async comparison: {filename}")
    
    # Create individual subplot images
    subplot_dir = os.path.join(output_dir, 'individual_subplots')
    os.makedirs(subplot_dir, exist_ok=True)
    
    for metric, title, ax in metrics:
        # Create individual figure for each metric
        fig, ax_individual = plt.subplots(1, 1, figsize=(10, 6))
        
        # Plot sync experiments
        for i, (exp_name, df) in enumerate(sync_experiments.items()):
            if not df.empty and metric in df.columns:
                data = df.dropna(subset=[metric])
                if max_iteration is not None:
                    data = data[data['iteration'] <= max_iteration]
                if not data.empty:
                    color = sync_colors[i % len(sync_colors)]
                    ax_individual.plot(data['iteration'], data[metric], 
                                     color=color, linewidth=2, 
                                     label=f'Sync {model_type} {exp_name}', alpha=0.8)
        
        # Plot async experiments
        for i, (exp_name, df) in enumerate(async_experiments.items()):
            if not df.empty and metric in df.columns:
                data = df.dropna(subset=[metric])
                if max_iteration is not None:
                    data = data[data['iteration'] <= max_iteration]
                if not data.empty:
                    color = async_colors[i % len(async_colors)]
                    ax_individual.plot(data['iteration'], data[metric], 
                                     color=color, linewidth=2, 
                                     label=f'Async {model_type} {exp_name}', alpha=0.8)
        
        ax_individual.set_title(f'Ember {title}', fontsize=14, fontweight='bold')
        ax_individual.set_xlabel('Iterations', fontsize=12)
        ax_individual.set_ylabel(title.replace('Average ', ''), fontsize=12)
        ax_individual.grid(True, alpha=0.3)
        ax_individual.legend()
        
        # Save individual subplot
        metric_name = metric.replace('avg_', '').replace('_', '_')
        individual_filename = os.path.join(subplot_dir, f'{model_type.lower()}_sync_vs_async_{metric_name}.png')
        plt.savefig(individual_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved individual subplot: {individual_filename}")

def plot_cross_model_comparison(data, output_dir, max_iteration=None):
    """
    Create plots comparing CNN vs MLP for sync and async separately
    
    Args:
        data: Dictionary containing all experiment data
        output_dir: Directory to save plots
        max_iteration: Maximum iteration to plot (None = plot all)
    """
    print("\n📊 Creating cross-model comparison plots...")
    
    # Sync Comparison (CNN vs MLP)
    if 'sync_cnn' in data and 'sync_mlp' in data:
        plot_cnn_vs_mlp(data['sync_cnn'], data['sync_mlp'], 'Sync', output_dir, max_iteration)
    
    # Async Comparison (CNN vs MLP)
    if 'async_cnn' in data and 'async_mlp' in data:
        plot_cnn_vs_mlp(data['async_cnn'], data['async_mlp'], 'Async', output_dir, max_iteration)

def plot_cnn_vs_mlp(cnn_experiments, mlp_experiments, mode, output_dir, max_iteration=None):
    """
    Plot CNN vs MLP comparison for a specific mode (sync/async)
    
    Args:
        cnn_experiments: Dictionary of CNN experiment dataframes
        mlp_experiments: Dictionary of MLP experiment dataframes
        mode: String indicating training mode (Sync/Async)
        output_dir: Directory to save plots
        max_iteration: Maximum iteration to plot (None = plot all)
    """
    if not cnn_experiments or not mlp_experiments:
        print(f"⚠️ No data available for {mode} CNN vs MLP comparison")
        return
    
    # Create figure with 4 subplots (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Ember {mode} Training: CNN vs MLP Comparison', fontsize=16, fontweight='bold')
    
    # Define metrics and their positions
    metrics = [
        ('avg_train_loss', 'Average Training Loss', axes[0, 0]),
        ('avg_train_acc', 'Average Training Accuracy', axes[0, 1]),
        ('avg_val_loss', 'Average Validation Loss', axes[1, 0]),
        ('avg_val_acc', 'Average Validation Accuracy', axes[1, 1])
    ]
    
    # Define colors for different experiments
    cnn_colors = ['blue', 'navy', 'darkblue']
    mlp_colors = ['red', 'darkred', 'maroon']
    
    for metric, title, ax in metrics:
        # Plot CNN experiments
        for i, (exp_name, df) in enumerate(cnn_experiments.items()):
            if not df.empty and metric in df.columns:
                data = df.dropna(subset=[metric])
                # Filter by max_iteration if specified
                if max_iteration is not None:
                    data = data[data['iteration'] <= max_iteration]
                if not data.empty:
                    color = cnn_colors[i % len(cnn_colors)]
                    ax.plot(data['iteration'], data[metric], 
                           color=color, linewidth=2, 
                           label=f'{mode} CNN {exp_name}', alpha=0.8)
        
        # Plot MLP experiments
        for i, (exp_name, df) in enumerate(mlp_experiments.items()):
            if not df.empty and metric in df.columns:
                data = df.dropna(subset=[metric])
                # Filter by max_iteration if specified
                if max_iteration is not None:
                    data = data[data['iteration'] <= max_iteration]
                if not data.empty:
                    color = mlp_colors[i % len(mlp_colors)]
                    ax.plot(data['iteration'], data[metric], 
                           color=color, linewidth=2, 
                           label=f'{mode} MLP {exp_name}', alpha=0.8)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Iterations', fontsize=10)
        ax.set_ylabel(title.replace('Average ', ''), fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    
    # Save combined plot
    filename = os.path.join(output_dir, f'{mode.lower()}_cnn_vs_mlp_comparison.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved {mode} CNN vs MLP comparison: {filename}")
    
    # Create individual subplot images
    subplot_dir = os.path.join(output_dir, 'individual_subplots')
    os.makedirs(subplot_dir, exist_ok=True)
    
    for metric, title, ax in metrics:
        # Create individual figure for each metric
        fig, ax_individual = plt.subplots(1, 1, figsize=(10, 6))
        
        # Plot CNN experiments
        for i, (exp_name, df) in enumerate(cnn_experiments.items()):
            if not df.empty and metric in df.columns:
                data = df.dropna(subset=[metric])
                if max_iteration is not None:
                    data = data[data['iteration'] <= max_iteration]
                if not data.empty:
                    color = cnn_colors[i % len(cnn_colors)]
                    ax_individual.plot(data['iteration'], data[metric], 
                                     color=color, linewidth=2, 
                                     label=f'{mode} CNN {exp_name}', alpha=0.8)
        
        # Plot MLP experiments
        for i, (exp_name, df) in enumerate(mlp_experiments.items()):
            if not df.empty and metric in df.columns:
                data = df.dropna(subset=[metric])
                if max_iteration is not None:
                    data = data[data['iteration'] <= max_iteration]
                if not data.empty:
                    color = mlp_colors[i % len(mlp_colors)]
                    ax_individual.plot(data['iteration'], data[metric], 
                                     color=color, linewidth=2, 
                                     label=f'{mode} MLP {exp_name}', alpha=0.8)
        
        ax_individual.set_title(f'Ember {title}', fontsize=14, fontweight='bold')
        ax_individual.set_xlabel('Iterations', fontsize=12)
        ax_individual.set_ylabel(title.replace('Average ', ''), fontsize=12)
        ax_individual.grid(True, alpha=0.3)
        ax_individual.legend()
        
        # Save individual subplot
        metric_name = metric.replace('avg_', '').replace('_', '_')
        individual_filename = os.path.join(subplot_dir, f'{mode.lower()}_cnn_vs_mlp_{metric_name}.png')
        plt.savefig(individual_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved individual subplot: {individual_filename}")

def plot_all_models_together(data, output_dir, max_iteration=None):
    """
    Create comprehensive plots with all four combinations
    
    Args:
        data: Dictionary containing all experiment data
        output_dir: Directory to save plots
        max_iteration: Maximum iteration to plot (None = plot all)
    """
    print("\n📊 Creating comprehensive all-models comparison plots...")
    
    if len(data) < 2:
        print("⚠️ Need at least 2 experiment types for comprehensive comparison")
        return
    
    # Create figure with 4 subplots (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Ember Comprehensive Model Comparison: All Training Modes', fontsize=16, fontweight='bold')
    
    # Define metrics and their positions
    metrics = [
        ('avg_train_loss', 'Average Training Loss', axes[0, 0]),
        ('avg_train_acc', 'Average Training Accuracy', axes[0, 1]),
        ('avg_val_loss', 'Average Validation Loss', axes[1, 0]),
        ('avg_val_acc', 'Average Validation Accuracy', axes[1, 1])
    ]
    
    # Define colors and labels for each experiment type
    base_colors = {
        'sync_cnn': 'blue',
        'sync_mlp': 'red', 
        'async_cnn': 'green',
        'async_mlp': 'orange'
    }
    
    labels = {
        'sync_cnn': 'Sync CNN',
        'sync_mlp': 'Sync MLP',
        'async_cnn': 'Async CNN', 
        'async_mlp': 'Async MLP'
    }
    
    for metric, title, ax in metrics:
        for exp_type, experiments in data.items():
            if not experiments:
                continue
                
            base_color = base_colors.get(exp_type, 'black')
            base_label = labels.get(exp_type, exp_type)
            
            # Plot each experiment for this type
            for i, (exp_name, df) in enumerate(experiments.items()):
                if not df.empty and metric in df.columns:
                    filtered_data = df.dropna(subset=[metric])
                    # Filter by max_iteration if specified
                    if max_iteration is not None:
                        filtered_data = filtered_data[filtered_data['iteration'] <= max_iteration]
                    if not filtered_data.empty:
                        # Use different shades of the base color for different experiments
                        color = base_color
                        label = f"{base_label} {exp_name}"
                        
                        ax.plot(filtered_data['iteration'], filtered_data[metric], 
                               color=color, linewidth=2, label=label, alpha=0.8)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Iterations', fontsize=10)
        ax.set_ylabel(title.replace('Average ', ''), fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    
    # Save combined plot
    filename = os.path.join(output_dir, 'comprehensive_all_models_comparison.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved comprehensive comparison: {filename}")
    
    # Create individual subplot images
    subplot_dir = os.path.join(output_dir, 'individual_subplots')
    os.makedirs(subplot_dir, exist_ok=True)
    
    for metric, title, ax in metrics:
        # Create individual figure for each metric
        fig, ax_individual = plt.subplots(1, 1, figsize=(12, 8))
        
        for exp_type, experiments in data.items():
            if not experiments:
                continue
                
            base_color = base_colors.get(exp_type, 'black')
            base_label = labels.get(exp_type, exp_type)
            
            # Plot each experiment for this type
            for i, (exp_name, df) in enumerate(experiments.items()):
                if not df.empty and metric in df.columns:
                    filtered_data = df.dropna(subset=[metric])
                    if max_iteration is not None:
                        filtered_data = filtered_data[filtered_data['iteration'] <= max_iteration]
                    if not filtered_data.empty:
                        color = base_color
                        label = f"{base_label} {exp_name}"
                        
                        ax_individual.plot(filtered_data['iteration'], filtered_data[metric], 
                                         color=color, linewidth=2, label=label, alpha=0.8)
        
        ax_individual.set_title(f'Ember {title}', fontsize=14, fontweight='bold')
        ax_individual.set_xlabel('Iterations', fontsize=12)
        ax_individual.set_ylabel(title.replace('Average ', ''), fontsize=12)
        ax_individual.grid(True, alpha=0.3)
        ax_individual.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Save individual subplot
        metric_name = metric.replace('avg_', '').replace('_', '_')
        individual_filename = os.path.join(subplot_dir, f'comprehensive_all_models_{metric_name}.png')
        plt.savefig(individual_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved individual subplot: {individual_filename}")

def plot_combined_metrics(data, output_dir, max_iteration=None):
    """
    Create combined plots showing all accuracy metrics and all loss metrics together
    
    Args:
        data: Dictionary containing all experiment data
        output_dir: Directory to save plots
        max_iteration: Maximum iteration to plot (None = plot all)
    """
    print("\n📊 Creating combined metrics plots...")
    
    if len(data) < 2:
        print("⚠️ Need at least 2 experiment types for combined comparison")
        return
    
    # Create figure with 2 subplots (accuracy and loss)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('Ember Combined Metrics Comparison: All Training Modes', fontsize=16, fontweight='bold')
    
    # Define colors and labels for each experiment type
    base_colors = {
        'sync_cnn': 'blue',
        'sync_mlp': 'red', 
        'async_cnn': 'green',
        'async_mlp': 'orange'
    }
    
    labels = {
        'sync_cnn': 'Sync CNN',
        'sync_mlp': 'Sync MLP',
        'async_cnn': 'Async CNN', 
        'async_mlp': 'Async MLP'
    }
    
    # Plot 1: All Accuracy Metrics
    ax1.set_title('All Accuracy Metrics vs Iterations', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Iterations', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    for exp_type, experiments in data.items():
        if not experiments:
            continue
            
        base_color = base_colors.get(exp_type, 'black')
        base_label = labels.get(exp_type, exp_type)
        
        # Plot each experiment for this type
        for i, (exp_name, df) in enumerate(experiments.items()):
            if not df.empty:
                # Plot training accuracy
                train_data = df.dropna(subset=['avg_train_acc'])
                if max_iteration is not None:
                    train_data = train_data[train_data['iteration'] <= max_iteration]
                if not train_data.empty:
                    color = base_color
                    label = f"{base_label} {exp_name} Train"
                    ax1.plot(train_data['iteration'], train_data['avg_train_acc'], 
                           color=color, linewidth=2, label=label, alpha=0.8, linestyle='-')
                
                # Plot validation accuracy
                val_data = df.dropna(subset=['avg_val_acc'])
                if max_iteration is not None:
                    val_data = val_data[val_data['iteration'] <= max_iteration]
                if not val_data.empty:
                    color = base_color
                    label = f"{base_label} {exp_name} Val"
                    ax1.plot(val_data['iteration'], val_data['avg_val_acc'], 
                           color=color, linewidth=2, label=label, alpha=0.8, linestyle='--')
    
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.set_ylim(0, 1)  # Accuracy is between 0 and 1
    
    # Plot 2: All Loss Metrics
    ax2.set_title('All Loss Metrics vs Iterations', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Iterations', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    for exp_type, experiments in data.items():
        if not experiments:
            continue
            
        base_color = base_colors.get(exp_type, 'black')
        base_label = labels.get(exp_type, exp_type)
        
        # Plot each experiment for this type
        for i, (exp_name, df) in enumerate(experiments.items()):
            if not df.empty:
                # Plot training loss
                train_data = df.dropna(subset=['avg_train_loss'])
                if max_iteration is not None:
                    train_data = train_data[train_data['iteration'] <= max_iteration]
                if not train_data.empty:
                    color = base_color
                    label = f"{base_label} {exp_name} Train"
                    ax2.plot(train_data['iteration'], train_data['avg_train_loss'], 
                           color=color, linewidth=2, label=label, alpha=0.8, linestyle='-')
                
                # Plot validation loss
                val_data = df.dropna(subset=['avg_val_loss'])
                if max_iteration is not None:
                    val_data = val_data[val_data['iteration'] <= max_iteration]
                if not val_data.empty:
                    color = base_color
                    label = f"{base_label} {exp_name} Val"
                    ax2.plot(val_data['iteration'], val_data['avg_val_loss'], 
                           color=color, linewidth=2, label=label, alpha=0.8, linestyle='--')
    
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Save combined plot
    filename = os.path.join(output_dir, 'combined_metrics_comparison.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved combined metrics comparison: {filename}")
    
    # Create individual subplot images for accuracy and loss
    subplot_dir = os.path.join(output_dir, 'individual_subplots')
    os.makedirs(subplot_dir, exist_ok=True)
    
    # Individual Accuracy Plot
    fig, ax_acc = plt.subplots(1, 1, figsize=(12, 8))
    ax_acc.set_title('Ember All Accuracy Metrics vs Iterations', fontsize=14, fontweight='bold')
    ax_acc.set_xlabel('Iterations', fontsize=12)
    ax_acc.set_ylabel('Accuracy', fontsize=12)
    ax_acc.grid(True, alpha=0.3)
    
    for exp_type, experiments in data.items():
        if not experiments:
            continue
            
        base_color = base_colors.get(exp_type, 'black')
        base_label = labels.get(exp_type, exp_type)
        
        # Plot each experiment for this type
        for i, (exp_name, df) in enumerate(experiments.items()):
            if not df.empty:
                # Plot training accuracy
                train_data = df.dropna(subset=['avg_train_acc'])
                if max_iteration is not None:
                    train_data = train_data[train_data['iteration'] <= max_iteration]
                if not train_data.empty:
                    color = base_color
                    label = f"{base_label} {exp_name} Train"
                    ax_acc.plot(train_data['iteration'], train_data['avg_train_acc'], 
                               color=color, linewidth=2, label=label, alpha=0.8, linestyle='-')
                
                # Plot validation accuracy
                val_data = df.dropna(subset=['avg_val_acc'])
                if max_iteration is not None:
                    val_data = val_data[val_data['iteration'] <= max_iteration]
                if not val_data.empty:
                    color = base_color
                    label = f"{base_label} {exp_name} Val"
                    ax_acc.plot(val_data['iteration'], val_data['avg_val_acc'], 
                               color=color, linewidth=2, label=label, alpha=0.8, linestyle='--')
    
    ax_acc.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_acc.set_ylim(0, 1)  # Accuracy is between 0 and 1
    
    # Save individual accuracy plot
    acc_filename = os.path.join(subplot_dir, 'combined_metrics_accuracy.png')
    plt.savefig(acc_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved individual subplot: {acc_filename}")
    
    # Individual Loss Plot
    fig, ax_loss = plt.subplots(1, 1, figsize=(12, 8))
    ax_loss.set_title('Ember All Loss Metrics vs Iterations', fontsize=14, fontweight='bold')
    ax_loss.set_xlabel('Iterations', fontsize=12)
    ax_loss.set_ylabel('Loss', fontsize=12)
    ax_loss.grid(True, alpha=0.3)
    
    for exp_type, experiments in data.items():
        if not experiments:
            continue
            
        base_color = base_colors.get(exp_type, 'black')
        base_label = labels.get(exp_type, exp_type)
        
        # Plot each experiment for this type
        for i, (exp_name, df) in enumerate(experiments.items()):
            if not df.empty:
                # Plot training loss
                train_data = df.dropna(subset=['avg_train_loss'])
                if max_iteration is not None:
                    train_data = train_data[train_data['iteration'] <= max_iteration]
                if not train_data.empty:
                    color = base_color
                    label = f"{base_label} {exp_name} Train"
                    ax_loss.plot(train_data['iteration'], train_data['avg_train_loss'], 
                                color=color, linewidth=2, label=label, alpha=0.8, linestyle='-')
                
                # Plot validation loss
                val_data = df.dropna(subset=['avg_val_loss'])
                if max_iteration is not None:
                    val_data = val_data[val_data['iteration'] <= max_iteration]
                if not val_data.empty:
                    color = base_color
                    label = f"{base_label} {exp_name} Val"
                    ax_loss.plot(val_data['iteration'], val_data['avg_val_loss'], 
                                color=color, linewidth=2, label=label, alpha=0.8, linestyle='--')
    
    ax_loss.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save individual loss plot
    loss_filename = os.path.join(subplot_dir, 'combined_metrics_loss.png')
    plt.savefig(loss_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved individual subplot: {loss_filename}")

def generate_summary_statistics(data, output_dir):
    """
    Generate summary statistics for all experiments
    """
    print("\n📊 Generating summary statistics...")
    
    summary_file = os.path.join(output_dir, 'experiment_summary.txt')
    
    with open(summary_file, 'w') as f:
        f.write("Ember Distributed Training Experiment Summary\n")
        f.write("=" * 50 + "\n\n")
        
        for exp_type, experiments in data.items():
            f.write(f"{exp_type.upper()}:\n")
            f.write("-" * 20 + "\n")
            
            if not experiments:
                f.write("  No experiments found\n\n")
                continue
            
            f.write(f"  Number of experiments: {len(experiments)}\n")
            
            # Analyze each experiment individually
            for exp_name, df in experiments.items():
                if df.empty:
                    continue
                    
                f.write(f"\n  {exp_name}:\n")
                f.write(f"    Iteration range: {df['iteration'].min()} to {df['iteration'].max()}\n")
                
                # Get results at iteration 1000 or last iteration if 1000 doesn't exist
                target_iteration = 1000
                if target_iteration in df['iteration'].values:
                    result_row = df[df['iteration'] == target_iteration].iloc[0]
                    iteration_label = f"at iteration {target_iteration}"
                else:
                    result_row = df.iloc[-1]
                    iteration_label = f"at final iteration {int(result_row['iteration'])}"
                
                f.write(f"    Results {iteration_label}:\n")
                f.write(f"    Training Loss: {result_row['avg_train_loss']:.4f}\n")
                f.write(f"    Training Acc: {result_row['avg_train_acc']:.4f}\n")
                f.write(f"    Validation Loss: {result_row['avg_val_loss']:.4f}\n")
                f.write(f"    Validation Acc: {result_row['avg_val_acc']:.4f}\n")
                
                # Add F1, precision, recall if available
                if 'avg_val_f1' in df.columns and not pd.isna(result_row['avg_val_f1']):
                    f.write(f"    Validation F1: {result_row['avg_val_f1']:.4f}\n")
                if 'avg_val_precision' in df.columns and not pd.isna(result_row['avg_val_precision']):
                    f.write(f"    Validation Precision: {result_row['avg_val_precision']:.4f}\n")
                if 'avg_val_recall' in df.columns and not pd.isna(result_row['avg_val_recall']):
                    f.write(f"    Validation Recall: {result_row['avg_val_recall']:.4f}\n")
                
                # Best performance
                best_train_acc = df['avg_train_acc'].max()
                best_val_acc = df['avg_val_acc'].max()
                f.write(f"    Best Training Accuracy: {best_train_acc:.4f}\n")
                f.write(f"    Best Validation Accuracy: {best_val_acc:.4f}\n")
                
                # Best F1, precision, recall if available
                if 'avg_val_f1' in df.columns and not df['avg_val_f1'].isna().all():
                    best_val_f1 = df['avg_val_f1'].max()
                    f.write(f"    Best Validation F1: {best_val_f1:.4f}\n")
                if 'avg_val_precision' in df.columns and not df['avg_val_precision'].isna().all():
                    best_val_precision = df['avg_val_precision'].max()
                    f.write(f"    Best Validation Precision: {best_val_precision:.4f}\n")
                if 'avg_val_recall' in df.columns and not df['avg_val_recall'].isna().all():
                    best_val_recall = df['avg_val_recall'].max()
                    f.write(f"    Best Validation Recall: {best_val_recall:.4f}\n")
            
            f.write("\n")
    
    print(f"✅ Summary statistics saved: {summary_file}")

def main():
    """
    Main function to analyze all experimental results
    """
    # Set up paths
    result_data_dir = "/home/abhattar/serverlogs/ember_results"
    output_dir = result_data_dir + "/" + f"ember_experiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # ⚙️ CONFIGURATION: Set maximum iteration to plot
    # Set to None to plot all iterations, or specify a number (e.g., 2000, 5000, etc.)
    max_iteration = 1000  # Change this to limit iteration range
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Created output directory: {output_dir}")
    
    if max_iteration is not None:
        print(f"📊 Limiting plots to iterations <= {max_iteration}")
        output_dir += f"_iter{max_iteration}"
        os.makedirs(output_dir, exist_ok=True)
    
    # Load all experiment data
    print("🔄 Loading experiment data...")
    data = load_experiment_data(result_data_dir)
    
    if not data:
        print("❌ No experiment data found!")
        return
    
    # Generate all types of plots
    plot_individual_model_comparison(data, output_dir, max_iteration)
    plot_cross_model_comparison(data, output_dir, max_iteration)
    plot_all_models_together(data, output_dir, max_iteration)
    plot_combined_metrics(data, output_dir, max_iteration)
    
    # Generate summary statistics
    generate_summary_statistics(data, output_dir)
    
    print(f"\n🎉 Analysis complete! All results saved to: {output_dir}")

if __name__ == "__main__":
    main()
