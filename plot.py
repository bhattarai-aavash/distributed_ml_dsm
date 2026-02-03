import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np

def load_and_plot_training_logs(logs_dir, max_iteration=None):
    """Load training logs from all servers and create plots for loss and accuracy"""
    
    # Colors for each server
    colors = {'yangra1': 'blue', 'yangra2': 'red', 'yangra3': 'green', 'yangra6': 'orange', 
              'mteverest1': 'blue', 'mteverest3': 'red', 'mteverest4': 'green'}
    
    # Store data for each server
    server_data = {}
    
    # Load data from each server
    for server_dir in os.listdir(logs_dir):
        if os.path.isdir(os.path.join(logs_dir, server_dir)):
            server_path = os.path.join(logs_dir, server_dir)
            
            # Find CSV files in server directory
            csv_files = [f for f in os.listdir(server_path) if f.endswith('.csv')]
            
            if csv_files:
                csv_file = os.path.join(server_path, csv_files[0])
                print(f"Loading data from {server_dir}: {csv_file}")
                
                # Load CSV data
                df = pd.read_csv(csv_file)
                
                # Check if the CSV has headers (first row should contain 'timestamp')
                if 'timestamp' not in df.columns:
                    # If no headers, add them manually
                    expected_columns = ['timestamp', 'server_id', 'iteration', 'epoch', 'batch_loss', 'batch_acc', 
                                      'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_f1', 'val_precision', 'val_recall',
                                      'learning_rate', 'time_elapsed']
                    df.columns = expected_columns
                    print(f"  ⚠️ Added missing column headers to {server_dir}")
                
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Filter by max_iteration if specified
                if max_iteration is not None:
                    df = df[df['iteration'] <= max_iteration]
                    print(f"  Filtered to iterations <= {max_iteration}: {len(df)} rows")
                
                # Store data
                server_data[server_dir] = df
                
                print(f"  Loaded {len(df)} rows")
                print(f"  Iterations: {df['iteration'].min()} to {df['iteration'].max()}")
                print(f"  Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                print(f"  Elapsed time: {df['time_elapsed'].min():.1f}s to {df['time_elapsed'].max():.1f}s")
    
    if not server_data:
        print("No training data found!")
        return
    
    # Create unique plots directory
    plots_dir = f"training_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if max_iteration is not None:
        plots_dir += f"_iter{max_iteration}"
    os.makedirs(plots_dir, exist_ok=True)
    print(f"\n📁 Created plots directory: {plots_dir}")
    
    # Calculate average metrics across all servers
    average_data = calculate_average_metrics(server_data)
    
    # Save average metrics to CSV
    average_csv_path = os.path.join(plots_dir, "average_metrics.csv")
    average_data.to_csv(average_csv_path, index=False)
    print(f"📊 Average metrics saved to: {average_csv_path}")
    
    # Also compute average elapsed time for all iterations available in the log
    if not average_data.empty and 'iteration' in average_data.columns and 'time_elapsed' in average_data.columns:
        # Create a separate CSV with just iteration and average elapsed time for all iterations
        elapsed_time_data = average_data[['iteration', 'time_elapsed']].copy()
        elapsed_time_data.rename(columns={'time_elapsed': 'avg_time_elapsed'}, inplace=True)
        elapsed_time_csv_path = os.path.join(plots_dir, "average_elapsed_time_all_iterations.csv")
        elapsed_time_data.to_csv(elapsed_time_csv_path, index=False)
        print(f"⏱️ Average elapsed time for all iterations saved to: {elapsed_time_csv_path}")
        print(f"   Total iterations with elapsed time data: {len(elapsed_time_data)}")
    
    # Create plots: 9 rows, 2 columns (including F1, precision, recall)
    fig, axes = plt.subplots(9, 2, figsize=(16, 45))
    
    # Plot titles and labels
    plot_configs = [
        # Row 1: Training Loss
        (axes[0, 0], 'Training Loss vs Elapsed Time (seconds)', 'Elapsed Time (seconds)', 'Training Loss'),
        (axes[0, 1], 'Training Loss vs Iterations', 'Iterations', 'Training Loss'),
        
        # Row 2: Training Accuracy
        (axes[1, 0], 'Training Accuracy vs Elapsed Time (seconds)', 'Elapsed Time (seconds)', 'Training Accuracy'),
        (axes[1, 1], 'Training Accuracy vs Iterations', 'Iterations', 'Training Accuracy'),
        
        # Row 3: Validation Loss
        (axes[2, 0], 'Validation Loss vs Elapsed Time (seconds)', 'Elapsed Time (seconds)', 'Validation Loss'),
        (axes[2, 1], 'Validation Loss vs Iterations', 'Iterations', 'Validation Loss'),
        
        # Row 4: Validation Accuracy
        (axes[3, 0], 'Validation Accuracy vs Elapsed Time (seconds)', 'Elapsed Time (seconds)', 'Validation Accuracy'),
        (axes[3, 1], 'Validation Accuracy vs Iterations', 'Iterations', 'Validation Accuracy'),
        
        # Row 5: Average Loss (Training + Validation)
        (axes[4, 0], 'Average Loss Across All Servers vs Elapsed Time (seconds)', 'Elapsed Time (seconds)', 'Average Loss'),
        (axes[4, 1], 'Average Loss Across All Servers vs Iterations', 'Iterations', 'Average Loss'),
        
        # Row 6: Average Accuracy (Training + Validation)
        (axes[5, 0], 'Average Accuracy Across All Servers vs Elapsed Time (seconds)', 'Elapsed Time (seconds)', 'Average Accuracy'),
        (axes[5, 1], 'Average Accuracy Across All Servers vs Iterations', 'Iterations', 'Average Accuracy'),
        
        # Row 7: Validation F1 Score
        (axes[6, 0], 'Validation F1 Score vs Elapsed Time (seconds)', 'Elapsed Time (seconds)', 'Validation F1 Score'),
        (axes[6, 1], 'Validation F1 Score vs Iterations', 'Iterations', 'Validation F1 Score'),
        
        # Row 8: Validation Precision
        (axes[7, 0], 'Validation Precision vs Elapsed Time (seconds)', 'Elapsed Time (seconds)', 'Validation Precision'),
        (axes[7, 1], 'Validation Precision vs Iterations', 'Iterations', 'Validation Precision'),
        
        # Row 9: Validation Recall
        (axes[8, 0], 'Validation Recall vs Elapsed Time (seconds)', 'Elapsed Time (seconds)', 'Validation Recall'),
        (axes[8, 1], 'Validation Recall vs Iterations', 'Iterations', 'Validation Recall')
    ]
    
    # Configure all plots
    for ax, title, xlabel, ylabel in plot_configs:
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3)
    
    # Plot data for each server
    for server_name, df in server_data.items():
        color = colors.get(server_name, 'black')
        
        # Filter out rows with NaN values
        valid_train_data = df.dropna(subset=['train_loss', 'train_acc'])
        valid_val_data = df.dropna(subset=['val_loss', 'val_acc'])
        valid_val_metrics_data = df.dropna(subset=['val_f1', 'val_precision', 'val_recall'])
        
        if len(valid_train_data) > 0:
            # Row 1: Training Loss
            axes[0, 0].plot(valid_train_data['time_elapsed'], valid_train_data['train_loss'], 
                           color=color, linewidth=2, 
                           label=f'{server_name}', alpha=0.8)
            axes[0, 1].plot(valid_train_data['iteration'], valid_train_data['train_loss'], 
                           color=color, linewidth=2, 
                           label=f'{server_name}', alpha=0.8)
            
            # Row 2: Training Accuracy
            axes[1, 0].plot(valid_train_data['time_elapsed'], valid_train_data['train_acc'], 
                           color=color, linewidth=2, 
                           label=f'{server_name}', alpha=0.8)
            axes[1, 1].plot(valid_train_data['iteration'], valid_train_data['train_acc'], 
                           color=color, linewidth=2, 
                           label=f'{server_name}', alpha=0.8)
        
        if len(valid_val_data) > 0:
            # Row 3: Validation Loss
            axes[2, 0].plot(valid_val_data['time_elapsed'], valid_val_data['val_loss'], 
                           color=color, linewidth=2, 
                           label=f'{server_name}', alpha=0.8)
            axes[2, 1].plot(valid_val_data['iteration'], valid_val_data['val_loss'], 
                           color=color, linewidth=2, 
                           label=f'{server_name}', alpha=0.8)
            
            # Row 4: Validation Accuracy
            axes[3, 0].plot(valid_val_data['time_elapsed'], valid_val_data['val_acc'], 
                           color=color, linewidth=2, 
                           label=f'{server_name}', alpha=0.8)
            axes[3, 1].plot(valid_val_data['iteration'], valid_val_data['val_acc'], 
                           color=color, linewidth=2, 
                           label=f'{server_name}', alpha=0.8)
        
        if len(valid_val_metrics_data) > 0:
            # Row 7: Validation F1 Score
            axes[6, 0].plot(valid_val_metrics_data['time_elapsed'], valid_val_metrics_data['val_f1'], 
                           color=color, linewidth=2, 
                           label=f'{server_name}', alpha=0.8)
            axes[6, 1].plot(valid_val_metrics_data['iteration'], valid_val_metrics_data['val_f1'], 
                           color=color, linewidth=2, 
                           label=f'{server_name}', alpha=0.8)
            
            # Row 8: Validation Precision
            axes[7, 0].plot(valid_val_metrics_data['time_elapsed'], valid_val_metrics_data['val_precision'], 
                           color=color, linewidth=2, 
                           label=f'{server_name}', alpha=0.8)
            axes[7, 1].plot(valid_val_metrics_data['iteration'], valid_val_metrics_data['val_precision'], 
                           color=color, linewidth=2, 
                           label=f'{server_name}', alpha=0.8)
            
            # Row 9: Validation Recall
            axes[8, 0].plot(valid_val_metrics_data['time_elapsed'], valid_val_metrics_data['val_recall'], 
                           color=color, linewidth=2, 
                           label=f'{server_name}', alpha=0.8)
            axes[8, 1].plot(valid_val_metrics_data['iteration'], valid_val_metrics_data['val_recall'], 
                           color=color, linewidth=2, 
                           label=f'{server_name}', alpha=0.8)
    
    # Plot average metrics (Rows 5 & 6) - use consistent filtering
    if not average_data.empty:
        # Filter out rows where we don't have complete average data
        valid_avg_data = average_data.dropna(subset=['avg_train_loss', 'avg_train_acc', 'avg_val_loss', 'avg_val_acc'])
        
        if len(valid_avg_data) > 0:
            # Row 5: Average Loss vs Elapsed Time
            axes[4, 0].plot(valid_avg_data['time_elapsed'], valid_avg_data['avg_train_loss'], 
                           color='purple', linewidth=3, 
                           label='Avg Train Loss', alpha=0.9)
            axes[4, 0].plot(valid_avg_data['time_elapsed'], valid_avg_data['avg_val_loss'], 
                           color='red', linewidth=3, 
                           label='Avg Val Loss', alpha=0.9)
            
            # Row 5: Average Loss vs Iterations
            axes[4, 1].plot(valid_avg_data['iteration'], valid_avg_data['avg_train_loss'], 
                           color='purple', linewidth=3, 
                           label='Avg Train Loss', alpha=0.9)
            axes[4, 1].plot(valid_avg_data['iteration'], valid_avg_data['avg_val_loss'], 
                           color='red', linewidth=3, 
                           label='Avg Val Loss', alpha=0.9)
            
            # Row 6: Average Accuracy vs Elapsed Time
            axes[5, 0].plot(valid_avg_data['time_elapsed'], valid_avg_data['avg_train_acc'], 
                           color='orange', linewidth=3, 
                           label='Avg Train Acc', alpha=0.9)
            axes[5, 0].plot(valid_avg_data['time_elapsed'], valid_avg_data['avg_val_acc'], 
                           color='green', linewidth=3, 
                           label='Avg Val Acc', alpha=0.9)
            
            # Row 6: Average Accuracy vs Iterations
            axes[5, 1].plot(valid_avg_data['iteration'], valid_avg_data['avg_train_acc'], 
                           color='orange', linewidth=3, 
                           label='Avg Train Acc', alpha=0.9)
            axes[5, 1].plot(valid_avg_data['iteration'], valid_avg_data['avg_val_acc'], 
                           color='green', linewidth=3, 
                           label='Avg Val Acc', alpha=0.9)
    
    # Add legends to all plots
    for i in range(9):
        for j in range(2):
            axes[i, j].legend(fontsize=11)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot in unique directory
    plot_filename = os.path.join(plots_dir, "training_validation_loss_accuracy_plots.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved as: {plot_filename}")
    
    # Save individual plots for each metric (including averages) - pass filtered average data
    if not average_data.empty:
        valid_avg_data = average_data.dropna(subset=['avg_train_loss', 'avg_train_acc', 'avg_val_loss', 'avg_val_acc'])
        save_individual_plots(server_data, plots_dir, colors, valid_avg_data)
    else:
        save_individual_plots(server_data, plots_dir, colors, pd.DataFrame())
    
    # Show plot
    plt.show()
    
    # Print summary statistics
    print(f"\n📊 Summary saved to: {plots_dir}/summary_statistics.txt")
    save_summary_statistics(server_data, plots_dir, average_data)
    
    print(f"\n All plots and data saved to: {plots_dir}")

def calculate_average_metrics(server_data):
    """Calculate average metrics across all servers for each iteration"""
    
    # Get all unique iterations across all servers
    all_iterations = set()
    for df in server_data.values():
        all_iterations.update(df['iteration'].tolist())
    
    all_iterations = sorted(list(all_iterations))
    
    # Calculate averages for each iteration
    average_metrics = []
    
    for iteration in all_iterations:
        iteration_data = {
            'iteration': iteration,
            'time_elapsed': 0.0,
            'avg_train_loss': 0.0,
            'avg_train_acc': 0.0,
            'avg_val_loss': 0.0,
            'avg_val_acc': 0.0,
            'avg_val_f1': 0.0,
            'avg_val_precision': 0.0,
            'avg_val_recall': 0.0,
            'server_count': 0
        }
        
        valid_servers = 0
        total_time = 0.0
        total_train_loss = 0.0
        total_train_acc = 0.0
        total_val_loss = 0.0
        total_val_acc = 0.0
        total_val_f1 = 0.0
        total_val_precision = 0.0
        total_val_recall = 0.0
        
        # Track which metrics we actually have data for
        has_train_loss = False
        has_train_acc = False
        has_val_loss = False
        has_val_acc = False
        has_val_f1 = False
        has_val_precision = False
        has_val_recall = False
        
        for server_name, df in server_data.items():
            server_iteration = df[df['iteration'] == iteration]
            
            if not server_iteration.empty:
                valid_servers += 1
                
                # Get metrics for this iteration
                row = server_iteration.iloc[0]
                
                if pd.notna(row['time_elapsed']):
                    total_time += row['time_elapsed']
                
                if pd.notna(row['train_loss']):
                    total_train_loss += row['train_loss']
                    has_train_loss = True
                
                if pd.notna(row['train_acc']):
                    total_train_acc += row['train_acc']
                    has_train_acc = True
                
                if pd.notna(row['val_loss']):
                    total_val_loss += row['val_loss']
                    has_val_loss = True
                
                if pd.notna(row['val_acc']):
                    total_val_acc += row['val_acc']
                    has_val_acc = True
                
                if pd.notna(row['val_f1']):
                    total_val_f1 += row['val_f1']
                    has_val_f1 = True
                
                if pd.notna(row['val_precision']):
                    total_val_precision += row['val_precision']
                    has_val_precision = True
                
                if pd.notna(row['val_recall']):
                    total_val_recall += row['val_recall']
                    has_val_recall = True
        
        # Calculate averages only if we have valid data
        if valid_servers > 0:
            iteration_data['time_elapsed'] = total_time / valid_servers
            iteration_data['server_count'] = valid_servers
            
            # Calculate averages only if we actually have data for that metric
            if has_train_loss:
                iteration_data['avg_train_loss'] = total_train_loss / valid_servers
            else:
                iteration_data['avg_train_loss'] = np.nan
                
            if has_train_acc:
                iteration_data['avg_train_acc'] = total_train_acc / valid_servers
            else:
                iteration_data['avg_train_acc'] = np.nan
                
            if has_val_loss:
                iteration_data['avg_val_loss'] = total_val_loss / valid_servers
            else:
                iteration_data['avg_val_loss'] = np.nan
                
            if has_val_acc:
                iteration_data['avg_val_acc'] = total_val_acc / valid_servers
            else:
                iteration_data['avg_val_acc'] = np.nan
                
            if has_val_f1:
                iteration_data['avg_val_f1'] = total_val_f1 / valid_servers
            else:
                iteration_data['avg_val_f1'] = np.nan
                
            if has_val_precision:
                iteration_data['avg_val_precision'] = total_val_precision / valid_servers
            else:
                iteration_data['avg_val_precision'] = np.nan
                
            if has_val_recall:
                iteration_data['avg_val_recall'] = total_val_recall / valid_servers
            else:
                iteration_data['avg_val_recall'] = np.nan
            
            average_metrics.append(iteration_data)
    
    return pd.DataFrame(average_metrics)

def save_individual_plots(server_data, plots_dir, colors, filtered_average_data):
    """Save individual plots for each metric (including averages) - now uses pre-filtered average data"""
    
    metrics = ['train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_f1', 'val_precision', 'val_recall']
    metric_names = ['Training Loss', 'Training Accuracy', 'Validation Loss', 'Validation Accuracy', 
                   'Validation F1 Score', 'Validation Precision', 'Validation Recall']
    
    for metric, metric_name in zip(metrics, metric_names):
        # Create figure for this metric
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: vs Elapsed Time
        ax1.set_title(f'{metric_name} vs Elapsed Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Elapsed Time (seconds)', fontsize=12)
        ax1.set_ylabel(metric_name, fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: vs Iterations
        ax2.set_title(f'{metric_name} vs Iterations', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Iterations', fontsize=12)
        ax2.set_ylabel(metric_name, fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Plot data for each server
        for server_name, df in server_data.items():
            color = colors.get(server_name, 'black')
            valid_data = df.dropna(subset=[metric])
            
            if len(valid_data) > 0:
                ax1.plot(valid_data['time_elapsed'], valid_data[metric], 
                        color=color, linewidth=2, 
                        label=f'{server_name}', alpha=0.8)
                ax2.plot(valid_data['iteration'], valid_data[metric], 
                        color=color, linewidth=2, 
                        label=f'{server_name}', alpha=0.8)
        
        # Plot average metrics using the pre-filtered data
        if not filtered_average_data.empty:
            avg_metric = f'avg_{metric}'
            if avg_metric in filtered_average_data.columns:
                # No need to filter again - use the pre-filtered data
                ax1.plot(filtered_average_data['time_elapsed'], filtered_average_data[avg_metric], 
                        color='purple', linewidth=3, 
                        label='Average', alpha=0.9)
                ax2.plot(filtered_average_data['iteration'], filtered_average_data[avg_metric], 
                        color='purple', linewidth=3, 
                        label='Average', alpha=0.9)
        
        # Add legends
        ax1.legend(fontsize=11)
        ax2.legend(fontsize=11)
        
        # Save individual plot
        plt.tight_layout()
        individual_filename = os.path.join(plots_dir, f"{metric.replace('_', '_')}_plots.png")
        plt.savefig(individual_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  📈 {metric_name} plots saved")
    
    # Create average metrics overview plot using the pre-filtered data
    if not filtered_average_data.empty:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: All average metrics vs Elapsed Time
        ax1.set_title('All Average Metrics vs Elapsed Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Elapsed Time (seconds)', fontsize=12)
        ax1.set_ylabel('Metrics', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: All average metrics vs Iterations
        ax2.set_title('All Average Metrics vs Iterations', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Iterations', fontsize=12)
        ax2.set_ylabel('Metrics', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Use the pre-filtered data directly
        ax1.plot(filtered_average_data['time_elapsed'], filtered_average_data['avg_train_loss'], 
                color='purple', linewidth=3, 
                label='Avg Train Loss', alpha=0.9)
        ax1.plot(filtered_average_data['time_elapsed'], filtered_average_data['avg_train_acc'], 
                color='orange', linewidth=3, 
                label='Avg Train Acc', alpha=0.9)
        ax1.plot(filtered_average_data['time_elapsed'], filtered_average_data['avg_val_loss'], 
                color='red', linewidth=3, 
                label='Avg Val Loss', alpha=0.9)
        ax1.plot(filtered_average_data['time_elapsed'], filtered_average_data['avg_val_acc'], 
                color='green', linewidth=3, 
                label='Avg Val Acc', alpha=0.9)
        
        ax2.plot(filtered_average_data['iteration'], filtered_average_data['avg_train_loss'], 
                color='purple', linewidth=3, 
                label='Avg Train Loss', alpha=0.9)
        ax2.plot(filtered_average_data['iteration'], filtered_average_data['avg_train_acc'], 
                color='orange', linewidth=3, 
                label='Avg Train Acc', alpha=0.9)
        ax2.plot(filtered_average_data['iteration'], filtered_average_data['avg_val_loss'], 
                color='red', linewidth=3, 
                label='Avg Val Loss', alpha=0.9)
        ax2.plot(filtered_average_data['iteration'], filtered_average_data['avg_val_acc'], 
                color='green', linewidth=3, 
                label='Avg Val Acc', alpha=0.9)
        
        # Add legends
        ax1.legend(fontsize=11)
        ax2.legend(fontsize=11)
        
        # Save average overview plot
        plt.tight_layout()
        avg_overview_filename = os.path.join(plots_dir, "average_metrics_overview.png")
        plt.savefig(avg_overview_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  📊 Average metrics overview plot saved")

def save_summary_statistics(server_data, plots_dir, average_data):
    """Save summary statistics to text file (including averages)"""
    
    with open(os.path.join(plots_dir, 'summary_statistics.txt'), 'w') as f:
        f.write("Training Summary Statistics\n")
        f.write("=" * 50 + "\n\n")
        
        # Individual server statistics
        for server_name, df in server_data.items():
            valid_train_data = df.dropna(subset=['train_loss', 'train_acc'])
            valid_val_data = df.dropna(subset=['val_loss', 'val_acc'])
            valid_val_metrics_data = df.dropna(subset=['val_f1', 'val_precision', 'val_recall'])
            
            f.write(f"{server_name}:\n")
            f.write(f"  Total iterations: {len(df)}\n")
            f.write(f"  Training metrics points: {len(valid_train_data)}\n")
            f.write(f"  Validation metrics points: {len(valid_val_data)}\n")
            f.write(f"  Validation F1/Precision/Recall points: {len(valid_val_metrics_data)}\n")
            f.write(f"  Elapsed time range: {df['time_elapsed'].min():.1f}s to {df['time_elapsed'].max():.1f}s\n")
            if 'time_elapsed' in df.columns and not df['time_elapsed'].dropna().empty:
                avg_elapsed_time = df['time_elapsed'].dropna().mean()
                f.write(f"  Average elapsed time: {avg_elapsed_time:.1f}s\n")
            
            if len(valid_train_data) > 0:
                f.write(f"  Training Loss - Min: {valid_train_data['train_loss'].min():.4f}, Max: {valid_train_data['train_loss'].max():.4f}, Final: {valid_train_data['train_loss'].iloc[-1]:.4f}\n")
                f.write(f"  Training Acc - Min: {valid_train_data['train_acc'].min():.4f}, Max: {valid_train_data['train_acc'].max():.4f}, Final: {valid_train_data['train_acc'].iloc[-1]:.4f}\n")
            
            if len(valid_val_data) > 0:
                f.write(f"  Validation Loss - Min: {valid_val_data['val_loss'].min():.4f}, Max: {valid_val_data['val_loss'].max():.4f}, Final: {valid_val_data['val_loss'].iloc[-1]:.4f}\n")
                f.write(f"  Validation Acc - Min: {valid_val_data['val_acc'].min():.4f}, Max: {valid_val_data['val_acc'].max():.4f}, Final: {valid_val_data['val_acc'].iloc[-1]:.4f}\n")
            
            if len(valid_val_metrics_data) > 0:
                f.write(f"  Validation F1 - Min: {valid_val_metrics_data['val_f1'].min():.4f}, Max: {valid_val_metrics_data['val_f1'].max():.4f}, Final: {valid_val_metrics_data['val_f1'].iloc[-1]:.4f}\n")
                f.write(f"  Validation Precision - Min: {valid_val_metrics_data['val_precision'].min():.4f}, Max: {valid_val_metrics_data['val_precision'].max():.4f}, Final: {valid_val_metrics_data['val_precision'].iloc[-1]:.4f}\n")
                f.write(f"  Validation Recall - Min: {valid_val_metrics_data['val_recall'].min():.4f}, Max: {valid_val_metrics_data['val_recall'].max():.4f}, Final: {valid_val_metrics_data['val_recall'].iloc[-1]:.4f}\n")
            
            f.write("\n")
        
        # Average statistics across all servers
        if not average_data.empty:
            f.write("Average Metrics Across All Servers:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total iterations with averages: {len(average_data)}\n")
            f.write(f"Average server count per iteration: {average_data['server_count'].mean():.1f}\n")
            if 'time_elapsed' in average_data.columns and not average_data['time_elapsed'].dropna().empty:
                avg_elapsed_time_overall = average_data['time_elapsed'].dropna().mean()
                f.write(f"Average elapsed time across servers: {avg_elapsed_time_overall:.1f}s\n")
            
            # Use the same filtering logic as the plots
            valid_avg_data = average_data.dropna(subset=['avg_train_loss', 'avg_train_acc', 'avg_val_loss', 'avg_val_acc'])
            f.write(f"Total iterations with complete average data: {len(valid_avg_data)}\n")
            
            if len(valid_avg_data) > 0:
                f.write(f"Training Loss - Min: {valid_avg_data['avg_train_loss'].min():.4f}, Max: {valid_avg_data['avg_train_loss'].max():.4f}, Final: {valid_avg_data['avg_train_loss'].iloc[-1]:.4f}\n")
                f.write(f"Training Acc - Min: {valid_avg_data['avg_train_acc'].min():.4f}, Max: {valid_avg_data['avg_train_acc'].max():.4f}, Final: {valid_avg_data['avg_train_acc'].iloc[-1]:.4f}\n")
                f.write(f"Validation Loss - Min: {valid_avg_data['avg_val_loss'].min():.4f}, Max: {valid_avg_data['avg_val_loss'].max():.4f}, Final: {valid_avg_data['avg_val_loss'].iloc[-1]:.4f}\n")
                f.write(f"Validation Acc - Min: {valid_avg_data['avg_val_acc'].min():.4f}, Max: {valid_avg_data['avg_val_acc'].max():.4f}, Final: {valid_avg_data['avg_val_acc'].iloc[-1]:.4f}\n")

# Usage
if __name__ == "__main__":
    # Change this to your logs directory path
    logs_directory = "/home/abhattar/serverlogs/ember_cnn_async_1"
    
    # Set max_iteration to limit plotting (None = plot all iterations)
    max_iteration = 2000  # Change this to your desired iteration limit
    
    if os.path.exists(logs_directory):
        load_and_plot_training_logs(logs_directory, max_iteration)
    else:
        print(f"❌ Directory not found: {logs_directory}")
        print("Please update the logs_directory variable to the correct path.")