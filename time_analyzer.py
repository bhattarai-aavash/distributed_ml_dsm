import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import glob

def load_training_logs(logs_dir):
    """
    Load training logs from all servers and calculate time per epoch statistics
    
    Args:
        logs_dir: Directory containing training logs
        
    Returns:
        dict: Dictionary with server data and time statistics
    """
    print(f"🔄 Loading training logs from: {logs_dir}")
    
    server_data = {}
    time_stats = {}
    
    # Find all server directories
    for server_dir in os.listdir(logs_dir):
        server_path = os.path.join(logs_dir, server_dir)
        
        if os.path.isdir(server_path):
            # Find CSV files in server directory
            csv_files = [f for f in os.listdir(server_path) if f.endswith('.csv')]
            
            if csv_files:
                csv_file = os.path.join(server_path, csv_files[0])
                print(f"📊 Loading data from {server_dir}: {csv_file}")
                
                try:
                    # Load CSV data
                    df = pd.read_csv(csv_file)
                    
                    # Convert timestamp to datetime
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    
                    # Store data
                    server_data[server_dir] = df
                    
                    # Calculate time per epoch statistics
                    time_stats[server_dir] = calculate_time_per_epoch(df)
                    
                    print(f"  ✅ Loaded {len(df)} rows")
                    print(f"  📈 Iterations: {df['iteration'].min()} to {df['iteration'].max()}")
                    print(f"  ⏱️  Time range: {df['time_elapsed'].min():.1f}s to {df['time_elapsed'].max():.1f}s")
                    
                except Exception as e:
                    print(f"  ❌ Error loading {csv_file}: {e}")
    
    return server_data, time_stats

def calculate_time_per_epoch(df):
    """
    Calculate time per epoch statistics from training data
    
    Args:
        df: DataFrame with training data
        
    Returns:
        dict: Time statistics
    """
    if df.empty:
        return {}
    
    # Calculate time per iteration
    df_sorted = df.sort_values('iteration')
    df_sorted['time_per_iteration'] = df_sorted['time_elapsed'].diff()
    
    # Remove first row (NaN) and negative values (server restarts)
    df_sorted = df_sorted[df_sorted['time_per_iteration'] > 0]
    
    if df_sorted.empty:
        return {}
    
    # Calculate epoch-based statistics
    epochs = df_sorted['epoch'].unique()
    epoch_times = []
    
    for epoch in epochs:
        epoch_data = df_sorted[df_sorted['epoch'] == epoch]
        if len(epoch_data) > 1:
            epoch_duration = epoch_data['time_elapsed'].max() - epoch_data['time_elapsed'].min()
            epoch_times.append(epoch_duration)
    
    # Calculate statistics
    stats = {
        'total_iterations': len(df),
        'total_epochs': len(epochs),
        'total_time': df['time_elapsed'].max() - df['time_elapsed'].min(),
        'avg_time_per_iteration': df_sorted['time_per_iteration'].mean(),
        'std_time_per_iteration': df_sorted['time_per_iteration'].std(),
        'min_time_per_iteration': df_sorted['time_per_iteration'].min(),
        'max_time_per_iteration': df_sorted['time_per_iteration'].max(),
        'iterations_per_second': 1.0 / df_sorted['time_per_iteration'].mean() if df_sorted['time_per_iteration'].mean() > 0 else 0,
        'epoch_times': epoch_times,
        'avg_time_per_epoch': np.mean(epoch_times) if epoch_times else 0,
        'std_time_per_epoch': np.std(epoch_times) if epoch_times else 0,
        'min_time_per_epoch': np.min(epoch_times) if epoch_times else 0,
        'max_time_per_epoch': np.max(epoch_times) if epoch_times else 0
    }
    
    return stats

def plot_time_statistics(server_data, time_stats, output_dir):
    """
    Create plots for time per epoch statistics
    
    Args:
        server_data: Dictionary with server data
        time_stats: Dictionary with time statistics
        output_dir: Directory to save plots
    """
    print("\n📊 Creating time statistics plots...")
    
    if not time_stats:
        print("⚠️ No time statistics available")
        return
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Training Time Statistics Analysis', fontsize=16, fontweight='bold')
    
    # Extract data for plotting
    servers = list(time_stats.keys())
    
    # Plot 1: Time per iteration
    ax1 = axes[0, 0]
    avg_times = [time_stats[server]['avg_time_per_iteration'] for server in servers]
    std_times = [time_stats[server]['std_time_per_iteration'] for server in servers]
    
    bars1 = ax1.bar(servers, avg_times, yerr=std_times, capsize=5, alpha=0.7)
    ax1.set_title('Average Time per Iteration', fontweight='bold')
    ax1.set_ylabel('Time (seconds)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, (bar, avg, std) in enumerate(zip(bars1, avg_times, std_times)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                f'{avg:.3f}±{std:.3f}s', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Iterations per second
    ax2 = axes[0, 1]
    ips = [time_stats[server]['iterations_per_second'] for server in servers]
    
    bars2 = ax2.bar(servers, ips, alpha=0.7, color='green')
    ax2.set_title('Training Speed (Iterations per Second)', fontweight='bold')
    ax2.set_ylabel('Iterations/Second')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars2, ips):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Time per epoch
    ax3 = axes[0, 2]
    epoch_times = [time_stats[server]['avg_time_per_epoch'] for server in servers]
    epoch_stds = [time_stats[server]['std_time_per_epoch'] for server in servers]
    
    bars3 = ax3.bar(servers, epoch_times, yerr=epoch_stds, capsize=5, alpha=0.7, color='orange')
    ax3.set_title('Average Time per Epoch', fontweight='bold')
    ax3.set_ylabel('Time (seconds)')
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, (bar, avg, std) in enumerate(zip(bars3, epoch_times, epoch_stds)):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.5,
                f'{avg:.1f}±{std:.1f}s', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Total training time
    ax4 = axes[1, 0]
    total_times = [time_stats[server]['total_time'] for server in servers]
    
    bars4 = ax4.bar(servers, total_times, alpha=0.7, color='red')
    ax4.set_title('Total Training Time', fontweight='bold')
    ax4.set_ylabel('Time (seconds)')
    ax4.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars4, total_times):
        hours = value / 3600
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(total_times)*0.01,
                f'{hours:.1f}h', ha='center', va='bottom', fontsize=9)
    
    # Plot 5: Total iterations
    ax5 = axes[1, 1]
    total_iters = [time_stats[server]['total_iterations'] for server in servers]
    
    bars5 = ax5.bar(servers, total_iters, alpha=0.7, color='purple')
    ax5.set_title('Total Iterations Completed', fontweight='bold')
    ax5.set_ylabel('Iterations')
    ax5.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars5, total_iters):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(total_iters)*0.01,
                f'{value}', ha='center', va='bottom', fontsize=9)
    
    # Plot 6: Time per iteration over time (for first server as example)
    ax6 = axes[1, 2]
    if servers:
        first_server = servers[0]
        df = server_data[first_server]
        df_sorted = df.sort_values('iteration')
        df_sorted['time_per_iteration'] = df_sorted['time_elapsed'].diff()
        df_sorted = df_sorted[df_sorted['time_per_iteration'] > 0]
        
        if not df_sorted.empty:
            # Sample every 50th point to avoid overcrowding
            sample_data = df_sorted.iloc[::50]
            ax6.plot(sample_data['iteration'], sample_data['time_per_iteration'], 
                    'b-', alpha=0.7, linewidth=1)
            ax6.set_title(f'Time per Iteration Over Time\n({first_server})', fontweight='bold')
            ax6.set_xlabel('Iteration')
            ax6.set_ylabel('Time per Iteration (seconds)')
            ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    filename = os.path.join(output_dir, 'time_statistics_analysis.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved time statistics plot: {filename}")

def plot_time_comparison(server_data, time_stats, output_dir):
    """
    Create comparison plots for different training modes
    
    Args:
        server_data: Dictionary with server data
        time_stats: Dictionary with time statistics
        output_dir: Directory to save plots
    """
    print("\n📊 Creating time comparison plots...")
    
    if not time_stats:
        print("⚠️ No time statistics available")
        return
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Training Time Comparison: Different Modes', fontsize=16, fontweight='bold')
    
    # Extract data for plotting
    servers = list(time_stats.keys())
    
    # Plot 1: Training speed comparison
    ax1.set_title('Training Speed Comparison', fontweight='bold')
    ax1.set_ylabel('Iterations per Second')
    ax1.grid(True, alpha=0.3)
    
    ips = [time_stats[server]['iterations_per_second'] for server in servers]
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    bars1 = ax1.bar(servers, ips, color=colors[:len(servers)], alpha=0.7)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, value in zip(bars1, ips):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.2f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 2: Efficiency comparison (iterations per hour)
    ax2.set_title('Training Efficiency Comparison', fontweight='bold')
    ax2.set_ylabel('Iterations per Hour')
    ax2.grid(True, alpha=0.3)
    
    iph = [time_stats[server]['iterations_per_second'] * 3600 for server in servers]
    
    bars2 = ax2.bar(servers, iph, color=colors[:len(servers)], alpha=0.7)
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, value in zip(bars2, iph):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(iph)*0.01,
                f'{value:.0f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    filename = os.path.join(output_dir, 'time_comparison_analysis.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved time comparison plot: {filename}")

def generate_time_summary(time_stats, output_dir):
    """
    Generate detailed time statistics summary
    
    Args:
        time_stats: Dictionary with time statistics
        output_dir: Directory to save summary
    """
    print("\n📊 Generating time statistics summary...")
    
    summary_file = os.path.join(output_dir, 'time_statistics_summary.txt')
    
    with open(summary_file, 'w') as f:
        f.write("Training Time Statistics Summary\n")
        f.write("=" * 50 + "\n\n")
        
        for server, stats in time_stats.items():
            f.write(f"{server.upper()}:\n")
            f.write("-" * 20 + "\n")
            f.write(f"  Total Iterations: {stats['total_iterations']}\n")
            f.write(f"  Total Epochs: {stats['total_epochs']}\n")
            f.write(f"  Total Training Time: {stats['total_time']:.1f} seconds ({stats['total_time']/3600:.2f} hours)\n")
            f.write(f"  Average Time per Iteration: {stats['avg_time_per_iteration']:.3f} ± {stats['std_time_per_iteration']:.3f} seconds\n")
            f.write(f"  Min Time per Iteration: {stats['min_time_per_iteration']:.3f} seconds\n")
            f.write(f"  Max Time per Iteration: {stats['max_time_per_iteration']:.3f} seconds\n")
            f.write(f"  Training Speed: {stats['iterations_per_second']:.2f} iterations/second\n")
            f.write(f"  Training Efficiency: {stats['iterations_per_second']*3600:.0f} iterations/hour\n")
            f.write(f"  Average Time per Epoch: {stats['avg_time_per_epoch']:.1f} ± {stats['std_time_per_epoch']:.1f} seconds\n")
            f.write(f"  Min Time per Epoch: {stats['min_time_per_epoch']:.1f} seconds\n")
            f.write(f"  Max Time per Epoch: {stats['max_time_per_epoch']:.1f} seconds\n")
            f.write("\n")
        
        # Overall comparison
        f.write("OVERALL COMPARISON:\n")
        f.write("-" * 20 + "\n")
        
        if time_stats:
            fastest_server = max(time_stats.keys(), key=lambda x: time_stats[x]['iterations_per_second'])
            slowest_server = min(time_stats.keys(), key=lambda x: time_stats[x]['iterations_per_second'])
            
            f.write(f"Fastest Training: {fastest_server} ({time_stats[fastest_server]['iterations_per_second']:.2f} iter/s)\n")
            f.write(f"Slowest Training: {slowest_server} ({time_stats[slowest_server]['iterations_per_second']:.2f} iter/s)\n")
            
            speed_ratio = time_stats[fastest_server]['iterations_per_second'] / time_stats[slowest_server]['iterations_per_second']
            f.write(f"Speed Difference: {speed_ratio:.2f}x faster\n")
    
    print(f"✅ Time statistics summary saved: {summary_file}")

def main():
    """
    Main function to analyze training time statistics
    """
    # Set up paths - you can modify these
    logs_directories = [
        "/home/abhattar/ml_cars_main/seq_cnn_log/training_logs_20250901_161520",
        "/home/abhattar/ml_cars_main/mlp_seq_logs/training_logs_20250902_135719",
        # Add more log directories as needed
    ]
    
    output_dir = f"time_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Created output directory: {output_dir}")
    
    all_server_data = {}
    all_time_stats = {}
    
    # Process each logs directory
    for logs_dir in logs_directories:
        if os.path.exists(logs_dir):
            print(f"\n🔄 Processing: {logs_dir}")
            server_data, time_stats = load_training_logs(logs_dir)
            
            # Add prefix to distinguish different experiments
            experiment_name = os.path.basename(logs_dir)
            for server, data in server_data.items():
                all_server_data[f"{experiment_name}_{server}"] = data
            for server, stats in time_stats.items():
                all_time_stats[f"{experiment_name}_{server}"] = stats
        else:
            print(f"⚠️ Directory not found: {logs_dir}")
    
    if not all_time_stats:
        print("❌ No training data found!")
        return
    
    # Generate all plots and summaries
    plot_time_statistics(all_server_data, all_time_stats, output_dir)
    plot_time_comparison(all_server_data, all_time_stats, output_dir)
    generate_time_summary(all_time_stats, output_dir)
    
    print(f"\n🎉 Time analysis complete! All results saved to: {output_dir}")

if __name__ == "__main__":
    main()
