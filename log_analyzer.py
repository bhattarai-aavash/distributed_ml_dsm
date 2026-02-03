import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import argparse

def analyze_single_log(log_file_path, save_plots=True, output_dir=None):
    """
    Analyze a single training log file and generate detailed statistics and plots.
    
    Args:
        log_file_path (str): Path to the CSV log file
        save_plots (bool): Whether to save plots to files
        output_dir (str): Directory to save plots (if None, creates timestamped directory)
    
    Returns:
        dict: Analysis results including statistics and metrics
    """
    
    if not os.path.exists(log_file_path):
        raise FileNotFoundError(f"Log file not found: {log_file_path}")
    
    print(f"🔍 Analyzing log file: {log_file_path}")
    
    # Load the log data
    df = pd.read_csv(log_file_path)
    print(f"📊 Loaded {len(df)} rows of data")
    
    # Basic data info
    print(f"📅 Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"🔄 Iterations: {df['iteration'].min()} to {df['iteration'].max()}")
    print(f"⏱️  Duration: {df['time_elapsed'].max():.1f} seconds ({df['time_elapsed'].max()/60:.1f} minutes)")
    print(f"🖥️  Server: {df['server_id'].iloc[0]}")
    
    # Create output directory
    if output_dir is None:
        server_name = df['server_id'].iloc[0]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"log_analysis_{server_name}_{timestamp}"
    
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Output directory: {output_dir}")
    
    # Calculate statistics
    analysis_results = calculate_log_statistics(df)
    
    # Generate plots
    if save_plots:
        create_analysis_plots(df, output_dir, analysis_results)
    
    # Save analysis report
    if save_plots:
        save_analysis_report(analysis_results, output_dir, log_file_path)
    
    return analysis_results

def calculate_log_statistics(df):
    """Calculate comprehensive statistics from the log data"""
    
    stats = {
        'basic_info': {
            'total_iterations': len(df),
            'server_id': df['server_id'].iloc[0],
            'start_time': df['timestamp'].min(),
            'end_time': df['timestamp'].max(),
            'total_duration_seconds': df['time_elapsed'].max(),
            'total_duration_minutes': df['time_elapsed'].max() / 60,
            'iterations_per_second': len(df) / df['time_elapsed'].max() if df['time_elapsed'].max() > 0 else 0
        }
    }
    
    # Training metrics statistics
    train_data = df.dropna(subset=['train_loss', 'train_acc'])
    if len(train_data) > 0:
        stats['training'] = {
            'data_points': len(train_data),
            'loss': {
                'initial': train_data['train_loss'].iloc[0],
                'final': train_data['train_loss'].iloc[-1],
                'min': train_data['train_loss'].min(),
                'max': train_data['train_loss'].max(),
                'mean': train_data['train_loss'].mean(),
                'std': train_data['train_loss'].std(),
                'improvement': train_data['train_loss'].iloc[0] - train_data['train_loss'].iloc[-1],
                'improvement_percent': ((train_data['train_loss'].iloc[0] - train_data['train_loss'].iloc[-1]) / train_data['train_loss'].iloc[0]) * 100
            },
            'accuracy': {
                'initial': train_data['train_acc'].iloc[0],
                'final': train_data['train_acc'].iloc[-1],
                'min': train_data['train_acc'].min(),
                'max': train_data['train_acc'].max(),
                'mean': train_data['train_acc'].mean(),
                'std': train_data['train_acc'].std(),
                'improvement': train_data['train_acc'].iloc[-1] - train_data['train_acc'].iloc[0],
                'improvement_percent': ((train_data['train_acc'].iloc[-1] - train_data['train_acc'].iloc[0]) / train_data['train_acc'].iloc[0]) * 100
            }
        }
    
    # Validation metrics statistics
    val_data = df.dropna(subset=['val_loss', 'val_acc'])
    if len(val_data) > 0:
        stats['validation'] = {
            'data_points': len(val_data),
            'loss': {
                'initial': val_data['val_loss'].iloc[0],
                'final': val_data['val_loss'].iloc[-1],
                'min': val_data['val_loss'].min(),
                'max': val_data['val_loss'].max(),
                'mean': val_data['val_loss'].mean(),
                'std': val_data['val_loss'].std(),
                'improvement': val_data['val_loss'].iloc[0] - val_data['val_loss'].iloc[-1],
                'improvement_percent': ((val_data['val_loss'].iloc[0] - val_data['val_loss'].iloc[-1]) / val_data['val_loss'].iloc[0]) * 100
            },
            'accuracy': {
                'initial': val_data['val_acc'].iloc[0],
                'final': val_data['val_acc'].iloc[-1],
                'min': val_data['val_acc'].min(),
                'max': val_data['val_acc'].max(),
                'mean': val_data['val_acc'].mean(),
                'std': val_data['val_acc'].std(),
                'improvement': val_data['val_acc'].iloc[-1] - val_data['val_acc'].iloc[0],
                'improvement_percent': ((val_data['val_acc'].iloc[-1] - val_data['val_acc'].iloc[0]) / val_data['val_acc'].iloc[0]) * 100
            }
        }
    
    # Overfitting analysis
    if len(train_data) > 0 and len(val_data) > 0:
        # Find common iterations
        common_iterations = set(train_data['iteration']).intersection(set(val_data['iteration']))
        if common_iterations:
            common_iterations = sorted(list(common_iterations))
            train_common = train_data[train_data['iteration'].isin(common_iterations)].sort_values('iteration')
            val_common = val_data[val_data['iteration'].isin(common_iterations)].sort_values('iteration')
            
            if len(train_common) > 0 and len(val_common) > 0:
                final_gap_loss = val_common['val_loss'].iloc[-1] - train_common['train_loss'].iloc[-1]
                final_gap_acc = train_common['train_acc'].iloc[-1] - val_common['val_acc'].iloc[-1]
                
                stats['overfitting'] = {
                    'final_loss_gap': final_gap_loss,
                    'final_accuracy_gap': final_gap_acc,
                    'is_overfitting_loss': final_gap_loss > 0.1,  # Threshold for overfitting
                    'is_overfitting_acc': final_gap_acc > 0.05   # Threshold for overfitting
                }
    
    # Learning rate analysis
    if 'learning_rate' in df.columns:
        lr_data = df.dropna(subset=['learning_rate'])
        if len(lr_data) > 0:
            stats['learning_rate'] = {
                'initial': lr_data['learning_rate'].iloc[0],
                'final': lr_data['learning_rate'].iloc[-1],
                'min': lr_data['learning_rate'].min(),
                'max': lr_data['learning_rate'].max(),
                'mean': lr_data['learning_rate'].mean(),
                'std': lr_data['learning_rate'].std(),
                'was_scheduled': lr_data['learning_rate'].nunique() > 1
            }
    
    # Performance analysis
    if len(df) > 1:
        time_diffs = df['time_elapsed'].diff().dropna()
        iteration_diffs = df['iteration'].diff().dropna()
        
        stats['performance'] = {
            'avg_time_per_iteration': time_diffs.mean(),
            'std_time_per_iteration': time_diffs.std(),
            'min_time_per_iteration': time_diffs.min(),
            'max_time_per_iteration': time_diffs.max(),
            'iterations_per_second': len(df) / df['time_elapsed'].max() if df['time_elapsed'].max() > 0 else 0
        }
    
    return stats

def create_analysis_plots(df, output_dir, stats):
    """Create comprehensive plots for the log analysis"""
    
    # Create a large figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Plot 1: Training and Validation Loss
    plt.subplot(3, 3, 1)
    train_data = df.dropna(subset=['train_loss'])
    val_data = df.dropna(subset=['val_loss'])
    
    if len(train_data) > 0:
        plt.plot(train_data['iteration'], train_data['train_loss'], 'b-', label='Training Loss', linewidth=2)
    if len(val_data) > 0:
        plt.plot(val_data['iteration'], val_data['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    
    plt.title('Loss vs Iterations', fontsize=14, fontweight='bold')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Training and Validation Accuracy
    plt.subplot(3, 3, 2)
    train_data = df.dropna(subset=['train_acc'])
    val_data = df.dropna(subset=['val_acc'])
    
    if len(train_data) > 0:
        plt.plot(train_data['iteration'], train_data['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    if len(val_data) > 0:
        plt.plot(val_data['iteration'], val_data['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    
    plt.title('Accuracy vs Iterations', fontsize=14, fontweight='bold')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Loss vs Time
    plt.subplot(3, 3, 3)
    train_data = df.dropna(subset=['train_loss'])
    val_data = df.dropna(subset=['val_loss'])
    
    if len(train_data) > 0:
        plt.plot(train_data['time_elapsed'], train_data['train_loss'], 'b-', label='Training Loss', linewidth=2)
    if len(val_data) > 0:
        plt.plot(val_data['time_elapsed'], val_data['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    
    plt.title('Loss vs Time', fontsize=14, fontweight='bold')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Accuracy vs Time
    plt.subplot(3, 3, 4)
    train_data = df.dropna(subset=['train_acc'])
    val_data = df.dropna(subset=['val_acc'])
    
    if len(train_data) > 0:
        plt.plot(train_data['time_elapsed'], train_data['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    if len(val_data) > 0:
        plt.plot(val_data['time_elapsed'], val_data['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    
    plt.title('Accuracy vs Time', fontsize=14, fontweight='bold')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Learning Rate (if available)
    plt.subplot(3, 3, 5)
    if 'learning_rate' in df.columns:
        lr_data = df.dropna(subset=['learning_rate'])
        if len(lr_data) > 0:
            plt.plot(lr_data['iteration'], lr_data['learning_rate'], 'g-', linewidth=2)
            plt.title('Learning Rate vs Iterations', fontsize=14, fontweight='bold')
            plt.xlabel('Iterations')
            plt.ylabel('Learning Rate')
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'No Learning Rate Data', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Learning Rate vs Iterations', fontsize=14, fontweight='bold')
    else:
        plt.text(0.5, 0.5, 'No Learning Rate Data', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Learning Rate vs Iterations', fontsize=14, fontweight='bold')
    
    # Plot 6: Time per Iteration
    plt.subplot(3, 3, 6)
    if len(df) > 1:
        time_diffs = df['time_elapsed'].diff().dropna()
        plt.plot(df['iteration'].iloc[1:], time_diffs, 'purple', linewidth=1, alpha=0.7)
        plt.title('Time per Iteration', fontsize=14, fontweight='bold')
        plt.xlabel('Iterations')
        plt.ylabel('Time per Iteration (seconds)')
        plt.grid(True, alpha=0.3)
    
    # Plot 7: Loss Distribution
    plt.subplot(3, 3, 7)
    train_data = df.dropna(subset=['train_loss'])
    val_data = df.dropna(subset=['val_loss'])
    
    if len(train_data) > 0:
        plt.hist(train_data['train_loss'], bins=30, alpha=0.7, label='Training Loss', color='blue')
    if len(val_data) > 0:
        plt.hist(val_data['val_loss'], bins=30, alpha=0.7, label='Validation Loss', color='red')
    
    plt.title('Loss Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Loss')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 8: Accuracy Distribution
    plt.subplot(3, 3, 8)
    train_data = df.dropna(subset=['train_acc'])
    val_data = df.dropna(subset=['val_acc'])
    
    if len(train_data) > 0:
        plt.hist(train_data['train_acc'], bins=30, alpha=0.7, label='Training Accuracy', color='blue')
    if len(val_data) > 0:
        plt.hist(val_data['val_acc'], bins=30, alpha=0.7, label='Validation Accuracy', color='red')
    
    plt.title('Accuracy Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 9: Summary Statistics
    plt.subplot(3, 3, 9)
    plt.axis('off')
    
    # Create summary text
    summary_text = f"""
    Training Summary:
    • Server: {stats['basic_info']['server_id']}
    • Total Iterations: {stats['basic_info']['total_iterations']}
    • Duration: {stats['basic_info']['total_duration_minutes']:.1f} minutes
    • Speed: {stats['basic_info']['iterations_per_second']:.2f} iter/sec
    
    Training Loss:
    • Initial: {stats['training']['loss']['initial']:.4f}
    • Final: {stats['training']['loss']['final']:.4f}
    • Improvement: {stats['training']['loss']['improvement_percent']:.1f}%
    
    Validation Loss:
    • Initial: {stats['validation']['loss']['initial']:.4f}
    • Final: {stats['validation']['loss']['final']:.4f}
    • Improvement: {stats['validation']['loss']['improvement_percent']:.1f}%
    """
    
    plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save the comprehensive plot
    plot_filename = os.path.join(output_dir, "comprehensive_analysis.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📈 Comprehensive analysis plot saved: {plot_filename}")
    
    # Create individual metric plots
    create_individual_metric_plots(df, output_dir)

def create_individual_metric_plots(df, output_dir):
    """Create individual plots for each metric"""
    
    metrics = [
        ('train_loss', 'Training Loss', 'blue'),
        ('train_acc', 'Training Accuracy', 'blue'),
        ('val_loss', 'Validation Loss', 'red'),
        ('val_acc', 'Validation Accuracy', 'red')
    ]
    
    for metric, title, color in metrics:
        if metric in df.columns:
            data = df.dropna(subset=[metric])
            if len(data) > 0:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Plot vs iterations
                ax1.plot(data['iteration'], data[metric], color=color, linewidth=2)
                ax1.set_title(f'{title} vs Iterations', fontsize=14, fontweight='bold')
                ax1.set_xlabel('Iterations')
                ax1.set_ylabel(title)
                ax1.grid(True, alpha=0.3)
                
                # Plot vs time
                ax2.plot(data['time_elapsed'], data[metric], color=color, linewidth=2)
                ax2.set_title(f'{title} vs Time', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Time (seconds)')
                ax2.set_ylabel(title)
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plot_filename = os.path.join(output_dir, f"{metric}_analysis.png")
                plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"📈 {title} plot saved: {plot_filename}")

def save_analysis_report(stats, output_dir, log_file_path):
    """Save detailed analysis report to text file"""
    
    report_filename = os.path.join(output_dir, "analysis_report.txt")
    
    with open(report_filename, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TRAINING LOG ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Log File: {log_file_path}\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Basic Information
        f.write("BASIC INFORMATION\n")
        f.write("-" * 40 + "\n")
        basic = stats['basic_info']
        f.write(f"Server ID: {basic['server_id']}\n")
        f.write(f"Total Iterations: {basic['total_iterations']}\n")
        f.write(f"Start Time: {basic['start_time']}\n")
        f.write(f"End Time: {basic['end_time']}\n")
        f.write(f"Total Duration: {basic['total_duration_minutes']:.2f} minutes\n")
        f.write(f"Training Speed: {basic['iterations_per_second']:.2f} iterations/second\n\n")
        
        # Training Metrics
        if 'training' in stats:
            f.write("TRAINING METRICS\n")
            f.write("-" * 40 + "\n")
            train = stats['training']
            f.write(f"Data Points: {train['data_points']}\n")
            f.write(f"Loss - Initial: {train['loss']['initial']:.6f}, Final: {train['loss']['final']:.6f}\n")
            f.write(f"Loss - Min: {train['loss']['min']:.6f}, Max: {train['loss']['max']:.6f}\n")
            f.write(f"Loss - Mean: {train['loss']['mean']:.6f}, Std: {train['loss']['std']:.6f}\n")
            f.write(f"Loss Improvement: {train['loss']['improvement']:.6f} ({train['loss']['improvement_percent']:.2f}%)\n")
            f.write(f"Accuracy - Initial: {train['accuracy']['initial']:.4f}, Final: {train['accuracy']['final']:.4f}\n")
            f.write(f"Accuracy - Min: {train['accuracy']['min']:.4f}, Max: {train['accuracy']['max']:.4f}\n")
            f.write(f"Accuracy - Mean: {train['accuracy']['mean']:.4f}, Std: {train['accuracy']['std']:.4f}\n")
            f.write(f"Accuracy Improvement: {train['accuracy']['improvement']:.4f} ({train['accuracy']['improvement_percent']:.2f}%)\n\n")
        
        # Validation Metrics
        if 'validation' in stats:
            f.write("VALIDATION METRICS\n")
            f.write("-" * 40 + "\n")
            val = stats['validation']
            f.write(f"Data Points: {val['data_points']}\n")
            f.write(f"Loss - Initial: {val['loss']['initial']:.6f}, Final: {val['loss']['final']:.6f}\n")
            f.write(f"Loss - Min: {val['loss']['min']:.6f}, Max: {val['loss']['max']:.6f}\n")
            f.write(f"Loss - Mean: {val['loss']['mean']:.6f}, Std: {val['loss']['std']:.6f}\n")
            f.write(f"Loss Improvement: {val['loss']['improvement']:.6f} ({val['loss']['improvement_percent']:.2f}%)\n")
            f.write(f"Accuracy - Initial: {val['accuracy']['initial']:.4f}, Final: {val['accuracy']['final']:.4f}\n")
            f.write(f"Accuracy - Min: {val['accuracy']['min']:.4f}, Max: {val['accuracy']['max']:.4f}\n")
            f.write(f"Accuracy - Mean: {val['accuracy']['mean']:.4f}, Std: {val['accuracy']['std']:.4f}\n")
            f.write(f"Accuracy Improvement: {val['accuracy']['improvement']:.4f} ({val['accuracy']['improvement_percent']:.2f}%)\n\n")
        
        # Overfitting Analysis
        if 'overfitting' in stats:
            f.write("OVERFITTING ANALYSIS\n")
            f.write("-" * 40 + "\n")
            overfit = stats['overfitting']
            f.write(f"Final Loss Gap (Val - Train): {overfit['final_loss_gap']:.6f}\n")
            f.write(f"Final Accuracy Gap (Train - Val): {overfit['final_accuracy_gap']:.4f}\n")
            f.write(f"Overfitting (Loss): {'Yes' if overfit['is_overfitting_loss'] else 'No'}\n")
            f.write(f"Overfitting (Accuracy): {'Yes' if overfit['is_overfitting_acc'] else 'No'}\n\n")
        
        # Learning Rate Analysis
        if 'learning_rate' in stats:
            f.write("LEARNING RATE ANALYSIS\n")
            f.write("-" * 40 + "\n")
            lr = stats['learning_rate']
            f.write(f"Initial LR: {lr['initial']:.6f}\n")
            f.write(f"Final LR: {lr['final']:.6f}\n")
            f.write(f"Min LR: {lr['min']:.6f}, Max LR: {lr['max']:.6f}\n")
            f.write(f"Mean LR: {lr['mean']:.6f}, Std LR: {lr['std']:.6f}\n")
            f.write(f"LR Scheduling: {'Yes' if lr['was_scheduled'] else 'No'}\n\n")
        
        # Performance Analysis
        if 'performance' in stats:
            f.write("PERFORMANCE ANALYSIS\n")
            f.write("-" * 40 + "\n")
            perf = stats['performance']
            f.write(f"Avg Time per Iteration: {perf['avg_time_per_iteration']:.4f} seconds\n")
            f.write(f"Std Time per Iteration: {perf['std_time_per_iteration']:.4f} seconds\n")
            f.write(f"Min Time per Iteration: {perf['min_time_per_iteration']:.4f} seconds\n")
            f.write(f"Max Time per Iteration: {perf['max_time_per_iteration']:.4f} seconds\n")
            f.write(f"Overall Speed: {perf['iterations_per_second']:.2f} iterations/second\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"📄 Analysis report saved: {report_filename}")

def main():
    """Command line interface for log analysis"""
    parser = argparse.ArgumentParser(description='Analyze a single training log file')
    parser.add_argument('log_file', help='Path to the CSV log file to analyze')
    parser.add_argument('--output-dir', '-o', help='Output directory for plots and reports')
    parser.add_argument('--no-plots', action='store_true', help='Skip generating plots')
    
    args = parser.parse_args()
    
    try:
        results = analyze_single_log(
            log_file_path=args.log_file,
            save_plots=not args.no_plots,
            output_dir=args.output_dir
        )
        
        print("\n✅ Analysis completed successfully!")
        print(f"📊 Analyzed {results['basic_info']['total_iterations']} iterations")
        print(f"⏱️  Total duration: {results['basic_info']['total_duration_minutes']:.1f} minutes")
        
        if 'training' in results:
            print(f"📈 Training loss improvement: {results['training']['loss']['improvement_percent']:.1f}%")
        
        if 'validation' in results:
            print(f"📈 Validation loss improvement: {results['validation']['loss']['improvement_percent']:.1f}%")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
