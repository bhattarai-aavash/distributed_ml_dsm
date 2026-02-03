#!/usr/bin/env python3
import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from datetime import datetime


def plot_throughput_and_training(exp_dir: str, output_filename: str | None = None, y_max_throughput: int | None = None, throughput_skip_seconds: int = 0, plot_client: bool = True) -> None:
    """
    Generate two separate figures in a timestamped output folder inside the experiment:
      1) throughput_combined.png  — Throughput over elapsed time (all servers & sources)
      2) metrics_over_time.png    — Train/Val/Test accuracy and loss vs elapsed time

    Looks for throughput data in either:
      - <experiment>_keydb/<subdir>/*_breakdown.csv (original structure, two levels deep)
      - keydb_logs/*_breakdown.csv (alternative structure, one level deep)
    
    Uses average_metrics.csv (+ average_elapsed_time_all_iterations.csv if present).
    
    Args:
        plot_client: If True, include client throughput in plots. If False, only plot server throughput.
    """
    exp_dir = os.path.abspath(exp_dir)
    if not os.path.isdir(exp_dir):
        raise SystemExit(f"Not a directory: {exp_dir}")

    # 1) Load training/validation metrics
    metrics_csv = os.path.join(exp_dir, 'average_metrics.csv')
    metrics_df = None
    if os.path.isfile(metrics_csv):
        try:
            metrics_df = pd.read_csv(metrics_csv)
        except Exception as e:
            print(f"❌ Failed to read {metrics_csv}: {e}")
            metrics_df = None
    else:
        print(f"⚠️ average_metrics.csv not found in {exp_dir}; metrics panel will be blank")

    # 2) Find *_breakdown.csv files
    # Try two patterns:
    #   a) Original: *_keydb/<subdir>/*_breakdown.csv (two levels deep)
    #   b) Alternative: keydb_logs/*_breakdown.csv (one level deep)
    breakdown_paths: List[str] = []
    keydb_root = None
    
    # First, try to find *_keydb directory (original structure)
    for d in os.listdir(exp_dir):
        if d.endswith('_keydb') and os.path.isdir(os.path.join(exp_dir, d)):
            keydb_root = os.path.join(exp_dir, d)
            # Look for files two levels deep: *_keydb/*/*_breakdown.csv
            breakdown_paths = sorted(glob.glob(os.path.join(keydb_root, '*', '*_breakdown.csv')))
            if breakdown_paths:
                break
    
    # If not found, try keydb_logs directory (alternative structure)
    if not breakdown_paths:
        keydb_logs_dir = os.path.join(exp_dir, 'keydb_logs')
        if os.path.isdir(keydb_logs_dir):
            # Look for files directly in keydb_logs: keydb_logs/*_breakdown.csv
            breakdown_paths = sorted(glob.glob(os.path.join(keydb_logs_dir, '*_breakdown.csv')))
            if breakdown_paths:
                print(f"📁 Found breakdown files in keydb_logs/ directory")
    
    if not breakdown_paths:
        print(f"⚠️ No breakdown CSV files found. Looked for:")
        print(f"   - {exp_dir}/*_keydb/*/*_breakdown.csv")
        print(f"   - {exp_dir}/keydb_logs/*_breakdown.csv")

    # Prepare output directory (timestamped) if caller didn't provide an explicit file path
    if output_filename is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = os.path.join(exp_dir, f'combined_plots_{ts}')
        os.makedirs(out_dir, exist_ok=True)
    else:
        # If a single output path was requested, place both plots in its directory
        out_dir = os.path.dirname(output_filename) or '.'
        os.makedirs(out_dir, exist_ok=True)

    # Build single figure with two subplots
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})

    # Throughput subplot
    any_series = False
    for csv_path in breakdown_paths:
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        server = str(df['server'].iloc[0])
        meta_cols = {"experiment", "server", "second", "elapsed_seconds"}
        source_cols = [c for c in df.columns if c not in meta_cols]
        # Filter columns based on plot_client setting
        if plot_client:
            # Include 'client' (prioritized first) and all others except 'total'
            source_cols = (
                ([c for c in source_cols if c == 'client']) +
                [c for c in source_cols if c not in ('client', 'total')]
            )
        else:
            # Exclude both 'client' and 'total', only plot server throughput
            source_cols = [c for c in source_cols if c not in ('client', 'total')]
        df = df.sort_values(['elapsed_seconds']).reset_index(drop=True)
        # Optional downsampling by seconds to reduce clutter
        if throughput_skip_seconds and throughput_skip_seconds > 0:
            # Keep rows where int(elapsed_seconds) modulo skip == 0
            df = df[(df['elapsed_seconds'].astype(float).astype(int) % int(throughput_skip_seconds)) == 0]
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

    # Metrics subplot (prefer built-in elapsed column in metrics; else require mapping file)
    if metrics_df is None or metrics_df.empty:
        ax = ax_bottom
        ax.text(0.5, 0.5, 'No metrics available', ha='center', va='center')
        ax.axis('off')
    else:
        elapsed_csv = os.path.join(exp_dir, 'average_elapsed_time_all_iterations.csv')
        dfm = metrics_df.copy()
        # Prefer a direct elapsed time column in metrics (e.g., time_elapsed)
        direct_time_candidates = ['time_elapsed', 'elapsed_seconds', 'elapsed_time', 'seconds']
        direct_x = next((c for c in direct_time_candidates if c in dfm.columns), None)
        if direct_x is not None:
            # Ensure sorted order
            dfm = dfm.dropna(subset=[direct_x]).sort_values([direct_x]).copy()
            x_col = direct_x
            x_label = 'Elapsed seconds'
            use_mapping = False
        else:
            # Fall back to mapping file
            dfm = dfm.dropna(subset=['iteration']).sort_values(['iteration']).copy()
            use_mapping = True
        # If we need mapping file but it's missing – render message and save
        if use_mapping and not os.path.isfile(elapsed_csv):
            ax = ax_bottom
            ax.text(0.5, 0.5, 'average_elapsed_time_all_iterations.csv not found – cannot plot metrics vs elapsed time',
                    ha='center', va='center')
            ax.axis('off')
            combined_png = os.path.join(out_dir, 'combined_throughput_and_metrics.png')
            plt.tight_layout()
            plt.savefig(combined_png, dpi=200)
            plt.close()
            print(f"✅ Saved combined plot (no metrics): {combined_png}")
            return
        else:
            success = False
            try:
                if use_mapping:
                    el = pd.read_csv(elapsed_csv)
                    el_iter_col = 'iteration' if 'iteration' in el.columns else None
                    time_candidates = ['time_elapsed', 'elapsed_seconds', 'average_elapsed_seconds', 'elapsed_time', 'elapsed', 'seconds', 'time_s']
                    el_time_col = next((c for c in time_candidates if c in el.columns), None)
                    if el_iter_col is None or el_time_col is None:
                        raise ValueError('Required columns not found in elapsed time CSV')
                    el_slim = el[[el_iter_col, el_time_col]].dropna()
                    el_slim.columns = ['iteration', 'elapsed_seconds']
                    dfm = dfm.merge(el_slim, on='iteration', how='left')
                    if not dfm['elapsed_seconds'].notna().any():
                        raise ValueError('No valid elapsed time values after merge')
                    x_col = 'elapsed_seconds'
                    x_label = 'Elapsed seconds'
                # If direct_x was present, success already true with x_col set
                success = True
            except Exception as e:
                ax = ax_bottom
                ax.text(0.5, 0.5, f'Failed to use elapsed time mapping: {e}', ha='center', va='center')
                ax.axis('off')
                combined_png = os.path.join(out_dir, 'combined_throughput_and_metrics.png')
                plt.tight_layout()
                plt.savefig(combined_png, dpi=200)
                plt.close()
                print(f"✅ Saved combined plot (metrics mapping failed): {combined_png}")
                return

            if success:
                ax_acc = ax_bottom
                ax_loss = ax_acc.twinx()

                if 'avg_train_acc' in dfm.columns:
                    ax_acc.plot(dfm[x_col], dfm['avg_train_acc'], label='Train Acc', color='tab:blue', linewidth=2)
                if 'avg_val_acc' in dfm.columns:
                    ax_acc.plot(dfm[x_col], dfm['avg_val_acc'], label='Val Acc', color='tab:cyan', linewidth=2, linestyle='--')
                if 'avg_test_acc' in dfm.columns:
                    ax_acc.plot(dfm[x_col], dfm['avg_test_acc'], label='Test Acc', color='tab:purple', linewidth=2, linestyle=':')
                ax_acc.set_ylabel('Accuracy', color='tab:blue')
                ax_acc.tick_params(axis='y', labelcolor='tab:blue')

                if 'avg_train_loss' in dfm.columns:
                    ax_loss.plot(dfm[x_col], dfm['avg_train_loss'], label='Train Loss', color='tab:red', linewidth=2)
                if 'avg_val_loss' in dfm.columns:
                    ax_loss.plot(dfm[x_col], dfm['avg_val_loss'], label='Val Loss', color='tab:orange', linewidth=2, linestyle='--')
                if 'avg_test_loss' in dfm.columns:
                    ax_loss.plot(dfm[x_col], dfm['avg_test_loss'], label='Test Loss', color='tab:pink', linewidth=2, linestyle=':')
                ax_loss.set_ylabel('Loss', color='tab:red')
                ax_loss.tick_params(axis='y', labelcolor='tab:red')

                ax_acc.set_xlabel(x_label)
                ax_acc.grid(True, alpha=0.3)
                ax_bottom.set_title('Training/Validation/Test Metrics vs ' + x_label)

                lines, labels = [], []
                for ax in (ax_acc, ax_loss):
                    h, l = ax.get_legend_handles_labels()
                    lines += h; labels += l
                ax_acc.legend(lines, labels, loc='upper right')

        ax_acc = ax_bottom
        ax_loss = ax_acc.twinx()

        if 'avg_train_acc' in dfm.columns:
            ax_acc.plot(dfm[x_col], dfm['avg_train_acc'], label='Train Acc', color='tab:blue', linewidth=2)
        if 'avg_val_acc' in dfm.columns:
            ax_acc.plot(dfm[x_col], dfm['avg_val_acc'], label='Val Acc', color='tab:cyan', linewidth=2, linestyle='--')
        if 'avg_test_acc' in dfm.columns:
            ax_acc.plot(dfm[x_col], dfm['avg_test_acc'], label='Test Acc', color='tab:purple', linewidth=2, linestyle=':')
        ax_acc.set_ylabel('Accuracy', color='tab:blue')
        ax_acc.tick_params(axis='y', labelcolor='tab:blue')

        if 'avg_train_loss' in dfm.columns:
            ax_loss.plot(dfm[x_col], dfm['avg_train_loss'], label='Train Loss', color='tab:red', linewidth=2)
        if 'avg_val_loss' in dfm.columns:
            ax_loss.plot(dfm[x_col], dfm['avg_val_loss'], label='Val Loss', color='tab:orange', linewidth=2, linestyle='--')
        if 'avg_test_loss' in dfm.columns:
            ax_loss.plot(dfm[x_col], dfm['avg_test_loss'], label='Test Loss', color='tab:pink', linewidth=2, linestyle=':')
        ax_loss.set_ylabel('Loss', color='tab:red')
        ax_loss.tick_params(axis='y', labelcolor='tab:red')

        ax_acc.set_xlabel(x_label)
        ax_acc.grid(True, alpha=0.3)
        ax_bottom.set_title('Training/Validation/Test Metrics vs ' + x_label)

        lines, labels = [], []
        for ax in (ax_acc, ax_loss):
            h, l = ax.get_legend_handles_labels()
            lines += h; labels += l
        ax_acc.legend(lines, labels, loc='upper right')

    # Save single combined image
    combined_png = os.path.join(out_dir, 'combined_throughput_and_metrics.png')
    plt.tight_layout()
    plt.savefig(combined_png, dpi=200)
    plt.close()
    print(f"✅ Saved combined plot: {combined_png}")


def plot_individual_server_throughput(exp_dir: str, output_dir: str | None = None, y_max_throughput: int | None = None, throughput_skip_seconds: int = 0, plot_client: bool = True) -> None:
    """
    Create separate throughput plots for each server.
    
    Args:
        exp_dir: Experiment directory
        output_dir: Output directory (if None, creates timestamped directory)
        y_max_throughput: Optional y-axis maximum
        throughput_skip_seconds: Optional downsampling (0 = no downsampling)
        plot_client: If True, include client throughput in plots. If False, only plot server throughput.
    """
    exp_dir = os.path.abspath(exp_dir)
    if not os.path.isdir(exp_dir):
        raise SystemExit(f"Not a directory: {exp_dir}")
    
    # Find breakdown CSV files (same logic as main function)
    breakdown_paths: List[str] = []
    keydb_root = None
    
    # First, try to find *_keydb directory (original structure)
    for d in os.listdir(exp_dir):
        if d.endswith('_keydb') and os.path.isdir(os.path.join(exp_dir, d)):
            keydb_root = os.path.join(exp_dir, d)
            breakdown_paths = sorted(glob.glob(os.path.join(keydb_root, '*', '*_breakdown.csv')))
            if breakdown_paths:
                break
    
    # If not found, try keydb_logs directory (alternative structure)
    if not breakdown_paths:
        keydb_logs_dir = os.path.join(exp_dir, 'keydb_logs')
        if os.path.isdir(keydb_logs_dir):
            breakdown_paths = sorted(glob.glob(os.path.join(keydb_logs_dir, '*_breakdown.csv')))
    
    if not breakdown_paths:
        print(f"⚠️ No breakdown CSV files found")
        return
    
    # Prepare output directory
    if output_dir is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(exp_dir, f'individual_server_plots_{ts}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a separate plot for each server
    for csv_path in breakdown_paths:
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        
        server = str(df['server'].iloc[0])
        experiment = str(df['experiment'].iloc[0]) if 'experiment' in df.columns else 'experiment'
        
        # Extract source columns
        meta_cols = {"experiment", "server", "second", "elapsed_seconds"}
        source_cols = [c for c in df.columns if c not in meta_cols]
        # Filter columns based on plot_client setting
        if plot_client:
            # Include 'client' (prioritized first) and all others except 'total'
            source_cols = (
                ([c for c in source_cols if c == 'client']) +
                [c for c in source_cols if c not in ('client', 'total')]
            )
        else:
            # Exclude both 'client' and 'total', only plot server throughput
            source_cols = [c for c in source_cols if c not in ('client', 'total')]
        
        df = df.sort_values(['elapsed_seconds']).reset_index(drop=True)
        
        # Optional downsampling
        if throughput_skip_seconds and throughput_skip_seconds > 0:
            df = df[(df['elapsed_seconds'].astype(float).astype(int) % int(throughput_skip_seconds)) == 0]
        
        # Create individual plot for this server
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        for col in source_cols:
            ax.plot(df['elapsed_seconds'], df[col], label=col, linewidth=2, alpha=0.9)
        
        ax.set_title(f'Throughput over elapsed time - {server} ({experiment})', fontsize=14, fontweight='bold')
        ax.set_xlabel('Elapsed seconds', fontsize=12)
        ax.set_ylabel('Commands per second (GET/SET)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', ncol=2, fontsize=10)
        
        if y_max_throughput is not None:
            ax.set_ylim(0, y_max_throughput)
        
        # Save individual plot
        server_png = os.path.join(output_dir, f'{server}_throughput.png')
        plt.tight_layout()
        plt.savefig(server_png, dpi=200)
        plt.close()
        print(f"✅ Saved {server} plot: {server_png}")
    
    print(f"📁 All individual server plots saved to: {output_dir}")


def main() -> None:
    # Edit these constants to your desired experiment and output:
    EXPERIMENT_DIR = "/home/abhattar/ml_cars_main/ember_mlp_sync"
    OUTPUT_IMAGE = None  # or set a path, e.g., "/home/abhattar/combined.png"
    Y_MAX = None         # e.g., 2000 to cap y-axis
    THROUGHPUT_SKIP_SECONDS = 10  # e.g., 10 plots only every 10 seconds; 0 disables skipping
    PLOT_CLIENT = False  # Set to False to exclude client throughput and only plot server throughput
    
    # Create combined plot (throughput + metrics)
    plot_throughput_and_training(exp_dir=EXPERIMENT_DIR, output_filename=OUTPUT_IMAGE, y_max_throughput=Y_MAX, throughput_skip_seconds=THROUGHPUT_SKIP_SECONDS, plot_client=PLOT_CLIENT)
    
    # Create individual server throughput plots
    plot_individual_server_throughput(exp_dir=EXPERIMENT_DIR, output_dir=None, y_max_throughput=Y_MAX, throughput_skip_seconds=THROUGHPUT_SKIP_SECONDS, plot_client=PLOT_CLIENT)


if __name__ == '__main__':
    main()


