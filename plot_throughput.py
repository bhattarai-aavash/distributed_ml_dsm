#!/usr/bin/env python3
import argparse
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt


def plot_server_breakdown(csv_path: str, out_dir: str, y_max: int | None = None) -> None:
    df = pd.read_csv(csv_path)
    # Expected columns: experiment,server,second,elapsed_seconds,client,<peer1>,<peer2>,...,total
    required = {"experiment", "server", "elapsed_seconds", "client", "total"}
    missing = required - set(df.columns)
    if missing:
        print(f"Skipping {csv_path}: missing columns {missing}")
        return

    server = str(df["server"].iloc[0]) if not df.empty else os.path.basename(csv_path).split("_")[0]
    experiment = str(df["experiment"].iloc[0]) if not df.empty else "experiment"

    # Identify source columns (exclude metadata and total)
    meta_cols = {"experiment", "server", "second", "elapsed_seconds", "total"}
    source_cols = [c for c in df.columns if c not in meta_cols]
    # Ensure consistent sort by elapsed time
    df = df.sort_values(["elapsed_seconds"]).reset_index(drop=True)

    plt.figure(figsize=(12, 6))
    for col in source_cols:
        plt.plot(df["elapsed_seconds"], df[col], label=col, linewidth=1.8)

    plt.title(f"Throughput over time for {server} ({experiment})")
    plt.xlabel("Elapsed seconds")
    plt.ylabel("Commands per second (GET/SET)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", ncol=2)
    if y_max is not None:
        plt.ylim(0, y_max)

    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, f"{server}_throughput.png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"Wrote {png_path}")


def plot_all_combined(exp_dir: str, out_dir: str, y_max: int | None = None) -> None:
    csv_paths = sorted(glob.glob(os.path.join(exp_dir, "*_breakdown.csv")))
    if not csv_paths:
        print(f"No *_breakdown.csv files found under {exp_dir}")
        return

    plt.figure(figsize=(14, 7))
    any_series = False

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        server = str(df["server"].iloc[0])
        experiment = str(df["experiment"].iloc[0])

        meta_cols = {"experiment", "server", "second", "elapsed_seconds"}
        source_cols = [c for c in df.columns if c not in meta_cols]
        # Keep consistent order: client first, then peers, then total last if present
        source_cols = (
            ([c for c in source_cols if c == "client"])
            + [c for c in source_cols if c not in ("client", "total")]
            + ([c for c in source_cols if c == "total"])
        )

        df = df.sort_values(["elapsed_seconds"]).reset_index(drop=True)

        for col in source_cols:
            label = f"{server}:{col}"
            plt.plot(df["elapsed_seconds"], df[col], label=label, linewidth=1.5, alpha=0.85)
            any_series = True

    if not any_series:
        print("No data to plot in combined figure")
        plt.close()
        return

    # Use last experiment name encountered for title; mixed experiments unlikely within one dir
    title_exp = experiment if 'experiment' in locals() else os.path.basename(exp_dir)
    plt.title(f"Throughput over time (all servers) ({title_exp})")
    plt.xlabel("Elapsed seconds")
    plt.ylabel("Commands per second (GET/SET)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", ncol=3, fontsize=8)
    if y_max is not None:
        plt.ylim(0, y_max)

    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, "combined_all_servers_throughput.png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"Wrote {png_path}")


def main() -> None:
    # Hardcode your experiment directory here
    EXPERIMENT_DIR = "/home/abhattar/ml_cars_main/cmd_throughput_20250929_114558/20250929_104430"
    OUTPUT_DIR = None  # or set to a custom path, else defaults to <EXPERIMENT_DIR>/plots
    Y_MAX = None       # set to an int to cap Y-axis (e.g., 2000)

    exp_dir = os.path.abspath(EXPERIMENT_DIR)
    if not os.path.isdir(exp_dir):
        raise SystemExit(f"Not a directory: {exp_dir}")

    out_dir = OUTPUT_DIR or os.path.join(exp_dir, "plots")

    csv_paths = sorted(glob.glob(os.path.join(exp_dir, "*_breakdown.csv")))
    if not csv_paths:
        print(f"No *_breakdown.csv files found under {exp_dir}")
        return

    for csv_path in csv_paths:
        plot_server_breakdown(csv_path, out_dir, y_max=Y_MAX)

    # Also generate a combined figure with all servers/series together
    plot_all_combined(exp_dir, out_dir, y_max=Y_MAX)


if __name__ == "__main__":
    main()


