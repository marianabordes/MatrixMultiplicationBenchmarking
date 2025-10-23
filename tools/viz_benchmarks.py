"""
Generate comprehensive benchmark visualization charts for matrix multiplication.

This script creates a full suite of charts to compare Python, Java, and C
matrix multiplication performance across various metrics including execution
time, CPU usage, memory consumption, and computational throughput.

Input Files
-----------
- results_summary.csv: Aggregated statistics per language and size
  Columns: run_id;language;size;runs;avg_time_ms;min_time_ms;max_time_ms;cpu_pct_avg;peak_mib

- results_raw.csv: Per-run raw measurements
  Columns: run_id;language;size;run_idx;time_ms;cpu_pct;peak_mib

Output Files
------------
PNG figures saved to ./figs/ directory:
- time_vs_size.png: Execution time vs matrix size with error bars
- time_vs_size_loglog.png: Log-log plot to verify O(n³) scaling
- speedup_vs_fastest.png: Relative speedup compared to fastest implementation
- cpu_vs_size.png: CPU usage percentage vs matrix size
- mem_vs_size.png: Peak memory consumption vs matrix size
- boxplots_time_by_size.png: Per-run time distribution by size and language
- efficiency_gflops.png: Computational throughput in GFLOP/s

Notes
-----
- Handles semicolon-separated CSVs with comma decimal separators
- Uses matplotlib for all visualizations (no seaborn dependency)
- Language colors: Python=blue, Java=orange, C=purple
"""

import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==================== Configuration ====================

# Input file paths
SUMMARY_PATH = "results_summary.csv"
RAW_PATH = "results_raw.csv"

# Output directory for figures
OUT_DIR = "figs"

# Color scheme for each language
COLORS = {
    "Python": "blue",
    "Java": "orange",
    "C": "purple",
}

# Consistent ordering of languages across all plots
LANG_ORDER = ["Python", "Java", "C"]

# ==================== Utility Functions ====================


def _to_num(x):
    """
    Convert a possibly locale-formatted string/number to float.
    
    Handles various numeric formats including:
    - European format: '1.234,56' (dot thousands, comma decimal)
    - US format: '1,234.56' (comma thousands, dot decimal)
    - Simple format: '1234.56' or '1234'
    
    Args:
        x: String, number, or NaN-like value to convert
    
    Returns:
        Float value, or np.nan if conversion fails
    
    Examples:
        >>> _to_num('1.234,56')
        1234.56
        >>> _to_num('1234.56')
        1234.56
        >>> _to_num('nan')
        nan
    """
    s = str(x).strip()
    
    # Handle empty or NaN strings
    if s == "" or s.lower() == "nan":
        return np.nan
    
    # If both comma and dot present, assume European format
    if "," in s and "." in s:
        # Remove thousands separator (dot), replace decimal separator (comma)
        s = s.replace(".", "").replace(",", ".")
    else:
        # Otherwise, replace comma with dot for decimal
        s = s.replace(",", ".")
    
    try:
        return float(s)
    except Exception:
        return np.nan


def load_summary(path=SUMMARY_PATH):
    """
    Load and sanitize the summary CSV file.
    
    Reads the aggregated benchmark summary, converts all numeric columns
    to appropriate types, and filters to include only expected languages.
    
    Args:
        path: Path to the summary CSV file
    
    Returns:
        DataFrame with sanitized numeric columns and categorical language column
    """
    df = pd.read_csv(path, sep=";", dtype=str)
    
    # Convert integer columns
    for col in ["size", "runs"]:
        df[col] = df[col].apply(_to_num).astype("Int64")
    
    # Convert float columns
    num_cols = ["avg_time_ms", "min_time_ms", "max_time_ms", "cpu_pct_avg", "peak_mib"]
    for col in num_cols:
        df[col] = df[col].apply(_to_num)
    
    # Filter to expected languages and create ordered categorical
    df = df[df["language"].isin(LANG_ORDER)].copy()
    df["language"] = pd.Categorical(df["language"], categories=LANG_ORDER, ordered=True)
    
    return df.sort_values(["language", "size"])


def load_raw(path=RAW_PATH):
    """
    Load and sanitize the raw per-run CSV file.
    
    Reads the raw benchmark data, converts all numeric columns to appropriate
    types, and filters to include only expected languages.
    
    Args:
        path: Path to the raw CSV file
    
    Returns:
        DataFrame with sanitized numeric columns and categorical language column
    """
    df = pd.read_csv(path, sep=";", dtype=str)
    
    # Convert integer columns
    for col in ["size", "run_idx"]:
        df[col] = df[col].apply(_to_num).astype("Int64")
    
    # Convert float columns
    for col in ["time_ms", "cpu_pct", "peak_mib"]:
        df[col] = df[col].apply(_to_num)
    
    # Filter to expected languages and create ordered categorical
    df = df[df["language"].isin(LANG_ORDER)].copy()
    df["language"] = pd.Categorical(df["language"], categories=LANG_ORDER, ordered=True)
    
    return df.sort_values(["language", "size", "run_idx"])


def ensure_outdir():
    """
    Ensure output directory exists, creating it if necessary.
    """
    os.makedirs(OUT_DIR, exist_ok=True)


def savefig(name):
    """
    Save current matplotlib figure to output directory.
    
    Applies tight layout and saves with high DPI for quality output.
    Closes the figure after saving to free memory.
    
    Args:
        name: Filename for the output PNG (e.g., "time_vs_size.png")
    """
    path = os.path.join(OUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  Saved: {name}")


# ==================== Plot Functions ====================


def plot_time_vs_size(df_sum):
    """
    Plot average execution time vs matrix size with error bars.
    
    Creates a line chart showing how execution time scales with matrix size
    for each language. Error bars represent the min-max range across runs.
    
    Args:
        df_sum: Summary DataFrame with avg/min/max time columns
    """
    plt.figure(figsize=(7, 4.5))
    
    for lang in LANG_ORDER:
        d = df_sum[df_sum["language"] == lang]
        if d.empty:
            continue
        
        # Extract data
        x = d["size"].astype(int).values
        y = d["avg_time_ms"].values
        
        # Calculate error bars (asymmetric: distance to min and max)
        yerr = np.vstack([
            y - d["min_time_ms"].values,  # Lower error
            d["max_time_ms"].values - y   # Upper error
        ])
        
        # Plot with error bars
        plt.errorbar(
            x, y, yerr=yerr, 
            fmt="o-", capsize=3, 
            label=lang, color=COLORS[lang]
        )
    
    plt.title("Execution Time vs Matrix Size")
    plt.xlabel("Matrix size (n × n)")
    plt.ylabel("Time (ms)")
    plt.legend()
    savefig("time_vs_size.png")


def plot_time_vs_size_loglog(df_sum):
    """
    Plot execution time vs size on log-log axes.
    
    Log-log plot reveals the scaling behavior (expected O(n³)) and shows
    constant-factor differences between languages as vertical offsets.
    
    Args:
        df_sum: Summary DataFrame with avg_time_ms column
    """
    plt.figure(figsize=(7, 4.5))
    
    for lang in LANG_ORDER:
        d = df_sum[df_sum["language"] == lang]
        if d.empty:
            continue
        
        x = d["size"].astype(int).values
        y = d["avg_time_ms"].values
        plt.plot(x, y, "o-", label=lang, color=COLORS[lang])
    
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Execution Time vs Size (log–log)")
    plt.xlabel("Matrix size (n)")
    plt.ylabel("Time (ms)")
    plt.legend()
    savefig("time_vs_size_loglog.png")


def plot_speedup_vs_fastest(df_sum):
    """
    Plot relative speedup compared to the fastest implementation per size.
    
    Speedup is calculated as: fastest_time / language_time
    Values of 1.0 indicate the fastest implementation, values < 1.0 are slower.
    
    Args:
        df_sum: Summary DataFrame with avg_time_ms column
    """
    sizes = sorted(df_sum["size"].dropna().astype(int).unique())
    
    plt.figure(figsize=(7, 4.5))
    
    for lang in LANG_ORDER:
        speedups = []
        
        # Calculate speedup for each matrix size
        for n in sizes:
            d = df_sum[df_sum["size"] == n]
            if d.empty:
                speedups.append(np.nan)
                continue
            
            # Find fastest time for this size
            tmin = d["avg_time_ms"].min()
            
            # Get this language's time
            t_lang = d[d["language"] == lang]["avg_time_ms"]
            
            # Calculate speedup (higher is better)
            if not t_lang.empty:
                speedups.append(float(tmin) / float(t_lang.iloc[0]))
            else:
                speedups.append(np.nan)
        
        plt.plot(sizes, speedups, "o-", label=lang, color=COLORS[lang])
    
    plt.title("Speedup vs Fastest (per size)")
    plt.xlabel("Matrix size (n)")
    plt.ylabel("Speedup (× fastest)")
    plt.legend()
    savefig("speedup_vs_fastest.png")


def plot_cpu_vs_size(df_sum):
    """
    Plot average CPU usage percentage vs matrix size.
    
    Shows how CPU utilization varies with problem size for each language.
    Expected to converge around 12.5% for single-threaded execution on 8 cores.
    
    Args:
        df_sum: Summary DataFrame with cpu_pct_avg column
    """
    plt.figure(figsize=(7, 4.5))
    
    for lang in LANG_ORDER:
        d = df_sum[df_sum["language"] == lang]
        if d.empty:
            continue
        
        plt.plot(
            d["size"].astype(int), d["cpu_pct_avg"], 
            "o-", label=lang, color=COLORS[lang]
        )
    
    plt.title("Average CPU Usage vs Matrix Size")
    plt.xlabel("Matrix size (n)")
    plt.ylabel("CPU usage (%)")
    plt.legend()
    savefig("cpu_vs_size.png")


def plot_mem_vs_size(df_sum):
    """
    Plot peak memory consumption vs matrix size.
    
    Shows how memory usage scales (expected O(n²)) for storing three n×n matrices.
    Reveals language-specific memory overhead.
    
    Args:
        df_sum: Summary DataFrame with peak_mib column
    """
    plt.figure(figsize=(7, 4.5))
    
    for lang in LANG_ORDER:
        d = df_sum[df_sum["language"] == lang]
        if d.empty:
            continue
        
        plt.plot(
            d["size"].astype(int), d["peak_mib"], 
            "o-", label=lang, color=COLORS[lang]
        )
    
    plt.title("Peak Memory vs Matrix Size")
    plt.xlabel("Matrix size (n)")
    plt.ylabel("Peak memory (MiB)")
    plt.legend()
    savefig("mem_vs_size.png")


def plot_boxplots_time_by_size(df_raw):
    """
    Create boxplots showing per-run time distribution for each size and language.
    
    Displays the variance in execution times across multiple runs, revealing
    the stability and reproducibility of each implementation.
    
    Args:
        df_raw: Raw DataFrame with per-run time_ms measurements
    """
    sizes = sorted(df_raw["size"].dropna().astype(int).unique())
    if not sizes:
        return
    
    # Calculate subplot grid dimensions
    cols = min(3, len(sizes))
    rows = math.ceil(len(sizes) / cols)
    
    plt.figure(figsize=(6 * cols, 4 * rows))
    
    # Create one subplot per matrix size
    for idx, n in enumerate(sizes, start=1):
        ax = plt.subplot(rows, cols, idx)
        d = df_raw[df_raw["size"] == n]
        
        # Collect data for each language
        data = [
            d[d["language"] == lang]["time_ms"].dropna().values 
            for lang in LANG_ORDER
        ]
        
        # Create boxplot
        ax.boxplot(data, labels=LANG_ORDER, showmeans=True)
        
        # Color x-axis labels to match language colors
        for ticklabel, lang in zip(ax.get_xticklabels(), LANG_ORDER):
            ticklabel.set_color(COLORS[lang])
        
        ax.set_title(f"Per-run Time at n={n}")
        ax.set_xlabel("Language")
        ax.set_ylabel("Time (ms)")
    
    plt.suptitle("Time Dispersion by Size and Language", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    savefig("boxplots_time_by_size.png")


def plot_efficiency_gflops(df_sum):
    """
    Plot computational throughput in GFLOP/s vs matrix size.
    
    Calculates floating-point operations per second using:
    GFLOP/s = 2*n³ / (time_seconds * 10⁹)
    
    Higher values indicate better computational efficiency.
    
    Args:
        df_sum: Summary DataFrame with avg_time_ms column
    """
    plt.figure(figsize=(7, 4.5))
    
    for lang in LANG_ORDER:
        d = df_sum[df_sum["language"] == lang]
        if d.empty:
            continue
        
        # Extract matrix sizes and convert time to seconds
        n = d["size"].astype(int).values
        t_s = d["avg_time_ms"].values / 1000.0
        
        # Calculate GFLOP/s: 2n³ operations / (time * 10⁹)
        gflops = (2.0 * (n.astype(float) ** 3)) / (t_s * 1e9)
        
        plt.plot(n, gflops, "o-", label=lang, color=COLORS[lang])
    
    plt.title("Throughput (GFLOP/s) vs Matrix Size")
    plt.xlabel("Matrix size (n)")
    plt.ylabel("GFLOP/s")
    plt.legend()
    savefig("efficiency_gflops.png")


# ==================== Main Entry Point ====================


if __name__ == "__main__":
    print("Generating benchmark visualizations...")
    
    # Ensure output directory exists
    ensure_outdir()
    
    # Load data
    print(f"Loading data from {SUMMARY_PATH} and {RAW_PATH}...")
    summary = load_summary(SUMMARY_PATH)
    raw = load_raw(RAW_PATH)
    
    # Generate all plots
    print("Creating plots...")
    plot_time_vs_size(summary)
    plot_time_vs_size_loglog(summary)
    plot_speedup_vs_fastest(summary)
    plot_cpu_vs_size(summary)
    plot_mem_vs_size(summary)
    plot_boxplots_time_by_size(raw)
    plot_efficiency_gflops(summary)
    
    print(f"\n✓ All figures saved to: {OUT_DIR}/")