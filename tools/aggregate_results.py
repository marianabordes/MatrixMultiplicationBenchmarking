"""
Aggregate per-run benchmark results into summary statistics.

This script reads raw benchmark results from a CSV file, computes summary
statistics (mean, min, max) per language and matrix size, and writes the
aggregated results to a new CSV file with Excel-friendly decimal formatting
(comma as decimal separator).

Input CSV format (semicolon-separated):
    run_id;language;size;run_idx;time_ms;cpu_pct;peak_mib

Output CSV format (semicolon-separated):
    run_id;language;size;runs;avg_time_ms;min_time_ms;max_time_ms;cpu_pct_avg;peak_mib

Usage:
    python aggregate_results.py --inp results_raw.csv --out results_summary.csv
"""

import argparse
import pandas as pd

# CSV delimiter used in input and output files
SEP = ";"


def fmt(x: float, nd: int) -> str:
    """
    Format a number as a string with comma decimal separator for Excel compatibility.
    
    This function converts a float to a string with a specified number of decimal
    places, replacing the period with a comma to match European Excel locale settings.
    
    Args:
        x: Number to format
        nd: Number of decimal places
    
    Returns:
        Formatted string with comma as decimal separator (e.g., "1234,567")
    
    Examples:
        >>> fmt(1234.567, 2)
        '1234,57'
        >>> fmt(3.14159, 3)
        '3,142'
    """
    return f"{float(x):.{nd}f}".replace(".", ",")


def main():
    """
    Main entry point for the aggregation script.
    
    Parses command-line arguments, reads raw benchmark data, computes summary
    statistics grouped by run_id, language, and size, and writes the results
    to a CSV file with Excel-friendly formatting.
    """
    # Parse command-line arguments
    ap = argparse.ArgumentParser(
        description="Aggregate per-run results with Excel-friendly decimals."
    )
    ap.add_argument(
        "--inp", 
        type=str, 
        default="results_raw.csv",
        help="Input CSV file with raw benchmark results"
    )
    ap.add_argument(
        "--out", 
        type=str, 
        default="results_summary.csv",
        help="Output CSV file for aggregated summary statistics"
    )
    args = ap.parse_args()
    
    # Read raw benchmark data
    df = pd.read_csv(args.inp, sep=SEP)
    
    # Convert columns to appropriate numeric types
    # Integer columns: size and run index
    for col in ["size", "run_idx"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    
    # Float columns: timing, CPU, and memory metrics
    for col in ["time_ms", "cpu_pct", "peak_mib"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Group by run_id, language, and size to compute statistics
    g = df.groupby(["run_id", "language", "size"], as_index=False)
    
    # Aggregate statistics for each group
    summary = g.agg(
        runs=("run_idx", "count"),            # Total number of runs
        avg_time_ms=("time_ms", "mean"),      # Average execution time
        min_time_ms=("time_ms", "min"),       # Minimum execution time
        max_time_ms=("time_ms", "max"),       # Maximum execution time
        cpu_pct_avg=("cpu_pct", "mean"),      # Average CPU usage
        peak_mib=("peak_mib", "max"),         # Peak memory consumption
    ).sort_values(["language", "size", "run_id"])
    
    # Round and format numeric columns with comma decimal separator for Excel
    # This ensures compatibility with European Excel locale settings
    summary["avg_time_ms"] = summary["avg_time_ms"].round(3).map(lambda v: fmt(v, 3))
    summary["min_time_ms"] = summary["min_time_ms"].round(3).map(lambda v: fmt(v, 3))
    summary["max_time_ms"] = summary["max_time_ms"].round(3).map(lambda v: fmt(v, 3))
    summary["cpu_pct_avg"] = summary["cpu_pct_avg"].round(1).map(lambda v: fmt(v, 1))
    summary["peak_mib"] = summary["peak_mib"].round(2).map(lambda v: fmt(v, 2))
    
    # Write summary to output CSV with UTF-8-BOM encoding for Excel compatibility
    summary.to_csv(args.out, index=False, sep=SEP, encoding="utf-8-sig")
    
    print(f"Aggregated {len(df)} raw records into {len(summary)} summary rows")
    print(f"Results written to: {args.out}")


if __name__ == "__main__":
    main()