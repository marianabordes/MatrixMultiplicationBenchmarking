"""
Matrix multiplication benchmarking harness for Python implementation.

This module benchmarks the performance of matrix multiplication across
multiple matrix sizes and runs, recording execution time, CPU usage, and
memory consumption to a CSV file.

Command-line arguments:
    --sizes: Space-separated matrix sizes (default: 64 128 256 512 1024)
    --runs: Number of runs per size (default: 3)
    --out: Output CSV file path (default: "results_raw.csv")
    --seed: Random seed (default: 27)
    --check_n: Matrix size for correctness check (default: 5)

Example:
    python benchmark.py --sizes 64 128 256 --runs 5 --out output.csv --seed 42
"""

import argparse
import os
import time
from typing import Tuple

import numpy as np
import psutil

from matrix_mult import matrixMultiplication

# CSV header format
HEADER = "run_id;language;size;run_idx;time_ms;cpu_pct;peak_mib\n"


def check_correctness(n: int, seed: int = 27, atol: float = 1e-8) -> bool:
    """
    Verify correctness of matrix multiplication implementation.
    
    Tests the multiplication with an n×n matrix and compares the result
    against NumPy's built-in matrix multiplication.
    
    Args:
        n: Dimension of the square matrix
        seed: Random seed for reproducibility
        atol: Absolute tolerance for element-wise comparison
    
    Returns:
        True if implementation is correct, False otherwise
    """
    rng = np.random.default_rng(seed)
    A = rng.random((n, n), dtype=np.float64)
    B = rng.random((n, n), dtype=np.float64)
    
    # Compare custom implementation against NumPy's optimized version
    return np.allclose(A @ B, matrixMultiplication(A, B), atol=atol)


def one_run(
    A: np.ndarray, 
    B: np.ndarray, 
    proc: psutil.Process, 
    ncpu: int
) -> Tuple[float, float, float]:
    """
    Execute a single matrix multiplication run and collect metrics.
    
    Args:
        A: First input matrix (n×n)
        B: Second input matrix (n×n)
        proc: psutil Process object for the current process
        ncpu: Number of logical CPU cores
    
    Returns:
        Tuple of (execution_time_ms, cpu_percentage, peak_memory_mib)
    """
    # Capture metrics before execution
    mem_before = proc.memory_info().rss / (1024 * 1024)  # Convert bytes to MiB
    cpu_before = proc.cpu_times()
    t0 = time.perf_counter()
    
    # Execute matrix multiplication
    matrixMultiplication(A, B)
    
    # Capture metrics after execution
    t1 = time.perf_counter()
    cpu_after = proc.cpu_times()
    mem_after = proc.memory_info().rss / (1024 * 1024)  # Convert bytes to MiB
    
    # Calculate performance metrics
    wall = max(t1 - t0, 1e-12)  # Avoid division by zero
    cpu_used = (cpu_after.user - cpu_before.user) + (cpu_after.system - cpu_before.system)
    cpu_pct = 100.0 * cpu_used / (wall * ncpu)
    
    return wall * 1000.0, cpu_pct, max(mem_before, mem_after)


def write_header_if_needed(path: str) -> None:
    """
    Write CSV header to file if it doesn't already exist.
    
    Args:
        path: Path to the output CSV file
    """
    if not os.path.isfile(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(HEADER)


def main() -> None:
    """
    Main benchmarking entry point.
    
    Parses command-line arguments, verifies correctness, generates random
    matrices, and runs the matrix multiplication benchmark for each
    specified size and run count.
    """
    # Parse command-line arguments
    p = argparse.ArgumentParser(description="Raw-run Python benchmark.")
    p.add_argument("--sizes", type=int, nargs="+", default=[64, 128, 256, 512, 1024],
                   help="Matrix sizes to benchmark")
    p.add_argument("--runs", type=int, default=3,
                   help="Number of runs per matrix size")
    p.add_argument("--out", type=str, default="results_raw.csv",
                   help="Output CSV file path")
    p.add_argument("--seed", type=int, default=27,
                   help="Random seed for reproducibility")
    p.add_argument("--check_n", type=int, default=5,
                   help="Matrix size for correctness verification")
    args = p.parse_args()

    # Verify implementation correctness before benchmarking
    if not check_correctness(args.check_n, args.seed):
        raise SystemExit("Verification failed")

    # Initialize components
    rng = np.random.default_rng(args.seed)
    proc = psutil.Process(os.getpid())
    ncpu = psutil.cpu_count(logical=True) or 1
    language = "Python"
    run_id = time.strftime("%d/%m/%H/%M", time.localtime())

    # Prepare output file
    write_header_if_needed(args.out)

    # Main benchmarking loop: iterate over all matrix sizes
    for n in args.sizes:
        # Generate random matrices for this size
        A = rng.random((n, n), dtype=np.float32)
        B = rng.random((n, n), dtype=np.float32)
        
        # Perform multiple runs for statistical stability
        for r in range(1, args.runs + 1):
            # Execute run and collect metrics
            t_ms, cpu_pct, peak_mib = one_run(A, B, proc, ncpu)
            
            # Print results to console
            print(f"n={n} run={r} time={t_ms:.2f} ms CPU={cpu_pct:.1f}% MEM={peak_mib:.2f} MiB")
            
            # Append results to CSV file
            with open(args.out, "a", encoding="utf-8") as f:
                f.write(f"{run_id};{language};{n};{r};{t_ms:.3f};{cpu_pct:.1f};{peak_mib:.2f}\n")


if __name__ == "__main__":
    main()