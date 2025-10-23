"""
Basic matrix multiplication implementation using triple-nested loops.

This module provides a naive O(n³) matrix multiplication algorithm for
benchmarking purposes. It intentionally avoids optimizations and external
libraries (except NumPy for array handling) to provide a fair baseline
comparison across languages.
"""

import numpy as np


def matrixMultiplication(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Multiply two matrices using the classical triple-loop algorithm.
    
    Computes C = A × B using the standard mathematical definition:
        C[i,j] = Σ(k=0 to m-1) A[i,k] * B[k,j]
    
    This implementation uses explicit loops rather than optimized BLAS routines
    to provide a consistent baseline for cross-language performance comparisons.
    
    Time Complexity: O(n * m * p) where A is (n×m) and B is (m×p)
    Space Complexity: O(n * p) for the output matrix
    
    Args:
        A: First input matrix of shape (n, m)
        B: Second input matrix of shape (m, p)
    
    Returns:
        Result matrix C of shape (n, p)
    
    Raises:
        ValueError: If matrix dimensions are incompatible (A.shape[1] != B.shape[0])
    
    Examples:
        >>> A = np.array([[1, 2], [3, 4]])
        >>> B = np.array([[5, 6], [7, 8]])
        >>> C = matrixMultiplication(A, B)
        >>> C
        array([[19., 22.],
               [43., 50.]])
    
    Notes:
        - Uses explicit type casting to float to ensure consistent behavior
        - Accumulates in a scalar variable for clarity and consistency
        - Output dtype matches input A's dtype
    """
    # Validate matrix dimensions for multiplication
    if A.shape[1] != B.shape[0]:
        raise ValueError(
            f"Incompatible shapes: A.shape={A.shape}, B.shape={B.shape}. "
            f"A.shape[1] must equal B.shape[0]"
        )
    
    # Extract dimensions: A is (n × m), B is (m × p)
    n = A.shape[0]  # Number of rows in A
    m = A.shape[1]  # Number of columns in A (= number of rows in B)
    p = B.shape[1]  # Number of columns in B
    
    # Initialize output matrix with zeros
    C = np.zeros((n, p), dtype=A.dtype)
    
    # Triple-nested loop: O(n³) for square matrices
    for i in range(n):              # Iterate over rows of A
        for j in range(p):          # Iterate over columns of B
            acc = 0.0               # Accumulator for dot product
            for k in range(m):      # Iterate over common dimension
                # Compute dot product: row i of A with column j of B
                acc += float(A[i, k]) * float(B[k, j])
            C[i, j] = acc           # Store result
    
    return C