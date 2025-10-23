
/**
 * @file matrix_mult.c
 * @brief Implementation of basic matrix multiplication using triple-nested loops
 * 
 * This implementation provides a naive O(n³) matrix multiplication algorithm
 * for benchmarking purposes. It intentionally avoids optimizations such as:
 * - Loop unrolling
 * - SIMD vectorization
 * - Cache blocking/tiling
 * - Parallel execution
 * 
 * This ensures a fair baseline comparison across programming languages.
 */

#include "matrix_mult.h"

/**
 * @brief Multiply two square matrices using the classical triple-loop algorithm
 * 
 * Implementation details:
 * - Uses row-major indexing: element (i,j) is at index [i*n + j]
 * - Optimizes memory access by computing row offset once per outer loop
 * - Accumulates result in a scalar variable before writing to output
 * - Inner loop accesses B in column-major pattern (potential cache misses)
 * 
 * @param A Pointer to first input matrix (n×n floats, row-major)
 * @param B Pointer to second input matrix (n×n floats, row-major)
 * @param C Pointer to output matrix (n×n floats, row-major, will be overwritten)
 * @param n Dimension of the square matrices
 */
void matrix_multiplication(const float* A, const float* B, float* C, int n) {
    /* Triple-nested loop structure */
    for (int i = 0; i < n; i++) {               /* Iterate over rows of A */
        int row = i * n;                        /* Precompute row offset for A and C */
        
        for (int j = 0; j < n; j++) {           /* Iterate over columns of B */
            float acc = 0.0f;                   /* Accumulator for dot product */
            
            for (int k = 0; k < n; k++) {       /* Iterate over common dimension */
                /* Compute dot product: row i of A with column j of B
                 * A[i,k] is at index [row + k]
                 * B[k,j] is at index [k*n + j]
                 */
                acc += A[row + k] * B[k * n + j];
            }
            
            /* Store accumulated result in output matrix
             * C[i,j] is at index [row + j]
             */
            C[row + j] = acc;
        }
    }
}