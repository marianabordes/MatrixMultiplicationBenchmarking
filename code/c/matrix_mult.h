
/**
 * @file matrix_mult.h
 * @brief Header file for basic matrix multiplication implementation
 * 
 * This header declares the matrix multiplication function using the
 * classical triple-loop algorithm. The implementation is designed for
 * benchmarking purposes and intentionally avoids SIMD or other optimizations.
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Multiply two square matrices using the classical O(n³) algorithm
 * 
 * Computes C = A × B where all matrices are n×n square matrices stored
 * in row-major order (C-style contiguous arrays).
 * 
 * Mathematical definition:
 *     C[i,j] = Σ(k=0 to n-1) A[i,k] * B[k,j]
 * 
 * Memory Layout:
 *     Matrices are stored as 1D arrays in row-major order
 *     Element (i,j) is at index [i*n + j]
 * 
 * Time Complexity: O(n³)
 * Space Complexity: O(1) auxiliary space (output provided by caller)
 * 
 * @param A Pointer to first input matrix (n×n elements in row-major order)
 * @param B Pointer to second input matrix (n×n elements in row-major order)
 * @param C Pointer to output matrix (n×n elements, will be overwritten)
 * @param n Dimension of the square matrices (all are n×n)
 * 
 * @note All pointers must point to valid memory of at least n*n floats
 * @note The function does not validate inputs; caller must ensure correctness
 * @note Matrices are read-only (const), output C is modified in-place
 * 
 * @example
 * float A[4] = {1.0f, 2.0f, 3.0f, 4.0f};  // 2×2 matrix
 * float B[4] = {5.0f, 6.0f, 7.0f, 8.0f};  // 2×2 matrix
 * float C[4];                              // 2×2 output
 * matrix_multiplication(A, B, C, 2);
 * // C = {19.0f, 22.0f, 43.0f, 50.0f}
 */
void matrix_multiplication(const float* A, const float* B, float* C, int n);

#ifdef __cplusplus
}
#endif


