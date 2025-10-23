package code.java;

/**
 * Basic matrix multiplication implementation using triple-nested loops.
 *
 * This class provides a naive O(n³) matrix multiplication algorithm for
 * benchmarking purposes. It intentionally avoids optimizations and external
 * libraries to provide a fair baseline comparison across languages.
 *
 * The implementation follows the classical mathematical definition:
 *
 *     C[i, j] = Σ (k = 0 to m-1) A[i, k] * B[k, j]
 */
public class MatrixMultiplier {

    /**
     * Multiplies two matrices using the classical triple-loop algorithm.
     *
     * Computes C = A × B where:
     *   - A has dimensions (n × m)
     *   - B has dimensions (m × p)
     *   - C has dimensions (n × p)
     *
     * Time Complexity:  O(n * m * p)
     * Space Complexity: O(n * p) for the output matrix
     *
     * @param A First input matrix of shape (n × m)
     * @param B Second input matrix of shape (m × p)
     * @return Result matrix C of shape (n × p)
     * @throws IllegalArgumentException if the number of columns in A
     *         does not equal the number of rows in B
     *
     * Example:
     * <pre>
     * double[][] A = {{1, 2}, {3, 4}};
     * double[][] B = {{5, 6}, {7, 8}};
     * double[][] C = MatrixMultiplier.multiply(A, B);
     * // C = {{19, 22}, {43, 50}}
     * </pre>
     */
    public static double[][] multiply(double[][] A, double[][] B) {
        int n = A.length;
        int m = A[0].length;
        int p = B[0].length;

        if (m != B.length) {
            throw new IllegalArgumentException(
                String.format(
                    "Incompatible shapes: A is %dx%d, B is %dx%d. " +
                    "A's columns must equal B's rows.",
                    n, m, B.length, p
                )
            );
        }

        double[][] C = new double[n][p];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < p; j++) {
                double acc = 0.0;
                for (int k = 0; k < m; k++) {
                    acc += A[i][k] * B[k][j];
                }
                C[i][j] = acc;
            }
        }

        return C;
    }
}
