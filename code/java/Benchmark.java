package code.java;

import java.io.*;
import java.lang.management.ManagementFactory;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Arrays;
import java.util.Locale;
import java.util.Random;

/**
 * Benchmark harness for the Java matrix multiplication implementation.
 *
 * Runs the algorithm for multiple matrix sizes and repeated trials,
 * recording wall time, normalized CPU usage, and peak memory into a CSV file.
 *
 * Command-line arguments:
 *   args[0]  comma-separated sizes, e.g. "64,128,256"  (default: 64,128,256,512,1024)
 *   args[1]  runs per size (default: 3)
 *   args[2]  output CSV path (default: results_raw.csv)
 *   args[3]  random seed (default: 27)
 *
 * Example:
 *   java Benchmark "64,128,256" 5 output.csv 42
 *
 * CSV schema (semicolon-separated):
 *   run_id;language;size;run_idx;time_ms;cpu_pct;peak_mib
 */
public class Benchmark {
    /** CSV header written once when creating the file. */
    static final String HEADER = "run_id;language;size;run_idx;time_ms;cpu_pct;peak_mib\n";

    /**
     * Generates an n×n matrix with entries in [0,1).
     *
     * @param n   matrix dimension
     * @param rnd random generator
     * @return n×n matrix of doubles
     */
    static double[][] gen(int n, Random rnd) {
        double[][] M = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                M[i][j] = rnd.nextDouble();
            }
        }
        return M;
    }

    /**
     * Ensures the CSV header exists for the given path.
     *
     * @param path output CSV path
     * @throws IOException on I/O failure
     */
    static void writeHeaderIfNeeded(String path) throws IOException {
        File f = new File(path);
        if (!f.exists()) {
            try (FileWriter fw = new FileWriter(f, true)) {
                fw.write(HEADER);
            }
        }
    }

    /**
     * Returns current process memory (MiB).
     *
     * @return resident memory estimate in MiB
     */
    static double procMemMiB() {
        Runtime rt = Runtime.getRuntime();
        return (rt.totalMemory() - rt.freeMemory()) / (1024.0 * 1024.0);
    }

    /**
     * Computes normalized CPU percentage from CPU time and wall time.
     *
     * @param wallStart wall-clock start (ns)
     * @param wallEnd   wall-clock end (ns)
     * @param cpuStart  process CPU start (ns)
     * @param cpuEnd    process CPU end (ns)
     * @param ncpu      logical core count
     * @return CPU usage percentage normalized by cores
     */
    static double cpuPct(long wallStart, long wallEnd, long cpuStart, long cpuEnd, int ncpu) {
        double wall = (wallEnd - wallStart) / 1e9;
        double cpu  = (cpuEnd - cpuStart) / 1e9;
        return 100.0 * cpu / (wall * Math.max(ncpu, 1));
    }

    /**
     * Lightweight correctness check on a 5×5 case.
     *
     * @return true if multiply() matches a manual computation
     */
    static boolean checkCorrectness() {
        Random rnd = new Random(27);
        int n = 5;
        double[][] A = gen(n, rnd);
        double[][] B = gen(n, rnd);
        double[][] C = MatrixMultiplier.multiply(A, B);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double acc = 0;
                for (int k = 0; k < n; k++) acc += A[i][k] * B[k][j];
                if (Math.abs(acc - C[i][j]) > 1e-9) return false;
            }
        }
        return true;
    }

    /**
     * Entry point. Parses parameters, verifies correctness, executes benchmarks,
     * and appends results to the CSV file.
     *
     * @param args sizes, runs, output, seed
     * @throws Exception on verification or I/O error
     */
    public static void main(String[] args) throws Exception {
        Locale.setDefault(Locale.US);

        if (!checkCorrectness()) {
            throw new RuntimeException("Verification failed");
        }

        int[] sizes = (args.length > 0)
            ? Arrays.stream(args[0].split(",")).mapToInt(Integer::parseInt).toArray()
            : new int[]{64, 128, 256, 512, 1024};

        int runs  = (args.length > 1) ? Integer.parseInt(args[1]) : 3;
        String out = (args.length > 2) ? args[2] : "results_raw.csv";
        int seed  = (args.length > 3) ? Integer.parseInt(args[3]) : 27;

        String language = "Java";
        String runId = LocalDateTime.now().format(DateTimeFormatter.ofPattern("dd/MM/HH/mm"));

        writeHeaderIfNeeded(out);

        var osBean = (com.sun.management.OperatingSystemMXBean)
                ManagementFactory.getOperatingSystemMXBean();
        int ncpu = osBean.getAvailableProcessors();
        Random rnd = new Random(seed);

        for (int n : sizes) {
            double[][] A = gen(n, rnd);
            double[][] B = gen(n, rnd);

            for (int r = 1; r <= runs; r++) {
                System.gc();

                double memBefore = procMemMiB();
                long cpuStart = osBean.getProcessCpuTime();
                long t0 = System.nanoTime();

                MatrixMultiplier.multiply(A, B);

                long t1 = System.nanoTime();
                long cpuEnd = osBean.getProcessCpuTime();
                double memAfter = procMemMiB();

                double timeMs = (t1 - t0) / 1e6;
                double cpu = cpuPct(t0, t1, cpuStart, cpuEnd, ncpu);
                double peakMiB = Math.max(memBefore, memAfter);

                System.out.printf(Locale.US,
                        "n=%d run=%d time=%.2f ms CPU=%.1f%%%n", n, r, timeMs, cpu);

                try (FileWriter fw = new FileWriter(out, true)) {
                    fw.write(String.format(Locale.US,
                            "%s;%s;%d;%d;%.3f;%.1f;%.2f%n",
                            runId, language, n, r, timeMs, cpu, peakMiB));
                }
            }
        }
    }
}
