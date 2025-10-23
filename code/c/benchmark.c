/**
 * @file benchmark.c
 * @brief Matrix multiplication benchmarking harness for C implementation
 * 
 * This program benchmarks the performance of matrix multiplication across
 * multiple matrix sizes and runs, recording execution time, CPU usage, and
 * memory consumption to a CSV file.
 * 
 * Command-line arguments:
 *   argv[1]: Comma-separated matrix sizes (e.g., "64,128,256")
 *   argv[2]: Number of runs per size (default: 3)
 *   argv[3]: Output CSV file path (default: "results_raw.csv")
 *   argv[4]: Random seed (default: 27)
 * 
 * Example: benchmark.exe "64,128,256" 5 output.csv 42
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <windows.h>
#include <psapi.h>
#include "matrix_mult.h"

/* CSV header format */
#define HEADER "run_id;language;size;run_idx;time_ms;cpu_pct;peak_mib\n"

/**
 * @brief Get current time in seconds with high precision
 * @return Current time in seconds as a double
 */
static double now_sec(void) {
    LARGE_INTEGER frequency, counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)frequency.QuadPart;
}

/**
 * @brief Get total CPU time consumed by the current process
 * @return CPU time in seconds (user + kernel time)
 */
static double proc_cpu_seconds(void) {
    FILETIME creation_time, exit_time, kernel_time, user_time;
    GetProcessTimes(GetCurrentProcess(), &creation_time, &exit_time, 
                    &kernel_time, &user_time);
    
    /* Convert FILETIME to 64-bit integer (100-nanosecond intervals) */
    ULONGLONG kernel_ticks = ((ULONGLONG)kernel_time.dwLowDateTime) | 
                             ((ULONGLONG)kernel_time.dwHighDateTime << 32);
    ULONGLONG user_ticks = ((ULONGLONG)user_time.dwLowDateTime) | 
                           ((ULONGLONG)user_time.dwHighDateTime << 32);
    
    /* Convert to seconds (1e7 = 100ns per second) */
    return (kernel_ticks + user_ticks) / 1e7;
}

/**
 * @brief Get number of logical CPU cores
 * @return Number of logical processors, or 1 if detection fails
 */
static int logical_cpus(void) {
    SYSTEM_INFO system_info;
    GetSystemInfo(&system_info);
    return system_info.dwNumberOfProcessors > 0 ? 
           (int)system_info.dwNumberOfProcessors : 1;
}

/**
 * @brief Get current process memory usage in MiB
 * @return Working set size in mebibytes (MiB), or 0.0 on error
 */
static double current_mem_mib(void) {
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize / (1024.0 * 1024.0);
    }
    return 0.0;
}

/**
 * @brief Write CSV header to file if it doesn't exist
 * @param path Path to the output CSV file
 */
static void write_header_if_needed(const char* path) {
    FILE* f = fopen(path, "r");
    if (f) {
        fclose(f);
        return; /* File exists, header already written */
    }
    
    /* File doesn't exist, create it and write header */
    f = fopen(path, "a");
    fputs(HEADER, f);
    fclose(f);
}

/**
 * @brief Generate a run identifier string from current local time
 * @param buf Buffer to store the run ID string
 * @param n Size of the buffer
 * 
 * Format: DD/MM/HH/MM (day/month/hour/minute)
 */
static void run_id_str(char* buf, size_t n) {
    SYSTEMTIME st;
    GetLocalTime(&st);
    _snprintf_s(buf, n, _TRUNCATE, "%02d/%02d/%02d/%02d", 
                st.wDay, st.wMonth, st.wHour, st.wMinute);
}

/**
 * @brief Main benchmarking entry point
 * 
 * Parses command-line arguments, generates random matrices, and runs
 * the matrix multiplication benchmark for each specified size and run count.
 */
int main(int argc, char** argv) {
    /* Default configuration */
    int sizes_default[] = {64, 128, 256, 512, 1024};
    int sizes[64];  /* Maximum 64 different sizes */
    int nsizes = 0;
    int runs = 3;
    const char* out = "results_raw.csv";
    int seed = 27;
    
    /* Parse command-line argument 1: comma-separated matrix sizes */
    if (argc > 1 && argv[1][0]) {
        const char* p = argv[1];
        while (*p && nsizes < 64) {
            sizes[nsizes++] = (int)strtol(p, NULL, 10);
            p = strchr(p, ',');
            if (!p) break;
            ++p;
        }
    }
    
    /* Use default sizes if none specified */
    if (nsizes == 0) {
        memcpy(sizes, sizes_default, sizeof(sizes_default));
        nsizes = (int)(sizeof(sizes_default) / sizeof(sizes_default[0]));
    }
    
    /* Parse command-line argument 2: number of runs */
    if (argc > 2) runs = atoi(argv[2]);
    
    /* Parse command-line argument 3: output file path */
    if (argc > 3) out = argv[3];
    
    /* Parse command-line argument 4: random seed */
    if (argc > 4) seed = atoi(argv[4]);
    
    /* Initialize random number generator */
    srand(seed);
    
    /* Prepare output file */
    write_header_if_needed(out);
    
    /* Generate run identifier */
    char run_id[32];
    run_id_str(run_id, sizeof(run_id));
    const char* language = "C";
    const int ncpu = logical_cpus();
    
    /* Main benchmarking loop: iterate over all matrix sizes */
    for (int si = 0; si < nsizes; ++si) {
        int n = sizes[si];
        
        /* Allocate matrices A, B, and C */
        float* A = (float*)malloc((size_t)n * (size_t)n * sizeof(float));
        float* B = (float*)malloc((size_t)n * (size_t)n * sizeof(float));
        float* C = (float*)malloc((size_t)n * (size_t)n * sizeof(float));
        
        /* Initialize matrices with random values in [0, 1] */
        for (int i = 0; i < n * n; i++) {
            A[i] = (float)rand() / RAND_MAX;
            B[i] = (float)rand() / RAND_MAX;
        }
        
        /* Perform multiple runs for statistical stability */
        for (int r = 1; r <= runs; ++r) {
            /* Capture metrics before execution */
            double mem_before = current_mem_mib();
            double cpu0 = proc_cpu_seconds();
            double t0 = now_sec();
            
            /* Execute matrix multiplication */
            matrix_multiplication(A, B, C, n);
            
            /* Capture metrics after execution */
            double t1 = now_sec();
            double cpu1 = proc_cpu_seconds();
            double mem_after = current_mem_mib();
            
            /* Calculate performance metrics */
            double wall = t1 - t0;                          /* Wall-clock time */
            double time_ms = wall * 1000.0;                 /* Convert to milliseconds */
            double cpu_pct = 100.0 * (cpu1 - cpu0) / (wall * ncpu);  /* CPU percentage */
            double peak_mib = mem_after > mem_before ? mem_after : mem_before;
            
            /* Print results to console */
            printf("n=%d run=%d time=%.2f ms CPU=%.1f%% MEM=%.2f MiB\n", 
                   n, r, time_ms, cpu_pct, peak_mib);
            
            /* Append results to CSV file */
            FILE* f = fopen(out, "a");
            fprintf(f, "%s;%s;%d;%d;%.3f;%.1f;%.2f\n", 
                    run_id, language, n, r, time_ms, cpu_pct, peak_mib);
            fclose(f);
        }
        
        /* Free allocated matrices */
        free(A);
        free(B);
        free(C);
    }
    
    return 0;
}