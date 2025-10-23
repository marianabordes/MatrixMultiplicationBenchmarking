# Matrix Multiplication Benchmark Project

## Project Structure

```
matrix-benchmark
├── code
│   ├── c
│   │   ├── matrix_mult.c
│   │   ├── matrix_mult.h
│   │   └── benchmark.c
│   ├── java
│   │   ├── MatrixMultiplier.java
│   │   └── Benchmark.java
│   └── python
│       ├── matrix_mult.py
│       └── benchmark.py
├── tools
│   ├── aggregate_results.py
│   └── viz_benchmarks.py
├── figs
│   └── (generated figures)
├── results_raw.csv
└── results_summary.csv
```

## Directory Description

- code: Contains implementation in different programming languages
  - `c/`: C implementation files
  - `java/`: Java implementation files
  - `python/`: Python implementation files
- tools: Analysis and visualization tools
- figs: Generated figures and plots
- results_raw.csv: Raw benchmark results
- results_summary.csv: Processed and aggregated results

## Requirements

* Python 3.10+ and `pip`
* Java JDK 11+
* C: Windows with MSVC (`cl`) for `make c-win`, or GCC for `make c-unix` if the code is portable

## Usage

1. Run benchmarks for each language
2. Aggregate results using aggregate_results.py
3. Generate visualizations with viz_benchmarks.py

### Quick Usage

```bash
make venv
make py            # runs the Python benchmark and writes results_raw.csv
make aggregate     # generates results_summary.csv
make figs          # creates PNG figures in figs/
```

## Build Instructions

See Makefile for build and run commands.


## Authors

Mariana Bordes Bueno
