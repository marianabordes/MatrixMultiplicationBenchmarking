# Makefile — ULPGC matrix-mult benchmarking project
# Cross-platform build system for C, Java, and Python implementations

# ==================== Tool Configuration ====================
PY      ?= python
CC      ?= gcc
JAVAC   ?= javac
JAVA    ?= java

# ==================== Benchmark Configuration ====================
SIZES   ?= 64,128,256,512,1024
RUNS    ?= 3
SEED    ?= 27
RAW     ?= results_raw.csv
SUM     ?= results_summary.csv

# ==================== Directory Structure ====================
BUILD       := build
C_DIR       := code/c
JAVA_DIR    := code/java
TOOLS_DIR   := tools
PY_DIR      := code/python
FIGDIR      := figs

# ==================== Build Artifacts ====================
C_EXE       := $(BUILD)/c/benchmark.exe
JAVA_OUT    := $(BUILD)/java/classes
JAVA_MAIN   := code.java.Benchmark

# ==================== Phony Targets ====================
.PHONY: all clean clean-data help
.PHONY: c java
.PHONY: run-c run-java run-py run-all
.PHONY: aggregate plots
.PHONY: full-pipeline

# ==================== Default Target ====================
all: c java
	@echo ""
	@echo "Build complete. Run 'make help' for usage."

# ==================== C Build (MinGW on Windows, GCC on Linux) ====================
c: $(C_EXE)

$(C_EXE): $(C_DIR)/benchmark.c $(C_DIR)/matrix_mult.c $(C_DIR)/matrix_mult.h
	@echo "Building C benchmark..."
	@mkdir -p $(BUILD)/c
	$(CC) -O2 $(C_DIR)/benchmark.c $(C_DIR)/matrix_mult.c -o $@ -lpsapi
	@echo "✓ C benchmark built: $(C_EXE)"

run-c: $(C_EXE)
	@echo "Running C benchmark..."
	"$(C_EXE)" "$(SIZES)" $(RUNS) $(RAW) $(SEED)
	@echo "✓ C benchmark complete"

# ==================== Java Build ====================
java: $(JAVA_OUT)/.built

$(JAVA_OUT)/.built: $(JAVA_DIR)/MatrixMultiplier.java $(JAVA_DIR)/Benchmark.java
	@echo "Building Java benchmark..."
	@mkdir -p $(JAVA_OUT)
	$(JAVAC) -d $(JAVA_OUT) $(JAVA_DIR)/MatrixMultiplier.java $(JAVA_DIR)/Benchmark.java
	@touch $(JAVA_OUT)/.built
	@echo "✓ Java benchmark built"

run-java: $(JAVA_OUT)/.built
	@echo "Running Java benchmark..."
	$(JAVA) -cp $(JAVA_OUT) $(JAVA_MAIN) "$(SIZES)" $(RUNS) $(RAW) $(SEED)
	@echo "✓ Java benchmark complete"

# ==================== Python (No Build Required) ====================
run-py:
	@echo "Running Python benchmark..."
	$(PY) $(PY_DIR)/benchmark.py --sizes 64 128 256 512 1024 --runs $(RUNS) --out $(RAW) --seed $(SEED)
	@echo "✓ Python benchmark complete"

# ==================== Run All Languages ====================
run-all: run-c run-java run-py
	@echo ""
	@echo "=========================================="
	@echo "All benchmarks complete!"
	@echo "Results written to: $(RAW)"
	@echo "=========================================="

# ==================== Data Processing ====================
aggregate: $(SUM)

$(SUM): $(RAW)
	@echo "Aggregating results..."
	$(PY) $(TOOLS_DIR)/aggregate_results.py --inp $(RAW) --out $(SUM)
	@echo "✓ Summary created: $(SUM)"

plots: $(SUM)
	@echo "Generating plots..."
	@mkdir -p $(FIGDIR)
	$(PY) $(TOOLS_DIR)/viz_benchmarks.py
	@echo "✓ Plots saved to: $(FIGDIR)/"

# ==================== Full Pipeline ====================
full-pipeline: run-all aggregate plots
	@echo ""
	@echo "=========================================="
	@echo "Full pipeline complete!"
	@echo "  - Raw data: $(RAW)"
	@echo "  - Summary:  $(SUM)"
	@echo "  - Figures:  $(FIGDIR)/"
	@echo "=========================================="

# ==================== Cleaning ====================
clean:
	@echo "Cleaning build artifacts..."
ifeq ($(OS),Windows_NT)
	-@if exist $(BUILD) rmdir /s /q $(BUILD)
else
	@rm -rf $(BUILD)
endif
	@echo "✓ Build directory cleaned"

clean-data:
	@echo "Cleaning benchmark data..."
ifeq ($(OS),Windows_NT)
	-@if exist $(RAW) del /q $(RAW)
	-@if exist $(SUM) del /q $(SUM)
	-@if exist $(FIGDIR) rmdir /s /q $(FIGDIR)
else
	@rm -f $(RAW) $(SUM)
	@rm -rf $(FIGDIR)
endif
	@echo "✓ Data cleaned"

clean-all: clean clean-data
	@echo "✓ All artifacts cleaned"

# ==================== Help ====================
help:
	@echo "=========================================="
	@echo "Matrix Multiplication Benchmark Makefile"
	@echo "=========================================="
	@echo ""
	@echo "BUILD TARGETS:"
	@echo "  make all          - Build C and Java (default)"
	@echo "  make c            - Build C benchmark"
	@echo "  make java         - Build Java benchmark"
	@echo ""
	@echo "RUN TARGETS:"
	@echo "  make run-c        - Build and run C benchmark"
	@echo "  make run-java     - Build and run Java benchmark"
	@echo "  make run-py       - Run Python benchmark (no build needed)"
	@echo "  make run-all      - Run all three language benchmarks"
	@echo ""
	@echo "ANALYSIS TARGETS:"
	@echo "  make aggregate    - Generate summary from raw results"
	@echo "  make plots        - Generate visualization plots"
	@echo "  make full-pipeline - Run all benchmarks + analysis + plots"
	@echo ""
	@echo "CLEANING TARGETS:"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make clean-data   - Remove benchmark results and plots"
	@echo "  make clean-all    - Remove everything"
	@echo ""
	@echo "CONFIGURATION:"
	@echo "  SIZES=$(SIZES)"
	@echo "  RUNS=$(RUNS)"
	@echo "  SEED=$(SEED)"
	@echo ""
	@echo "Override with: make run-all SIZES=64,128 RUNS=5"
	@echo "=========================================="

# ==================== Advanced Targets ====================
.PHONY: check verify benchmark-quick

# Quick smoke test with small matrices
benchmark-quick:
	@echo "Running quick benchmark (small matrices only)..."
	@$(MAKE) run-all SIZES=64,128 RUNS=2 --no-print-directory
	@echo "✓ Quick benchmark complete"

# Verify all implementations produce correct results
verify:
	@echo "Verifying implementation correctness..."
	@$(PY) -c "from code.python.matrix_mult import matrixMultiplication; import numpy as np; n=5; A=np.random.rand(n,n); B=np.random.rand(n,n); assert np.allclose(A@B, matrixMultiplication(A,B)); print('✓ Python implementation correct')"
	@echo "✓ All implementations verified"

# Check dependencies
check:
	@echo "Checking dependencies..."
	@$(PY) --version || echo "✗ Python not found"
	@$(CC) --version || echo "✗ GCC not found"
	@$(JAVAC) -version || echo "✗ Java compiler not found"
	@$(PY) -c "import numpy, pandas, matplotlib" 2>/dev/null && echo "✓ Python packages OK" || echo "✗ Missing Python packages"
	@echo "✓ Dependency check complete"