# High-Performance GEMM and DCU Convolution Optimization

## Overview

**This project is qualified for participation in the "National Pioneer Cup on Intelligent Computing – Shandong University of Science and Technology Selection.**

This repository provides high-performance implementations of General Matrix-Matrix Multiplication (GEMM) and image convolution kernels optimized for multi-core CPU (AVX2 + OpenMP) and DCU GPU (HIP) platforms. It includes:

- **GEMM Variants**: basic, loop-unrolled, blocked, SIMD-optimized, and sparse implementations.
- **Convolution Kernels**: basic, optimized (shared memory & register reuse), and tiled (block-based) designs on DCU.

## Repository Structure

```
project_root/

├── conv/                      # Convolution project
│   ├── build/                 # Build outputs
│   ├── include/               # Header files
│   ├── results/               # Benchmark results & HTML reports
│   ├── scripts/               # Build and test automation scripts
│   ├── src/                   # Convolution source code
│   └── test/                  # Convolution test cases

└── gemm/                      # GEMM project
    ├── obj/                   # Compiled object files
    ├── src/                   # GEMM source code variants
    ├── test/                  # GEMM test cases & benchmark data
    └── Makefile               # Build configuration
```

## Prerequisites

- **OS**: Ubuntu 20.04 or compatible Linux
- **CPU**: x86_64 with AVX2 support (recommended 32 cores)
- **GPU**: DCU with HIP support (16 GB VRAM)
- **Compiler**: GCC ≥9.4.0 (C++17), hipcc for HIP code
- **Libraries**: OpenMP 4.5, HIP 5.4.0, rocBLAS 4.5.0

## Build & Run Instructions

### Convolution (DCU GPU)

```bash
cd conv
env setup (e.g., module load hip/5.4.0)
chmod +x scripts/build.sh scripts/run_tests.sh
./scripts/build.sh         # Compile convolution kernels
./scripts/run_tests.sh     # Run benchmarks & generate HTML report
```

### GEMM (CPU)

```bash
cd gemm
make                       # Compile GEMM implementations
./gemm_test                # Execute GEMM benchmarks
```

## Key Results

- **Convolution**: Up to **879.67×** speedup over single-threaded CPU baseline.
- **GEMM (Dense)**: Up to **7.06 GFLOPS** (~34× speedup) on 1024³ matrices.
- **GEMM (Sparse)**: Over **3000×** speedup for low-density (≤1%) sparse matrices.

## Reproducibility

All build and test steps are automated via `scripts/` and `Makefile`. Full logs and raw data are available under `conv/results/` and `gemm/test/`.

## License

This project is released under the MIT License.
