# LLM kernels in CUDA

The memory-bound kernels used inside transformers, written from scratch in CUDA
on an NVIDIA Tesla T4 and measured in GB/s against the T4's memory bandwidth
(320 GB/s spec, ~250 GB/s achievable in practice).

## 1. Parallel reduction (sum)

N = 16,777,216 floats (64 MB), 256 threads per block. Checked against a CPU sum in
double precision. Average of 20 runs.

| Version | Idea | GB/s | vs v1 |
| --- | --- | --- | --- |
| v1 | Interleaved addressing | ~26 | 1x |
| v2 | Packed active threads (no divergence) | ~34 | 1.3x |
| v3 | Sequential addressing (no bank conflicts) | ~50 | 2x |
| v4 | Thread coarsening: grid-stride loop, 64 elements per thread | ~215 | 8x |
| v5 | + warp shuffles (1 barrier per block instead of 9) | ~221 | 8.5x |

What I learned:
- **Divergence:** v1's active threads are scattered across all 8 warps, so every
  warp runs every step. Packing them (v2) cuts the warp-steps from 47 to 12.
- **Bank conflicts:** v2's packed threads read addresses 2, 4, 8... apart, landing
  on the same shared memory banks. Sequential addressing (v3) makes neighbouring
  threads read neighbouring addresses: conflict-free.
- **Thread coarsening was the real win:** v1 to v3 all run one reduction tree per
  256 elements, so the tree's barriers dominate. Having each thread first sum 64
  elements in a register (v4) means one tree per 16,384 elements: 4x faster than v3,
  at ~89% of the measured memory limit.
- **Warp shuffles** (v5) remove shared memory and barriers from the tree, but since
  the tree was already negligible after coarsening, they only add ~3% here.
  They matter more in softmax, where each row is a short reduction.

## Build and run

    make
    ./bin/reduction
