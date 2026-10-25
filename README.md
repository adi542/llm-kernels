# LLM kernels in CUDA

The memory-bound kernels used inside transformers, written from scratch in CUDA
on an NVIDIA Tesla T4 and measured in GB/s against the T4's memory bandwidth
(320 GB/s spec, ~250 GB/s achievable in practice). Every version is checked for
correctness against a CPU reference.

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

## 2. Softmax (row-wise, numerically safe)

4096 rows x 1024 columns, FP32. Checked against a double-precision CPU softmax.
GB/s is effective bandwidth: the minimum traffic (read input once + write output once)
divided by time.

| Version | Idea | GB/s |
| --- | --- | --- |
| v1 | One thread per row, three passes | ~21 |
| v2 | One block per row, shared memory trees for max and sum | ~138 |
| v3 | One block per row, warp-shuffle block reductions | ~176 |
| v4 | Online softmax: max and sum in one pass, then the write pass | ~147 |

What I learned:
- **One block per row** (v2) fixed three problems at once: coalesced reads (neighbouring
  threads read neighbouring elements), enough threads to fill all 40 SMs, and short
  per-thread work instead of a 1024-step chain. 6.6x faster than v1.
- **Warp shuffles mattered much more here than in reduction** (+29% vs +3%): each row is
  short and needs two reductions, so the trees were a large share of the work.
- **Online softmax was slower standalone** on the T4. Rows fit in cache, so the pass it
  saves was cheap, while its update adds two exponentials and a dependency chain per
  element. Its real value is enabling fusion: in Flash Attention the scores are never
  stored, so softmax must be computed in a single streaming pass.
- **Testing lesson:** my first checker reported NaN outputs as "correct" (every
  comparison with NaN is false). It now checks for NaN explicitly.

## Build and run

    make
    ./bin/reduction
    ./bin/softmax

Requires the CUDA toolkit and an NVIDIA GPU. Set `-arch` in the Makefile for your GPU
(`sm_75` for a T4).
