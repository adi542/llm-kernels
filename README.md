# LLM kernels in CUDA

The memory-bound kernels used inside transformers, written from scratch in CUDA
on an NVIDIA Tesla T4 and measured in GB/s against the T4's memory bandwidth
(320 GB/s spec, ~250 GB/s achievable in practice). Every version is checked for
correctness against a CPU reference, and the final kernels are callable from PyTorch.

Why GB/s and not GFLOPS: these kernels do only a few operations per number while
reading and writing every number (arithmetic intensity ~0.5 to 3 FLOPs/byte, far below
the T4's ridge point of ~18 to 25). They are limited by memory, not math, so they are
measured against the memory limit.

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

## 3. RMSNorm and LayerNorm

4096 rows x 1024 columns, FP32, one block per row, warp-shuffle block reductions
(shared in `reduce.cuh`). Checked against double-precision CPU references.

| Kernel | Reductions per row | Passes over each row | GB/s |
| --- | --- | --- | --- |
| RMSNorm (Llama-style) | 1 (sum of squares) | 2 | ~196 |
| LayerNorm (GPT-2-style) | 2 (mean, then variance) | 3 | ~164 |

What I learned:
- **RMSNorm is ~16% faster than LayerNorm** in the same run: one fewer reduction and one
  fewer pass. That cost difference is part of why modern LLMs switched to RMSNorm.
- **LayerNorm's variance uses two passes** (mean first, then squared distances from the
  mean) instead of the one-pass shortcut E[x^2] - mean^2, which loses precision through
  catastrophic cancellation.
- **`rsqrtf`** computes 1/sqrt in one instruction, so each output is a multiplication
  instead of a division.

## 4. Calling the kernels from PyTorch

`python/torch_ext.py` compiles softmax v3 and RMSNorm as a PyTorch extension
(`torch.utils.cpp_extension.load_inline`): a C++ wrapper validates the tensors
(device, dtype, shape, contiguity), allocates the output, and launches the kernel on
the tensors' raw GPU pointers. Outputs match PyTorch to within ~1e-6.

| Kernel | Mine | PyTorch | Notes |
| --- | --- | --- | --- |
| Softmax | ~160 GB/s | `torch.softmax`: ~217 GB/s | |
| RMSNorm | ~185 GB/s | plain PyTorch ops: ~65 GB/s | **2.9x faster: fusion** |
| RMSNorm | ~185 GB/s | `F.rms_norm`: ~234 GB/s | PyTorch's own fused kernel |

What I learned:
- **Kernel fusion:** written as separate PyTorch operations, RMSNorm runs as several
  kernels that each read and write full tensors in memory (~6 trips of 16 MB). One
  fused kernel reads the input once and writes the output once: 2.9x faster.
- **The gap to PyTorch's best kernels** comes from two things mine don't do yet: loading
  each row into registers once (mine reads each row two or three times) and wider
  vectorized memory loads.

## Build and run

    make
    ./bin/reduction
    ./bin/softmax
    ./bin/norms
    python python/torch_ext.py

Requires the CUDA toolkit, PyTorch with CUDA, and an NVIDIA GPU. Set `-arch` in the
Makefile for your GPU (`sm_75` for a T4).
