# LLM kernels log
Reduction v1 (interleaved, divergent): N=16M, 2.50 ms, 26.9 GB/s (8% of 320). Overhead-bound: modulo + divergence + barriers.
Reduction v2 (strided index): fixes divergence, adds bank conflicts. v3 (sequential addressing): fixes both. Same run: v1 25.6, v2 39.1, v3 51.1 GB/s.
Reduction v4 (grid-stride, 64 numbers/thread, 1024 blocks): 221.6 GB/s (69% of 320, ~89% of vector add). 4x v3. Thread coarsening was the big win.
Reduction v5 (warp shuffles, 1 barrier/block): 221.1 GB/s, ~3% over v4. Tree was already negligible after coarsening.
