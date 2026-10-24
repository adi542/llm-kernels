# LLM kernels log
Reduction v1 (interleaved, divergent): N=16M, 2.50 ms, 26.9 GB/s (8% of 320). Overhead-bound: modulo + divergence + barriers.
Reduction v2 (strided index): fixes divergence, adds bank conflicts. v3 (sequential addressing): fixes both. Same run: v1 25.6, v2 39.1, v3 51.1 GB/s.
Reduction v4 (grid-stride, 64 numbers/thread, 1024 blocks): 221.6 GB/s (69% of 320, ~89% of vector add). 4x v3. Thread coarsening was the big win.
Reduction v5 (warp shuffles, 1 barrier/block): 221.1 GB/s, ~3% over v4. Tree was already negligible after coarsening.
Softmax v1 (one thread per row, 3 passes): 4096x1024, 1.72 ms, 19.5 GB/s effective (6% of 320). Uncoalesced, too few threads, reads input 3x.
Softmax v2 (one block per row, shared memory trees): 0.256 ms, 131.0 GB/s (41%). 6.7x v1: coalesced, all SMs busy.
