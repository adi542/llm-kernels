# LLM kernels log
Reduction v1 (interleaved, divergent): N=16M, 2.50 ms, 26.9 GB/s (8% of 320). Overhead-bound: modulo + divergence + barriers.
Reduction v2 (strided index): fixes divergence, adds bank conflicts. v3 (sequential addressing): fixes both. Same run: v1 25.6, v2 39.1, v3 51.1 GB/s.
