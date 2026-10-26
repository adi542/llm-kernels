# LLM kernels log
Reduction v1 (interleaved, divergent): N=16M, 2.50 ms, 26.9 GB/s (8% of 320). Overhead-bound: modulo + divergence + barriers.
Reduction v2 (strided index): fixes divergence, adds bank conflicts. v3 (sequential addressing): fixes both. Same run: v1 25.6, v2 39.1, v3 51.1 GB/s.
Reduction v4 (grid-stride, 64 numbers/thread, 1024 blocks): 221.6 GB/s (69% of 320, ~89% of vector add). 4x v3. Thread coarsening was the big win.
Reduction v5 (warp shuffles, 1 barrier/block): 221.1 GB/s, ~3% over v4. Tree was already negligible after coarsening.
Softmax v1 (one thread per row, 3 passes): 4096x1024, 1.72 ms, 19.5 GB/s effective (6% of 320). Uncoalesced, too few threads, reads input 3x.
Softmax v2 (one block per row, shared memory trees): 0.256 ms, 131.0 GB/s (41%). 6.7x v1: coalesced, all SMs busy.
Softmax v3 (warp shuffles, 4 barriers/row instead of ~18): 0.195 ms, 172.0 GB/s (54%). +29% over v2: trees matter with short rows.
Softmax v4 (online softmax, 2 passes): 146.7 GB/s vs v3 175.7 in the same run. Slower standalone: rows fit in cache so the saved pass was cheap, and the update adds 2 exps + a dependency chain per element. Its value is enabling fusion (Flash Attention). Also fixed the checker: NaN used to pass as 'correct'.
RMSNorm (one block per row, 1 block reduction, rsqrtf): 4096x1024, 0.172 ms, 194.6 GB/s (61%). Faster than softmax v3: 1 reduction, no exps.
LayerNorm (one block per row, 2 reductions: mean then variance, 3 passes): 0.204 ms, 164.4 GB/s (51%). ~16% slower than RMSNorm (195.8 GB/s, same run): extra reduction + extra pass.
PyTorch extension (load_inline): softmax v3 160 GB/s vs torch.softmax 217. RMSNorm 185 GB/s vs plain PyTorch 65 (2.9x, fusion) vs F.rms_norm 234. Gap to PyTorch's fused kernels: they read each row once (registers) + wider loads.
Triton vector add: 244.7 GB/s vs torch 245.2 (same run), CUDA ~248 earlier. ~10 lines of Python reach the memory limit.
Triton fused softmax (one program per row, row loaded once): 232.2 GB/s vs torch 215.6 vs CUDA v3 139.7 (same run). ~93% of practical limit.
Triton RMSNorm (row loaded once): 233.7 GB/s = F.rms_norm 233.6, CUDA 173.1 (same run). Among fused kernels, reading the row once is the difference.
