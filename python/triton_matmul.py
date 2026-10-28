import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ks   = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptr + rows[:, None] * K + (k + ks)[None, :])
        b = tl.load(b_ptr + (k + ks)[:, None] * N + cols[None, :])
        acc += tl.dot(a, b, input_precision="ieee")

    tl.store(c_ptr + rows[:, None] * N + cols[None, :], acc)

def matmul(a, b, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,num_warps=4):
    assert a.is_contiguous() and b.is_contiguous()
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "A's columns must match B's rows"
    assert M % BLOCK_M == 0 and N % BLOCK_N == 0 and K % BLOCK_K == 0, "sizes must divide by the tiles"
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = (M // BLOCK_M, N // BLOCK_N)
    matmul_kernel[grid](a, b, c, M, N, K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                        num_warps=num_warps)
    return c

def time_ms(fn, runs=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(runs):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / runs

if __name__ == "__main__":
    N = 4096
    a = torch.randn(N, N, device="cuda")
    b = torch.randn(N, N, device="cuda")

    # ---------- test against cuBLAS (torch.matmul)
    c_triton = matmul(a, b)
    c_ref    = torch.matmul(a, b)
    max_diff = (c_triton - c_ref).abs().max().item()
    rel_err  = max_diff / c_ref.abs().max().item()
    print("max difference:", max_diff, "  relative:", rel_err)

    # ---------- race, in GFLOPS
    flops = 2 * N * N * N
    for name, fn in [("triton", lambda: matmul(a, b)),
                     ("cuBLAS", lambda: torch.matmul(a, b))]:
        ms = time_ms(fn)
        print(name, round(ms, 2), "ms", round(flops / (ms / 1000) / 1e9), "GFLOPS")
    # print("triton row 0:", c_triton[0, :4].tolist())
    # print("cuBLAS row 0:", c_ref[0, :4].tolist())
    # exact = (a[:2].double() @ b.double())[:, :4]              # 2 rows, in double precision
    # print("double row 0:", exact[0].tolist())
    # print("triton vs double:", (c_triton[:2, :4].double() - exact).abs().max().item())

    print("\nBLOCK_M BLOCK_N BLOCK_K warps -> GFLOPS")
    for bm, bn, bk, nw in [(32, 32, 32, 4), (64, 64, 32, 4), (128, 64, 32, 4),
                           (64, 128, 32, 4), (128, 128, 32, 4), (128, 128, 32, 8),
                           (128, 128, 16, 8)]:
        try:
            ms = time_ms(lambda: matmul(a, b, bm, bn, bk, nw))
            print(bm, bn, bk, nw, "->", round(flops / (ms / 1000) / 1e9))
        except Exception as e:
            print(bm, bn, bk, nw, "-> failed:", str(e)[:80])
    
