import torch
import triton
import triton.language as tl
from triton_matmul_autotune import matmul as matmul_plain     # the version without grouping

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 16}, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                  GROUP_M: tl.constexpr):
    # ---- grouped ordering: turn my number into a (row, column) of tiles
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    local = pid % num_pid_in_group
    pid_m = first_pid_m + (local % group_size_m)
    pid_n = local // group_size_m

    # ---- the rest is unchanged
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ks   = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptr + rows[:, None] * K + (k + ks)[None, :])
        b = tl.load(b_ptr + (k + ks)[:, None] * N + cols[None, :])
        acc += tl.dot(a, b, input_precision="ieee")

    tl.store(c_ptr + rows[:, None] * N + cols[None, :], acc)

def matmul(a, b):
    assert a.is_contiguous() and b.is_contiguous()
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "A's columns must match B's rows"
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = lambda meta: ((M // meta["BLOCK_M"]) * (N // meta["BLOCK_N"]),)
    matmul_kernel[grid](a, b, c, M, N, K, GROUP_M=8)
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

    c_grouped = matmul(a, b)
    c_ref     = torch.matmul(a, b)
    rel_err   = (c_grouped - c_ref).abs().max().item() / c_ref.abs().max().item()
    print("relative error:", rel_err)
    print("best config:", matmul_kernel.best_config)

    flops = 2 * N * N * N
    for name, fn in [("triton grouped", lambda: matmul(a, b)),
                     ("triton plain  ", lambda: matmul_plain(a, b)),
                     ("cuBLAS        ", lambda: torch.matmul(a, b))]:
        ms = time_ms(fn)
        print(name, round(ms, 2), "ms", round(flops / (ms / 1000) / 1e9), "GFLOPS")
