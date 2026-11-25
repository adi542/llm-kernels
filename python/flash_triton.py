import math
import torch
import triton
import triton.language as tl
from flash_torch import naive_attention

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_Q": 32,  "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 64,  "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 64,  "BLOCK_K": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_Q": 128, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_Q": 128, "BLOCK_K": 64}, num_warps=8, num_stages=1),
    ],
    key=["N"],
)



@triton.jit
def flash_attn_kernel(q_ptr, k_ptr, v_ptr, o_ptr, N, scale,HQ, HKV, GROUP,
                      D: tl.constexpr, BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr,
                      CAUSAL: tl.constexpr):
    pid = tl.program_id(axis=0)
    bh  = tl.program_id(axis=1)
    b = bh // HQ
    hq = bh % HQ
    kv_bh = b * HKV + hq // GROUP
    offset = bh * N * D
    q_ptr += offset
    k_ptr += kv_bh * N * D
    v_ptr += kv_bh * N * D
    o_ptr += offset

    rows = pid * BLOCK_Q + tl.arange(0, BLOCK_Q)
    dims = tl.arange(0, D)
    rows_ok = rows < N
    q = tl.load(q_ptr + rows[:, None] * D + dims[None, :], mask=rows_ok[:, None], other=0.0)

    m   = tl.full((BLOCK_Q,), float("-inf"), dtype=tl.float32)
    l   = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_Q, D), dtype=tl.float32)

    # ---- no old loop here: the helper does the looping
    if CAUSAL:
        lo = pid * BLOCK_Q
        hi = tl.minimum((pid + 1) * BLOCK_Q, N)
        acc, l, m = _attn_inner(acc, l, m, q, k_ptr, v_ptr, rows, dims, 0, lo, N, scale,
                                D, BLOCK_K, CHECK_N=False, CAUSAL_MASK=False)
        acc, l, m = _attn_inner(acc, l, m, q, k_ptr, v_ptr, rows, dims, lo, hi, N, scale,
                                D, BLOCK_K, CHECK_N=True, CAUSAL_MASK=True)
    else:
        acc, l, m = _attn_inner(acc, l, m, q, k_ptr, v_ptr, rows, dims, 0, N, N, scale,
                                D, BLOCK_K, CHECK_N=True, CAUSAL_MASK=False)

    o = acc / l[:, None]
    tl.store(o_ptr + rows[:, None] * D + dims[None, :], o, mask=rows_ok[:, None])

@triton.jit
def _attn_inner(acc, l, m, q, k_ptr, v_ptr, rows, dims, start, end, N, scale,
                D: tl.constexpr, BLOCK_K: tl.constexpr,
                CHECK_N: tl.constexpr, CAUSAL_MASK: tl.constexpr):
    for j in range(start, end, BLOCK_K):
        cols = j + tl.arange(0, BLOCK_K)
        col_ok = cols < N
        k = tl.load(k_ptr + cols[:, None] * D + dims[None, :], mask=col_ok[:, None], other=0.0)
        v = tl.load(v_ptr + cols[:, None] * D + dims[None, :], mask=col_ok[:, None], other=0.0)

        s = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
        if CHECK_N:                                                  # switch 1: keys past the end
            s = tl.where(col_ok[None, :], s, float("-inf"))
        if CAUSAL_MASK:                                              # switch 2: keys in the future
            s = tl.where(cols[None, :] <= rows[:, None], s, float("-inf"))

        m_new = tl.maximum(m, tl.max(s, axis=1))
        p     = tl.exp2(s - m_new[:, None])
        alpha = tl.exp2(m - m_new)
        l     = l * alpha + tl.sum(p, axis=1)
        acc   = acc * alpha[:, None] + tl.dot(p, v, input_precision="ieee")
        m     = m_new
    return acc, l, m



def flash_attention_triton(q, k, v, causal=False):
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    *lead, N, D = q.shape
    BH = math.prod(lead)
    HQ  = q.shape[-3] if q.dim() >= 3 else 1    # query heads: Q's third-from-last size
    HKV = k.shape[-3] if k.dim() >= 3 else 1    # KV heads: K's third-from-last size
    assert HQ % HKV == 0
    o = torch.empty_like(q)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_Q"]), BH)
    flash_attn_kernel[grid](q, k, v, o, N, 1.0 / math.sqrt(D) * 1.4426950408889634,HQ, HKV, HQ // HKV,
                            D=D, CAUSAL=causal)
    return o


if __name__ == "__main__":
    torch.manual_seed(0)
    for causal in [False, True]:
        for N in [1024, 1000]:
            q = torch.randn(N, 64, device="cuda")
            k = torch.randn(N, 64, device="cuda")
            v = torch.randn(N, 64, device="cuda")
            ref = naive_attention(q, k, v, causal=causal)
            out = flash_attention_triton(q, k, v, causal=causal)
            print("causal =", causal, " ", N, "tokens   max difference:", (out - ref).abs().max().item())
        
        
        
    import torch.nn.functional as F
    B, HQ, HKV, N, D = 2, 8, 2, 1024, 64
    q = torch.randn(B, HQ, N, D, device="cuda")     # 8 query heads
    k = torch.randn(B, HKV, N, D, device="cuda")    # only 2 KV heads
    v = torch.randn(B, HKV, N, D, device="cuda")
    G = HQ // HKV
    k_rep = k.repeat_interleave(G, dim=1)           # (B, 8, N, D): each KV head copied 4 times
    v_rep = v.repeat_interleave(G, dim=1)
    for causal in [False, True]:
        ref = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=causal)
        out = flash_attention_triton(q, k, v, causal=causal)
        print("GQA 8:2  causal =", causal, "  max difference:", (out - ref).abs().max().item())
    
