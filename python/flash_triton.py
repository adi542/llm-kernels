import math
import torch
import triton
import triton.language as tl
from flash_torch import naive_attention

@triton.jit
def flash_attn_kernel(q_ptr, k_ptr, v_ptr, o_ptr, N, scale,
                      D: tl.constexpr, BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr,CAUSAL: tl.constexpr):
    pid = tl.program_id(axis=0)
    rows = pid * BLOCK_Q + tl.arange(0,BLOCK_Q)
    dims = tl.arange(0,D)
    rows_ok = rows < N

    q = tl.load(q_ptr + rows[:,None] * D + dims[None,:],mask=rows_ok[:,None],other=0.0)
    m = tl.full((BLOCK_Q,),float("-inf"),dtype=tl.float32)
    l = tl.zeros((BLOCK_Q,),dtype=tl.float32)
    acc = tl.zeros((BLOCK_Q,D),dtype=tl.float32)

    if CAUSAL:
        hi = tl.minimum(N,(pid+1) * BLOCK_Q)
    else:
        hi = N

    for j in range(0,hi,BLOCK_K):
        cols = j + tl.arange(0,BLOCK_K)
        col_ok = cols < N
        k = tl.load(k_ptr + cols[:, None] * D + dims[None, :], mask=col_ok[:, None], other=0.0)
        v = tl.load(v_ptr + cols[:, None] * D + dims[None, :], mask=col_ok[:, None], other=0.0)

        s = tl.dot(q,tl.trans(k),input_precision="ieee") * scale
        keep = col_ok[None,:]
        if CAUSAL:
            keep = keep & (cols[None, :] <= rows[:, None])
        s = tl.where(keep, s, float("-inf"))

        m_new = tl.maximum(m,tl.max(s,axis=1))
        p = tl.exp(s - m_new[:,None])
        alpha = tl.exp(m - m_new)
        l = l * alpha + tl.sum(p,axis=1)
        acc = acc * alpha[:,None] + tl.dot(p, v, input_precision="ieee")
        m = m_new

    o = acc / l[:,None]
    tl.store(o_ptr + rows[:, None] * D + dims[None, :], o, mask=rows_ok[:, None])

def flash_attention_triton(q, k, v,causal=False, BLOCK_Q=64, BLOCK_K=32):
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    N, D = q.shape 
    o = torch.empty_like(q)
    grid = (triton.cdiv(N, BLOCK_Q),) 
    flash_attn_kernel[grid](q, k, v, o, N, 1.0 / math.sqrt(D),
                            D=D, BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K,CAUSAL=causal)
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
        
        
        

    

    
