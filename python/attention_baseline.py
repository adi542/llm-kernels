import math
import torch
import torch.nn.functional as F

def naive_attention(q,k,v):
    N,d = q.shape[-2],q.shape[-1]
    s = (q @ k.transpose(-2,-1)) / math.sqrt(d)
    future = torch.triu(torch.ones(N, N, dtype=torch.bool, device=q.device), diagonal=1)
    s = s.masked_fill(future,float("-inf"))
    p = torch.softmax(s,dim=-1)
    return p @ v

def fused_attention(q, k, v):                                    # PyTorch's own, for comparison
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    
def measure(fn,q,k,v,runs=50):
    torch.cuda.synchronize();
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    out = fn(q,k,v)
    torch.cuda.synchronize()
    extra_mb = (torch.cuda.max_memory_allocated() - before) / 1e6
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(runs):
        fn(q, k, v)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / runs, extra_mb, out

B, H, d = 1, 12, 64                                              # one sequence, GPT-2's 12 heads of 64
print("   N    naive ms   naive MB    fused ms   fused MB   max diff")
for N in [512, 1024, 2048, 4096]:
    q = torch.randn(B, H, N, d, device="cuda")
    k = torch.randn(B, H, N, d, device="cuda")
    v = torch.randn(B, H, N, d, device="cuda")
    ms_n, mb_n, out_n = measure(naive_attention, q, k, v)
    ms_f, mb_f, out_f = measure(fused_attention, q, k, v)
    diff = (out_n - out_f).abs().max().item()
    print(f"{N:5d}   {ms_n:8.2f}   {mb_n:8.0f}   {ms_f:8.2f}   {mb_f:8.0f}   {diff:.1e}")



    
    
    
