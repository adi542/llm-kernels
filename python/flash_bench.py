import torch
import torch.nn.functional as F
from flash_triton import flash_attention_triton
from attention_baseline import naive_attention as naive_4d, measure

def fused(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)

def mine(q, k, v):
    return flash_attention_triton(q, k, v, causal=True)

B, H, d = 1, 12, 64
print("   N |  naive ms   MB |  fused ms   MB |   mine ms   MB")
for N in [512, 1024, 2048, 4096]:
    q = torch.randn(B, H, N, d, device="cuda")
    k = torch.randn(B, H, N, d, device="cuda")
    v = torch.randn(B, H, N, d, device="cuda")
    row = f"{N:5d}"
    for fn in [naive_4d, fused, mine]:
        ms, mb, _ = measure(fn, q, k, v)
        row += f" | {ms:8.2f} {mb:5.0f}"
    print(row)
