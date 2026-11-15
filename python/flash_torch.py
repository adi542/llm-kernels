import math
import torch

def naive_attention(q, k, v,causal=False):
    d = q.shape[-1];
    s = (q @ k.T) / math.sqrt(d);
    if(causal):
        future = torch.triu(torch.ones(N, N, dtype=torch.bool, device=q.device), diagonal=1)
        s = s.masked_fill(future,float("-inf"))
    p = torch.softmax(s,dim=-1);
    return p @ v

def flash_attention(q, k, v, BLOCK_Q=64, BLOCK_K=64,causal=False,watch_row=None):
    N,d = q.shape
    scale = 1 / math.sqrt(d)
    out = torch.empty_like(q)
    for i in range(0,N,BLOCK_Q):
        qi = q[i:i+BLOCK_Q]
        Bq = qi.shape[0]
        m = torch.full((Bq,),float("-inf"),device=q.device)
        l = torch.zeros(Bq,device=q.device)
        acc = torch.zeros(Bq,d,device=q.device)
        for j in range(0,N,BLOCK_K):
            if causal and j > i + Bq -1:
                break
            kj = k[j:j+BLOCK_K]
            vj = v[j:j+BLOCK_K]
            s = (qi @ kj.T) * scale
            if causal:                                                       
                rows = i + torch.arange(Bq, device=q.device)[:, None]            
                cols = j + torch.arange(kj.shape[0], device=q.device)[None, :]  
                s = s.masked_fill(cols > rows, float("-inf"))
            m_new = torch.maximum(m,s.max(dim=1).values)
            p = torch.exp(s - m_new[:,None])
            alpha = torch.exp(m - m_new)
            l = l * alpha + p.sum(dim=1)
            acc   = acc * alpha[:, None] + p @ vj
            m     = m_new
            if watch_row is not None and i == 0:
             print(f"K tile {j // BLOCK_K:2d}:  max = {m[watch_row].item():6.3f}   "
                    f"alpha = {alpha[watch_row].item():.4f}   l = {l[watch_row].item():8.4f}")
        out[i:i + BLOCK_Q] = acc / l[:, None] 
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    N, d = 1024, 64
    q = torch.randn(N, d, device="cuda")
    k = torch.randn(N, d, device="cuda")
    v = torch.randn(N, d, device="cuda")

    ref = naive_attention(q, k, v)
    out = flash_attention(q, k, v)
    print("max difference:", (out - ref).abs().max().item())

    flash_attention(q, k, v, watch_row=0)


