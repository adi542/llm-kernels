import torch
from triton_rmsnorm import rmsnorm                     # your Triton RMSNorm

def rms_torch(x, w, eps):                              # plain PyTorch: several separate kernels
    return x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + eps) * w

rms_compiled = torch.compile(rms_torch)                # the same function, compiled

def time_ms(fn, runs=50):
    for _ in range(5):
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

eps = 1e-5
x = torch.randn(4096, 1024, device="cuda")
w = torch.rand(1024, device="cuda") + 0.5

y_ref = rms_torch(x, w, eps)
print("compiled max diff:", (rms_compiled(x, w, eps) - y_ref).abs().max().item())   # first call compiles

bytes_moved = 2 * x.numel() * 4
for name, fn in [("plain PyTorch   ", lambda: rms_torch(x, w, eps)),
                 ("torch.compile   ", lambda: rms_compiled(x, w, eps)),
                 ("your Triton     ", lambda: rmsnorm(x, w, eps))]:
    ms = time_ms(fn)
    print(name, round(ms, 3), "ms", round(bytes_moved / (ms / 1000) / 1e9, 1), "GB/s")
