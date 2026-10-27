import torch
import triton
import triton.language as tl
from triton_rmsnorm import rmsnorm                    # your Triton RMSNorm

@triton.jit
def add_rmsnorm_kernel(a_ptr, r_ptr, h_ptr, y_ptr, w_ptr, n_cols, eps, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(axis = 0);
    cols = tl.arange(0,BLOCK_SIZE);
    mask = cols < n_cols;
    a = tl.load(a_ptr + row*n_cols + cols,mask=mask,other=0.0);
    r = tl.load(r_ptr + row*n_cols + cols,mask=mask,other=0.0);
    h = a + r;
    tl.store(h_ptr + row*n_cols + cols,h,mask=mask);
    ms = tl.sum(h*h,axis=0)/n_cols;
    inv_rms = tl.rsqrt(ms + eps);
    w = tl.load(w_ptr + cols,mask=mask,other=0.0);
    tl.store(y_ptr + row*n_cols + cols, h * inv_rms * w, mask=mask)
    

def add_rmsnorm(a, residual, w, eps):
     assert residual.is_contiguous() and residual.dim() == 2;
     assert a.is_contiguous() and a.dim() == 2;
     rows,cols = residual.shape;
     h = torch.empty_like(residual);
     y = torch.empty_like(residual);
     BLOCK = triton.next_power_of_2(cols)
     add_rmsnorm_kernel[(rows,)](a, residual, h,y, w,cols, eps, BLOCK_SIZE=BLOCK)
     return h,y

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

# ---------- test on two sizes: both outputs must match
eps = 1e-5
for cols in [1024, 1000]:
    a        = torch.randn(4096, cols, device="cuda")
    residual = torch.randn(4096, cols, device="cuda")
    w        = torch.rand(cols, device="cuda") + 0.5
    h, y = add_rmsnorm(a, residual, w, eps)
    h_ref = residual + a
    y_ref = torch.nn.functional.rms_norm(h_ref, (cols,), w, eps)
    print(cols, "columns   h diff:", (h - h_ref).abs().max().item(),
          "  y diff:", (y - y_ref).abs().max().item())

# ---------- race
a        = torch.randn(4096, 1024, device="cuda")
residual = torch.randn(4096, 1024, device="cuda")
w        = torch.rand(1024, device="cuda") + 0.5
bytes_moved = 4 * a.numel() * 4             # fused minimum: read a, read residual, write h, write y

for name, fn in [("fused triton         ", lambda: add_rmsnorm(
    a, residual, w, eps)),
                 ("add + triton rmsnorm ", lambda: rmsnorm(residual + a, w, eps)),
                 ("add + F.rms_norm     ", lambda: torch.nn.functional.rms_norm(residual + a, (1024,), w, eps))]:
    ms = time_ms(fn)
    print(name, round(ms, 3), "ms", round(bytes_moved / (ms / 1000) / 1e9, 1), "GB/s")
