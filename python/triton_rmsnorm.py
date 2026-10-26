import torch
import triton
import triton.language as tl
from torch_ext import ext

@triton.jit
def rmsnorm_kernel(x_ptr, y_ptr, n_cols, w_ptr, eps, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(axis = 0);
    cols = tl.arange(0,BLOCK_SIZE);
    mask = cols < n_cols;

    x = tl.load(x_ptr + row*n_cols+cols,mask=mask,other=0.0);
    w = tl.load(w_ptr+cols,mask=mask,other=0.0);
    ms = tl.sum(x*x,axis=0) /n_cols;
    inv_rms = tl.rsqrt(ms + eps);
    tl.store(y_ptr + row*n_cols+cols,x * inv_rms*w,mask=mask);


def rmsnorm(x, w, eps):
    assert x.is_contiguous() and x.dim() == 2;
    rows,cols = x.shape;
    y = torch.empty_like(x);
    BLOCK = triton.next_power_of_2(cols)
    rmsnorm_kernel[(rows,)](x, y, cols, w, eps, BLOCK_SIZE=BLOCK)
    return y
    

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

# ---------- test
eps = 1e-5
x = torch.randn(4096, 1024, device="cuda")
w = torch.rand(1024, device="cuda") + 0.5
y_triton = rmsnorm(x, w, eps)
y_ref    = torch.nn.functional.rms_norm(x, (1024,), w, eps)
print("max difference:", (y_triton - y_ref).abs().max().item())

# ---------- race
bytes_moved = 2 * x.numel() * 4
for name, fn in [("triton    ", lambda: rmsnorm(x, w, eps)),
                 ("CUDA      ", lambda: ext.rms_norm(x, w, eps)),
                 ("F.rms_norm", lambda: torch.nn.functional.rms_norm(x, (1024,), w, eps))]:
    ms = time_ms(fn)
    print(name, round(ms, 3), "ms", round(bytes_moved / (ms / 1000) / 1e9, 1), "GB/s")
