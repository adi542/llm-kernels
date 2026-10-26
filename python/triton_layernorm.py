import torch
import triton
import triton.language as tl

@triton.jit
def layernorm_kernel(x_ptr, y_ptr, gamma_ptr, beta_ptr, n_cols, eps, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(axis = 0);
    cols = tl.arange(0,BLOCK_SIZE);
    mask = cols < n_cols;
    x = tl.load(x_ptr+ row*n_cols + cols,mask=mask,other=0.0);
    gamma = tl.load(gamma_ptr + cols,mask=mask,other=0.0);
    beta = tl.load(beta_ptr + cols,mask=mask,other=0.0);
    mean = tl.sum(x,axis=0) / n_cols;
    diff = tl.where(mask,x - mean,0.0);
    var = tl.sum(diff * diff,axis = 0) / n_cols;
    inv_std = tl.rsqrt(var + eps);
    tl.store(y_ptr + row*n_cols + cols,diff * inv_std * gamma + beta,mask=mask)
    

def layernorm(x, gamma, beta, eps):
    assert x.is_contiguous() and x.dim() == 2;
    rows,cols = x.shape;
    assert gamma.shape[0] == cols and beta.shape[0] == cols
    y = torch.empty_like(x);
    BLOCK = triton.next_power_of_2(cols)
    layernorm_kernel[(rows,)](x, y, gamma, beta,cols, eps, BLOCK_SIZE=BLOCK)
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

# ---------- test, on two sizes: 1024 columns, and 1000 (to test the padding!)
eps = 1e-5
for cols in [1024, 1000]:
    x     = torch.randn(4096, cols, device="cuda")
    gamma = torch.rand(cols, device="cuda") + 0.5
    beta  = torch.rand(cols, device="cuda") - 0.5
    y_triton = layernorm(x, gamma, beta, eps)
    y_ref    = torch.nn.functional.layer_norm(x, (cols,), gamma, beta, eps)
    print(cols, "columns, max difference:", (y_triton - y_ref).abs().max().item())

# ---------- race (on 1024 columns)
x     = torch.randn(4096, 1024, device="cuda")
gamma = torch.rand(1024, device="cuda") + 0.5
beta  = torch.rand(1024, device="cuda") - 0.5
bytes_moved = 2 * x.numel() * 4
for name, fn in [("triton      ", lambda: layernorm(x, gamma, beta, eps)),
                 ("F.layer_norm", lambda: torch.nn.functional.layer_norm(x, (1024,), gamma, beta, eps))]:
    ms = time_ms(fn)
    print(name, round(ms, 3), "ms", round(bytes_moved / (ms / 1000) / 1e9, 1), "GB/s")
