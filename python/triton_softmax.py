import torch
import triton
import triton.language as tl
from torch_ext import ext 
@triton.jit
def softmax_kernel(x_ptr, y_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(axis=0);
    cols = tl.arange(0,BLOCK_SIZE);
    mask = cols < n_cols;
    x = tl.load(x_ptr + row*n_cols+cols,mask=mask, other=-float("inf"));
    x = x - tl.max(x,axis = 0);
    num = tl.exp(x);
    den = tl.sum(num,axis = 0);
    tl.store(y_ptr + row*n_cols+cols,num/den,mask = mask);

def softmax(x):
    assert x.is_contiguous() and x.dim() == 2;
    rows,cols = x.shape;
    y = torch.empty_like(x);
    BLOCK = triton.next_power_of_2(cols)
    softmax_kernel[(rows,)](x, y, cols, BLOCK_SIZE=BLOCK)
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
x = torch.randn(4096, 1024, device="cuda")
y_triton = softmax(x)
y_torch  = torch.softmax(x, dim=1)
print("max difference:", (y_triton - y_torch).abs().max().item())

# ---------- race all three
bytes_moved = 2 * x.numel() * 4
for name, fn in [("triton   ", lambda: softmax(x)),
                 ("CUDA v3  ", lambda: ext.softmax(x)),
                 ("torch    ", lambda: torch.softmax(x, dim=1))]:
    ms = time_ms(fn)
    print(name, round(ms, 3), "ms", round(bytes_moved / (ms / 1000) / 1e9, 1), "GB/s")
    
