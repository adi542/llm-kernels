import torch
import triton
import triton.language as tl
@triton.jit
def add_kernal(x_ptr,y_ptr,out_ptr,n,BLOCK_SIZE : tl.constexpr):
    pid = tl.program_id(axis = 0);
    offset = pid * BLOCK_SIZE + tl.arange(0,BLOCK_SIZE);
    mask = offset < n;
    x = tl.load(x_ptr+offset,mask=mask);
    y = tl.load(y_ptr+offset,mask=mask);
    tl.store(out_ptr+offset,x+y,mask = mask);

def add(x,y):
    out = torch.empty_like(x);
    n = x.numel();
    BLOCK = 1024;
    grid = (triton.cdiv(n,BLOCK),)
    add_kernal[grid](x,y,out,n,BLOCK_SIZE=BLOCK)
    return out

def time_ms(fn,runs=50):
    for i in range(5):
        fn()
    # gpu and cpu both are here as cpu is fast it will jsut throw the gpu function and move ahead
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record();
    for _ in range(runs):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / runs

n = 1 << 24;
x = torch.rand(n,device="cuda")
y = torch.rand(n,device="cuda")
out_mine = add(x,y);
out_torch = x + y;
print("max difference:", (out_mine - out_torch).abs().max().item());

bytes_moved = 3 * n * 4                             
ms = time_ms(lambda: add(x, y))
print("triton:", round(ms, 3), "ms", round(bytes_moved / (ms / 1000) / 1e9, 1), "GB/s")

ms = time_ms(lambda: x + y)
print("torch :", round(ms, 3), "ms", round(bytes_moved / (ms / 1000) / 1e9, 1), "GB/s")
    
