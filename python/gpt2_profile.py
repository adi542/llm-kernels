import torch
from torch.profiler import profile, ProfilerActivity
from gpt2_model import GPT

model = GPT.from_pretrained("gpt2").cuda().eval()
x = torch.randint(0, 50257, (8, 1024), device="cuda")      # 8 sequences of 1,024 random tokens

with torch.no_grad():
    for _ in range(2):                                     # warmup
        model(x)
    torch.cuda.synchronize()

    # ---------- total time for one forward pass
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    model(x)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end)
    print(f"forward pass: {ms:.1f} ms   ({8 * 1024 / (ms / 1000):.0f} tokens/s)")

    # ---------- the profile: which kernels take the time
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        model(x)
        torch.cuda.synchronize()

try:
    print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=15))
except Exception:
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))
