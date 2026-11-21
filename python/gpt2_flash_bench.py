import torch
import gpt2_model
from gpt2_model import GPT

model = GPT.from_pretrained("gpt2").cuda().eval()
x = torch.randint(0, 50257, (8, 1024), device="cuda")      # 8 sequences of 1,024 tokens
tokens = 8 * 1024

print("attention |  forward ms |  tokens/s | peak memory MB")
with torch.no_grad():
    for mode in ["sdpa", "mine", "naive"]:
        gpt2_model.ATTENTION = mode
        for _ in range(2):                                  # warmup (and autotuning, for "mine")
            model(x)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(3):
            model(x)
        end.record()
        torch.cuda.synchronize()

        ms = start.elapsed_time(end) / 3
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        print(f"{mode:9s} | {ms:10.1f} | {tokens / (ms / 1000):9,.0f} | {peak_mb:8.0f}")
