import torch
from gpt2_model import GPT

def time_forward(model, x, runs=5):
    with torch.no_grad():
        for _ in range(2):
            model(x)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(runs):
            model(x)
        end.record()
        torch.cuda.synchronize()
    return start.elapsed_time(end) / runs

model = GPT.from_pretrained("gpt2").cuda().eval()
x = torch.randint(0, 50257, (8, 1024), device="cuda")
tokens = 8 * 1024

# ---------- FP32
ms32 = time_forward(model, x)
with torch.no_grad():
    logits32, _ = model(x[:1, :64])                 # a small input, kept for the accuracy check

# ---------- FP16: convert every weight to half precision
model = model.half()
ms16 = time_forward(model, x)
with torch.no_grad():
    logits16, _ = model(x[:1, :64])
logits16 = logits16.float()                         # back to FP32, to compare precisely

# ---------- results
print(f"FP32: {ms32:.1f} ms   {tokens / (ms32 / 1000):,.0f} tokens/s")
print(f"FP16: {ms16:.1f} ms   {tokens / (ms16 / 1000):,.0f} tokens/s")
print(f"speedup: {ms32 / ms16:.2f}x")

rel_err = (logits16 - logits32).abs().max().item() / logits32.abs().max().item()
same_top1 = (logits16.argmax(-1) == logits32.argmax(-1)).float().mean().item()
print(f"relative error of the logits: {rel_err:.5f}")
print(f"same most-likely next token: {same_top1 * 100:.1f}% of positions")
