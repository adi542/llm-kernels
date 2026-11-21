import torch
import tiktoken
from transformers import GPT2LMHeadModel
import gpt2_model
from gpt2_model import GPT

enc = tiktoken.get_encoding("gpt2")
model = GPT.from_pretrained("gpt2").cuda().eval()
hf = GPT2LMHeadModel.from_pretrained("gpt2").cuda().eval()
tokens = torch.tensor([enc.encode("Hello, I'm a language model,")], device="cuda")

# ---------- 1. logits against Hugging Face, for each attention
with torch.no_grad():
    ref = hf(tokens).logits
    for mode in ["sdpa", "mine", "naive"]:
        gpt2_model.ATTENTION = mode
        logits, _ = model(tokens)
        print(f"{mode:6s} max logit difference vs Hugging Face: {(logits - ref).abs().max().item():.2e}")

# ---------- 2. generate text with YOUR kernel
gpt2_model.ATTENTION = "mine"
torch.manual_seed(0)
idx = tokens
with torch.no_grad():
    for _ in range(30):
        logits, _ = model(idx)
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        topk_probs, topk_ids = torch.topk(probs, 50, dim=-1)
        choice = torch.multinomial(topk_probs, 1)
        idx = torch.cat([idx, torch.gather(topk_ids, -1, choice)], dim=1)
print(enc.decode(idx[0].tolist()))
