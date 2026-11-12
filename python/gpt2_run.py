import torch
import tiktoken
from transformers import GPT2LMHeadModel
from gpt2_model import GPT

torch.manual_seed(0)
enc = tiktoken.get_encoding("gpt2")

# ---------- load your model with OpenAI's weights
model = GPT.from_pretrained("gpt2").cuda().eval()

# ---------- 1. correctness: same input into your model and Hugging Face's
tokens = torch.tensor([enc.encode("Hello, I'm a language model,")], device="cuda")
hf = GPT2LMHeadModel.from_pretrained("gpt2").cuda().eval()
with torch.no_grad():
    mine, _ = model(tokens)
    ref = hf(tokens).logits
print("max logit difference vs Hugging Face:", (mine - ref).abs().max().item())

# ---------- 2. generate 30 new tokens, one at a time
idx = tokens
with torch.no_grad():
    for _ in range(30):
        logits, _ = model(idx)                                   # (1, T, 50257)
        probs = torch.softmax(logits[:, -1, :], dim=-1)          # last position only
        topk_probs, topk_ids = torch.topk(probs, 50, dim=-1)     # keep the 50 most likely tokens
        choice = torch.multinomial(topk_probs, 1)                # pick one, at random by probability
        next_id = torch.gather(topk_ids, -1, choice)             # turn the choice back into a token ID
        idx = torch.cat([idx, next_id], dim=1)                   # append it
print(enc.decode(idx[0].tolist()))
