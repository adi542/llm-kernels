from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)    # q, k, v in one matmul
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)        # mixes the heads
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)   # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu   = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp  = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size, config.n_embd),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight              # weight sharing

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"sequence too long: {T} > {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel
        config_args = {
            "gpt2":        dict(n_layer=12, n_head=12, n_embd=768),     # 124M
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),    # 350M
            "gpt2-large":  dict(n_layer=36, n_head=20, n_embd=1280),    # 774M
            "gpt2-xl":     dict(n_layer=48, n_head=25, n_embd=1600),    # 1558M
        }[model_type]
        model = GPT(GPTConfig(**config_args))
        sd = model.state_dict()

        sd_hf = GPT2LMHeadModel.from_pretrained(model_type).state_dict()
        keys_hf = [k for k in sd_hf if not k.endswith((".attn.masked_bias", ".attn.bias"))]
        # OpenAI stored these four as "Conv1D" weights, transposed compared with nn.Linear
        transposed = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]
        assert len(keys_hf) == len(sd), f"mismatched keys: {len(keys_hf)} != {len(sd)}"
        for k in keys_hf:
            with torch.no_grad():
                if any(k.endswith(w) for w in transposed):
                    assert sd_hf[k].shape[::-1] == sd[k].shape
                    sd[k].copy_(sd_hf[k].t())
                else:
                    assert sd_hf[k].shape == sd[k].shape
                    sd[k].copy_(sd_hf[k])
        return model
