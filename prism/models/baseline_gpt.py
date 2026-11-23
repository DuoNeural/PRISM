"""
EXPERIMENTAL RESEARCH CODE — NOT FOR PRODUCTION USE.

Matched Baseline: Causal Transformer (Pre-RMSNorm + RoPE + Flash/SDPA + SwiGLU)
with identical factorized tied V x d_emb embedding bottleneck.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from prism.kernels.common import RMSNorm, FactorizedTiedHead, build_rope_cache, apply_rope


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim)
        v = v.view(B, T, self.n_heads, self.head_dim)

        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        q = q.transpose(1, 2)  # [B, H, T, dh]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int):
        super().__init__()
        self.w13 = nn.Linear(d_model, 2 * ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, v = self.w13(x).chunk(2, dim=-1)
        return self.w2(F.silu(u) * v)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, ffn_dim)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class GPTBaselineLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 49152,
        d_emb: int = 256,
        d_model: int = 512,
        n_layers: int = 10,
        n_heads: int = 8,
        ffn_dim: int = 1472,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_emb = d_emb
        self.d_model = d_model
        self.head_dim = d_model // n_heads

        self.embed = nn.Embedding(vocab_size, d_emb)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.emb_up = nn.Linear(d_emb, d_model, bias=False)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_dim) for _ in range(n_layers)
        ])
        self.norm_out = RMSNorm(d_model)
        self.head = FactorizedTiedHead(d_model, d_emb, chunk_size=256)

        cos, sin = build_rope_cache(max_seq_len, self.head_dim)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_res_scales)

    def _init_res_scales(self, m):
        if isinstance(m, nn.Linear) and m is not self.emb_up:
            nn.init.normal_(m.weight, std=0.02)

    def forward_hidden(self, input_ids: torch.Tensor):
        x = self.emb_up(self.embed(input_ids))
        for blk in self.blocks:
            x = blk(x, self.rope_cos, self.rope_sin)
        return self.norm_out(x)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor = None):
        h = self.forward_hidden(input_ids)
        if targets is None:
            return self.head.forward_logits(h, self.embed.weight), {}
        ce_loss = self.head.forward_loss(h, self.embed.weight, targets)
        return ce_loss, {"ce_loss": ce_loss.detach().item(), "aux_loss": 0.0}
