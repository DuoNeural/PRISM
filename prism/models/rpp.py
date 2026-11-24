"""
EXPERIMENTAL RESEARCH CODE — NOT FOR PRODUCTION USE.

Candidate 3: RPP — RevPort-POVM Hybrid Architecture
Combines:
1. Bit-exact FP32 Reversible Port-Hamiltonian conjugate streams (q, p in R^d_model)
   eliminating BF16 drift and matching full d_model=512 capacity per stream.
2. Sub-block F(p): Exact Causal Sliding-Window RoPE Attention (W=64) via chunked-64
   overlapping Flash/SDPA (diagonal j=i always valid, zero NaN rows) + compact SwiGLU.
3. Sub-block G(q_next): Dual-Timescale Unital OU Triton Scan (50% fast poles tau_f in [1,32],
   50% slow poles tau_s in [64,1024] strictly ordered in FP32 continuous log-rate space,
   exact OU drive variance sqrt(1 - a^2)) + Port-Hamiltonian J-R coupling.
4. Chunked FP32 Complex Born-Log POVM Output Head tied to shared V x 256 embedding.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from prism.kernels.common import RMSNorm, FactorizedTiedHead, build_rope_cache, apply_rope
from prism.kernels.complex_scan import complex_linear_scan
from prism.kernels.born_povm import BornLogPOVMHead
from prism.models.ldg import chunked_wy_delta_scan, lorentz_origin_logmap


def sliding_window_causal_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: int = 64) -> torch.Tensor:
    """
    Exact causal sliding-window attention (0 <= i - j < window) using chunked SDPA.
    q, k, v: [B, T, H, dh]. Zero Python token loops!
    """
    B, T, H, dh = q.shape
    if window is None or window <= 0:
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )
        return out.transpose(1, 2).contiguous()
    if T <= window:
        i_idx = torch.arange(T, device=q.device).unsqueeze(1)
        j_idx = torch.arange(T, device=q.device).unsqueeze(0)
        mask = (j_idx <= i_idx) & ((i_idx - j_idx) < window)
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask)
        return out.transpose(1, 2).contiguous()

    pad_t = (window - (T % window)) % window
    if pad_t > 0:
        q = F.pad(q, (0, 0, 0, 0, 0, pad_t))
        k = F.pad(k, (0, 0, 0, 0, 0, pad_t))
        v = F.pad(v, (0, 0, 0, 0, 0, pad_t))
    T_pad = T + pad_t
    NC = T_pad // window
    W = window
    # Pad previous chunk with zeros on the left for chunk 0
    pad_k = torch.zeros(B, W, H, dh, device=k.device, dtype=k.dtype)
    pad_v = torch.zeros(B, W, H, dh, device=v.device, dtype=v.dtype)
    k_pad = torch.cat([pad_k, k], dim=1)  # [B, T_pad+W, H, dh]
    v_pad = torch.cat([pad_v, v], dim=1)

    # Gather [prev_W, curr_W] KV blocks for each query chunk c in 0..NC-1
    k_chunks = []
    v_chunks = []
    for c in range(NC):
        start = c * W
        k_chunks.append(k_pad[:, start:start + 2 * W])  # [B, 2W, H, dh]
        v_chunks.append(v_pad[:, start:start + 2 * W])
    k_win = torch.stack(k_chunks, dim=1).reshape(B * NC, 2 * W, H, dh).transpose(1, 2)  # [B*NC, H, 2W, dh]
    v_win = torch.stack(v_chunks, dim=1).reshape(B * NC, 2 * W, H, dh).transpose(1, 2)
    q_win = q.view(B, NC, W, H, dh).reshape(B * NC, W, H, dh).transpose(1, 2)          # [B*NC, H, W, dh]

    # Query i in [0..W-1] maps to global pos W + i in the 2W KV window [0..2W-1]
    qi = torch.arange(W, device=q.device).unsqueeze(1) + W
    kj = torch.arange(2 * W, device=q.device).unsqueeze(0)
    win_mask = (kj <= qi) & ((qi - kj) < W)  # Every query row sees W valid keys (self j=qi always True!)

    out_win = F.scaled_dot_product_attention(q_win, k_win, v_win, attn_mask=win_mask)  # [B*NC, H, W, dh]
    return out_win.transpose(1, 2).contiguous().view(B, T_pad, H, dh)[:, :T].contiguous()


class WindowAttnSubBlock(nn.Module):
    """F(p): Causal Sliding-Window RoPE Attention (W=64) + compact SwiGLU."""
    def __init__(self, d_model: int, n_heads: int = 8, window: int = 64, ffn_dim: int = 768, scale_init: float = 0.25):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window = window
        self.scale_init = scale_init

        self.norm1 = RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.ffn_dim = ffn_dim
        if ffn_dim > 0:
            self.norm2 = RMSNorm(d_model)
            self.w13 = nn.Linear(d_model, 2 * ffn_dim, bias=False)
            self.w2 = nn.Linear(ffn_dim, d_model, bias=False)

    def forward(self, p: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> torch.Tensor:
        B, T, D = p.shape
        h = self.norm1(p)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = apply_rope(q.view(B, T, self.n_heads, self.head_dim), rope_cos, rope_sin)
        k = apply_rope(k.view(B, T, self.n_heads, self.head_dim), rope_cos, rope_sin)
        v = v.view(B, T, self.n_heads, self.head_dim)
        out = sliding_window_causal_sdpa(q, k, v, window=self.window).view(B, T, D)
        delta = self.o_proj(out)
        if self.ffn_dim <= 0:
            return delta

        h2 = self.norm2(p + delta)
        u, g = self.w13(h2).chunk(2, dim=-1)
        return delta + self.w2(F.silu(u) * g)


class DualTimescaleScanSubBlock(nn.Module):
    """G(q): Dual-Timescale Unital OU Triton Scan (Fast/Slow ordered poles) + SwiGLU."""
    def __init__(self, d_model: int, n_heads: int = 8, ffn_dim: int = 768, d_scan: int = None):
        super().__init__()
        self.d_model = d_model
        self.d_scan = d_model if d_scan is None else d_scan
        self.n_heads = n_heads
        self.head_dim = self.d_scan // n_heads
        self.n_osc = self.head_dim // 2
        self.h_half = n_heads // 2  # half fast heads, half slow heads

        self.norm1 = RMSNorm(d_model)
        self.drive_proj = nn.Linear(d_model, self.d_scan, bias=False)
        self.gate_proj = nn.Linear(d_model, self.d_scan, bias=False)
        self.o_proj = nn.Linear(self.d_scan, d_model, bias=False)

        # Strictly ordered disjoint decay rates:
        # Fast: gamma in [1/32, 1.0] (tau_f in [1, 32])
        # Slow: gamma in [1/1024, 1/64] (tau_s in [64, 1024])
        self.theta_fast = nn.Parameter(torch.zeros(1, 1, self.h_half, self.n_osc))
        self.theta_slow = nn.Parameter(torch.zeros(1, 1, self.h_half, self.n_osc))
        self.omega_base = nn.Parameter(torch.linspace(0.05, 1.2, n_heads * self.n_osc).view(1, 1, n_heads, self.n_osc))

        self.norm2 = RMSNorm(d_model)
        self.w13 = nn.Linear(d_model, 2 * ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, d_model, bias=False)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        B, T, _ = q.shape
        H, M = self.n_heads, self.n_osc
        h = self.norm1(q)
        u_pair = F.silu(self.drive_proj(h)).view(B, T, H, self.head_dim).float()
        u_r, u_i = u_pair.chunk(2, dim=-1)

        # Ordered disjoint log-rates in FP32
        gamma_fast = (1.0 / 32.0) + (1.0 - 1.0 / 32.0) * torch.sigmoid(self.theta_fast.float())
        gamma_slow = (1.0 / 1024.0) + (1.0 / 64.0 - 1.0 / 1024.0) * torch.sigmoid(self.theta_slow.float())
        gamma = torch.cat([gamma_fast, gamma_slow], dim=2).expand(B, T, H, M)

        log_decay = -gamma
        phase = (2.0 * torch.atan(0.5 * self.omega_base.float())).expand(B, T, H, M)

        # Exact OU / ZOH drive scaling sqrt(1 - a^2) = sqrt(-expm1(2 * log_decay))
        ou_gain = torch.sqrt((-torch.expm1(2.0 * log_decay)).clamp(min=1e-6))
        s_r, s_i = complex_linear_scan(log_decay, phase, u_r * ou_gain, u_i * ou_gain)
        s_out = torch.cat([s_r, s_i], dim=-1).view(B, T, self.d_scan).to(q.dtype)

        delta = self.o_proj(F.silu(self.gate_proj(h)) * s_out)
        h2 = self.norm2(q + delta)
        u1, g1 = self.w13(h2).chunk(2, dim=-1)
        return delta + self.w2(F.silu(u1) * g1)


class DeltaSubBlock(nn.Module):
    """G_alt(q): Chunk-64 WY Gated Delta-Rule + Lorentz Log-Map + SwiGLU for Reversible Port Streams."""
    def __init__(self, d_model: int, n_heads: int = 8, chunk_size: int = 64, ffn_dim: int = 416):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.chunk_size = chunk_size

        self.norm1 = RMSNorm(d_model)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.conv = nn.Conv1d(3 * d_model, 3 * d_model, kernel_size=4, padding=3, groups=3 * d_model, bias=False)
        self.log_c = nn.Parameter(torch.tensor(-0.5))
        self.q_lorentz_norm = RMSNorm(self.head_dim)
        self.k_lorentz_norm = RMSNorm(self.head_dim)
        self.ab_proj = nn.Linear(d_model, 2 * n_heads, bias=True)
        with torch.no_grad():
            self.ab_proj.bias[:n_heads].copy_(torch.linspace(1.6, 5.2, n_heads))
            self.ab_proj.bias[n_heads:].fill_(-0.7)

        self.out_gate = nn.Linear(d_model, d_model, bias=False)
        self.out_norm = RMSNorm(self.head_dim)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.norm2 = RMSNorm(d_model)
        self.w13 = nn.Linear(d_model, 2 * ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, d_model, bias=False)

    def forward(self, q_in: torch.Tensor, rope_cos: torch.Tensor = None, rope_sin: torch.Tensor = None) -> torch.Tensor:
        B, T, D = q_in.shape
        h = self.norm1(q_in)
        qkv = F.silu(self.conv(self.qkv_proj(h).transpose(1, 2))[..., :T].transpose(1, 2))
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim)
        v = v.view(B, T, self.n_heads, self.head_dim)

        kappa = 0.05 + 9.95 * torch.sigmoid(self.log_c)
        q = lorentz_origin_logmap(self.q_lorentz_norm(q), kappa) / math.sqrt(self.head_dim)
        k = lorentz_origin_logmap(self.k_lorentz_norm(k), kappa)
        if rope_cos is not None:
            q = apply_rope(q, rope_cos, rope_sin)
            k = apply_rope(k, rope_cos, rope_sin)
        k_hat = k.float() / torch.sqrt((k.float() ** 2).sum(dim=-1, keepdim=True) + 1e-6)

        a_logit, b_logit = self.ab_proj(h).float().chunk(2, dim=-1)
        alpha = 0.01 + 0.989 * torch.sigmoid(a_logit)
        beta = 1.5 * torch.sigmoid(b_logit)
        mem, _ = chunked_wy_delta_scan(q, k_hat, v, alpha, beta, chunk_size=self.chunk_size)
        mem = self.out_norm(mem).reshape(B, T, D)
        delta = self.o_proj(F.silu(self.out_gate(h)) * mem)

        h2 = self.norm2(q_in + delta)
        u1, g1 = self.w13(h2).chunk(2, dim=-1)
        return delta + self.w2(F.silu(u1) * g1)


class _ReversiblePortBlockFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q_in, p_in, f_blk, g_blk, rope_cos, rope_sin, mix_theta, *params):
        ctx.f_blk = f_blk
        ctx.g_blk = g_blk
        ctx.has_mix = mix_theta is not None
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=q_in.is_cuda):
                f_out = f_blk(p_in, rope_cos, rope_sin).float()
            q_mid = q_in.float() + f_out
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=q_in.is_cuda):
                g_out = (g_blk(q_mid, rope_cos, rope_sin) if isinstance(g_blk, DeltaSubBlock) else g_blk(q_mid)).float()
            p_mid = p_in.float() + g_out
            if ctx.has_mix:
                c = torch.cos(mix_theta.float())
                s = torch.sin(mix_theta.float())
                q_out = c * q_mid - s * p_mid
                p_out = s * q_mid + c * p_mid
                ctx.save_for_backward(q_out.detach(), p_out.detach(), rope_cos, rope_sin, mix_theta)
            else:
                q_out, p_out = q_mid, p_mid
                ctx.save_for_backward(q_out.detach(), p_out.detach(), rope_cos, rope_sin)

        return q_out, p_out

    @staticmethod
    def backward(ctx, dq_out, dp_out):
        if ctx.has_mix:
            q_out, p_out, rope_cos, rope_sin, mix_theta = ctx.saved_tensors
            with torch.no_grad():
                c = torch.cos(mix_theta.float())
                s = torch.sin(mix_theta.float())
                q_mid = c * q_out + s * p_out
                p_mid = -s * q_out + c * p_out
            with torch.enable_grad():
                q_r = q_mid.detach().requires_grad_(True)
                p_r = p_mid.detach().requires_grad_(True)
                c_g = torch.cos(mix_theta.float())
                s_g = torch.sin(mix_theta.float())
                qo_g = c_g * q_r - s_g * p_r
                po_g = s_g * q_r + c_g * p_r
                dq_mid_rot, dp_mid_rot, d_theta = torch.autograd.grad(
                    (qo_g, po_g), (q_r, p_r, mix_theta), (dq_out, dp_out)
                )
        else:
            q_out, p_out, rope_cos, rope_sin = ctx.saved_tensors
            q_mid, p_mid = q_out, p_out
            dq_mid_rot, dp_mid_rot, d_theta = dq_out, dp_out, None

        f_blk, g_blk = ctx.f_blk, ctx.g_blk

        with torch.enable_grad():
            q_tmp = q_mid.detach().requires_grad_(True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=q_mid.is_cuda):
                g_out = (g_blk(q_tmp, rope_cos, rope_sin) if isinstance(g_blk, DeltaSubBlock) else g_blk(q_tmp)).float()
            p_in = (p_mid.detach() - g_out.detach())

            g_params = list(g_blk.parameters())
            g_grads = torch.autograd.grad(g_out, [q_tmp] + g_params, dp_mid_rot, allow_unused=True)
            dq_mid = dq_mid_rot + g_grads[0]
            dg_params = g_grads[1:]

            p_tmp = p_in.detach().requires_grad_(True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=p_in.is_cuda):
                f_out = f_blk(p_tmp, rope_cos, rope_sin).float()
            f_params = list(f_blk.parameters())
            f_grads = torch.autograd.grad(f_out, [p_tmp] + f_params, dq_mid, allow_unused=True)
            dp_in = dp_mid_rot + f_grads[0]
            df_params = f_grads[1:]

        return (dq_mid, dp_in, None, None, None, None, d_theta) + tuple(df_params) + tuple(dg_params)


class RPPLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 49152,
        d_emb: int = 256,
        d_model: int = 512,
        n_layers: int = 10,
        n_heads: int = 8,
        window: int = 64,
        global_layers: tuple[int, ...] = (),
        ffn_dim: int = 480,
        f_ffn_dim: int = None,
        g_ffn_dim: int = None,
        d_scan: int = None,
        use_symp_mix: bool = False,
        symp_mix_init: float = 0.0,
        max_seq_len: int = 1024,
        use_hybrid_delta: bool = False,
        use_povm_head: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_emb = d_emb
        self.d_model = d_model
        self.global_layers = tuple(sorted(set(global_layers or ())))
        self.use_symp_mix = use_symp_mix
        self.use_hybrid_delta = use_hybrid_delta
        self.use_povm_head = use_povm_head

        f_ffn = ffn_dim if f_ffn_dim is None else f_ffn_dim
        g_ffn = ffn_dim if g_ffn_dim is None else g_ffn_dim

        self.embed = nn.Embedding(vocab_size, d_emb)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.emb_up_q = nn.Linear(d_emb, d_model, bias=False)
        self.emb_up_p = nn.Linear(d_emb, d_model, bias=False)

        global_set = set(self.global_layers)
        self.f_blocks = nn.ModuleList([
            WindowAttnSubBlock(d_model, n_heads=n_heads, window=(0 if i in global_set else window), ffn_dim=f_ffn)
            for i in range(n_layers)
        ])
        self.g_blocks = nn.ModuleList([
            DeltaSubBlock(d_model, n_heads=n_heads, chunk_size=window, ffn_dim=320)
            if (use_hybrid_delta and i % 2 == 0)
            else DualTimescaleScanSubBlock(d_model, n_heads=n_heads, ffn_dim=g_ffn, d_scan=d_scan)
            for i in range(n_layers)
        ])
        self.mix_thetas = (
            nn.ParameterList([nn.Parameter(torch.full((d_model,), float(symp_mix_init))) for _ in range(n_layers)])
            if use_symp_mix else None
        )
        self.norm_out = RMSNorm(d_model)
        if use_povm_head:
            self.head = BornLogPOVMHead(d_model, d_emb, init_scale=12.0, eps=1e-6, chunk_size=128, mode="amplitude")
        else:
            self.head = FactorizedTiedHead(d_model, d_emb, chunk_size=256)

        cos, sin = build_rope_cache(max_seq_len, d_model // n_heads)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Scale reversible output projections by 0.1 / sqrt(2 * n_layers) for exact precision stability
        res_std = 0.02 / math.sqrt(2 * n_layers)
        for fb, gb in zip(self.f_blocks, self.g_blocks):
            nn.init.normal_(fb.o_proj.weight, std=res_std)
            if hasattr(fb, "w2"):
                nn.init.normal_(fb.w2.weight, std=res_std)
            nn.init.normal_(gb.o_proj.weight, std=res_std)
            if hasattr(gb, "w2"):
                nn.init.normal_(gb.w2.weight, std=res_std)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) and m not in (self.emb_up_q, self.emb_up_p):
            nn.init.normal_(m.weight, std=0.02)

    def forward_streams(self, input_ids: torch.Tensor):
        emb = self.embed(input_ids)
        q = self.emb_up_q(emb).float()
        p = self.emb_up_p(emb).float()
        for idx, (fb, gb) in enumerate(zip(self.f_blocks, self.g_blocks)):
            th = self.mix_thetas[idx] if self.mix_thetas is not None else None
            params = list(fb.parameters()) + list(gb.parameters())
            q, p = _ReversiblePortBlockFn.apply(q, p, fb, gb, self.rope_cos, self.rope_sin, th, *params)
        return q, p

    def forward_hidden(self, input_ids: torch.Tensor):
        q, p = self.forward_streams(input_ids)
        h = self.norm_out(0.5 * (q + p))
        return h

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor = None):
        q, p = self.forward_streams(input_ids)
        h = self.norm_out(0.5 * (q + p))
        with torch.no_grad():
            stream_q_rms = q.pow(2).mean().sqrt().item()
            stream_p_rms = p.pow(2).mean().sqrt().item()
            tau_scale = self.head.scale.item() if self.use_povm_head else 0.0
        m = {
            "stream_q_rms": stream_q_rms,
            "stream_p_rms": stream_p_rms,
            "povm_scale": tau_scale,
            "aux_loss": 0.0,
        }
        if targets is None:
            return self.head.forward_logits(h, self.embed.weight), m
        ce_loss = self.head.forward_loss(h, self.embed.weight, targets)
        m["ce_loss"] = ce_loss.detach().item()
        return ce_loss, m
