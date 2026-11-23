"""
EXPERIMENTAL RESEARCH CODE — NOT FOR PRODUCTION USE.

Candidate 1: LDG — Lorentz-Delta-TSSP GENERIC Architecture
Combines:
1. Chunked-64 WY Gated Delta-Rule Associative State (exact FP32 triangular solve,
   log-space head-isotropic decay alpha_t in (0, 1), bounded beta_t in (0, 1.5),
   state S_t in R^{d_v x d_k} matching key/value axes cleanly).
2. Cancellation-free Lorentz Origin Log-Map radial hierarchy warping on q/k
   with learnable bounded block curvature kappa_l in [0.05, 10.0], plus RoPE & conv4.
3. FFN-free CliffordNet Geometric Product + Projection-Sandwiched FP32 GENERIC mixer
   with Post-RMSNorm and gated residual branch to prevent quadratic blowup.
4. Collapse-proof TSSP auxiliary predictor over chunk-pooled hidden states
   (normalized cosine + VICReg variance hinge, warmed up cleanly).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from prism.kernels.common import (
    RMSNorm,
    FactorizedTiedHead,
    build_rope_cache,
    apply_rope,
    projection_sandwiched_generic_fp32,
)
from prism.kernels.born_povm import BornLogPOVMHead


def lorentz_origin_logmap(x: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """
    Cancellation-free origin log-map onto R^d tangent space from upper hyperboloid sheet:
        log_0^kappa(x_s) = (asinh(sqrt(kappa) * ||x_s||) / (sqrt(kappa) * ||x_s||)) * x_s
    Uses smooth Taylor expansion 1 - u^2 / 6 near ||x_s|| -> 0 (zero 0/0 singularity).
    """
    xf = x.float()
    sq_norm = (xf * xf).sum(dim=-1, keepdim=True)
    r = torch.sqrt(sq_norm.clamp(min=1e-12))
    sqrt_k = torch.sqrt(kappa.float())
    u = sqrt_k * r
    small = u < 1e-3
    scale_taylor = 1.0 - (u * u) / 6.0 + (u.pow(4)) / 120.0
    scale_exact = torch.asinh(u.clamp(max=1e4)) / u.clamp(min=1e-6)
    factor = torch.where(small, scale_taylor, scale_exact)
    return (xf * factor).to(x.dtype)


def chunked_wy_delta_scan(
    q: torch.Tensor,          # [B, T, H, dk]
    k_hat: torch.Tensor,      # [B, T, H, dk] unit Euclidean norm
    v: torch.Tensor,          # [B, T, H, dv]
    alpha: torch.Tensor,      # [B, T, H] in (0, 1)
    beta: torch.Tensor,       # [B, T, H] in (0, 1.5)
    chunk_size: int = 64,
) -> torch.Tensor:
    """
    Exact FP32 chunked WY Gated Delta-Rule recurrence:
        S_t = alpha_t * S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T
        o_t = S_t k_q -> q_t @ S_t^T
    where S_t has shape [B, H, dv, dk]. Zero Python token loop over T; only T // C = 8 steps!
    """
    B, T, H, dk = q.shape
    dv = v.shape[-1]
    assert T % chunk_size == 0
    NC = T // chunk_size
    C = chunk_size

    # Reshape into [B, NC, H, C, d] strictly in FP32
    qf = q.float().view(B, NC, C, H, dk).permute(0, 1, 3, 2, 4).contiguous()       # [B, NC, H, C, dk]
    kf = k_hat.float().view(B, NC, C, H, dk).permute(0, 1, 3, 2, 4).contiguous()   # [B, NC, H, C, dk]
    vf = v.float().view(B, NC, C, H, dv).permute(0, 1, 3, 2, 4).contiguous()       # [B, NC, H, C, dv]
    af = alpha.float().view(B, NC, C, H).permute(0, 1, 3, 2).contiguous()          # [B, NC, H, C]
    bf = beta.float().view(B, NC, C, H).permute(0, 1, 3, 2).contiguous()           # [B, NC, H, C]

    # Log-space decays within each chunk: gamma[i] = sum_{m=0}^i log(alpha_m)
    log_a = torch.log(af.clamp(min=1e-6))
    gamma = torch.cumsum(log_a, dim=-1)  # [B, NC, H, C] (<= 0)

    # Pairwise Lambda[i, j] = exp(gamma_i - gamma_j) for i >= j, 0 for i < j
    tril_mask = torch.tril(torch.ones(C, C, device=q.device, dtype=torch.bool))
    strict_tril = torch.tril(torch.ones(C, C, device=q.device, dtype=torch.bool), diagonal=-1)

    diff = gamma.unsqueeze(-1) - gamma.unsqueeze(-2)  # [B, NC, H, C, C]
    # Note for i >= j, diff <= 0; mask strictly upper triangle before exp to avoid overflow
    diff_safe = torch.where(tril_mask, diff.clamp(max=0.0), torch.full_like(diff, -50.0))
    Lambda = torch.where(tril_mask, torch.exp(diff_safe), torch.zeros_like(diff_safe))  # [B, NC, H, C, C]

    # Pre-state decay Lambda_0 = exp(gamma) [B, NC, H, C]
    Lambda0 = torch.exp(gamma.clamp(min=-50.0, max=0.0))
    chunk_decay_end = Lambda0[..., -1]  # [B, NC, H]

    # Build strict lower triangular A = I + strict_tril(diag(beta) * Lambda * (K @ K^T))
    KKT = torch.matmul(kf, kf.transpose(-1, -2))  # [B, NC, H, C, C]
    M = bf.unsqueeze(-1) * Lambda * KKT
    M = torch.where(strict_tril, M, torch.zeros_like(M))
    eye = torch.eye(C, device=q.device, dtype=torch.float32)
    A = eye + M  # unit lower triangular

    # Parallel triangular solves across all [B * NC * H] chunks at once!
    rhs_W = (bf * Lambda0).unsqueeze(-1) * kf  # [B, NC, H, C, dk]
    rhs_U = bf.unsqueeze(-1) * vf              # [B, NC, H, C, dv]

    BNH = B * NC * H
    A_flat = A.reshape(BNH, C, C)
    W_mat = torch.linalg.solve_triangular(A_flat, rhs_W.reshape(BNH, C, dk), upper=False, unitriangular=True).view(B, NC, H, C, dk)
    U_mat = torch.linalg.solve_triangular(A_flat, rhs_U.reshape(BNH, C, dv), upper=False, unitriangular=True).view(B, NC, H, C, dv)

    # Precompute intra-chunk Q @ K^T scaled by Lambda (causal tril)
    QKT_causal = torch.matmul(qf, kf.transpose(-1, -2)) * Lambda  # [B, NC, H, C, C]

    # Precompute chunk-end decay weights Lambda[C-1, j] = exp(gamma[C-1] - gamma[j])
    tail_decay = Lambda[:, :, :, -1, :]  # [B, NC, H, C]

    # Scan sequentially across ONLY NC chunks (e.g. 8 steps for T=512!)
    S_curr = torch.zeros(B, H, dv, dk, device=q.device, dtype=torch.float32)
    outputs = []
    state_norms = []
    for c in range(NC):
        q_c = qf[:, c]              # [B, H, C, dk]
        k_c = kf[:, c]              # [B, H, C, dk]
        W_c = W_mat[:, c]           # [B, H, C, dk]
        U_c = U_mat[:, c]           # [B, H, C, dv]
        lam0_c = Lambda0[:, c]      # [B, H, C]

        # u_corr = U_c - W_c @ S_curr^T  ([B,H,C,dk] @ [B,H,dk,dv] -> [B,H,C,dv])
        u_corr = U_c - torch.matmul(W_c, S_curr.transpose(-1, -2))

        # out_c = diag(lam0_c) * (q_c @ S_curr^T) + QKT_causal_c @ u_corr
        q_read = torch.matmul(q_c, S_curr.transpose(-1, -2)) * lam0_c.unsqueeze(-1)
        out_c = q_read + torch.matmul(QKT_causal[:, c], u_corr)
        outputs.append(out_c)

        # Update chunk state: S_next = decay_end * S_curr + u_corr^T @ (tail_decay * k_c)
        k_scaled = k_c * tail_decay[:, c].unsqueeze(-1)
        S_curr = chunk_decay_end[:, c].unsqueeze(-1).unsqueeze(-1) * S_curr + torch.matmul(u_corr.transpose(-1, -2), k_scaled)
        if not torch.jit.is_scripting():
            with torch.no_grad():
                state_norms.append(S_curr.pow(2).sum(dim=(-1, -2)).sqrt().mean())

    out = torch.stack(outputs, dim=1)  # [B, NC, H, C, dv]
    out = out.permute(0, 1, 3, 2, 4).contiguous().view(B, T, H, dv)
    mean_s_norm = torch.stack(state_norms).mean().item() if state_norms else 0.0
    return out.to(q.dtype), mean_s_norm


class CliffordGenericMixer(nn.Module):
    """
    CliffordNet dot + wedge geometric product mixer + Projection-Sandwiched FP32 GENERIC step.
    Uses Cl_{3,0} vector pairs (a, b) with dot + cross (wedge) interaction, Post-RMSNorm,
    gated SiLU residual, and rank-complete linear down projection.
    """
    def __init__(self, d_model: int, n_vec: int = 192):
        super().__init__()
        self.n_vec = n_vec
        self.d_cliff = n_vec * 3
        # dot produces n_vec scalars, wedge produces n_vec*3 bivectors -> 4 * n_vec features!
        d_feat = n_vec * 4

        self.in_proj = nn.Linear(d_model, 2 * self.d_cliff + d_feat, bias=False)
        self.norm_prod = RMSNorm(d_feat)
        self.out_proj = nn.Linear(d_feat, d_model, bias=False)

        # Compact bottleneck GENERIC projections (512 -> 64 -> 512) saving ~4.6M params
        self.generic_dim = 64
        assert d_model % self.generic_dim == 0
        self.e_proj = nn.Sequential(nn.Linear(d_model, 64, bias=False), nn.Linear(64, d_model, bias=False))
        self.s_proj = nn.Sequential(nn.Linear(d_model, 64, bias=False), nn.Linear(64, d_model, bias=False))
        self.J_generic = nn.Parameter(torch.randn(self.generic_dim, self.generic_dim) * 0.02)
        self.gamma_raw = nn.Parameter(torch.tensor(-2.0))
        self.post_gen_norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape
        proj = self.in_proj(x)
        a_raw, b_raw, gate = torch.split(proj, [self.d_cliff, self.d_cliff, self.n_vec * 4], dim=-1)

        a = a_raw.float().view(B, T, self.n_vec, 3)
        b = b_raw.float().view(B, T, self.n_vec, 3)
        a = a / math.sqrt(3.0)
        b = b / math.sqrt(3.0)
        dot = (a * b).sum(dim=-1)                        # [B, T, n_vec]
        wedge = torch.linalg.cross(a, b, dim=-1).reshape(B, T, self.n_vec * 3)
        prod = torch.cat([dot, wedge], dim=-1)           # [B, T, 4*n_vec]

        prod_normed = self.norm_prod(prod.to(x.dtype))
        h = self.out_proj(F.silu(gate) * prod_normed)

        # Projection-sandwiched FP32 GENERIC step across grouped 64-dim blocks
        n_grp = D // self.generic_dim
        x_g = h.view(B, T, n_grp, self.generic_dim)
        gE = self.e_proj(h).view(B, T, n_grp, self.generic_dim)
        gS = self.s_proj(h).view(B, T, n_grp, self.generic_dim)
        gamma = F.softplus(self.gamma_raw) + 1e-4
        x_gen, g_metrics = projection_sandwiched_generic_fp32(x_g, gE, gS, self.J_generic, gamma, dt=0.08)
        return self.post_gen_norm(x_gen.view(B, T, D)), g_metrics


class LDGBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 16, chunk_size: int = 64, n_vec: int = 240):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.chunk_size = chunk_size

        self.norm1 = RMSNorm(d_model)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.conv = nn.Conv1d(3 * d_model, 3 * d_model, kernel_size=4, padding=3, groups=3 * d_model, bias=False)

        # Lorentz per-block curvature log_c in [0.05, 10.0]
        self.log_c = nn.Parameter(torch.tensor(-0.5))
        self.q_lorentz_norm = RMSNorm(self.head_dim)
        self.k_lorentz_norm = RMSNorm(self.head_dim)

        # Gate projections (head-isotropic alpha_t, beta_t) + geodesic gate modulation
        self.ab_proj = nn.Linear(d_model, 2 * n_heads, bias=True)
        self.gate_geo = nn.Parameter(torch.zeros(2 * n_heads))
        # Logarithmic multi-timescale head bias init: tau spanning ~[4, 270] tokens
        with torch.no_grad():
            self.ab_proj.bias[:n_heads].copy_(torch.linspace(1.2, 5.6, n_heads))
            self.ab_proj.bias[n_heads:].fill_(-0.7)

        self.out_gate = nn.Linear(d_model, d_model, bias=False)
        self.out_norm = RMSNorm(self.head_dim)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.norm2 = RMSNorm(d_model)
        self.mixer = CliffordGenericMixer(d_model, n_vec=n_vec)

    def forward(self, x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor):
        B, T, D = x.shape
        h = self.norm1(x)

        # Causal depthwise conv4
        qkv = self.qkv_proj(h)
        qkv_conv = self.conv(qkv.transpose(1, 2))[..., :T].transpose(1, 2)
        qkv = F.silu(qkv_conv)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim)
        v = v.view(B, T, self.n_heads, self.head_dim)

        # Lorentz Origin Log-Map warping (bounded kappa in [0.05, 10.0]) + RoPE
        # NOTE: q stays UN-NORMALIZED so hyperbolic radial factor shapes retrieval strength & log_c grads!
        # Only k_hat is Euclidean unit-normalized for exact Householder stability.
        kappa = 0.05 + 9.95 * torch.sigmoid(self.log_c)
        q = lorentz_origin_logmap(self.q_lorentz_norm(q), kappa) / math.sqrt(self.head_dim)
        k = lorentz_origin_logmap(self.k_lorentz_norm(k), kappa)

        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        k_hat = k.float() / torch.sqrt((k.float() ** 2).sum(dim=-1, keepdim=True) + 1e-6)

        # Geodesic radius couples hyperbolic norm directly into retention/erasure gates
        h_norm = torch.sqrt((h.float() ** 2).mean(dim=-1, keepdim=True) + 1e-6)
        geo_rad = torch.asinh(torch.sqrt(kappa.float()) * h_norm) / torch.sqrt(kappa.float())
        ab = self.ab_proj(h).float() + (geo_rad - 1.0) * self.gate_geo.float()
        a_logit, b_logit = ab.chunk(2, dim=-1)
        alpha = 0.01 + 0.989 * torch.sigmoid(a_logit)   # (0.01, 0.999) FP32
        beta = 1.5 * torch.sigmoid(b_logit)              # (0, 1.5) convex/bounded Householder

        mem, mean_s_norm = chunked_wy_delta_scan(q, k_hat, v, alpha, beta, chunk_size=self.chunk_size)
        mem = self.out_norm(mem).reshape(B, T, D)
        attn_out = self.o_proj(F.silu(self.out_gate(h)) * mem)
        x = x + attn_out

        mix_out, g_metrics = self.mixer(self.norm2(x))
        x = x + mix_out

        metrics = {
            "alpha_mean": alpha.detach().mean().item(),
            "beta_mean": beta.detach().mean().item(),
            "kappa": kappa.detach().item(),
            "s_norm": mean_s_norm,
            **g_metrics,
        }
        return x, metrics


class LDGLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 49152,
        d_emb: int = 256,
        d_model: int = 512,
        n_layers: int = 10,
        n_heads: int = 16,
        chunk_size: int = 64,
        max_seq_len: int = 1024,
        tssp_weight: float = 0.05,
        use_povm_head: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_emb = d_emb
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.tssp_weight = tssp_weight
        self.use_povm_head = use_povm_head

        self.embed = nn.Embedding(vocab_size, d_emb)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.emb_up = nn.Linear(d_emb, d_model, bias=False)

        self.blocks = nn.ModuleList([
            LDGBlock(d_model, n_heads=n_heads, chunk_size=chunk_size, n_vec=240) for _ in range(n_layers)
        ])
        self.norm_out = RMSNorm(d_model)
        if use_povm_head:
            self.head = BornLogPOVMHead(d_model, d_emb, init_scale=12.0, eps=1e-6, chunk_size=128, mode="amplitude")
        else:
            self.head = FactorizedTiedHead(d_model, d_emb, chunk_size=256)

        # Collapse-proof TSSP chunk predictor over mean-pooled hidden summaries
        self.tssp_pred = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=False),
        )

        cos, sin = build_rope_cache(max_seq_len, d_model // n_heads)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)
        if use_povm_head:
            res_std = 0.02 / math.sqrt(2 * n_layers)
            for blk in self.blocks:
                nn.init.normal_(blk.o_proj.weight, std=res_std)
                nn.init.normal_(blk.mixer.out_proj.weight, std=res_std)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) and m is not self.emb_up:
            nn.init.normal_(m.weight, std=0.02)

    def forward_hidden(self, input_ids: torch.Tensor):
        x = self.emb_up(self.embed(input_ids))
        agg = {"alpha_mean": 0.0, "beta_mean": 0.0, "kappa": 0.0, "s_norm": 0.0, "se_res": 0.0, "es_res": 0.0, "ds_dt": 0.0}
        for blk in self.blocks:
            x, m = blk(x, self.rope_cos, self.rope_sin)
            for k in agg:
                if k == "se_res" or k == "es_res":
                    agg[k] = max(agg[k], m[k])
                else:
                    agg[k] += m[k] / len(self.blocks)
        return self.norm_out(x), agg

    def compute_tssp_loss(self, hidden: torch.Tensor) -> torch.Tensor:
        B, T, D = hidden.shape
        C = self.chunk_size
        NC = T // C
        if NC < 2:
            return torch.zeros((), device=hidden.device, dtype=torch.float32)
        pooled = hidden[:, :NC * C].float().view(B, NC, C, D).mean(dim=2)  # [B, NC, D]
        curr = pooled[:, :-1]       # [B, NC-1, D]
        nxt = pooled[:, 1:].detach()

        pred = self.tssp_pred(curr.to(hidden.dtype)).float()
        pred_n = F.normalize(pred, dim=-1, eps=1e-6)
        nxt_n = F.normalize(nxt, dim=-1, eps=1e-6)
        cos_loss = (1.0 - (pred_n * nxt_n).sum(dim=-1)).mean()

        # VICReg variance hinge across batch & chunks to prevent state freeze/collapse
        std = torch.sqrt(curr.reshape(-1, D).var(dim=0, unbiased=False) + 1e-4)
        var_hinge = F.relu(1.0 - std).mean()
        return cos_loss + 0.25 * var_hinge

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor = None, tssp_scale: float = 1.0):
        h, m = self.forward_hidden(input_ids)
        if targets is None:
            return self.head.forward_logits(h, self.embed.weight), m
        ce_loss = self.head.forward_loss(h, self.embed.weight, targets)
        tssp_loss = self.compute_tssp_loss(h) if self.tssp_weight > 0 else torch.zeros_like(ce_loss)
        total = ce_loss + (self.tssp_weight * tssp_scale) * tssp_loss
        m["ce_loss"] = ce_loss.detach().item()
        m["tssp_loss"] = tssp_loss.detach().item()
        m["aux_loss"] = (self.tssp_weight * tssp_scale * tssp_loss).detach().item()
        if self.use_povm_head:
            m["povm_scale"] = self.head.scale.detach().item()
        return total, m
