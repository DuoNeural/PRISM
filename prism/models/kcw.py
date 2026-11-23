"""
EXPERIMENTAL RESEARCH CODE — NOT FOR PRODUCTION USE.

Candidate 2: KCW — Kuramoto-Clifford WaveNet
Combines:
1. Coalesced Triton FP32 Complex Harmonic Associative Scan with Hurwitz Quasicrystal
   frequency spacing (INV_PHI), Cayley phases, ZOH/OU variance-balanced drive,
   and causal depthwise conv4 + RoPE phase alignment.
2. Post-scan Singularity-Free Algebraic Kuramoto Inter-Head Phase Coupling
   (ZERO atan2! Uses unit phasor z / sqrt(|z|^2 + 1e-8) and Im(z_j * z_h^*))
   with row-bounded coupling matrix |eta * K|_inf <= 0.6.
3. Exact Bernoulli Closed-Form Radial Double-Well Bifurcation Flow:
   r_dot = alpha * r - beta * r^3 (Taylor branch for |alpha * dt| < 1e-4),
   eliminating Euler divergence and 0/0 singularities.
4. Singularity-Free Taylor-Sinc Cl_{3,0} Rotor + Grade-Mixing SwiGLU MLP.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from prism.kernels.common import (
    INV_PHI,
    PHI,
    RMSNorm,
    FactorizedTiedHead,
    build_rope_cache,
    apply_rope,
    taylor_sinc_rodrigues,
)
from prism.kernels.complex_scan import complex_linear_scan


def exact_bernoulli_double_well(
    z_r: torch.Tensor,
    z_i: torch.Tensor,
    alpha: torch.Tensor,   # [1, 1, H, n_osc] can be signed (- damped, + bistable)
    beta: torch.Tensor,    # [1, 1, H, n_osc] strictly > 0
    dt: float = 0.2,
):
    """
    Exact closed-form integration of radial Landau ODE: dr/dt = alpha * r - beta * r^3.
        r(dt) / r(0) = exp(alpha*dt) / sqrt(1 + (beta/alpha)*(exp(2*alpha*dt) - 1) * r^2)
    Applies scale m(r) directly to (z_r, z_i) with Taylor branch at |alpha*dt| < 1e-4.
    Zero division by r! Unconditionally positive and bounded.
    """
    r2 = z_r * z_r + z_i * z_i
    adt = alpha.float() * dt
    b_f = beta.float().clamp(min=1e-4)

    small = adt.abs() < 1e-4
    # For |adt| < 1e-4: exp(2*adt) - 1 ~= 2*adt -> (beta/alpha)*(2*alpha*dt) = 2 * beta * dt
    denom_small = 1.0 + 2.0 * b_f * dt * r2
    m_small = (1.0 + adt) * torch.rsqrt(denom_small.clamp(min=1e-6))

    adt_safe = torch.where(small, torch.full_like(adt, 1e-2), adt).clamp(min=-8.0, max=4.0)
    alpha_safe = adt_safe / dt
    e_adt = torch.exp(adt_safe)
    exp_factor = torch.expm1(2.0 * adt_safe) / alpha_safe  # strictly >= 0 whether alpha > 0 or < 0!
    denom_exact = 1.0 + b_f * exp_factor * r2
    m_exact = e_adt * torch.rsqrt(denom_exact.clamp(min=1e-6))

    m = torch.where(small, m_small, m_exact)
    return z_r * m, z_i * m


class KCWBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 8, ffn_dim: int = 1280, layer_idx: int = 0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim % 2 == 0
        self.n_osc = self.head_dim // 2  # complex oscillators per head (32)

        self.norm1 = RMSNorm(d_model)
        self.drive_proj = nn.Linear(d_model, d_model, bias=False)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=4, padding=3, groups=d_model, bias=False)

        # Query-like readout key/gate (with RoPE alignment on real pairs) + dynamic decay/phase mod
        self.read_gate = nn.Linear(d_model, d_model, bias=False)
        self.dyn_proj = nn.Linear(d_model, n_heads * self.n_osc * 2, bias=True)

        # Hurwitz quasicrystal base frequencies + bounded continuous log-rates
        idx = torch.arange(n_heads * self.n_osc, dtype=torch.float32)
        omega_init = (0.8 * (PHI ** (-layer_idx / 4.0)) + 0.35 * torch.frac((layer_idx * n_heads * self.n_osc + idx) * INV_PHI))
        self.omega_base = nn.Parameter(omega_init.view(1, 1, n_heads, self.n_osc))

        # Log-decay initialized spanning tau in [8, 256] (slow enough for T=512 DHP survival)
        tau_grid = torch.exp(torch.linspace(math.log(8.0), math.log(256.0), self.n_osc)).repeat(n_heads)
        gamma_init = torch.log(torch.expm1(1.0 / tau_grid).clamp(min=1e-4))
        self.gamma_raw = nn.Parameter(gamma_init.view(1, 1, n_heads, self.n_osc))

        # Inter-head algebraic Kuramoto phase coupling K in R^{H x H}
        self.kuramoto_K = nn.Parameter(torch.zeros(n_heads, n_heads))

        # Exact Bernoulli Landau radial parameters: alpha_landau starts lightly near critical point (~+0.2)
        self.alpha_landau = nn.Parameter(torch.full((1, 1, n_heads, self.n_osc), 0.25))
        self.beta_landau = nn.Parameter(torch.full((1, 1, n_heads, self.n_osc), 0.0))

        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # Singularity-free Cl_{3,0} Taylor-sinc Rotor + Grade-Mixing SwiGLU
        self.norm2 = RMSNorm(d_model)
        self.bivector_proj = nn.Linear(d_model, (d_model // 3) * 3, bias=False)  # on main chunks
        self.rot_dim = (d_model // 3) * 3
        self.w13 = nn.Linear(d_model, 2 * ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor):
        B, T, D = x.shape
        H, M = self.n_heads, self.n_osc
        h = self.norm1(x)

        u = self.drive_proj(h)
        u = F.silu(self.conv(u.transpose(1, 2))[..., :T].transpose(1, 2))
        u_pair = u.view(B, T, H, self.head_dim)
        u_pair = apply_rope(u_pair, rope_cos, rope_sin)
        u_r, u_i = u_pair.float().chunk(2, dim=-1)  # [B, T, H, M] each

        # Continuous FP32 scan parameters (strictly clamped to prevent underflow)
        d_mod, p_mod = self.dyn_proj(h).float().view(B, T, H, M, 2).unbind(-1)
        rate = F.softplus(self.gamma_raw.float() + 0.5 * d_mod) + 1e-4  # gamma > 0
        log_decay = (-rate).clamp(min=-10.0, max=-1e-4)
        # Cayley phase around Hurwitz frequencies
        phase = 2.0 * torch.atan(0.5 * (self.omega_base.float() + 0.3 * torch.tanh(p_mod)))

        # ZOH / OU amplitude balance: sqrt(1 - exp(2 * log_decay))
        ou_gain = torch.sqrt((-torch.expm1(2.0 * log_decay)).clamp(min=1e-5))
        s_r, s_i = complex_linear_scan(log_decay, phase, u_r * ou_gain, u_i * ou_gain)

        # Post-scan singularity-free algebraic Kuramoto coupling across heads H
        r2 = s_r * s_r + s_i * s_i
        inv_r = torch.rsqrt(r2 + 1e-8)
        z_hat_r, z_hat_i = s_r * inv_r, s_i * inv_r  # smooth unit phasor (zero atan2!)

        # Im(z_hat_j * conj(z_hat_h)) = sum_j K_{hj} (z_hat_j_i * z_hat_h_r - z_hat_j_r * z_hat_h_i)
        K_bounded = (0.6 / H) * torch.tanh(self.kuramoto_K.float())  # [H, H], |row sum| <= 0.6
        cross_r = torch.einsum("hj,btjm->bthm", K_bounded, z_hat_r)
        cross_i = torch.einsum("hj,btjm->bthm", K_bounded, z_hat_i)
        d_theta = (cross_i * z_hat_r - cross_r * z_hat_i).clamp(min=-1.5, max=1.5)
        cd, sd = torch.cos(d_theta), torch.sin(d_theta)
        k_r = s_r * cd - s_i * sd
        k_i = s_r * sd + s_i * cd

        # Exact Bernoulli closed-form radial double-well bifurcation
        beta_pos = F.softplus(self.beta_landau) + 1e-2
        alpha_clamped = self.alpha_landau.clamp(min=-2.0, max=2.0)
        k_r, k_i = exact_bernoulli_double_well(k_r, k_i, alpha_clamped, beta_pos, dt=0.25)

        wave_out = torch.cat([k_r, k_i], dim=-1).view(B, T, D).to(x.dtype)
        gate = F.silu(self.read_gate(h))
        x = x + self.o_proj(gate * wave_out)

        # Taylor-Sinc Cl_{3,0} Rotor + Grade-Mixing SwiGLU
        h2 = self.norm2(x)
        B_vec = 0.5 * self.bivector_proj(h2).view(B, T, D // 3, 3)
        h_rot = taylor_sinc_rodrigues(h2[..., :self.rot_dim].view(B, T, D // 3, 3), B_vec).reshape(B, T, self.rot_dim)
        if self.rot_dim < D:
            h_rot = torch.cat([h_rot, h2[..., self.rot_dim:]], dim=-1)
        g1, g2 = self.w13(h_rot).chunk(2, dim=-1)
        x = x + self.w2(F.silu(g1) * g2)

        with torch.no_grad():
            # Kuramoto Order Parameter R = |mean_h z_hat_h|
            r_mean = z_hat_r.mean(dim=2)
            i_mean = z_hat_i.mean(dim=2)
            R_order = torch.sqrt(r_mean * r_mean + i_mean * i_mean).mean().item()
            frac_bistable = (self.alpha_landau > 0).float().mean().item()
            amp_mean = torch.sqrt(k_r * k_r + k_i * k_i + 1e-12).mean().item()
            exp_decay_mean = torch.exp(log_decay).mean().item()

        return x, {
            "R_order": R_order,
            "frac_bistable": frac_bistable,
            "amp_mean": amp_mean,
            "exp_decay": exp_decay_mean,
        }


class KCWLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 49152,
        d_emb: int = 256,
        d_model: int = 512,
        n_layers: int = 10,
        n_heads: int = 8,
        ffn_dim: int = 1280,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_emb = d_emb
        self.d_model = d_model

        self.embed = nn.Embedding(vocab_size, d_emb)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.emb_up = nn.Linear(d_emb, d_model, bias=False)

        self.blocks = nn.ModuleList([
            KCWBlock(d_model, n_heads=n_heads, ffn_dim=ffn_dim, layer_idx=l) for l in range(n_layers)
        ])
        self.norm_out = RMSNorm(d_model)
        self.head = FactorizedTiedHead(d_model, d_emb, chunk_size=256)

        cos, sin = build_rope_cache(max_seq_len, d_model // n_heads)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) and m is not self.emb_up:
            nn.init.normal_(m.weight, std=0.02)

    def forward_hidden(self, input_ids: torch.Tensor):
        x = self.emb_up(self.embed(input_ids))
        agg = {"R_order": 0.0, "frac_bistable": 0.0, "amp_mean": 0.0, "exp_decay": 0.0}
        for blk in self.blocks:
            x, m = blk(x, self.rope_cos, self.rope_sin)
            for k in agg:
                agg[k] += m[k] / len(self.blocks)
        return self.norm_out(x), agg

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor = None):
        h, m = self.forward_hidden(input_ids)
        if targets is None:
            return self.head.forward_logits(h, self.embed.weight), m
        ce_loss = self.head.forward_loss(h, self.embed.weight, targets)
        m["ce_loss"] = ce_loss.detach().item()
        m["aux_loss"] = 0.0
        return ce_loss, m
