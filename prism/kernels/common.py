"""
EXPERIMENTAL RESEARCH CODE — NOT FOR PRODUCTION USE.

Shared building blocks:
- RMSNorm
- Precomputed RoPE + causal application
- ChunkedTiedCrossEntropy (FP32 logits, zero full [N, V] graph bloat)
- Taylor-Sinc Cl_{3,0} Rotor + Projection-Sandwiched GENERIC Step (FP32)
- CliffordNet Geometric Dot+Wedge Product Mixer
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

INV_PHI = (1.0 + math.sqrt(5.0)) / 2.0 - 1.0  # 0.6180339887498949
PHI = (1.0 + math.sqrt(5.0)) / 2.0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        inv = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (xf * inv).to(x.dtype) * self.weight


def build_rope_cache(seq_len: int, head_dim: int, base: float = 10000.0, device=None):
    assert head_dim % 2 == 0
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # [T, dh/2]
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, T, H, dh], cos/sin: [T, dh//2]"""
    T = x.shape[1]
    c = cos[:T].view(1, T, 1, -1).to(x.dtype)
    s = sin[:T].view(1, T, 1, -1).to(x.dtype)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


class _ChunkedTiedCEFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor, chunk_size: int):
        N, D = hidden.shape
        V, _ = weight.shape
        h_f = hidden.float()
        w_f = weight.float()

        valid = (targets >= 0) & (targets < V)
        n_valid = valid.sum().clamp(min=1).float()

        loss_total = torch.zeros((), dtype=torch.float32, device=hidden.device)
        d_h = torch.zeros_like(h_f)
        d_w = torch.zeros_like(w_f)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            h_c = h_f[start:end]
            t_c = targets[start:end]
            v_c = valid[start:end]
            t_safe = torch.where(v_c, t_c, torch.zeros_like(t_c))

            logits = h_c @ w_f.t()  # [c, V] FP32
            lse = torch.logsumexp(logits, dim=-1, keepdim=True)
            probs = torch.exp(logits - lse)

            ce_c = lse.squeeze(1) - logits.gather(1, t_safe.unsqueeze(1)).squeeze(1)
            loss_total = loss_total + (ce_c * v_c.float()).sum() / n_valid

            d_logits = probs
            d_logits.scatter_add_(
                1,
                t_safe.unsqueeze(1),
                torch.full((end - start, 1), -1.0, dtype=torch.float32, device=hidden.device),
            )
            d_logits = d_logits * (v_c.float().unsqueeze(1) / n_valid)

            d_h[start:end] = d_logits @ w_f
            d_w.add_(d_logits.t() @ h_c)

        ctx.save_for_backward(d_h.to(hidden.dtype), d_w.to(weight.dtype))
        return loss_total

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        d_h, d_w = ctx.saved_tensors
        return d_h * grad_out.to(d_h.dtype), d_w * grad_out.to(d_w.dtype), None, None


class FactorizedTiedHead(nn.Module):
    """Shared factorized tied head [B, T, d_model] -> d_emb -> vocab logits."""
    def __init__(self, d_model: int, d_emb: int = 256, chunk_size: int = 256):
        super().__init__()
        self.proj = nn.Linear(d_model, d_emb, bias=False)
        self.chunk_size = chunk_size

    def forward_loss(self, hidden: torch.Tensor, embed_weight: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        h = self.proj(hidden).reshape(-1, embed_weight.shape[1])
        return _ChunkedTiedCEFn.apply(h, embed_weight, targets.reshape(-1), self.chunk_size)

    def forward_logits(self, hidden: torch.Tensor, embed_weight: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden).float() @ embed_weight.float().t()


def taylor_sinc_rodrigues(v: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Singularity-free SO(3) rotation exp(-B_hat) acting on v (..., 3) via bivector B (..., 3).
    Zero sqrt(0) or 0/0 singularities when ||B|| -> 0 using Taylor series for phi^2 < 1e-6.
    """
    vf = v.float()
    Bf = B.float()
    phi2 = (Bf * Bf).sum(dim=-1, keepdim=True)  # [..., 1]
    phi = torch.sqrt(phi2.clamp(min=1e-12))

    # sin(phi)/phi and (1 - cos(phi))/phi^2
    small = phi2 < 1e-6
    s1_taylor = 1.0 - phi2 / 6.0 + (phi2 * phi2) / 120.0
    s2_taylor = 0.5 - phi2 / 24.0 + (phi2 * phi2) / 720.0

    s1_exact = torch.sin(phi) / phi
    s2_exact = (1.0 - torch.cos(phi)) / phi2.clamp(min=1e-12)

    s1 = torch.where(small, s1_taylor, s1_exact)
    s2 = torch.where(small, s2_taylor, s2_exact)

    b_cross_v = torch.linalg.cross(Bf, vf, dim=-1)
    b_cross_2v = torch.linalg.cross(Bf, b_cross_v, dim=-1)
    return (vf + s1 * b_cross_v + s2 * b_cross_2v).to(v.dtype)


def projection_sandwiched_generic_fp32(
    x: torch.Tensor,         # [..., D]
    grad_E: torch.Tensor,    # [..., D]
    grad_S: torch.Tensor,    # [..., D]
    J_raw: torch.Tensor,     # [D, D] or small block [G, G]
    gamma: torch.Tensor,     # [1] or [...] positive scalar
    dt: float = 0.2,
):
    """
    Exact Projection-Sandwiched GENERIC update:
        dx/dt = J(grad_E) + R(grad_S)
    where J is projected orthogonal to grad_S and skew-symmetrized, and R_gradS is projected
    orthogonal to grad_E so {S, E} = grad_S^T J grad_E == 0 and [E, S] = grad_E^T R grad_S == 0.
    Returns (x_next, metrics_dict) cleanly in FP32 then cast back.
    """
    xf = x.float()
    a = grad_E.float()
    b = grad_S.float()
    eps = 1e-6

    b_norm2 = (b * b).sum(dim=-1, keepdim=True).clamp(min=eps)
    a_norm2 = (a * a).sum(dim=-1, keepdim=True).clamp(min=eps)

    # Project a orthogonal to b: P_b(a)
    ab_dot = (a * b).sum(dim=-1, keepdim=True)
    Pa = a - (ab_dot / b_norm2) * b

    # Skew-symmetric J0
    J0 = 0.5 * (J_raw.float() - J_raw.float().t())
    J0_Pa = Pa @ J0.t()  # or J0 @ Pa along last dim
    # Project result orthogonal to b again -> P_b J0 P_b (a)
    jb_dot = (J0_Pa * b).sum(dim=-1, keepdim=True)
    J_gradE = J0_Pa - (jb_dot / b_norm2) * b

    # Project b orthogonal to a: P_a(b)
    R_gradS = gamma.float() * (b - (ab_dot / a_norm2) * a)

    with torch.no_grad():
        se_bracket = (b * J_gradE).sum(dim=-1).abs().max().item()
        es_bracket = (a * R_gradS).sum(dim=-1).abs().max().item()
        ds_dt = (b * R_gradS).sum(dim=-1).mean().item()

    out = xf + dt * (J_gradE + R_gradS)
    return out.to(x.dtype), {"se_res": se_bracket, "es_res": es_bracket, "ds_dt": ds_dt}
