"""
EXPERIMENTAL RESEARCH CODE — NOT FOR PRODUCTION USE.

Chunked FP32 Complex Born-Log POVM Cross-Entropy Head.
Implements rank-1 complex unit-normalized POVM measurement logits:
    mod_sq_v = |<psi, E_v>|^2
    logits_v = scale * (log(mod_sq_v + eps) - log(1.0 + eps))
with eps=1e-3 (eliminates sqrt(eps) gradient explosion) and learnable positive scale.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ChunkedBornLogPOVMFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        h_raw: torch.Tensor,      # [N, D] real (split [D//2, D//2] -> complex)
        E_raw: torch.Tensor,      # [V, D] real (tied embed.weight)
        targets: torch.Tensor,    # [N] int64
        scale: torch.Tensor,      # scalar positive tensor
        eps: float,
        chunk_size: int,
        mode: str = "amplitude",
    ):
        ctx.mode = mode
        assert h_raw.ndim == 2 and E_raw.ndim == 2
        N, D = h_raw.shape
        V, D_e = E_raw.shape
        assert D == D_e and D % 2 == 0
        Dc = D // 2

        h_f = h_raw.float()
        E_f = E_raw.float()
        sc = scale.float().clamp(min=1e-3)
        log_denom = math.log(1.0 + eps)

        psi_norm = torch.sqrt((h_f * h_f).sum(dim=-1, keepdim=True) + 1e-6)
        psi = h_f / psi_norm
        E_norm = torch.sqrt((E_f * E_f).sum(dim=-1, keepdim=True) + 1e-6)
        E_u = E_f / E_norm

        E_r, E_i = E_u[:, :Dc], E_u[:, Dc:]
        E_u_T = E_u.t().contiguous()
        E_conj_swap_T = torch.cat([-E_i, E_r], dim=1).t().contiguous()  # for im = psi_i E_r - psi_r E_i -> [psi_r, psi_i] @ [-E_i; E_r]

        valid_mask = (targets >= 0) & (targets < V)
        n_valid = valid_mask.sum().clamp(min=1).float()

        loss_total = torch.zeros((), dtype=torch.float32, device=h_raw.device)
        d_psi = torch.zeros_like(psi)
        d_E_u = torch.zeros_like(E_u)
        d_scale = torch.zeros_like(sc)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            psi_c = psi[start:end]  # [c, D]
            tgt_c = targets[start:end]
            vm_c = valid_mask[start:end]

            # re = psi_r @ E_r^T + psi_i @ E_i^T
            re = psi_c @ E_u_T
            # im = psi_i @ E_r^T - psi_r @ E_i^T = psi_r @ (-E_i)^T + psi_i @ E_r^T
            im = psi_c @ E_conj_swap_T

            mod_sq = re * re + im * im
            if ctx.mode == "amplitude":
                # Pure quantum probability amplitude: sqrt(|<psi, E_v>|^2 + eps), C^infty smooth
                feat = torch.sqrt(mod_sq + eps)
            else:
                feat = torch.log(mod_sq + eps) - log_denom
            logits = sc * feat

            # Stable FP32 cross-entropy + softmax probs
            tgt_safe = torch.where(vm_c, tgt_c, torch.zeros_like(tgt_c))
            lse = torch.logsumexp(logits, dim=-1, keepdim=True)
            probs = torch.exp(logits - lse)

            tgt_logits = logits.gather(1, tgt_safe.unsqueeze(1)).squeeze(1)
            ce_c = lse.squeeze(1) - tgt_logits
            loss_total = loss_total + (ce_c * vm_c.float()).sum() / n_valid

            # Gradients
            d_logits = probs
            d_logits.scatter_add_(
                1,
                tgt_safe.unsqueeze(1),
                torch.full((end - start, 1), -1.0, dtype=torch.float32, device=h_raw.device),
            )
            d_logits = d_logits * (vm_c.float().unsqueeze(1) / n_valid)

            d_scale = d_scale + (d_logits * feat).sum()

            if ctx.mode == "amplitude":
                d_mod_sq = d_logits * (0.5 * sc / feat)
            else:
                d_mod_sq = d_logits * (sc / (mod_sq + eps))
            d_re = (2.0 * d_mod_sq) * re
            d_im = (2.0 * d_mod_sq) * im

            d_re_r, d_re_i = d_re @ E_r, d_re @ E_i
            d_im_r, d_im_i = d_im @ E_r, d_im @ E_i
            # d_psi_r = d_re @ E_r - d_im @ E_i ; d_psi_i = d_re @ E_i + d_im @ E_r
            d_psi[start:end, :Dc] = d_re_r - d_im_i
            d_psi[start:end, Dc:] = d_re_i + d_im_r

            psi_r_c, psi_i_c = psi_c[:, :Dc], psi_c[:, Dc:]
            d_E_u[:, :Dc].add_(d_re.t() @ psi_r_c + d_im.t() @ psi_i_c)
            d_E_u[:, Dc:].add_(d_re.t() @ psi_i_c - d_im.t() @ psi_r_c)

        # Tangent projection through joint complex L2 normalizations
        psi_dot = (d_psi * psi).sum(dim=-1, keepdim=True)
        d_h_raw = (d_psi - psi_dot * psi) / psi_norm

        E_dot = (d_E_u * E_u).sum(dim=-1, keepdim=True)
        d_E_raw = (d_E_u - E_dot * E_u) / E_norm

        ctx.save_for_backward(d_h_raw.to(h_raw.dtype), d_E_raw.to(E_raw.dtype), d_scale.to(scale.dtype))
        return loss_total

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        d_h_raw, d_E_raw, d_scale = ctx.saved_tensors
        g = grad_out.to(d_h_raw.dtype)
        return d_h_raw * g, d_E_raw * grad_out.to(d_E_raw.dtype), None, d_scale * grad_out.to(d_scale.dtype), None, None, None


class BornLogPOVMHead(nn.Module):
    """
    Complex Born POVM Cross-Entropy Head tied to a real [V, d_emb] embedding table.
    Supports mode="amplitude" (scale * sqrt(|<psi,E>|^2 + eps), breaks Welch floor cleanly)
    or mode="log" (scale * log(|<psi,E>|^2 + eps)).
    """
    def __init__(self, d_model: int, d_emb: int = 256, init_scale: float = 12.0, eps: float = 1e-6, chunk_size: int = 128, mode: str = "amplitude"):
        super().__init__()
        assert d_emb % 2 == 0
        self.proj = nn.Linear(d_model, d_emb, bias=False)
        init_sp = math.log(math.expm1(max(init_scale - 0.2, 0.1)))
        self.tau_povm = nn.Parameter(torch.tensor(init_sp, dtype=torch.float32))
        self.eps = eps
        self.chunk_size = chunk_size
        self.mode = mode

    @property
    def scale(self) -> torch.Tensor:
        return 0.2 + F.softplus(self.tau_povm)

    def forward_loss(self, hidden: torch.Tensor, embed_weight: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        h_raw = self.proj(hidden).reshape(-1, embed_weight.shape[1])
        tgt = targets.reshape(-1)
        return _ChunkedBornLogPOVMFn.apply(h_raw, embed_weight, tgt, self.scale, self.eps, self.chunk_size, self.mode)

    def forward_logits(self, hidden: torch.Tensor, embed_weight: torch.Tensor) -> torch.Tensor:
        """Unchunked logits helper for generation/probes on short sequences."""
        h_raw = self.proj(hidden).float()
        E_f = embed_weight.float()
        Dc = E_f.shape[1] // 2
        psi = h_raw / torch.sqrt((h_raw * h_raw).sum(dim=-1, keepdim=True) + 1e-6)
        E_u = E_f / torch.sqrt((E_f * E_f).sum(dim=-1, keepdim=True) + 1e-6)
        psi_r, psi_i = psi[..., :Dc], psi[..., Dc:]
        E_r, E_i = E_u[:, :Dc], E_u[:, Dc:]
        re = psi_r @ E_r.t() + psi_i @ E_i.t()
        im = psi_i @ E_r.t() - psi_r @ E_i.t()
        mod_sq = re * re + im * im
        if self.mode == "amplitude":
            return self.scale.float() * torch.sqrt(mod_sq + self.eps)
        return self.scale.float() * (torch.log(mod_sq + self.eps) - math.log(1.0 + self.eps))
