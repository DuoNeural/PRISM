"""
EXPERIMENTAL RESEARCH CODE — NOT FOR PRODUCTION USE.

Hybrid Muon + AdamW Optimizer for PRISM.
Follows Red Team Partition Rule:
- Muon (Newton-Schulz5 BF16 orthogonalization): ONLY interior pure 2D linear mixing matrices
  (explicitly excluding embeddings, factorized projection layers, Kuramoto couplings, norms, gates, physics scalars).
- AdamW (0.1x LR, wd=0.0): Physics scalars (curvature, decay rates, frequencies, Landau params, POVM temp).
- AdamW (1.0x LR): Embeddings, projections, convs, couplings, norms, 1D params.
"""
import math
import torch
from torch.optim import AdamW


@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    assert G.ndim >= 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    transposed = False
    if G.size(-2) > G.size(-1):
        X = X.T
        transposed = True
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.1):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.mul_(mu).add_(g)
                u = g.add(buf, alpha=mu) if nesterov else buf
                o = zeropower_via_newtonschulz5(u, steps=ns_steps)
                rms = o.float().pow(2).mean().sqrt().clamp(min=1e-8)
                scale = 0.2 / rms
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(o.to(p.dtype), alpha=-lr * scale.item())
        return loss


class HybridMuonAdamW:
    """Cleanly coordinates Muon + multi-group AdamW with independent base LR ratios."""
    PHYSICS_KEYWORDS = (
        "log_c", "a_log", "gamma_raw", "omega_base", "alpha_landau",
        "beta_landau", "tau_povm", "theta_fast", "theta_slow", "nernst_xi", "J_generic",
    )
    ADAMW_ONLY_KEYWORDS = (
        "embed", "emb_up", "head", "kuramoto_K", "q_lorentz", "k_lorentz",
        "conv", "norm", "tssp",
    )

    def __init__(self, model: torch.nn.Module, adamw_lr: float = 2e-4, muon_lr: float = 8e-4, weight_decay: float = 0.1):
        muon_params = []
        physics_params = []
        adamw_decay_params = []
        adamw_nodecay_params = []

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            lname = name.lower()
            if any(k.lower() in lname for k in self.PHYSICS_KEYWORDS):
                physics_params.append(p)
            elif p.ndim == 2 and p.shape[0] != getattr(model, "vocab_size", 49152) and not any(k.lower() in lname for k in self.ADAMW_ONLY_KEYWORDS):
                muon_params.append(p)
            elif p.ndim >= 2 and "embed" not in lname:
                adamw_decay_params.append(p)
            else:
                adamw_nodecay_params.append(p)

        self.muon = Muon(muon_params, lr=muon_lr, weight_decay=weight_decay) if muon_params else None
        adamw_groups = []
        if adamw_decay_params:
            adamw_groups.append({"params": adamw_decay_params, "lr": adamw_lr, "weight_decay": 0.05, "base_lr": adamw_lr})
        if adamw_nodecay_params:
            adamw_groups.append({"params": adamw_nodecay_params, "lr": adamw_lr, "weight_decay": 0.0, "base_lr": adamw_lr})
        if physics_params:
            adamw_groups.append({"params": physics_params, "lr": adamw_lr * 0.1, "weight_decay": 0.0, "base_lr": adamw_lr * 0.1})

        self.adamw = AdamW(adamw_groups, betas=(0.9, 0.95), eps=1e-8)
        self.base_muon_lr = muon_lr
        self.counts = {
            "muon": sum(p.numel() for p in muon_params),
            "adamw": sum(p.numel() for p in adamw_decay_params + adamw_nodecay_params),
            "physics": sum(p.numel() for p in physics_params),
        }

    def zero_grad(self, set_to_none: bool = True):
        if self.muon is not None:
            self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def set_lr_scale(self, scale: float):
        if self.muon is not None:
            for g in self.muon.param_groups:
                g["lr"] = self.base_muon_lr * scale
        for g in self.adamw.param_groups:
            g["lr"] = g["base_lr"] * scale

    def step(self):
        if self.muon is not None:
            self.muon.step()
        self.adamw.step()
