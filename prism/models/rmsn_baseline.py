"""
EXPERIMENTAL RESEARCH CODE — NOT FOR PRODUCTION USE.

PRISM-Matched RMSN (Omega = 4/5) Baseline — Archon Spec (D=640, L=10, M_w=80)
Wraps DuoNeural's hardened RMSNLayer (rmsn_core.py) inside PRISM's shared
factorized tied 49152 <-> 256 embedding bottleneck for strict parameter/split parity.
"""
import torch
import torch.nn as nn

from prism.kernels.common import RMSNorm, FactorizedTiedHead
from prism.models.rmsn_core import RMSNLayer


class RMSNOmega45LM(nn.Module):
    """
    Resonant Metriplectic Sheaf Network configured at the proven Arnold Tongue
    optimum Omega = D / (M_w * L) = 640 / (80 * 10) = 0.80 = 4/5.
    """
    def __init__(
        self,
        vocab_size: int = 49152,
        d_emb: int = 256,
        d_model: int = 640,
        n_layers: int = 10,
        num_slots: int = 80,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_emb = d_emb
        self.d_model = d_model
        self.n_layers = n_layers
        self.num_slots = num_slots
        self.omega_ratio = d_model / (num_slots * n_layers)

        self.embed = nn.Embedding(vocab_size, d_emb)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.emb_up = nn.Linear(d_emb, d_model, bias=False)
        nn.init.normal_(self.emb_up.weight, std=0.02)

        self.layers = nn.ModuleList([
            RMSNLayer(dim=d_model, num_slots=num_slots) for _ in range(n_layers)
        ])
        self.norm_out = RMSNorm(d_model)
        self.head = FactorizedTiedHead(d_model, d_emb, chunk_size=256)
        nn.init.normal_(self.head.proj.weight, std=0.02)

    def forward_hidden(self, input_ids: torch.Tensor):
        x = self.emb_up(self.embed(input_ids))
        max_lyap = -1e9
        mean_energy = 0.0
        ws_entropy = 0.0
        for layer in self.layers:
            x, t = layer(x)
            max_lyap = max(max_lyap, t["max_lyapunov"])
            mean_energy += t["mean_energy"] / self.n_layers
            ws_entropy += t["ws_entropy"] / self.n_layers
        return self.norm_out(x), {
            "lyap": max_lyap,
            "mean_energy": mean_energy,
            "ws_entropy": ws_entropy,
            "omega": self.omega_ratio,
        }

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor = None):
        h, m = self.forward_hidden(input_ids)
        if targets is None:
            return self.head.forward_logits(h, self.embed.weight), m
        ce_loss = self.head.forward_loss(h, self.embed.weight, targets)
        m["ce_loss"] = ce_loss.detach().item()
        m["aux_loss"] = 0.0
        return ce_loss, m
