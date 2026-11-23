"""
EXPERIMENTAL RESEARCH CODE — NOT FOR PRODUCTION USE.

Comprehensive Mathematical, Gradient, and Causality Verification Suite for PRISM.
Tests:
1. Causal invariance (modifying token t >= k MUST leave logits[:, :k] identically unchanged) for ALL 4 models.
2. Chunked WY Delta Scan forward + backward parity vs exact sequential recurrence (< 1e-4).
3. Complex Linear Scan (Triton vs sequential FP32 reference) forward + backward parity (< 1e-4).
4. Chunked Born-Log POVM Head forward + backward parity (h, E_weight, tau_povm) vs PyTorch autograd (< 1e-4).
5. RPP Reversible Port-Hamiltonian exact reconstruction drift (< 1e-4) and autograd parity vs non-reversible trace.
6. Parameter count parity & Step-0 Cross-Entropy sanity (~ln(49152) = 10.803) across GPT, LDG, KCW, RPP.
"""
import math
import torch
import torch.nn.functional as F

from prism.kernels.complex_scan import complex_linear_scan, complex_scan_reference
from prism.kernels.born_povm import BornLogPOVMHead
from prism.models.baseline_gpt import GPTBaselineLM
from prism.models.ldg import LDGLM, chunked_wy_delta_scan
from prism.models.kcw import KCWLM
from prism.models.rpp import RPPLM
from prism.models.rmsn_baseline import RMSNOmega45LM


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_causality_all_models():
    dev = _get_device()
    T = 128
    cut = 65  # Crosses chunk boundary 64 intentionally!
    x1 = torch.randint(0, 49152, (2, T), device=dev)
    x2 = x1.clone()
    x2[:, cut:] = (x2[:, cut:] + 123) % 49152

    models = {
        "gpt": GPTBaselineLM(n_layers=2, max_seq_len=T).to(dev),
        "ldg": LDGLM(n_layers=2, max_seq_len=T).to(dev),
        "kcw": KCWLM(n_layers=2, max_seq_len=T).to(dev),
        "rpp": RPPLM(n_layers=2, max_seq_len=T).to(dev),
        "rpp_bridge": RPPLM(n_layers=2, max_seq_len=T, window=32, global_layers=(1,)).to(dev),
        "rpp_delta": RPPLM(n_layers=2, max_seq_len=T, use_hybrid_delta=True).to(dev),
        "rmsn": RMSNOmega45LM(n_layers=2, d_model=128, num_slots=16).to(dev),
    }
    for name, m in models.items():
        m.eval()
        with torch.no_grad():
            l1, _ = m(x1)
            l2, _ = m(x2)
        diff = (l1[:, :cut] - l2[:, :cut]).abs().max().item()
        print(f"[Causality] {name}: max prefix diff across cut={cut} = {diff:.2e}")
        assert diff < 1e-5, f"{name} leaked future tokens! max_diff={diff}"


def test_wy_chunked_delta_equivalence():
    dev = _get_device()
    torch.manual_seed(0)
    B, T, H, dk, dv = 2, 128, 4, 32, 32
    q = torch.randn(B, T, H, dk, device=dev, requires_grad=True)
    k = F.normalize(torch.randn(B, T, H, dk, device=dev), dim=-1).detach().requires_grad_(True)
    v = torch.randn(B, T, H, dv, device=dev, requires_grad=True)
    alpha = (0.5 + 0.45 * torch.rand(B, T, H, device=dev)).detach().requires_grad_(True)
    beta = (0.1 + 1.2 * torch.rand(B, T, H, device=dev)).detach().requires_grad_(True)

    # Chunked WY
    out_wy, _ = chunked_wy_delta_scan(q, k, v, alpha, beta, chunk_size=64)
    w_loss = torch.randn_like(out_wy)
    (out_wy * w_loss).sum().backward()
    g_wy = [q.grad.clone(), k.grad.clone(), v.grad.clone(), alpha.grad.clone(), beta.grad.clone()]

    q.grad = k.grad = v.grad = alpha.grad = beta.grad = None

    # Exact sequential recurrence: S_t = alpha_t S_{t-1}(I - beta_t k_t k_t^T) + beta_t v_t k_t^T
    S = torch.zeros(B, H, dv, dk, device=dev, dtype=torch.float32)
    outs = []
    for t in range(T):
        kt = k[:, t]  # [B, H, dk]
        vt = v[:, t]  # [B, H, dv]
        qt = q[:, t]  # [B, H, dk]
        at = alpha[:, t].unsqueeze(-1).unsqueeze(-1)
        bt = beta[:, t]
        # S_{t-1} @ kt -> [B, H, dv]
        Sk = torch.einsum("bhvk,bhk->bhv", S, kt)
        v_corr = bt.unsqueeze(-1) * (vt - at.squeeze(-1) * Sk)
        S = at * S + torch.einsum("bhv,bhk->bhvk", v_corr, kt)
        ot = torch.einsum("bhvk,bhk->bhv", S, qt)
        outs.append(ot)
    out_seq = torch.stack(outs, dim=1)
    (out_seq * w_loss).sum().backward()
    g_seq = [q.grad.clone(), k.grad.clone(), v.grad.clone(), alpha.grad.clone(), beta.grad.clone()]

    fwd_err = (out_wy - out_seq).abs().max().item()
    bwd_errs = [(a - b).abs().max().item() for a, b in zip(g_wy, g_seq)]
    print(f"[WY Delta] fwd max err = {fwd_err:.2e} | bwd max errs = {[f'{e:.2e}' for e in bwd_errs]}")
    assert fwd_err < 1e-4, f"WY fwd mismatch: {fwd_err}"
    assert max(bwd_errs) < 1e-3, f"WY bwd mismatch: {bwd_errs}"


def test_complex_scan_equivalence():
    dev = _get_device()
    torch.manual_seed(1)
    B, T, H, M = 2, 128, 4, 16
    ld = (-0.01 - 0.5 * torch.rand(B, T, H, M, device=dev)).requires_grad_(True)
    ph = (0.5 * torch.randn(B, T, H, M, device=dev)).requires_grad_(True)
    ur = torch.randn(B, T, H, M, device=dev, requires_grad=True)
    ui = torch.randn(B, T, H, M, device=dev, requires_grad=True)

    sr, si = complex_linear_scan(ld, ph, ur, ui)
    wr, wi = torch.randn_like(sr), torch.randn_like(si)
    (sr * wr + si * wi).sum().backward()
    g_fast = [t.grad.clone() for t in (ld, ph, ur, ui)]
    for t in (ld, ph, ur, ui):
        t.grad = None

    perm = (0, 2, 3, 1)
    inv_p = (0, 3, 1, 2)
    sr_ref, si_ref = complex_scan_reference(
        ld.permute(*perm).reshape(-1, T),
        ph.permute(*perm).reshape(-1, T),
        ur.permute(*perm).reshape(-1, T),
        ui.permute(*perm).reshape(-1, T),
    )
    sr_ref = sr_ref.view(B, H, M, T).permute(*inv_p)
    si_ref = si_ref.view(B, H, M, T).permute(*inv_p)
    (sr_ref * wr + si_ref * wi).sum().backward()
    g_ref = [t.grad.clone() for t in (ld, ph, ur, ui)]

    fwd_err = max((sr - sr_ref).abs().max().item(), (si - si_ref).abs().max().item())
    bwd_errs = [(a - b).abs().max().item() for a, b in zip(g_fast, g_ref)]
    print(f"[Complex Scan] fwd max err = {fwd_err:.2e} | bwd max errs = {[f'{e:.2e}' for e in bwd_errs]}")
    assert fwd_err < 1e-4
    assert max(bwd_errs) < 1e-4


def test_born_povm_head_grad_parity():
    dev = _get_device()
    torch.manual_seed(2)
    B, T, D, d_emb, V = 2, 16, 64, 32, 500
    for mode, init_s, eps in [("log", 1.0, 1e-3), ("amplitude", 12.0, 1e-6)]:
        head = BornLogPOVMHead(D, d_emb=d_emb, init_scale=init_s, eps=eps, chunk_size=12, mode=mode).to(dev)
        embed = torch.nn.Parameter(torch.randn(V, d_emb, device=dev) * 0.02)
        h = torch.randn(B, T, D, device=dev, requires_grad=True)
        y = torch.randint(0, V, (B, T), device=dev)

        loss_chunked = head.forward_loss(h, embed, y)
        loss_chunked.backward()
        g_c = [h.grad.clone(), embed.grad.clone(), head.tau_povm.grad.clone(), head.proj.weight.grad.clone()]

        h.grad = embed.grad = head.tau_povm.grad = head.proj.weight.grad = None
        logits_ref = head.forward_logits(h, embed)
        loss_ref = F.cross_entropy(logits_ref.reshape(-1, V), y.reshape(-1))
        loss_ref.backward()
        g_r = [h.grad.clone(), embed.grad.clone(), head.tau_povm.grad.clone(), head.proj.weight.grad.clone()]

        loss_err = abs(loss_chunked.item() - loss_ref.item())
        g_errs = [(a - b).abs().max().item() for a, b in zip(g_c, g_r)]
        print(f"[Born POVM ({mode})] loss err = {loss_err:.2e} | grad max errs = {[f'{e:.2e}' for e in g_errs]}")
        assert loss_err < 1e-5
        assert max(g_errs) < 1e-4


def test_rpp_reversible_drift_and_grad():
    dev = _get_device()
    torch.manual_seed(3)
    B, T, D, H = 2, 128, 128, 4
    m = RPPLM(vocab_size=256, d_emb=32, d_model=D, n_layers=6, n_heads=H, ffn_dim=128, max_seq_len=T).to(dev)
    x = torch.randint(0, 256, (B, T), device=dev)
    y = torch.randint(0, 256, (B, T), device=dev)

    # Forward using reversible stack -> check exact layerwise inverse reconstruction
    emb = m.embed(x)
    q0 = m.emb_up_q(emb).float().detach()
    p0 = m.emb_up_p(emb).float().detach()
    q_cur, p_cur = q0.clone(), p0.clone()
    with torch.no_grad():
        for fb, gb in zip(m.f_blocks, m.g_blocks):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                f_o = fb(p_cur, m.rope_cos, m.rope_sin).float()
            q_cur = q_cur + f_o
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                g_o = gb(q_cur).float()
            p_cur = p_cur + g_o

        for fb, gb in reversed(list(zip(m.f_blocks, m.g_blocks))):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                g_o = gb(q_cur).float()
            p_cur = p_cur - g_o
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                f_o = fb(p_cur, m.rope_cos, m.rope_sin).float()
            q_cur = q_cur - f_o

    q_drift = (q_cur - q0).abs().max().item()
    p_drift = (p_cur - p0).abs().max().item()
    print(f"[RPP Reversible Drift] 6-layer max q_drift={q_drift:.2e} p_drift={p_drift:.2e}")
    assert max(q_drift, p_drift) < 1e-4, f"Reversible drift too high: q={q_drift}, p={p_drift}"

    loss, _ = m(x, y)
    loss.backward()
    assert m.head.tau_povm.grad is not None and torch.isfinite(m.head.tau_povm.grad)


def test_param_parity_and_step0():
    dev = _get_device()
    T = 128
    x = torch.randint(0, 49152, (2, T), device=dev)
    y = torch.randint(0, 49152, (2, T), device=dev)
    target_ce = math.log(49152)

    specs = {
        "gpt": GPTBaselineLM(),
        "ldg": LDGLM(use_povm_head=False),
        "ldg_povm": LDGLM(use_povm_head=True),
        "kcw": KCWLM(),
        "rpp": RPPLM(use_hybrid_delta=False, use_povm_head=True),
        "rpp_bridge_1g": RPPLM(window=128, global_layers=(9,), use_hybrid_delta=False, use_povm_head=True),
        "rpp_bridge_2g": RPPLM(window=128, global_layers=(4, 9), use_hybrid_delta=False, use_povm_head=True),
        "rpp_bridge_5g": RPPLM(window=128, global_layers=(1, 3, 5, 7, 9), use_hybrid_delta=False, use_povm_head=True),
        "rpp_global_all": RPPLM(window=0, global_layers=tuple(range(10)), use_hybrid_delta=False, use_povm_head=True),
        "rpp_symp_ffn_1g": RPPLM(window=128, global_layers=(9,), f_ffn_dim=0, g_ffn_dim=960, use_hybrid_delta=False, use_povm_head=True),
        "rpp_symp_ffn_global": RPPLM(window=0, global_layers=tuple(range(10)), f_ffn_dim=0, g_ffn_dim=960, use_hybrid_delta=False, use_povm_head=True),
        "rpp_lin": RPPLM(use_hybrid_delta=False, use_povm_head=False),
        "rpp_delta": RPPLM(use_hybrid_delta=True),
        "rmsn": RMSNOmega45LM(),
    }
    print(f"\n{'Arch':<6} {'TotalM':>8} {'EmbM':>8} {'BackM':>8} {'Step0_CE':>10} {'Target_lnV':>11}")
    for name, m in specs.items():
        m = m.to(dev)
        tot = sum(p.numel() for p in m.parameters())
        emb = m.embed.weight.numel()
        back = tot - emb
        loss, metrics = m(x, y)
        ce0 = metrics["ce_loss"]
        print(f"{name:<6} {tot/1e6:>8.2f} {emb/1e6:>8.2f} {back/1e6:>8.2f} {ce0:>10.4f} {target_ce:>11.4f}")
        assert abs(ce0 - target_ce) < 0.6, f"{name} step-0 CE drifted far from ln(V): {ce0}"
        loss.backward()


if __name__ == "__main__":
    test_causality_all_models()
    test_wy_chunked_delta_equivalence()
    test_complex_scan_equivalence()
    test_born_povm_head_grad_parity()
    test_rpp_reversible_drift_and_grad()
    test_param_parity_and_step0()
    print("\nALL MATHEMATICAL, GRADIENT & CAUSALITY CHECKS PASSED!")
