"""
EXPERIMENTAL RESEARCH CODE — NOT FOR PRODUCTION USE.

Coalesced Triton FP32 complex associative scan + exact Wirtinger adjoint backward,
with a bit-clean pure-PyTorch reference fallback for non-CUDA / unit checks.
"""
import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _complex_affine_combine(
        a_l_r, a_l_i, u_l_r, u_l_i,
        a_r_r, a_r_i, u_r_r, u_r_i,
    ):
        new_a_r = a_r_r * a_l_r - a_r_i * a_l_i
        new_a_i = a_r_r * a_l_i + a_r_i * a_l_r
        new_u_r = a_r_r * u_l_r - a_r_i * u_l_i + u_r_r
        new_u_i = a_r_r * u_l_i + a_r_i * u_l_r + u_r_i
        return new_a_r, new_a_i, new_u_r, new_u_i

    @triton.jit
    def _cscan_fwd_kernel(
        ld_ptr, ph_ptr, ur_ptr, ui_ptr,
        sr_ptr, si_ptr,
        T: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_t = tl.arange(0, T)
        idx = pid * T + offs_t

        ld = tl.load(ld_ptr + idx).to(tl.float32)
        ph = tl.load(ph_ptr + idx).to(tl.float32)
        ur = tl.load(ur_ptr + idx).to(tl.float32)
        ui = tl.load(ui_ptr + idx).to(tl.float32)

        mag = tl.exp(ld)
        ar = mag * tl.cos(ph)
        ai = mag * tl.sin(ph)

        _, _, sr, si = tl.associative_scan(
            (ar, ai, ur, ui), axis=0, combine_fn=_complex_affine_combine
        )
        tl.store(sr_ptr + idx, sr)
        tl.store(si_ptr + idx, si)

    @triton.jit
    def _cscan_bwd_kernel(
        ld_ptr, ph_ptr, sr_ptr, si_ptr,
        dsr_ptr, dsi_ptr,
        d_ld_ptr, d_ph_ptr, dur_ptr, dui_ptr,
        T: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_r = tl.arange(0, T)
        offs_t = (T - 1) - offs_r
        base = pid * T
        idx_rev = base + offs_t

        dsr = tl.load(dsr_ptr + idx_rev).to(tl.float32)
        dsi = tl.load(dsi_ptr + idx_rev).to(tl.float32)

        offs_next = tl.minimum(offs_t + 1, T - 1)
        idx_next = base + offs_next
        ld_next = tl.load(ld_ptr + idx_next).to(tl.float32)
        ph_next = tl.load(ph_ptr + idx_next).to(tl.float32)

        mag_next = tl.where(offs_r > 0, tl.exp(ld_next), 0.0)
        ar_conj = mag_next * tl.cos(ph_next)
        ai_conj = -mag_next * tl.sin(ph_next)

        _, _, gr_rev, gi_rev = tl.associative_scan(
            (ar_conj, ai_conj, dsr, dsi), axis=0, combine_fn=_complex_affine_combine
        )
        tl.store(dur_ptr + idx_rev, gr_rev)
        tl.store(dui_ptr + idx_rev, gi_rev)

        offs_prev = tl.maximum(offs_t - 1, 0)
        idx_prev = base + offs_prev
        sr_prev = tl.where(offs_t > 0, tl.load(sr_ptr + idx_prev).to(tl.float32), 0.0)
        si_prev = tl.where(offs_t > 0, tl.load(si_ptr + idx_prev).to(tl.float32), 0.0)

        # Adjoint wrt complex multiplier a_t: g_t * conj(s_{t-1})
        da_r = gr_rev * sr_prev + gi_rev * si_prev
        da_i = gi_rev * sr_prev - gr_rev * si_prev

        ld_curr = tl.load(ld_ptr + idx_rev).to(tl.float32)
        ph_curr = tl.load(ph_ptr + idx_rev).to(tl.float32)
        mag_curr = tl.exp(ld_curr)
        ar_curr = mag_curr * tl.cos(ph_curr)
        ai_curr = mag_curr * tl.sin(ph_curr)

        # Since a = exp(ld + i*ph), dL/d(ld) = Re(dL/da * conj(a)?) -> da_r*ar + da_i*ai
        # and dL/d(ph) = da_i*ar - da_r*ai
        d_ld = da_r * ar_curr + da_i * ai_curr
        d_ph = da_i * ar_curr - da_r * ai_curr

        d_ld = tl.where(offs_t > 0, d_ld, 0.0)
        d_ph = tl.where(offs_t > 0, d_ph, 0.0)
        tl.store(d_ld_ptr + idx_rev, d_ld)
        tl.store(d_ph_ptr + idx_rev, d_ph)


class _TritonComplexScanFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_decay: torch.Tensor, phase: torch.Tensor, u_r: torch.Tensor, u_i: torch.Tensor):
        # Inputs: [N_fibers, T] contiguous float32
        N, T = log_decay.shape
        assert T in (64, 128, 256, 512, 1024), f"Unsupported scan T={T}; must be power of 2 in [64, 1024]"
        ld = log_decay.contiguous().float()
        ph = phase.contiguous().float()
        ur = u_r.contiguous().float()
        ui = u_i.contiguous().float()

        sr = torch.empty_like(ur)
        si = torch.empty_like(ui)
        _cscan_fwd_kernel[(N,)](ld, ph, ur, ui, sr, si, T=T, num_warps=8)

        ctx.save_for_backward(ld, ph, sr, si)
        ctx.T = T
        return sr, si

    @staticmethod
    def backward(ctx, dsr: torch.Tensor, dsi: torch.Tensor):
        ld, ph, sr, si = ctx.saved_tensors
        N = ld.shape[0]
        T = ctx.T
        dsr_c = dsr.contiguous().float()
        dsi_c = dsi.contiguous().float()

        d_ld = torch.empty_like(ld)
        d_ph = torch.empty_like(ph)
        dur = torch.empty_like(sr)
        dui = torch.empty_like(si)
        _cscan_bwd_kernel[(N,)](
            ld, ph, sr, si,
            dsr_c, dsi_c,
            d_ld, d_ph, dur, dui,
            T=T, num_warps=8,
        )
        return d_ld, d_ph, dur, dui


def complex_scan_reference(log_decay: torch.Tensor, phase: torch.Tensor, u_r: torch.Tensor, u_i: torch.Tensor):
    """Exact sequential FP32 complex linear recurrence: s_t = exp(ld_t + i*ph_t) s_{t-1} + u_t."""
    ld = log_decay.float()
    ph = phase.float()
    ur = u_r.float()
    ui = u_i.float()
    mag = torch.exp(ld)
    ar = mag * torch.cos(ph)
    ai = mag * torch.sin(ph)

    N, T = ld.shape
    sr_list = []
    si_list = []
    sr_prev = torch.zeros(N, dtype=torch.float32, device=ld.device)
    si_prev = torch.zeros(N, dtype=torch.float32, device=ld.device)
    for t in range(T):
        ar_t = ar[:, t]
        ai_t = ai[:, t]
        sr_t = ar_t * sr_prev - ai_t * si_prev + ur[:, t]
        si_t = ar_t * si_prev + ai_t * sr_prev + ui[:, t]
        sr_list.append(sr_t)
        si_list.append(si_t)
        sr_prev, si_prev = sr_t, si_t
    return torch.stack(sr_list, dim=1), torch.stack(si_list, dim=1)


def complex_linear_scan(log_decay: torch.Tensor, phase: torch.Tensor, u_r: torch.Tensor, u_i: torch.Tensor):
    """
    Causal complex linear scan s_t = exp(log_decay_t + i * phase_t) * s_{t-1} + (u_r_t + i * u_i_t)
    Strictly FP32. Accepts arbitrary shape [B, T, ...], transposes to [N_fibers, T] for coalesced memory access.
    """
    orig_shape = log_decay.shape  # [B, T, ...]
    B, T = orig_shape[0], orig_shape[1]
    rest = orig_shape[2:]
    # Permute [B, T, ...] -> [B, ..., T] -> [N_fibers, T]
    perm_to = (0,) + tuple(range(2, len(orig_shape))) + (1,)
    ld_f = log_decay.permute(*perm_to).reshape(-1, T).contiguous()
    ph_f = phase.permute(*perm_to).reshape(-1, T).contiguous()
    ur_f = u_r.permute(*perm_to).reshape(-1, T).contiguous()
    ui_f = u_i.permute(*perm_to).reshape(-1, T).contiguous()

    # Check if T is power of 2 in [64, 1024] and on CUDA with Triton
    use_triton = HAS_TRITON and ld_f.is_cuda and (T in (64, 128, 256, 512, 1024))
    if use_triton:
        sr_f, si_f = _TritonComplexScanFn.apply(ld_f, ph_f, ur_f, ui_f)
    elif ld_f.is_cuda and HAS_TRITON and T < 1024:
        # Pad right to next power of 2 >= 64
        T_pad = 64
        while T_pad < T:
            T_pad *= 2
        pad = T_pad - T
        zero_p = torch.zeros(ld_f.shape[0], pad, device=ld_f.device, dtype=torch.float32)
        sr_p, si_p = _TritonComplexScanFn.apply(
            torch.cat([ld_f, zero_p], dim=1),
            torch.cat([ph_f, zero_p], dim=1),
            torch.cat([ur_f, zero_p], dim=1),
            torch.cat([ui_f, zero_p], dim=1),
        )
        sr_f, si_f = sr_p[:, :T], si_p[:, :T]
    else:
        sr_f, si_f = complex_scan_reference(ld_f, ph_f, ur_f, ui_f)

    # Restore [B, ..., T] -> [B, T, ...]
    inv_shape = (B,) + rest + (T,)
    perm_back = (0, len(orig_shape) - 1) + tuple(range(1, len(orig_shape) - 1))
    sr = sr_f.view(inv_shape).permute(*perm_back).contiguous()
    si = si_f.view(inv_shape).permute(*perm_back).contiguous()
    return sr, si
