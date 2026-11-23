# PRISM — Physics-Regulated Information-State Models

> [!CAUTION]
> **STRICTLY EXPERIMENTAL RESEARCH — WORK IN PROGRESS — NOT FOR PRODUCTION USE.**
>
> Everything in this repository is an active, rapidly evolving architecture research prototype under live ablation on RTX 4080 Super hardware. APIs, mathematical formulations, checkpoints, and training recipes will change without notice. None of these models are hardened, evaluated for safety, or intended for production deployment.

---

## Research Thesis

**PRISM** synthesizes DuoNeural's empirical sequence-modeling discoveries (`RMSN`, `CRSN`, `SRSN`, `QSMN`, `PULSE`, `Bismuth-104M`, and Dynamical Horizon Principle / Thermodynamic papers 24–41) with modern linear-state and oscillator mechanics into a controlled benchmark family. Every candidate is parameter-, vocabulary-, split-, and optimizer-matched against a standard Causal RoPE-SwiGLU Transformer baseline on **WikiText-103** (`vocab = 49152`, `cosmo2-tokenizer`, `seq_len = 512`, `12.29M` tokens per smoke run).

---

## Core Concepts Explained

### 1. TSSP — Temporal State-Space Self-Prediction (Paper 34)
Autoregressive token loss (`CE`) only supervises local token-by-token predictions, leaving linear recurrent states vulnerable to the **Dynamical Horizon Principle (DHP)** horizon cliff ($\tau^* / \tau_L \approx 0.72$, where recurrent predictors forget beyond $\sim 0.72$ Lyapunov times). **TSSP** adds an auxiliary self-supervised trajectory objective directly in latent state space: from pooled chunk state/hidden summary $\bar{h}_c$ over chunk $c$ ($C = 64$), a lightweight 2-layer predictor learns to match the stop-gradient normalized next-chunk representation $\operatorname{sg}(\bar{h}_{c+1} / \|\bar{h}_{c+1}\|_2)$.
- **Why Naive TSSP Collapses (and Our Fix):** Naive MSE TSSP $\mathbb{E}\|P(S_c) - \operatorname{sg}(S_{c+1})\|_2^2$ can be trivially cheated by shrinking $\|S\| \to 0$ or freezing state updates ($\alpha \to 1, \gamma \to 0$). PRISM uses a **collapse-proof cosine similarity loss + VICReg variance hinge** ($\operatorname{ReLU}(1 - \operatorname{std}(\bar{h}_c))$) warmed up cleanly after step 150 and logged strictly *separately* from pure validation CE/PPL.

### 2. GENERIC Metriplectic Constraints ($\{S, E\} = 0$, $[E, S] = 0$)
Thermodynamically consistent neural state evolution splits updates into a reversible **Hamiltonian** part $J(\nabla E)$ (skew-symmetric $J = -J^T$, conserving energy $E$) and an irreversible **Dissipative** part $R(\nabla S)$ (symmetric PSD $R \succeq 0$, guaranteeing non-negative entropy production $\dot{S} \ge 0$). PRISM enforces exact **projection-sandwiched orthogonality in FP32**: $\nabla E$ is projected orthogonal to $\nabla S$ before applying $J$, and $\nabla S$ is projected orthogonal to $\nabla E$ before applying $R$, keeping degeneracy residuals `max(|{S, E}|), max(|[E, S]|) < 1e-4` (typically `0.0–5e-5` live) while actively monitoring entropy production rate $\mathrm{d}S/\mathrm{d}t$.

### 3. Dynamical Horizon Principle (DHP) & Multi-Timescale Poles
Open-system recurrent states with dissipation rate $\gamma$ suffer exponential memory decay with horizon ceiling $\tau^* \approx 0.7 / \gamma$. To span `seq_len = 512` cleanly without horizon collapse or BF16 pole quantization (where BF16 snaps $a_s > 0.9961 \to 1.0$), PRISM initializes continuous FP32 log-rates logarithmically across fast ("juvenile" local syntax) and slow ("condensed" long-range context) slots ($\tau \in [6, 512]$).

### 4. Born POVM Measurement & The Welch Bound Floor (`v0.1` Discovery)
Replacing standard softmax dot-product logits with **Positive Operator-Valued Measure (POVM)** Born probabilities over unit-normalized complex embeddings $E_v \in \mathbb{C}^{D_c}$ ($D_c = d_{\text{emb}}/2 = 128$, $V = 49{,}152$):
- In **v0.1**, using log-probability logits $s \cdot \log(|\langle \psi, E_v \rangle|^2 + \epsilon)$ at natural scale $s = 1.0$ hit a **hard mathematical Welch ceiling**: for $49{,}152$ unit vectors in $\mathbb{C}^{128}$, isotropic background overlap sums to $\sum_{u \neq v^*} (|z_u|^2 + 10^{-3}) \approx 432.5$, capping oracle probability at $P(v^*) \le 1/433 \implies \text{CE}_{\min} \ge 6.07$ nats ($\text{PPL} \ge 432$).
- In **v0.2**, switching to **Quantum Amplitude POVM logits** $s \cdot \sqrt{|\langle \psi, E_v \rangle|^2 + \epsilon}$ (`init_scale = 12.0`) exploits the tight random amplitude variance ($\operatorname{std} \approx 0.04 \implies$ step-0 CE $\approx 10.97$) while widening matched-vs-background separation to $\sim 11.06$ nats ($e^{11.06} > V$), completely shattering the Welch ceiling.

---

## Architecture Catalog (`~45.7M–46.1M` Matched Budget)

All models share an identical **factorized tied `49152 → 256` embedding bottleneck** (`12.58M` params) so backbone capacity (`33.07M–33.52M` params) is directly comparable without vocabulary-table confounders.

| Key | Name | Detailed Mechanics & Observables |
|---|---|---|
| `gpt` | **Causal RoPE-SwiGLU Baseline** | Standard 10-layer (`d=512, H=8`) causal Flash/SDPA attention with RoPE + Pre-RMSNorm + SwiGLU FFN (`ffn=1472`). Pure Transformer yardstick. |
| `ldg` | **Lorentz-Delta-TSSP GENERIC (`LDG`)** | **Sequence mixer:** 10-layer FP32 chunked-64 WY Gated DeltaNet (`S_t = alpha_t S_{t-1}(I - beta_t k_hat k_hat^T) + beta_t v_t k_hat^T`) solved via exact parallel `solve_triangular` on `(I + M)` with causal depthwise `conv4` + RoPE + logarithmic multi-timescale head decays. **Geometry:** Cancellation-free Lorentz origin log-map (`asinh(sqrt(kappa)*r)/(sqrt(kappa)*r)`) warping spatial `q, k` under learnable block curvature `kappa in [0.05, 10.0]` (`q` kept un-normalized so hyperbolic radial factor drives retrieval, plus geodesic norm coupling into `(alpha, beta)`). **FFN-free mixer:** `Cl_{3,0}` CliffordNet dot+wedge geometric product followed by FP32 projection-sandwiched GENERIC step (`Post-RMSNorm` stabilized) and collapse-proof TSSP chunk prediction. **Metrics:** `alpha_mean`, `beta_mean`, `kappa`, `s_norm`, `se_res`, `es_res`, `ds_dt`, `tssp_loss`. |
| `ldg_povm` | **Lorentz-Delta GENERIC + Quantum POVM (`LDG-POVM`)** | Pairs the full `LDG v0.2` Lorentz-Delta GENERIC TSSP backbone with the chunked FP32 complex `BornAmplitudePOVMHead`, isolating whether quantum Born measurement readout complements GENERIC recurrent state processing. |
| `kcw` | **Kuramoto-Clifford WaveNet (`KCW`)** | **Wave scan:** Coalesced Triton FP32 complex associative scan (`tl.associative_scan` + Wirtinger adjoint backward) with Hurwitz quasicrystal frequency spacing (`INV_PHI`), Cayley phases, ZOH/OU variance-balanced drive, and causal `conv4` + RoPE. **Synchronization:** Post-scan singularity-free algebraic Kuramoto phase coupling across heads (`sin(theta_j - theta_h) = Im(z_hat_j * conj(z_hat_h))`, zero `atan2`!) bounded to `\|eta K\|_inf <= 0.6` so oscillators stay in the expressive sync band rather than collapsing (`R -> 1`) or uncoupling (`R -> 0`). **Bifurcation:** Exact closed-form Bernoulli ODE solution for radial Landau flow `dr/dt = alpha*r - beta*r^3` (50/50 damped/bistable init in `v0.2`), followed by a singularity-free Taylor-sinc `Cl_{3,0}` rotor + SwiGLU. **Metrics:** Kuramoto order parameter `R_order`, `frac_bistable`, `amp_mean`, `exp_decay`. |
| `rpp` | **RevPort-POVM Hybrid (`RPP`)** | **Backbone:** 10-layer bit-exact FP32 reversible Port-Hamiltonian conjugate streams `(q, p in R^512)` (BF16 sub-block compute cast to FP32 additions; backward reconstruction drift `< 1e-4`), slashing peak training VRAM down to **~1.37 GB**. **Sub-blocks:** `F(p)` runs exact causal sliding-window RoPE attention (`W=64`, diagonal-safe overlapping SDPA); `G(q)` runs a Dual-Timescale Unital OU Triton scan (50% fast poles `tau in [1, 32]`, 50% slow poles `tau in [64, 1024]`, exact OU drive gain `sqrt(1 - a^2)`). **Readout:** Chunked FP32 complex `BornAmplitudePOVMHead`. **Metrics:** `stream_q_rms`, `stream_p_rms`, `povm_scale`. |
| `rmsn` | *Planned v0.2 Addition* | Matched `Omega=4/5` Resonant Metriplectic Sheaf Network (`D=512`) baseline for direct head-to-head comparison against DuoNeural's proven `sub-50 PPL` architecture. |

---

## Experimental Safeguards & Verification

1. **Automated Pre-Flight Gate (`tests/test_math_and_grads.py`):** Verifies `0.00e+00` causal prefix leakage across chunk boundaries (`cut=65`), chunked WY DeltaNet fwd/bwd parity (`<= 5.34e-5`), Triton complex scan Wirtinger adjoint parity (`<= 8.58e-6`), chunked Born POVM loss/gradient parity (`<= 9.54e-6`), 6-layer RPP reversible reconstruction drift (`<= 9.16e-5`), parameter alignment (`45.65M–46.10M`), and Step-0 CE proximity to $\ln(49152) = 10.8027$.
2. **Optimizer Partition (`prism/optim.py`):** Muon (5-step BF16 Newton-Schulz) is restricted strictly to interior real 2D mixing matrices; embeddings, projections, Kuramoto couplings, norms, convs, curvature, and POVM temperature use AdamW; scan poles and physics scalars receive `0.1x` LR with `wd = 0.0`.
3. **Pure Validation Tracking:** Held-out evaluation reports pure token cross-entropy (`val_ce`) and perplexity (`val_ppl`) on the final 5% split of `tokens_wiki103_full.pt`, cleanly separated from auxiliary losses.

---

## Smoke-Test Results (`WikiText-103`, 1,500 steps / 12.29M tokens, RTX 4080S)

### v0.1 Controlled Leaderboard
| Arch | Total (Backbone) | Peak VRAM | Speed (tok/s) | Step-0 `val_ce` | Step-250 `val_ppl` | Step-1000 `val_ppl` | Final `val_ce` | Final `val_ppl` | NaNs |
|---|---|---|---|---|---|---|---|---|---|
| `gpt` | `45.95M` (`33.37M`) | `2,884 MB` | `~64,000` | `10.8170` | `442.62` | `147.46` | **`4.8206`** | **`124.04`** | `0` |
| `kcw` | `45.65M` (`33.07M`) | `6,583 MB` | `~36,700` | `10.8131` | **`410.99`** | `170.74` | **`4.9896`** | **`146.87`** | `0` |
| `ldg` | `45.69M` (`33.10M`) | `9,134 MB` | `~20,900` | `10.8133` | `591.53` | `186.37` | **`5.0720`** | **`159.50`** | `0` |
| `rpp` | `46.10M` (`33.52M`) | **`1,394 MB`** | `~26,200` | `11.1639` | `1355.11` | `817.69` | **`6.6231`** | **`752.31`** | `0` |

### v0.2 Observations (In Progress)
- **`rpp v0.2` (`BornAmplitudePOVMHead`)**: **`val_ce = 5.1043` (`val_ppl = 164.73`)** at 1,500 steps (`0` NaNs, **`1,370.4 MB` peak VRAM**, `~27.3k tok/s`) — shattering the v0.1 Welch ceiling (`752.31 -> 164.73`, a **4.57x** perplexity improvement) and beating `gpt` at step 250 (`401.55` vs `442.62`).
- **`kcw v0.2`** running live (`~37.7k tok/s`, `R_order = 0.327`, `frac_bistable = 0.5`, `exp_decay = 0.9696`); **`ldg v0.2`** and **`ldg_povm`** queued next.

---

## Quick Usage (Research Harness)

```bash
# Run mathematical, gradient & causality verification suite
PYTHONPATH=. python3 tests/test_math_and_grads.py

# Launch a matched 1,500-step smoke test
PYTHONPATH=. python3 train_smoke.py --arch gpt --data /workspace/rmsn_data/tokens_wiki103_full.pt --out runs/v01_gpt
PYTHONPATH=. python3 train_smoke.py --arch ldg --data /workspace/rmsn_data/tokens_wiki103_full.pt --out runs/v02_ldg
PYTHONPATH=. python3 train_smoke.py --arch ldg_povm --data /workspace/rmsn_data/tokens_wiki103_full.pt --out runs/v02_ldg_povm
PYTHONPATH=. python3 train_smoke.py --arch kcw --data /workspace/rmsn_data/tokens_wiki103_full.pt --out runs/v02_kcw
PYTHONPATH=. python3 train_smoke.py --arch rpp --data /workspace/rmsn_data/tokens_wiki103_full.pt --out runs/v02_rpp
```
