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
| `rpp_delta` | **RevPort-Delta-POVM (`RPP-Delta`, v0.4)** | Alternates `G` blocks between Dual-Timescale Unital OU Triton Scans (odd layers) and Chunk-64 WY Gated Delta-Rule + Lorentz Log-Map associative memory (even layers) inside the bit-exact FP32 reversible `(q, p)` carrier + `BornAmplitudePOVMHead` (`46.19M` params). |
| `rmsn` | **RMSN (`Omega = 4/5`)** | Matched `Omega = D / (M_w * L) = 640 / (80 * 10) = 0.80 = 4/5` Resonant Metriplectic Sheaf Network (`49.28M` params) wrapping hardened `rmsn_core.py` (`CliffordStalkBundle`, `MetriplecticParallelScanFn`, `FractionalCaputoBank`, `CausalTrophicGlobalWorkspace`) inside PRISM's shared tied `49152 <-> 256` head. **Metrics:** `lyap`, `mean_energy`, `ws_entropy`, `omega`. |

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

### v0.2 & `RMSN` Benchmark Results (`1,500` steps, `12.29M` tokens, `0` NaNs)
| Arch | Total (Backbone) | Peak VRAM | Speed (tok/s) | Step-250 `val_ppl` | Step-1000 `val_ppl` | Final `val_ce` | Final `val_ppl` | Notes |
|---|---|---|---|---|---|---|---|---|
| `ldg` v0.2 | `45.69M` (`33.11M`) | `9,525 MB` | `~20,800` | — | — | **`4.9869`** | **`146.48`** | `-13.02 PPL` vs v0.1; `kappa: 3.8065 -> 3.8847`, `alpha=0.875`, `ds_dt=6.16` |
| `kcw` v0.2 | `45.65M` (`33.07M`) | `6,583 MB` | `~37,400` | — | — | **`5.0339`** | **`153.52`** | Ablation: 50/50 damped/bistable (`-0.25/+0.25`) slightly worse than v0.1 `+0.25` (`146.87`) |
| `rpp` v0.2 | `46.10M` (`33.52M`) | **`1,370 MB`** | `~27,200` | **`401.55`** | `190.29` | **`5.1043`** | **`164.73`** | **`-587.58 PPL` vs v0.1** (`BornAmplitudePOVMHead` broke Welch floor) |
| `rmsn` (`Ω=4/5`) | `49.28M` (`36.69M`) | `19,599 MB` | `~11,500` | `640.57` | `208.53` | **`5.2210`** | **`185.13`** | `lyap: -0.0699 -> -0.00123` (TRUE SILENT); `ws_entropy: 4.382 -> 4.249` (needs 10k-50k+ for full soliton condensation) |
| `ldg_povm` v0.2 | `45.69M` (`33.11M`) | `9,721 MB` | `~17,100` | `2110.26` | `330.61` | **`5.6046`** | **`271.67`** | Unscaled residuals spiked early `gn` -> fixed with `0.02/sqrt(2L)` + `25x` `tau_povm` LR in `v0.3` |

### v0.3 Leaderboard (`1,500` steps, `12.29M` tokens, `0` NaNs — Quantum POVM Beats Full-Attention GPT)
| Arch | Total (Backbone) | Peak VRAM | Speed (tok/s) | Final `val_ce` | Final `val_ppl` | Key Finding |
|---|---|---|---|---|---|---|
| **`rpp` v0.3** | `46.10M` (`33.52M`) | **`1,370 MB`** | `~26,800` | **`4.8053`** | **`122.15`** | **Beats matched Causal GPT (`124.04`) by `-1.89 PPL`** with no global attention & `2.1x` less VRAM (`povm_scale: 12.0 -> 16.03`) |
| `gpt` (Baseline) | `45.95M` (`33.37M`) | `2,884 MB` | `~64,000` | `4.8206` | `124.04` | Full $O(T^2)$ causal RoPE-SwiGLU reference |
| `ldg` v0.2 (`H=8`) | `45.69M` (`33.11M`) | `9,525 MB` | `~20,800` | `4.9869` | `146.48` | `v0.3` ablation (`H=16, dh=32`, bottleneck GENERIC) scored `151.71`, confirming `dh=64` matrix capacity + full GENERIC (`ds_dt ~ 6`) |
| `kcw` v0.1 (`H=8`) | `45.65M` (`33.07M`) | `6,583 MB` | `~36,700` | `4.9896` | `146.87` | `v0.3` ablation (`H=16`) scored `152.29` (`R_order: 0.320 -> 0.227`), confirming `H=8` deeper oscillator banks |
| `rmsn` (`Ω=4/5`) | `49.28M` (`36.69M`) | `19,599 MB` | `~11,500` | `5.2210` | `185.13` | `lyap: -0.0699 -> -0.00123` (TRUE SILENT); `ws_entropy: 4.382 -> 4.249` (needs 10k-50k+ for full soliton condensation) |
| `ldg_povm` v0.3 | `45.28M` (`32.70M`) | `11,099 MB` | `~16,400` | `5.4633` | `235.87` | Improved `-35.8 PPL` vs `v0.2` (`271.67`), `povm_scale -> 15.5`, though unattenuated TSSP+GENERIC gradients (`gn=22` early) still hinder unit-sphere POVM head |

### v0.4 Hybrid Check (`rpp_delta`, `1,500` steps) & Extended 5k Showdown (`40.96M` tokens)
- **`rpp_delta` (`47.56M`, `1,534 MB` VRAM, `~19.5k tok/s`)**: `val_ce = 4.8390`, **`val_ppl = 126.34`** (`0` NaNs, `povm_scale -> 16.005`). Clean ablation: substituting half of `rpp`'s Dual-Timescale OU scans with WY Gated DeltaNet (`H=8`) slowed throughput and cost `+4.19 PPL` vs pure `rpp v0.3` (`122.15 PPL`), proving the **Dual-Timescale OU Scan + W=64 Attention + Born-Amplitude POVM** synergy.
### 5,000-Step (`40.96M` tokens) Championship Showdown (`p19285`, Completed + Full-Val `774`-batch Verified)
| Rank | Arch | Total (Backbone) | Peak VRAM | Speed (tok/s) | 40-batch `val_ce` / `val_ppl` | **Full 774-batch `val_ce` / `val_ppl`** | Key Finding |
|---|---|---|---|---|---|---|---|
| 1 | `gpt` | `45.95M` (`33.37M`) | `2,884 MB` | `~64,400` | `3.9637` / `52.65` | **`3.9266` / `50.73`** | $O(T^2)$ exact causal attention overtakes after step 500–1,000 |
| 2 | **`rpp` (`W=256` + POVM)** | `46.10M` (`33.52M`) | **`1,370 MB`** | `~27,400` | **`4.0199` / `55.70`** | **`3.9734` / `53.16`** | Saturates `-0.42 PPL` past `W=128` with identical `1,370 MB` VRAM (`povm_scale -> 25.42`) |
| 3 | **`rpp` (`W=128` + POVM)** | `46.10M` (`33.52M`) | **`1,370 MB`** | **`~27,700`** | `4.0228` / `55.86` | **`3.9811` / `53.58`** | **Zero extra params/VRAM vs `W=64`, closes 48% of gap to GPT (`-2.59 PPL`)** with faster speed and `2.1×` less VRAM than GPT (`povm_scale -> 25.42`) |
| 4 | **`rpp`** (`W=64` + POVM) | `46.10M` (`33.52M`) | **`1,370 MB`** | `~26,900` | `4.0699` / `58.55` | **`4.0284` / `56.17`** | **Beats linear-head `rpp_lin` by `-1.04 PPL` on full val**; `2.1×` less VRAM than GPT, leads early (`122.15 < 124.04` at 1.5k) |
| 5 | `rpp_lin` (`W=64` + Linear) | `46.10M` (`33.52M`) | **`1,174 MB`** | `~36,000` | `4.0835` / `59.35` | **`4.0468` / `57.21`** | Controlled head ablation: linear softmax starts faster (`210.96` vs `232.52` at 500) before Born POVM overtakes at step ~2,000–2,500 |
| 6 | **`ldg`** | `45.69M` (`33.11M`) | `9,525 MB` | `~20,700` | `4.2524` / `70.27` | **`4.2175` / `67.87`** | Pure linear $O(T)$ (`0` attention); `kappa: 3.8065 -> 4.1623`, `{S,E} <= 9e-5`, `ds_dt -> 58.84`, `tssp -> 0.249` |
| 7 | **`kcw`** | `45.65M` (`33.07M`) | `6,583 MB` | `~38,000` | `4.3669` / `78.80` | **`4.3288` / `75.86`** | Non-attention complex harmonic + Kuramoto phase lock held `R_order = 0.3223` (`0` NaNs across 40.96M tokens) |
| 8 | **`rmsn`** (`Ω=4/5`) | `49.28M` (`36.69M`) | `19,599 MB` | `~11,500` | `4.5935` / `98.84` | **`4.5618` / `95.75`** | `lyap` locked TRUE SILENT `-0.00104`, `ws_entropy: 4.382 -> 1.757` (soliton condensation confirmed!) |

### 10,000-Step (`81.92M` tokens) Extended Showdown (`p19285`, Complete + Full-Val `774`-batch Verified)
| Arch | Total (Backbone) | Peak VRAM | Speed (tok/s) | 40-batch `val_ce` / `val_ppl` | **Full 774-batch `val_ce` / `val_ppl`** | Key Finding |
|---|---|---|---|---|---|---|
| **`gpt`** | `45.95M` (`33.37M`) | `2,884 MB` | `~64,000` | `3.6615` / `38.92` | **`3.6212` / `37.38`** | Clean scaling (`-13.35 PPL` from 5k); global causal attention widens edge to `0.0911 nats` over local `W=128` |
| **`rpp` (`W=128` + POVM)** | `46.10M` (`33.52M`) | **`1,370 MB`** | `~27,900` | `3.7636` / `43.10` | **`3.7123` / `40.95`** | **Smashes 5k GPT (`50.73`) by nearly 10 PPL at `2.1×` less VRAM** (`povm_scale -> 38.53`, `-12.63 PPL` from 5k, `0` NaNs) |
| **`rmsn`** (`Ω=4/5`) | `49.28M` (`36.69M`) | `19,599 MB` | `~11,500` | `4.2720` / `71.67` | **`4.2339` / `68.98`** | `-26.77 PPL` from 5k; `lyap` locked `-0.00100`, **`ws_entropy` stabilized at `1.52–1.64` (`~5.0` effective slots) from step 3,600–10,000**; fluent grammar without prompt domain-lock yet |

### v0.5–v0.6 Global-Bridge & Ceiling Ablation (`p19285`, 10k `81.92M`-token Verified)
Adding `1`, `2`, `5`, or all `10` global causal layers inside the exact same **`46.10M`-param** reversible `(q,p)` Port-Hamiltonian stack cleanly isolates the retrieval vs FFN/stream gap to `gpt` (`37.38 PPL` at `2,884 MB`):
| Arch | Global Layers | Peak VRAM | Speed (tok/s) | 40-batch `val_ce` / `val_ppl` | **Full 774-batch `val_ce` / `val_ppl`** | Key Finding |
|---|---|---|---|---|---|---|
| **`rpp` (`W=128`)** | None (`0/10`) | `1,370 MB` | `~27,900` | `3.7636` / `43.10` | **`3.7123` / `40.95`** | Pure local `W=128` + OU scan + Born-POVM baseline |
| **`rpp_bridge_1g`** | `L9` (`1/10`) | **`1,376 MB`** | `~27,700` | `3.7363` / `41.94` | **`3.6845` / `39.83`** | **Pareto sweet spot (`+6 MB`, sub-40 PPL):** captures **89%** of the total global-attention gain (`-1.12` of `-1.26 PPL`) right before POVM readout |
| **`rpp_bridge_2g`** | `L4 + L9` (`2/10`) | **`1,376 MB`** | `~27,300` | `3.7346` / `41.87` | **`3.6833` / `39.78`** | Saturates `-0.05 PPL` past `1g` (`L4` global writes only enter `q` via diagonal OU) |
| **`rpp_bridge_5g`** | `L1,3,5,7,9` (`5/10`) | **`1,378 MB`** | `~27,700` | **`3.7274` / `41.57`** | **`3.6752` / `39.46`** | **Best RPP variant so far — BEATS `global_all` (`39.69`) by `-0.23 PPL`!** Alternating local `W=128` + global forms an optimal hybrid U-curve |
| **`rpp_global_all`** | All (`10/10`) | **`1,379 MB`** | **`~28,400`** | `3.7341` / `41.85` | **`3.6810` / `39.69`** | **Definitive saturation:** remaining `2.31 PPL` (`0.0598 nat`) gap to `gpt` is **structural** (`ffn_dim=480` bottleneck / `(q,p)` split), motivating `v0.7` |

### v0.7 Symplectic Wide-FFN Interleaving (`p19285`, Complete `2×2` Falsification)
Tests whether eliminating `F(p)`'s split `480`-dim FFN (`f_ffn_dim=0`, pure Causal Attention) and consolidating associative capacity into a **2× wider SwiGLU (`g_ffn_dim=960`)** inside `G(q)` (`OU + Wide-SwiGLU`, **`46.09M` exact parity**, `~1.38 GB` VRAM) closes the remaining structural gap to `gpt` (`37.38 PPL`).
| Arch | Global Layers | Peak VRAM | Speed (tok/s) | 40-batch `val_ce` / `val_ppl` | **Full 774-batch `val_ce` / `val_ppl`** | Key Finding |
|---|---|---|---|---|---|---|
| **`rpp_symp_ffn_1g`** | `L9` (`1/10`) | `1,377 MB` | **`~30,300`** | `3.7565` / `42.80` | **`3.7042` / `40.62`** | **Falsified (`+0.79 PPL` vs `rpp_bridge_1g` `39.83`):** `+10%` faster, but removing `F(p)`'s nonlinear FFN starves `q`-stream mixing (`stream_q_rms` drops `0.74` vs `p`'s `2.69`) |
| **`rpp_symp_ffn_global`** | All (`10/10`) | `1,379 MB` | **`~30,600`** | `3.7585` / `42.88` | **`3.7054` / `40.67`** | **Confirmed falsification (`+0.98 PPL` vs `rpp_global_all` `39.69`):** proves BOTH `q` and `p` streams require nonlinear FFN channel mixing |

### v0.8 Exact Symplectic Givens `(q,p)` Mixing + Compact-OU Wide FFN (`p19285`, In Progress)
Builds directly on top of **`rpp_bridge_5g`** (`39.46 PPL` champion) without starving either stream:
| Arch | Config | Peak VRAM | Speed (tok/s) | 40-batch `val_ce` / `val_ppl` | **Full 774-batch `val_ce` / `val_ppl`** | Key Finding |
|---|---|---|---|---|---|---|
| **`rpp_5g_symp_mix`** | `R(θ)` only (`46.10M`) | `1,378 MB` | `~27,000` | `3.7299` / `41.67` | **`3.6789` / `39.60`** | Learned `|mix_thetas|` stayed tiny (`0.0074–0.0146` rad, `max=0.0451`); rotation alone didn't beat `bridge_5g` (`39.46`) |
| **`rpp_5g_wide_ou`** | `d_scan=128`, `f=g=672` (`46.10M`) | `1,383 MB` | `~27,600` | *Running...* | *Running...* | Launched `00:21Z`; faster early drop (`786.36` vs `797.99` at step 150, `gn=1.429`) |

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
PYTHONPATH=. python3 train_smoke.py --arch rpp_bridge_1g --window 128 --steps 10000 --data /workspace/rmsn_data/tokens_wiki103_full.pt --out runs/v05_10k_rpp_bridge_1g
PYTHONPATH=. python3 train_smoke.py --arch rpp_bridge_2g --window 128 --steps 10000 --data /workspace/rmsn_data/tokens_wiki103_full.pt --out runs/v05_10k_rpp_bridge_2g
PYTHONPATH=. python3 train_smoke.py --arch rpp_bridge_5g --window 128 --steps 10000 --data /workspace/rmsn_data/tokens_wiki103_full.pt --out runs/v06_10k_rpp_bridge_5g
PYTHONPATH=. python3 train_smoke.py --arch rpp_global_all --window 0 --steps 10000 --data /workspace/rmsn_data/tokens_wiki103_full.pt --out runs/v06_10k_rpp_global_all
PYTHONPATH=. python3 train_smoke.py --arch rpp_symp_ffn_1g --window 128 --steps 10000 --data /workspace/rmsn_data/tokens_wiki103_full.pt --out runs/v07_10k_rpp_symp_ffn_1g
PYTHONPATH=. python3 train_smoke.py --arch rpp_symp_ffn_global --window 0 --steps 10000 --data /workspace/rmsn_data/tokens_wiki103_full.pt --out runs/v07_10k_rpp_symp_ffn_global
PYTHONPATH=. python3 train_smoke.py --arch rpp_5g_symp_mix --window 128 --steps 10000 --data /workspace/rmsn_data/tokens_wiki103_full.pt --out runs/v08_10k_rpp_5g_symp_mix
PYTHONPATH=. python3 train_smoke.py --arch rpp_5g_wide_ou --window 128 --steps 10000 --data /workspace/rmsn_data/tokens_wiki103_full.pt --out runs/v08_10k_rpp_5g_wide_ou
```
