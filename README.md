# PRISM — Physics-Regulated Information-State Models

> [!CAUTION]
> **STRICTLY EXPERIMENTAL RESEARCH — WORK IN PROGRESS — NOT FOR PRODUCTION USE.**
>
> Everything in this repository is an active, rapidly evolving architecture research prototype under live ablation on RTX 4080 Super hardware. APIs, mathematical formulations, checkpoints, and training recipes will change without notice. None of these models are hardened, evaluated for safety, or intended for production deployment.

---

## Research Thesis

`PRISM` synthesizes DuoNeural's empirical sequence-modeling findings (`RMSN`, `CRSN`, `SRSN`, `QSMN`, `PULSE`, `Bismuth-104M`, and Dynamical Horizon Principle papers 24–41) with 2025–2026 linear-state architectures into three distinct physics-regulated non-Transformer candidates, benchmarked against a strictly parameter- and token-matched Causal Transformer baseline on WikiText-103 (`vocab = 49152`, `cosmo2-tokenizer`, `seq_len = 512`).

### Benchmark Suite (`~46M–48M` Total Params, Identical Factorized Tied `V×256` Embedding)

| Key | Architecture | Core Mechanism |
|---|---|---|
| `gpt` | **Causal RoPE-SwiGLU Baseline** | Standard Flash/SDPA causal attention + RoPE + SwiGLU MLP |
| `ldg` | **Lorentz-Delta-TSSP GENERIC (`LDG`)** | FP32 chunked-64 WY Gated DeltaNet + cancellation-free Lorentz origin log-map + CliffordNet geometric product mixer + collapse-proof TSSP self-prediction |
| `kcw` | **Kuramoto-Clifford WaveNet (`KCW`)** | Coalesced Triton complex harmonic associative scan + singularity-free algebraic Kuramoto inter-head phase coupling + exact Bernoulli ODE radial double-well bifurcation + Taylor-sinc $Cl_{3,0}$ rotor |
| `rpp` | **RevPort-POVM Hybrid (`RPP`)** | Bit-exact FP32 reversible Port-Hamiltonian streams + causal sliding-window $W=64$ RoPE attention + dual-timescale unital OU Triton scan + chunked FP32 complex Born-Log POVM head |

---

## Experimental Safeguards Built Into `PRISM`

1. **Zero Embedding Confounder:** All four models share an identical factorized tied `49152 → 256` embedding bottleneck (`12.58M` params) so backbone capacity (`~33M–35M` params) is directly comparable.
2. **Numerical Stability & Singularity Guards:** Log-space FP32 decay scans, `atan2`-free complex phasor coupling, exact Bernoulli closed-form double-well integration, Taylor-sinc $Cl_{3,0}$ rotors, and shifted bounded Born-Log POVM scaling.
3. **Causality & Unit Verification:** `tests/test_math_and_grads.py` verifies zero future-token leakage, Triton/WY equivalence to sequential recurrences, reversible reconstruction drift (`< 1e-4`), and GENERIC Poisson-orthogonality residuals (`< 1e-6`).
4. **Controlled Validation:** Evaluates pure held-out cross-entropy loss and perplexity on the final 5% split of WikiText-103 (`tokens_wiki103_full.pt`), keeping auxiliary losses separate from reported `val_ppl`.

---

## Quick Usage (Research Harness)

```bash
# Run mathematical & causality verification suite
pytest -s tests/test_math_and_grads.py

# Launch a matched 3,000-step smoke test
python3 train_smoke.py --arch gpt --data /workspace/rmsn_data/tokens_wiki103_full.pt --out runs/gpt_smoke
python3 train_smoke.py --arch ldg --data /workspace/rmsn_data/tokens_wiki103_full.pt --out runs/ldg_smoke
python3 train_smoke.py --arch kcw --data /workspace/rmsn_data/tokens_wiki103_full.pt --out runs/kcw_smoke
python3 train_smoke.py --arch rpp --data /workspace/rmsn_data/tokens_wiki103_full.pt --out runs/rpp_smoke
```
