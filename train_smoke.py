"""
EXPERIMENTAL RESEARCH CODE — NOT FOR PRODUCTION USE.

PRISM Matched Smoke-Test Training Harness
Evaluates GPT-Baseline vs LDG vs KCW vs RPP on WikiText-103 (tokens_wiki103_full.pt, vocab=49152):
- Identical 95% train / 5% held-out validation split (zero sequence overlap across split).
- Identical seed, batch tokens (batch=16 x seq_len=512 = 8192 tok/step), warmup & cosine LR.
- Pure validation CE & PPL logged cleanly separately from auxiliary losses.
- Rich telemetry written to metrics.jsonl + train.log + final qualitative probe.
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from prism.optim import HybridMuonAdamW
from prism.models.baseline_gpt import GPTBaselineLM
from prism.models.ldg import LDGLM
from prism.models.kcw import KCWLM
from prism.models.rpp import RPPLM
from prism.models.rmsn_baseline import RMSNOmega45LM


class ChunkDataset(Dataset):
    def __init__(self, tokens: torch.Tensor, seq_len: int, stride: int):
        self.tokens = tokens
        self.seq_len = seq_len
        self.stride = stride
        self.n = (len(tokens) - seq_len - 1) // stride

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        s = idx * self.stride
        chunk = self.tokens[s: s + self.seq_len + 1]
        return chunk[:-1].long(), chunk[1:].long()


def build_model(arch: str, vocab: int, seq_len: int):
    if arch == "gpt":
        return GPTBaselineLM(vocab_size=vocab, max_seq_len=max(seq_len, 1024))
    if arch == "ldg":
        return LDGLM(vocab_size=vocab, max_seq_len=max(seq_len, 1024), use_povm_head=False)
    if arch == "ldg_povm":
        return LDGLM(vocab_size=vocab, max_seq_len=max(seq_len, 1024), use_povm_head=True)
    if arch == "kcw":
        return KCWLM(vocab_size=vocab, max_seq_len=max(seq_len, 1024))
    if arch == "rpp":
        return RPPLM(vocab_size=vocab, max_seq_len=max(seq_len, 1024))
    if arch == "rmsn":
        return RMSNOmega45LM(vocab_size=vocab)
    raise ValueError(f"Unknown arch: {arch}")


def cosine_lr_scale(step: int, warmup: int, total: int, min_ratio: float = 0.1) -> float:
    if step <= warmup:
        return step / max(warmup, 1)
    prog = (step - warmup) / max(total - warmup, 1)
    return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * min(prog, 1.0)))


@torch.no_grad()
def evaluate(model, val_dl, device, max_batches: int = 40):
    model.eval()
    total_ce = 0.0
    n_b = 0
    for vx, vy in val_dl:
        vx = vx.to(device, non_blocking=True)
        vy = vy.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            _, metrics = model(vx, vy)
        total_ce += metrics["ce_loss"]
        n_b += 1
        if n_b >= max_batches:
            break
    model.train()
    mean_ce = total_ce / max(n_b, 1)
    return mean_ce, math.exp(min(mean_ce, 20.0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", required=True, choices=["gpt", "ldg", "ldg_povm", "kcw", "rpp", "rmsn"])
    p.add_argument("--data", default="/workspace/rmsn_data/tokens_wiki103_full.pt")
    p.add_argument("--out", required=True)
    p.add_argument("--vocab", type=int, default=49152)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--warmup", type=int, default=150)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--adamw-lr", type=float, default=2e-4)
    p.add_argument("--muon-lr", type=float, default=8e-4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    metrics_path = out_dir / "metrics.jsonl"

    def log(msg: str):
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    log(f"PRISM Smoke Test (EXPERIMENTAL RESEARCH) | arch={args.arch} | seed={args.seed}")
    tokens = torch.load(args.data, map_location="cpu")
    n_tok = len(tokens)
    split = int(0.95 * n_tok)
    train_ds = ChunkDataset(tokens[:split], args.seq_len, stride=args.seq_len // 2)
    val_ds = ChunkDataset(tokens[split:], args.seq_len, stride=args.seq_len)
    log(f"Tokens={n_tok:,} | train_examples={len(train_ds):,} | val_examples={len(val_ds):,}")

    g = torch.Generator().manual_seed(args.seed)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, generator=g, drop_last=True, num_workers=2, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, drop_last=True, num_workers=1, pin_memory=True)
    train_iter = iter(train_dl)

    model = build_model(args.arch, args.vocab, args.seq_len).to(device)
    tot_p = sum(p.numel() for p in model.parameters())
    emb_p = model.embed.weight.numel()
    opt = HybridMuonAdamW(model, adamw_lr=args.adamw_lr, muon_lr=args.muon_lr)
    log(f"Params: total={tot_p/1e6:.2f}M | emb={emb_p/1e6:.2f}M | backbone={(tot_p-emb_p)/1e6:.2f}M | groups={opt.counts}")

    val_ce0, val_ppl0 = evaluate(model, val_dl, device)
    log(f"Step 0 | val_ce={val_ce0:.4f} val_ppl={val_ppl0:.2f} (expected ~ln({args.vocab})={math.log(args.vocab):.2f})")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    best_val_ce = float("inf")
    skipped_nan = 0
    consec_nan = 0
    t_win = time.perf_counter()
    toks_win = 0
    ce_win = []
    gn_win = []

    model.train()
    for step in range(1, args.steps + 1):
        scale = cosine_lr_scale(step, args.warmup, args.steps)
        opt.set_lr_scale(scale)

        try:
            bx, by = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dl)
            bx, by = next(train_iter)

        bx = bx.to(device, non_blocking=True)
        by = by.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            if args.arch in ("ldg", "ldg_povm"):
                tssp_s = min(1.0, max(0.0, (step - 150) / 150.0))
                loss, m = model(bx, by, tssp_scale=tssp_s)
            else:
                loss, m = model(bx, by)

        if not torch.isfinite(loss):
            skipped_nan += 1
            consec_nan += 1
            log(f"WARNING step {step}: non-finite loss={loss.item()} (consec={consec_nan})")
            if consec_nan >= 50:
                log("ABORTING: 50 consecutive non-finite losses")
                break
            continue

        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gn):
            opt.zero_grad(set_to_none=True)
            skipped_nan += 1
            consec_nan += 1
            log(f"WARNING step {step}: non-finite grad_norm={gn} (consec={consec_nan})")
            if consec_nan >= 50:
                log("ABORTING: 50 consecutive non-finite grad norms")
                break
            continue

        consec_nan = 0
        opt.step()

        toks_win += bx.numel()
        ce_win.append(m["ce_loss"])
        gn_win.append(float(gn))

        if step % args.log_every == 0 or step == 1:
            dt = max(time.perf_counter() - t_win, 1e-6)
            tps = toks_win / dt
            avg_ce = sum(ce_win) / len(ce_win)
            avg_gn = sum(gn_win) / len(gn_win)
            vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if torch.cuda.is_available() else 0.0
            extra = {k: round(v, 5) if isinstance(v, float) else v for k, v in m.items() if k != "ce_loss"}
            extra_str = " ".join(f"{k}={v}" for k, v in extra.items())
            log(f"step={step:4d}/{args.steps} | ce={avg_ce:.4f} ppl={math.exp(min(avg_ce,20)):.2f} | gn={avg_gn:.3f} | {tps:.0f} tok/s | vram={vram_mb:.0f}MB | {extra_str}")
            rec = {"step": step, "train_ce": avg_ce, "train_ppl": math.exp(min(avg_ce, 20)), "grad_norm": avg_gn, "tps": tps, "vram_mb": vram_mb, **extra}
            with open(metrics_path, "a") as mf:
                mf.write(json.dumps(rec) + "\n")
            t_win = time.perf_counter()
            toks_win = 0
            ce_win.clear()
            gn_win.clear()

        if step % args.eval_every == 0 or step == args.steps:
            val_ce, val_ppl = evaluate(model, val_dl, device)
            is_best = val_ce < best_val_ce
            if is_best:
                best_val_ce = val_ce
                torch.save({"step": step, "arch": args.arch, "model": model.state_dict(), "val_ce": val_ce, "val_ppl": val_ppl}, out_dir / "best.pt")
            log(f"EVAL step={step:4d} | val_ce={val_ce:.4f} val_ppl={val_ppl:.2f} | best_ppl={math.exp(min(best_val_ce,20)):.2f} | skipped_nan={skipped_nan}")
            with open(metrics_path, "a") as mf:
                mf.write(json.dumps({"step": step, "eval": True, "val_ce": val_ce, "val_ppl": val_ppl, "best_val_ce": best_val_ce}) + "\n")

    summary = {
        "arch": args.arch,
        "total_params_M": round(tot_p / 1e6, 2),
        "backbone_params_M": round((tot_p - emb_p) / 1e6, 2),
        "steps": args.steps,
        "tokens_seen": args.steps * args.batch * args.seq_len,
        "best_val_ce": round(best_val_ce, 4),
        "best_val_ppl": round(math.exp(min(best_val_ce, 20)), 2),
        "skipped_nan": skipped_nan,
        "peak_vram_mb": round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 1) if torch.cuda.is_available() else 0.0,
    }
    with open(out_dir / "summary.json", "w") as sf:
        json.dump(summary, sf, indent=2)
    log(f"DONE SUMMARY: {json.dumps(summary)}")


if __name__ == "__main__":
    main()
