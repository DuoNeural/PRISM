"""
EXPERIMENTAL RESEARCH CODE — NOT FOR PRODUCTION USE.

PRISM Post-Training Evaluation & Qualitative Generation Probe
Runs full held-out WikiText-103 validation pass (all batches) + side-by-side
top-p continuation sampling across trained PRISM checkpoints.
"""
import argparse
import math
from pathlib import Path
import torch
import torch.nn.functional as F

from train_smoke import ChunkDataset, build_model


@torch.no_grad()
def full_val_eval(model, val_tokens: torch.Tensor, seq_len: int, batch_size: int, device: torch.device):
    ds = ChunkDataset(val_tokens, seq_len, stride=seq_len)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=True)
    model.eval()
    total_ce = 0.0
    n_b = 0
    for vx, vy in dl:
        vx, vy = vx.to(device), vy.to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            _, m = model(vx, vy)
        total_ce += m["ce_loss"]
        n_b += 1
    mean_ce = total_ce / max(n_b, 1)
    return mean_ce, math.exp(min(mean_ce, 20.0)), n_b


@torch.no_grad()
def sample_text(model, tok, prompt: str, max_new: int = 80, temp: float = 0.8, top_k: int = 40, device="cuda"):
    model.eval()
    ids = tok.encode(prompt, return_tensors="pt").to(device)
    for _ in range(max_new):
        x_in = ids[:, -512:]
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(x_in)
            logits = out[0] if isinstance(out, tuple) else out
        next_logits = logits[:, -1, :].float() / max(temp, 1e-4)
        if top_k > 0:
            v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits[next_logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(next_logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, nxt], dim=1)
    return tok.decode(ids[0].tolist(), skip_special_tokens=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/workspace/rmsn_data/tokens_wiki103_full.pt")
    p.add_argument("--tokenizer", default="HuggingFaceTB/cosmo2-tokenizer")
    p.add_argument("--cache-dir", default="/workspace/rmsn_data/hf")
    p.add_argument("--window", type=int, default=None, help="Override sliding window if not saved in checkpoint")
    p.add_argument("--runs", nargs="+", required=True, help="List of run dirs with best.pt")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokens = torch.load(args.data, map_location="cpu")
    val_tokens = tokens[int(0.95 * len(tokens)):]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, cache_dir=args.cache_dir, local_files_only=True)

    prompts = [
        "The history of quantum mechanics began in",
        "In computer science, a balanced tree is",
    ]

    for rdir in args.runs:
        ckpt_path = Path(rdir) / "best.pt"
        if not ckpt_path.exists():
            print(f"[SKIP] {ckpt_path} not found")
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu")
        arch = ckpt["arch"]
        win = args.window if args.window is not None else ckpt.get("window", 64)
        model = build_model(arch, vocab=49152, seq_len=512, window=win).to(device)
        model.load_state_dict(ckpt["model"])
        ce, ppl, nb = full_val_eval(model, val_tokens, seq_len=512, batch_size=16, device=device)
        print(f"\n=== {rdir} (arch={arch}, step={ckpt['step']}) | FULL VAL ({nb} batches): CE={ce:.4f} PPL={ppl:.2f} ===")
        for pr in prompts:
            gen = sample_text(model, tok, pr, max_new=60, device=device)
            print(f"  Prompt: {pr!r}\n  -> {gen}\n")


if __name__ == "__main__":
    main()
