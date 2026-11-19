"""Fine-tune a (multilingual) encoder for DiCo-NLI and dump per-instance log-probabilities.

Batches are built from whole reversal groups, so both directions of a pair always share a batch.
With --cons_lambda > 0 a symmetric-KL term ties p(.|a,b) to Rev(p(.|b,a)) (E4); with 0 the
objective is plain cross-entropy and the grouping has no effect on the loss.

Model weights are not saved; log-probabilities for the held-out val slice and the dev split are
written to results/runs/<run_name>/ and all decoding/scoring happens offline (src/evaluate.py).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from dico.common import L2I, REV_PERM, RESULTS, TRACKS, decode, groups_of, load_rows, score, write_logp


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--train_tracks", nargs="+", required=True, choices=TRACKS)
    p.add_argument("--eval_tracks", nargs="+", default=None, choices=TRACKS)
    p.add_argument("--run_name", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--max_len", type=int, default=64)
    p.add_argument("--warmup", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--cons_lambda", type=float, default=0.0)
    p.add_argument(
        "--train_embeddings",
        action="store_true",
        help="also update the (250k-vocab) word embeddings; frozen by default to fit in 18GB unified memory",
    )
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_batches(rows: list[dict], bs: int, rng: random.Random) -> list[tuple[list[int], list[tuple[int, int]]]]:
    """Shuffle reversal groups and pack them into batches; returns (row indices, in-batch pairs)."""
    groups = groups_of(rows)
    rng.shuffle(groups)
    batches, cur, pairs = [], [], []
    for g in groups:
        if len(cur) + len(g) > bs and cur:
            batches.append((cur, pairs))
            cur, pairs = [], []
        if len(g) == 2:
            pairs.append((len(cur), len(cur) + 1))
        cur.extend(g)
    if cur:
        batches.append((cur, pairs))
    return batches


def encode(tok, rows: list[dict], idxs: list[int], max_len: int, device) -> dict:
    enc = tok(
        [rows[i]["text1"] for i in idxs],
        [rows[i]["text2"] for i in idxs],
        padding=True,
        truncation=True,
        max_length=max_len,
        pad_to_multiple_of=16,  # few distinct shapes -> bounded MPS kernel/buffer cache
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in enc.items()}


@torch.no_grad()
def predict_logp(model, tok, rows: list[dict], max_len: int, device, bs: int = 128) -> np.ndarray:
    model.eval()
    out = []
    for s in range(0, len(rows), bs):
        idxs = list(range(s, min(s + bs, len(rows))))
        logits = model(**encode(tok, rows, idxs, max_len, device)).logits.float()
        out.append(F.log_softmax(logits, dim=-1).cpu().numpy())
    model.train()
    return np.concatenate(out)


def main() -> None:
    args = parse_args()
    eval_tracks = args.eval_tracks or args.train_tracks
    set_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    out_dir = RESULTS / "runs" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    fit = [r for t in args.train_tracks for r in load_rows("train", t, "fit")]
    val = {t: load_rows("train", t, "val") for t in args.train_tracks}
    y_fit = torch.tensor([L2I[r["label"]] for r in fit])
    rev_perm = torch.tensor(REV_PERM, device=device)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=4, ignore_mismatched_sizes=True
    ).to(device)
    if not args.train_embeddings:
        model.get_input_embeddings().weight.requires_grad_(False)
    model.train()

    rng = random.Random(args.seed)
    steps_per_epoch = math.ceil(len(fit) / args.bs)
    total = steps_per_epoch * args.epochs
    no_decay = ("bias", "LayerNorm.weight", "layernorm", "layer_norm")
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params = [
        {"params": [p for n, p in named if not any(k in n for k in no_decay)], "weight_decay": args.wd},
        {"params": [p for n, p in named if any(k in n for k in no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(params, lr=args.lr)
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup * total), total)

    best_f1, best_state, log = -1.0, None, []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        ce_sum = kl_sum = 0.0
        batches = make_batches(fit, args.bs, rng)
        for idxs, pairs in batches:
            logits = model(**encode(tok, fit, idxs, args.max_len, device)).logits.float()
            loss = F.cross_entropy(logits, y_fit[idxs].to(device))
            ce_sum += loss.item()
            if args.cons_lambda > 0 and pairs:
                logp = F.log_softmax(logits, dim=-1)
                a = torch.tensor([i for i, _ in pairs], device=device)
                b = torch.tensor([j for _, j in pairs], device=device)
                lp_a, lp_b_rev = logp[a], logp[b][:, rev_perm]
                kl = 0.5 * (
                    F.kl_div(lp_b_rev, lp_a, log_target=True, reduction="batchmean")
                    + F.kl_div(lp_a, lp_b_rev, log_target=True, reduction="batchmean")
                )
                kl_sum += kl.item()
                loss = loss + args.cons_lambda * kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

        val_scores = {}
        for t, rows in val.items():
            preds = decode(rows, predict_logp(model, tok, rows, args.max_len, device), "indep")
            val_scores[t] = {k: v for k, v in score("train", t, preds).items() if k != "confusion"}
        mean_f1 = float(np.mean([s["weighted_f1"] for s in val_scores.values()]))
        entry = {
            "epoch": epoch,
            "ce": ce_sum / len(batches),
            "kl": kl_sum / len(batches),
            "val_mean_weighted_f1": mean_f1,
            "val": val_scores,
            "elapsed_s": round(time.time() - t0, 1),
        }
        log.append(entry)
        print(json.dumps({k: v for k, v in entry.items() if k != "val"}), flush=True)
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    best_epoch = max(log, key=lambda e: e["val_mean_weighted_f1"])["epoch"]
    for t in sorted(set(eval_tracks) | set(args.train_tracks)):
        for split, part in [("train", "val"), ("dev", "all")]:
            rows = load_rows(split, t, part)
            name = "val" if split == "train" else "dev"
            write_logp(out_dir / f"logp_{name}_{t}.csv", rows, predict_logp(model, tok, rows, args.max_len, device))
    (out_dir / "train_log.json").write_text(
        json.dumps({"best_epoch": best_epoch, "best_val_mean_weighted_f1": best_f1, "epochs": log}, indent=2) + "\n"
    )
    print(f"done: best epoch {best_epoch}, val F1 {best_f1:.4f}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
