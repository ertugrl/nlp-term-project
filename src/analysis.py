"""Error analysis on dev: pair-level error decomposition and per-language-combination scores.

Pair-level outcome categories for each gold reversible pair (x, x_rev):
  correct      : both directions right (HardCons hit)
  neg          : at least one direction predicted NEGATIVE_OTHER
  inconsistent : both reversible labels but f(x_rev) != Rev(f(x))  (e.g. FWD/FWD)
  consistent_wrong : consistent but wrong relation (e.g. EQ/EQ for gold FWD/BWD)
Averaged over the seeds of a config; written to results/analysis.json and printed.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from dico.common import RESULTS, TRACKS, decode, groups_of, load_rows, read_logp, reverse_label, score

SEED_RE = re.compile(r"^(?P<config>.+)_s(?P<seed>\d+)$")
CATS = ["correct", "neg", "inconsistent", "consistent_wrong"]


def decompose(rows: list[dict], preds: dict[str, str]) -> dict[str, float]:
    counts = defaultdict(int)
    n = 0
    for g in groups_of(rows):
        if len(g) != 2:
            continue
        a, b = rows[g[0]], rows[g[1]]
        pa, pb = preds[a["instance_id"]], preds[b["instance_id"]]
        n += 1
        if pa == a["label"] and pb == b["label"]:
            counts["correct"] += 1
        elif "NEGATIVE_OTHER" in (pa, pb):
            counts["neg"] += 1
        elif pb != reverse_label(pa):
            counts["inconsistent"] += 1
        else:
            counts["consistent_wrong"] += 1
    return {c: counts[c] / n for c in CATS}


def neg_detection(rows: list[dict], preds: dict[str, str]) -> dict[str, float]:
    tp = sum(1 for r in rows if r["label"] == "NEGATIVE_OTHER" and preds[r["instance_id"]] == "NEGATIVE_OTHER")
    pp = sum(1 for r in rows if preds[r["instance_id"]] == "NEGATIVE_OTHER")
    gp = sum(1 for r in rows if r["label"] == "NEGATIVE_OTHER")
    p, rc = tp / max(pp, 1), tp / max(gp, 1)
    return {"neg_precision": p, "neg_recall": rc, "neg_f1": 2 * p * rc / max(p + rc, 1e-9)}


def lang_combo_scores(rows: list[dict], preds: dict[str, str]) -> dict[str, dict]:
    """Mixed track: score each unordered language combination separately."""
    by_combo = defaultdict(dict)
    for r in rows:
        combo = "-".join(sorted({r["text1_lang"], r["text2_lang"]}))
        by_combo[combo][r["instance_id"]] = preds[r["instance_id"]]
    return {c: {k: v for k, v in score("dev", "track4", p).items() if k in ("weighted_f1", "soft_cons", "hard_cons")}
            for c, p in sorted(by_combo.items())}


def main() -> None:
    runs = defaultdict(list)
    for d in sorted((RESULTS / "runs").iterdir()):
        m = SEED_RE.match(d.name)
        if m and (d / "train_log.json").exists():
            runs[m["config"]].append(d)

    out: dict = {}
    for config, dirs in sorted(runs.items()):
        # config looks like <model>_<track>
        track = config.rsplit("_", 1)[-1]
        if track not in TRACKS:
            continue
        rows = load_rows("dev", track)
        res = defaultdict(lambda: defaultdict(list))
        for d in dirs:
            f = d / f"logp_dev_{track}.csv"
            if not f.exists():
                continue
            lp = read_logp(f, rows)
            for mode in ["indep", "joint", "struct"]:
                preds = decode(rows, lp, mode)
                for k, v in {**decompose(rows, preds), **neg_detection(rows, preds)}.items():
                    res[mode][k].append(v)
                if track == "track4":
                    for combo, s in lang_combo_scores(rows, preds).items():
                        for k, v in s.items():
                            res[mode][f"{combo}:{k}"].append(v)
        out[config] = {mode: {k: float(np.mean(v)) for k, v in d.items()} for mode, d in res.items()}
        print(config)
        for mode, d in out[config].items():
            print(f"  {mode:6s} " + " ".join(f"{k}={100 * v:.1f}" for k, v in d.items()))
    (RESULTS / "analysis.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
