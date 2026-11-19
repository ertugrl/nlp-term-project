"""E0: trivial baselines on the dev split of every track (official scorer)."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np

from dico.common import LABELS, RESULTS, TRACKS, decode, groups_of, load_rows, score


def run() -> dict:
    out: dict = {}
    for track in TRACKS:
        train = load_rows("train", track)
        dev = load_rows("dev", track)
        majority = Counter(r["label"] for r in train).most_common(1)[0][0]
        res = {}

        res["majority"] = score("dev", track, {r["instance_id"]: majority for r in dev})

        rand = []
        for seed in range(5):
            rng = np.random.default_rng(seed)
            rand.append(score("dev", track, {r["instance_id"]: LABELS[rng.integers(4)] for r in dev}))
        res["random"] = {k: float(np.mean([s[k] for s in rand])) for k in ["weighted_f1", "soft_cons", "hard_cons"]}

        # Constant EQUIVALENCE: self-consistent by construction, ignores the input entirely.
        res["constant_eq"] = score("dev", track, {r["instance_id"]: "EQUIVALENCE" for r in dev})

        # Structure-only heuristic: one-direction groups -> NEGATIVE_OTHER, else EQUIVALENCE.
        flat = np.zeros((len(dev), 4))
        res["structure_only"] = score("dev", track, decode(dev, flat + np.log([0.4, 0.3, 0.3, 1e-9]), "struct"))

        out[track] = res
        sizes = Counter(len(g) for g in groups_of(dev))
        print(track, "group sizes", dict(sizes))
        for name, s in res.items():
            print(f"  {name:15s} F1={s['weighted_f1']:.3f} Soft={s['soft_cons']:.3f} Hard={s['hard_cons']:.3f}")
    return out


if __name__ == "__main__":
    scores = run()
    path = RESULTS / "baselines" / "scores_dev.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    for t in scores.values():
        for s in t.values():
            s.pop("confusion", None)
    path.write_text(json.dumps(scores, indent=2) + "\n")
    print("wrote", path)
