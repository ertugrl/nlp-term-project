"""Decode saved log-probabilities with every strategy and score them with the official metrics.

Writes results/runs/<run>/scores.json for each run, plus seed-ensembles (mean of probabilities
over runs named <config>_s<seed>) under results/ensembles/<config>/scores.json.
Also writes official-format dev prediction CSVs for the joint decoder.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from dico.common import RESULTS, decode, load_rows, read_logp, score, write_predictions

MODES = ["indep", "joint", "struct"]
SEED_RE = re.compile(r"^(?P<config>.+)_s(?P<seed>\d+)$")


def rows_for(split: str, track: str) -> list[dict]:
    return load_rows("train", track, "val") if split == "val" else load_rows("dev", track)


def score_logps(logps: dict[tuple[str, str], np.ndarray], pred_dir: Path | None = None) -> dict:
    out: dict = defaultdict(dict)
    for (split, track), lp in sorted(logps.items()):
        rows = rows_for(split, track)
        out[split][track] = {}
        for mode in MODES:
            preds = decode(rows, lp, mode)
            out[split][track][mode] = score("train" if split == "val" else "dev", track, preds)
            if pred_dir is not None and split == "dev" and mode in ("indep", "joint"):
                write_predictions(pred_dir / f"pred_dev_{track}_{mode}.csv", preds)
    return out


def load_run(run_dir: Path) -> dict[tuple[str, str], np.ndarray]:
    logps = {}
    for f in run_dir.glob("logp_*_track*.csv"):
        _, split, track = f.stem.split("_")
        logps[(split, track)] = read_logp(f, rows_for(split, track))
    return logps


def main() -> None:
    by_config: dict[str, list[dict]] = defaultdict(list)
    for run_dir in sorted((RESULTS / "runs").iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("_") or not (run_dir / "train_log.json").exists():
            continue
        logps = load_run(run_dir)
        scores = score_logps(logps, pred_dir=run_dir)
        (run_dir / "scores.json").write_text(json.dumps(scores, indent=2) + "\n")
        m = SEED_RE.match(run_dir.name)
        if m:
            by_config[m["config"]].append(logps)
        print("scored", run_dir.name)

    for config, runs in by_config.items():
        if len(runs) < 2:
            continue
        keys = set.intersection(*(set(r) for r in runs))
        # Ensemble = log of the mean probability over seeds.
        ens = {k: np.log(np.mean([np.exp(r[k]) for r in runs], axis=0)) for k in keys}
        d = RESULTS / "ensembles" / config
        d.mkdir(parents=True, exist_ok=True)
        (d / "scores.json").write_text(json.dumps(score_logps(ens), indent=2) + "\n")
        print("ensembled", config, f"({len(runs)} seeds)")


if __name__ == "__main__":
    main()
