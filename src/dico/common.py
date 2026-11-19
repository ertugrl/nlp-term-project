"""Shared paths, label algebra, data loading, decoding and official scoring."""

from __future__ import annotations

import csv
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT / "third_party" / "dico-nli"))

from evaluation_functions.io import read_reference_file  # noqa: E402
from evaluation_functions.labels import ALL_LABELS, reverse_label  # noqa: E402
from evaluation_functions.metrics import compute_evaluation  # noqa: E402
from evaluation_functions.validation import validate_reference_records  # noqa: E402

LABELS = list(ALL_LABELS)  # EQUIVALENCE, FORWARD_ENTAILMENT, BACKWARD_ENTAILMENT, NEGATIVE_OTHER
L2I = {label: i for i, label in enumerate(LABELS)}
NEG = L2I["NEGATIVE_OTHER"]
# Index permutation implementing Rev(.) on the label simplex; NEGATIVE_OTHER is mapped to itself
# (it has no official reverse, but this lets us compare distributions of the two directions).
REV_PERM = np.array([L2I[reverse_label(l)] if i != NEG else NEG for i, l in enumerate(LABELS)])

TRACKS = ["track1", "track2", "track3", "track4"]
TRACK_NAMES = {"track1": "EN", "track2": "ES", "track3": "EU", "track4": "Mixed"}
VAL_FRACTION = 0.10


def participant_path(split: str, track: str) -> Path:
    return DATA / split / f"dico_nli_{split}_{track}_participant_labeled.csv"


def reference_path(split: str, track: str) -> Path:
    return DATA / split / f"dico_nli_{split}_{track}_reference.csv"


def is_val_pair(pair_id: str) -> bool:
    """Deterministic pair-level hold-out; identical source pairs are held out in every track."""
    h = int(hashlib.md5(pair_id.encode("utf-8")).hexdigest(), 16)
    return (h % 1000) < VAL_FRACTION * 1000


def load_rows(split: str, track: str, part: str = "all") -> list[dict]:
    """part: 'all' | 'fit' (train minus held-out pairs) | 'val' (held-out pairs)."""
    with open(participant_path(split, track), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if part == "fit":
        rows = [r for r in rows if not is_val_pair(r["pair_id"])]
    elif part == "val":
        rows = [r for r in rows if is_val_pair(r["pair_id"])]
    for r in rows:
        r["track"] = track
    return rows


def group_key(row: dict) -> tuple:
    """The reverse of (a, b) is (b, a): same pair_id, same {(lang, text)} set with slots swapped.

    Only participant-visible columns are used, so this works on unlabeled test files too
    (in the mixed track the reverse of an en-es item is an es-en item).
    """
    slots = frozenset({(row["text1_lang"], row["text1"]), (row["text2_lang"], row["text2"])})
    return (row["pair_id"], slots)


def groups_of(rows: list[dict]) -> list[list[int]]:
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        groups[group_key(r)].append(i)
    return list(groups.values())


def decode(rows: list[dict], logp: np.ndarray, mode: str) -> dict[str, str]:
    """Turn per-instance log-probabilities into labels.

    indep  : argmax per instance.
    joint  : for each two-direction group, argmax_y logp(y|a,b) + logp(Rev(y)|b,a); both directions
             receive y* and Rev(y*). NEGATIVE_OTHER stays a joint option (both directions NEG).
    struct : artifact-exploiting variant: one-direction groups -> NEGATIVE_OTHER, two-direction
             groups -> joint decoding restricted to the reversible labels.
    """
    preds: dict[str, str] = {}
    for idxs in groups_of(rows):
        if mode == "indep" or len(idxs) != 2:
            for i in idxs:
                k = NEG if (mode == "struct" and len(idxs) == 1) else int(np.argmax(logp[i]))
                preds[rows[i]["instance_id"]] = LABELS[k]
            continue
        i, j = idxs
        scores = logp[i] + logp[j][REV_PERM]
        if mode == "struct":
            scores = scores.copy()
            scores[NEG] = -np.inf
        k = int(np.argmax(scores))
        preds[rows[i]["instance_id"]] = LABELS[k]
        preds[rows[j]["instance_id"]] = LABELS[REV_PERM[k]]
    return preds


def score(split: str, track: str, preds: dict[str, str]) -> dict:
    """Official DiCo-NLI metrics, restricted to the instances present in `preds`."""
    gold = [g for g in read_reference_file(reference_path(split, track)) if g.instance_id in preds]
    gold_by_id = validate_reference_records(gold)
    rep = compute_evaluation(gold_by_id, preds).to_dict()
    cls = rep["classification"]
    return {
        "weighted_f1": rep["weighted_f1"],
        "soft_cons": rep["soft_cons"],
        "hard_cons": rep["hard_cons"],
        "macro_f1": cls["macro_f1"],
        "accuracy": cls["accuracy"],
        "n": cls["total_instances"],
        "pairs": rep["consistency"]["reversible_pairs"],
        "confusion": cls["confusion_matrix"],
    }


def write_logp(path: Path, rows: list[dict], logp: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["instance_id"] + [f"logp_{l}" for l in LABELS])
        for r, lp in zip(rows, logp):
            w.writerow([r["instance_id"]] + [f"{x:.6f}" for x in lp])


def read_logp(path: Path, rows: list[dict]) -> np.ndarray:
    with open(path, newline="", encoding="utf-8") as f:
        table = {r["instance_id"]: [float(r[f"logp_{l}"]) for l in LABELS] for r in csv.DictReader(f)}
    return np.array([table[r["instance_id"]] for r in rows])


def write_predictions(path: Path, preds: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["instance_id", "label"])
        for iid, lab in preds.items():
            w.writerow([iid, lab])
