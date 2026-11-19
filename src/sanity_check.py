"""Checks the data assumptions the decoders rely on, and prints dataset statistics."""

from __future__ import annotations

import csv
from collections import Counter

from dico.common import TRACKS, groups_of, load_rows, reference_path

for split in ["train", "dev"]:
    for track in TRACKS:
        rows = load_rows(split, track)
        with open(reference_path(split, track), newline="", encoding="utf-8") as f:
            rev = {r["instance_id"]: r["reverse_pair_id"] for r in csv.DictReader(f)}
        sizes = Counter()
        for g in groups_of(rows):
            sizes[len(g)] += 1
            ids = [rows[i]["instance_id"] for i in g]
            if len(g) == 2:
                assert rev[ids[0]] == ids[1] and rev[ids[1]] == ids[0], ids
            else:
                assert len(g) == 1 and not rev[ids[0]], ids
        neg_single = all(
            (rows[g[0]]["label"] == "NEGATIVE_OTHER") == (len(g) == 1) for g in groups_of(rows)
        )
        n_fit = len(load_rows(split, track, "fit"))
        n_val = len(load_rows(split, track, "val"))
        langs = Counter(f'{r["text1_lang"]}-{r["text2_lang"]}' for r in rows)
        print(
            f"{split:5s} {track}: n={len(rows):5d} groups={dict(sizes)} fit/val={n_fit}/{n_val} "
            f"NEG<=>singleton: {neg_single} langs={dict(langs)}"
        )
print("all group/reverse_pair_id checks passed")
