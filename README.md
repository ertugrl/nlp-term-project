# BLG505 Term Project — SemEval-2027 Task 2 (DiCo-NLI)

Directional-consistent fine-grained NLI over EN / ES / EU / mixed-language phrase pairs.
Task page: https://inigolopezgazpio.net/SemEval-2027-Task-2-DiCo-NLI/

**Approach.** Fine-tune a multilingual encoder (XLM-R, mDeBERTa-v3, NLI-pretrained mDeBERTa) as a
4-way directional classifier, then decode both directions of every reversible pair jointly:
`y* = argmax_y log p(y|a,b) + log p(Rev(y)|b,a)`, which makes predictions consistent by
construction. Also: a symmetric-KL consistency loss, and cross-lingual transfer experiments.

## Layout

| Path | Content |
|---|---|
| `data/` | Official public train/dev CSVs (copied from the task repo) |
| `third_party/dico-nli/evaluation_functions/` | Official scorer (GPL-3.0, see `LICENSE`, upstream commit in `UPSTREAM_COMMIT`) |
| `src/dico/common.py` | Labels, reversal algebra, data loading, reversal grouping, decoders, official scoring |
| `src/train.py` | Fine-tuning; writes per-instance log-probs to `results/runs/<run>/` (no weights saved) |
| `src/evaluate.py` | Decodes saved log-probs (`indep` / `joint` / `struct`), scores, seed-ensembles |
| `src/baselines.py` | Majority / random / constant-EQ / structure-only baselines |
| `src/analysis.py` | Pair-level error decomposition, NEGATIVE_OTHER detection, per-language scores |
| `src/make_tables.py` | `results/summary.md` (mean ± std over seeds) |
| `scripts/` | Experiment drivers (`run_main.sh`, `run_extra.sh`) |
| `report/` | ACL-format LaTeX report: English `main.tex` (submitted) and Turkish `main_tr.tex` (reference copy, tables via `make_latex.py --lang tr`); build with `tectonic -X compile <file> --outdir build` |

## Reproduce

```bash
/opt/homebrew/bin/python3.10 -m venv .venv && .venv/bin/pip install -r requirements.txt
PYTHONPATH=src .venv/bin/python src/sanity_check.py      # data/pairing checks
PYTHONPATH=src .venv/bin/python src/baselines.py
zsh scripts/run_main.sh                                   # E1/E2: 3 encoders x 4 tracks x 3 seeds
zsh scripts/run_extra.sh                                  # E2/E4/E5
PYTHONPATH=src .venv/bin/python src/evaluate.py && PYTHONPATH=src .venv/bin/python src/make_tables.py
PYTHONPATH=src .venv/bin/python src/analysis.py
```

Runs on an Apple M3 Pro (MPS, 18 GB): ~3.5 min per monolingual run, ~10 min per mixed-track run.
Word embeddings are frozen by default to fit in memory (`--train_embeddings` to unfreeze).

## Protocol

- Model selection (epoch, LR) uses a 10% held-out slice of **train**, split by source `pair_id`
  (same pairs held out in every track). The public **dev** split is used as the test set.
- `struct` decoding exploits a dataset property (NEGATIVE_OTHER items are exactly the
  single-direction items) and is reported only as a diagnostic, not as our system.
