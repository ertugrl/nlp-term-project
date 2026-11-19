"""Generate the LaTeX tables of the report from results/ (run after evaluate.py, make_tables.py, analysis.py).

Rows whose runs are missing are skipped, so this can be re-run as experiments finish.
"""

from __future__ import annotations

import json

import numpy as np

from dico.common import RESULTS, ROOT, TRACKS

OUT = ROOT / "report" / "tables"
METRICS = ["weighted_f1", "soft_cons", "hard_cons"]
SUMMARY = json.loads((RESULTS / "summary.json").read_text())["dev"]
BASE = json.loads((RESULTS / "baselines" / "scores_dev.json").read_text())
ANALYSIS = json.loads((RESULTS / "analysis.json").read_text()) if (RESULTS / "analysis.json").exists() else {}


def seed_mean(model: str, track: str, mode: str, metric: str, std: bool = False):
    """Per-track models are stored as <model>_<track>; multi-track runs as <model>."""
    for config in (f"{model}_{track}", model):
        try:
            mean, sd = SUMMARY[config][mode][track][metric]
            return (mean, sd) if std else mean
        except KeyError:
            continue
    return None


def ens(model: str, track: str, mode: str, metric: str):
    f = RESULTS / "ensembles" / f"{model}_{track}" / "scores.json"
    if not f.exists():
        return None
    return json.loads(f.read_text())["dev"][track][mode][metric]


def cells(values: list) -> list[str]:
    return [f"{100 * v:.1f}" if v is not None else "--" for v in values]


def bold_best(rows: list[tuple[str, list]], skip: set[int]) -> list[tuple[str, list[str]]]:
    """Bold the column-wise best value among non-diagnostic rows."""
    ncol = len(rows[0][1])
    best = []
    for c in range(ncol):
        vals = [r[1][c] for i, r in enumerate(rows) if i not in skip and r[1][c] is not None]
        best.append(max(vals) if vals else None)
    out = []
    for i, (name, vals) in enumerate(rows):
        txt = []
        for c, v in enumerate(vals):
            s = cells([v])[0]
            if i not in skip and v is not None and best[c] is not None and abs(v - best[c]) < 1e-9:
                s = f"\\textbf{{{s}}}"
            txt.append(s)
        out.append((name, txt))
    return out


def row_values(fn) -> list:
    vals = [fn(t, m) for t in TRACKS for m in METRICS]
    avg = []
    for k, m in enumerate(METRICS):
        col = [vals[i * 3 + k] for i in range(len(TRACKS))]
        avg.append(float(np.mean(col)) if all(v is not None for v in col) else None)
    return vals + avg


def main_table() -> str:
    spec = [
        ("Random", lambda t, m: BASE[t]["random"][m]),
        ("Constant \\EQ", lambda t, m: BASE[t]["constant_eq"][m]),
    ]
    models = [("xlmr", "XLM-R"), ("mdeberta", "mDeBERTa"), ("mdeberta-xnli", "mDeBERTa-NLI"), ("mdeberta-cons", "mDeBERTa + $\\mathcal{L}_\\text{cons}$")]
    for key, name in models:
        if seed_mean(key, "track1", "indep", "weighted_f1") is None:
            continue
        spec.append((name, lambda t, m, k=key: seed_mean(k, t, "indep", m)))
        spec.append(("\\quad + joint decoding", lambda t, m, k=key: seed_mean(k, t, "joint", m)))
    if ens("mdeberta-xnli", "track1", "indep", "weighted_f1") is not None:
        spec.append(("mDeBERTa-NLI 3-seed ens.", lambda t, m: ens("mdeberta-xnli", t, "indep", m)))
        spec.append(("\\quad + joint decoding", lambda t, m: ens("mdeberta-xnli", t, "joint", m)))
    diag_idx = len(spec)
    spec.append(("\\textit{mDeBERTa + struct (O2)}", lambda t, m: seed_mean("mdeberta", t, "struct", m)))

    rows = [(name, row_values(fn)) for name, fn in spec]
    rows = bold_best(rows, skip={0, 1, diag_idx})
    lines = [
        "\\begin{table*}[t]",
        "\\centering\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{l" + "ccc" * (len(TRACKS) + 1) + "}",
        "\\toprule",
        " & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{n}}}" for n in ["EN", "ES", "EU", "Mixed", "Average"]) + " \\\\",
        " ".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(TRACKS) + 1)),
        "System & " + " & ".join(["F1", "Soft", "Hard"] * (len(TRACKS) + 1)) + " \\\\",
        "\\midrule",
    ]
    for i, (name, txt) in enumerate(rows):
        if i in (2, diag_idx) or (i > 2 and name.startswith(("XLM", "mDeBERTa"))):
            lines.append("\\midrule")
        if i == diag_idx:
            txt = [f"\\textit{{{t}}}" for t in txt]
        lines.append(f"{name} & " + " & ".join(txt) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}}",
        "\\caption{Dev results (\\%). Learned systems: mean over 3 seeds (std.\\ in Appendix~\\ref{app:std}). "
        "\\emph{joint}: joint bidirectional decoding (Eq.~\\ref{eq:joint}). Bold: best non-diagnostic value. "
        "The last row exploits observation O2 and is shown as a diagnostic only.}",
        "\\label{tab:main}",
        "\\end{table*}",
    ]
    return "\n".join(lines)


def error_table() -> str:
    cats = [("correct", "OK"), ("neg", "\\NO"), ("inconsistent", "Inc."), ("consistent_wrong", "Wrong")]
    lines = [
        "\\begin{table}[t]",
        "\\centering\\small",
        "\\setlength{\\tabcolsep}{2.5pt}",
        "\\begin{tabular}{@{}llccccc@{}}",
        "\\toprule",
        "Model & Dec. & " + " & ".join(c[1] for c in cats) + " & F1\\textsubscript{\\NO} \\\\",
        "\\midrule",
    ]
    for key, name in [("xlmr", "XLM-R"), ("mdeberta", "mDeBERTa")]:
        for mode in ["indep", "joint"]:
            vals = []
            for c in [c[0] for c in cats] + ["neg_f1"]:
                v = [ANALYSIS.get(f"{key}_{t}", {}).get(mode, {}).get(c) for t in TRACKS]
                vals.append(float(np.mean(v)) if all(x is not None for x in v) else None)
            lines.append(f"{name if mode == 'indep' else ''} & {mode} & " + " & ".join(cells(vals)) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Outcome of gold reversible pairs (\\% of pairs, dev, mean over tracks and seeds): both correct (OK), "
        "a direction predicted \\NO, inconsistent (Inc.), or consistent but wrong. F1\\textsubscript{\\NO}: instance-level F1 of \\NO.}",
        "\\label{tab:errors}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def transfer_table() -> str | None:
    """E5: in-track training vs EN-only (zero-shot) vs pooled multilingual training (joint decoding)."""
    systems = [("mdeberta", "In-track"), ("mdeberta-en2all", "EN only"), ("mdeberta-pooled", "Pooled")]
    if all(seed_mean(k, "track2", "joint", "weighted_f1") is None for k, _ in systems[1:]):
        return None
    lines = [
        "\\begin{table}[t]",
        "\\centering\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{@{}lcccc@{}}",
        "\\toprule",
        "Training data & EN & ES & EU & Mixed \\\\",
        "\\midrule",
    ]
    for key, name in systems:
        vals = [seed_mean(key, t, "joint", "hard_cons") for t in TRACKS]
        if all(v is None for v in vals):
            continue
        lines.append(f"{name} & " + " & ".join(cells(vals)) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Dev HardCons (\\%) of mDeBERTa with joint decoding (3 seeds) by training data: the track's own (in-track), "
        "Track~1 only (zero-shot on the others), or all four tracks (pooled).}",
        "\\label{tab:transfer}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def std_table() -> str:
    lines = [
        "\\begin{table*}[t]",
        "\\centering\\small",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{ll" + "ccc" * len(TRACKS) + "}",
        "\\toprule",
        " & & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{n}}}" for n in ["EN", "ES", "EU", "Mixed"]) + " \\\\",
        "Model & Dec. & " + " & ".join(["F1", "Soft", "Hard"] * len(TRACKS)) + " \\\\",
        "\\midrule",
    ]
    for key, name in [("xlmr", "XLM-R"), ("mdeberta", "mDeBERTa"), ("mdeberta-xnli", "mDeBERTa-NLI"), ("mdeberta-cons", "mDeBERTa+$\\mathcal{L}_\\text{cons}$")]:
        if seed_mean(key, "track1", "indep", "weighted_f1") is None:
            continue
        for mode in ["indep", "joint"]:
            txt = []
            for t in TRACKS:
                for m in METRICS:
                    r = seed_mean(key, t, mode, m, std=True)
                    txt.append(f"{100 * r[0]:.1f}$\\pm${100 * r[1]:.1f}" if r else "--")
            lines.append(f"{name if mode == 'indep' else ''} & {mode} & " + " & ".join(txt) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}}",
        "\\caption{Dev results (\\%), mean $\\pm$ standard deviation over three seeds.}",
        "\\label{tab:std}",
        "\\end{table*}",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "main.tex").write_text(main_table() + "\n")
    (OUT / "errors.tex").write_text(error_table() + "\n")
    (OUT / "std.tex").write_text(std_table() + "\n")
    tr = transfer_table()
    (OUT / "transfer.tex").write_text((tr or "% transfer experiments not finished yet") + "\n")
    print("wrote", ", ".join(p.name for p in OUT.glob("*.tex")))


if __name__ == "__main__":
    main()
