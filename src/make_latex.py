"""Generate the LaTeX tables of the report from results/ (run after evaluate.py, make_tables.py, analysis.py).

    python src/make_latex.py            -> report/tables/     (English report)
    python src/make_latex.py --lang tr  -> report/tables_tr/  (Turkish report)

Rows whose runs are missing are skipped, so this can be re-run as experiments finish.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from dico.common import RESULTS, ROOT, TRACKS

METRICS = ["weighted_f1", "soft_cons", "hard_cons"]
SUMMARY = json.loads((RESULTS / "summary.json").read_text())["dev"]
BASE = json.loads((RESULTS / "baselines" / "scores_dev.json").read_text())
ANALYSIS = json.loads((RESULTS / "analysis.json").read_text()) if (RESULTS / "analysis.json").exists() else {}

STRINGS = {
    "en": {
        "random": "Random",
        "const": "Constant \\EQ",
        "joint_row": "\\quad + joint decoding",
        "ens": "mDeBERTa-NLI 3-seed ens.",
        "mixed": "Mixed",
        "avg": "Average",
        "system": "System",
        "dec": "Dec.",
        "indep": "indep",
        "joint": "joint",
        "ok": "OK",
        "inc": "Inc.",
        "wrong": "Wrong",
        "train_data": "Training data",
        "in_track": "In-track",
        "en_only": "EN only",
        "pooled": "Pooled",
        "cap_main": "Dev results (\\%). Learned systems: mean over 3 seeds (std.\\ in Appendix~\\ref{app:std}); "
        "the \\emph{3-seed ens.} rows are a single ensemble (averaged probabilities), not a mean. "
        "\\emph{joint}: joint bidirectional decoding (Eq.~\\ref{eq:joint}). Bold: best learned (non-diagnostic) value. "
        "The last row exploits observation O2 and is shown as a diagnostic only.",
        "cap_err": "Gold reversible pairs (\\%, dev, mean over tracks and seeds): both correct (OK), a direction predicted \\NO, "
        "inconsistent (Inc.), consistent but wrong. F1\\textsubscript{\\NO}: instance-level F1.",
        "cap_transfer": "Dev F1\\,/\\,HardCons (\\%) of mDeBERTa, joint decoding, 3 seeds. EN only is zero-shot on ES, EU, Mixed; "
        "on EN it reruns the in-track configuration (gap: GPU non-determinism). Full: Table~\\ref{tab:transfer_full}.",
        "cap_transfer_full": "Cross-lingual transfer of mDeBERTa on dev (\\%), mean $\\pm$ std.\\ over three seeds, "
        "with independent and joint decoding. Training data as in Table~\\ref{tab:transfer}.",
        "cap_std": "Dev results (\\%), mean $\\pm$ standard deviation over three seeds.",
    },
    "tr": {
        "random": "Rastgele",
        "const": "Sabit \\EQ",
        "joint_row": "\\quad + ortak çözümleme",
        "ens": "mDeBERTa-NLI, 3 tohum topluluğu",
        "mixed": "Karma",
        "avg": "Ortalama",
        "system": "Sistem",
        "dec": "Çöz.",
        "indep": "ayrı",
        "joint": "ortak",
        "ok": "Doğru",
        "inc": "Tuts.",
        "wrong": "Yanlış",
        "train_data": "Eğitim verisi",
        "in_track": "Kendi verisi",
        "en_only": "Yalnız EN",
        "pooled": "Birleşik",
        "cap_main": "Dev sonuçları (\\%). Öğrenen sistemler: 3 tohumun ortalaması (standart sapmalar Ek~\\ref{app:std}'de); "
        "\\emph{3 tohum topluluğu} satırları ortalama değil, tek bir topluluğun (olasılık ortalaması) sonucudur. "
        "\\emph{ortak}: ortak çift yönlü çözümleme (Denk.~\\ref{eq:joint}). Kalın: en iyi öğrenen (tanı dışı) değer. "
        "Son satır O2'yi kullanır; yalnızca tanı amaçlıdır.",
        "cap_err": "Altın tersinir çiftler (\\%; dev, track ve tohum ortalaması): iki yön de doğru (Doğru), bir yöne \\NO, "
        "tutarsız (Tuts.), tutarlı ama yanlış. F1\\textsubscript{\\NO}: örnek düzeyinde F1.",
        "cap_transfer": "mDeBERTa, ortak çözümleme, 3 tohum: dev F1\\,/\\,HardCons (\\%). Yalnız EN, ES, EU ve Karma'da sıfır-örnektir; "
        "EN'de kendi verisi yapılandırmasının tekrar koşusudur (fark: GPU deterministik değil). Tümü: Tablo~\\ref{tab:transfer_full}.",
        "cap_transfer_full": "mDeBERTa'nın diller arası aktarımı (dev, \\%), üç tohum üzerinden ortalama $\\pm$ std., "
        "ayrı ve ortak çözümlemeyle. Eğitim verileri Tablo~\\ref{tab:transfer}'teki gibidir.",
        "cap_std": "Dev sonuçları (\\%), üç tohum üzerinden ortalama $\\pm$ standart sapma.",
    },
}
S = STRINGS["en"]


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
    for k in range(len(METRICS)):
        col = [vals[i * 3 + k] for i in range(len(TRACKS))]
        avg.append(float(np.mean(col)) if all(v is not None for v in col) else None)
    return vals + avg


def track_names(with_avg: bool) -> list[str]:
    return ["EN", "ES", "EU", S["mixed"]] + ([S["avg"]] if with_avg else [])


def main_table() -> str:
    spec = [
        (S["random"], lambda t, m: BASE[t]["random"][m]),
        (S["const"], lambda t, m: BASE[t]["constant_eq"][m]),
    ]
    models = [("xlmr", "XLM-R"), ("mdeberta", "mDeBERTa"), ("mdeberta-xnli", "mDeBERTa-NLI"), ("mdeberta-cons", "mDeBERTa + $\\mathcal{L}_\\text{cons}$")]
    for key, name in models:
        if seed_mean(key, "track1", "indep", "weighted_f1") is None:
            continue
        spec.append((name, lambda t, m, k=key: seed_mean(k, t, "indep", m)))
        spec.append((S["joint_row"], lambda t, m, k=key: seed_mean(k, t, "joint", m)))
    if ens("mdeberta-xnli", "track1", "indep", "weighted_f1") is not None:
        spec.append((S["ens"], lambda t, m: ens("mdeberta-xnli", t, "indep", m)))
        spec.append((S["joint_row"], lambda t, m: ens("mdeberta-xnli", t, "joint", m)))
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
        " & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{n}}}" for n in track_names(with_avg=True)) + " \\\\",
        " ".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(TRACKS) + 1)),
        S["system"] + " & " + " & ".join(["F1", "Soft", "Hard"] * (len(TRACKS) + 1)) + " \\\\",
        "\\midrule",
    ]
    for i, (name, txt) in enumerate(rows):
        if i in (2, diag_idx) or (i > 2 and name.startswith(("XLM", "mDeBERTa"))):
            lines.append("\\midrule")
        if i == diag_idx:
            txt = [f"\\textit{{{x}}}" for x in txt]
        lines.append(f"{name} & " + " & ".join(txt) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}}",
        f"\\caption{{{S['cap_main']}}}",
        "\\label{tab:main}",
        "\\end{table*}",
    ]
    return "\n".join(lines)


def error_table() -> str:
    cats = [("correct", S["ok"]), ("neg", "\\NO"), ("inconsistent", S["inc"]), ("consistent_wrong", S["wrong"])]
    lines = [
        "\\begin{table}[t]",
        "\\centering\\small",
        "\\setlength{\\tabcolsep}{2.5pt}",
        "\\begin{tabular}{@{}llccccc@{}}",
        "\\toprule",
        f"Model & {S['dec']} & " + " & ".join(c[1] for c in cats) + " & F1\\textsubscript{\\NO} \\\\",
        "\\midrule",
    ]
    for key, name in [("xlmr", "XLM-R"), ("mdeberta", "mDeBERTa")]:
        for mode in ["indep", "joint"]:
            vals = []
            for c in [c[0] for c in cats] + ["neg_f1"]:
                v = [ANALYSIS.get(f"{key}_{t}", {}).get(mode, {}).get(c) for t in TRACKS]
                vals.append(float(np.mean(v)) if all(x is not None for x in v) else None)
            lines.append(f"{name if mode == 'indep' else ''} & {S[mode]} & " + " & ".join(cells(vals)) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{{S['cap_err']}}}",
        "\\label{tab:errors}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def transfer_table() -> str | None:
    """E5: in-track training vs EN-only (zero-shot) vs pooled multilingual training (joint decoding)."""
    systems = [("mdeberta", S["in_track"]), ("mdeberta-en2all", S["en_only"]), ("mdeberta-pooled", S["pooled"])]
    if all(seed_mean(k, "track2", "joint", "weighted_f1") is None for k, _ in systems[1:]):
        return None
    lines = [
        "\\begin{table}[t]",
        "\\centering\\small",
        "\\setlength{\\tabcolsep}{2.5pt}",
        "\\begin{tabular}{@{}lcccc@{}}",
        "\\toprule",
        f"{S['train_data']} & " + " & ".join(track_names(with_avg=False)) + " \\\\",
        "\\midrule",
    ]
    for key, name in systems:
        f1 = [seed_mean(key, t, "joint", "weighted_f1") for t in TRACKS]
        hc = [seed_mean(key, t, "joint", "hard_cons") for t in TRACKS]
        if all(v is None for v in hc):
            continue
        lines.append(f"{name} & " + " & ".join(f"{a}\\,/\\,{b}" for a, b in zip(cells(f1), cells(hc))) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{{S['cap_transfer']}}}",
        "\\label{tab:transfer}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def std_table(models: list[tuple[str, str]] | None = None, caption: str = "cap_std", label: str = "tab:std",
              head: str = "Model") -> str:
    if models is None:
        models = [("xlmr", "XLM-R"), ("mdeberta", "mDeBERTa"), ("mdeberta-xnli", "mDeBERTa-NLI"),
                  ("mdeberta-cons", "mDeBERTa+$\\mathcal{L}_\\text{cons}$")]
    lines = [
        "\\begin{table*}[t]",
        "\\centering\\small",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{ll" + "ccc" * len(TRACKS) + "}",
        "\\toprule",
        " & & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{n}}}" for n in track_names(with_avg=False)) + " \\\\",
        f"{head} & {S['dec']} & " + " & ".join(["F1", "Soft", "Hard"] * len(TRACKS)) + " \\\\",
        "\\midrule",
    ]
    for key, name in models:
        if seed_mean(key, "track1", "indep", "weighted_f1") is None:
            continue
        for mode in ["indep", "joint"]:
            txt = []
            for t in TRACKS:
                for m in METRICS:
                    r = seed_mean(key, t, mode, m, std=True)
                    txt.append(f"{100 * r[0]:.1f}$\\pm${100 * r[1]:.1f}" if r else "--")
            lines.append(f"{name if mode == 'indep' else ''} & {S[mode]} & " + " & ".join(txt) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}}",
        f"\\caption{{{S[caption]}}}",
        f"\\label{{{label}}}",
        "\\end{table*}",
    ]
    return "\n".join(lines)


def main() -> None:
    global S
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=sorted(STRINGS), default="en")
    lang = ap.parse_args().lang
    S = STRINGS[lang]
    out = ROOT / "report" / ("tables" if lang == "en" else f"tables_{lang}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "main.tex").write_text(main_table() + "\n")
    (out / "errors.tex").write_text(error_table() + "\n")
    (out / "std.tex").write_text(std_table() + "\n")
    (out / "transfer_full.tex").write_text(std_table(
        [("mdeberta", S["in_track"]), ("mdeberta-en2all", S["en_only"]), ("mdeberta-pooled", S["pooled"])],
        caption="cap_transfer_full", label="tab:transfer_full", head=S["train_data"]) + "\n")
    tr = transfer_table()
    (out / "transfer.tex").write_text((tr or "% transfer experiments not finished yet") + "\n")
    print("wrote", out.relative_to(ROOT), ":", ", ".join(sorted(p.name for p in out.glob("*.tex"))))


if __name__ == "__main__":
    main()
