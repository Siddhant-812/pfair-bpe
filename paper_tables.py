#!/usr/bin/env python3
"""Turn result folders into the paper's tables (LaTeX) and chart data (CSV for pgfplots).

    python paper_tables.py front     --dir results_errorbars            # Table: Gini / compression, mean +- sd, with baselines at matched compression
    python paper_tables.py perlang   --dir results_errorbars            # Table: tokens per sentence per language (mean over samples)
    python paper_tables.py single    --dir results_sweep                # same two tables from a single run (no error bars)
    python paper_tables.py sens      --dirs "1x=results_equal,4x=results,8x=results_eng8,64k=results_64k"   # Table: Gini by setting
    python paper_tables.py morph     --dir results_morph                # Table: lambda x p -> Gini, compression, boundary F1 per language
    python paper_tables.py alloc     --dir results_alloc                # Table: allocation methods
    python paper_tables.py lm        --dirs "Classical=results_lm/p1,p=8=results_lm/p8,..."   # Table: bits per byte per language
    python paper_tables.py regress   --dir results_sweep                # Table: worst regression vs classical per method (max_l t_l / t_l^classical)
    python paper_tables.py underserved --dir results_sweep              # Table: do under-served languages (above the classical mean) ever get worse?

All commands print LaTeX to stdout and write the same table as CSV next to the
results (<dir>/paper_<name>.csv). Numbers are taken from the files written by
run_all.py / error_bars.py / allocate.py / train_lm.py, nothing is retyped.
"""
import argparse
import csv
import glob
import os
import statistics as st
import sys
from collections import defaultdict


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def esc(s):
    return s.replace("%", "\\%").replace("_", "\\_")


def latex_table(header, rows, caption, label, align=None):
    align = align or "l" + "c" * (len(header) - 1)
    out = ["\\begin{table}[t]", "\\centering", "\\small", f"\\begin{{tabular}}{{{align}}}", "\\toprule",
           " & ".join(header) + "\\\\", "\\midrule"]
    out += [" & ".join(str(x) for x in r) + "\\\\" for r in rows]
    out += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table}"]
    return "\n".join(out)


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"% written {path}", file=sys.stderr)


def samples_of(d):
    """[(summary rows, per-language rows)] for every sample folder, or the folder itself"""
    subs = sorted(glob.glob(os.path.join(d, "sample*")))
    folders = subs if subs else [d]
    return [(read_csv(os.path.join(f, "results_summary.csv")), read_csv(os.path.join(f, "results_per_language.csv"))) for f in folders]


def method_order(methods):
    def key(m):
        if m == "Classical":
            return (0, 0)
        if m.startswith("p=") and "lambda" not in m:
            return (0, float(m[2:]))
        if m == "Parity-aware":
            return (0, 1e9)
        return (1, m)
    return sorted(methods, key=key)


def pm(vals, nd=4):
    if len(vals) == 1:
        return f"{vals[0]:.{nd}f}"
    return f"${st.mean(vals):.{nd}f}\\pm{st.stdev(vals):.{nd}f}$"


def cmd_front(a):
    smp = samples_of(a.dir)
    methods = method_order({r["method"] for s, _ in smp for r in s})
    rows, csv_rows = [], []
    for m in methods:
        g = [float(r["gini_per_line"]) for s, _ in smp for r in s if r["method"] == m]
        c = [float(r["train_bytes_per_token"]) for s, _ in smp for r in s if r["method"] == m]
        d = [float(r["bytes_per_token"]) for s, _ in smp for r in s if r["method"] == m]
        rows.append([esc(m), pm(g), pm(c, 3), pm(d, 3)])
        csv_rows.append([m, st.mean(g), st.stdev(g) if len(g) > 1 else 0, st.mean(c), st.stdev(c) if len(c) > 1 else 0, st.mean(d)])
    print(latex_table(["Tokenizer", "Gini $\\downarrow$", "Compr.\\ train $\\uparrow$", "Compr.\\ test $\\uparrow$"], rows,
                      f"Parity and compression (mean $\\pm$ sd over {len(smp)} samples).", "tab:front"))
    write_csv(os.path.join(a.dir, "paper_front.csv"), ["method", "gini", "gini_sd", "cr_train", "cr_train_sd", "cr_test"], csv_rows)


def cmd_perlang(a):
    smp = samples_of(a.dir)
    langs = []
    for r in smp[0][1]:
        if r["lang"] not in langs:
            langs.append(r["lang"])
    methods = method_order({r["method"] for _, d in smp for r in d})
    rows, csv_rows = [], []
    for m in methods:
        vals = []
        for l in langs:
            v = [float(r["tokens_per_line"]) for _, d in smp for r in d if r["method"] == m and r["lang"] == l]
            vals.append(st.mean(v))
        rows.append([esc(m)] + [f"{v:.2f}" for v in vals] + [f"{max(vals) / min(vals):.2f}"])
        csv_rows.append([m] + vals + [max(vals) / min(vals)])
    print(latex_table(["Tokenizer"] + langs + ["Max/min"], rows, "Tokens per sentence on FLORES-200 devtest.", "tab:perlang"))
    write_csv(os.path.join(a.dir, "paper_perlang.csv"), ["method"] + langs + ["max_over_min"], csv_rows)


def cmd_regress(a):
    """worst regression against classical: max over languages of t_l(method) / t_l(classical)"""
    smp = samples_of(a.dir)
    methods = method_order({r["method"] for _, d in smp for r in d})
    rows = []
    for m in methods:
        ratios = []
        for _, d in smp:
            base = {r["lang"]: float(r["tokens_per_line"]) for r in d if r["method"] == "Classical"}
            cur = {r["lang"]: float(r["tokens_per_line"]) for r in d if r["method"] == m}
            ratios.append(max(cur[l] / base[l] for l in cur if l != "eng"))
        rows.append([esc(m), pm(ratios, 3)])
    print(latex_table(["Tokenizer", "max$_{\\ell\\neq\\mathrm{eng}}\\; t_\\ell / t_\\ell^{\\mathrm{classical}}$"], rows,
                      "Worst regression of a non-English language against classical BPE (1.000 = no language worse off).", "tab:regress"))


def cmd_underserved(a):
    """For languages that cost MORE than the classical mean (the under-served ones): does the
    method lower every one of them?  Reports max_l t_l/t_l^classical over those languages and
    how many of them got more expensive. Languages below the classical mean are expected to
    pay for parity; what distinguishes the rules is whether mid-cost languages pay too."""
    smp = samples_of(a.dir)
    methods = method_order({r["method"] for _, d in smp for r in d})
    rows, csv_rows = [], []
    for m in methods:
        worst, n_worse, n_under = [], [], 0
        for _, d in smp:
            base = {r["lang"]: float(r["tokens_per_line"]) for r in d if r["method"] == "Classical"}
            cur = {r["lang"]: float(r["tokens_per_line"]) for r in d if r["method"] == m}
            mean = st.mean(base.values())
            under = [l for l in base if base[l] > mean]
            n_under = len(under)
            worst.append(max(cur[l] / base[l] for l in under))
            n_worse.append(sum(cur[l] > base[l] * 1.001 for l in under))
        rows.append([esc(m), pm(worst, 3), f"{st.mean(n_worse):.1f} / {n_under}"])
        csv_rows.append([m, st.mean(worst), st.mean(n_worse), n_under])
    print(latex_table(["Tokenizer", "max ratio vs.\ classical", "made worse / under-served"], rows,
                      "Languages above the classical mean cost: largest cost ratio against classical BPE and how many of them a method makes more expensive.", "tab:underserved"))
    write_csv(os.path.join(a.dir, "paper_underserved.csv"), ["method", "max_ratio", "n_made_worse", "n_underserved"], csv_rows)


def cmd_sens(a):
    rows = []
    for item in a.dirs.split(","):
        label, d = item.split("=")
        smp = samples_of(d)
        methods = method_order({r["method"] for s, _ in smp for r in s})
        g = {m: st.mean([float(r["gini_per_line"]) for s, _ in smp for r in s if r["method"] == m]) for m in methods}
        c = {m: st.mean([float(r["train_bytes_per_token"]) for s, _ in smp for r in s if r["method"] == m]) for m in methods}
        cost = 100 * (1 - c.get("Parity-aware", c["Classical"]) / c["Classical"])
        rows.append([esc(label)] + [f"{g[m]:.4f}" for m in methods] + [f"{cost:.1f}\\%"])
    print(latex_table(["Setting"] + [esc(m) for m in methods] + ["Cost of parity"], rows, "Gini by setting.", "tab:sens"))


def cmd_morph(a):
    summ = read_csv(os.path.join(a.dir, "results_summary.csv"))
    morph = read_csv(os.path.join(a.dir, "results_morph.csv"))
    langs = []
    for r in morph:
        if r["lang"] not in langs:
            langs.append(r["lang"])
    rows = []
    for r in summ:
        m = r["method"]
        f1 = {x["lang"]: float(x["f1"]) for x in morph if x["method"] == m}
        p, lam = (m.split(",lambda=") + ["0"])[:2]
        rows.append([esc(p), lam, r["gini_per_line"], r["train_bytes_per_token"]] + [f"{f1.get(l, float('nan')):.3f}" for l in langs])
    print(latex_table(["$p$", "$\\lambda$", "Gini $\\downarrow$", "Compr. $\\uparrow$"] + [f"F1 {l}" for l in langs], rows,
                      "Morphology penalty: boundary F1 on held-out gold segmentations.", "tab:morph"))


def cmd_alloc(a):
    summ = read_csv(os.path.join(a.dir, "results_summary.csv"))
    rows = [[esc(r["method"]), r["vocab"], r["gini_per_line"], r["train_bytes_per_token"], r["bytes_per_token"]] for r in summ]
    print(latex_table(["Allocation", "Vocab.", "Gini $\\downarrow$", "Compr.\\ train $\\uparrow$", "Compr.\\ test $\\uparrow$"], rows,
                      "Vocabulary allocation across scripts.", "tab:alloc"))


def cmd_lm(a):
    data = {}
    for item in a.dirs.split(","):
        label, d = item.rsplit("=", 1)
        data[label] = read_csv(os.path.join(d, "lm_results.csv"))
    langs = [r["lang"] for r in next(iter(data.values()))]
    rows = []
    for label, rs in data.items():
        b = {r["lang"]: float(r["bits_per_byte"]) for r in rs}
        rows.append([esc(label)] + [f"{b[l]:.4f}" for l in langs] + [f"{st.mean(b.values()):.4f}", f"{max(b.values()) / min(b.values()):.3f}"])
    print(latex_table(["Tokenizer"] + langs + ["Mean", "Max/min"], rows, "Bits per byte of a small LM on FLORES-200 devtest.", "tab:lm"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["front", "perlang", "single", "sens", "morph", "alloc", "lm", "regress", "underserved"])
    ap.add_argument("--dir", default="")
    ap.add_argument("--dirs", default="")
    a = ap.parse_args()
    if a.what in ("front", "perlang", "regress", "underserved", "morph", "alloc") and not a.dir:
        sys.exit("--dir is required")
    if a.what == "single":
        cmd_front(a)
        cmd_perlang(a)
    else:
        globals()["cmd_" + a.what](a)


if __name__ == "__main__":
    main()
