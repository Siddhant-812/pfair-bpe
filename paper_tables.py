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
    python paper_tables.py chart     --dir results_errorbars            # pgfplots CSVs: p-fair curve / hybrid points / other baselines

All commands print LaTeX to stdout and write the same table as CSV next to the
results (<dir>/paper_<name>.csv). Numbers are taken from the files written by
run_all.py / error_bars.py / allocate.py / train_lm.py, nothing is retyped.
"""
import argparse
import csv
import glob
import json
import os
import statistics as st
import sys
from collections import defaultdict

from compare_hybrid import is_pfair


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


def lead_share(folder, method, lang):
    """share of merges whose score the language `lang` contributes most to (train_log.tsv, column lead_lang)"""
    with open(os.path.join(folder, "runs.json"), encoding="utf-8") as f:
        model = json.load(f)[method]
    n = led = 0
    with open(os.path.join(folder, model, "train_log.tsv"), encoding="utf-8") as f:
        head = f.readline().rstrip("\n").split("\t")
        k = head.index("lead_lang")
        for line in f:
            n += 1
            led += line.split("\t")[k] == lang
    return led / n


def cmd_front(a):
    smp = samples_of(a.dir)
    folders = sorted(glob.glob(os.path.join(a.dir, "sample*"))) or [a.dir]
    methods = method_order({r["method"] for s, _ in smp for r in s})
    rows, csv_rows = [], []
    for m in methods:
        g = [float(r["gini_per_line"]) for s, _ in smp for r in s if r["method"] == m]
        c = [float(r["train_bytes_per_token"]) for s, _ in smp for r in s if r["method"] == m]
        d = [float(r["bytes_per_token"]) for s, _ in smp for r in s if r["method"] == m]
        led = [lead_share(f, m, a.lead) for f in folders] if a.lead else []
        rows.append([esc(m), pm(g), pm(c, 3), pm(d, 3)] + ([f"{100 * st.mean(led):.0f}\\%"] if led else []))
        csv_rows.append([m, st.mean(g), st.stdev(g) if len(g) > 1 else 0, st.mean(c), st.stdev(c) if len(c) > 1 else 0,
                         st.mean(d), st.stdev(d) if len(d) > 1 else 0] + ([st.mean(led)] if led else []))
    print(latex_table(["Tokenizer", "Gini $\\downarrow$", "Compr.\\ train $\\uparrow$", "Compr.\\ test $\\uparrow$"]
                      + ([f"Merges led by {a.lead}"] if a.lead else []), rows,
                      f"Parity and compression (mean $\\pm$ sd over {len(smp)} samples).", "tab:front"))
    write_csv(os.path.join(a.dir, "paper_front.csv"),
              ["method", "gini", "gini_sd", "cr_train", "cr_train_sd", "cr_test", "cr_test_sd"] + ([f"lead_{a.lead}"] if a.lead else []), csv_rows)


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
    rows, csv_rows = [], []
    for m in methods:
        ratios = []
        for _, d in smp:
            base = {r["lang"]: float(r["tokens_per_line"]) for r in d if r["method"] == "Classical"}
            cur = {r["lang"]: float(r["tokens_per_line"]) for r in d if r["method"] == m}
            ratios.append(max(cur[l] / base[l] for l in cur if l != "eng"))
        rows.append([esc(m), pm(ratios, 3)])
        csv_rows.append([m, st.mean(ratios), st.stdev(ratios) if len(ratios) > 1 else 0])
    print(latex_table(["Tokenizer", "max$_{\\ell\\neq\\mathrm{eng}}\\; t_\\ell / t_\\ell^{\\mathrm{classical}}$"], rows,
                      "Worst regression of a non-English language against classical BPE (1.000 = no language worse off).", "tab:regress"))
    write_csv(os.path.join(a.dir, "paper_regress.csv"), ["method", "max_ratio", "max_ratio_sd"], csv_rows)


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
    print(latex_table(["Tokenizer", "max ratio vs.\\ classical", "made worse / under-served"], rows,
                      "Languages above the classical mean cost: largest cost ratio against classical BPE and how many of them a method makes more expensive.", "tab:underserved"))
    write_csv(os.path.join(a.dir, "paper_underserved.csv"), ["method", "max_ratio", "n_made_worse", "n_underserved"], csv_rows)


def cmd_sens(a):
    rows, csv_rows = [], []
    for item in a.dirs.split(","):
        label, d = item.split("=")
        smp = samples_of(d)
        methods = method_order({r["method"] for s, _ in smp for r in s})
        if a.methods:
            methods = [m for m in a.methods.split(",") if m in methods]
        g = {m: st.mean([float(r["gini_per_line"]) for s, _ in smp for r in s if r["method"] == m]) for m in methods}
        c = {m: st.mean([float(r["train_bytes_per_token"]) for s, _ in smp for r in s if r["method"] == m]) for m in methods}
        cost = 100 * (1 - c.get("Parity-aware", c["Classical"]) / c["Classical"])
        rows.append([esc(label)] + [f"{g[m]:.4f}" for m in methods] + [f"{cost:.1f}\\%"])
        csv_rows.append([label] + [g[m] for m in methods] + [cost])
    print(latex_table(["Setting"] + [esc(m) for m in methods] + ["Cost of parity"], rows, "Gini by setting.", "tab:sens"))
    if a.csv:
        write_csv(a.csv, ["setting"] + methods + ["cost_of_parity_pct"], csv_rows)


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
    """bits per byte per language; --seeds "label=dir,..." adds second-seed runs: cells are the mean over seeds and the
    Mean column carries half the seed-to-seed range. Also writes paper_lm.csv next to the first results folder."""
    data, seeds = {}, {}
    for item in a.dirs.split(","):
        label, d = item.rsplit("=", 1)
        data[label] = [read_csv(os.path.join(d, "lm_results.csv"))]
    if a.seeds:
        for item in a.seeds.split(","):
            label, d = item.rsplit("=", 1)
            data[label].append(read_csv(os.path.join(d, "lm_results.csv")))
    langs = [r["lang"] for r in data[next(iter(data))][0]]
    rows, csv_rows = [], []
    for label, runs in data.items():
        per = [{r["lang"]: float(r["bits_per_byte"]) for r in rs} for rs in runs]
        b = {l: st.mean(x[l] for x in per) for l in langs}
        means = [st.mean(x.values()) for x in per]
        mean = st.mean(means)
        spread = (max(means) - min(means)) / 2 if len(means) > 1 else None
        mean_cell = f"{mean:.4f}" + (f" $\\pm$ {spread:.4f}" if spread is not None else "")
        rows.append([esc(label)] + [f"{b[l]:.4f}" for l in langs] + [mean_cell, f"{max(b.values()) / min(b.values()):.3f}"])
        csv_rows.append([label, len(runs)] + [f"{b[l]:.4f}" for l in langs] + [f"{mean:.4f}", "" if spread is None else f"{spread:.4f}"])
    out = os.path.join(os.path.dirname(a.dirs.split(",")[0].rsplit("=", 1)[1].rstrip("/")), "paper_lm.csv")
    write_csv(out, ["method", "seeds"] + langs + ["mean_bpb", "seed_halfrange"], csv_rows)
    print(latex_table(["Tokenizer"] + langs + ["Mean", "Max/min"], rows,
                      "Bits per byte of a small language model on FLORES-200 devtest.", "tab:lm"))


def cmd_chart(a):
    """pgfplots data: results_errorbars.csv split into the p-fair curve, the hybrid points and the other baselines"""
    rows = read_csv(os.path.join(a.dir, "results_errorbars.csv"))
    groups = {"pfair": [r for r in rows if is_pfair(r["method"])],
              "hybrid": [r for r in rows if r["method"].startswith("Hybrid")],
              "other": [r for r in rows if not is_pfair(r["method"]) and not r["method"].startswith("Hybrid")]}
    for g, rs in groups.items():
        if rs:
            write_csv(os.path.join(a.dir, f"paper_chart_{g}.csv"), ["method", "cr", "gini", "cr_err", "gini_err"],
                      [[r["method"], r["cr"], r["gini"], r["cr_err"], r["gini_err"]] for r in rs])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["front", "perlang", "single", "sens", "morph", "alloc", "lm", "regress", "underserved", "chart"])
    ap.add_argument("--dir", default="")
    ap.add_argument("--dirs", default="")
    ap.add_argument("--methods", default="", help="sens: only these methods (comma separated), in this order")
    ap.add_argument("--csv", default="", help="sens: also write the table as CSV to this path")
    ap.add_argument("--seeds", default="", help="lm: second-seed folders as label=dir,...")
    ap.add_argument("--lead", default="eng", help="front: add the share of merges led by this language ('' for none)")
    a = ap.parse_args()
    if a.what in ("front", "perlang", "regress", "underserved", "morph", "alloc", "chart") and not a.dir:
        sys.exit("--dir is required")
    if a.what == "single":
        cmd_front(a)
        cmd_perlang(a)
    else:
        globals()["cmd_" + a.what](a)


if __name__ == "__main__":
    main()
