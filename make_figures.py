#!/usr/bin/env python3
"""Figures for the README from the result CSVs (no numbers are typed here).

    python make_figures.py --main results_main --l14 results_l14_eb --out figures

Reads paper_chart_{pfair,hybrid,other}.csv (written by `paper_tables.py chart`), paper_perlang.csv
and paper_sens.csv from the result folders and writes PNGs. Falls back to reference_results/
when a folder is missing.
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"   # validated categorical order
LANG = {"hin": "Hindi", "mar": "Marathi", "guj": "Gujarati", "eng": "English"}


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def style(ax, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)


def front(d, out, title, label_every):
    pf, hy, ot = (rows(os.path.join(d, f"paper_chart_{g}.csv")) for g in ("pfair", "hybrid", "other"))
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=160, facecolor=SURFACE)
    x = [float(r["cr"]) for r in pf]; y = [float(r["gini"]) for r in pf]
    ax.errorbar(x, y, xerr=[float(r["cr_err"]) for r in pf], yerr=[float(r["gini_err"]) for r in pf],
                color=BLUE, lw=2, marker="o", ms=5, capsize=2, elinewidth=1, label="p-fair BPE (p = 1 … ∞)", zorder=3)
    for grp, col, name, mk in ((hy, ORANGE, "Hybrid schedule", "s"), (ot, AQUA, "Worst-N / moving window", "D")):
        ax.errorbar([float(r["cr"]) for r in grp], [float(r["gini"]) for r in grp],
                    xerr=[float(r["cr_err"]) for r in grp], yerr=[float(r["gini_err"]) for r in grp],
                    color=col, ls="none", marker=mk, ms=6, capsize=2, elinewidth=1, label=name, zorder=4,
                    markeredgecolor=SURFACE, markeredgewidth=1)
    for r in pf:
        if r["method"] in label_every:
            if r["method"] == "Parity-aware":
                ax.annotate("Parity-aware (p = ∞)", (float(r["cr"]), float(r["gini"])), xytext=(-4, 12),
                            textcoords="offset points", fontsize=8, color=INK, ha="left")
            else:
                ax.annotate(r["method"], (float(r["cr"]), float(r["gini"])), xytext=(6, 4),
                            textcoords="offset points", fontsize=8, color=INK)
    for r in hy:
        ax.annotate(r["method"], (float(r["cr"]), float(r["gini"])), xytext=(6, -10), textcoords="offset points",
                    fontsize=8, color=INK)
    style(ax, "Compression: bytes per token on the training text (higher is better)",
          "Disparity: Gini of per-language cost (lower is better)")
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout(); fig.savefig(out, facecolor=SURFACE); plt.close(fig)


def perlang(d, out):
    rs = [r for r in rows(os.path.join(d, "paper_perlang.csv"))
          if r["method"] == "Classical" or r["method"].startswith("p=") or r["method"] == "Parity-aware"]
    labels = ["1" if r["method"] == "Classical" else "∞" if r["method"] == "Parity-aware" else r["method"][2:] for r in rs]
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=160, facecolor=SURFACE)
    for lang, col in zip(("hin", "mar", "guj", "eng"), (BLUE, ORANGE, AQUA, YELLOW)):
        ys = [float(r[lang]) for r in rs]
        ax.plot(range(len(rs)), ys, color=col, lw=2, marker="o", ms=5, label=LANG[lang], markeredgecolor=SURFACE)
    ax.set_xticks(range(len(rs))); ax.set_xticklabels(labels)
    style(ax, "p  (1 = classical BPE, ∞ = Parity-aware BPE)", "Tokens per FLORES sentence")
    ax.set_title("Tokens per sentence along the front (4 languages, 32k, 4× English)", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower left")
    ax.set_xlim(-0.3, len(rs) - 0.7)
    fig.tight_layout(); fig.savefig(out, facecolor=SURFACE); plt.close(fig)


def sens(d, out):
    rs = rows(os.path.join(d, "paper_sens.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6), dpi=160, facecolor=SURFACE, sharey=True)
    for ax, keys, title, xl in ((axes[0], ["1x", "4x", "8x"], "Data balance (32k vocabulary)", "English text relative to each Indian language"),
                                (axes[1], ["4x", "64k", "128k", "256k"], "Vocabulary size (4× English)", "Vocabulary")):
        sel = [r for k in keys for r in rs if r["setting"] == k]
        names = ["32k" if k == "4x" and ax is axes[1] else k for k in keys]
        for m, col, lab in (("Classical", BLUE, "Classical"), ("p=8", ORANGE, "p = 8"), ("Parity-aware", AQUA, "Parity-aware")):
            ys = [float(r[m]) for r in sel]
            ax.plot(range(len(sel)), ys, color=col, lw=2, marker="o", ms=5, label=lab, markeredgecolor=SURFACE)
            ax.annotate(lab, (len(sel) - 1, ys[-1]), xytext=(5, 0), textcoords="offset points", fontsize=8, color=INK, va="center")
        ax.set_xticks(range(len(sel))); ax.set_xticklabels(names)
        style(ax, xl, "Gini of tokens per sentence")
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.set_xlim(-0.3, len(sel) - 1 + 1.2)
    axes[1].set_ylabel("")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    fig.tight_layout(); fig.savefig(out, facecolor=SURFACE); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main", default="results_main")
    ap.add_argument("--l14", default="results_l14_eb")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    front(a.main, os.path.join(a.out, "front_4lang.png"),
          "Hindi, Marathi, Gujarati, English: 32k vocabulary, 4× English, 5 training samples",
          {"Classical", "p=8", "p=32", "Parity-aware"})
    front(a.l14, os.path.join(a.out, "front_14lang.png"),
          "14 languages in 10 scripts: 64k vocabulary, natural data balance, 3 training samples",
          {"Classical", "p=8", "p=32", "Parity-aware"})
    perlang(a.main, os.path.join(a.out, "per_language.png"))
    sens(a.main, os.path.join(a.out, "sensitivity.png"))
    print("wrote", sorted(os.listdir(a.out)))


if __name__ == "__main__":
    main()
