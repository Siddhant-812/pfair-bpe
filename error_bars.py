#!/usr/bin/env python3
"""Error bars: repeat the whole experiment on several different samples of
training text and report the average and the spread.

    python error_bars.py                # 5 samples; about 3 minutes per sample
    python error_bars.py --samples 3    # quicker
    python error_bars.py --run-args "--langs hin,mar,guj,ben,tam,eng --worst-ns 2,3 --window 100"
                                        # anything after --run-args is passed to run_all.py

Sample 0 is the text you already have. Every other sample downloads a
different, non-overlapping part of Sangraha (about 250 MB each; the text is
deleted again afterwards unless you pass --keep-data).

Written to results_errorbars/ :
    results_errorbars.csv    method,cr,gini,cr_err,gini_err   average and standard deviation over the samples
    chart_pfair.csv          the same, p-fair rows only        (for the slide chart)
    chart_hybrid.csv         the same, hybrid rows only
    all_samples.csv          every number from every sample
    bootstrap_test_sentences.csv   95% interval of the Gini when the TEST sentences are resampled (sample 0)
"""
import argparse
import csv
import json
import os
import random
import shlex
import shutil
import statistics
import subprocess
import sys

from compare_hybrid import is_hybrid, pfair_gini_at

HERE = os.path.dirname(os.path.abspath(__file__))
LANGS = ["hin", "mar", "guj", "eng"]  # replaced by --langs inside --run-args if given


def gini(x):
    n, mean = len(x), sum(x) / len(x)
    return sum(abs(a - b) for a in x for b in x) / (2 * n * n * mean)


def data_matches(folder, chars_m, eng_m, langs, extra):
    """True if `folder` already holds sample 0 at exactly these sizes."""
    if "--sizes" in extra or "--natural" in extra:
        return False  # sizes set by hand: always let get_data.py decide (it skips files that match)
    for l in langs:
        note = os.path.join(HERE, folder, "train", f"{l}.size")
        want = str(int((eng_m if l == "eng" else chars_m) * 1e6))
        if not os.path.exists(note) or open(note).read().strip() != want:
            return False
    return True


def read_summary(folder):
    with open(os.path.join(HERE, folder, "results_summary.csv"), encoding="utf-8") as f:
        return {r["method"]: (float(r["train_bytes_per_token"]), float(r["gini_per_line"])) for r in csv.DictReader(f)}


def mean_sd(v):
    return statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else 0.0)


def bootstrap(folder, methods_to_models, out_csv, n_boot=1000):
    """Resample the test sentences (the same resample for every method) and
    report the 95% interval of each method's Gini."""
    table = {}
    for method, model in methods_to_models.items():
        with open(os.path.join(HERE, folder, "models", model, "test_tokens_per_line.tsv"), encoding="utf-8") as f:
            rows = [list(map(int, line.split("\t"))) for line in f.read().split("\n")[1:] if line]
        table[method] = rows  # [sentence][language]
    n = len(next(iter(table.values())))
    rng = random.Random(0)
    draws = {m: [] for m in table}
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        for m, rows in table.items():
            sums = [0] * len(LANGS)
            for i in idx:
                r = rows[i]
                for k in range(len(LANGS)):
                    sums[k] += r[k]
            draws[m].append(gini(sums))
    out = []
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "gini", "ci_low", "ci_high"])
        for m, rows in table.items():
            g = gini([sum(r[k] for r in rows) for k in range(len(LANGS))])
            d = sorted(draws[m])
            lo, hi = d[int(0.025 * n_boot)], d[int(0.975 * n_boot) - 1]
            w.writerow([m, f"{g:.4f}", f"{lo:.4f}", f"{hi:.4f}"])
            out.append((m, g, lo, hi))
    return out, draws


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--ps", default="1,2,4,8,16,32,inf")
    ap.add_argument("--hybrids", default="0.5,0.75,0.9", help="hybrid fractions to include ('' for none)")
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--chars-m", type=float, default=20)
    ap.add_argument("--eng-chars-m", type=float, default=80)
    ap.add_argument("--out", default="results_errorbars")
    ap.add_argument("--keep-data", action="store_true", help="do not delete the extra training text afterwards")
    ap.add_argument("--run-args", default="", help="extra arguments passed to run_all.py, in quotes")
    a = ap.parse_args()
    os.makedirs(os.path.join(HERE, a.out), exist_ok=True)
    extra = shlex.split(a.run_args)
    global LANGS
    if "--langs" in extra:
        LANGS = extra[extra.index("--langs") + 1].split(",")

    # ---- 1. run the experiment once per sample ----
    folders = []
    for i in range(a.samples):
        reuse = i == 0 and data_matches("data", a.chars_m, a.eng_chars_m, LANGS, extra)
        data = "data" if reuse else os.path.join(a.out, f"data_sample{i}")
        res = os.path.join(a.out, f"sample{i}")
        flores_have, flores_want = os.path.join(HERE, "data", "flores"), os.path.join(HERE, data, "flores")
        if os.path.exists(flores_have) and not os.path.exists(flores_want):
            shutil.copytree(flores_have, flores_want)  # no need to download the test set again
        print(f"\n##### sample {i + 1} of {a.samples} #####", flush=True)
        subprocess.run(
            [sys.executable, "run_all.py", "--part", str(i), "--data", data, "--out", res, "--ps", a.ps,
             "--hybrids", a.hybrids, "--vocab-size", str(a.vocab_size),
             "--chars-m", str(a.chars_m), "--eng-chars-m", str(a.eng_chars_m)] + extra,
            check=True, cwd=HERE, stdout=subprocess.DEVNULL)
        folders.append(res)
        if not reuse and not a.keep_data:
            shutil.rmtree(os.path.join(HERE, data, "train"), ignore_errors=True)
        print("   " + "  ".join(f"{m}: {g:.4f}" for m, (_, g) in read_summary(res).items()), flush=True)

    # ---- 2. average and spread over the samples ----
    per_sample = [read_summary(f) for f in folders]
    methods = list(per_sample[0])
    with open(os.path.join(HERE, a.out, "all_samples.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample", "method", "train_bytes_per_token", "gini"])
        for i, s in enumerate(per_sample):
            for m in methods:
                w.writerow([i, m, f"{s[m][0]:.4f}", f"{s[m][1]:.4f}"])
    stats = {}
    for m in methods:
        (cr, cr_sd), (g, g_sd) = mean_sd([s[m][0] for s in per_sample]), mean_sd([s[m][1] for s in per_sample])
        stats[m] = (cr, g, cr_sd, g_sd)
    for name, keep in [("results_errorbars.csv", lambda m: True), ("chart_pfair.csv", lambda m: not is_hybrid(m)),
                       ("chart_hybrid.csv", is_hybrid)]:
        with open(os.path.join(HERE, a.out, name), "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["method", "cr", "gini", "cr_err", "gini_err"])
            for m in methods:
                if keep(m):
                    w.writerow([m] + [f"{x:.4f}" for x in stats[m]])

    n = len(per_sample)
    print("\n" + "=" * 78)
    print(f"Average over {n} samples of training text  (+- one standard deviation)\n")
    print(f"{'method':<13} {'Gini':>18} {'compression (training text)':>32}")
    for m in methods:
        cr, g, cr_sd, g_sd = stats[m]
        print(f"{m:<13} {g:>9.4f} +- {g_sd:.4f} {cr:>20.4f} +- {cr_sd:.4f}")

    # ---- 3. does the pattern hold in every sample? ----
    pf = [m for m in methods if not is_hybrid(m)]  # in the order of --ps
    print("\nDoes each step up in p lower the Gini AND the compression?")
    for x, y in zip(pf, pf[1:]):
        g_ok = sum(s[y][1] < s[x][1] for s in per_sample)
        c_ok = sum(s[y][0] < s[x][0] for s in per_sample)
        print(f"  {x:>12} -> {y:<12}  Gini lower in {g_ok}/{n} samples, compression lower in {c_ok}/{n}")

    # ---- 4. p-fair against the baselines (hybrid, Worst-N, window), at equal compression ----
    hy = [m for m in methods if is_hybrid(m)]
    if hy:
        print("\np-fair against each baseline at the same compression")
        print("(difference = baseline Gini minus p-fair Gini; positive means p-fair is fairer)")
        for m in hy:
            diffs = []
            for s in per_sample:
                g_pf = pfair_gini_at(s, s[m][0])
                if g_pf is not None:
                    diffs.append(s[m][1] - g_pf)
            if not diffs:
                print(f"  {m:<11} outside the p-fair range")
                continue
            d, d_sd = mean_sd(diffs)
            print(f"  {m:<11} difference {d:+.4f} +- {d_sd:.4f}   p-fair fairer in {sum(x > 0 for x in diffs)}/{len(diffs)} samples")

    # ---- 5. noise from the test set: resample the test sentences ----
    with open(os.path.join(HERE, folders[0], "runs.json"), encoding="utf-8") as f:
        models = {m: os.path.basename(path) for m, path in json.load(f).items()}
    boot, draws = bootstrap(folders[0], models, os.path.join(HERE, a.out, "bootstrap_test_sentences.csv"))
    print("\nNoise from the test set (sample 0, 1000 resamples of the test sentences): 95% interval of the Gini")
    for m, g, lo, hi in boot:
        print(f"  {m:<13} {g:.4f}   [{lo:.4f}, {hi:.4f}]")
    print("\nIs each step up in p a real drop in Gini, or test-set luck?")
    for x, y in zip(pf, pf[1:]):
        share = sum(b < a_ for a_, b in zip(draws[x], draws[y])) / len(draws[x])
        print(f"  {x:>12} -> {y:<12}  Gini lower in {100 * share:.1f}% of resamples")

    print(f"\nFiles are in {a.out}/")


if __name__ == "__main__":
    main()
