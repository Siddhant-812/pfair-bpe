#!/usr/bin/env python3
"""Vocabulary allocation across scripts (the "allocation level" of the paper).

    python allocate.py --clusters "Deva=hin,mar;Gujr=guj;Latn=eng" --ps 1,8 [--vocab-size 32000]

Steps (all automatic):
 1. For every cluster c, train tokenizers on the cluster's languages only at a few
    sizes, measure the cluster cost t_c(k) = share-weighted mean tokens per sentence
    of its languages on FLORES dev, and fit  t_c(k) = a_c * k^(-b_c) + d_c.
 2. Solve  min sum_c pi_c t_c(k_c)^p  s.t.  sum_c k_c = K  by water-filling on the
    KKT condition  p pi_c t_c(k_c)^(p-1) |t_c'(k_c)| = mu  (bisection on mu).
 3. Baselines: uniform split; data-proportional split; union-proportion
    (Brahma et al. 2026: proportional to the summed size of the clusters'
    monolingual vocabularies, which with equal-size monolingual vocabularies is
    proportional to the number of languages in the cluster).
 4. For each allocation, train one tokenizer per cluster at its budget, merge them
    (`pfair merge`) and evaluate the merged tokenizer on FLORES devtest.
    A joint tokenizer at K (no allocation) is included for reference.

Outputs in --out (default results_alloc):
    curves.csv       the measured points and the fitted a, b, d per cluster
    allocations.csv  the budgets of every method
    results_summary.csv / results_per_language.csv   as in run_all.py
"""
import argparse
import csv
import json
import math
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXE = os.path.join(HERE, "target", "release", "pfair.exe" if os.name == "nt" else "pfair")


def sh(cmd, capture=False):
    print("$ " + " ".join(map(str, cmd)), flush=True)
    r = subprocess.run(list(map(str, cmd)), check=True, cwd=HERE, capture_output=capture, text=True)
    return r.stdout if capture else None


def train_chars(data, lang):
    """size of a language's training text in characters (its share pi)"""
    n = 0
    with open(os.path.join(HERE, data, "train", f"{lang}.txt"), encoding="utf-8") as f:
        for line in f:
            n += len(line.rstrip("\n"))
    return n


def cluster_cost(model, langs, data, shares):
    """share-weighted tokens per sentence of the cluster's languages on FLORES dev"""
    out = sh([EXE, "eval", "--langs", ",".join(langs), "--model", model, "--test-dir", f"{data}/flores/dev", "--unit", "line"], capture=True)
    cost = {}
    for line in out.splitlines():
        parts = line.split()
        if parts and parts[0] in langs and len(parts) >= 8:
            cost[parts[0]] = float(parts[-1])  # tok/line is the last column
    w = np.array([shares[l] for l in langs])
    return float(sum(w[i] * cost[l] for i, l in enumerate(langs)) / w.sum()), cost


def fit_curve(ks, ts):
    """least squares fit of t = a k^-b + d with a>0, b>0, d>=0 (grid on b, linear in a, d)"""
    ks, ts = np.array(ks, float), np.array(ts, float)
    best = None
    for b in np.concatenate([np.linspace(0.02, 0.6, 59), np.linspace(0.62, 2.0, 70)]):
        X = np.stack([ks ** (-b), np.ones_like(ks)], 1)
        (aa, dd), *_ = np.linalg.lstsq(X, ts, rcond=None)
        if aa <= 0:
            continue
        dd = max(dd, 0.0)
        err = float(np.sum((X @ np.array([aa, dd]) - ts) ** 2))
        if best is None or err < best[0]:
            best = (err, aa, b, dd)
    if best is None:
        sys.exit("could not fit a decreasing curve; check the measured points")
    return best[1], best[2], best[3], best[0]


def t_of(k, a, b, d):
    return a * k ** (-b) + d


def marginal(k, a, b, d, pi, p):
    """p pi t^(p-1) |t'(k)| : the gain from one more slot (decreasing in k)"""
    t = t_of(k, a, b, d)
    return p * pi * t ** (p - 1) * a * b * k ** (-b - 1)


def water_fill(curves, pis, K, p, kmin):
    """budgets k_c with sum K that equalise the marginal gain (KKT), by bisection on mu"""
    names = list(curves)

    def budgets(mu):
        out = {}
        for c in names:
            a, b, d = curves[c]
            lo, hi = kmin[c], K
            if marginal(lo, a, b, d, pis[c], p) <= mu:
                out[c] = lo
                continue
            for _ in range(100):  # marginal gain is decreasing in k: bisection
                mid = (lo + hi) / 2
                if marginal(mid, a, b, d, pis[c], p) > mu:
                    lo = mid
                else:
                    hi = mid
            out[c] = lo
        return out

    lo, hi = 0.0, max(marginal(kmin[c], *curves[c], pis[c], p) for c in names)
    for _ in range(200):
        mu = (lo + hi) / 2
        if sum(budgets(mu).values()) > K:
            lo = mu  # too many slots handed out: raise the bar
        else:
            hi = mu
    k = budgets(hi)
    # round to integers that sum to K
    rounded = {c: int(round(v)) for c, v in k.items()}
    diff = K - sum(rounded.values())
    big = max(rounded, key=rounded.get)
    rounded[big] += diff
    return rounded


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clusters", required=True, help='e.g. "Deva=hin,mar;Gujr=guj;Latn=eng"')
    ap.add_argument("--ps", default="1,8", help="p values for the convex allocation (and the within-cluster rule)")
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--probe-sizes", default="2000,4000,8000,16000,32000", help="per-cluster sizes used to fit the curves")
    ap.add_argument("--within-p", default="1", help="p used inside each cluster's tokenizer (1 = classical)")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results_alloc")
    ap.add_argument("--unit", default="line")
    a = ap.parse_args()

    clusters = {}
    for item in a.clusters.split(";"):
        name, langs = item.split("=")
        clusters[name.strip()] = [l.strip() for l in langs.split(",") if l.strip()]
    langs_all = [l for ls in clusters.values() for l in ls]
    out = os.path.join(HERE, a.out)
    os.makedirs(os.path.join(out, "models"), exist_ok=True)
    chars = {l: train_chars(a.data, l) for l in langs_all}
    total = sum(chars.values())
    shares = {l: chars[l] / total for l in langs_all}
    pis = {c: sum(shares[l] for l in ls) for c, ls in clusters.items()}
    print("cluster shares of the training text:", {c: round(v, 3) for c, v in pis.items()})

    # ---- 1. probe sizes and fit the curves ----
    probe = [int(x) for x in a.probe_sizes.split(",")]
    curves, kmin, rows = {}, {}, []
    for c, ls in clusters.items():
        ks, ts = [], []
        for k in probe:
            model = os.path.join(a.out, "models", f"probe_{c}_{k}")
            if not os.path.exists(os.path.join(HERE, model, "model.txt")):
                try:
                    sh([EXE, "train", "--langs", ",".join(ls), "--train-dir", f"{a.data}/train", "--dev-dir", f"{a.data}/flores/dev",
                        "--unit", a.unit, "--p", a.within_p, "--vocab-size", k, "--out", model])
                except subprocess.CalledProcessError:
                    print(f"   (size {k} is not larger than the alphabet of {c}; skipped)")
                    continue
            t, _ = cluster_cost(model, ls, a.data, shares)
            ks.append(k)
            ts.append(t)
            rows.append({"cluster": c, "k": k, "cost": round(t, 4), "kind": "measured"})
        if len(ks) < 3:
            sys.exit(f"cluster {c}: need at least 3 probe sizes above its alphabet size")
        aa, bb, dd, err = fit_curve(ks, ts)
        curves[c] = (aa, bb, dd)
        kmin[c] = min(ks)
        fitted = [t_of(k, aa, bb, dd) for k in ks]
        rows.append({"cluster": c, "k": "", "cost": "", "kind": f"fit a={aa:.4g} b={bb:.4g} d={dd:.4g} rmse={math.sqrt(err / len(ks)):.4f}"})
        print(f"{c}: measured {dict(zip(ks, [round(t, 2) for t in ts]))}; fit a={aa:.1f} b={bb:.3f} d={dd:.2f}; fitted {[round(x, 2) for x in fitted]}")
    with open(os.path.join(out, "curves.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["cluster", "k", "cost", "kind"])
        w.writeheader()
        w.writerows(rows)

    # ---- 2. allocations ----
    K = a.vocab_size
    names = list(clusters)
    allocs = {}
    allocs["Uniform"] = {c: K // len(names) for c in names}
    allocs["Data-proportional"] = {c: int(round(K * pis[c])) for c in names}
    nlang = {c: len(ls) for c, ls in clusters.items()}
    allocs["Union-proportion"] = {c: int(round(K * nlang[c] / len(langs_all))) for c in names}
    for p in [float(x) for x in a.ps.split(",")]:
        allocs[f"Convex,p={p:g}"] = water_fill(curves, pis, K, p, kmin)
    for name, al in allocs.items():  # make every allocation sum to K exactly
        diff = K - sum(al.values())
        al[max(al, key=al.get)] += diff
    def write_allocations():
        with open(os.path.join(out, "allocations.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["method"] + names + ["predicted_cost_p1"])
            for name, al in allocs.items():
                pred = sum(pis[c] * t_of(al[c], *curves[c]) for c in names)
                w.writerow([name] + [al[c] for c in names] + [round(pred, 3)])
                print(f"{name:<20} " + "  ".join(f"{c}={al[c]}" for c in names) + f"   predicted cost {pred:.3f}")
    write_allocations()

    # ---- 3. build and evaluate ----
    csvs = {k: os.path.join(a.out, v) for k, v in [("summary", "results_summary.csv"), ("detail", "results_per_language.csv"), ("csv-train", "results_train_cr.csv")]}
    for v in csvs.values():
        if os.path.exists(os.path.join(HERE, v)):
            os.remove(os.path.join(HERE, v))
    evals = []
    for p in a.ps.split(","):  # joint tokenizers for reference (no allocation)
        model = os.path.join(a.out, "models", f"joint_p{p}")
        sh([EXE, "train", "--langs", ",".join(langs_all), "--train-dir", f"{a.data}/train", "--dev-dir", f"{a.data}/flores/dev",
            "--unit", a.unit, "--p", p, "--vocab-size", K, "--out", model])
        evals.append((f"Joint,p={p}", model))
    def build(name, al):
        parts = []
        for c, ls in clusters.items():
            model = os.path.join(a.out, "models", f"{name.replace(',', '_').replace('=', '')}_{c}")
            sh([EXE, "train", "--langs", ",".join(ls), "--train-dir", f"{a.data}/train", "--dev-dir", f"{a.data}/flores/dev",
                "--unit", a.unit, "--p", a.within_p, "--vocab-size", al[c], "--out", model])
            parts.append(model)
        merged = os.path.join(a.out, "models", f"{name.replace(',', '_').replace('=', '')}_merged")
        sh([EXE, "merge", "--models", ",".join(parts), "--out", merged])
        with open(os.path.join(HERE, merged, "model.txt"), encoding="utf-8") as f:
            lines = f.read().split("\n")
        n_alpha = int(lines[2].split()[1])
        return merged, 2 + n_alpha + int(lines[3 + n_alpha].split()[1])

    for name, al in allocs.items():
        # Per-cluster alphabets overlap (digits, punctuation, Latin letters inside Indic
        # text), so the merged vocabulary is smaller than sum k_c. Top the budgets up
        # once or twice so that every merged tokenizer has (about) K entries, like the
        # joint tokenizer it is compared with.
        budgets = dict(al)
        for _ in range(3):
            merged, size = build(name, budgets)
            deficit = K - size
            if abs(deficit) <= max(20, K // 500):
                break
            tot = sum(budgets.values())
            budgets = {c: budgets[c] + int(round(deficit * budgets[c] / tot)) for c in budgets}
        print(f"   {name}: merged vocabulary {size} (budgets {budgets})")
        allocs[name] = budgets
        evals.append((name, merged))
    for name, model in evals:
        sh([EXE, "eval", "--langs", ",".join(langs_all), "--model", model, "--test-dir", f"{a.data}/flores/devtest", "--unit", a.unit,
            "--name", name, "--train-dir", f"{a.data}/train", "--summary", csvs["summary"], "--detail", csvs["detail"], "--csv-train", csvs["csv-train"]])
    print("\nfinal budgets after topping up for overlapping alphabets:")
    write_allocations()
    print("\n" + open(os.path.join(HERE, csvs["summary"]), encoding="utf-8").read())
    json.dump({"clusters": clusters, "shares": shares, "curves": {c: list(v) for c, v in curves.items()}, "allocations": allocs},
              open(os.path.join(out, "allocation.json"), "w"), indent=2)
    print(f"Done. See {a.out}/allocations.csv, curves.csv and results_summary.csv")


if __name__ == "__main__":
    main()
