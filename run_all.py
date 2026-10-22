#!/usr/bin/env python3
"""Run one experiment with one command:

    python run_all.py

It builds the Rust trainer, downloads the data (first run only), trains the
tokenizers, scores each one on FLORES-200 devtest and writes

    results/results.csv               method,cr,gini   (cr = compression on FLORES devtest)
    results/results_train_cr.csv      method,cr,gini   (cr = compression on the training text)
    results/results_summary.csv       every metric, one row per tokenizer
    results/results_per_language.csv  token cost of each language under each tokenizer
    results/results_morph.csv         boundary F1 / MorphScore per language (only with --morph-gold)
    results/config.json               the exact settings of this run

Which tokenizers to train:
    --ps 1,inf,2,4,8          p values (1 = classical, inf = Parity-aware)      -> "Classical", "p=4", "Parity-aware"
    --hybrids 0.5,0.75,0.9    hybrid baseline, share of classical merges        -> "Hybrid-75"
    --worst-ns 2,3            Worst-N parity baseline                           -> "WorstN-2"
    --window 100 --alpha 2    moving-window Parity-aware baseline               -> "Window-100"
    --lambdas 0.5,1           morphology penalty, crossed with --ps              -> "p=4,lambda=0.5"
      --morph-lexicon FILE[,FILE]   segmentations used for the penalty (training words)
      --morph-gold hin=FILE,mar=FILE  held-out gold segmentations to score boundary F1 on
    --pi-uniform              weight languages equally instead of by text share

Data and setting:
    --langs hin,mar,guj,eng   languages (names from get_data.py)
    --vocab-size 32000
    --unit line               cost unit: line (default), byte or char   (see README)
    --chars-m 20 --eng-chars-m 80   training text per language / for English
    --sizes eng=80,hin=40     per-language sizes (million characters), overrides the two above
    --natural --total-chars-m 400   sizes in proportion to each language's Sangraha split
    --part 1                  a different sample of the training text
    --data DIR --out DIR      where the text lives / where the results go
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def sh(cmd, quiet=False):
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=HERE, stdout=subprocess.DEVNULL if quiet else None)


def method_name(p):
    return {"1": "Classical", "inf": "Parity-aware"}.get(p, f"p={p}")


def hybrid_name(f):
    return f"Hybrid-{round(100 * float(f))}"  # e.g. Hybrid-50 = first 50% of merges classical


def parse_list(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ps", default="1,inf,2,4", help="comma-separated p values")
    ap.add_argument("--hybrids", default="", help="hybrid baseline: shares of classical merges, e.g. 0.5,0.75,0.9")
    ap.add_argument("--worst-ns", default="", help="Worst-N baseline: values of N, e.g. 2,3")
    ap.add_argument("--window", type=int, default=0, help="moving-window Parity-aware baseline with this window (0 = off)")
    ap.add_argument("--alpha", type=float, default=2.0, help="alpha for --window")
    ap.add_argument("--lambdas", default="", help="morphology penalties, crossed with --ps, e.g. 0.5,1")
    ap.add_argument("--morph-lexicon", default="", help="segmentation file(s) for the penalty, comma separated")
    ap.add_argument("--morph-gold", default="", help="held-out gold segmentations per language: hin=FILE,mar=FILE")
    ap.add_argument("--pi-uniform", action="store_true", help="weight languages equally instead of by text share")
    ap.add_argument("--langs", default="hin,mar,guj,eng")
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--unit", default="line", choices=["line", "byte", "char"])
    ap.add_argument("--chars-m", type=float, default=20, help="million characters of training text per language")
    ap.add_argument("--eng-chars-m", type=float, default=80, help="million characters of English training text")
    ap.add_argument("--sizes", default="", help="per-language sizes in million characters, e.g. eng=80,hin=40")
    ap.add_argument("--natural", action="store_true", help="sizes in proportion to each language's Sangraha split")
    ap.add_argument("--total-chars-m", type=float, default=400, help="total for --natural")
    ap.add_argument("--part", type=int, default=0, help="which sample of the training text (0, 1, 2, ...)")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--skip-data", action="store_true", help="do not call get_data.py (the text is already there)")
    a = ap.parse_args()
    langs = a.langs

    # 1. build
    if shutil.which("cargo") is None:
        sys.exit("Rust is not installed. Install it from https://rustup.rs (one command), open a new terminal, and run this again.")
    sh(["cargo", "build", "--release"], quiet=True)
    exe = os.path.join(HERE, "target", "release", "pfair.exe" if os.name == "nt" else "pfair")

    # 2. data (skipped automatically if the files are already there)
    if not a.skip_data:
        cmd = [sys.executable, "get_data.py", "--out", a.data, "--langs", langs, "--chars-m", str(a.chars_m),
               "--eng-chars-m", str(a.eng_chars_m), "--part", str(a.part), "--total-chars-m", str(a.total_chars_m)]
        if a.sizes:
            cmd += ["--sizes", a.sizes]
        if a.natural:
            cmd += ["--natural"]
        sh(cmd)

    # 3. train + evaluate
    out_abs = os.path.join(HERE, a.out)
    os.makedirs(out_abs, exist_ok=True)
    with open(os.path.join(out_abs, "config.json"), "w") as f:
        json.dump(vars(a), f, indent=2)
    csvs = {k: os.path.join(a.out, f) for k, f in
            [("csv", "results.csv"), ("csv-train", "results_train_cr.csv"),
             ("summary", "results_summary.csv"), ("detail", "results_per_language.csv"),
             ("morph", "results_morph.csv")]}
    for f in csvs.values():  # start clean so that reruns do not duplicate rows
        if os.path.exists(os.path.join(HERE, f)):
            os.remove(os.path.join(HERE, f))

    # every run: (display name, folder tag, extra trainer arguments)
    runs = []
    lambdas = [None] + parse_list(a.lambdas)
    for p in parse_list(a.ps):
        for lam in lambdas:
            name, tag, extra = method_name(p), f"p{p}", ["--p", p]
            if lam is not None:
                name, tag = f"{name},lambda={lam}", f"{tag}_lambda{lam}"
                extra += ["--lambda", lam, "--morph-lexicon", a.morph_lexicon]
            runs.append((name, tag, extra))
    for f in parse_list(a.hybrids):
        runs.append((hybrid_name(f), f"hybrid{f}", ["--p", "inf", "--hybrid", f]))
    for n in parse_list(a.worst_ns):
        runs.append((f"WorstN-{n}", f"worstn{n}", ["--p", "inf", "--worst-n", n]))
    if a.window > 0:
        runs.append((f"Window-{a.window}", f"window{a.window}", ["--p", "inf", "--window", str(a.window), "--alpha", str(a.alpha)]))
    if a.lambdas and not a.morph_lexicon:
        sys.exit("--lambdas needs --morph-lexicon FILE (see morph_prepare.py)")
    gold = dict(item.split("=") for item in parse_list(a.morph_gold))

    with open(os.path.join(out_abs, "runs.json"), "w") as f:  # name -> model folder, for the other scripts
        json.dump({name: os.path.join("models", f"{tag}_v{a.vocab_size}_{a.unit}") for name, tag, _ in runs}, f, indent=2)
    for name, tag, extra in runs:
        model = os.path.join(a.out, "models", f"{tag}_v{a.vocab_size}_{a.unit}")
        t0 = time.time()
        sh([exe, "train", "--langs", langs, "--train-dir", f"{a.data}/train",
            # each language's current cost is measured on FLORES dev while training ...
            "--dev-dir", f"{a.data}/flores/dev",
            "--unit", a.unit, "--vocab-size", str(a.vocab_size), "--out", model]
           + extra + (["--pi-uniform"] if a.pi_uniform else []))
        print(f"   trained in {time.time() - t0:.0f} s")
        # ... and the final numbers come from FLORES devtest, which training never saw
        sh([exe, "eval", "--langs", langs, "--model", model, "--test-dir", f"{a.data}/flores/devtest",
            "--unit", a.unit, "--name", name,
            "--csv", csvs["csv"], "--csv-train", csvs["csv-train"],
            "--summary", csvs["summary"], "--detail", csvs["detail"],
            "--per-line", os.path.join(model, "test_tokens_per_line.tsv")])
        for lang, path in gold.items():
            sh([exe, "morph-eval", "--model", model, "--gold", path, "--lang", lang, "--name", name, "--csv", csvs["morph"]])

    print("\n" + "=" * 60)
    for k, what in [("csv", "compression on FLORES devtest"), ("csv-train", "compression on the training text")]:
        print(f"{csvs[k]}   ({what})")
        print(open(os.path.join(HERE, csvs[k]), encoding="utf-8").read())
    if gold:
        print(csvs["morph"])
        print(open(os.path.join(HERE, csvs["morph"]), encoding="utf-8").read())
    print(f"Done. Results are in {a.out}/ (see results_summary.csv and results_per_language.csv).")


if __name__ == "__main__":
    main()
