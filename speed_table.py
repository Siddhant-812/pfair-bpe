#!/usr/bin/env python3
"""E8: training time with and without lazy evaluation (Appendix table).

    python speed_table.py --out results_speed                 # 4 languages, 32k, p = 1, 8, inf, lazy and naive
    python speed_table.py --out results_speed --langs hin,...,eng --train-dir data_l14/train --vocab-size 64000 --tag 14

Runs the trainer once per setting, records the wall-clock time and the
"heap re-checks per merge" that the trainer prints, and writes
<out>/timings.csv (one row per run) and prints the LaTeX rows. Run it on an
otherwise idle machine.
"""
import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EXE = os.path.join(HERE, "target", "release", "pfair")


def run(args):
    t0 = time.perf_counter()
    r = subprocess.run([EXE] + args, cwd=HERE, capture_output=True, text=True, check=True)
    secs = time.perf_counter() - t0
    m = re.search(r"heap re-checks per merge: ([0-9.]+)", r.stdout + r.stderr)
    return secs, (float(m.group(1)) if m else float("nan"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--langs", default="hin,mar,guj,eng")
    ap.add_argument("--train-dir", default="data/train")
    ap.add_argument("--dev-dir", default="data/flores/dev")
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--ps", default="1,8,inf")
    ap.add_argument("--tag", default="", help="label for the 'languages' column (default: number of languages)")
    ap.add_argument("--skip-naive", default="", help="p values for which the naive run is skipped (comma separated)")
    ap.add_argument("--out", default="results_speed")
    a = ap.parse_args()
    os.makedirs(os.path.join(HERE, a.out), exist_ok=True)
    tag = a.tag or str(len(a.langs.split(",")))
    path = os.path.join(HERE, a.out, "timings.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["languages", "p", "vocab", "lazy_s", "naive_s", "rechecks_per_merge"])
        for p in a.ps.split(","):
            base = ["train", "--langs", a.langs, "--train-dir", a.train_dir, "--dev-dir", a.dev_dir,
                    "--unit", "line", "--p", p, "--vocab-size", str(a.vocab_size)]
            out = os.path.join(a.out, f"tmp_{tag}_{p}")
            lazy, rechecks = run(base + ["--out", out])
            shutil.rmtree(os.path.join(HERE, out), ignore_errors=True)
            if p in a.skip_naive.split(","):
                naive = float("nan")
            else:
                naive, _ = run(base + ["--out", out + "_naive", "--naive"])
                shutil.rmtree(os.path.join(HERE, out + "_naive"), ignore_errors=True)
            w.writerow([tag, p, a.vocab_size, f"{lazy:.1f}", f"{naive:.1f}", f"{rechecks:.1f}"])
            f.flush()
            print(f"{tag} & {p} & {lazy:.0f} & {naive:.0f} & {rechecks:.1f}\\\\", flush=True)
    print(f"written {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
